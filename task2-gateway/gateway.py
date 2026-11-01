"""MCP Security Gateway (Task 2).

An HTTP reverse proxy between an AI agent and a downstream MCP server
(Streamable HTTP transport). For every request it:

  1. authenticates the caller: "Authorization: Bearer <JWT>" -> user + role
  2. checks the MCP session belongs to that user
  3. inspects the JSON-RPC message (POST):
       tools/list and every other method   -> forwarded transparently
       tools/call of an admin_* tool        -> admins only; anyone else gets
                                               JSON-RPC error -32001 "Unauthorized
                                               Tool Call" from the gateway itself
  4. forwards what's allowed and streams the response (JSON or SSE) back.

It fails closed: a message it can't read unambiguously is never forwarded.

Run: uv run uvicorn gateway:app --port 8000
"""

import json
import logging
import os
import sys
from collections import OrderedDict
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

import auth
from auth import AuthError, Identity, bearer_token, verify
from policy import blocked_tool

DOWNSTREAM_URL = os.environ.get("DOWNSTREAM_MCP_URL", "http://127.0.0.1:9000/mcp")
MAX_BODY_BYTES = 1_000_000
MAX_TRACKED_SESSIONS = 10_000

# JSON-RPC error codes. -32000..-32099 is the range reserved for server-defined errors.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
UNAUTHENTICATED = -32000  # no valid credentials
UNAUTHORIZED_TOOL_CALL = -32001  # valid credentials, but this role may not call this tool
SESSION_FORBIDDEN = -32002  # the MCP session belongs to another user
DOWNSTREAM_UNAVAILABLE = -32003  # the MCP server behind the gateway can't be reached

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="gateway: %(message)s")
log = logging.getLogger("gateway")
logging.getLogger("httpx").setLevel(logging.WARNING)  # our own log line already covers each request

# Hop-by-hop headers describe one network hop, not the message, so a proxy
# must never forward them (RFC 9110). Host and Content-Length are recomputed
# by httpx for the downstream request.
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if auth.SECRET == auth.DEV_SECRET:
        log.warning("GATEWAY_JWT_SECRET is not set: using the built-in development secret. Never do this in production.")
    # One shared connection pool for all downstream calls.
    # No read timeout: SSE streams from the server may stay open for a long time.
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None))
    # MCP session id -> user who opened it, oldest first (bounded).
    app.state.session_owner = OrderedDict()
    yield
    await app.state.http.aclose()


app = FastAPI(title="MCP Security Gateway", lifespan=lifespan)


# --- Small helpers ---------------------------------------------------------------


class Rejected(Exception):
    """Stop processing and answer the client with this JSON-RPC error."""

    def __init__(self, code: int, message: str, status: int = 200, id_=None, headers: dict | None = None) -> None:
        super().__init__(message)
        self.code, self.message, self.status, self.id, self.headers = code, message, status, id_, headers


def jsonrpc_error(id_, code: int, message: str, status: int = 200, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}},
        status_code=status,
        headers=headers,
    )


def forwardable(headers) -> dict[str, str]:
    return {name: value for name, value in headers.items() if name.lower() not in HOP_BY_HOP}


def _no_duplicate_keys(pairs: list[tuple]) -> dict:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        # Parsers disagree on duplicates (first wins vs last wins). If the gateway
        # read one value and the server another, the check could be bypassed.
        raise ValueError("duplicate key")
    return dict(pairs)


def best_effort_id(body: bytes):
    """JSON-RPC id of the request if it can be read, so error replies can be matched."""
    try:
        message = json.loads(body)
    except ValueError:
        return None
    return message.get("id") if isinstance(message, dict) else None


# --- The four checks ------------------------------------------------------------------


async def read_body(request: Request) -> bytes:
    """Read the body, refusing anything the gateway couldn't inspect safely."""
    if request.headers.get("content-encoding", "identity").lower() != "identity":
        # A compressed body can't be inspected without decompressing it; refuse instead.
        raise Rejected(INVALID_REQUEST, "Content-Encoding is not supported", status=415)
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise Rejected(INVALID_REQUEST, "Request body too large", status=413)
    body = b""
    async for chunk in request.stream():  # also catches bodies without a Content-Length
        body += chunk
        if len(body) > MAX_BODY_BYTES:
            raise Rejected(INVALID_REQUEST, "Request body too large", status=413)
    return body


def authenticate(request: Request, body: bytes) -> Identity:
    try:
        return verify(bearer_token(request.headers.get("authorization")))
    except AuthError as exc:
        # HTTP 401 + WWW-Authenticate is how MCP clients learn they must authenticate.
        raise Rejected(UNAUTHENTICATED, str(exc), status=401, id_=best_effort_id(body),
                       headers={"WWW-Authenticate": 'Bearer realm="mcp-gateway"'}) from None


def check_session(request: Request, identity: Identity) -> None:
    """A session opened by one user can't be used by another (no session hijacking)."""
    session_id = request.headers.get("mcp-session-id")
    if not session_id:
        return  # no session yet: this should be an initialize request
    owner = request.app.state.session_owner.get(session_id)
    if owner is None:
        # Unknown to the gateway (e.g. after a gateway restart). 404 tells MCP
        # clients to start a new session (Streamable HTTP spec).
        raise Rejected(INVALID_REQUEST, "Session not found", status=404)
    if owner != identity.user:
        raise Rejected(SESSION_FORBIDDEN, "Session belongs to another user", status=403)


def check_message(body: bytes, identity: Identity) -> None:
    """Inspect the JSON-RPC message (single or batch) before anything is forwarded."""
    try:
        message = json.loads(body, object_pairs_hook=_no_duplicate_keys)
    except ValueError:
        # Fail closed: what the gateway can't read unambiguously, it doesn't forward.
        raise Rejected(PARSE_ERROR, "Parse error") from None

    batch = message if isinstance(message, list) else [message]
    if not batch or not all(isinstance(item, dict) for item in batch):
        raise Rejected(INVALID_REQUEST, "Invalid Request")
    for item in batch:
        tool = blocked_tool(item, identity)
        if tool is not None:
            log.warning("BLOCKED tools/call %r user=%s role=%s", tool, identity.user, identity.role)
            # A batch with one forbidden call is refused as a whole: nothing in it runs.
            raise Rejected(UNAUTHORIZED_TOOL_CALL, "Unauthorized Tool Call", id_=None if isinstance(message, list) else item.get("id"))


# --- Forwarding ---------------------------------------------------------------------------


def downstream_headers(request: Request, identity: Identity) -> dict[str, str]:
    """Headers for the downstream call.

    The client's token is NOT passed on: it was issued for the gateway, and
    forwarding it ("token passthrough") would let the downstream server replay
    it elsewhere. Instead the gateway states who the verified caller is.
    Any X-Gateway-* headers the client sent are dropped first, so they can't be spoofed.
    """
    headers = {
        name: value
        for name, value in forwardable(request.headers).items()
        if name.lower() != "authorization" and not name.lower().startswith("x-gateway-")
    }
    headers["X-Gateway-User"] = identity.user
    headers["X-Gateway-Role"] = identity.role
    return headers


def track_session(request: Request, response: httpx.Response, identity: Identity) -> None:
    sessions: OrderedDict = request.app.state.session_owner
    if request.method == "DELETE" and response.status_code < 300:
        sessions.pop(request.headers.get("mcp-session-id"), None)
        return
    new_session = response.headers.get("mcp-session-id")
    if new_session and new_session not in sessions:
        sessions[new_session] = identity.user
        if len(sessions) > MAX_TRACKED_SESSIONS:
            sessions.popitem(last=False)


async def forward(request: Request, body: bytes, identity: Identity) -> Response:
    """Send the request downstream and stream the response back as it arrives."""
    downstream = request.app.state.http.build_request(
        request.method,
        DOWNSTREAM_URL,
        headers=downstream_headers(request, identity),
        content=body if request.method == "POST" else b"",  # GET/DELETE carry no message
    )
    try:
        response = await request.app.state.http.send(downstream, stream=True)
    except httpx.HTTPError as exc:
        log.error("downstream unavailable: %s", type(exc).__name__)
        # Generic message: no internal hostnames or stack traces reach the client.
        return jsonrpc_error(best_effort_id(body), DOWNSTREAM_UNAVAILABLE, "Downstream MCP server unavailable", status=502)

    track_session(request, response, identity)
    return StreamingResponse(
        response.aiter_raw(),
        status_code=response.status_code,
        headers=forwardable(response.headers),
        background=BackgroundTask(response.aclose),
    )


@app.api_route("/mcp", methods=["POST", "GET", "DELETE"])
async def mcp_endpoint(request: Request) -> Response:
    # POST carries JSON-RPC messages; GET opens the server-to-client SSE stream;
    # DELETE ends the session. All three are part of Streamable HTTP, and all need a valid token.
    try:
        body = await read_body(request)
        identity = authenticate(request, body)
        check_session(request, identity)
        if request.method == "POST":
            check_message(body, identity)
    except Rejected as rejected:
        if rejected.code != UNAUTHORIZED_TOOL_CALL:
            log.info("%s /mcp rejected (%d): %s", request.method, rejected.code, rejected.message)
        return jsonrpc_error(rejected.id, rejected.code, rejected.message, rejected.status, rejected.headers)

    log.info("%s /mcp user=%s role=%s (%d bytes) -> downstream", request.method, identity.user, identity.role, len(body))
    return await forward(request, body, identity)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}
