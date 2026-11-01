"""LLM Gateway model router (Task 4).

POST /v1/chat/completions (OpenAI-compatible):

  1. Authenticate the tenant by API key ("Authorization: Bearer <key>").
  2. Estimate the request's tokens and reserve them in the tenant's sliding
     window (50,000 tokens / 60 s by default, stored in SQLite). Over the
     limit: 429 with Retry-After.
  3. Send it to the primary model. On 429 or no answer within 3000 ms (or the
     provider being down / failing with 5xx), fail over to the secondary.
  4. Settle the reservation with the real token usage, or release it if no
     provider could serve the request.

Every error, ours or a provider's, comes back in one shape and never contains
upstream bodies, stack traces or internal details:

  {"error": {"type": "rate_limit_exceeded", "message": "...", "request_id": "..."}}

Run: uv run uvicorn app:app --port 8200
"""

import json
import logging
import math
import os
import sys
import uuid
from contextlib import asynccontextmanager

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rate_limiter import RateLimited, RequestTooLarge, SlidingWindowLimiter, estimate_tokens
from router import AllProvidersFailed, ModelRouter, Provider, ProviderRejected
from tenants import TenantStore

DB_PATH = os.environ.get("ROUTER_DB_PATH", "router.db")
TOKENS_PER_WINDOW = int(os.environ.get("TOKENS_PER_MINUTE", "50000"))
WINDOW_SECONDS = float(os.environ.get("WINDOW_SECONDS", "60"))
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "512"))
MAX_BODY_BYTES = 1_000_000

PROVIDERS = [
    Provider(
        name="primary",
        base_url=os.environ.get("PRIMARY_BASE_URL", "http://127.0.0.1:9201/v1"),
        model=os.environ.get("PRIMARY_MODEL", "primary-model"),
        api_key=os.environ.get("PRIMARY_API_KEY"),
        timeout=int(os.environ.get("PRIMARY_TIMEOUT_MS", "3000")) / 1000,
    ),
    Provider(
        name="secondary",
        base_url=os.environ.get("SECONDARY_BASE_URL", "http://127.0.0.1:9202/v1"),
        model=os.environ.get("SECONDARY_MODEL", "secondary-model"),
        api_key=os.environ.get("SECONDARY_API_KEY"),
        timeout=int(os.environ.get("SECONDARY_TIMEOUT_MS", "10000")) / 1000,
    ),
]

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="router: %(message)s")
log = logging.getLogger("router")
logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.limiter = SlidingWindowLimiter(DB_PATH, limit=TOKENS_PER_WINDOW, window=WINDOW_SECONDS)
    app.state.tenants = TenantStore(DB_PATH)
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as http:
        app.state.router = ModelRouter(PROVIDERS, http)
        yield


app = FastAPI(title="LLM Gateway Model Router", lifespan=lifespan)


# --- Standardized errors ----------------------------------------------------------------


class GatewayError(Exception):
    def __init__(self, status: int, kind: str, message: str, headers: dict | None = None) -> None:
        super().__init__(message)
        self.status, self.kind, self.message, self.headers = status, kind, message, headers or {}


def error_response(request: Request, status: int, kind: str, message: str, headers: dict | None = None) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
    return JSONResponse(
        {"error": {"type": kind, "message": message, "request_id": request_id}},
        status_code=status,
        headers={"X-Request-ID": request_id, **(headers or {})},
    )


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex
    response = await call_next(request)
    response.headers.setdefault("X-Request-ID", request.state.request_id)
    return response


@app.exception_handler(GatewayError)
async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
    return error_response(request, exc.status, exc.kind, exc.message, exc.headers)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    kinds = {404: "not_found", 405: "method_not_allowed"}
    return error_response(request, exc.status_code, kinds.get(exc.status_code, "invalid_request"),
                          "Not found" if exc.status_code == 404 else "Request not allowed")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response(request, 400, "invalid_request", "Invalid request")


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # The traceback goes to our logs only; the client gets a generic message and the request id.
    log.exception("unexpected error request_id=%s", getattr(request.state, "request_id", "-"))
    return error_response(request, 500, "internal_error", "An internal error occurred")


# --- Request handling ----------------------------------------------------------------------


def authenticate(request: Request) -> str:
    scheme, _, key = request.headers.get("authorization", "").partition(" ")
    tenant = request.app.state.tenants.tenant_for(key.strip()) if scheme.lower() == "bearer" and key.strip() else None
    if tenant is None:
        raise GatewayError(401, "authentication_error", "Invalid or missing API key", {"WWW-Authenticate": "Bearer"})
    return tenant


async def parse_body(request: Request) -> dict:
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise GatewayError(413, "request_too_large", "Request body too large")
    try:
        body = json.loads(raw)
    except ValueError:
        raise GatewayError(400, "invalid_request", "Request body must be JSON") from None
    if not isinstance(body, dict):
        raise GatewayError(400, "invalid_request", "Request body must be a JSON object")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
        raise GatewayError(400, "invalid_request", "'messages' must be a non-empty list of objects")
    max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise GatewayError(400, "invalid_request", "'max_tokens' must be a positive integer")
    if body.get("stream"):
        raise GatewayError(400, "invalid_request", "Streaming is not supported by this endpoint")
    body["max_tokens"] = max_tokens
    return body


def actual_tokens(completion: dict, estimate: int, max_tokens: int) -> int:
    """Tokens the request really used: the provider's count, or our estimate if it gave none."""
    usage = completion.get("usage") or {}
    total = usage.get("total_tokens")
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        return total
    # No usage reported: prompt estimate + ~4 characters per completion token, never above the reservation.
    text = "".join(str((c.get("message") or {}).get("content", "")) for c in completion.get("choices", []))
    return min(estimate, (estimate - max_tokens) + math.ceil(len(text) / 4))


def rate_limit_headers(limit: int, remaining: int, reset_after: float) -> dict:
    return {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(max(0, remaining)),
        "X-RateLimit-Reset": str(math.ceil(reset_after)),
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    tenant = authenticate(request)
    body = await parse_body(request)
    limiter: SlidingWindowLimiter = request.app.state.limiter
    estimate = estimate_tokens(body["messages"], body["max_tokens"])

    try:
        # SQLite calls are blocking: run them in a worker thread so the event loop keeps serving others.
        reservation = await anyio.to_thread.run_sync(limiter.reserve, tenant, estimate)
    except RequestTooLarge:
        raise GatewayError(400, "request_too_large",
                           f"This request needs about {estimate} tokens; the limit is {limiter.limit} per window") from None
    except RateLimited as exc:
        retry = max(1, math.ceil(exc.retry_after))  # never "retry in 0s": that invites a tight retry loop
        log.info("rate limited tenant=%s used=%d estimate=%d", tenant, exc.used, estimate)
        raise GatewayError(429, "rate_limit_exceeded", f"Token rate limit exceeded; retry in {retry}s",
                           {"Retry-After": str(retry), **rate_limit_headers(exc.limit, exc.limit - exc.used, exc.retry_after)}) from None

    settled = False
    try:
        routed = await request.app.state.router.complete(body)
        used = actual_tokens(routed.completion, estimate, body["max_tokens"])
        await anyio.to_thread.run_sync(limiter.settle, reservation, used)
        settled = True
    except ProviderRejected as exc:
        raise GatewayError(400, "invalid_request", "The model provider rejected the request") from exc
    except AllProvidersFailed as exc:
        log.error("all providers failed tenant=%s attempts=%s", tenant, exc.attempts)
        raise GatewayError(503, "upstream_unavailable", "No model provider is available right now; please retry") from None
    finally:
        if not settled:
            # Not served (or the client went away): don't charge the tenant for it.
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(limiter.release, reservation)

    log.info("served tenant=%s provider=%s tokens=%d (estimated %d) attempts=%s",
             tenant, routed.provider, used, estimate, routed.attempts)
    remaining = reservation.remaining + (estimate - used)
    return JSONResponse(routed.completion, headers={
        "X-Model-Provider": routed.provider,
        **rate_limit_headers(limiter.limit, remaining, reservation.reset_after or WINDOW_SECONDS),
    })


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}
