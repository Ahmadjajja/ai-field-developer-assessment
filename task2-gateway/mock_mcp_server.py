"""Mock downstream MCP server for Task 2.

A real MCP server (official SDK, Streamable HTTP transport) that the gateway
sits in front of. It has no security of its own on purpose: protecting it is
the gateway's job.

Tools:
  get_status, list_keys          normal tools, anyone may call them
  admin_reset_key, admin_delete_user   admin-only tools (names start with admin_)

It also records every tool call it actually executes, readable at
GET /debug/calls, so tests can prove that blocked calls never arrive here.

Run: uv run python mock_mcp_server.py   (listens on http://127.0.0.1:9000/mcp)
"""

import logging
import os
import sys
from datetime import datetime, timezone

from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse

HOST = os.environ.get("MOCK_MCP_HOST", "127.0.0.1")
PORT = int(os.environ.get("MOCK_MCP_PORT", "9000"))
# "1" (default): reply with plain JSON. "0": reply with SSE event streams, as many real servers do.
JSON_RESPONSE = os.environ.get("MOCK_JSON_RESPONSE", "1") == "1"

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="mock-mcp: %(message)s")
log = logging.getLogger("mock-mcp")

mcp = MCPServer(name="mock-key-service", log_level="WARNING")

# Every tool call that really ran here, newest last.
EXECUTED_CALLS: list[dict] = []


def _record(tool: str, **arguments: str) -> None:
    EXECUTED_CALLS.append({"tool": tool, "arguments": arguments, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    log.info("executed %s %s", tool, arguments)


@mcp.tool()
def get_status() -> dict:
    """Health of the key service."""
    _record("get_status")
    return {"status": "ok", "keys_active": 3}


@mcp.tool()
def list_keys() -> list[dict]:
    """List API keys (IDs only, never the secret values)."""
    _record("list_keys")
    return [{"key_id": "key-1", "owner": "billing"}, {"key_id": "key-2", "owner": "search"}, {"key_id": "key-3", "owner": "ops"}]


@mcp.tool()
def admin_reset_key(key_id: str) -> dict:
    """ADMIN: rotate an API key. Its old value stops working immediately."""
    _record("admin_reset_key", key_id=key_id)
    return {"key_id": key_id, "status": "rotated"}


@mcp.tool()
def admin_delete_user(user_id: str) -> dict:
    """ADMIN: permanently delete a user account."""
    _record("admin_delete_user", user_id=user_id)
    return {"user_id": user_id, "status": "deleted"}


@mcp.custom_route("/debug/calls", methods=["GET"])
async def debug_calls(request: Request) -> JSONResponse:
    """Test hook: which tool calls actually reached this server."""
    return JSONResponse(EXECUTED_CALLS)


@mcp.custom_route("/debug/calls", methods=["DELETE"])
async def clear_calls(request: Request) -> JSONResponse:
    EXECUTED_CALLS.clear()
    RECEIVED_HEADERS.clear()
    return JSONResponse({"cleared": True})


# Headers of every request that reached /mcp, newest last (test hook).
RECEIVED_HEADERS: list[dict] = []


@mcp.custom_route("/debug/headers", methods=["GET"])
async def debug_headers(request: Request) -> JSONResponse:
    return JSONResponse(RECEIVED_HEADERS)


class RecordHeaders:
    """ASGI middleware: remember the headers of each /mcp request."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and scope["path"] == "/mcp":
            RECEIVED_HEADERS.append({k.decode(): v.decode() for k, v in scope["headers"]})
        await self.app(scope, receive, send)


app = RecordHeaders(mcp.streamable_http_app(json_response=JSON_RESPONSE, host=HOST))

if __name__ == "__main__":
    import uvicorn

    log.info("listening on http://%s:%d/mcp", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
