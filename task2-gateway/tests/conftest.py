"""Test fixtures: real mock MCP server + real gateway processes on free ports."""

import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auth import create_token  # noqa: E402

PROTOCOL = "2025-11-25"
INIT_PARAMS = {"protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"}}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until_up(url: str, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"process exited early: {proc.stderr.read()}")
        try:
            httpx.get(url, timeout=0.5)
            return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError(f"{url} did not come up")


def spawn(args: list[str], env: dict, ready_url: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, *args], cwd=ROOT, env={**os.environ, **env},
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    wait_until_up(ready_url, proc)
    return proc


@dataclass
class Stack:
    gateway: str  # http://127.0.0.1:<port>/mcp
    mock: str  # http://127.0.0.1:<port>
    procs: list

    def executed(self) -> list[str]:
        return [c["tool"] for c in httpx.get(f"{self.mock}/debug/calls").json()]

    def received_headers(self) -> list[dict]:
        return httpx.get(f"{self.mock}/debug/headers").json()

    def reset(self) -> None:
        httpx.delete(f"{self.mock}/debug/calls")

    def gateway_log(self) -> str:
        return self.procs[1].stderr.read() if self.procs[1].poll() is not None else ""


def start_stack(json_response: bool = True, downstream_url: str | None = None) -> Stack:
    mock_port, gw_port = free_port(), free_port()
    mock = spawn(
        ["mock_mcp_server.py"],
        {"MOCK_MCP_PORT": str(mock_port), "MOCK_JSON_RESPONSE": "1" if json_response else "0"},
        f"http://127.0.0.1:{mock_port}/debug/calls",
    )
    gateway = spawn(
        ["-m", "uvicorn", "gateway:app", "--port", str(gw_port), "--log-level", "warning"],
        {"DOWNSTREAM_MCP_URL": downstream_url or f"http://127.0.0.1:{mock_port}/mcp"},
        f"http://127.0.0.1:{gw_port}/healthz",
    )
    return Stack(f"http://127.0.0.1:{gw_port}/mcp", f"http://127.0.0.1:{mock_port}", [mock, gateway])


def stop_stack(stack: Stack) -> None:
    for proc in stack.procs:
        proc.terminate()
        proc.wait(timeout=10)
        proc.stderr.close()


@pytest.fixture(scope="session")
def stack():
    s = start_stack()
    yield s
    stop_stack(s)


@pytest.fixture(scope="session")
def sse_stack():
    s = start_stack(json_response=False)
    yield s
    stop_stack(s)


@pytest.fixture(autouse=True)
def _clean_calls(request):
    for name in ("stack", "sse_stack"):
        if name in request.fixturenames:
            request.getfixturevalue(name).reset()


def token(role: str, user: str) -> str:
    return create_token(user, role)


# One HTTP connection pool shared by every test client, closed at the end of the run.
HTTP = httpx.Client(timeout=10)


@pytest.fixture(scope="session", autouse=True)
def _close_http():
    yield
    HTTP.close()


class McpClient:
    """Minimal Streamable HTTP client that speaks to the gateway with a Bearer token."""

    def __init__(self, url: str, bearer: str | None) -> None:
        self.url, self.bearer, self.session_id = url, bearer, None
        self.http = HTTP

    def headers(self, **extra) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": PROTOCOL}
        if self.bearer:
            h["Authorization"] = f"Bearer {self.bearer}"
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        h.update(extra)
        return h

    def post(self, body, raw: bytes | None = None, **headers) -> httpx.Response:
        content = raw if raw is not None else httpx.Request("POST", self.url, json=body).content
        return self.http.post(self.url, content=content, headers=self.headers(**headers))

    def initialize(self) -> "McpClient":
        r = self.post({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": INIT_PARAMS})
        assert r.status_code == 200, r.text
        self.session_id = r.headers["mcp-session-id"]
        self.post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return self

    def call(self, name: str, arguments: dict | None = None, id_: int = 1) -> httpx.Response:
        return self.post({"jsonrpc": "2.0", "id": id_, "method": "tools/call", "params": {"name": name, "arguments": arguments or {}}})


def body(response: httpx.Response) -> dict:
    """JSON-RPC message from a plain JSON or SSE response."""
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        data = [line[5:].strip() for line in response.text.splitlines() if line.startswith("data:")]
        return httpx.Response(200, content=data[-1]).json()
    return response.json()
