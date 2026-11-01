"""Tiny raw JSON-RPC client for tests.

Starts a server as a child process and talks to it over stdin/stdout, exactly
like a real MCP host does. Raw JSON is used on purpose, so the tests check the
real wire format and not an SDK's interpretation of it.
"""

import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPLY_TIMEOUT = 30  # seconds

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"}},
}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}


def start(script: str = "server.py") -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(ROOT / script)],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
    )


def tool_call(request_id: int, name: str, arguments: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}}


def run_session(messages: list[dict | str], script: str = "server.py", null_id_replies: int | None = None) -> tuple[list[str], str]:
    """Send a full session (handshake + messages) and return (raw stdout lines, stderr).

    A message can be a dict (sent as JSON) or a str (sent as-is, for malformed
    input). By default each str is expected to get one reply with id null;
    pass null_id_replies to expect a different number.

    Like a real host, stdin stays open until every request has its response;
    closing it means "shut down" and the server would drop in-flight calls.
    """
    proc = start(script)
    all_messages = [INITIALIZE, INITIALIZED, *messages]
    pending = {m["id"] for m in all_messages if isinstance(m, dict) and "id" in m}
    if null_id_replies is None:
        null_id_replies = sum(isinstance(m, str) for m in all_messages)
    proc.stdin.write("".join((m if isinstance(m, str) else json.dumps(m)) + "\n" for m in all_messages))
    proc.stdin.flush()

    # Read stdout on a thread, so a missing reply makes the test fail at the
    # deadline instead of blocking forever.
    replies: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: [replies.put(line) for line in proc.stdout], daemon=True).start()

    lines, deadline = [], time.monotonic() + REPLY_TIMEOUT
    while (pending or null_id_replies > 0) and time.monotonic() < deadline:
        try:
            line = replies.get(timeout=0.1)
        except queue.Empty:
            continue
        lines.append(line.rstrip("\n"))
        try:
            reply_id = json.loads(line).get("id")
        except ValueError:
            continue  # keep the bad line; the test's purity check reports it
        if reply_id is None:
            null_id_replies -= 1
        pending.discard(reply_id)

    proc.stdin.close()  # end of session: the server shuts down
    try:
        proc.wait(timeout=REPLY_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    time.sleep(0.05)  # let the reader thread pick up the last lines
    while not replies.empty():
        lines.append(replies.get().rstrip("\n"))
    stderr = proc.stderr.read()
    proc.stdout.close()
    proc.stderr.close()
    return lines, stderr


class Client:
    """Long-lived connection: one request at a time, wait for its response."""

    def __init__(self) -> None:
        self.proc = start()
        self._next_id = 1
        self.request(INITIALIZE["method"], INITIALIZE["params"])
        self._send(INITIALIZED)

    def _send(self, message: dict) -> None:
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict | None = None) -> dict:
        request_id, self._next_id = self._next_id, self._next_id + 1
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"server closed stdout; stderr: {self.proc.stderr.read()}")
            message = json.loads(line)
            if message.get("id") == request_id:
                return message

    def call(self, name: str, arguments: dict) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=10)
        self.proc.stdout.close()
        self.proc.stderr.close()
