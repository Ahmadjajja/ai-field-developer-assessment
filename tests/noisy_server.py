"""The real server plus deliberately noisy code.

Used by the stdout-purity test: it prints at startup, and its tool prints,
writes straight to the stdout file descriptor and starts a child process that
prints. None of it may reach the JSON-RPC wire.
"""

import stdio_guard

stdio_guard.claim()

import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import anyio  # noqa: E402

print("stray print at startup, before the transport starts")

import server  # noqa: E402


@server.tools.tool()
def noisy_tool() -> str:
    """Make as much stray stdout output as possible."""
    print("stray print()")
    sys.stdout.write("stray sys.stdout.write without newline")
    os.write(1, b"stray os.write to fd 1\n")
    subprocess.run(["echo", "stray output from a child process"], check=True)
    return "done"


@server.tools.tool()
def crashing_tool() -> str:
    """Fail with an unexpected exception carrying internal details."""
    raise RuntimeError("db password=hunter2 at /srv/internal/billing.py")


if __name__ == "__main__":
    try:
        anyio.run(server.serve_stdio)
    finally:
        server.log_handler.drain(timeout=2)
