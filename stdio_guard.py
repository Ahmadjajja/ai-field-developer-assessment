"""Reserve the real stdout for JSON-RPC, before anything else can write to it.

With stdio transport, stdout IS the wire to the client: a single stray line
breaks it. claim() saves a private copy of stdout for the transport, then
points file descriptor 1 at stderr. From then on print(), sys.stdout.write,
raw os.write(1, ...) and child processes all land on stderr, buffered or not.

Standard library only, so it can run first, before importing anything that
might print at import time.
"""

import os
import sys

_wire_fd: int | None = None


def claim() -> int:
    """Claim stdout once and return the private file descriptor of the wire."""
    global _wire_fd
    if _wire_fd is None:
        sys.stdout.flush()
        _wire_fd = os.dup(sys.stdout.fileno())
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return _wire_fd
