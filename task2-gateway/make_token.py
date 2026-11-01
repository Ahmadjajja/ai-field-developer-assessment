"""Create a test token for the gateway.

Usage:
  uv run python make_token.py admin            # token for user "alice" with role admin
  uv run python make_token.py viewer bob 600   # role, user, lifetime in seconds
"""

import sys

from auth import create_token

if __name__ == "__main__":
    role = sys.argv[1] if len(sys.argv) > 1 else "viewer"
    user = sys.argv[2] if len(sys.argv) > 2 else "alice"
    ttl = int(sys.argv[3]) if len(sys.argv) > 3 else 3600
    print(create_token(user, role, ttl))
