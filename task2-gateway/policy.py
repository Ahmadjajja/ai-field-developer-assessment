"""Tool-level authorization policy: who may call which tool.

Rule: tools whose name starts with "admin_" may only be called by the admin
role. Every other method and tool passes through.

The name is normalized before the check, so lookalikes are treated as admin
tools too: "ADMIN_reset_key", fullwidth "ａdmin_reset_key", " admin_x" or
"admin​_x". The downstream server might match names loosely; the gateway
must never be the looser of the two.
"""

import unicodedata

from auth import Identity

ADMIN_TOOL_PREFIX = "admin_"


def normalize(name: str) -> str:
    name = unicodedata.normalize("NFKC", name)  # fullwidth/compatibility forms -> plain ASCII
    name = "".join(ch for ch in name if unicodedata.category(ch) != "Cf")  # drop zero-width/format chars
    return name.strip().casefold()


def is_admin_tool(name: str) -> bool:
    return normalize(name).startswith(ADMIN_TOOL_PREFIX)


def blocked_tool(message: dict, identity: Identity) -> str | None:
    """Name of the tool this JSON-RPC message may not call, or None if it's allowed."""
    if identity.role == "admin":
        return None
    if not isinstance(message.get("method"), str) or normalize(message["method"]) != "tools/call":
        return None  # tools/list, initialize, notifications... are forwarded as they are
    params = message.get("params")
    name = params.get("name") if isinstance(params, dict) else None
    if isinstance(name, str) and is_admin_tool(name):
        return name
    return None
