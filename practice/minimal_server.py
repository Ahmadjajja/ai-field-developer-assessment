"""Minimal Task 1 server, for practising the 60-minute live-coding version.

The full submission is ../server.py. This is the smallest version that still
meets every requirement: official SDK, two tools, Pydantic validation, -32602
for bad input, stdio transport and logs on stderr only.

Run: uv run python practice/minimal_server.py
"""

import logging
import sys
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_PARAMS
from pydantic import Field, StringConstraints, ValidationError

# 1. Logs go to stderr only: stdout is the JSON-RPC wire.
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger("minimal")

# 2. Input rules, declared once with Pydantic.
CustomerId = Annotated[str, Field(pattern=r"^CUST-[0-9]{5}$")]
Amount = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
Reason = Annotated[str, StringConstraints(strip_whitespace=True, min_length=10)]


# 3. Turn validation failures into JSON-RPC -32602 (the SDK default is an isError result).
class StrictServer(MCPServer):
    async def call_tool(self, name: str, arguments: dict[str, Any], context: Any = None) -> Any:
        try:
            return await super().call_tool(name, arguments, context)
        except ToolError as exc:
            if isinstance(exc.__cause__, ValidationError):
                fields = [".".join(map(str, e["loc"])) for e in exc.__cause__.errors()]
                raise MCPError(code=INVALID_PARAMS, message="Invalid params", data=fields) from exc
            raise


mcp = StrictServer(name="customer-support-minimal")


# 4. The two tools.
@mcp.tool()
def get_customer_record(customer_id: CustomerId) -> dict:
    """Look up a customer by ID (CUST-XXXXX)."""
    log.info("get_customer_record %s", customer_id)
    return {"customer_id": customer_id, "name": "Test Customer", "balance": 500.0}


@mcp.tool()
def trigger_refund(customer_id: CustomerId, amount: Amount, reason: Reason) -> dict:
    """Refund a customer. Reason must be at least 10 characters."""
    log.info("trigger_refund %s %.2f", customer_id, amount)
    return {"customer_id": customer_id, "amount": amount, "status": "processed"}


# 5. stdio transport.
if __name__ == "__main__":
    mcp.run(transport="stdio")
