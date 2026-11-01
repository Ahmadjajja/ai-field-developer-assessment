"""Customer support MCP server (Task 1).

Two tools over stdio: get_customer_record and trigger_refund.
Bad input is rejected with a JSON-RPC -32602 (Invalid params) error before
any tool code runs. stdout carries JSON-RPC only; all logs go to stderr.
"""

import stdio_guard

if __name__ == "__main__":
    # First thing, before other imports: nothing printed from here on can reach the wire.
    stdio_guard.claim()

import io  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import queue  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from decimal import Decimal  # noqa: E402
from typing import Annotated, Literal  # noqa: E402

import anyio  # noqa: E402
from mcp.server import Server, ServerRequestContext  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from mcp.shared.exceptions import MCPError  # noqa: E402
from mcp.shared.message import SessionMessage  # noqa: E402
from mcp_types import (  # noqa: E402
    INVALID_PARAMS,
    INVALID_REQUEST,
    PARSE_ERROR,
    CallToolRequestParams,
    CallToolResult,
    ErrorData,
    JSONRPCError,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    ToolAnnotations,
)
from pydantic import AfterValidator, Field, StringConstraints, ValidationError  # noqa: E402

import data  # noqa: E402

# --- Logging: stderr only ----------------------------------------------------


class JsonFormatter(logging.Formatter):
    """One JSON object per line, easy for log tools to parse."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


class NonBlockingHandler(logging.Handler):
    """Hands log records to a background thread that writes them to stderr.

    If the host never reads stderr, the pipe fills up and writes block. Without
    this, the whole server would freeze. With it, only the writer thread waits:
    requests keep being answered, and once the buffer is full new log lines are
    dropped (and counted) instead of blocking.
    """

    def __init__(self, target: logging.Handler, capacity: int = 10_000) -> None:
        super().__init__()
        self.target = target
        self.pending: queue.Queue[logging.LogRecord] = queue.Queue(capacity)
        self.dropped = 0
        threading.Thread(target=self._write_forever, name="log-writer", daemon=True).start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.pending.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def _write_forever(self) -> None:
        while True:
            record = self.pending.get()
            self.target.handle(record)
            self.pending.task_done()

    def drain(self, timeout: float) -> None:
        """Give queued lines a moment to reach stderr before the process exits."""
        deadline = time.monotonic() + timeout
        while self.pending.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)


def setup_logging() -> tuple[logging.Logger, NonBlockingHandler]:
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(JsonFormatter())
    handler = NonBlockingHandler(stderr_handler)
    logger = logging.getLogger("customer_support")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger, handler


log, log_handler = setup_logging()


# --- Input rules (Pydantic) -------------------------------------------------

# "CUST-" followed by exactly 5 ASCII digits.
# [0-9] instead of \d, because \d also matches non-ASCII digits like "١٢٣٤٥".
CustomerId = Annotated[
    str,
    Field(pattern=r"^CUST-[0-9]{5}$", description="Customer ID in the format CUST-XXXXX (5 digits), e.g. CUST-10001"),
]

# Positive float. strict=True rejects strings like "40" and booleans;
# allow_inf_nan=False rejects NaN and Infinity.
Amount = Annotated[
    float,
    Field(strict=True, gt=0, allow_inf_nan=False, description="Refund amount, a positive number"),
]

ALLOWED_CONTROL_CHARS = {"\t", "\n", "\r"}


def _meaningful_reason(value: str) -> str:
    """Count only characters a person can see, and refuse hidden control characters.

    Without this, 12 zero-width spaces ("\\u200b") would pass a plain length check.
    """
    if any(unicodedata.category(ch) == "Cc" and ch not in ALLOWED_CONTROL_CHARS for ch in value):
        raise ValueError("reason must not contain control characters")
    visible = "".join(ch for ch in value if unicodedata.category(ch) != "Cf").strip()
    if len(visible) < 10:
        raise ValueError("reason must have at least 10 visible characters")
    return value


# Leading/trailing spaces are stripped first, so "          " (10 spaces) fails;
# invisible characters (zero-width spaces etc.) don't count towards the 10.
Reason = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=10),
    AfterValidator(_meaningful_reason),
    Field(description="Why the refund is issued, at least 10 visible characters"),
]

# Optional category from a fixed list. Not part of the task's required inputs,
# so callers that send only customer_id, amount and reason still work.
ReasonCode = Annotated[
    Literal["duplicate_charge", "billing_error", "service_outage", "cancellation", "goodwill"],
    Field(description="Optional refund category; omit it or send one of the listed values"),
]


# --- Tools ------------------------------------------------------------------
# MCPServer is used as the tool registry: it turns each function into a tool
# with a JSON schema and validates arguments with Pydantic.

tools = MCPServer(name="customer-support", log_level="WARNING")


@tools.tool(annotations=ToolAnnotations(read_only_hint=True))
def get_customer_record(customer_id: CustomerId) -> dict:
    """Look up a customer's record by their ID (format: CUST-XXXXX)."""
    return data.get_customer(customer_id)


# Business rules. The input schema follows the spec exactly ("positive float",
# "reason min 10"); these extra limits are valid input the tool refuses, so they
# come back as isError results, not protocol errors.
MAX_REASON_LENGTH = 1000


@tools.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True))
def trigger_refund(customer_id: CustomerId, amount: Amount, reason: Reason, reason_code: ReasonCode = None) -> dict:  # type: ignore[assignment]
    """Issue a refund to a customer. Moves real money: only use for a verified billing problem.

    The amount must be in whole cents (at least 0.01) and cannot exceed the
    customer's refundable balance (see get_customer_record).
    """
    money = Decimal(str(amount))  # str() first, so 0.1 stays exactly 0.1
    if money < data.CENT:
        raise ToolError("Refund amount must be at least 0.01")
    if money.as_tuple().exponent < -2:
        raise ToolError("Refund amount must be in whole cents (at most 2 decimal places)")
    if len(reason) > MAX_REASON_LENGTH:
        raise ToolError(f"Reason must be at most {MAX_REASON_LENGTH} characters")

    try:
        refund = data.refund(customer_id, money, reason, reason_code)
    except data.RefundRefused as exc:
        raise ToolError(str(exc)) from exc

    log.info("refund processed", extra={"fields": {"refund_id": refund["refund_id"], "customer_id": customer_id, "amount": refund["amount"], "reason_code": reason_code}})
    return refund


# --- Protocol: how tools/list and tools/call are answered ---------------------


def invalid_params(tool: str, errors: list[dict]) -> MCPError:
    return MCPError(code=INVALID_PARAMS, message=f"Invalid params for tool '{tool}'", data=errors)


async def handle_list_tools(ctx: ServerRequestContext, params: PaginatedRequestParams | None) -> ListToolsResult:
    return ListToolsResult(tools=await tools.list_tools())


async def handle_call_tool(ctx: ServerRequestContext, params: CallToolRequestParams) -> CallToolResult:
    """Validate, run the tool and write an audit log line for every call.

    Bad input            -> JSON-RPC error -32602 (the request itself is wrong)
    Tool says no         -> result with isError=true (the model can read why)
    Unexpected crash     -> isError=true with a generic message; traceback to stderr only
    """
    name, arguments = params.name, params.arguments or {}
    start = time.perf_counter()
    outcome = "ok"
    try:
        schemas = {tool.name: tool.input_schema for tool in await tools.list_tools()}
        if name not in schemas:
            # The MCP spec reports an unknown tool as -32602 (Invalid params).
            raise MCPError(code=INVALID_PARAMS, message=f"Unknown tool: {name}")

        # Unexpected extra fields are rejected instead of silently ignored.
        unexpected = sorted(set(arguments) - set(schemas[name].get("properties", {})))
        if unexpected:
            raise invalid_params(name, [{"field": field, "message": "Unexpected field"} for field in unexpected])

        try:
            return await tools.call_tool(name, arguments)
        except ToolError as exc:
            if isinstance(exc.__cause__, ValidationError):
                # List which fields failed and why, without echoing the rejected values back.
                errors = [
                    {"field": ".".join(str(part) for part in err["loc"]), "message": err["msg"]}
                    for err in exc.__cause__.errors()
                ]
                raise invalid_params(name, errors) from exc
            if exc.__cause__ is not None and not isinstance(exc.__cause__, ToolError):
                outcome = "crash"
                log.error("tool crashed", exc_info=exc.__cause__, extra={"fields": {"tool": name}})
            else:
                outcome = "tool_error"
            # ToolError messages are safe by design: the SDK hides details of unexpected crashes.
            return CallToolResult(content=[TextContent(type="text", text=str(exc))], is_error=True)
    except MCPError as exc:
        outcome = f"rejected:{exc.code}"
        raise
    finally:
        customer_id = arguments.get("customer_id")
        log.info(
            "tool call",
            extra={"fields": {
                "tool": name,
                "customer_id": customer_id[:20] if isinstance(customer_id, str) else None,
                "outcome": outcome,
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            }},
        )


# Only tool handlers are registered, so the server advertises the tools capability only.
server = Server(name="customer-support", on_list_tools=handle_list_tools, on_call_tool=handle_call_tool)


# --- Startup ------------------------------------------------------------------


def malformed_message_error(exc: Exception) -> SessionMessage:
    """JSON-RPC 2.0 reply for a line that isn't a valid message (the SDK would drop it silently).

    Not JSON at all              -> -32700 Parse error
    JSON, but not a valid message -> -32600 Invalid Request
    The id can't be trusted from a malformed message, so it's null, as the spec requires.
    """
    not_json = isinstance(exc, ValidationError) and any(err["type"] == "json_invalid" for err in exc.errors())
    error = ErrorData(code=PARSE_ERROR, message="Parse error") if not_json else ErrorData(code=INVALID_REQUEST, message="Invalid Request")
    log.info("malformed message", extra={"fields": {"code": error.code}})
    return SessionMessage(JSONRPCError(jsonrpc="2.0", id=None, error=error))


def is_blank_line(exc: Exception) -> bool:
    if not isinstance(exc, ValidationError):
        return False
    return all(err["type"] == "json_invalid" and str(err.get("input", "x")).strip() == "" for err in exc.errors())


async def serve_stdio() -> None:
    # The transport writes JSON-RPC to the private copy of stdout from stdio_guard;
    # the regular stdout (fd 1) already points at stderr.
    wire = anyio.wrap_file(io.TextIOWrapper(os.fdopen(stdio_guard.claim(), "wb"), encoding="utf-8"))
    async with stdio_server(stdout=wire) as (read_stream, write_stream):
        # Answer malformed lines here; pass every valid message on to the server.
        to_server, from_transport = anyio.create_memory_object_stream[SessionMessage | Exception](0)

        async def filter_malformed() -> None:
            async with to_server:
                async for item in read_stream:
                    if not isinstance(item, Exception):
                        await to_server.send(item)
                    elif not is_blank_line(item):  # blank lines carry no message: skip them quietly
                        await write_stream.send(malformed_message_error(item))

        async with anyio.create_task_group() as tg:
            tg.start_soon(filter_malformed)
            await server.run(from_transport, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    log.info("server starting", extra={"fields": {"transport": "stdio"}})
    try:
        anyio.run(serve_stdio)
    finally:
        log_handler.drain(timeout=2)
