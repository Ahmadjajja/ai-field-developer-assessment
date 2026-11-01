# AI Field Developer Assessment

Four tasks on MCP servers, MCP and LLM gateways, security guardrails and
system integration. All in Python 3.12 with [uv](https://docs.astral.sh/uv/);
each task is a self-contained project with its own tests and README.

| Task | Folder | What it is | Tests |
|---|---|---|---|
| 1. MCP server with strict validation & transport handling | [`./`](#task-1-customer-support-mcp-server) (this folder) | stdio MCP server, Pydantic validation, JSON-RPC errors, stdout isolation | 81 |
| 2. MCP security gateway (tool filtering & auth) | [`task2-gateway/`](task2-gateway/README.md) | Reverse proxy: JWT roles, `admin_*` tools blocked with `-32001`, red-teamed | 50 |
| 3. LLM gateway streaming guardrail (PII redaction) | [`task3-llm-redaction/`](task3-llm-redaction/README.md) | Streams answers while redacting emails, SSNs and cards in real time | 77 |
| 4. Rate limiting & model fallback router | [`task4-model-router/`](task4-model-router/README.md) | 50k tokens/min per tenant in SQLite, 429/3 s timeout failover, sanitized errors | 47 |

Run every suite:

```bash
uv run pytest -q
for t in task2-gateway task3-llm-redaction task4-model-router; do (cd $t && uv run pytest -q); done
```

No API keys are needed: every model provider and downstream server is a local
mock, and the upstream URLs are configurable to point at real ones.

---

# Task 1: Customer Support MCP Server

An MCP server in Python, built with the official SDK (`mcp` 2.x), that exposes two
customer-support tools over **stdio** with strict **Pydantic** validation.

| Tool | Inputs | Annotations |
|---|---|---|
| `get_customer_record` | `customer_id`: `CUST-XXXXX` | read-only |
| `trigger_refund` | `customer_id`: `CUST-XXXXX`, `amount`: positive float, `reason`: string, min 10 chars, optional `reason_code` | destructive |

## Quick start

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.12 for the project automatically.

```bash
uv sync                      # install dependencies
uv run pytest -q             # run the 81 end-to-end tests
uv run python server.py      # start the server (it waits for JSON-RPC on stdin)
```

**Try it in MCP Inspector** (visual UI that shows every JSON-RPC message):

```bash
npx @modelcontextprotocol/inspector uv run python server.py
```

**Use it from Claude Code:**

```bash
claude mcp add customer-support -- uv run --directory "$(pwd)" python server.py
```

Then ask: *"What's CUST-10001's refundable balance? Refund them $40 for a duplicate charge."*

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "customer-support": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/project", "python", "server.py"]
    }
  }
}
```

## Mock data

`data.py` stands in for a real CRM/billing system and lives in memory, so it
resets on restart. Three named customers are seeded (`CUST-10001` to
`CUST-10003`). **Any other valid ID** (e.g. `CUST-12345`) gets a generated
record with a refundable balance of 500.00 on first use, so every well-formed
ID works for testing.

Money is stored as `Decimal` (exact cents), so three refunds of 0.10 total exactly
0.30, not 0.30000000000000004. Checking the balance, paying out and recording the
refund happen under one lock, because the SDK runs tools in worker threads and
concurrent refunds must never overspend a balance.

## Project layout

```
server.py              Validation rules, tools, protocol handlers (error mapping + audit log), stdio startup
stdio_guard.py         Reserves the real stdout for JSON-RPC before anything else is imported
data.py                Mock CRM/billing store (stand-in for real company APIs)
tests/test_server.py   End-to-end tests: the server runs as a real child process over stdio
tests/mcp_client.py    Minimal raw JSON-RPC client used by the tests
tests/noisy_server.py  Server plus deliberately noisy and crashing code, for the isolation tests
tests/slow_server.py   Server with a slow mock payment provider, for the concurrency test
practice/              Minimal ~60-line version of the same server (core requirements only)
```

`server.py` uses only public SDK APIs. `MCPServer` is the tool registry: it
turns each function into a tool with a JSON schema and validates arguments
with Pydantic. The low-level `Server` is wired with explicit `tools/list` and
`tools/call` handlers, which is where errors are mapped and every call is
logged. Because only tool handlers are registered, the server advertises the
`tools` capability only.

## Validation

All input rules are declared once, as Pydantic types, and serve two purposes:
they're published in `tools/list` so the model knows the rules up front, and
they're enforced on every `tools/call` before any tool code runs.

| Field | Rule | Edge cases covered |
|---|---|---|
| `customer_id` | `^CUST-[0-9]{5}$` | wrong case, too short/long, letters, leading space, trailing newline, non-ASCII digits (`[0-9]` instead of `\d`), non-strings |
| `amount` | strict number, `> 0`, no NaN/Infinity | `0`, negatives, `"40"`, `true`, `null`, `NaN`, `±Infinity`, lists/objects |
| `reason` | whitespace stripped, then at least 10 **visible** chars; no control characters (tab/newline allowed) | empty, 9 chars, only spaces, padded short text, 12 zero-width spaces, NUL/BEL, non-strings. Unicode, emoji and newlines inside are fine. |
| `reason_code` *(optional)* | one of `duplicate_charge`, `billing_error`, `service_outage`, `cancellation`, `goodwill` | unknown values, wrong case, empty, `null`, non-strings |
| *(any)* | unknown fields rejected | e.g. an injected `"admin": true` |

**Assumptions:**
- `CUST-XXXXX` is read as `CUST-` plus **5 digits**, the usual meaning of the
  X placeholder in ID formats. It's a one-line change in `CustomerId` if
  letters should be allowed.
- `amount` follows the spec exactly ("positive float"). Integers like `40` are
  accepted, since JSON has a single number type. Limits on size are business
  rules, not input rules: an amount above the refundable balance is valid input
  that the tool refuses (see the error model).

**Business rules** (valid input the tool refuses, returned as `isError: true`,
so the input schema still matches the spec exactly):

| Rule | Example refused |
|---|---|
| Amount at least 0.01 | `0.001`, `5e-324` |
| Amount in whole cents | `10.555` |
| Amount not above the refundable balance | `15000` on a 500.00 balance |
| Reason at most 1000 characters (not echoed back) | a 1 MB reason |

**Why `reason_code`?** A length rule can't tell a real reason from `"i don't know"`
(12 characters, so it passes). A fixed category makes refunds reportable and
easier to audit, and the model can only pick from the list. It's **optional**, so
calls with exactly the three fields the task specifies still work. In
production it would become required, with `reason` kept for free-text detail.

## Error model

There are two kinds of failure, and they're reported differently on purpose:

| Situation | Response | Why |
|---|---|---|
| Bad input (schema violation) | JSON-RPC error **`-32602` Invalid params**, with `data: [{field, message}]` listing every failing field | The request itself is malformed. That's a protocol-level error. |
| Unknown tool | JSON-RPC error **`-32602`** | As specified by MCP for unknown tools |
| Unknown method (e.g. `tools/explode`) | JSON-RPC error **`-32601` Method not found** | Handled by the SDK |
| Line that isn't JSON | JSON-RPC error **`-32700` Parse error** (id `null`) | The SDK drops these silently; `serve_stdio` answers them as JSON-RPC 2.0 requires. The session keeps working. |
| JSON, but not a valid JSON-RPC message (e.g. no `method`) | JSON-RPC error **`-32600` Invalid Request** (id `null`) | Same as above |
| Blank line | Ignored | It carries no message, so there's nothing to answer |
| Valid input, but a business rule says no (see the table above) | Normal result with **`isError: true`** | The model can read the reason and react |
| Unexpected crash | `isError: true` with a generic message; the full traceback goes to stderr only | Internal details never reach the client |

Error responses never echo the rejected values back. They name the field and the rule only.

The SDK's default is to report schema violations as `isError: true` results.
The task asks for standard JSON-RPC error codes, so `handle_call_tool`
converts them into `-32602` errors.

## stdio isolation

With stdio transport, stdout **is** the JSON-RPC wire. A single stray line breaks the client.

1. **Logs go to stderr only**, as one JSON object per line: an audit entry for
   every tool call (tool, customer, outcome, duration), every processed refund,
   and full tracebacks for crashes. Logging never blocks: lines go through a
   bounded queue to a background writer. If the host never reads stderr, the
   pipe fills up, but only the writer waits; requests keep being answered, and
   overflow lines are dropped instead of freezing the server.
2. **stdout is claimed at the file-descriptor level, first thing at startup**
   (`stdio_guard.claim()`, before any other import). A private copy of fd 1
   becomes the wire, and fd 1 itself is pointed at stderr. So `print()`,
   `sys.stdout.write`, raw `os.write(1, …)`, child processes and libraries
   printing at import time all land on stderr, buffered or not.

The SDK 2.x already diverts fd 1 while serving, but only from when the
transport starts until it stops. Output from startup, and buffered `print()`
output flushed at exit, still reached stdout. With the SDK's default startup,
the isolation test catches stray lines on stdout; with `stdio_guard`, 0 reach it.

## Tests

`uv run pytest -q` runs 81 tests. Each one spawns the server as a real child
process and speaks raw JSON-RPC over its stdin/stdout, so they test the actual
wire format the way a host sees it.

- protocol: tool listing, schemas, annotations, capabilities, `-32700`/`-32600` for malformed lines, unknown tool and unknown method codes
- every validation edge case in the table above
- business rules: balance check, minimum and whole-cent amounts, reason size, exact money math, any valid ID works
- concurrency: 20 simultaneous refunds against a slow payment provider never overspend the balance
- resilience: the server keeps answering when the host never reads stderr; blank lines are ignored
- error hygiene: rejected values aren't echoed back; a crash returns a generic error and logs the traceback to stderr
- stdio: every stdout line is valid JSON-RPC, logs are on stderr, and stray output from startup code, tool code and child processes can't reach stdout

Each fix from QA was mutation-tested: undoing it (removing the lock, using a
blocking logger, dropping the visible-character check, using `Decimal(float)`,
answering blank lines) makes its test fail.

## Known limitations (SDK behavior)

- A request with `"id": null` gets no reply. MCP requires string or number ids, and the SDK drops these.
- Ctrl-C (SIGINT) only stops the server once stdin closes, because the SDK's stdin reader blocks. Hosts stop servers by closing stdin or with SIGTERM, which both exit cleanly.
- A tool call sent before `initialize` returns `-32602` from the SDK's pre-initialization check.
- Duplicate JSON keys: the last one wins (`"amount": -5, "amount": 5` refunds 5). That's standard JSON parsing, but a gateway that reads the first key would see a different value. Validate on the server, never only at a proxy.

## Path to production

This is a take-home with mock data. To run it for real users:

- **Real backend:** replace `data.py` with calls to the existing billing/CRM APIs (and return "not found" as an `isError` result). The tools and validation stay the same.
- **Idempotency:** accept an `idempotency_key` on `trigger_refund`, so a retried call can't refund twice.
- **Authorization:** check who is calling (e.g. an OAuth token over Streamable HTTP, or behind an MCP gateway that filters tools by role) and limit refund amounts per role.
- **Human in the loop:** the `destructiveHint` annotation already lets hosts ask the user before running `trigger_refund`. Large refunds could also require approval.
- **Transport:** stdio suits a local, single user. A shared deployment would use Streamable HTTP, with TLS, rate limits and the audit log shipped to a central store.
- **Observability:** the JSON stderr logs are ready for any log pipeline. Add request IDs and metrics (call counts, rejection rates).
