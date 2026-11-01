# MCP Security Gateway (Task 2)

An HTTP/JSON-RPC reverse proxy that sits between an AI agent and a downstream
MCP server. It authenticates every request with a Bearer token, reads the
user's role from it, and enforces **tool-level authorization**: tools named
`admin_*` can only be called by admins. Anyone else gets JSON-RPC error
**`-32001` Unauthorized Tool Call** straight from the gateway, and the
downstream server never sees the request.

```
AI agent ──HTTP + Bearer JWT──►  Gateway :8000  ──HTTP──►  MCP server :9000
                                    │
                                    ├─ 401  bad / missing token
                                    ├─ -32001  viewer calls admin_* (blocked here)
                                    └─ tools/list & allowed calls → forwarded, response streamed back
```

Built in Python with FastAPI, httpx and PyJWT. The downstream mock is a real
MCP server (official SDK, Streamable HTTP transport).

## Quick start

```bash
uv sync
uv run pytest -q                                   # 50 end-to-end tests

uv run python mock_mcp_server.py                   # terminal 1: MCP server on :9000
uv run uvicorn gateway:app --port 8000             # terminal 2: gateway on :8000
```

Make tokens and try it (terminal 3):

```bash
VIEWER=$(uv run python make_token.py viewer bob)
ADMIN=$(uv run python make_token.py admin alice)
H='-H Content-Type:application/json -H Accept:application/json,text/event-stream'

# Open a session as the viewer and note the Mcp-Session-Id response header
curl -si $H -H "Authorization: Bearer $VIEWER" localhost:8000/mcp \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'

SID=<session id from above>
curl -s $H -H "Authorization: Bearer $VIEWER" -H "Mcp-Session-Id: $SID" localhost:8000/mcp \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"key_id":"key-1"}}}'
# -> {"jsonrpc":"2.0","id":2,"error":{"code":-32001,"message":"Unauthorized Tool Call"}}
```

MCP Inspector works through the gateway too: connect to `http://localhost:8000/mcp`
with transport "Streamable HTTP" and add the header `Authorization: Bearer <token>`.

## Files

```
gateway.py           The proxy: body checks, auth, session binding, JSON-RPC inspection, forwarding
auth.py              Bearer/JWT parsing and verification, token creation
policy.py            Who may call which tool (admin_* rule, name normalization)
make_token.py        CLI to mint test tokens
mock_mcp_server.py   Downstream MCP server with normal and admin_* tools, plus /debug hooks for tests
tests/               End-to-end tests against real gateway + server processes
```

## How a request is handled

1. **Read the body**, refusing what can't be inspected safely: compressed bodies (`415`) and bodies over 1 MB (`413`).
2. **Authenticate**: `Authorization: Bearer <JWT>` → user and role. Every method (POST, GET, DELETE) needs it.
3. **Check the session**: an `Mcp-Session-Id` may only be used by the user who opened it.
4. **Inspect the JSON-RPC message** (POST):
   - `tools/list`, `initialize`, notifications and non-admin tools → forwarded unchanged.
   - `tools/call` where `params.name` is an `admin_*` tool and the role isn't `admin` → `-32001`, never forwarded.
5. **Forward** with httpx and **stream** the response back as it arrives (plain JSON or SSE).

## Authentication

Tokens are JWTs signed with HS256 (`GATEWAY_JWT_SECRET`; a dev default exists so
the demo runs out of the box, and the gateway logs a warning at startup when
it's in use). Payload: `{"sub": "alice", "role": "admin", "aud": "mcp-gateway", "exp": ...}`.

A token is accepted only if the signature matches, the algorithm is exactly
HS256 (blocks `alg: none` and algorithm-swap tricks), it hasn't expired (`exp`
required), the audience is `mcp-gateway`, and the role is `admin` or `viewer`.

Failures return HTTP **401** with `WWW-Authenticate: Bearer` (how MCP clients
learn to authenticate) and a JSON-RPC error `-32000`. Every kind of bad token
gets the same "Invalid token" message, so attackers learn nothing about why;
only "expired" is distinguished, because clients need it to refresh.

**No token passthrough.** The client's token is never forwarded downstream:
it was issued for the gateway. Instead the gateway sends `X-Gateway-User` and
`X-Gateway-Role`, after dropping any `X-Gateway-*` headers the client sent, so
they can't be spoofed.

## Error codes

| Situation | HTTP | JSON-RPC code |
|---|---|---|
| Missing / invalid / expired token | 401 | `-32000` |
| Non-admin calls an `admin_*` tool | 200 | **`-32001` Unauthorized Tool Call** |
| Session belongs to another user | 403 | `-32002` |
| Unknown session (e.g. after gateway restart) | 404 | `-32600` (tells MCP clients to re-initialize) |
| Downstream server unreachable | 502 | `-32003` (no internal details) |
| Not JSON, or ambiguous JSON (duplicate keys) | 200 | `-32700` |
| Not a JSON-RPC object, empty batch | 200 | `-32600` |
| Compressed body / body over 1 MB | 415 / 413 | `-32600` |

JSON-RPC errors carry the request's `id` when it can be read, so clients can match them.

## Hardening: fail closed

The gateway was red-teamed with 26 bypass attempts. Each is now a test, and
none reaches the downstream server:

| Attack | Defense |
|---|---|
| Batch `[get_status, admin_reset_key]` | Every item in a batch is checked; one forbidden call refuses the whole batch |
| Duplicate keys `{"name":"get_status","name":"admin_reset_key"}` | Rejected. Parsers disagree on which duplicate wins, so the gateway could check one name while the server runs another |
| `ADMIN_reset_key`, fullwidth `ａｄｍｉｎ_…`, `" admin_…"`, zero-width chars, `admin` | Names (and the method) are normalized (NFKC, format chars removed, trimmed, case-folded) before the check. The gateway is never looser than the server |
| Viewer reuses the admin's `Mcp-Session-Id` | Sessions are bound to the user who opened them → 403 |
| Spoofed `X-Gateway-Role: admin` | Client `X-Gateway-*` headers are dropped |
| UTF-16 / BOM bodies, notification form, upper-case method | Decoded and checked like any other message |
| gzip body, trailing garbage, not JSON | Refused, never forwarded |
| GET / DELETE with a JSON-RPC body | Only POST bodies are forwarded |

Each defense was **mutation-tested**: removing it (duplicate-key check, batch
inspection, name normalization, session binding, header stripping, body
dropping on GET/DELETE) makes a test fail.

## Tests

`uv run pytest -q` starts the mock server and the gateway as real processes on
free ports and talks to them over HTTP, so headers, sessions and SSE behave as
in production. 50 tests cover:

- the requirements: `tools/list` forwarded transparently, `admin_*` → `-32001` for viewers and allowed for admins, with `/debug/calls` proving blocked calls never ran downstream
- authentication: missing, malformed, expired, forged, `alg: none`, wrong audience, unknown role, no `exp`
- the 26 bypass attempts above
- sessions: binding, token refresh, unknown session, DELETE
- SSE: streamed responses pass through and the block still applies
- downstream down → 502 with no internal details; oversized and compressed bodies

## Design decisions

- **`tools/list` is passed through unfiltered**, as the task specifies. A
  natural extension is to remove `admin_*` tools from the list for viewers, so
  the model never even sees them.
- **Batches with a forbidden call are refused entirely** rather than partially
  executed: simpler to reason about, and MCP has dropped batching since
  2025-06-18 anyway.
- **Session ownership is kept in memory** (bounded to 10,000 sessions). With
  several gateway replicas it would move to a shared store such as Redis.

## Path to production

- Tokens from a real identity provider (OIDC), verified with its public keys (RS256/ES256 + JWKS) instead of a shared secret.
- Per-tool policy from config (allow-lists per role) instead of the `admin_` prefix rule.
- TLS, rate limiting per user, and the audit log (`BLOCKED ...` lines) shipped to a SIEM.
