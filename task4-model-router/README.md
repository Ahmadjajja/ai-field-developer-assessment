# LLM Gateway Model Router: Rate Limiting & Fallback (Task 4)

A resilient model-routing module for an LLM gateway. It accepts
OpenAI-compatible completion requests and:

1. **Rate limits per tenant API key** with a token-aware sliding window:
   at most **50,000 tokens per 60 seconds**, stored in **SQLite on disk**.
2. **Fails over** from the primary model to the secondary when the primary
   returns **429** or doesn't answer within **3000 ms** (also when it's
   unreachable or fails with a 5xx).
3. Returns **standardized errors** that never contain upstream bodies, stack
   traces or internal details.

```
client ─Bearer tenant key─► router :8200 ─┬─► primary :9201   (3000 ms deadline)
                              │            └─► secondary :9202 (on 429 / timeout / down / 5xx)
                              └─ SQLite: api_keys, token_usage (sliding window)
```

## Quick start

```bash
uv sync
uv run pytest -q                                              # 47 tests

PROVIDER_NAME=primary   uv run uvicorn mock_provider:app --port 9201   # terminal 1
PROVIDER_NAME=secondary uv run uvicorn mock_provider:app --port 9202   # terminal 2
uv run uvicorn app:app --port 8200                                     # terminal 3

KEY=$(uv run python tenants.py add acme)
curl -si localhost:8200/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -d '{"model":"any","max_tokens":100,"messages":[{"role":"user","content":"hello"}]}'
```

Make the primary misbehave, per request or as a default:

```bash
# per request: the mock reads the "mock" field
-d '{..., "mock": {"primary": "slow:4000"}}'      # → served by secondary after 3 s
-d '{..., "mock": {"primary": "429"}}'            # → served by secondary at once
# as a default
curl -X POST localhost:9201/mock/behavior -d '{"mode": "hang"}'
```

Response headers show what happened: `X-Model-Provider: primary|secondary`,
`X-RateLimit-Limit/Remaining/Reset`, `X-Request-ID`.

## Files

```
rate_limiter.py   SlidingWindowLimiter (SQLite) and token estimation
router.py         ModelRouter: deadline, failover rules, timeout-race handling
app.py            FastAPI endpoint: auth, validation, reserve → route → settle, standardized errors
tenants.py        Tenant API keys (stored as SHA-256 hashes) + CLI to create them
mock_provider.py  OpenAI-compatible mock provider: ok / 429 / 500 / slow:<ms> / hang
tests/            Limiter and router unit tests, end-to-end tests against real processes
```

## Rate limiting

**Token-aware.** Requests are limited by tokens, not by request count. Before
the call the router reserves an upper-bound estimate (prompt at about 4
characters per token, plus per-message overhead, plus `max_tokens`). After
the call it **settles** the reservation to the provider's reported
`usage.total_tokens`. Unused tokens become available again immediately.

**Sliding window.** Each reservation is a row `(tenant, ts, tokens)`. A
request fits if the tenant's rows from the last 60 seconds plus the new
estimate stay within 50,000. There's no fixed minute boundary to burst
across.

**Atomic under concurrency.** Eviction, check and insert run in one
`BEGIN IMMEDIATE` transaction, which takes SQLite's write lock up front.
Concurrent requests, including requests from **several gateway processes
sharing the database file**, line up one by one and can never overshoot
together. WAL mode keeps readers from blocking. SQLite calls run in a worker
thread (`anyio.to_thread`) so the event loop isn't blocked.

**Eviction.** Rows older than the window are deleted inside the same
transaction, so the table only ever holds about a minute of data.

**Accounting rules:**

| Outcome | Tenant is charged |
|---|---|
| Served by primary or secondary | that provider's real usage, **once** |
| Primary timed out, secondary served | only the secondary's usage (the primary call was cancelled) |
| No provider could serve it (503) | nothing (reservation released) |
| Client disconnected mid-request | nothing (released in a shielded cleanup step) |

**Responses.** Over the limit: `429 rate_limit_exceeded` with `Retry-After`,
computed from when enough old reservations expire, not a fixed guess (at least
1 second, so clients never get "retry in 0s" and spin). A
request whose estimate exceeds the whole budget gets `400 request_too_large`,
since it could never succeed.

## Failover and the timeout race

The primary gets a hard 3000 ms deadline for the **entire** call (connect, wait,
read), enforced with `asyncio.timeout()`:

| Primary result | Action |
|---|---|
| 2xx with a valid completion | use it |
| **429** | fail over |
| **no answer within 3000 ms** | cancel it, fail over |
| unreachable, 5xx, invalid JSON | fail over |
| other 4xx (bad request) | no failover: the request itself is wrong → 400 |

**Race safety.** When the deadline hits, the primary request is **cancelled**:
httpx closes the connection, so a late primary answer can never be read after
the switch. Exactly one provider's answer is returned, and exactly one usage
is charged. The unit tests race the deadline (primary finishing at 290 to 310
ms against a 300 ms deadline) and check that either the primary finished
in time and nothing was cancelled, or it was cancelled and the secondary
served. Never both.

The secondary has its own deadline (10 s by default). Each provider gets its
own model name and API key (`PRIMARY_*` / `SECONDARY_*` env vars).

## Standardized errors

Every error, from validation, auth, rate limiting, providers, unknown
routes or unexpected crashes, has the same shape:

```json
{"error": {"type": "rate_limit_exceeded", "message": "Token rate limit exceeded; retry in 12s", "request_id": "4cc5…"}}
```

| Status | `type` |
|---|---|
| 400 | `invalid_request`, `request_too_large` |
| 401 | `authentication_error` |
| 404 / 405 | `not_found` / `method_not_allowed` |
| 429 | `rate_limit_exceeded` (+ `Retry-After`, `X-RateLimit-*`) |
| 503 | `upstream_unavailable` (all providers failed) |
| 500 | `internal_error` (traceback logged server-side only) |

The `request_id` is also in the `X-Request-ID` header and in the server logs,
so support can find the internal details without exposing them. The tests
make the mock providers return bodies full of fake internals (`Traceback`,
`sk-live-…`, file paths) and assert that none of it reaches the client.

## Tests

`uv run pytest -q` runs 47 tests:

- **Limiter** (fake clock): limit edges, the window sliding precisely at 60 s,
  `Retry-After` math, eviction, tenant isolation, settle and release,
  persistence across restarts, **120 threads at once → exactly 50 accepted,
  70 limited, 0 errors**, **3 processes sharing the DB → exactly 50,000 tokens granted**.
- **Router** (in-process fake providers): failover on 429/500/503/unreachable/invalid
  JSON, deadline cut-off with real cancellation, the deadline race, no
  failover on 4xx, every attempt reported.
- **End to end** (real router and provider processes, spec defaults): primary
  slower than 3000 ms → secondary at about 3 s; 2500 ms → primary; hang → secondary;
  sanitized 503; standard error shape for every error kind; charged once after
  failover; failed requests not charged; 429 with `Retry-After` and recovery
  once the window slides; 30 concurrent requests → exactly the number that fits.

Mutation-tested: removing eviction, the window count, the deadline, failover
on 429, releasing failed requests or settling real usage each makes a test
fail. A plain deferred `BEGIN` without the up-front write lock produces
"database is locked" failures under concurrency, which the thread test catches.

## Design decisions

- **Upper-bound reservation, then settle**: guarantees the limit even with
  many requests in flight, without needing exact token counts up front.
- **Streaming is rejected** (`400`) in this module. Failover after the first
  streamed bytes would need different rules (hedging or replay); that belongs
  in the streaming path (see Task 3).
- **SQLite as the shared store** fits a single host or a few processes. Across
  many hosts the same algorithm maps directly onto Redis (a sorted set per
  tenant plus a Lua script for the atomic check-and-reserve).
