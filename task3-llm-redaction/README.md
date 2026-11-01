# LLM Gateway: Streaming PII Redaction (Task 3)

An OpenAI-compatible LLM gateway endpoint (`POST /v1/chat/completions`) that
forwards requests to the model provider and streams the answer back, replacing
**emails, SSNs and credit card numbers** with `[REDACTED]` in real time.

```
client ◄── SSE ── gateway :8100 ◄── SSE ── provider :9100
                    │
                    parse events → StreamRedactor (per choice) → re-emit
```

The response is never accumulated: each stream keeps only the short tail
that could still turn into PII, so time-to-first-token stays close to the
provider's and memory stays flat however long the answer is.

## Quick start

```bash
uv sync
uv run pytest -q                                   # 77 tests

uv run uvicorn mock_llm:app --port 9100            # terminal 1: mock provider
uv run uvicorn llm_gateway:app --port 8100         # terminal 2: gateway

curl -N localhost:8100/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"mock","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

The mock streams a canned answer full of PII in random 1-6 character chunks.
Send `"content": "echo:<your text>"` to make it stream your own text.

To use a real provider, point the gateway at any OpenAI-compatible API:
`UPSTREAM_BASE_URL=https://api.openai.com/v1 UPSTREAM_API_KEY=... uv run uvicorn llm_gateway:app`.

## Files

```
redactor.py      PII patterns and StreamRedactor (the core algorithm)
llm_gateway.py   FastAPI endpoint: SSE parsing, per-choice redaction, errors
mock_llm.py      OpenAI-compatible mock provider that streams in tiny random chunks
tests/           Unit tests (redactor, SSE parser) and end-to-end tests (real processes)
```

## The core idea: hold back only what could still become PII

PII gets split across chunks all the time (`"john.d"` + `"oe@gmail"` + `".com"`).
Checking each chunk alone misses it; waiting for the full answer ruins latency
and memory. The redactor uses one observation: every pattern is a run of
"candidate" characters, either email characters with no spaces, or digits,
spaces and dashes for cards and SSNs. Only the **trailing run** can still
become PII when more text arrives, so:

```
feed("Contact me at john.d")  → emits "Contact me at "        holds "john.d"
feed("oe@gmail.com today")    → emits "[REDACTED] "            holds "today"
flush()                       → emits "today"
```

Two rules keep it exact:

1. **Never cut inside a match, or right at its end**, because the next characters could still extend it. The cut moves back to the start of the match.
2. **Everything before the cut is final.** The cut always lands where a candidate run or a match starts, so later text can't change any match before it.

The result: **the streamed output is identical to redacting the complete text
in one go, however the stream is chunked.** That property is tested directly:
every possible two-chunk split, one character at a time, and 10,000 random
adversarial streams in the suite. During development, over 700,000 fuzzed
streams were checked, including hostile ones with very long runs, with 0
mismatches and 0 leaks.

**How much is held.** Only what could still become PII, using the real size
limits of each kind:

| Trailing run | Held at most | Why |
|---|---|---|
| digits / spaces / dashes | 40 characters | the longest card is 19 digits + 18 separators = 37 |
| email characters, no `@` yet | 64 characters | only the last 64 can become a local part (RFC 5321) |
| email characters with an `@` | 256 characters | longest valid address is 254 |

As a last resort for hostile input (e.g. `1 2 3 4 …`, where overlapping
card-like matches chain on and on), held text over 512 characters is
released anyway. Any match crossing that cut is redacted, including its
continuation in the next output: it over-redacts rather than leaks.

### Performance

- **Latency:** normally only the current partial word is held (a card number
  at most), so the client sees each token about one token later than the
  provider sends it. In the end-to-end test, the first token arrives within
  milliseconds and the answer arrives in over 50 separate pieces.
- **Memory:** constant per stream. The held run is capped (`MAX_HOLD`), so even
  a hostile stream with no separators can't grow the buffer: 200,000
  characters with no break keep the buffer under 512 characters.
- **CPU:** regexes run only over the held tail plus the new chunk, never the
  whole response, and card checks (a Luhn sum per candidate) are cached once a
  candidate can no longer change. Measured in 4-character chunks:

  | Input | Time per MB |
  |---|---|
  | normal prose with PII | 0.7 s |
  | hostile: letters / digits / email characters with no break | 1.6–2 s |
  | hostile: `1 2 3 4 …` / random digits with spaces (card-like matches everywhere) | ~26 s (was 228 s before caching) |

  Even the worst case is about 40 KB/s per stream, dozens of times faster than
  a model generates text, so a hostile answer can't stall the gateway.

## What's detected

| Type | Rule | Not redacted (to avoid false positives) |
|---|---|---|
| Email | `local@domain.tld`, including `+`, `_`, `.`, `%`, `-` and subdomains; local part ≤ 64, labels ≤ 63, at most 254 characters | `@handle`, `x@localhost`, `user at example dot com` |
| Credit card | 13-19 digits, optionally grouped with spaces or dashes, **must pass the Luhn checksum**; every length is tried, so `4111 1111 1111 1111 9 items` is still caught | order numbers, a card number with a wrong digit |
| SSN | `AAA-GG-SSSS` or `AAA GG SSSS` | numbers never issued as SSNs (area 000/666/9xx, group 00, serial 0000), dates, phone numbers, mixed separators |

Cards are checked at every start position and every length, so neither an
overlapping digit run nor a longer window that fails Luhn can hide a valid card.

## SSE handling

- Bytes go through an **incremental UTF-8 decoder**: a network chunk can split
  a multi-byte character (emoji, Arabic), and decoding chunks separately would
  corrupt it.
- Only complete lines and events are processed. CRLF, comments and multi-line
  `data:` fields are supported.
- Each choice gets its own redactor, so requests with `n > 1` work.
- Held text is released at `finish_reason`, before `[DONE]`, or when the
  upstream ends without `[DONE]`.
- Events whose content becomes empty after holding are skipped, not sent empty.
- An event that isn't valid JSON is dropped, never relayed unchecked (fail closed).

## Errors

All errors use one shape and never include upstream bodies or internal details:
`{"error": {"type": "...", "message": "..."}}`

| Situation | Response |
|---|---|
| Provider unreachable | 502 `upstream_unavailable` |
| Provider 4xx (e.g. 429) | same status, `upstream_error` |
| Provider 5xx | 502 `upstream_error` |
| Stream cut off mid-way | held text is flushed (redacted), then an `error` event |
| Body not JSON / over 1 MB | 400 / 413 `invalid_request` |

Non-streaming requests (`"stream": false`) are redacted too, in one pass.

## Tests

`uv run pytest -q` runs 77 tests:

- **Redactor:** positive and negative examples for each PII type, every
  two-chunk split, one character at a time, random adversarial streams, held
  size for normal prose and for a hostile stream, throughput, speed on six
  hostile input shapes, no PII ever surviving in the output of hostile streams,
  email length limits, a card followed by more digits, forced cuts through
  endless chains (random digits, back-to-back real cards) never leaking a digit.
- **SSE parser:** events split at every byte, including inside an emoji; CRLF; multi-line data.
- **End to end** (real gateway and provider processes): redaction under 5
  different chunkings, no PII fragment anywhere in the raw stream, incremental
  delivery and time-to-first-token, Unicode, non-streaming, interrupted
  upstream, sanitized upstream errors, provider down, bad request bodies.

Key defenses were mutation-tested: removing the hold-back, allowing cuts
inside a match, dropping the Luhn check, not holding digit runs, or decoding
UTF-8 per chunk each makes a test fail.

## Limitations and next steps

- Regex-based detection: names, addresses and free-form identifiers need an
  NER model. That could run on the same held-tail approach, at the cost of latency.
- Only message `content` is redacted. Tool-call arguments stream separately
  and could get their own redactors the same way.
- Requests aren't redacted (PII the user sends to the model). The same
  `redact()` could be applied to `messages` before forwarding.
