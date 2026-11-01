"""LLM Gateway with a streaming PII guardrail (Task 3).

POST /v1/chat/completions (OpenAI-compatible) is forwarded to the LLM
provider. With "stream": true, the provider's SSE stream is relayed to the
client as it arrives, with emails, SSNs and credit card numbers replaced by
[REDACTED] on the fly:

  provider SSE  ->  parse events  ->  StreamRedactor per choice  ->  re-emit SSE

The full response is never accumulated: each choice keeps only the short tail
that could still turn into PII (see redactor.py), so time-to-first-token stays
close to the provider's and memory stays flat however long the answer is.

Run: uv run uvicorn llm_gateway:app --port 8100
"""

import codecs
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from redactor import StreamRedactor, redact

UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "http://127.0.0.1:9100/v1")
UPSTREAM_API_KEY = os.environ.get("UPSTREAM_API_KEY")  # the gateway's own provider key, if any
MAX_REQUEST_BYTES = 1_000_000

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="llm-gateway: %(message)s")
log = logging.getLogger("llm-gateway")
logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # No read timeout: a stream may pause between tokens. Connect timeout stays short.
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None))
    yield
    await app.state.http.aclose()


app = FastAPI(title="LLM Gateway (PII redaction)", lifespan=lifespan)


def error(status: int, kind: str, message: str) -> JSONResponse:
    """One error shape for everything; never includes upstream bodies or tracebacks."""
    return JSONResponse({"error": {"type": kind, "message": message}}, status_code=status)


# --- SSE parsing ----------------------------------------------------------------------


async def sse_events(byte_chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """Yield the data of each Server-Sent Event, as it completes.

    Network chunks don't respect event, line or even character boundaries, so
    bytes go through an incremental UTF-8 decoder (a multi-byte character can
    be split across chunks) and only complete lines are processed.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending, data_lines = "", []
    async for raw in byte_chunks:
        pending += decoder.decode(raw)
        *lines, pending = pending.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        for line in lines:
            if line == "":  # blank line = end of event
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines = []
            elif line.startswith("data:"):
                data_lines.append(line[5:].removeprefix(" "))
            # comments (":") and other fields (event:, id:, retry:) are not needed here
    pending += decoder.decode(b"", final=True)
    if pending.startswith("data:"):
        data_lines.append(pending[5:].removeprefix(" "))
    if data_lines:
        yield "\n".join(data_lines)


def sse(data: str) -> bytes:
    return f"data: {data}\n\n".encode()


# --- Streaming redaction ------------------------------------------------------------------


class ChoiceRedaction:
    """Redacts each choice's content stream separately (requests with n > 1 have several)."""

    def __init__(self) -> None:
        self.redactors: dict[int, StreamRedactor] = {}
        self.template: dict = {}  # id/model/created of the last chunk, for flush events

    def apply(self, chunk: dict) -> bool:
        """Redact the chunk in place. Returns False if nothing is left worth sending."""
        self.template = {k: chunk[k] for k in ("id", "object", "created", "model") if k in chunk}
        worth_sending = bool(chunk.get("usage")) or not chunk.get("choices")
        for choice in chunk.get("choices") or []:
            redactor = self.redactors.setdefault(choice.get("index", 0), StreamRedactor())
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                delta["content"] = redactor.feed(delta["content"])
            if choice.get("finish_reason"):
                # The choice is done: release what was held, before the finish marker.
                delta["content"] = (delta.get("content") or "") + redactor.flush()
                choice["delta"] = delta
            if delta.get("content") or set(delta) - {"content"} or choice.get("finish_reason"):
                worth_sending = True
        return worth_sending

    def flush_events(self) -> list[dict]:
        """Chunks carrying any text still held (for streams that end without finish_reason)."""
        events = []
        for index, redactor in self.redactors.items():
            text = redactor.flush()
            if text:
                events.append({**self.template, "choices": [{"index": index, "delta": {"content": text}, "finish_reason": None}]})
        return events


async def redacted_stream(upstream: httpx.Response) -> AsyncIterator[bytes]:
    redaction = ChoiceRedaction()
    started, first_byte_at, sent_done = time.perf_counter(), None, False
    try:
        async for data in sse_events(upstream.aiter_bytes()):
            if data.strip() == "[DONE]":
                for event in redaction.flush_events():
                    yield sse(json.dumps(event))
                yield sse("[DONE]")
                sent_done = True
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                log.warning("dropped an unparseable stream event")  # fail closed: never relay unchecked text
                continue
            if isinstance(chunk, dict) and redaction.apply(chunk):
                if first_byte_at is None and any((c.get("delta") or {}).get("content") for c in chunk.get("choices") or []):
                    first_byte_at = time.perf_counter()  # time to first token, as the client sees it
                yield sse(json.dumps(chunk))
        if not sent_done:
            for event in redaction.flush_events():
                yield sse(json.dumps(event))
    except httpx.HTTPError as exc:
        log.error("upstream stream failed: %s", type(exc).__name__)
        for event in redaction.flush_events():
            yield sse(json.dumps(event))
        yield sse(json.dumps({"error": {"type": "upstream_error", "message": "The model stream was interrupted"}}))
    finally:
        await upstream.aclose()
        ttft = f"{(first_byte_at - started) * 1000:.0f}ms" if first_byte_at else "n/a"
        log.info("stream finished: first token after %s, total %.0fms", ttft, (time.perf_counter() - started) * 1000)


# --- Endpoint ------------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    raw = await request.body()
    if len(raw) > MAX_REQUEST_BYTES:
        return error(413, "invalid_request", "Request body too large")
    try:
        body = json.loads(raw)
    except ValueError:
        return error(400, "invalid_request", "Request body must be JSON")
    if not isinstance(body, dict):
        return error(400, "invalid_request", "Request body must be a JSON object")

    headers = {"Content-Type": "application/json"}
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    elif "authorization" in request.headers:
        headers["Authorization"] = request.headers["authorization"]

    http: httpx.AsyncClient = request.app.state.http
    upstream_request = http.build_request("POST", f"{UPSTREAM_BASE_URL}/chat/completions", headers=headers, content=raw)
    try:
        upstream = await http.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        log.error("upstream unavailable: %s", type(exc).__name__)
        return error(502, "upstream_unavailable", "The model provider is unavailable")

    if upstream.status_code >= 400:
        await upstream.aclose()
        log.warning("upstream returned HTTP %d", upstream.status_code)
        return error(upstream.status_code if upstream.status_code < 500 else 502, "upstream_error",
                     "The model provider rejected the request" if upstream.status_code < 500 else "The model provider failed")

    if body.get("stream"):
        return StreamingResponse(redacted_stream(upstream), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # Non-streaming: the whole answer is here anyway, so redact it in one go.
    try:
        completion = json.loads(await upstream.aread())
    finally:
        await upstream.aclose()
    for choice in completion.get("choices") or []:
        message = choice.get("message") or {}
        if isinstance(message.get("content"), str):
            message["content"] = redact(message["content"])
    return JSONResponse(completion)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}
