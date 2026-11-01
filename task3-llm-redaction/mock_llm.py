"""Mock LLM provider with an OpenAI-compatible /v1/chat/completions endpoint.

It streams its answer in random 1-6 character chunks, so PII is split across
chunks all the time, which is exactly what the gateway has to cope with.

  - If the last user message starts with "echo:", the rest of it is the answer
    (lets tests choose the exact text). Otherwise a canned answer full of PII.
  - "seed" in the request makes the chunking reproducible.
  - MOCK_CHUNK_DELAY_MS sets the delay between chunks (default 20 ms).
  - "fail_after_chunks": N in the request drops the connection after N chunks.
  - "status": 429 (or any code) in the request makes it fail with that HTTP status.

Run: uv run uvicorn mock_llm:app --port 9100
"""

import asyncio
import json
import os
import random
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

CANNED = (
    "Sure! I found the customer. Their email is jane.doe@example.com and the "
    "backup contact is first_last@corp.example.org. The card on file is "
    "4111 1111 1111 1111 (expires 12/27) and their SSN is 123-45-6789. "
    "Order number 1234567890123 shipped on 2026-01-15."
)

app = FastAPI(title="Mock LLM provider")


def answer_for(request: dict) -> str:
    messages = request.get("messages") or []
    last = messages[-1].get("content", "") if messages else ""
    if isinstance(last, str) and last.startswith("echo:"):
        return last[len("echo:"):]
    return CANNED


def chunks(text: str, seed) -> list[str]:
    rng = random.Random(seed)
    out, i = [], 0
    while i < len(text):
        n = rng.randint(1, 6)
        out.append(text[i:i + n])
        i += n
    return out


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    if body.get("status"):
        return JSONResponse({"error": {"message": "internal trace: db=prod-7 key=sk-live-123"}}, status_code=int(body["status"]))
    text = answer_for(body)
    model = body.get("model", "mock-model")
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if not body.get("stream"):
        return JSONResponse({
            "id": completion_id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        })

    delay = int(os.environ.get("MOCK_CHUNK_DELAY_MS", "20")) / 1000

    def event(delta: dict, finish_reason=None) -> str:
        chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
        return f"data: {json.dumps(chunk)}\n\n"

    fail_after = body.get("fail_after_chunks")  # test hook: cut the stream off mid-way

    async def stream():
        yield event({"role": "assistant", "content": ""})
        for i, piece in enumerate(chunks(text, body.get("seed"))):
            if fail_after is not None and i >= fail_after:
                raise RuntimeError("simulated provider crash mid-stream")
            await asyncio.sleep(delay)
            yield event({"content": piece})
        yield event({}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
