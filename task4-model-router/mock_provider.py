"""Mock LLM provider (OpenAI-compatible) whose behavior can be switched per request.

Run one as the primary and one as the secondary:
  PROVIDER_NAME=primary   uv run uvicorn mock_provider:app --port 9201
  PROVIDER_NAME=secondary uv run uvicorn mock_provider:app --port 9202

Behavior, chosen per request with the "mock" field in the request body, e.g.
{"mock": {"primary": "429", "secondary": "ok"}}, or as the default with
POST /mock/behavior {"mode": "slow:4000"}:

  ok          normal completion
  429         rate limited by the provider
  500         internal error (the body contains fake internal details)
  slow:<ms>   answer after <ms> milliseconds
  hang        never answer
"""

import asyncio
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

NAME = os.environ.get("PROVIDER_NAME", "provider")
app = FastAPI(title=f"Mock provider ({NAME})")
app.state.default_mode = os.environ.get("MOCK_MODE", "ok")
app.state.calls = 0


@app.post("/mock/behavior")
async def set_behavior(request: Request) -> dict:
    app.state.default_mode = (await request.json())["mode"]
    return {"provider": NAME, "mode": app.state.default_mode}


@app.get("/mock/calls")
async def calls() -> dict:
    return {"provider": NAME, "calls": app.state.calls}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    app.state.calls += 1
    body = await request.json()
    mode = (body.get("mock") or {}).get(NAME, app.state.default_mode)

    if mode == "429":
        return JSONResponse({"error": {"message": "Rate limit exceeded for org-internal-42"}}, status_code=429,
                            headers={"Retry-After": "20"})
    if mode == "500":
        return JSONResponse({"error": {"message": "Traceback (most recent call last): File /srv/llm/engine.py, "
                                                  "line 88, in generate  KeyError: api_key=sk-live-9f8e7d"}}, status_code=500)
    if mode == "hang":
        await asyncio.sleep(3600)
    if mode.startswith("slow:"):
        await asyncio.sleep(int(mode.split(":")[1]) / 1000)

    prompt = " ".join(str(m.get("content", "")) for m in body.get("messages", []))
    answer = f"Answer from {NAME}."
    prompt_tokens = max(1, len(prompt) // 4)
    completion_tokens = min(body.get("max_tokens") or 64, 6)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", f"{NAME}-model"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }
