"""End-to-end: client -> LLM gateway process -> mock LLM provider process."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
REDACTED = "[REDACTED]"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn(app: str, port: int, env: dict, ready: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", app, "--port", str(port), "--log-level", "warning"],
        cwd=ROOT, env={**os.environ, **env}, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            httpx.request("GET" if "healthz" in ready else "POST", ready, timeout=0.5)
            return proc
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError(f"{app} did not start")


def stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    proc.wait(timeout=10)
    proc.stderr.close()


@pytest.fixture(scope="module")
def gateway():
    mock_port, gw_port = free_port(), free_port()
    mock = spawn("mock_llm:app", mock_port, {"MOCK_CHUNK_DELAY_MS": "15"}, f"http://127.0.0.1:{mock_port}/docs")
    gw = spawn("llm_gateway:app", gw_port, {"UPSTREAM_BASE_URL": f"http://127.0.0.1:{mock_port}/v1"}, f"http://127.0.0.1:{gw_port}/healthz")
    yield f"http://127.0.0.1:{gw_port}/v1/chat/completions"
    for p in (mock, gw):
        stop(p)


def request(text: str | None = None, stream: bool = True, **extra) -> dict:
    content = f"echo:{text}" if text is not None else "hello"
    return {"model": "mock", "stream": stream, "messages": [{"role": "user", "content": content}], **extra}


def read_stream(url: str, body: dict) -> tuple[str, list[dict], list[float], float]:
    """Return (assembled content, all events, arrival time of each content event, total time)."""
    text, events, arrivals = "", [], []
    start = time.perf_counter()
    with httpx.stream("POST", url, json=body, timeout=30) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                events.append({"done": True})
                continue
            event = json.loads(data)
            events.append(event)
            for choice in event.get("choices", []):
                content = (choice.get("delta") or {}).get("content")
                if content:
                    text += content
                    arrivals.append(time.perf_counter() - start)
    return text, events, arrivals, time.perf_counter() - start


PII_TEXT = (
    "Email jane.doe@example.com, backup first_last@corp.example.org. Card 4111 1111 1111 1111 and "
    "5500-0000-0000-0004. SSN 123-45-6789. Order 1234567890123 on 2026-01-15, phone 555-123-4567."
)
EXPECTED = (
    f"Email {REDACTED}, backup {REDACTED}. Card {REDACTED} and {REDACTED}. SSN {REDACTED}. "
    "Order 1234567890123 on 2026-01-15, phone 555-123-4567."
)


@pytest.mark.parametrize("seed", range(5))
def test_stream_is_redacted_whatever_the_chunking(gateway, seed):
    text, events, _, _ = read_stream(gateway, request(PII_TEXT, seed=seed))
    assert text == EXPECTED
    assert events[-1] == {"done": True}
    assert any(c.get("finish_reason") == "stop" for e in events[:-1] for c in e.get("choices", []))


def test_no_pii_fragment_ever_reaches_the_client(gateway):
    with httpx.stream("POST", gateway, json=request(PII_TEXT, seed=3), timeout=30) as r:
        raw = r.read().decode()
    for fragment in ["jane", "doe@", "example.com", "first_last", "4111", "5500", "45-6789"]:
        assert fragment not in raw


def test_stream_is_relayed_incrementally_not_buffered(gateway):
    long_text = "The quick brown fox jumps over the lazy dog. " * 12  # ~540 chars, ~150 chunks x 15 ms
    text, _, arrivals, total = read_stream(gateway, request(long_text, seed=1))
    assert text == long_text
    assert arrivals[0] < 0.5  # first token arrives quickly...
    assert arrivals[0] < total / 4  # ...long before the stream ends
    assert len(arrivals) > 50  # and the answer comes in many pieces, as it's generated


def test_unicode_survives(gateway):
    text = "Héllo 👋 مرحبا — mail é@x.io? no: ok@xn--bcher-kva.de. 日本語テキスト"
    out, _, _, _ = read_stream(gateway, request(text, seed=2))
    assert out == "Héllo 👋 مرحبا — mail é@x.io? no: [REDACTED]. 日本語テキスト"


def test_non_streaming_response_is_redacted(gateway):
    r = httpx.post(gateway, json=request(PII_TEXT, stream=False), timeout=30)
    assert r.json()["choices"][0]["message"]["content"] == EXPECTED


def test_interrupted_upstream_ends_with_a_clean_error_event(gateway):
    _, events, _, _ = read_stream(gateway, request(PII_TEXT, seed=1, fail_after_chunks=10))
    assert events[-1] == {"error": {"type": "upstream_error", "message": "The model stream was interrupted"}}


@pytest.mark.parametrize(("status", "expected"), [(429, 429), (400, 400), (500, 502), (503, 502)])
def test_upstream_errors_are_sanitized(gateway, status, expected):
    r = httpx.post(gateway, json=request("x", status=status), timeout=30)
    assert r.status_code == expected
    assert "sk-live" not in r.text and "prod-7" not in r.text
    assert r.json()["error"]["type"] == "upstream_error"


def test_provider_down_gives_502():
    port = free_port()
    gw = spawn("llm_gateway:app", port, {"UPSTREAM_BASE_URL": "http://127.0.0.1:9/v1"}, f"http://127.0.0.1:{port}/healthz")
    try:
        r = httpx.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=request("x"), timeout=30)
        assert r.status_code == 502
        assert r.json() == {"error": {"type": "upstream_unavailable", "message": "The model provider is unavailable"}}
    finally:
        stop(gw)


@pytest.mark.parametrize("raw", [b"not json", b"[1,2]"])
def test_bad_request_bodies(gateway, raw):
    r = httpx.post(gateway, content=raw, headers={"Content-Type": "application/json"}, timeout=30)
    assert r.status_code == 400
