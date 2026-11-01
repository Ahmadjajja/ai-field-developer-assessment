"""Unit tests for failover and the timeout race, with in-process fake providers."""

import asyncio
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router import AllProvidersFailed, ModelRouter, Provider, ProviderRejected  # noqa: E402

OK = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}], "usage": {"total_tokens": 5}}
PROVIDERS = [Provider("primary", "http://primary/v1", "p", timeout=0.3), Provider("secondary", "http://secondary/v1", "s", timeout=1.0)]


class FakeProviders:
    """httpx transport whose responses per host are scripted by the test."""

    def __init__(self, **behavior) -> None:
        self.behavior = behavior  # host -> callable(request) -> awaitable Response
        self.started: list[str] = []
        self.finished: list[str] = []
        self.cancelled: list[str] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        self.started.append(host)
        try:
            response = await self.behavior[host](request)
        except asyncio.CancelledError:
            self.cancelled.append(host)
            raise
        self.finished.append(host)
        return response


def ok(delay: float = 0.0, status: int = 200, body=None):
    async def respond(request):
        await asyncio.sleep(delay)
        return httpx.Response(status, json=OK if body is None else body)
    return respond


def unreachable():
    async def respond(request):
        raise httpx.ConnectError("connection refused", request=request)
    return respond


def run(fake: FakeProviders, body: dict | None = None):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handle)) as http:
            return await ModelRouter(PROVIDERS, http).complete(body or {"messages": [{"role": "user", "content": "x"}]})
    return asyncio.run(go())


def test_primary_answers():
    fake = FakeProviders(primary=ok(), secondary=ok())
    assert run(fake).provider == "primary"
    assert fake.started == ["primary"]


@pytest.mark.parametrize(("label", "primary"), [
    ("429", ok(status=429)),
    ("500", ok(status=500)),
    ("503", ok(status=503)),
    ("unreachable", unreachable()),
    ("not JSON", lambda r: asyncio.sleep(0, httpx.Response(200, content=b"<html>oops"))),
    ("JSON without choices", ok(body={"weird": True})),
])
def test_fails_over_to_secondary(label, primary):
    fake = FakeProviders(primary=primary, secondary=ok())
    routed = run(fake)
    assert routed.provider == "secondary", label
    assert [a["provider"] for a in routed.attempts] == ["primary", "secondary"]


def test_slow_primary_is_cut_off_at_the_deadline_and_cancelled():
    fake = FakeProviders(primary=ok(delay=5), secondary=ok())
    start = time.perf_counter()
    routed = run(fake)
    elapsed = time.perf_counter() - start

    assert routed.provider == "secondary"
    assert routed.attempts[0]["outcome"] == "timeout"
    assert 0.3 <= elapsed < 0.6  # waited for the deadline, not for the slow answer
    assert fake.cancelled == ["primary"]  # the primary call was really cancelled...
    assert "primary" not in fake.finished  # ...so its late answer can never be used


def test_primary_just_inside_the_deadline_is_used():
    fake = FakeProviders(primary=ok(delay=0.2), secondary=ok())
    assert run(fake).provider == "primary"
    assert fake.started == ["primary"]


def test_exactly_one_result_even_when_primary_finishes_right_at_the_deadline():
    # Race the deadline many times: whoever wins, exactly one provider's answer is returned
    # and a timed-out primary is always cancelled, never "finished" after the switch.
    for delay in [0.29, 0.295, 0.3, 0.305, 0.31] * 4:
        fake = FakeProviders(primary=ok(delay=delay), secondary=ok())
        routed = run(fake)
        if routed.provider == "primary":
            assert fake.started == ["primary"] and fake.cancelled == []
        else:
            assert fake.cancelled == ["primary"] and "primary" not in fake.finished


def test_client_errors_are_not_retried_elsewhere():
    fake = FakeProviders(primary=ok(status=400), secondary=ok())
    with pytest.raises(ProviderRejected):
        run(fake)
    assert fake.started == ["primary"]


def test_all_providers_failing_reports_each_attempt():
    fake = FakeProviders(primary=ok(status=429), secondary=ok(delay=5))
    with pytest.raises(AllProvidersFailed) as exc:
        run(fake)
    assert [(a["provider"], a["outcome"]) for a in exc.value.attempts] == [("primary", "rate_limited"), ("secondary", "timeout")]


def test_each_provider_gets_its_own_model_name():
    seen = []

    async def record(request):
        import json
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(429 if request.url.host == "primary" else 200, json=OK)

    run(FakeProviders(primary=record, secondary=record), {"model": "client-choice", "messages": []})
    assert seen == ["p", "s"]
