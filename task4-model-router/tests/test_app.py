"""End-to-end: client -> router process -> primary/secondary mock provider processes."""

import concurrent.futures
import os
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tenants import TenantStore  # noqa: E402

LEAKS = ["Traceback", "sk-live", "/srv/", "org-internal", "engine.py", "KeyError"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn(app: str, port: int, env: dict) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", app, "--port", str(port), "--log-level", "warning"],
        cwd=ROOT, env={**os.environ, **env}, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{port}/docs", timeout=0.5)
            return proc
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError(f"{app} did not start")


class Stack:
    def __init__(self, tmp: Path, **router_env) -> None:
        self.db = str(tmp / "router.db")
        self.key = TenantStore(self.db).add("acme")
        self.other_key = TenantStore(self.db).add("globex")
        p1, p2, rp = free_port(), free_port(), free_port()
        self.providers = {"primary": f"http://127.0.0.1:{p1}", "secondary": f"http://127.0.0.1:{p2}"}
        self.procs = [
            spawn("mock_provider:app", p1, {"PROVIDER_NAME": "primary"}),
            spawn("mock_provider:app", p2, {"PROVIDER_NAME": "secondary"}),
            spawn("app:app", rp, {"ROUTER_DB_PATH": self.db, "PRIMARY_BASE_URL": f"http://127.0.0.1:{p1}/v1",
                                  "SECONDARY_BASE_URL": f"http://127.0.0.1:{p2}/v1", **router_env}),
        ]
        self.url = f"http://127.0.0.1:{rp}/v1/chat/completions"

    def post(self, key: str | None = "default", timeout: float = 30, **body) -> httpx.Response:
        payload = {"model": "any", "max_tokens": 10, "messages": [{"role": "user", "content": "hello"}], **body}
        headers = {} if key is None else {"Authorization": f"Bearer {self.key if key == 'default' else key}"}
        return httpx.post(self.url, json=payload, headers=headers, timeout=timeout)

    def used(self, tenant: str = "acme") -> int:
        with sqlite3.connect(self.db) as db:
            return db.execute("SELECT COALESCE(SUM(tokens),0) FROM token_usage WHERE tenant=? AND ts > ?",
                              (tenant, time.time() - 3600)).fetchone()[0]

    def stop(self) -> None:
        for p in self.procs:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()  # a mock still sleeping in "hang" mode delays graceful shutdown
                p.wait()
            p.stderr.close()


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    s = Stack(tmp_path_factory.mktemp("main"))  # spec defaults: 50,000 tokens / 60 s, primary timeout 3000 ms
    yield s
    s.stop()


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    s = Stack(tmp_path_factory.mktemp("small"), TOKENS_PER_MINUTE="1000", WINDOW_SECONDS="3")
    yield s
    s.stop()


# --- Failover --------------------------------------------------------------------------------------------


def test_primary_serves_normally(stack):
    r = stack.post()
    assert r.status_code == 200
    assert r.headers["x-model-provider"] == "primary"
    assert r.json()["choices"][0]["message"]["content"] == "Answer from primary."


@pytest.mark.parametrize("failure", ["429", "500"])
def test_primary_failure_falls_over_to_secondary(stack, failure):
    r = stack.post(mock={"primary": failure})
    assert r.status_code == 200
    assert r.headers["x-model-provider"] == "secondary"


def test_primary_slower_than_3000ms_falls_over(stack):
    start = time.perf_counter()
    r = stack.post(mock={"primary": "slow:4500"})
    elapsed = time.perf_counter() - start
    assert r.headers["x-model-provider"] == "secondary"
    assert 3.0 <= elapsed < 4.0  # switched at the 3 s deadline, didn't wait 4.5 s


def test_primary_answering_within_3000ms_is_kept(stack):
    r = stack.post(mock={"primary": "slow:2500"})
    assert r.headers["x-model-provider"] == "primary"


def test_hanging_primary_falls_over(stack):
    r = stack.post(mock={"primary": "hang"})
    assert r.headers["x-model-provider"] == "secondary"


# --- Error sanitization ------------------------------------------------------------------------------------


def test_all_providers_down_gives_a_clean_503(stack):
    r = stack.post(mock={"primary": "500", "secondary": "500"})
    error = r.json()["error"]
    assert r.status_code == 503
    assert error["type"] == "upstream_unavailable"
    assert error["request_id"] == r.headers["x-request-id"]
    assert not any(leak in r.text for leak in LEAKS)


def test_no_upstream_detail_ever_leaks(stack):
    for mock in [{"primary": "500"}, {"primary": "429", "secondary": "500"}, {"primary": "500", "secondary": "429"}]:
        r = stack.post(mock=mock)
        assert not any(leak in r.text for leak in LEAKS), mock
        assert not any(leak in str(r.headers) for leak in LEAKS), mock


@pytest.mark.parametrize(
    ("kwargs", "status", "kind"),
    [
        ({"key": None}, 401, "authentication_error"),
        ({"key": "sk-wrong"}, 401, "authentication_error"),
        ({"messages": []}, 400, "invalid_request"),
        ({"messages": "hi"}, 400, "invalid_request"),
        ({"max_tokens": 0}, 400, "invalid_request"),
        ({"max_tokens": True}, 400, "invalid_request"),
        ({"stream": True}, 400, "invalid_request"),
        ({"max_tokens": 60_000}, 400, "request_too_large"),
    ],
)
def test_client_errors_use_the_standard_shape(stack, kwargs, status, kind):
    r = stack.post(**kwargs)
    assert r.status_code == status
    assert set(r.json()["error"]) == {"type", "message", "request_id"}
    assert r.json()["error"]["type"] == kind


def test_framework_errors_use_the_standard_shape_too(stack):
    for r in [httpx.get(stack.url), httpx.post(stack.url.replace("/chat/completions", "/nope")),
              httpx.post(stack.url, content=b"not json", headers={"Authorization": f"Bearer {stack.key}"})]:
        assert set(r.json()["error"]) == {"type", "message", "request_id"}


# --- Token accounting ---------------------------------------------------------------------------------------


def test_tenant_is_charged_real_usage_once_even_after_failover(stack):
    before = stack.used("globex")
    r = stack.post(key=stack.other_key, mock={"primary": "slow:4000"})  # times out, then secondary serves
    assert r.headers["x-model-provider"] == "secondary"
    assert stack.used("globex") - before == r.json()["usage"]["total_tokens"]


def test_failed_requests_are_not_charged(stack):
    before = stack.used()
    stack.post(mock={"primary": "500", "secondary": "429"})
    assert stack.used() == before


def test_rate_limit_headers_on_success(stack):
    r = stack.post()
    assert r.headers["x-ratelimit-limit"] == "50000"
    assert 0 < int(r.headers["x-ratelimit-remaining"]) <= 50_000


# --- Rate limiting end to end (1,000 tokens per 3 s window) -------------------------------------------------------


def test_limit_is_enforced_with_retry_after_and_then_recovers(small):
    time.sleep(3.2)  # clean window
    body = {"max_tokens": 300, "messages": [{"role": "user", "content": "x" * 36}]}  # estimate: 9+4+300 = 313
    r = small.post(**body, mock={"primary": "slow:200"})
    assert r.status_code == 200  # settled down to real usage (~15), then:
    for _ in range(3):
        small.post(max_tokens=1, messages=[{"role": "user", "content": "x" * 1200}])  # ~305 each, real usage small
    big = {"max_tokens": 980, "messages": [{"role": "user", "content": "x"}]}  # estimate 985: won't fit with other usage
    r = small.post(**big)
    assert r.status_code == 429
    assert r.json()["error"]["type"] == "rate_limit_exceeded"
    assert 1 <= int(r.headers["retry-after"]) <= 3
    time.sleep(3.2)  # the window slides past everything
    assert small.post(**big).status_code == 200


def test_concurrent_requests_never_exceed_the_budget(small):
    time.sleep(3.2)  # clean window
    # 30 requests at once, each reserving ~105 tokens against a 1,000 budget. The primary is slow
    # (500 ms) so all reservations happen before any is settled: exactly 9 can fit.
    def one(_):
        return small.post(max_tokens=100, messages=[{"role": "user", "content": "hi"}], mock={"primary": "slow:500"}).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as pool:
        statuses = list(pool.map(one, range(30)))
    assert statuses.count(200) == 9
    assert statuses.count(429) == 21
