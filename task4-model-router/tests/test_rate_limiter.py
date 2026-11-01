"""Unit tests for the SQLite sliding-window limiter."""

import multiprocessing
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rate_limiter import RateLimited, RequestTooLarge, SlidingWindowLimiter, estimate_tokens  # noqa: E402


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def limiter(tmp_path, clock):
    return SlidingWindowLimiter(str(tmp_path / "rl.db"), limit=50_000, window=60, clock=clock)


def test_allows_up_to_the_limit_and_no_further(limiter):
    limiter.reserve("acme", 30_000)
    assert limiter.reserve("acme", 20_000).remaining == 0
    with pytest.raises(RateLimited) as exc:
        limiter.reserve("acme", 1)
    assert exc.value.used == 50_000


def test_window_slides_instead_of_resetting_on_the_minute(limiter, clock):
    limiter.reserve("acme", 30_000)  # t = 0
    clock.now += 30
    limiter.reserve("acme", 20_000)  # t = 30
    clock.now += 29.9  # t = 59.9: the first reservation is still inside the window
    with pytest.raises(RateLimited):
        limiter.reserve("acme", 1)
    clock.now += 0.2  # t = 60.1: it has expired, 30,000 tokens are free again
    limiter.reserve("acme", 30_000)
    with pytest.raises(RateLimited):
        limiter.reserve("acme", 1)


def test_retry_after_says_when_enough_tokens_free_up(limiter, clock):
    limiter.reserve("acme", 10_000)  # t = 0
    clock.now += 20
    limiter.reserve("acme", 40_000)  # t = 20
    clock.now += 5  # t = 25
    with pytest.raises(RateLimited) as small:
        limiter.reserve("acme", 5_000)  # fits once the t=0 reservation expires (t=60)
    assert small.value.retry_after == pytest.approx(35)
    with pytest.raises(RateLimited) as big:
        limiter.reserve("acme", 45_000)  # needs both to expire (t=80)
    assert big.value.retry_after == pytest.approx(55)


def test_expired_rows_are_evicted_from_the_database(limiter, clock):
    for _ in range(10):
        limiter.reserve("acme", 1_000)
    clock.now += 61
    limiter.reserve("acme", 1)
    count = limiter._connect().execute("SELECT COUNT(*) FROM token_usage").fetchone()[0]
    assert count == 1


def test_tenants_are_independent(limiter):
    limiter.reserve("acme", 50_000)
    assert limiter.reserve("globex", 50_000).remaining == 0


def test_settle_replaces_the_estimate_with_real_usage(limiter):
    reservation = limiter.reserve("acme", 40_000)
    limiter.settle(reservation, 1_200)
    assert limiter.used("acme") == 1_200
    limiter.reserve("acme", 48_800)  # the freed tokens are usable again


def test_release_gives_tokens_back(limiter):
    reservation = limiter.reserve("acme", 50_000)
    limiter.release(reservation)
    assert limiter.used("acme") == 0


def test_request_bigger_than_the_window_is_rejected_outright(limiter):
    with pytest.raises(RequestTooLarge):
        limiter.reserve("acme", 50_001)


def test_usage_survives_a_restart(tmp_path, clock):
    path = str(tmp_path / "rl.db")
    SlidingWindowLimiter(path, clock=clock).reserve("acme", 45_000)
    restarted = SlidingWindowLimiter(path, clock=clock)  # e.g. after a deploy
    with pytest.raises(RateLimited):
        restarted.reserve("acme", 10_000)


def test_concurrent_threads_never_overshoot_or_error(tmp_path):
    limiter = SlidingWindowLimiter(str(tmp_path / "rl.db"), limit=50_000)
    accepted, limited, errors = [], [], []
    start = threading.Barrier(120)

    def worker():
        start.wait()  # release all 120 at the same moment
        try:
            limiter.reserve("race", 1_000)
            accepted.append(1)
        except RateLimited:
            limited.append(1)
        except Exception as exc:  # e.g. "database is locked": a request that failed instead of being answered
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker) for _ in range(120)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert (len(accepted), len(limited)) == (50, 70)
    assert limiter.used("race") == 50_000


def _process_worker(path: str, results) -> None:
    limiter = SlidingWindowLimiter(path, limit=50_000)
    for _ in range(40):
        try:
            limiter.reserve("multi", 1_000)
            results.put(1)
        except RateLimited:
            results.put(0)


def test_several_gateway_processes_sharing_the_db_never_overshoot(tmp_path):
    path = str(tmp_path / "rl.db")
    SlidingWindowLimiter(path)  # create the schema
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    procs = [ctx.Process(target=_process_worker, args=(path, results)) for _ in range(3)]  # 120 attempts
    for p in procs:
        p.start()
    outcomes = [results.get(timeout=60) for _ in range(120)]
    for p in procs:
        p.join(timeout=30)
    assert sum(outcomes) == 50
    assert SlidingWindowLimiter(path).used("multi") == 50_000


def test_estimate_tokens():
    messages = [{"role": "user", "content": "x" * 400}, {"role": "assistant", "content": ""}]
    assert estimate_tokens(messages, max_tokens=256) == (100 + 4) + (0 + 4) + 256
