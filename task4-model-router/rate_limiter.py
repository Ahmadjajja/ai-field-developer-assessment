"""Token-aware sliding-window rate limiter, stored in SQLite on disk.

Each tenant may use at most `limit` tokens in any `window`-second period
(default 50,000 tokens per 60 s). "Sliding" means the period is always the
last 60 seconds, not a calendar minute, so a burst can't double up at a
minute boundary.

Every request reserves its tokens *before* it is sent to a model:

    reserve()  check the last 60 s, then record an estimate; one atomic step
    settle()   after the response, replace the estimate with the real usage
    release()  the request failed and used nothing: give the tokens back

The check and the insert run inside one `BEGIN IMMEDIATE` transaction, which
takes SQLite's write lock. Concurrent requests, even from several gateway
processes sharing the database file, therefore line up one by one and can
never together exceed the limit.

Old rows are evicted (deleted) inside the same transaction, so the table only
ever holds the last window's worth of reservations.
"""

import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from collections.abc import Callable

SCHEMA = """
CREATE TABLE IF NOT EXISTS token_usage (
    id      INTEGER PRIMARY KEY,
    tenant  TEXT    NOT NULL,
    ts      REAL    NOT NULL,      -- when the tokens were reserved (unix seconds)
    tokens  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS token_usage_tenant_ts ON token_usage (tenant, ts);
CREATE INDEX IF NOT EXISTS token_usage_ts ON token_usage (ts);
"""


class RateLimited(Exception):
    def __init__(self, retry_after: float, used: int, limit: int) -> None:
        super().__init__(f"token limit reached ({used}/{limit})")
        self.retry_after, self.used, self.limit = retry_after, used, limit


class RequestTooLarge(Exception):
    """A single request needs more tokens than the whole window allows: it can never succeed."""


@dataclass(frozen=True)
class Reservation:
    id: int
    tenant: str
    tokens: int
    remaining: int  # tokens left in the window after this reservation
    reset_after: float  # seconds until the oldest reservation in the window expires


class SlidingWindowLimiter:
    def __init__(self, db_path: str, limit: int = 50_000, window: float = 60.0,
                 clock: Callable[[], float] = time.time) -> None:
        self.db_path, self.limit, self.window, self.clock = db_path, limit, window, clock
        self._local = threading.local()
        with self._connect() as db:
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        """One connection per thread (sqlite3 connections must not be shared across threads)."""
        db = getattr(self._local, "db", None)
        if db is None:
            # isolation_level=None: we issue BEGIN/COMMIT ourselves. timeout: wait for the write lock.
            db = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
            db.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
            db.execute("PRAGMA synchronous=NORMAL")
            self._local.db = db
        return db

    def reserve(self, tenant: str, tokens: int) -> Reservation:
        if tokens > self.limit:
            raise RequestTooLarge(f"request needs {tokens} tokens, limit is {self.limit} per window")
        db = self._connect()
        now = self.clock()
        window_start = now - self.window
        db.execute("BEGIN IMMEDIATE")  # take the write lock now: check + insert are one atomic step
        try:
            db.execute("DELETE FROM token_usage WHERE ts <= ?", (window_start,))  # evict expired rows
            rows = db.execute(
                "SELECT ts, tokens FROM token_usage WHERE tenant = ? ORDER BY ts", (tenant,)
            ).fetchall()
            used = sum(t for _, t in rows)
            if used + tokens > self.limit:
                db.execute("COMMIT")  # keep the eviction
                raise RateLimited(self._retry_after(rows, used, tokens, now), used, self.limit)
            cur = db.execute("INSERT INTO token_usage (tenant, ts, tokens) VALUES (?, ?, ?)", (tenant, now, tokens))
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        oldest = rows[0][0] if rows else now
        return Reservation(cur.lastrowid, tenant, tokens, self.limit - used - tokens, max(0.0, oldest + self.window - now))

    def _retry_after(self, rows: list[tuple[float, int]], used: int, tokens: int, now: float) -> float:
        """Seconds until enough old reservations expire for `tokens` to fit."""
        for ts, t in rows:
            used -= t
            if used + tokens <= self.limit:
                return max(0.0, ts + self.window - now)
        return self.window

    def settle(self, reservation: Reservation, actual_tokens: int) -> None:
        """Replace the estimate with what the request really used (keeps the original timestamp)."""
        db = self._connect()
        db.execute("UPDATE token_usage SET tokens = ? WHERE id = ?", (max(0, int(actual_tokens)), reservation.id))

    def release(self, reservation: Reservation) -> None:
        """The request was never served: give its tokens back."""
        db = self._connect()
        db.execute("DELETE FROM token_usage WHERE id = ?", (reservation.id,))

    def used(self, tenant: str) -> int:
        db = self._connect()
        row = db.execute(
            "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ? AND ts > ?",
            (tenant, self.clock() - self.window),
        ).fetchone()
        return int(row[0])


def estimate_tokens(messages: list, max_tokens: int) -> int:
    """Upper-bound estimate before the call: prompt (~4 characters per token) + the completion cap."""
    prompt = 0
    for message in messages:
        content = message.get("content", "") if isinstance(message, dict) else ""
        text = content if isinstance(content, str) else str(content)
        prompt += math.ceil(len(text) / 4) + 4  # + per-message overhead (role, separators)
    return prompt + max_tokens
