"""Tenant API keys, stored in the same SQLite database.

Keys are never stored in plain text, only their SHA-256 hash, so a leaked
database doesn't leak working keys.

Create a key: uv run python tenants.py add acme
"""

import hashlib
import secrets
import sqlite3
import sys


def _hash(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


class TenantStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        with sqlite3.connect(db_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS api_keys (key_hash TEXT PRIMARY KEY, tenant TEXT NOT NULL)")

    def add(self, tenant: str, api_key: str | None = None) -> str:
        api_key = api_key or f"sk-tenant-{secrets.token_urlsafe(24)}"
        with sqlite3.connect(self.db_path) as db:
            db.execute("INSERT OR REPLACE INTO api_keys (key_hash, tenant) VALUES (?, ?)", (_hash(api_key), tenant))
        return api_key

    def tenant_for(self, api_key: str) -> str | None:
        with sqlite3.connect(self.db_path) as db:
            row = db.execute("SELECT tenant FROM api_keys WHERE key_hash = ?", (_hash(api_key),)).fetchone()
        return row[0] if row else None


if __name__ == "__main__":
    import os

    if len(sys.argv) != 3 or sys.argv[1] != "add":
        sys.exit("usage: python tenants.py add <tenant>")
    print(TenantStore(os.environ.get("ROUTER_DB_PATH", "router.db")).add(sys.argv[2]))
