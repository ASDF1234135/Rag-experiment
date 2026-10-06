from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


def request_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ApiCache:
    """Request-hash -> response cache (sqlite). Re-runs and re-analysis cost nothing.

    Stochastic calls must put a replicate id in the hashed payload (`salt`), otherwise
    repeated runs would silently return the first sample.
    """

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=120)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._lock = threading.Lock()
        self._pending = 0

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache(key, value) VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False)),
            )
            # Commit every write: API calls take seconds, so this is cheap, and it never holds the
            # write lock across calls (several processes may share one cache file).
            self._conn.commit()

    def flush(self) -> None:
        with self._lock:
            self._conn.commit()
            self._pending = 0

    def close(self) -> None:
        self.flush()
        self._conn.close()
