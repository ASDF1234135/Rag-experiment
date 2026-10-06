from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Iterator

from .types import Chunk


class ChunkStore:
    """sqlite-backed chunk table. `row` is the position of the chunk in the embedding matrix."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks (row INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE NOT NULL,"
            " doc_id TEXT NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL, n_tokens INTEGER NOT NULL)"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_doc ON chunks(doc_id)")
        self._count = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def __len__(self) -> int:
        return self._count

    def add_many(self, chunks: list[Chunk]) -> None:
        with self._lock:
            rows = []
            for c in chunks:
                rows.append((self._count, c.chunk_id, c.doc_id, c.title, c.text, c.n_tokens))
                self._count += 1
            self._conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?)", rows)

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    @staticmethod
    def _chunk(r) -> Chunk:
        return Chunk(chunk_id=r[1], doc_id=r[2], title=r[3], text=r[4], n_tokens=r[5])

    def get(self, chunk_id: str) -> Chunk:
        with self._lock:
            r = self._conn.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        if r is None:
            raise KeyError(chunk_id)
        return self._chunk(r)

    def get_rows(self, rows: list[int]) -> list[Chunk]:
        out = []
        with self._lock:
            for i in rows:
                r = self._conn.execute("SELECT * FROM chunks WHERE row=?", (int(i),)).fetchone()
                out.append(self._chunk(r))
        return out

    def rows_for_ids(self, chunk_ids: list[str]) -> list[int]:
        out = []
        with self._lock:
            for cid in chunk_ids:
                r = self._conn.execute("SELECT row FROM chunks WHERE chunk_id=?", (cid,)).fetchone()
                if r is not None:
                    out.append(r[0])
        return out

    def doc_chunk_ids(self, doc_id: str) -> list[str]:
        with self._lock:
            rs = self._conn.execute("SELECT chunk_id FROM chunks WHERE doc_id=? ORDER BY row", (doc_id,)).fetchall()
        return [r[0] for r in rs]

    def doc_ids(self) -> list[str]:
        with self._lock:
            rs = self._conn.execute("SELECT DISTINCT doc_id FROM chunks ORDER BY doc_id").fetchall()
        return [r[0] for r in rs]

    def iter_chunks(self, batch: int = 2000, start_row: int = 0) -> Iterator[list[Chunk]]:
        row = start_row
        while True:
            with self._lock:
                rs = self._conn.execute(
                    "SELECT * FROM chunks WHERE row>=? ORDER BY row LIMIT ?", (row, batch)
                ).fetchall()
            if not rs:
                return
            yield [self._chunk(r) for r in rs]
            row = rs[-1][0] + 1

    def close(self) -> None:
        self.commit()
        self._conn.close()
