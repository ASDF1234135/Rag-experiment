from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

try:  # optional: psutil, only used to pick memory vs stream mode
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

try:  # optional: faiss-cpu
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _search_rows(vectors, q: np.ndarray, k: int, rows: list[int]) -> tuple[list[int], list[float]]:
    """Exact search restricted to a candidate pool (reduced-pool variant, spec section 9)."""
    sub = np.asarray(vectors[np.asarray(rows)], dtype=np.float32)
    sc = (sub @ normalize(q.reshape(1, -1))[0]).astype(np.float32)
    order = np.argsort(-sc, kind="stable")[:k]
    return [int(rows[i]) for i in order], [float(sc[i]) for i in order]


class FlatIndex:
    """Exact inner-product search over L2-normalised vectors (= cosine). No ANN, per spec.

    float16 vectors (used for the 1.7M-chunk BrowseComp-Plus corpus, where float32 would need
    ~10 GB) are scanned exhaustively with faiss's fp16 scalar quantiser: still brute force, with
    only fp16 rounding (measured max score error ~2e-5, identical top-10 sets). The whole matrix
    is held in RAM; use StreamingIndex when it does not fit."""

    def __init__(self, vectors: np.ndarray, use_faiss: bool | None = None):
        self.vectors = vectors  # (n, d), rows aligned with ChunkStore.row
        self.dim = vectors.shape[1]
        want = faiss is not None if use_faiss is None else use_faiss
        self._faiss = None
        if want:
            if faiss is None:
                raise RuntimeError("faiss is not installed")
            if vectors.dtype == np.float16:
                self._faiss = faiss.IndexScalarQuantizer(
                    self.dim, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
            else:
                self._faiss = faiss.IndexFlatIP(self.dim)
            for i in range(0, vectors.shape[0], 50_000):  # chunked: never materialise a full float32 copy
                self._faiss.add(np.ascontiguousarray(vectors[i : i + 50_000], dtype=np.float32))

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def search(self, q: np.ndarray, k: int, rows: list[int] | None = None) -> tuple[list[int], list[float]]:
        """Top-k rows. If `rows` is given, search only that candidate pool."""
        if rows is not None:
            return _search_rows(self.vectors, q, k, rows)
        q = normalize(q.reshape(1, -1))
        if self._faiss is not None:
            scores, ids = self._faiss.search(q, k)
            return [int(i) for i in ids[0] if i >= 0], [float(s) for s, i in zip(scores[0], ids[0]) if i >= 0]
        sc = np.concatenate(
            [np.asarray(self.vectors[i : i + 200_000], dtype=np.float32) @ q[0] for i in range(0, len(self), 200_000)]
        )
        order = np.argsort(-sc, kind="stable")[:k]
        return [int(i) for i in order], [float(sc[i]) for i in order]

    def save(self, path: str | Path) -> None:
        np.save(path, np.asarray(self.vectors))

    @classmethod
    def load(cls, path: str | Path, use_faiss: bool | None = None) -> "FlatIndex":
        return cls(np.load(path, mmap_mode="r"), use_faiss=use_faiss)


class _Coalescer:
    """Gathers concurrent single-query searches into one batch, so one pass over the data serves
    many queries (the pass is what costs disk bandwidth)."""

    def __init__(self, fn, max_batch: int, max_wait_s: float):
        self._fn, self._max_batch, self._max_wait = fn, max_batch, max_wait_s
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, q: np.ndarray, k: int):
        fut: Future = Future()
        self._q.put((q, k, fut))
        return fut.result()

    def _loop(self) -> None:
        while True:
            batch = [self._q.get()]
            deadline = time.monotonic() + self._max_wait
            while len(batch) < self._max_batch:
                try:
                    batch.append(self._q.get(timeout=max(0.0, deadline - time.monotonic())))
                except queue.Empty:
                    break
            by_k: dict[int, list] = {}
            for item in batch:
                by_k.setdefault(item[1], []).append(item)
            for k, items in by_k.items():
                try:
                    res = self._fn(np.stack([it[0] for it in items]), k)
                    for it, r in zip(items, res):
                        it[2].set_result(r)
                except Exception as e:  # noqa: BLE001 - hand the error to every waiting caller
                    for it in items:
                        it[2].set_exception(e)


class StreamingIndex:
    """Exact search without holding the index in RAM: the memory-mapped matrix is scanned in
    blocks of `block_rows`, keeping a running top-k. Peak memory is about
    `threads * block_rows * dim * 4` bytes whatever the corpus size. Concurrent callers are
    coalesced into batches (`search_batch`): N parallel queries cost one scan, not N."""

    _faiss = None

    def __init__(self, vectors: np.ndarray, block_rows: int = 20_000, threads: int = 4,
                 max_batch: int = 64, max_wait_ms: float = 10.0, coalesce: bool = True,
                 path: str | Path | None = None):
        self.vectors, self.dim = vectors, vectors.shape[1]
        # With a file path, blocks are read with explicit file reads: scanning a memory map would
        # keep every touched page in this process's working set and squeeze free RAM.
        self._path = str(path) if path else None
        self._offset = int(getattr(vectors, "offset", 0)) if path else 0
        self.block_rows, self.threads = block_rows, threads
        self._pool = ThreadPoolExecutor(max_workers=threads)
        self._co = _Coalescer(self.search_batch, max_batch, max_wait_ms / 1000) if coalesce else None

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def _scan_block(self, start: int, Q: np.ndarray, k: int):
        if self._path:
            rows = min(self.block_rows, len(self) - start)
            raw = np.fromfile(self._path, dtype=self.vectors.dtype, count=rows * self.dim,
                              offset=self._offset + start * self.dim * self.vectors.dtype.itemsize)
            blk = raw.reshape(rows, self.dim).astype(np.float32)
        else:
            blk = np.asarray(self.vectors[start : start + self.block_rows], dtype=np.float32)
        S = blk @ Q.T  # (rows in block, n_queries)
        if k < S.shape[0]:
            idx = np.argpartition(-S, k - 1, axis=0)[:k]
        else:
            idx = np.tile(np.arange(S.shape[0])[:, None], (1, S.shape[1]))
        return idx + start, np.take_along_axis(S, idx, axis=0)

    def search_batch(self, Q: np.ndarray, k: int) -> list[tuple[list[int], list[float]]]:
        Q = normalize(Q)
        parts = list(self._pool.map(lambda s: self._scan_block(s, Q, k), range(0, len(self), self.block_rows)))
        idx = np.concatenate([p[0] for p in parts])  # (blocks * min(k, block), n_queries)
        sc = np.concatenate([p[1] for p in parts])
        out = []
        for j in range(Q.shape[0]):
            order = np.lexsort((idx[:, j], -sc[:, j]))[:k]  # score desc, row asc: deterministic
            out.append(([int(idx[i, j]) for i in order], [float(sc[i, j]) for i in order]))
        return out

    def search(self, q: np.ndarray, k: int, rows: list[int] | None = None) -> tuple[list[int], list[float]]:
        if rows is not None:
            return _search_rows(self.vectors, q, k, rows)
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        return self._co.submit(q, k) if self._co else self.search_batch(q[None], k)[0]


def load_index(path: str | Path, cfg=None):
    """Choose the index implementation (`cfg` is config.IndexCfg). mode=auto uses the in-RAM faiss
    scan only if the matrix fits in `memory_fraction` of the RAM that is currently free."""
    from .config import IndexCfg

    cfg = cfg or IndexCfg()
    vecs = np.load(path, mmap_mode="r")
    mode = cfg.mode
    if mode == "auto":
        need = vecs.shape[0] * vecs.shape[1] * (2 if vecs.dtype == np.float16 else 4)
        avail = psutil.virtual_memory().available if psutil else float("inf")
        mode = "memory" if need <= cfg.memory_fraction * avail else "stream"
    if mode == "memory":
        return FlatIndex(vecs)
    if mode == "stream":
        return StreamingIndex(vecs, cfg.block_rows, cfg.threads, cfg.max_batch, cfg.max_wait_ms, path=path)
    raise ValueError(f"unknown index mode {mode!r}")
