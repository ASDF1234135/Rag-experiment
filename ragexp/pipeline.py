"""Corpus preparation: chunk -> store -> embed -> exact index; dev/test split; candidate pools."""

from __future__ import annotations

import hashlib
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np

from .chunking import chunk_document, embed_text
from .config import ChunkCfg, EmbedCfg
from .index import normalize
from .store import ChunkStore
from .types import Doc, Question


def build_store(docs: Iterable[Doc], store: ChunkStore, enc, cfg: ChunkCfg, batch_docs: int = 500) -> int:
    """Chunk every document into the store. Idempotent per doc: skips docs already present."""
    done = set(store.doc_ids()) if len(store) else set()
    buf, n = [], 0
    for d in docs:
        if d.doc_id in done:
            continue
        buf.extend(chunk_document(d, enc, cfg))
        n += 1
        if len(buf) >= batch_docs * 4:
            store.add_many(buf)
            store.commit()
            buf = []
    if buf:
        store.add_many(buf)
    store.commit()
    return n


def embed_store(store: ChunkStore, embedder, cfg: EmbedCfg, out_dir: str | Path, log=print) -> Path:
    """Embed all chunks (`title\\nbody`) into out_dir/shard_XXXXX.npy, resumable, then assemble emb.npy."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_rows = cfg.batch_size * cfg.workers * 2  # small shards: progress survives interruptions
    n = len(store)
    n_shards = (n + shard_rows - 1) // shard_rows
    for s in range(n_shards):
        path = out_dir / f"shard_{s:05d}.npy"
        if path.exists():
            continue
        chunks = next(store.iter_chunks(batch=shard_rows, start_row=s * shard_rows))
        batches, cur, cur_tok = [], [], 0
        for c in chunks:  # token-aware batching: at most batch_size texts and max_batch_tokens tokens
            t = c.n_tokens + len(c.title) // 3 + 4
            if cur and (len(cur) >= cfg.batch_size or cur_tok + t > cfg.max_batch_tokens):
                batches.append(cur)
                cur, cur_tok = [], 0
            cur.append(c)
            cur_tok += t
        if cur:
            batches.append(cur)

        def one(b):
            est = sum(c.n_tokens + len(c.title) / 4 + 2 for c in b)  # exact body tokens + title estimate
            v, meta = embedder.embed([embed_text(c) for c in b], use_cache=False, est_tokens=est)
            return v, meta.input_tokens

        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            res = list(ex.map(one, batches))
        np.save(path, normalize(np.concatenate([v for v, _ in res])).astype(np.float16))
        (out_dir / f"shard_{s:05d}.tokens").write_text(str(sum(t for _, t in res)))
        log(f"embedded shard {s + 1}/{n_shards}")
    if n_shards == 0:
        raise ValueError("no chunks to embed")
    first = np.load(out_dir / "shard_00000.npy", mmap_mode="r")
    full = np.lib.format.open_memmap(out_dir / "emb.npy", mode="w+", dtype=np.float16, shape=(n, first.shape[1]))
    for s in range(n_shards):  # assemble shard by shard, never all in RAM
        m = np.load(out_dir / f"shard_{s:05d}.npy", mmap_mode="r")
        full[s * shard_rows : s * shard_rows + len(m)] = m
    full.flush()
    del full
    return out_dir / "emb.npy"


def split_dev_test(qids: list[str], seed: int, dev_fraction: float) -> dict[str, list[str]]:
    """Fixed-seed dev/test split. Sorting first makes it independent of input order."""
    ids = sorted(qids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_dev = int(round(len(ids) * dev_fraction))
    return {"dev": sorted(ids[:n_dev]), "test": sorted(ids[n_dev:])}


def subsample(qids: list[str], fraction: float, seed: int) -> list[str]:
    """Seeded subsample of `fraction` of qids. Ordering by a hash of (seed, qid) makes the subsets nested:
    raising the fraction later only adds questions, so finished work is never invalidated."""
    ids = sorted(qids)
    if fraction >= 1.0:
        return ids
    n = max(1, round(len(ids) * fraction))
    ranked = sorted(ids, key=lambda q: hashlib.sha256(f"{seed}:{q}".encode()).hexdigest())
    return sorted(ranked[:n])


def build_pool(q: Question, store: ChunkStore, pool_size: int, seed: int) -> list[str]:
    """Reduced candidate pool (spec section 9): gold + evidence + hard-negative docs' chunks, topped up
    with fixed random distractor chunks to `pool_size`. Same pool must be used by G1-G4 and G5."""
    rel_docs = list(dict.fromkeys(q.gold_doc_ids + q.evidence_doc_ids + q.negative_doc_ids))
    pool: list[str] = []
    for d in rel_docs:
        pool.extend(store.doc_chunk_ids(d))
    have = set(pool)
    if len(pool) < pool_size:
        h = int(hashlib.sha256(f"{seed}:{q.qid}".encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(h)
        all_docs = store.doc_ids()
        rng.shuffle(all_docs)
        for d in all_docs:
            if len(pool) >= pool_size:
                break
            for cid in store.doc_chunk_ids(d):
                if cid not in have:
                    pool.append(cid)
                    have.add(cid)
                    if len(pool) >= pool_size:
                        break
    return pool
