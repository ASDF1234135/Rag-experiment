from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .clients import CallMeta, EmbedClient, JevClient
from .index import FlatIndex
from .prompts import JEV_RANK
from .store import ChunkStore
from .types import Hit


@dataclass
class RetrievalResult:
    hits: list[Hit]
    calls: list[tuple[str, CallMeta]] = field(default_factory=list)  # (component, meta)
    extra: dict = field(default_factory=dict)
    ranking: list[Hit] = field(default_factory=list)  # top max(k, 10), for NDCG@10 / MRR (hits = ranking[:k])

RANK_DEPTH = 10


class EmbeddingRetriever:
    """Exact cosine search (G1-G4, G6, G7 and the ReAct tool). `pool` restricts candidates
    (reduced-pool variant, spec section 9)."""

    def __init__(self, store: ChunkStore, index: FlatIndex, embedder: EmbedClient):
        self.store, self.index, self.embedder = store, index, embedder

    def search(self, query: str, k: int, pool: list[str] | None = None) -> RetrievalResult:
        vec, meta_e = self.embedder.embed([query])
        t0 = time.perf_counter()
        rows = self.store.rows_for_ids(pool) if pool is not None else None
        top, scores = self.index.search(vec[0], max(k, RANK_DEPTH), rows=rows)
        meta_r = CallMeta(provider="local", model=("faiss-flat" if self.index._faiss else
                                 "stream-flat" if hasattr(self.index, "block_rows") else "numpy-flat"),
                          latency_ms=(time.perf_counter() - t0) * 1000)
        ranking = [Hit(c.chunk_id, c.doc_id, s) for c, s in zip(self.store.get_rows(top), scores)]
        return RetrievalResult(ranking[:k], [("embed_query", meta_e), ("retrieve", meta_r)], ranking=ranking)


class RerankRetriever:
    """G8: embedding shortlists `shortlist` chunks, then Jev scores only those (same two questions as G5) and
    re-ranks them. Ties keep the embedding order. Costs ~shortlist Jev requests per round instead of the whole pool."""

    def __init__(self, store: ChunkStore, index: FlatIndex, embedder: EmbedClient, jev: JevClient,
                 shortlist: int = 200, workers: int = 16):
        self._emb = EmbeddingRetriever(store, index, embedder)
        self._jev = JevRetriever(store, jev, workers)
        self.shortlist = shortlist

    def search(self, query: str, k: int, pool: list[str] | None = None) -> RetrievalResult:
        first = self._emb.search(query, max(self.shortlist, k), pool=pool)
        second = self._jev.search(query, k, pool=[h.chunk_id for h in first.hits])
        return RetrievalResult(second.hits, first.calls + second.calls, extra=second.extra, ranking=second.ranking)


class JevRetriever:
    """G5: no embeddings. Every (query, chunk) pair is one Jev request; rank by expected
    `usefulness` level, ties broken by `is_relevant`. Cost scales with the candidate count,
    so a candidate pool is normally required for BrowseComp-Plus (spec section 9)."""

    def __init__(self, store: ChunkStore, jev: JevClient, workers: int = 16):
        self.store, self.jev, self.workers = store, jev, workers

    def _score(self, query: str, chunk) -> tuple[float, float, CallMeta]:
        state = {"query": query, "passage": {"id": chunk.chunk_id, "title": chunk.title, "text": chunk.text}}
        r = self.jev.ask(state, JEV_RANK)
        return (float(r.answers["usefulness"]["score"]), float(r.answers["is_relevant"]["noul"]), r.meta)

    def search(self, query: str, k: int, pool: list[str] | None = None) -> RetrievalResult:
        if pool is not None:
            chunks = [self.store.get(cid) for cid in pool]
        else:
            chunks = [c for batch in self.store.iter_chunks() for c in batch]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            res = list(ex.map(lambda c: self._score(query, c), chunks))
        wall = (time.perf_counter() - t0) * 1000
        order = sorted(range(len(chunks)), key=lambda i: (-res[i][0], -res[i][1], i))
        ranking = [Hit(chunks[i].chunk_id, chunks[i].doc_id, res[i][0], res[i][1]) for i in order[:max(k, RANK_DEPTH)]]
        hits = ranking[:k]
        ties = len(res) - len({(round(s, 6), round(a, 6)) for s, a, _ in res})
        agg = CallMeta(
            provider="typesafe", model=self.jev.cfg.model,
            input_tokens=sum(m.input_tokens for *_, m in res), output_tokens=sum(m.output_tokens for *_, m in res),
            latency_ms=wall, retry_wait_ms=sum(m.retry_wait_ms for *_, m in res),
            retries=sum(m.retries for *_, m in res), n_calls=len(res), cached=all(m.cached for *_, m in res) if res else False,
        )
        return RetrievalResult(hits, [("jev_rank", agg)], ranking=ranking,
                               extra={"n_calls": len(res), "n_tied_pairs": ties,
                                      "latency_sum_ms": sum(m.latency_ms for *_, m in res)})
