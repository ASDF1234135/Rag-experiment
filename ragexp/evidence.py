from __future__ import annotations

from .store import ChunkStore
from .types import Chunk, Hit


def serialize_chunk(c: Chunk) -> str:
    """Shared serialization for LLM and Jev: `[chunk_id] title\\ntext`."""
    return f"[{c.chunk_id}] {c.title}\n{c.text}"


class Evidence:
    """Accumulated evidence (spec 5.1): dedupe by chunk id, keep the best score per chunk, order by
    score descending, and keep the best prefix whose serialized size fits in `e_max` tokens.

    All chunks ever seen stay in the pool, so a chunk displaced by a later round can come back if it
    is retrieved again with a higher score. `seen_chunk_ids` is everything retrieved (for recall).
    """

    def __init__(self, store: ChunkStore, enc, e_max: int):
        self.store, self.enc, self.e_max = store, enc, e_max
        self._best: dict[str, tuple[float, float]] = {}  # chunk_id -> (score, aux)
        self._order: dict[str, int] = {}  # first-seen order, for stable ties
        self._chunks: dict[str, Chunk] = {}
        self._ntok: dict[str, int] = {}
        self.selected: list[Chunk] = []

    def _tokens(self, c: Chunk) -> int:
        if c.chunk_id not in self._ntok:
            self._ntok[c.chunk_id] = len(self.enc.encode(serialize_chunk(c), disallowed_special=()))
        return self._ntok[c.chunk_id]

    def add(self, hits: list[Hit]) -> None:
        for h in hits:
            key = (h.score, h.aux)
            if h.chunk_id not in self._best:
                self._order[h.chunk_id] = len(self._order)
                self._chunks[h.chunk_id] = self.store.get(h.chunk_id)
                self._best[h.chunk_id] = key
            elif key > self._best[h.chunk_id]:
                self._best[h.chunk_id] = key
        ranked = sorted(self._best, key=lambda i: (-self._best[i][0], -self._best[i][1], self._order[i]))
        total, sel = 0, []
        for cid in ranked:
            n = self._tokens(self._chunks[cid])
            if total + n > self.e_max:
                break
            total += n
            sel.append(self._chunks[cid])
        self.selected = sel

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.selected]

    @property
    def doc_ids(self) -> list[str]:
        return list(dict.fromkeys(c.doc_id for c in self.selected))

    @property
    def seen_chunk_ids(self) -> list[str]:
        return list(self._order)

    @property
    def n_tokens(self) -> int:
        return sum(self._tokens(c) for c in self.selected)

    def text(self) -> str:
        return "\n\n".join(serialize_chunk(c) for c in self.selected)

    def jev_items(self) -> list[dict]:
        return [{"id": c.chunk_id, "title": c.title, "text": c.text} for c in self.selected]


def coverage(doc_ids: list[str], gold: list[str]) -> tuple[bool, bool]:
    """(covered_all, covered_any) of the gold documents by a set of doc ids."""
    have = set(doc_ids)
    g = set(gold)
    if not g:
        return False, False
    return g <= have, bool(g & have)
