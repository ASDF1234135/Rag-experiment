from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str  # body only; the title prefix is added when embedding / serializing
    n_tokens: int  # body tokens (cl100k_base)


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    score: float
    aux: float = 0.0  # tie-breaker (Jev `is_relevant`); 0 for embedding retrieval


@dataclass
class Question:
    qid: str
    question: str
    answer: str
    gold_doc_ids: list[str]
    evidence_doc_ids: list[str] = field(default_factory=list)
    negative_doc_ids: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
