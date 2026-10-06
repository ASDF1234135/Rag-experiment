"""Shared multi-round trajectory (spec section 10).

G3/G4/G6/G7 use the same query-generation logic, so one full R_max trajectory per
(question, replicate) is run, with every round recording the LLM judgement, the Jev probability
and the oracle label. Each group's stopping round is then simulated offline (see simulate.py).
G5 (Jev retrieval) runs its own trajectory, optionally stopping live on a frozen tau.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

from .clients import CallMeta
from .config import SearchCfg
from .evidence import Evidence, coverage
from .judges import JevJudge, LLMJudge, QueryGenerator
from .types import Question


@dataclass
class Round:
    r: int
    query: str
    retrieved_chunk_ids: list[str]
    retrieved_doc_ids: list[str]
    scores: list[float]
    ranking_doc_ids: list[str]  # top-10 chunk ranking (doc ids, with repeats) for NDCG@10 / MRR
    evidence_chunk_ids: list[str]  # accumulated evidence after this round
    evidence_doc_ids: list[str]
    evidence_tokens: int
    covered_all: bool
    covered_any: bool
    llm_sufficient: bool | None = None
    llm_parse_ok: bool | None = None
    jev_p: float | None = None
    retrieval_calls: list[tuple[str, dict]] = field(default_factory=list)  # (component, CallMeta)
    judge_calls: dict[str, dict] = field(default_factory=dict)  # component -> CallMeta
    qgen_call: dict | None = None  # cost of writing the query for round r+1
    retrieval_extra: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    qid: str
    replicate: int
    dataset: str
    rounds: list[Round]
    live_stop: str | None = None  # set when the trajectory stopped early (G5)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        rounds = [Round(**{**r, "retrieval_calls": [tuple(x) for x in r["retrieval_calls"]]}) for r in d["rounds"]]
        return cls(d["qid"], d["replicate"], d["dataset"], rounds, d.get("live_stop"))


def run_trajectory(
    q: Question,
    dataset: str,
    replicate: int,
    retriever,
    store,
    enc,
    cfg: SearchCfg,
    qgen: QueryGenerator,
    llm_judge: LLMJudge | None,
    jev_judge: JevJudge | None,
    stop_fn: Callable[[Round], str | None] | None = None,
    pool: list[str] | None = None,
) -> Trajectory:
    """Run rounds until R_max (or until `stop_fn(round)` returns a reason). Query generation is
    identical whatever the judge says, which is what makes offline stop simulation exact."""
    salt = f"rep{replicate}"
    ev = Evidence(store, enc, cfg.e_max)
    queries: list[str] = []
    rounds: list[Round] = []
    query = q.question
    live_stop = None
    for r in range(1, cfg.r_max + 1):
        queries.append(query)
        res = retriever.search(query, cfg.top_k, pool=pool)
        ev.add(res.hits)
        ca, cn = coverage(ev.doc_ids, q.gold_doc_ids)
        rd = Round(
            r=r, query=query,
            retrieved_chunk_ids=[h.chunk_id for h in res.hits], retrieved_doc_ids=[h.doc_id for h in res.hits],
            scores=[h.score for h in res.hits], ranking_doc_ids=[h.doc_id for h in res.ranking],
            evidence_chunk_ids=ev.chunk_ids, evidence_doc_ids=ev.doc_ids, evidence_tokens=ev.n_tokens,
            covered_all=ca, covered_any=cn,
            retrieval_calls=[(c, m.to_dict()) for c, m in res.calls], retrieval_extra=res.extra,
        )
        etext, eitems = ev.text(), ev.jev_items()
        if llm_judge is not None:
            ok, meta, parsed = llm_judge.judge(q.question, etext, salt=salt)
            rd.llm_sufficient, rd.llm_parse_ok = ok, parsed
            rd.judge_calls["glm_judge"] = meta.to_dict()
        if jev_judge is not None:
            p, meta = jev_judge.judge(q.question, eitems)
            rd.jev_p = p
            rd.judge_calls["jev_judge"] = meta.to_dict()
        rounds.append(rd)
        if stop_fn is not None:
            live_stop = stop_fn(rd)
            if live_stop:
                break
        if r == cfg.r_max:
            break
        query, qmeta = qgen.generate(q.question, etext, queries, salt=salt)
        rd.qgen_call = qmeta.to_dict()
    return Trajectory(q.qid, replicate, dataset, rounds, live_stop)
