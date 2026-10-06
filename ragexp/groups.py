"""Experimental groups G1-G7 (spec section 4) -> uniform result rows + JSONL records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import prompts
from .clients import CallMeta, GLMClient, JevClient
from .config import Config
from .evidence import Evidence, coverage, serialize_chunk
from .judges import JevJudge, LLMJudge, QueryGenerator, generate_answer
from .recorder import Recorder
from .simulate import group_calls, stop_round, sum_components
from .store import ChunkStore
from .trajectory import Trajectory, run_trajectory
from .types import Question


@dataclass
class Env:
    cfg: Config
    store: ChunkStore
    enc: object
    llm: GLMClient
    emb_retriever: object  # EmbeddingRetriever
    recorder: Recorder
    run_id: str
    jev: JevClient | None = None
    jev_retriever: object | None = None  # JevRetriever (G5)
    rerank_retriever: object | None = None  # RerankRetriever (G8)
    qgen: QueryGenerator = field(init=False)
    llm_judge: LLMJudge = field(init=False)
    jev_judge: JevJudge | None = field(init=False, default=None)

    def __post_init__(self):
        self.qgen = QueryGenerator(self.llm)
        self.llm_judge = LLMJudge(self.llm)
        self.jev_judge = JevJudge(self.jev) if self.jev else None


def _emit(env: Env, group: str, dataset: str, q: Question, rep: int, rnd: int, component: str,
          meta: dict, **extra) -> None:
    env.recorder.write(
        run_id=env.run_id, group=group, dataset=dataset, query_id=q.qid, replicate=rep, round=rnd,
        component=component, model=meta.get("model"), provider=meta.get("provider"),
        input_tokens=meta.get("input_tokens"), output_tokens=meta.get("output_tokens"),
        reasoning_tokens=meta.get("reasoning_tokens"), latency_ms=meta.get("latency_ms"),
        retry_wait_ms=meta.get("retry_wait_ms"), http_status=meta.get("http_status"),
        retries=meta.get("retries"), n_calls=meta.get("n_calls", 1), cached=meta.get("cached"), input_hash=meta.get("input_hash"), **extra,
    )


def _evidence_text(store: ChunkStore, chunk_ids: list[str]) -> str:
    return "\n\n".join(serialize_chunk(store.get(c)) for c in chunk_ids)


def _row(group, dataset, q, rep, stop_r, reason, answer, ev_chunks, ev_docs, ca, cn, first_docs, comps,
         first_ranking=None, **extra) -> dict:
    return {
        "group": group, "dataset": dataset, "qid": q.qid, "replicate": rep, "stop_round": stop_r,
        "stop_reason": reason, "answer": answer, "evidence_chunk_ids": ev_chunks,
        "evidence_doc_ids": ev_docs, "covered_all": ca, "covered_any": cn,
        "first_round_doc_ids": first_docs, "first_round_ranking_doc_ids": first_ranking or first_docs,
        "components": comps,
        "latency_ms": sum(c["latency_ms"] for c in comps.values()), **extra,
    }


def finalize_from_trajectory(env: Env, q: Question, traj: Trajectory, group: str, stop_r: int,
                             reason: str) -> dict:
    """Generate the answer at the group's own stop round and emit its records."""
    rd = traj.rounds[stop_r - 1]
    ds, rep = traj.dataset, traj.replicate
    answer, ameta = generate_answer(env.llm, q.question, _evidence_text(env.store, rd.evidence_chunk_ids),
                                    salt=f"rep{rep}")
    calls = group_calls(traj, group, stop_r)
    for r, comp, meta in calls:
        extra = {}
        rr = traj.rounds[r - 1]
        if comp in ("retrieve", "jev_rank"):
            extra = dict(retrieved_chunk_ids=rr.retrieved_chunk_ids, doc_ids=rr.retrieved_doc_ids,
                         scores=rr.scores, covered_all=rr.covered_all, covered_any=rr.covered_any)
        elif comp == "glm_judge":
            extra = dict(judge_output=rr.llm_sufficient)
        elif comp == "jev_judge":
            extra = dict(judge_output=rr.jev_p)
        _emit(env, group, ds, q, rep, r, comp, meta, **extra)
    _emit(env, group, ds, q, rep, stop_r, "glm_answer", ameta.to_dict(), stop_reason=reason,
          covered_all=rd.covered_all, covered_any=rd.covered_any)
    comps = sum_components(calls + [(stop_r, "glm_answer", ameta.to_dict())])
    return _row(group, ds, q, rep, stop_r, reason, answer, rd.evidence_chunk_ids, rd.evidence_doc_ids,
                rd.covered_all, rd.covered_any, traj.rounds[0].retrieved_doc_ids, comps,
                first_ranking=traj.rounds[0].ranking_doc_ids, rounds_run=len(traj.rounds), evidence_tokens=rd.evidence_tokens)


def run_g1(env: Env, q: Question, dataset: str, rep: int, pool: list[str] | None = None) -> dict:
    """Single round: retrieve once with the original question, answer from top-k."""
    res = env.emb_retriever.search(q.question, env.cfg.search.top_k, pool=pool)
    ev = Evidence(env.store, env.enc, env.cfg.search.e_max)
    ev.add(res.hits)
    ca, cn = coverage(ev.doc_ids, q.gold_doc_ids)
    answer, ameta = generate_answer(env.llm, q.question, ev.text(), salt=f"rep{rep}")
    for comp, meta in res.calls:
        extra = {}
        if comp == "retrieve":
            extra = dict(retrieved_chunk_ids=[h.chunk_id for h in res.hits], doc_ids=[h.doc_id for h in res.hits],
                         scores=[h.score for h in res.hits], covered_all=ca, covered_any=cn)
        _emit(env, "G1", dataset, q, rep, 1, comp, meta.to_dict(), **extra)
    _emit(env, "G1", dataset, q, rep, 1, "glm_answer", ameta.to_dict(), stop_reason="answered",
          covered_all=ca, covered_any=cn)
    comps = sum_components([(1, c, m.to_dict()) for c, m in res.calls] + [(1, "glm_answer", ameta.to_dict())])
    return _row("G1", dataset, q, rep, 1, "answered", answer, ev.chunk_ids, ev.doc_ids, ca, cn,
                [h.doc_id for h in res.hits], comps,
                first_ranking=[h.doc_id for h in res.ranking], rounds_run=1, evidence_tokens=ev.n_tokens)


def run_g2(env: Env, q: Question, dataset: str, rep: int, pool: list[str] | None = None) -> dict:
    """ReAct with native tool calling. Full observation history is kept (no E_max); at most R_max
    searches, after which the model is forced to answer."""
    cfg, salt = env.cfg.search, f"rep{rep}"
    messages = [{"role": "system", "content": prompts.REACT_SYSTEM}, {"role": "user", "content": q.question}]
    seen_chunks: dict[str, str] = {}
    first_docs: list[str] | None = None
    first_ranking: list[str] | None = None
    n_search, step, answer, reason = 0, 0, "", "answered"
    calls: list[tuple[int, str, dict]] = []
    context_sizes: list[int] = []
    while True:
        step += 1
        force = n_search >= cfg.r_max
        r = env.llm.chat(messages, tools=[prompts.SEARCH_TOOL], tool_choice="none" if force else "auto", salt=salt)
        context_sizes.append(r.meta.input_tokens)
        calls.append((step, "glm_react_step", r.meta.to_dict()))
        if not r.tool_calls or force:
            answer, reason = r.content, ("r_max" if force else "answered")
            ca, cn = coverage(list(seen_chunks.values()), q.gold_doc_ids)
            _emit(env, "G2", dataset, q, rep, step, "glm_react_step", r.meta.to_dict(),
                  stop_reason=reason, covered_all=ca, covered_any=cn)
            break
        _emit(env, "G2", dataset, q, rep, step, "glm_react_step", r.meta.to_dict())
        messages.append({
            "role": "assistant", "content": r.content or None,
            "tool_calls": [{"id": t["id"], "type": "function",
                            "function": {"name": t["name"], "arguments": t["arguments"]}} for t in r.tool_calls],
        })
        for t in r.tool_calls:
            if n_search >= cfg.r_max:
                messages.append({"role": "tool", "tool_call_id": t["id"], "content": "Search budget exhausted."})
                continue
            try:
                query = str(json.loads(t["arguments"]).get("query", "")).strip() or q.question
            except (json.JSONDecodeError, AttributeError):
                query = q.question
            n_search += 1
            res = env.emb_retriever.search(query, cfg.top_k, pool=pool)
            chunks = [env.store.get(h.chunk_id) for h in res.hits]
            for c in chunks:
                seen_chunks.setdefault(c.chunk_id, c.doc_id)
            if first_docs is None:
                first_docs = [c.doc_id for c in chunks]
                first_ranking = [h.doc_id for h in res.ranking]
            ca, cn = coverage(list(seen_chunks.values()), q.gold_doc_ids)
            for comp, meta in res.calls:
                extra = {}
                if comp == "retrieve":
                    extra = dict(retrieved_chunk_ids=[c.chunk_id for c in chunks], doc_ids=[c.doc_id for c in chunks],
                                 scores=[h.score for h in res.hits], covered_all=ca, covered_any=cn)
                calls.append((step, comp, meta.to_dict()))
                _emit(env, "G2", dataset, q, rep, step, comp, meta.to_dict(), **extra)
            messages.append({"role": "tool", "tool_call_id": t["id"],
                             "content": "\n\n".join(serialize_chunk(c) for c in chunks) or "No results."})
    ca, cn = coverage(list(seen_chunks.values()), q.gold_doc_ids)
    return _row("G2", dataset, q, rep, step, reason, answer, list(seen_chunks), list(dict.fromkeys(seen_chunks.values())),
                ca, cn, first_docs or [], sum_components(calls),
                first_ranking=first_ranking, rounds_run=step, n_searches=n_search,
                context_tokens_per_step=context_sizes, evidence_tokens=None)


def _run_jev_live(env: Env, q: Question, dataset: str, rep: int, tau: float, pool, retriever, group: str) -> dict:
    """Multi-round flow with a Jev stop that runs live (stops at tau) because each round is expensive."""
    if retriever is None or env.jev_judge is None:
        raise RuntimeError(f"{group} needs a Jev client")
    traj = run_trajectory(
        q, dataset, rep, retriever, env.store, env.enc, env.cfg.search, env.qgen,
        llm_judge=None, jev_judge=env.jev_judge,
        stop_fn=lambda rd: "judge" if (rd.jev_p is not None and rd.jev_p >= tau) else None, pool=pool,
    )
    stop_r = traj.rounds[-1].r
    return finalize_from_trajectory(env, q, traj, group, stop_r, traj.live_stop or "r_max")


def run_g5(env: Env, q: Question, dataset: str, rep: int, tau: float, pool: list[str] | None = None) -> dict:
    """Jev retrieval (scores every candidate chunk) + Jev stop."""
    return _run_jev_live(env, q, dataset, rep, tau, pool, env.jev_retriever, "G5")


def run_g8(env: Env, q: Question, dataset: str, rep: int, tau: float, pool: list[str] | None = None) -> dict:
    """Embedding shortlist, Jev re-rank of the shortlist only, Jev stop (same tau as G4/G5)."""
    return _run_jev_live(env, q, dataset, rep, tau, pool, env.rerank_retriever, "G8")


def run_trajectory_groups(env: Env, q: Question, traj: Trajectory, tau: float | None, r_fixed: int | None,
                          groups=("G3", "G4", "G6", "G7")) -> list[dict]:
    rows = []
    for g in groups:
        stop_r, reason = stop_round(traj, g, tau=tau, r_fixed=r_fixed)
        rows.append(finalize_from_trajectory(env, q, traj, g, stop_r, reason))
    return rows
