"""Orchestration: prepare a dataset, run trajectories, tune on dev, run groups, grade."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .cache import ApiCache
from .chunking import get_encoding
from .clients import ApiError, EmbedClient, GLMClient, JevClient
from .config import Config
from .datasets import load_docs, load_questions
from .groups import Env, run_g1, run_g2, run_g5, run_g8, run_trajectory_groups
from .index import load_index
from .evidence import serialize_chunk
from .judges import grade_answer, judge_completeness, judge_faithfulness
from .pipeline import build_pool, build_store, embed_store, split_dev_test, subsample
from .recorder import Recorder
from .retrievers import EmbeddingRetriever, JevRetriever, RerankRetriever
from .store import ChunkStore
from .trajectory import Trajectory, run_trajectory
from .tune import tune
from .types import Question

ALL_GROUPS = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")


class Workspace:
    """work/<dataset>/: chunks.sqlite, emb/, split.json, frozen.json, api_cache.sqlite"""

    def __init__(self, cfg: Config, dataset: str):
        self.cfg, self.dataset = cfg, dataset
        self.dir = Path(cfg.work_dir) / dataset
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def chunks(self): return self.dir / "chunks.sqlite"
    @property
    def emb_dir(self): return self.dir / "emb"
    @property
    def split_path(self): return self.dir / "split.json"
    @property
    def frozen_path(self): return self.dir / "frozen.json"
    @property
    def cache_path(self): return self.dir / "api_cache.sqlite"
    @property
    def questions_path(self): return self.dir / "questions.jsonl"

    def eval_qids(self, split: str) -> list[str]:
        """Questions actually evaluated: all of dev, and a `test_fraction` subsample of test."""
        ids = self.split()[split]
        frac = self.cfg.test_fraction_by_dataset.get(self.dataset, self.cfg.test_fraction)
        return subsample(ids, frac, self.cfg.seed) if split == "test" else sorted(ids)

    def split(self) -> dict[str, list[str]]:
        return json.loads(self.split_path.read_text(encoding="utf-8"))

    def frozen(self) -> dict:
        return json.loads(self.frozen_path.read_text(encoding="utf-8")) if self.frozen_path.exists() else {}


def prepare(cfg: Config, dataset: str, data_dir: str | Path, embedder=None, log=print,
            embed: bool = True) -> Workspace:
    """Chunk -> embed -> index, and fix the dev/test split. Embedding cost is one-off and reported separately."""
    ws = Workspace(cfg, dataset)
    enc = get_encoding(cfg.chunk.encoding)
    store = ChunkStore(ws.chunks)
    n = build_store(load_docs(Path(data_dir) / "docs.jsonl"), store, enc, cfg.chunk)
    log(f"chunked {n} new docs -> {len(store)} chunks")
    questions = load_questions(Path(data_dir) / "questions.jsonl")
    ws.questions_path.write_text("\n".join(json.dumps(q.__dict__, ensure_ascii=False) for q in questions),
                                 encoding="utf-8")
    if not ws.split_path.exists():  # never re-split: dev/test must stay fixed
        ws.split_path.write_text(json.dumps(split_dev_test([q.qid for q in questions], cfg.seed, cfg.dev_fraction)))
    if embed:
        embedder = embedder or EmbedClient(cfg.embed)
        embed_store(store, embedder, cfg.embed, ws.emb_dir, log=log)
    store.close()
    return ws


def build_env(cfg: Config, ws: Workspace, run_id: str, fake: bool = False, with_jev: bool = True) -> Env:
    from . import fakes

    enc = get_encoding(cfg.chunk.encoding)
    store = ChunkStore(ws.chunks)
    emb_path = ws.emb_dir / "emb.npy"
    index = load_index(emb_path, cfg.index) if emb_path.exists() else None  # G5 needs no embeddings
    cache = ApiCache(ws.cache_path)
    run_dir = Path(cfg.runs_dir) / run_id
    rec = Recorder(run_dir / "records.jsonl")
    if fake:
        embedder, llm, jev = fakes.FakeEmbedder(), fakes.FakeLLM(), fakes.FakeJev()
    else:
        embedder = EmbedClient(cfg.embed, cache)
        llm = GLMClient(cfg.glm, cache)
        jev = JevClient(cfg.jev, cache) if with_jev else None
    return Env(cfg=cfg, store=store, enc=enc, llm=llm, emb_retriever=EmbeddingRetriever(store, index, embedder) if index is not None else None,
               recorder=rec, run_id=run_id, jev=jev,
               jev_retriever=JevRetriever(store, jev, cfg.jev.workers) if jev else None,
               rerank_retriever=(RerankRetriever(store, index, embedder, jev, cfg.search.rerank_shortlist, cfg.jev.workers)
                                 if jev and index is not None else None))


def _questions(ws: Workspace, split: str, limit: int | None) -> list[Question]:
    keep = set(ws.eval_qids(split))
    qs = [q for q in load_questions(ws.questions_path) if q.qid in keep]
    return qs[:limit] if limit else qs


def _traj_path(cfg: Config, run_id: str, dataset: str, qid: str, rep: int) -> Path:
    safe = qid.replace("/", "_")
    return Path(cfg.runs_dir) / run_id / "trajectories" / dataset / f"{safe}_r{rep}.json"


class Failures:
    """Per-(group, question) failures that are permanent for that question (HTTP 400, e.g. the provider's
    content filter). They are recorded, not fatal; reports drop such questions from *every* group so
    paired comparisons stay paired. Auth / credit / rate errors still abort the run."""

    def __init__(self, path: Path):
        self.path, self._lock = path, threading.Lock()

    def guard(self, group: str, q: Question, rep: int, fn):
        try:
            return fn()
        except ApiError as e:
            if e.status != 400:
                raise
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"group": group, "qid": q.qid, "replicate": rep, "status": e.status,
                                    "error": str(e)[:300]}, ensure_ascii=False) + "\n")
            return None


def load_failures(path: str | Path) -> list[dict]:
    p = Path(path)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def _parallel(fn, items, workers: int):
    if workers <= 1:
        return [fn(i) for i in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def run_trajectories(env: Env, ws: Workspace, split: str, repeats: int | None = None, limit: int | None = None,
                     workers: int = 4, pool_size: int | None = None, log=print) -> list[Trajectory]:
    cfg = env.cfg
    qs = _questions(ws, split, limit)
    units = [(q, r) for q in qs for r in range(repeats or cfg.repeats)]

    def one(u):
        q, rep = u
        path = _traj_path(cfg, env.run_id, ws.dataset, q.qid, rep)
        if path.exists():
            return Trajectory.from_dict(json.loads(path.read_text(encoding="utf-8")))
        pool = build_pool(q, env.store, pool_size, cfg.seed) if pool_size else None
        t = failures.guard("TRAJ", q, rep, lambda: run_trajectory(
            q, ws.dataset, rep, env.emb_retriever, env.store, env.enc, cfg.search, env.qgen,
            env.llm_judge, env.jev_judge, pool=pool))
        if t is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(t.to_dict(), ensure_ascii=False), encoding="utf-8")
        return t

    failures = Failures(Path(cfg.runs_dir) / env.run_id / "failures.jsonl")
    (Path(cfg.runs_dir) / env.run_id).mkdir(parents=True, exist_ok=True)
    trajs = [t for t in _parallel(one, units, workers) if t is not None]
    log(f"{len(trajs)} trajectories ({split}); {len(units) - len(trajs)} failed (see failures.jsonl)")
    return trajs


def load_trajectories(cfg: Config, run_id: str, dataset: str) -> list[Trajectory]:
    d = Path(cfg.runs_dir) / run_id / "trajectories" / dataset
    return [Trajectory.from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in sorted(d.glob("*.json"))]


def run_tune(cfg: Config, ws: Workspace, run_id: str) -> dict:
    out = tune(load_trajectories(cfg, run_id, ws.dataset))
    ws.frozen_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def run_groups(env: Env, ws: Workspace, split: str, groups=ALL_GROUPS, repeats: int | None = None,
               limit: int | None = None, workers: int = 4, pool_size: int | None = None, log=print) -> list[dict]:
    cfg = env.cfg
    fz = ws.frozen()
    tau, r_fixed = fz.get("tau"), fz.get("r_fixed")
    need = []
    if {"G4", "G5", "G8"} & set(groups) and tau is None:
        need.append("tau")
    if "G6" in groups and r_fixed is None:
        need.append("r_fixed")
    if need:
        raise RuntimeError(f"missing frozen {need}: run `tune` on dev first")
    qs = _questions(ws, split, limit)
    n_rep = repeats or cfg.repeats
    results_path = Path(cfg.runs_dir) / env.run_id / "results.jsonl"
    done: set[tuple] = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["group"], r["qid"], r["replicate"]))
    lock = threading.Lock()
    traj_groups = [g for g in groups if g in ("G3", "G4", "G6", "G7")]

    def emit(rows):
        with lock, open(results_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    failures = Failures(Path(cfg.runs_dir) / env.run_id / "failures.jsonl")

    def one(u):
        q, rep = u
        pool = build_pool(q, env.store, pool_size, cfg.seed) if pool_size else None
        rows = []

        def add(group, fn):
            r = failures.guard(group, q, rep, fn)
            if r is not None:
                rows.extend(r if isinstance(r, list) else [r])

        if "G1" in groups and ("G1", q.qid, rep) not in done:
            add("G1", lambda: run_g1(env, q, ws.dataset, rep, pool))
        if "G2" in groups and ("G2", q.qid, rep) not in done:
            add("G2", lambda: run_g2(env, q, ws.dataset, rep, pool))
        todo = [g for g in traj_groups if (g, q.qid, rep) not in done]
        if todo:
            path = _traj_path(cfg, env.run_id, ws.dataset, q.qid, rep)
            if path.exists():
                t = Trajectory.from_dict(json.loads(path.read_text(encoding="utf-8")))
            else:
                t = failures.guard("TRAJ", q, rep, lambda: run_trajectory(
                    q, ws.dataset, rep, env.emb_retriever, env.store, env.enc, cfg.search, env.qgen,
                    env.llm_judge, env.jev_judge, pool=pool))
                if t is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(t.to_dict(), ensure_ascii=False), encoding="utf-8")
            if t is not None:
                for g in todo:  # each group answers separately, so one blocked answer does not hide the others
                    add(g, lambda g=g: run_trajectory_groups(env, q, t, tau, r_fixed, groups=[g]))
        if "G5" in groups and ("G5", q.qid, rep) not in done:
            add("G5", lambda: run_g5(env, q, ws.dataset, rep, tau, pool))
        if "G8" in groups and ("G8", q.qid, rep) not in done:
            add("G8", lambda: run_g8(env, q, ws.dataset, rep, tau, pool))
        emit(rows)
        return rows

    out = [r for rows in _parallel(one, [(q, r) for q in qs for r in range(n_rep)], workers) for r in rows]
    nf = len(load_failures(Path(cfg.runs_dir) / env.run_id / "failures.jsonl"))
    log(f"{len(out)} new result rows -> {results_path}" + (f"; {nf} failures recorded" if nf else ""))
    return out


def grade_row(judge, q: Question, row: dict, store: ChunkStore | None) -> dict:
    """Fill the quality fields of one result row, with one fixed judge for every group.

    Datasets with a human/gold answer (BrowseComp-Plus): `correct`.
    RAGBench (no gold answer; its `response` is LLM-generated): claim-level `faithfulness` against the
    evidence the group used, and `completeness` against the annotated relevant sentences."""
    key = q.meta.get("relevant_sentences")
    if key is None:
        if row.get("correct") is None:
            row["correct"], _ = grade_answer(judge, q.question, q.answer, row["answer"])
        return row
    if row.get("faithfulness") is None and row.get("n_claims") is None:
        ev_text = "\n\n".join(serialize_chunk(store.get(c)) for c in row["evidence_chunk_ids"])
        f = judge_faithfulness(judge, ev_text, row["answer"])
        row.update(faithfulness=f["faithfulness"], fully_faithful=f["fully_faithful"], n_claims=f["n_claims"],
                   faithfulness_parse_ok=f["parse_ok"])
    if row.get("completeness") is None and row.get("n_key") is None:
        c = judge_completeness(judge, q.question, row["answer"], key)
        row.update(completeness=c["completeness"], n_key=c["n_key"], completeness_parse_ok=c["parse_ok"])
    return row


def grade(cfg: Config, ws: Workspace, run_id: str, judge=None, workers: int = 4, log=print,
          replicates: int | None = None) -> int:
    """Grade result rows (see grade_row). Safe to re-run: graded fields are not recomputed.
    `replicates=N` grades only repeats 0..N-1 (budget: judge cost is the largest line item); the other
    repeats still feed cost, latency, round and retrieval statistics."""
    qs = {q.qid: q for q in load_questions(ws.questions_path)}
    cache = ApiCache(ws.cache_path)
    judge = judge or GLMClient(cfg.judge, cache)
    path = Path(cfg.runs_dir) / run_id / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    store = ChunkStore(ws.chunks) if any(q.meta.get("relevant_sentences") is not None for q in qs.values()) else None
    keep = set(ws.eval_qids("test"))  # rows of questions outside the test subsample are not graded (no judge cost)
    todo = lambda r: r["qid"] in keep and (replicates is None or r["replicate"] < replicates)
    rows = _parallel(lambda r: grade_row(judge, qs[r["qid"]], r, store) if todo(r) else r, rows, workers)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    log(f"graded {len(rows)} rows")
    return len(rows)
