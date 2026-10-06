"""Offline simulation: how much of G5's (Jev scores every pool chunk) first-round retrieval quality survives if
embedding first shortlists K chunks and Jev only re-ranks those?  Uses cached Jev scores only (no API calls).

python -m scripts.jev_rerank_sim [dataset ...]
"""
import json
import sys

import numpy as np

from ragexp import prompts
from ragexp.cache import ApiCache, request_hash
from ragexp.clients import EmbedClient
from ragexp.config import Config
from ragexp.datasets import load_questions
from ragexp.envfile import load_dotenv
from ragexp.index import load_index
from ragexp.pipeline import build_pool
from ragexp.runner import Workspace
from ragexp.store import ChunkStore

POOL = {"bcp100": 1500}
KS = [5, 10, 20, 50, 100, 200, None]  # None = Jev scores the whole pool (= G5's retrieval)


def metrics(ranked_docs, gold):
    """ranked_docs: doc id per chunk, best first. Doc-level recall@5 chunks, any-gold hit@5, MRR over chunks."""
    top = ranked_docs[:5]
    g = set(gold)
    rec = len(g & set(top)) / len(g)
    hit = float(bool(g & set(top)))
    rr = next((1 / (i + 1) for i, d in enumerate(ranked_docs) if d in g), 0.0)
    return rec, hit, rr


def run(ds, cfg):
    ws = Workspace(cfg, ds)
    cache = ApiCache(ws.cache_path)
    store = ChunkStore(ws.chunks)
    index = load_index(ws.emb_dir / "emb.npy", cfg.index)
    emb = EmbedClient(cfg.embed, cache)
    qs = {q.qid: q for q in load_questions(ws.questions_path)}
    keep = [qs[i] for i in ws.eval_qids("test")]
    size = POOL.get(ds, 300)
    acc = {k: [] for k in KS}
    tok = []
    used = 0
    for q in keep:
        pool = build_pool(q, store, size, cfg.seed)
        chunks = [store.get(c) for c in pool]
        scores, ok = [], True
        for c in chunks:
            body = {"model": cfg.jev.model, "state": {"query": q.question, "passage": {"id": c.chunk_id, "title": c.title, "text": c.text}},
                    "questions": prompts.JEV_RANK}
            hit = cache.get(request_hash({"p": "typesafe", **body}))
            if hit is None:
                ok = False
                break
            a = hit["answers"]
            scores.append((float(a["usefulness"]["score"]), float(a["is_relevant"]["noul"])))
            tok.append(hit["meta"]["input_tokens"])
        if not ok:
            continue  # question whose pool was never fully scored by Jev (e.g. blocked)
        used += 1
        vec, _ = emb.embed([q.question])
        rows = store.rows_for_ids(pool)
        order, _ = index.search(vec[0], len(rows), rows=rows)
        by_row = {r: i for i, r in enumerate(rows)}
        emb_rank = [by_row[r] for r in order]  # indices into `chunks`, best embedding first
        for k in KS:
            short = emb_rank if k is None else emb_rank[:k]
            ranked = sorted(short, key=lambda i: (-scores[i][0], -scores[i][1], emb_rank.index(i)))
            acc[k].append(metrics([chunks[i].doc_id for i in ranked], q.gold_doc_ids))
    mean_tok = float(np.mean(tok)) if tok else 0.0
    out = {"questions": used, "pool": size, "tokens_per_request": mean_tok, "rows": {}}
    for k in KS:
        m = np.mean(acc[k], axis=0) if acc[k] else [float("nan")] * 3
        n_calls = size if k is None else k
        out["rows"]["full pool" if k is None else f"emb top-{k}" + (" (no Jev)" if k == 5 else " -> Jev")] = {
            "recall@5": float(m[0]), "hit@5": float(m[1]), "mrr": float(m[2]),
            "jev_calls": 0 if k == 5 else n_calls,
            "usd_per_query": 0.0 if k == 5 else n_calls * mean_tok * cfg.pricing.jev_input / 1e6}
    return out


def main(datasets):
    load_dotenv(".env")
    cfg = Config.load("configs/experiment.yaml")
    res = {ds: run(ds, cfg) for ds in datasets}
    for ds, r in res.items():
        print(f"\n== {ds}: {r['questions']} questions, pool {r['pool']} chunks, {r['tokens_per_request']:.0f} tokens/Jev request")
        print(f"{'first-round ranking':22s} {'recall@5':>9s} {'hit@5':>7s} {'MRR':>6s} {'Jev calls':>10s} {'$/query':>9s}")
        for name, v in r["rows"].items():
            print(f"{name:22s} {v['recall@5']:9.3f} {v['hit@5']:7.3f} {v['mrr']:6.3f} {v['jev_calls']:10d} {v['usd_per_query']:9.4f}")
    with open("work/jev_rerank_sim.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["bcp100", "covidqa", "hotpotqa", "techqa"])
