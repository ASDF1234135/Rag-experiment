"""Aggregate result rows into the spec's metric tables (section 7). Costs are computed offline from tokens."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import metrics
from .config import PricingCfg
from .types import Question

# (group A, group B, what it isolates) — spec section 4 "main contrasts"
CONTRASTS = [
    ("G3", "G4", "judge: LLM vs Jev"),
    ("G4", "G5", "retrieval base: embedding vs Jev"),
    ("G2", "G3", "ReAct vs separated judge/action"),
    ("G1", "G2", "single round vs ReAct"),
    ("G1", "G3", "single round vs multi-round"),
    ("G4", "G8", "embedding vs embedding + Jev re-rank"),
    ("G8", "G5", "Jev re-rank of top-K vs Jev on every chunk"),
    ("G4", "G6", "Jev stop vs fixed rounds"),
    ("G4", "G7", "Jev stop vs oracle stop"),
]


def row_cost(row: dict, p: PricingCfg) -> float | None:
    """USD cost of one result row; None if a needed price is still TBD."""
    total = 0.0
    for comp, c in row["components"].items():
        if comp.startswith("jev"):
            total += (c["input_tokens"] * p.jev_input + c["output_tokens"] * p.jev_output) / 1e6
        elif comp == "embed_query":
            total += c["input_tokens"] * p.embed_input / 1e6
        elif comp.startswith("glm"):
            if p.glm_input is None or p.glm_output is None:
                return None
            total += (c["input_tokens"] * p.glm_input + c["output_tokens"] * p.glm_output) / 1e6  # completion_tokens already includes reasoning
    return total


def load_results(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _graded_cost(rows: list[dict], pricing: PricingCfg) -> float:
    """Cost of exactly the rows that were graded, so cost-per-good-answer pairs like with like."""
    return float(sum(row_cost(r, pricing) or 0.0 for r in rows))


def summarize(rows: list[dict], questions: dict[str, Question], pricing: PricingCfg, r_max: int) -> list[dict]:
    """One line per group. Quality columns depend on the dataset: `accuracy` for datasets with a gold
    answer (BrowseComp-Plus); `faithfulness`, `fully_faithful`, `completeness` for RAGBench."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    g7: dict[tuple, int] = {(r["qid"], r["replicate"]): r["stop_round"] for r in by_group.get("G7", [])}
    out = []
    for g in sorted(by_group):
        rs = by_group[g]
        graded = [r for r in rs if r.get("correct") is not None]
        rb = [r for r in rs if r.get("n_claims") is not None]  # RAGBench-graded rows
        costs = [row_cost(r, pricing) for r in rs]
        have_cost = all(c is not None for c in costs)
        lat = [r["latency_ms"] for r in rs]
        rel = lambda r: questions[r["qid"]].gold_doc_ids
        first = [r["first_round_doc_ids"] for r in rs]
        rank = [r["first_round_ranking_doc_ids"] for r in rs]
        final_recall = [metrics.recall_at_k(r["evidence_doc_ids"], rel(r)) for r in rs]
        comp_tokens: dict[str, float] = defaultdict(float)
        for r in rs:
            for comp, c in r["components"].items():
                comp_tokens[f"{comp}.in"] += c["input_tokens"] / len(rs)
                comp_tokens[f"{comp}.out"] += c["output_tokens"] / len(rs)  # includes reasoning tokens
        n_correct = sum(r["correct"] for r in graded)
        # premature stop = stopped on the judge's say-so with the gold documents not all in evidence and
        # (where a gold answer exists) a wrong answer. RAGBench has no gold answer, so only the first two.
        pool = graded or rb
        premature = [r for r in pool if r["stop_reason"] == "judge" and not r["covered_all"]
                     and (r.get("correct") is None or not r["correct"])]
        faithful = [r for r in rb if r.get("fully_faithful")]
        d = {
            "group": g, "n": len(rs),
            "accuracy": n_correct / len(graded) if graded else None,
            "faithfulness": _mean([r.get("faithfulness") for r in rb]),
            "fully_faithful": _mean([r["fully_faithful"] for r in rb if r.get("fully_faithful") is not None]),
            "abstain_rate": _mean([r["n_claims"] == 0 for r in rb]),
            "completeness": _mean([r.get("completeness") for r in rb]),
            "mean_rounds": float(np.mean([r["stop_round"] for r in rs])),
            "r_max_truncated": float(np.mean([r["stop_reason"] == "r_max" for r in rs])),
            "premature_stop": len(premature) / len(pool) if pool else None,
            "excess_rounds_vs_G7": (float(np.mean([r["stop_round"] - g7[(r["qid"], r["replicate"])]
                                                  for r in rs if (r["qid"], r["replicate"]) in g7]))
                                    if g7 and g in ("G3", "G4", "G5", "G6") else None),
            "covered_all_final": float(np.mean([r["covered_all"] for r in rs])),
            "recall_first@5": float(np.nanmean([metrics.recall_at_k(f, rel(r), 5) for f, r in zip(first, rs)])),
            "recall_final": float(np.nanmean(final_recall)),
            "mrr_first": float(np.mean([metrics.mrr(k, rel(r)) for k, r in zip(rank, rs)])),
            "ndcg10_first": float(np.nanmean([metrics.ndcg_at_k(k, rel(r), 10) for k, r in zip(rank, rs)])),
            "cost_usd_per_q": float(np.mean(costs)) if have_cost else None,
            "cost_usd_per_correct": (_graded_cost(graded, pricing) / n_correct if have_cost and n_correct else None),
            "cost_usd_per_faithful": (_graded_cost(rb, pricing) / len(faithful) if have_cost and faithful else None),
            "latency_p50_ms": metrics.percentile(lat, 50), "latency_p95_ms": metrics.percentile(lat, 95),
            "retries_per_q": float(np.mean([sum(c["retries"] for c in r["components"].values()) for r in rs])),
            "tokens": dict(comp_tokens),
        }
        out.append(d)
    return out


# quality metrics compared between groups: name -> is it binary (McNemar applies)
QUALITY_METRICS = {"correct": True, "fully_faithful": True, "faithfulness": False, "completeness": False}


def contrasts(rows: list[dict], pricing: PricingCfg) -> list[dict]:
    """Paired comparisons on per-question means over replicates. Binary metrics (`correct`,
    `fully_faithful`) get an exact McNemar test on the majority vote plus a bootstrap CI of the
    difference; continuous ones (`faithfulness`, `completeness`) and cost get the bootstrap CI."""
    per: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        for m in QUALITY_METRICS:
            if r.get(m) is not None:
                per[r["group"]][r["qid"]][m].append(float(r[m]))
        c = row_cost(r, pricing)
        if c is not None:
            per[r["group"]][r["qid"]]["cost"].append(c)
    out = []
    for a, b, what in CONTRASTS:
        if a not in per or b not in per:
            continue
        qids = sorted(set(per[a]) & set(per[b]))
        if not qids:
            continue
        res: dict = {"a": a, "b": b, "what": what, "n": len(qids), "metrics": {}}
        for m, binary in QUALITY_METRICS.items():
            qs = [q for q in qids if per[a][q][m] and per[b][q][m]]
            if not qs:
                continue
            va = [float(np.mean(per[a][q][m])) for q in qs]
            vb = [float(np.mean(per[b][q][m])) for q in qs]
            entry = {"n": len(qs), "diff": metrics.paired_bootstrap_ci(va, vb)}
            if binary:
                entry["mcnemar"] = metrics.mcnemar_exact([x >= 0.5 for x in va], [x >= 0.5 for x in vb])
            res["metrics"][m] = entry
        if all(per[g][q]["cost"] for g in (a, b) for q in qids):
            res["cost_diff_usd"] = metrics.paired_bootstrap_ci(
                [np.mean(per[a][q]["cost"]) for q in qids], [np.mean(per[b][q]["cost"]) for q in qids])
        if res["metrics"] or "cost_diff_usd" in res:
            out.append(res)
    return out


def to_markdown(summary: list[dict]) -> str:
    cols = ["group", "n", "accuracy", "faithfulness", "fully_faithful", "abstain_rate", "completeness", "mean_rounds",
            "r_max_truncated", "premature_stop", "covered_all_final", "recall_first@5", "recall_final",
            "mrr_first", "ndcg10_first", "cost_usd_per_q", "cost_usd_per_correct", "cost_usd_per_faithful",
            "latency_p50_ms", "latency_p95_ms"]
    cols = [c for c in cols if any(s.get(c) is not None for s in summary) or c in ("group", "n")]
    fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(s[c]) for c in cols) + " |" for s in summary]
    return "\n".join(lines)
