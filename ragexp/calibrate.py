"""Calibrate the faithfulness / completeness judge against RAGBench's own annotations (spec section 7).

RAGBench ships, for each (documents, response) pair, `adherence_score` (response fully supported by the
documents) and `completeness_score` (share of relevant sentences the response uses). We run our judge
prompts on exactly those pairs and report how well they reproduce the annotation. The judge model is
the one in `config.judge`, so this is also the check that a candidate judge is fit for purpose.
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import metrics
from .datasets import relevant_sentences
from .judges import judge_completeness, judge_faithfulness


def load_rows(files: list[str | Path], n: int, seed: int) -> list[dict]:
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    rows.sort(key=lambda r: str(r["id"]))
    random.Random(seed).shuffle(rows)
    return rows[:n]


def _one(judge, row: dict) -> dict:
    evidence = "\n\n".join(f"[doc{i}] {d}" for i, d in enumerate(row["documents"]))
    keys = relevant_sentences(row)
    f = judge_faithfulness(judge, evidence, row["response"])
    c = judge_completeness(judge, row["question"], row["response"], keys)
    return {"id": row["id"], "ann_adherence": bool(row["adherence_score"]), "ann_completeness": row["completeness_score"],
            "has_keys": bool(keys), "faithfulness": f["faithfulness"], "fully_faithful": f["fully_faithful"],
            "n_claims": f["n_claims"], "completeness": c["completeness"],
            "parse_ok": f["parse_ok"] and c["parse_ok"]}


def agreement(results: list[dict]) -> dict:
    """Judge-vs-annotation agreement. Pure function of `results` (testable without an API)."""
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

    out: dict = {"n": len(results), "parse_failure_rate": float(np.mean([not r["parse_ok"] for r in results])) if results else None}
    a = [r for r in results if r["fully_faithful"] is not None]
    out["adherence"] = {"n": len(a), "no_claim_rate": float(np.mean([r["n_claims"] == 0 for r in results])) if results else None}
    if a:
        y = [r["ann_adherence"] for r in a]
        p = [bool(r["fully_faithful"]) for r in a]
        out["adherence"].update(
            accuracy=float(np.mean([x == z for x, z in zip(y, p)])),
            balanced_accuracy=metrics.balanced_accuracy(p, y),
            kappa=float(cohen_kappa_score(y, p)) if len(set(y)) > 1 or len(set(p)) > 1 else None,
            precision=float(precision_score(y, p, zero_division=0)), recall=float(recall_score(y, p, zero_division=0)),
            f1=float(f1_score(y, p, zero_division=0)),
            auc_fraction_supported=metrics.auc([r["faithfulness"] for r in a], y),
            annotated_positive_rate=float(np.mean(y)))
    c = [r for r in results if r["completeness"] is not None and r["has_keys"] and r["ann_completeness"] is not None]
    out["completeness"] = {"n": len(c)}
    if len(c) > 2:
        x = [float(r["ann_completeness"]) for r in c]
        z = [float(r["completeness"]) for r in c]
        out["completeness"].update(
            mae=float(np.mean(np.abs(np.array(x) - np.array(z)))),
            pearson=float(pearsonr(x, z)[0]) if np.std(x) > 0 and np.std(z) > 0 else None,
            spearman=float(spearmanr(x, z)[0]) if np.std(x) > 0 and np.std(z) > 0 else None,
            annotated_mean=float(np.mean(x)), judged_mean=float(np.mean(z)))
    return out


def calibrate(judge, files: list[str | Path], n: int = 100, seed: int = 0, workers: int = 4) -> dict:
    rows = load_rows(files, n, seed)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda r: _one(judge, r), rows))
    return {"agreement": agreement(results), "results": results}
