from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom


def dedupe_docs(doc_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(doc_ids))


def recall_at_k(ranked_doc_ids: list[str], relevant: list[str], k: int | None = None) -> float:
    rel = set(relevant)
    if not rel:
        return float("nan")
    docs = dedupe_docs(ranked_doc_ids)
    if k is not None:
        docs = docs[:k]
    return len(rel & set(docs)) / len(rel)


def mrr(ranked_doc_ids: list[str], relevant: list[str]) -> float:
    rel = set(relevant)
    for i, d in enumerate(dedupe_docs(ranked_doc_ids), 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_doc_ids: list[str], relevant: list[str], k: int = 10) -> float:
    rel = set(relevant)
    if not rel:
        return float("nan")
    docs = dedupe_docs(ranked_doc_ids)[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(docs) if d in rel)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg


def auc(scores: list[float], labels: list[bool]) -> float:
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def ece(probs: list[float], labels: list[bool], bins: int = 10) -> float:
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    if len(p) == 0:
        return float("nan")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def balanced_accuracy(pred: list[bool], labels: list[bool]) -> float:
    pred, y = np.asarray(pred, bool), np.asarray(labels, bool)
    pos, neg = y.sum(), (~y).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float(((pred & y).sum() / pos + (~pred & ~y).sum() / neg) / 2)


def best_threshold(probs: list[float], labels: list[bool]) -> tuple[float, float]:
    """tau maximising balanced accuracy of `p >= tau` vs the covered_all label (spec section 6)."""
    p = np.asarray(probs, float)
    cands = np.unique(np.concatenate([p, [0.0, 1.0]]))
    best, best_ba = 0.5, -1.0
    for t in cands:
        ba = balanced_accuracy(list(p >= t), labels)
        if not math.isnan(ba) and ba > best_ba + 1e-12:
            best, best_ba = float(t), ba
    return best, best_ba


def mcnemar_exact(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired per-question correctness."""
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    p = 1.0 if n == 0 else min(1.0, 2 * float(binom.cdf(min(n10, n01), n, 0.5)))
    return {"a_only": n10, "b_only": n01, "p_value": p}


def paired_bootstrap_ci(a: list[float], b: list[float], n_boot: int = 10_000, alpha: float = 0.05,
                        seed: int = 0) -> dict:
    """CI of mean(a - b) over questions (paired)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {"mean_diff": float(d.mean()), "lo": float(np.quantile(means, alpha / 2)),
            "hi": float(np.quantile(means, 1 - alpha / 2))}


def percentile(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")
