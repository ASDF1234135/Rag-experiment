"""Dev-only hyper-parameter freezing (spec section 6): tau for the Jev stop rule, R_fixed for G6."""

from __future__ import annotations

import statistics

from . import metrics
from .simulate import stop_round
from .trajectory import Trajectory


def tune(trajs: list[Trajectory]) -> dict:
    """Pools every (question, round) of the dev trajectories. Label = covered_all after that round."""
    jev = [(r.jev_p, r.covered_all) for t in trajs for r in t.rounds if r.jev_p is not None]
    llm = [(r.llm_sufficient, r.covered_all) for t in trajs for r in t.rounds if r.llm_sufficient is not None]
    out: dict = {"n_questions": len(trajs), "n_round_samples": len(jev)}
    if jev:
        p, y = [a for a, _ in jev], [b for _, b in jev]
        tau, ba = metrics.best_threshold(p, y)
        out["tau"] = tau
        out["jev"] = {"balanced_accuracy_at_tau": ba, "auc": metrics.auc(p, y), "ece": metrics.ece(p, y),
                      "positive_rate": sum(y) / len(y)}
        stops = [stop_round(t, "G4", tau=tau)[0] for t in trajs]
        out["r_fixed"] = max(1, round(statistics.median(stops)))
        out["g4_stop_round_median"] = statistics.median(stops)
    if llm:
        pred, y = [a for a, _ in llm], [b for _, b in llm]
        tp = sum(a and b for a, b in llm); fp = sum(a and not b for a, b in llm)
        fn = sum((not a) and b for a, b in llm); tn = sum((not a) and (not b) for a, b in llm)
        out["llm"] = {"balanced_accuracy": metrics.balanced_accuracy(pred, y),
                      "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                      "auc_binary": metrics.auc([float(a) for a in pred], y)}
    # R_max guidance: share of questions already covered_all after each round
    r_max = max((len(t.rounds) for t in trajs), default=0)
    out["covered_all_by_round"] = [
        sum(t.rounds[min(r, len(t.rounds)) - 1].covered_all for t in trajs) / max(1, len(trajs))
        for r in range(1, r_max + 1)
    ]
    return out
