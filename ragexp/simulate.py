"""Offline stop-rule simulation over a full trajectory (spec section 10)."""

from __future__ import annotations

from .trajectory import Trajectory

TRAJECTORY_GROUPS = ("G3", "G4", "G6", "G7")
JUDGE_COMPONENT = {"G3": "glm_judge", "G4": "jev_judge", "G5": "jev_judge", "G8": "jev_judge"}


def stop_round(traj: Trajectory, group: str, tau: float | None = None, r_fixed: int | None = None) -> tuple[int, str]:
    """Round at which `group` stops and why. Falls through to the last round with reason r_max."""
    last = traj.rounds[-1].r
    if group == "G6":
        if r_fixed is None:
            raise ValueError("G6 needs r_fixed")
        return min(r_fixed, last), "fixed"
    for rd in traj.rounds:
        if group == "G3" and rd.llm_sufficient:
            return rd.r, "judge"
        if group == "G4":
            if tau is None:
                raise ValueError("G4 needs tau")
            if rd.jev_p is not None and rd.jev_p >= tau:
                return rd.r, "judge"
        if group == "G7" and rd.covered_all:
            return rd.r, "judge"
    if group not in ("G3", "G4", "G7", "G5"):
        raise ValueError(f"unknown trajectory group {group}")
    return last, "r_max"


def group_calls(traj: Trajectory, group: str, stop_r: int) -> list[tuple[int, str, dict]]:
    """(round, component, meta) actually incurred by `group` when stopping at `stop_r`.
    A group never pays for rounds, judges or query generations past its own stop."""
    out: list[tuple[int, str, dict]] = []
    for rd in traj.rounds:
        if rd.r > stop_r:
            break
        for comp, meta in rd.retrieval_calls:
            out.append((rd.r, comp, meta))
        jc = JUDGE_COMPONENT.get(group)
        if jc and jc in rd.judge_calls:
            out.append((rd.r, jc, rd.judge_calls[jc]))
        if rd.r < stop_r and rd.qgen_call is not None:
            out.append((rd.r, "glm_query_gen", rd.qgen_call))
    return out


def sum_components(calls: list[tuple[int, str, dict]]) -> dict[str, dict]:
    agg: dict[str, dict] = {}
    for _, comp, m in calls:
        a = agg.setdefault(comp, {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
                                  "latency_ms": 0.0, "retries": 0, "n_calls": 0})
        a["input_tokens"] += m.get("input_tokens", 0)
        a["output_tokens"] += m.get("output_tokens", 0)
        a["reasoning_tokens"] += m.get("reasoning_tokens", 0)
        a["latency_ms"] += m.get("latency_ms", 0.0)
        a["retries"] += m.get("retries", 0)
        a["n_calls"] += m.get("n_calls", 1)
    return agg
