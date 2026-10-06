"""Collect everything the RAGBench report page needs into one JSON file.

python -m scripts.build_report_data [out.json]

Quality/retrieval numbers and paired contrasts come from runs/<run>/summary.json (already restricted to the
evaluated test questions, with failed questions removed from every group). Cost is split by vendor from the
result rows; *actual spend* is summed from every API call ever cached, plus embedding shards.
"""

import collections
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

from ragexp.config import Config
from ragexp.runner import Workspace, load_failures

DATASETS = {"covidqa": "CovidQA (biomedical)", "hotpotqa": "HotpotQA (multi-hop)", "techqa": "TechQA (technical docs)",
            "bcp100": "BrowseComp-Plus (reduced)"}
SETTINGS = {"full": ("final_{ds}_full", "Full corpus"), "pool": ("final_{ds}_pool{size}", "Candidate pool ({size} chunks)")}
POOL_SIZE = {"bcp100": 1500}  # chunks per candidate pool; RAGBench default below
RUN_SUFFIX = os.environ.get("RAGEXP_REPORT_RUN_PREFIX", "final")  # e.g. "_preview" to build from copies of in-progress runs
GROUP_NAMES = {
    "G1": "Single-round RAG", "G2": "ReAct multi-round", "G3": "Multi-round, LLM stop judge", "G4": "Multi-round, Jev stop judge",
    "G5": "Jev stop judge + Jev retrieval", "G6": "Fixed rounds", "G7": "Oracle stop", "G8": "Embedding top 200 + Jev re-rank",
}


def vendor_cost(row: dict, p) -> dict:
    """USD by vendor for one result row. completion tokens already include reasoning tokens."""
    out = collections.Counter()
    for comp, c in row["components"].items():
        if comp == "jev_rank":
            out["jev_rank"] += c["input_tokens"] * p.jev_input / 1e6
        elif comp == "jev_judge":
            out["jev_judge"] += c["input_tokens"] * p.jev_input / 1e6
        elif comp == "embed_query":
            out["embed"] += c["input_tokens"] * p.embed_input / 1e6
        elif comp.startswith("glm"):
            out["glm"] += (c["input_tokens"] * p.glm_input + c["output_tokens"] * p.glm_output) / 1e6
    return out


def spend(cfg) -> dict:
    """Actual spend: every cached live call (dev, pilots, test, grading, calibration) + embedding shards."""
    p = cfg.pricing
    judge_in, judge_out = 0.435, 0.87  # mimo-v2.6-pro list price, USD per 1M tokens (public price page)
    rows = {}
    for ds in DATASETS:
        t = collections.Counter()
        con = sqlite3.connect(f"work/{ds}/api_cache.sqlite")
        for (v,) in con.execute("select value from cache"):
            m = json.loads(v).get("meta")
            if not m:
                continue
            prov = m.get("provider")
            if prov == "z.ai":
                t["glm_in"] += m["input_tokens"]; t["glm_out"] += m["output_tokens"]; t["glm_calls"] += 1
            elif prov == "typesafe":
                t["jev_in"] += m["input_tokens"]; t["jev_calls"] += 1
            elif prov == "xiaomi":
                t["judge_in"] += m["input_tokens"]; t["judge_out"] += m["output_tokens"]; t["judge_calls"] += 1
            elif prov == "openai":
                t["embq_in"] += m["input_tokens"]
        emb = sum(int(f.read_text()) for f in Path(f"work/{ds}/emb").glob("shard_*.tokens"))
        rows[ds] = {
            "GLM (z.ai)": t["glm_in"] * p.glm_input / 1e6 + t["glm_out"] * p.glm_output / 1e6,
            "Jev": t["jev_in"] * p.jev_input / 1e6,
            "Judge (mimo)": t["judge_in"] * judge_in / 1e6 + t["judge_out"] * judge_out / 1e6,
            "OpenAI embedding": (emb + t["embq_in"]) * p.embed_input / 1e6,
            "_tokens": dict(t, embedding_build=emb),
        }
    cal = Path("work/calibration_cache.sqlite")
    cal_cost = 0.0
    if cal.exists():
        t = collections.Counter()
        for (v,) in sqlite3.connect(cal).execute("select value from cache"):
            m = json.loads(v).get("meta") or {}
            if m.get("provider") == "xiaomi":
                t["in"] += m["input_tokens"]; t["out"] += m["output_tokens"]
        cal_cost = t["in"] * judge_in / 1e6 + t["out"] * judge_out / 1e6
    return {"by_dataset": rows, "judge_calibration": cal_cost}


def main(out: str) -> None:
    cfg = Config.load("configs/experiment.yaml")
    data: dict = {"datasets": {}, "groups": GROUP_NAMES, "pricing": cfg.pricing.model_dump(),
                  "config": {"test_questions_per_dataset": None, "repeats": cfg.repeats, "r_max": cfg.search.r_max,
                             "top_k": cfg.search.top_k, "e_max": cfg.search.e_max}}
    for ds, label in DATASETS.items():
        ws = Workspace(cfg, ds)
        d = {"label": label, "settings": {}, "frozen": json.loads(ws.frozen_path.read_text())}
        size = POOL_SIZE.get(ds, 300)
        for key, (pat, slabel) in SETTINGS.items():
            run = pat.format(ds=ds, size=size)
            if ds == "bcp100":
                run = run.replace("final_", RUN_SUFFIX + "_", 1)
            slabel = slabel.format(size=f"{size:,}")
            summ = json.load(open(f"runs/{run}/summary.json", encoding="utf-8"))
            failed = {f["qid"] for f in load_failures(f"runs/{run}/failures.jsonl")}
            keep = set(ws.eval_qids("test")) - failed
            rows = [json.loads(l) for l in open(f"runs/{run}/results.jsonl", encoding="utf-8")]
            rows = [r for r in rows if r["qid"] in keep]
            by = collections.defaultdict(list)
            for r in rows:
                by[r["group"]].append(r)
            split = {}
            for g, rs in by.items():
                tot = collections.Counter()
                for r in rs:
                    tot.update(vendor_cost(r, cfg.pricing))
                n = len(rs)
                split[g] = {k: v / n for k, v in tot.items()}
                split[g]["_total"] = sum(tot.values()) / n
                lat = [r["latency_ms"] / 1000 for r in rs]
                split[g]["_lat_p50"], split[g]["_lat_p95"] = float(np.percentile(lat, 50)), float(np.percentile(lat, 95))
            groups = []
            for s in summ["summary"]:
                g = s["group"]
                groups.append({
                    "group": g, "name": GROUP_NAMES[g], "rows": s["n"], "faithfulness": s["faithfulness"],
                    "fully_faithful": s["fully_faithful"], "abstain_rate": s["abstain_rate"], "completeness": s["completeness"],
                    "mean_rounds": s["mean_rounds"], "r_max_truncated": s["r_max_truncated"],
                    "premature_stop": s["premature_stop"], "covered_all_final": s["covered_all_final"],
                    "recall_first5": s["recall_first@5"], "recall_final": s["recall_final"], "ndcg10_first": s["ndcg10_first"],
                    "cost": split[g], "latency_p50_s": split[g]["_lat_p50"], "latency_p95_s": split[g]["_lat_p95"],
                    "cost_per_faithful": s["cost_usd_per_faithful"], "accuracy": s["accuracy"],
                    "cost_per_correct": s["cost_usd_per_correct"],
                })
            d["settings"][key] = {"label": slabel, "run": run, "questions": len(keep), "failed_questions": sorted(failed),
                                  "groups": groups, "contrasts": summ["contrasts"]}
        data["datasets"][ds] = d
    data["config"]["test_questions_per_dataset"] = {ds: len(Workspace(cfg, ds).eval_qids("test")) for ds in DATASETS}
    data["spend"] = spend(cfg)
    data["spend"].pop("bcp100_embedding_so_far", None)
    cal = json.load(open("work/calibration_n150.json", encoding="utf-8"))["agreement"]
    data["calibration"] = cal
    Path(out).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "report/report_data.json")
