"""Render the README figures from report/report_data.json into docs/img/.

python -m scripts.make_readme_charts
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path("report/report_data.json")
OUT = Path("docs/img")
D = json.loads(DATA.read_text(encoding="utf-8"))

# Baselines are blue/grey; groups that use Jev are warm so they stand out.
COLOR = {"G1": "#9aa0a6", "G2": "#5b8fd1", "G3": "#2b5fa8", "G4": "#e08a1e", "G5": "#c0392b", "G6": "#7fb7a4", "G7": "#4d8f7a", "G8": "#8e44ad"}
NAME = {"G1": "G1 single round", "G2": "G2 ReAct", "G3": "G3 LLM stop", "G4": "G4 Jev stop", "G5": "G5 Jev retrieval",
        "G6": "G6 fixed rounds", "G7": "G7 oracle stop", "G8": "G8 Jev re-rank"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight"})


def groups(ds, setting):
    return {g["group"]: g for g in D["datasets"][ds]["settings"][setting]["groups"]}


def fig_bcp_accuracy():
    g = groups("bcp100", "pool")
    n = D["datasets"]["bcp100"]["settings"]["pool"]["questions"]
    order = ["G1", "G2", "G3", "G4", "G6", "G7", "G8", "G5"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    vals = [g[k]["accuracy"] for k in order]
    bars = ax.bar([NAME[k] for k in order], vals, color=[COLOR[k] for k in order])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"BrowseComp-Plus (reduced), candidate pool: accuracy by group (n={n})", fontsize=11, loc="left")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.savefig(OUT / "bcp_accuracy.png")
    plt.close(fig)


OFFSET = {"G7": (-20, -4), "G4": (8, -12), "G6": (-8, 9), "G3": (8, 4), "G2": (-20, -14)}  # keep overlapping labels apart


def fig_bcp_frontier():
    g = groups("bcp100", "pool")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for k, v in g.items():
        x, y = v["cost"]["_total"] * 1000, v["accuracy"]
        ax.scatter(x, y, s=90, color=COLOR[k], zorder=3)
        ax.annotate(k, (x, y), textcoords="offset points", xytext=OFFSET.get(k, (7, 5)), fontsize=10, color=COLOR[k], fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("Cost per 1,000 questions (USD, log scale, excluding grading)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("BrowseComp-Plus (reduced), candidate pool: accuracy vs cost", fontsize=11, loc="left")
    fig.savefig(OUT / "bcp_cost_accuracy.png")
    plt.close(fig)


def fig_ragbench_completeness():
    labels = {"covidqa": "CovidQA", "hotpotqa": "HotpotQA", "techqa": "TechQA"}
    order = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.115
    for i, k in enumerate(order):
        vals = [groups(ds, "pool")[k]["completeness"] for ds in labels]
        xs = [j + (i - 3) * w for j in range(len(labels))]
        ax.bar(xs, vals, width=w, color=COLOR[k], label=NAME[k])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels.values())
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Completeness")
    ax.set_title("RAGBench, candidate pool (300 chunks): completeness by group (n=66 each)", fontsize=11, loc="left")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.savefig(OUT / "ragbench_completeness.png")
    plt.close(fig)


def contrast(ds, setting, a, b):
    for c in D["datasets"][ds]["settings"][setting]["contrasts"]:
        if c["a"] == a and c["b"] == b:
            return c
    return None


def fig_jev_forest():
    # (dataset, setting, a, b, flip, row label); flip=True shows b - a so that Jev is always the positive side.
    rows = [
        ("bcp100", "pool", "G3", "G4", True, "Stop judge: Jev - LLM"),
        ("bcp100", "pool", "G4", "G5", True, "Retrieval: Jev full scoring - embedding"),
        ("bcp100", "pool", "G4", "G8", True, "Retrieval: Jev re-rank top 200 - embedding"),
        ("bcp100", "pool", "G8", "G5", True, "Jev full scoring - Jev re-rank"),
    ]
    for ds, name in [("covidqa", "CovidQA"), ("hotpotqa", "HotpotQA"), ("techqa", "TechQA")]:
        rows.append((ds, "pool", "G4", "G5", True, f"Retrieval: Jev - embedding ({name})"))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ys = list(range(len(rows)))[::-1]
    for y, (ds, st, a, b, flip, label) in zip(ys, rows):
        c = contrast(ds, st, a, b)
        metric = "correct" if ds == "bcp100" else "completeness"
        d = c["metrics"][metric]["diff"]
        m, lo, hi = d["mean_diff"], d["lo"], d["hi"]
        if flip:
            m, lo, hi = -m, -hi, -lo
        sig = lo > 0 or hi < 0
        col = "#e08a1e" if ds == "bcp100" else "#2b5fa8"
        ax.plot([lo, hi], [y, y], color=col, lw=2)
        ax.scatter([m], [y], s=60, color=col if sig else "white", edgecolor=col, zorder=3, linewidth=1.8)
        ax.text(1.02, y, f"{m:+.2f}", va="center", fontsize=9, transform=ax.get_yaxis_transform())
    ax.axvline(0, color="#555", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[-1] for r in rows])
    ax.set_xlabel("Difference (positive = first named is better); orange: accuracy, blue: completeness")
    ax.set_title("Paired differences with 95% bootstrap CI (filled dot = CI excludes 0)", fontsize=11, loc="left")
    ax.grid(axis="y", alpha=0)
    fig.savefig(OUT / "jev_forest.png")
    plt.close(fig)


def fig_judge_auc():
    ds_labels = {"covidqa": "CovidQA", "hotpotqa": "HotpotQA", "techqa": "TechQA", "bcp100": "BrowseComp-Plus"}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    xs = range(len(ds_labels))
    jev = [D["datasets"][d]["frozen"]["jev"]["auc"] for d in ds_labels]
    llm = [D["datasets"][d]["frozen"]["llm"]["auc_binary"] for d in ds_labels]
    ax.bar([x - 0.2 for x in xs], jev, 0.4, color="#e08a1e", label="Jev judge (AUC)")
    ax.bar([x + 0.2 for x in xs], llm, 0.4, color="#2b5fa8", label="LLM judge (AUC, binary)")
    for x, v in zip(xs, jev):
        ax.text(x - 0.2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for x, v in zip(xs, llm):
        ax.text(x + 0.2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(0.5, color="#555", lw=1, ls="--")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(ds_labels.values())
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC on dev")
    ax.set_title('Can the judge tell "all gold documents are in the evidence"?', fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(OUT / "judge_auc.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_bcp_accuracy, fig_bcp_frontier, fig_ragbench_completeness, fig_jev_forest, fig_judge_auc):
        f()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
