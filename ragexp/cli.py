"""python -m ragexp.cli <command> ...   (see README.md for the full workflow)"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import datasets, report, runner
from .config import Config


def _cfg(args) -> Config:
    return Config.load(args.config) if args.config and Path(args.config).exists() else Config()


def cmd_convert_bcp(args):
    print(datasets.convert_browsecomp_plus(args.queries, args.corpus, args.out))


def cmd_subset(args):
    print(datasets.subset_by_pools(args.src, args.out, args.n, args.seed))


def cmd_convert_ragbench(args):
    print(datasets.convert_ragbench(args.files, args.out, args.subset, args.max_questions, args.seed))


def cmd_prepare(args):
    runner.prepare(_cfg(args), args.dataset, args.data_dir, embed=not args.skip_embed)


def _env(args, cfg, ws, fake=False):
    return runner.build_env(cfg, ws, args.run_id, fake=fake, with_jev=not args.no_jev)


def cmd_traj(args):
    cfg = _cfg(args)
    ws = runner.Workspace(cfg, args.dataset)
    env = _env(args, cfg, ws)
    runner.run_trajectories(env, ws, args.split, args.repeats, args.limit, args.workers, args.pool_size)


def cmd_tune(args):
    cfg = _cfg(args)
    print(json.dumps(runner.run_tune(cfg, runner.Workspace(cfg, args.dataset), args.run_id), indent=2))


def cmd_groups(args):
    cfg = _cfg(args)
    ws = runner.Workspace(cfg, args.dataset)
    env = _env(args, cfg, ws)
    groups = tuple(args.groups.split(",")) if args.groups else runner.ALL_GROUPS
    runner.run_groups(env, ws, args.split, groups, args.repeats, args.limit, args.workers, args.pool_size)
    env.recorder.close()
    from .recorder import dedupe_records

    n = dedupe_records(Path(cfg.runs_dir) / args.run_id / "records.jsonl")
    if n:
        print(f"removed {n} duplicate record rows left by an interrupted run")


def cmd_grade(args):
    cfg = _cfg(args)
    runner.grade(cfg, runner.Workspace(cfg, args.dataset), args.run_id, workers=args.workers,
                 replicates=args.replicates)


def cmd_calibrate(args):
    from .cache import ApiCache
    from .calibrate import calibrate
    from .clients import GLMClient

    cfg = _cfg(args)
    judge = GLMClient(cfg.judge, ApiCache(Path(cfg.work_dir) / "calibration_cache.sqlite"))
    out = calibrate(judge, args.files, args.n, args.seed, args.workers)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["agreement"], indent=2))


def cmd_report(args):
    cfg = _cfg(args)
    ws = runner.Workspace(cfg, args.dataset)
    rows = report.load_results(Path(cfg.runs_dir) / args.run_id / "results.jsonl")
    in_scope = set(ws.eval_qids("test"))  # results of questions outside the test subsample are ignored
    rows = [r for r in rows if r["qid"] in in_scope]
    failed = {f["qid"] for f in runner.load_failures(Path(cfg.runs_dir) / args.run_id / "failures.jsonl")}
    if failed:  # drop the question from every group, so all groups are compared on the same questions
        rows = [r for r in rows if r["qid"] not in failed]
        print(f"excluded {len(failed)} question(s) with a blocked/failed call: {sorted(failed)}")
    qs = {q.qid: q for q in datasets.load_questions(ws.questions_path)}
    summary = report.summarize(rows, qs, cfg.pricing, cfg.search.r_max)
    md = report.to_markdown(summary)
    out = Path(cfg.runs_dir) / args.run_id
    (out / "summary.json").write_text(json.dumps({"summary": summary, "contrasts": report.contrasts(rows, cfg.pricing)},
                                                 indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)


def cmd_demo(args):
    """End-to-end run on a synthetic two-hop corpus with fake models. No network, no API keys."""
    from . import fakes
    from .chunking import get_encoding  # noqa: F401

    tmp = Path(tempfile.mkdtemp(prefix="ragexp_demo_"))
    cfg = Config(work_dir=str(tmp / "work"), runs_dir=str(tmp / "runs"), repeats=2, dev_fraction=0.34)
    cfg.search.top_k, cfg.search.r_max = 2, 4
    docs, qs = fakes.make_two_hop_corpus(n_questions=9)
    datasets.write_normalised(tmp / "data", docs, qs)
    ws = runner.prepare(cfg, "toy", tmp / "data", embedder=fakes.FakeEmbedder(), log=lambda *_: None)
    env = runner.build_env(cfg, ws, "demo", fake=True)
    runner.run_trajectories(env, ws, "dev", workers=1, log=lambda *_: None)
    print("tune:", json.dumps({k: v for k, v in runner.run_tune(cfg, ws, "demo").items()}, indent=1))
    runner.run_groups(env, ws, "test", workers=1, pool_size=40, log=lambda *_: None)
    env.recorder.close()
    rows = report.load_results(Path(cfg.runs_dir) / "demo" / "results.jsonl")
    qd = {q.qid: q for q in qs}
    for r in rows:  # RAGBench-style grading (toy questions carry relevant_sentences) with the fake judge
        runner.grade_row(env.llm, qd[r["qid"]], r, env.store)
    print(report.to_markdown(report.summarize(rows, qd, cfg.pricing, cfg.search.r_max)))
    print(f"\nartifacts in {tmp}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="ragexp")
    p.add_argument("--config", default="configs/experiment.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        sp = sub.add_parser(name, **kw)
        sp.set_defaults(fn=fn)
        return sp

    s = add("convert-bcp", cmd_convert_bcp)
    s.add_argument("--queries", required=True); s.add_argument("--corpus", required=True); s.add_argument("--out", required=True)
    s = add("subset", cmd_subset, help="n questions + the documents of their candidate pools -> new dataset")
    s.add_argument("--src", required=True); s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, required=True); s.add_argument("--seed", type=int, default=20261002)
    s = add("convert-ragbench", cmd_convert_ragbench)
    s.add_argument("--subset", required=True); s.add_argument("--out", required=True); s.add_argument("files", nargs="+")
    s.add_argument("--max-questions", type=int); s.add_argument("--seed", type=int, default=20261002)
    s = add("prepare", cmd_prepare)
    s.add_argument("--dataset", required=True); s.add_argument("--data-dir", required=True)
    s.add_argument("--skip-embed", action="store_true", help="chunk + split only (e.g. G5 trial without embeddings)")
    for name, fn in (("traj", cmd_traj), ("tune", cmd_tune), ("groups", cmd_groups), ("grade", cmd_grade), ("report", cmd_report)):
        s = add(name, fn)
        s.add_argument("--dataset", required=True); s.add_argument("--run-id", required=True)
        s.add_argument("--workers", type=int, default=4)
        if name in ("traj", "groups"):
            s.add_argument("--split", choices=["dev", "test"], required=True)
            s.add_argument("--repeats", type=int); s.add_argument("--limit", type=int)
            s.add_argument("--pool-size", type=int, help="reduced candidate pool (spec section 9)")
            s.add_argument("--no-jev", action="store_true")
        if name == "grade":
            s.add_argument("--replicates", type=int, help="grade only repeats 0..N-1 (budget)")
        if name == "groups":
            s.add_argument("--groups", help="comma list, default all, e.g. G1,G2,G3,G4,G6,G7")
    s = add("calibrate", cmd_calibrate)
    s.add_argument("files", nargs="+", help="raw RAGBench jsonl files (with annotations)")
    s.add_argument("--n", type=int, default=100); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=4); s.add_argument("--out", default="work/calibration.json")
    add("demo", cmd_demo)
    args = p.parse_args(argv)
    from .envfile import load_dotenv

    load_dotenv(".env")  # keys only; real environment variables take precedence
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
