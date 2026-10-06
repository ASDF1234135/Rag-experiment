"""Restartable end-to-end driver for the finished datasets.

  python -m scripts.run_experiment full  covidqa hotpotqa techqa
  python -m scripts.run_experiment pool  covidqa hotpotqa techqa      (waits for each dataset's tuned tau)

full : dev trajectories -> tune tau / R_fixed (dev only) -> test G1,G2,G3,G4,G6,G7 on the full corpus
       (3 repeats) -> grade -> report.
pool : after `full` has tuned a dataset: test G1-G7 on per-question candidate pools of POOL_SIZE chunks
       (1 repeat; G5 is Jev-rate-limited and Jev is deterministic), same frozen tau -> grade -> report.
Every step is resumable (trajectories, results rows and API calls are cached), so after a crash or
reboot simply run the same command again.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

POOL_SIZE = {"bcp100": 1500}   # candidate-pool chunks per question (default below)
DEFAULT_POOL = 300
GRADE_REPLICATES_FULL = 1       # budget: judge only repeat 0 of the 3-repeat full runs
DEV_WORKERS, TEST_WORKERS, JUDGE_WORKERS = 8, 10, 28   # judge call ~29 s: 28 workers ~ 48 req/min


def log(stage: str, msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} [{stage}] {msg}"
    print(line, flush=True)
    with open(f"work/run_experiment_{stage}.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def cli(stage: str, *args: str, tries: int = 4) -> None:
    cmd = [sys.executable, "-m", "ragexp.cli", *args]
    for attempt in range(1, tries + 1):
        log(stage, "RUN " + " ".join(args) + (f" (attempt {attempt})" if attempt > 1 else ""))
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        tail = (r.stdout.strip().splitlines() or [""])[-1][:300]
        if r.returncode == 0:
            log(stage, "ok: " + tail)
            return
        err = (r.stderr.strip().splitlines() or [""])[-1][:300]
        log(stage, f"FAILED rc={r.returncode}: {err}")
        if any(k in err.lower() for k in ("insufficient balance", "insufficient_quota", "credit_balance", "http 402")):
            log(stage, "OUT OF CREDIT -- stopping instead of retrying. Recharge, then rerun the same command (resumable).")
            raise SystemExit(2)
        time.sleep(60)
    raise SystemExit(f"{args[0]} failed {tries} times")


def wait_for_index(stage: str, ds: str) -> None:
    emb = Path("work") / ds / "emb" / "emb.npy"
    if not emb.exists():
        log(stage, f"{ds}: waiting for embeddings ({emb})")
    while not emb.exists():
        time.sleep(300)


def full(ds: str) -> None:
    wait_for_index("full", ds)
    run_dev, run_full = f"dev_{ds}", f"final_{ds}_full"
    marker = Path("runs") / run_dev / "TUNED"
    if not marker.exists():
        cli("full", "traj", "--dataset", ds, "--run-id", run_dev, "--split", "dev", "--repeats", "1",
            "--workers", str(DEV_WORKERS))
        cli("full", "tune", "--dataset", ds, "--run-id", run_dev)
        marker.write_text(datetime.now().isoformat())
    cli("full", "groups", "--dataset", ds, "--run-id", run_full, "--split", "test", "--groups",
        "G1,G2,G3,G4,G6,G7", "--repeats", "3", "--workers", str(TEST_WORKERS))
    cli("full", "grade", "--dataset", ds, "--run-id", run_full, "--workers", str(JUDGE_WORKERS),
        "--replicates", str(GRADE_REPLICATES_FULL))
    cli("full", "report", "--dataset", ds, "--run-id", run_full)


def pool(ds: str) -> None:
    marker = Path("runs") / f"dev_{ds}" / "TUNED"
    while not marker.exists():
        time.sleep(60)
    size = POOL_SIZE.get(ds, DEFAULT_POOL)
    run = f"final_{ds}_pool{size}"
    cli("pool", "groups", "--dataset", ds, "--run-id", run, "--split", "test", "--repeats", "1",
        "--pool-size", str(size), "--workers", str(TEST_WORKERS))
    cli("pool", "grade", "--dataset", ds, "--run-id", run, "--workers", str(JUDGE_WORKERS))
    cli("pool", "report", "--dataset", ds, "--run-id", run)


if __name__ == "__main__":
    stage, datasets = sys.argv[1], sys.argv[2:]
    Path("work").mkdir(exist_ok=True)
    for d in datasets:
        {"full": full, "pool": pool}[stage](d)
    log(stage, "ALL DONE")
