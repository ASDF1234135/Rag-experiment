"""Long-running, restartable embedding driver: python -m scripts.embed_all [dataset ...]

Runs `prepare` (chunk -> embed -> index) for each dataset in order. Embedding progress is saved in
small shards, so Ctrl-C / a crash / a reboot loses at most one shard; just run it again. Transient
failures are retried. Progress goes to stdout and work/embed_all.log.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

from ragexp import runner
from ragexp.config import Config
from ragexp.envfile import load_dotenv

DEFAULT_ORDER = ["covidqa", "hotpotqa", "techqa", "bcp100"]


def main(datasets: list[str]) -> int:
    load_dotenv(".env")
    cfg = Config.load("configs/experiment.yaml")
    log_path = Path(cfg.work_dir) / "embed_all.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    for ds in datasets:
        if (Path(cfg.work_dir) / ds / "emb" / "emb.npy").exists():
            log(f"{ds}: already embedded, skipping")
            continue
        for attempt in range(1, 21):
            try:
                log(f"{ds}: start (attempt {attempt})")
                runner.prepare(cfg, ds, Path("data") / ds, log=lambda m, ds=ds: log(f"{ds}: {m}"))
                log(f"{ds}: DONE")
                break
            except Exception as e:  # noqa: BLE001 - keep going; shards already written are kept
                log(f"{ds}: {type(e).__name__}: {str(e)[:200]} -- retrying in 60 s")
                time.sleep(60)
        else:
            log(f"{ds}: giving up after 20 attempts")
            return 1
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or DEFAULT_ORDER))
