from __future__ import annotations

import json
import threading
from pathlib import Path

COMPONENTS = (
    "embed_query",
    "retrieve",
    "glm_react_step",
    "glm_query_gen",
    "glm_judge",
    "glm_answer",
    "jev_judge",
    "jev_rank",
)

# Spec section 8. Tokens only, never money. `stop_reason` extends the spec's
# {judge, r_max, answered} with "fixed" for G6.
FIELDS = (
    "run_id", "group", "dataset", "query_id", "replicate", "round", "component",
    "model", "provider", "input_tokens", "output_tokens", "reasoning_tokens",
    "latency_ms", "retry_wait_ms", "http_status", "retries", "n_calls", "cached",
    "input_hash", "retrieved_chunk_ids", "doc_ids", "scores", "judge_output",
    "stop_reason", "covered_all", "covered_any",
)


class Recorder:
    """Append-only JSONL, one line per (run, group, dataset, query, replicate, round, component)."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self.rows: list[dict] | None = None if self.path else []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, **kw) -> dict:
        unknown = set(kw) - set(FIELDS)
        if unknown:
            raise KeyError(f"unknown record fields: {sorted(unknown)}")
        row = {f: kw.get(f) for f in FIELDS}
        if row["component"] is not None and row["component"] not in COMPONENTS:
            raise ValueError(f"unknown component {row['component']!r}")
        with self._lock:
            if self.path:
                self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                self._fh.flush()
            else:
                self.rows.append(row)
        return row

    def close(self) -> None:
        if self.path:
            self._fh.close()


def read_records(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dedupe_records(path: str | Path) -> int:
    """Rewrite a records file keeping the LAST row per (run, group, dataset, query, replicate, round,
    component). A run that is interrupted and resumed re-emits the calls of the unit it was in; the
    result table is unaffected, but the raw log would otherwise count those calls twice."""
    rows = read_records(path)
    last = {}
    for r in rows:
        last[(r["run_id"], r["group"], r["dataset"], r["query_id"], r["replicate"], r["round"], r["component"])] = r
    kept = [r for r in rows if last[(r["run_id"], r["group"], r["dataset"], r["query_id"], r["replicate"],
                                      r["round"], r["component"])] is r]
    if len(kept) != len(rows):
        Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n", encoding="utf-8")
    return len(rows) - len(kept)
