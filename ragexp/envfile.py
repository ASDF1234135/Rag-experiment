"""Minimal .env loader (no extra dependency). Real environment variables win over the file."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Read KEY=VALUE lines into os.environ without overriding variables that are already set.
    Returns the names that were loaded (never the values)."""
    p = Path(path)
    if not p.exists():
        return []
    loaded = []
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.removeprefix("export ").strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key and val and key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded
