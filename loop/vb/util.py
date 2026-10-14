"""Paths, JSON IO, run directories, .env loading and Harbor calls."""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("VB_DATA") or ("/root/vbench" if os.path.isdir("/root/vbench") else ROOT / "data"))  # isdir is False, not an error, without access
TB_ROOT = Path(os.environ.get("VB_TB_ROOT", "/root/tb4/terminal-bench/tasks"))  # the checkout moved tasks into tasks/
HARBOR = os.environ.get("VB_HARBOR", "/root/venv/bin/harbor")


def read_json(path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_sha() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "vb", "configs", "templates"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def new_run_dir(name: str, root: Path | None = None) -> Path:
    """<root>/<name>/<UTC stamp>_<git sha>[_n]/, created fresh; an existing path is never reused."""
    root = Path(root) if root is not None else DATA / "runs"
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = root / name / f"{stamp}_{git_sha()}"
    d, n = base, 1
    while d.exists():
        d = base.with_name(f"{base.name}_{n}")
        n += 1
    d.mkdir(parents=True)
    return d


def load_env(path: Path | None = None) -> list[str]:
    """Load KEY=VALUE lines into os.environ. Returns the names loaded, never the values."""
    path = Path(path) if path is not None else DATA / ".env"
    if not path.is_file():
        return []
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()
        names.append(key.strip())
    return names


def harbor_json(*args: str) -> Any:
    out = subprocess.run([HARBOR, *args, "--json"], capture_output=True, text=True, check=True).stdout
    return json.loads(out)
