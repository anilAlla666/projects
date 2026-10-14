"""Run a tests/ folder on any set of attempts, in the task's grading image and the hidden grader's sandbox, keeping
each run's output (templates/hud/probe_run.py). The verify loop uses it to test a written grader against independent
attempts and generated near misses, and to see why each one was rejected. Runs on the server.

    python -m vb.probe --task T --tests DIR --attempts DIR     # DIR/<id>/ mirrors the artifact paths, e.g. A/app/...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from .util import DATA, ROOT


def build_pool(attempts: Path, out: Path) -> list[dict]:
    """The content-addressed pool layout the hidden grader reads (manifest.json, blobs/<sha256>)."""
    (out / "blobs").mkdir(parents=True, exist_ok=True)
    manifest = []
    for a in sorted(p for p in attempts.iterdir() if p.is_dir()):
        files, dirs = {}, []
        for f in sorted(a.rglob("*")):
            rel = "/" + str(f.relative_to(a))
            if f.is_dir():
                if not any(f.iterdir()):
                    dirs.append(rel)
                continue
            data = f.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            blob = out / "blobs" / sha
            if not blob.exists():
                blob.write_bytes(data)
            files[rel] = [sha, f.stat().st_mode & 0o777]
        manifest.append({"id": a.name, "label": "unknown", "files": files, "dirs": dirs})
    (out / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def probe(task: str, tests: Path, attempts: Path, timeout: int = 7200) -> dict:
    grader = DATA / "hud_tasks" / f"write-verifier-{task}" / "tests" / "grader"
    with tempfile.TemporaryDirectory(prefix="vbprobe-") as tmp:
        d = Path(tmp)
        shutil.copy(grader / "config.json", d / "config.json")
        build_pool(attempts, d / "pool")
        shutil.copy(ROOT / "templates" / "hud" / "probe_run.py", d / "probe_run.py")
        cmd = ["docker", "run", "--rm", "--cap-add", "SYS_ADMIN", "--security-opt", "seccomp=unconfined",
               "--security-opt", "apparmor=unconfined", "-v", f"{tests.resolve()}:/verifier_out/tests:ro",
               "-v", f"{d}:/probe", f"vbhud-{task}", "python3", "/probe/probe_run.py"]
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"error": (r.stdout + r.stderr)[-1500:]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--tests", type=Path, required=True)
    ap.add_argument("--attempts", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(probe(a.task, a.tests, a.attempts)))
