#!/usr/bin/env python3
"""Runs inside a task's grading image: the hidden grader's sandbox and run loop (/grader/grader.py, imported, not
copied), on an arbitrary set of attempts at /probe/pool, keeping each run's output. For the verify loop, which needs
to know why a grader rejected an attempt, not only that it did. Prints one JSON line."""
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location("grader", "/grader/grader.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
G.GRADER = Path("/probe")
_bwrap = G.bwrap_argv


def _bwrap_hiding_both(*args, **kwargs):
    """The image's own labelled pool stays at /grader; the sandbox hides it as well as /probe, so a test run here can
    read neither the probe's attempts nor the hidden labels."""
    a = _bwrap(*args, **kwargs)
    i = a.index("/verifier_out")
    return a[:i + 1] + ["--tmpfs", "/grader"] + a[i + 1:]


G.bwrap_argv = _bwrap_hiding_both
TAIL = 6000


def run_one(mode: str, entry: dict, cfg: dict) -> dict:
    stage = G.STAGE / entry["id"]
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "logs").mkdir(parents=True)
    shutil.copytree(G.WRITTEN, stage / "tests", symlinks=False, ignore_dangling_symlinks=True)
    for d in G.writable_dirs(cfg["artifacts"]):
        shutil.copytree(d, stage / "rw" / d.lstrip("/"), symlinks=True)
    for d in entry.get("dirs", []):
        (stage / "art" / d.lstrip("/")).mkdir(parents=True, exist_ok=True)
    for path, (sha, fmode) in entry.get("files", {}).items():
        dst = stage / "art" / path.lstrip("/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(G.GRADER / "pool" / "blobs" / sha, dst)
        dst.chmod(fmode)
    argv = G.bwrap_argv(stage, cfg["artifacts"], cfg["env"], cfg["workdir"], ["/bin/bash", "/tests/test.sh"], proc=mode)
    out = stage / "out.txt"
    t0, timed_out = time.monotonic(), False
    with out.open("wb") as fh:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, cwd="/",
                                start_new_session=True, env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"})
        try:
            rc = proc.wait(timeout=cfg["attempt_timeout_sec"])
        except subprocess.TimeoutExpired:
            timed_out, rc = True, None
        G._kill_group(proc)
    text = out.read_text(encoding="utf-8", errors="replace")
    verdict = "fail" if timed_out else G.verdict_of(stage / "logs")
    shutil.rmtree(stage, ignore_errors=True)
    return {"id": entry["id"], "verdict": verdict, "timed_out": timed_out, "rc": rc,
            "seconds": round(time.monotonic() - t0, 1),
            "output": text if len(text) <= TAIL else text[:TAIL // 3] + "\n...[cut]...\n" + text[-2 * TAIL // 3:]}


def main() -> int:
    cfg = json.loads((G.GRADER / "config.json").read_text(encoding="utf-8"))
    pool = json.loads((G.GRADER / "pool" / "manifest.json").read_text(encoding="utf-8"))
    if not (G.WRITTEN / "test.sh").is_file():
        print(json.dumps({"error": "no tests/test.sh"}))
        return 0
    mode = G.probe(cfg)
    if mode is None:
        print(json.dumps({"error": "no sandbox"}))
        return 1
    print(json.dumps({"task": cfg["task"], "isolation": mode, "runs": [run_one(mode, e, cfg) for e in pool]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
