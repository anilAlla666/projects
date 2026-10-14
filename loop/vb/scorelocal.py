"""Score written graders with the HUD taskset's own hidden grader, locally in Docker.

Same grader, same pool, same sandbox as on HUD; the only difference is that Docker must be told to allow the
grader's nested sandbox (user namespaces, unconfined seccomp and AppArmor), since HUD's runtime already does.
Results are cached by (task, content hash of the tests folder).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .hudtask import HUD_TASKS
from .regrade import tests_hash
from .util import DATA, read_json, write_json

CACHE = DATA / "exp_scores"
DOCKER_OPTS = ["--cap-add", "SYS_ADMIN", "--security-opt", "seccomp=unconfined", "--security-opt",
               "apparmor=unconfined"]


def image(task: str) -> str:
    tag = f"vbhud-{task}"
    subprocess.run(["docker", "build", "-q", "-t", tag, str(HUD_TASKS / f"write-verifier-{task}" / "tests")],
                   check=True, capture_output=True)
    return tag


def score(task: str, tests: Path | None, img: str | None = None) -> dict:
    """The grader's JSON summary for one tests folder (None: no tests handed in)."""
    key = CACHE / task / f"{tests_hash(tests) if tests else 'none'}.json"
    if key.exists():
        return read_json(key)
    mount = ["-v", f"{Path(tests).resolve()}:/verifier_out/tests:ro"] if tests else []
    r = subprocess.run(["docker", "run", "--rm", *DOCKER_OPTS, *mount, img or image(task), "bash", "/tests/test.sh"],
                       capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
    out = json.loads(lines[-1]) if lines else {"error": "no summary", "stderr": r.stderr[-800:]}
    if "reward" in out:
        write_json(key, out)
    return out


def score_many(items: list[tuple[str, Path | None]], workers: int = 3) -> list[dict]:
    imgs = {t: image(t) for t in {t for t, _ in items}}
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda it: score(it[0], it[1], imgs[it[0]]), items))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("tests", nargs="*")
    a = ap.parse_args()
    for t, s in zip(a.tests or [None], score_many([(a.task, Path(x) if x else None) for x in (a.tests or [None])])):
        print(t, json.dumps({k: s.get(k) for k in ("reward", "balanced_accuracy", "tpr", "tnr", "seconds", "error")}))
