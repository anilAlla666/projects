"""Regrade recorded attempts with a task copy's verifier, and read the rewards back.

Two measured Harbor behaviours shape this module (26 Sep 2026, swap proof on Hetzner):
1. Harbor's own verifier build reused stale files when two test folders held files of equal size and
   timestamp, so 'always fail' passed. We build each verifier image ourselves with `docker build`, tagged by
   a hash of the folder's content, and pin the task copy to it.
2. Harbor enforces `[verifier] network_mode = "no-network"` through an egress sidecar that redirects every TCP
   connection to a local proxy. A bare TCP connect therefore succeeds even when the network is blocked; only an
   application-level reply proves reachability. The network_probe baseline asks GitHub for an HTTP reply, and was
   shown to answer 1 when Harbor's network is public and 0 under our offline setting."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from .taskdir import make_verifier_task
from .util import HARBOR

def run_regrade(job_dir: Path, task_dir: Path, out_root: Path, job_name: str, n_concurrent: int = 4,
                timeout_s: int = 21600) -> Path:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / f"{job_name}.log").open("w") as log:
        subprocess.run([HARBOR, "job", "regrade", str(job_dir), "-p", str(task_dir), "-o", str(out_root),
                        "--job-name", job_name, "-n", str(n_concurrent)],
                       stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
    return out_root / job_name


def read_rewards(regraded_job: Path) -> dict[str, float | None]:
    """{source attempt dir name: reward, or None when the verifier errored or wrote no reward}."""
    out: dict[str, float | None] = {}
    for rj in sorted(Path(regraded_job).glob("*/result.json")):
        r = json.loads(rj.read_text(encoding="utf-8"))
        src = ((r.get("config") or {}).get("source_trial") or {})
        key = Path(src.get("path") or "").name or r.get("trial_name") or rj.parent.name
        reward = (((r.get("verifier_result") or {}).get("rewards")) or {}).get("reward")
        out[key] = None if r.get("exception_info") or reward is None else float(reward)
    return out


# Failures of the grading machinery, never of the verifier: retried, never scored. A compose failure for any other
# reason (e.g. a writer's image whose ENTRYPOINT exits) stays an error charged to the verifier.
INFRA_MARKERS = ("all predefined address pools have been fully subnetted", "failed to create network",
                 "Cannot connect to the Docker daemon", "no space left on device")


def infra_failures(regraded_job: Path) -> list[str]:
    """Source attempts whose regrade failed for an infrastructure reason (see INFRA_MARKERS)."""
    out = []
    for rj in sorted(Path(regraded_job).glob("*/result.json")):
        r = json.loads(rj.read_text(encoding="utf-8"))
        msg = ((r.get("exception_info") or {}).get("exception_message") or "")
        if any(m in msg for m in INFRA_MARKERS):
            src = ((r.get("config") or {}).get("source_trial") or {})
            out.append(Path(src.get("path") or "").name or rj.parent.name)
    return out


def verdict(reward: float | None) -> str:
    if reward is None:
        return "error"
    return "pass" if reward >= 1.0 else "fail"


def tests_hash(tests_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(q for q in Path(tests_dir).rglob("*") if q.is_file()):
        h.update(p.relative_to(tests_dir).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def build_image(tests_dir: Path, log: Path | None = None) -> str | None:
    """Build a verifier image from tests_dir with the network available. Returns its tag, or None if the
    build failed (every attempt is then an error for this verifier).

    The contract is that the grader runs `bash /tests/test.sh` itself, so any ENTRYPOINT the writer set is cleared
    in a thin layer on top: with one, the container ran the tests and exited at start, and Harbor could not run
    them (our instruction had called test.sh "the entry point", which invites that reading)."""
    h = tests_hash(tests_dir)
    raw, tag = f"vbv-raw:{h}", f"vbv:{h}"
    proc = subprocess.run(["docker", "build", "-t", raw, str(tests_dir)], capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if proc.returncode == 0:
        wrap = subprocess.run(["docker", "build", "-t", tag, "-"], input=f"FROM {raw}\nENTRYPOINT []\n",
                              capture_output=True, text=True)
        out += wrap.stdout + wrap.stderr
        proc = wrap
    if log is not None:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text(out)
    return tag if proc.returncode == 0 else None


def grade(job_dir: Path, task_dir: Path, tests_dir: Path, work: Path, name: str, image: str | None = None,
          n_concurrent: int | None = None) -> dict[str, float | None]:
    """Grade every attempt in job_dir with the verifier in tests_dir: build the image (with network), then regrade
    (offline, by the task copy's network_mode).
    Pass `image` to use an existing image instead (the official one for the original tests)."""
    work = Path(work)
    if n_concurrent is None:  # 4 by default; 1 for heavy tasks (four JVM builds at once overloaded an 8-core, 16 GB host)
        n_concurrent = int(os.environ.get("VB_REGRADE_CONCURRENCY", "4"))
    if image is None:
        image = build_image(tests_dir, log=work / "builds" / f"{name}.log")
    if image is None:
        return {}                                            # build failed: the caller counts every entry as error
    vt = make_verifier_task(task_dir, tests_dir, work / "vtasks" / name, image=image)
    return read_rewards(run_regrade(job_dir, vt, work / "regrades", name, n_concurrent=n_concurrent))
