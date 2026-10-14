"""Grader writing, step 2: run each task's checks on the unfixed code (base) and on every attempt, in the instance's
image with no network.

    python scripts/run_checks.py REAL_DIR CHECKS_DIR OUT_DIR [--workers 16]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ["sudo", "-n", "docker"]


def one(inst: dict, key: str, patch: str, checks: Path, base: Path) -> None:
    d = base / inst["instance_id"] / key
    if (d / "out" / "result.json").is_file():
        return
    if d.exists():
        subprocess.run(DOCKER[:2] + ["rm", "-rf", str(d)], capture_output=True)
    (d / "out").mkdir(parents=True)
    (d / "spec.json").write_text(json.dumps({"mode": "base" if key == "base" else "attempt", "timeout": 120}))
    (d / "attempt.patch").write_text(patch if patch.endswith("\n") else patch + "\n")
    shutil.copy(checks, d / "checks.py")
    shutil.copy(ROOT / "vm" / "real" / "checks_in_container.py", d / "run.py")
    for x in (d, d / "out"):
        x.chmod(0o777)
    cmd = DOCKER + ["run", "--rm", "--network", "none", "--memory", "8g", "--cpus", "2", "-v", f"{d}:/work", "-w", "/testbed",
                    inst["docker_image"], "bash", "-c",
                    "P=/opt/conda/envs/testbed/bin/python; [ -x $P ] || P=$(command -v python); exec $P /work/run.py"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
        if not (d / "out" / "result.json").is_file():
            (d / "out" / "error.txt").write_text((r.stdout + r.stderr)[-2000:])
    except subprocess.TimeoutExpired:
        (d / "out" / "error.txt").write_text("container timed out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("real_dir", type=Path)
    ap.add_argument("checks_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    insts = {i["instance_id"]: i for i in json.loads((a.real_dir / "instances.json").read_text())}
    jobs = []
    for f in sorted(a.checks_dir.glob("*.py")):
        inst = insts.get(f.stem)
        if not inst:
            continue
        jobs.append((inst, "base", "", f))
        jobs += [(inst, f"a{i}", at["patch"], f) for i, at in enumerate(inst["attempts"])]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda j: one(*j, a.out), jobs))
    print(json.dumps({"runs": len(jobs), "results": len(list(a.out.glob("*/*/out/result.json")))}))


if __name__ == "__main__":
    main()
