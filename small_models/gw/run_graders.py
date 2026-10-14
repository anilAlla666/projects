"""Run each written grader (a test patch) on the unfixed code and on every labelled attempt of its task.

    python gw/run_graders.py REAL_DIR GEN_DIR OUT_DIR [--workers 24]
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


def one(inst, key, patch, grader: Path, base: Path):
    d = base / grader.stem / key  # one folder per grader: "<task>" or, with several drafts, "<task>.<k>"
    if (d / "out" / "result.json").is_file():
        return
    if d.exists():
        subprocess.run(DOCKER[:2] + ["rm", "-rf", str(d)], capture_output=True)
    (d / "out").mkdir(parents=True)
    (d / "spec.json").write_text(json.dumps({"mode": "base" if key == "base" else "attempt", "base_commit": inst["base_commit"], "timeout": 300}))
    (d / "attempt.patch").write_text(patch if patch.endswith("\n") else patch + "\n")
    g = grader.read_text()
    (d / "grader.patch").write_text(g if g.endswith("\n") else g + "\n")
    shutil.copy(ROOT / "gw" / "tests_in_container.py", d / "run.py")
    shutil.copy(ROOT / "vm" / "sandbox" / "vm_results_plugin.py", d / "vm_results_plugin.py")
    for x in (d, d / "out"):
        x.chmod(0o777)
    cmd = DOCKER + ["run", "--rm", "--network", "none", "--memory", "8g", "--cpus", "2", "-v", f"{d}:/work", "-w", "/testbed",
                    inst["docker_image"], "bash", "-c",
                    "P=/opt/conda/envs/testbed/bin/python; [ -x $P ] || P=$(command -v python); exec $P /work/run.py"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=500)
        if not (d / "out" / "result.json").is_file():
            (d / "out" / "error.txt").write_text((r.stdout + r.stderr)[-1500:])
    except subprocess.TimeoutExpired:
        (d / "out" / "error.txt").write_text("container timed out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("real", type=Path)
    ap.add_argument("gen", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--all", action="store_true", help="every task in the instances file, not only the labelled real set")
    a = ap.parse_args()
    insts = {i["instance_id"]: i for i in json.loads((a.real / "instances.json").read_text())}
    ex_file = a.real / "examples_realistic.jsonl"
    labelled = None if a.all else {json.loads(l)["example_id"].split("/")[0].removeprefix("real:") for l in ex_file.read_text().splitlines()}
    jobs = []
    for g in sorted(a.gen.glob("*.patch")):
        iid = g.stem if g.stem in insts else g.stem.rsplit(".", 1)[0]
        inst = insts.get(iid)
        if not inst or (labelled is not None and iid not in labelled) or not g.read_text().strip():
            continue
        jobs.append((inst, "base", "", g))
        jobs += [(inst, f"a{i}", at["patch"], g) for i, at in enumerate(inst["attempts"])]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda j: one(*j, a.out), jobs))
    print(json.dumps({"runs": len(jobs)}))


if __name__ == "__main__":
    main()
