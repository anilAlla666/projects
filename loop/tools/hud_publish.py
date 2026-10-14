"""Publish built verifier-writing tasks to HUD, checking each one on HUD first.

Runs where the HUD CLI is logged in. For each task:
  1. copy the built Harbor task directory from the server;
  2. build a check copy whose writer machine holds the official tests and the test graders (always pass, always
     fail, a cheating grader), adapt and deploy it, and run one row per grader with a cheap model told to copy
     that grader into place (plus a row that hands in nothing); expect the official tests to score 1 and every
     other row 0;
  3. only if that holds, adapt and deploy the real task and sync it into the taskset.

  python tools/hud_publish.py --task hof-topology-interpenetration [--hud /path/to/hud] [--skip-check]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

SERVER = "root@178.156.248.194"
LOCAL = pathlib.Path("/Users/anilkumar/vbench_hud")
REPO = pathlib.Path(__file__).resolve().parents[1]
CHECKS = ["official", "always_pass", "always_fail", "cheat", "none"]
EXPECT = {"official": 1.0}
TASKSET = "verifier-bench"


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, **kw)


def adapt_one(task_dir: pathlib.Path) -> tuple[pathlib.Path, dict]:
    from hud.integrations.harbor import adapt
    r = adapt(task_dir)
    if r.failures:
        raise SystemExit(f"adapt failed: {r.failures}")
    # One adapted environment per task (separate verifiers group by task); the newest one holding this task.
    found = []
    for d in (task_dir.parent / ".hud-adapt").glob("*"):
        for x in json.loads((d / "tasks.json").read_text()):
            if x["args"]["task"]["id"] == task_dir.name:
                found.append((d.stat().st_mtime, d, x))
    _, d, row = max(found, key=lambda f: f[0])
    if "runtime_config" in row["verifier"]:
        raise SystemExit("the verifier row has its own runtime_config: it would grade on a fresh machine")
    (d / "runtime.json").write_text(json.dumps(row["runtime_config"], indent=1))
    return d, row


def deploy(hud: str, d: pathlib.Path) -> None:
    out = subprocess.run([hud, "deploy", ".", "--no-env", "--runtime-config", "runtime.json"], cwd=d,
                         capture_output=True, text=True)
    if "Build SUCCEEDED" not in out.stdout + out.stderr:
        raise SystemExit(f"deploy failed in {d}:\n{out.stdout[-2000:]}{out.stderr[-1000:]}")


def run_checks(hud: str, d: pathlib.Path, row: dict) -> dict:
    rows = []
    for v in CHECKS:
        r = json.loads(json.dumps(row))
        r["args"]["instruction"] = (
            "This is a check run. Do not create or change any files. Reply with the single word done and stop."
            if v == "none" else "This is a check run. Run exactly this one shell command, then reply done and stop:\n\n"
            f"mkdir -p /verifier_out && cp -r /opt/vb_checks/{v} /verifier_out/tests && ls /verifier_out/tests")
        r["slug"], r["verifier"]["slug"] = f"check-{v}", f"check-{v}:verify"
        rows.append(r)
    (d / "check_tasks.json").write_text(json.dumps(rows, indent=1))
    out = subprocess.run([hud, "eval", "check_tasks.json", "claude-haiku-4-5", "--remote", "--all", "--yes",
                          "--max-steps", "10"], cwd=d, capture_output=True, text=True)
    out = out.stdout + out.stderr
    job = re.search(r"https://hud.ai/jobs/([a-z0-9-]+)", out).group(1)
    traces = json.loads(subprocess.run([hud, "jobs", "get", job, "--json"], capture_output=True, text=True).stdout)
    got = {t["task_slug"].removeprefix("check-"): (t["status"], t["reward"], t.get("error")) for t in traces}
    print(json.dumps({"job": job, "results": got}, indent=1))
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--hud", default="/Users/anilkumar/privacybench_v2/.venv/bin/hud")
    ap.add_argument("--skip-check", action="store_true")
    a = ap.parse_args()
    name = f"write-verifier-{a.task}"
    real = LOCAL / "verifier-bench" / name
    real.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):                 # large pools (0.4 to 0.7 GB) over one SSH link: resume, retry
        r = subprocess.run(["rsync", "-a", "--delete", "--partial", "--timeout=120",
                            f"{SERVER}:/root/vbench/hud_tasks/{name}", str(real.parent) + "/"])
        if r.returncode == 0:
            break
    else:
        raise SystemExit("copy from the server failed four times")

    if not a.skip_check:
        check = LOCAL / "verifier-bench-check" / name
        check.parent.mkdir(parents=True, exist_ok=True)
        sh(["rsync", "-a", "--delete", str(real) + "/", str(check) + "/"])
        vb = check / "environment" / "vb_checks"
        vb.mkdir(parents=True, exist_ok=True)
        shutil.copytree(real / "solution" / "official_tests", vb / "official", dirs_exist_ok=True)
        for v in ("always_pass", "always_fail", "cheat"):
            shutil.copytree(REPO / "templates" / "hud" / "checks" / v, vb / v, dirs_exist_ok=True)
        df = check / "environment" / "Dockerfile"
        df.write_text(df.read_text().rstrip() + "\n\nCOPY vb_checks /opt/vb_checks\n")
        d, row = adapt_one(check)
        deploy(a.hud, d)
        got = run_checks(a.hud, d, row)
        bad = {v: g for v, g in got.items() if g[0] != "completed" or g[1] != EXPECT.get(v, 0.0)}
        if bad:
            print(f"CHECK FAILED on HUD: {bad}")
            return 1

    d, row = adapt_one(real)
    deploy(a.hud, d)
    print(subprocess.run([a.hud, "sync", "tasks", TASKSET, "tasks.json", "--yes"], cwd=d, capture_output=True,
                         text=True).stdout[-600:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
