"""Run the intent × goal experiment (docs/research_intent_goal.md) as HUD rollouts.

  base, intent   the published task environments; only the instruction differs
  goal, both     the same tasks rebuilt with the reference solution at /solution in the writer's machine

Every row is graded by the task's own hidden grader on HUD (reward = max(0, 2 * BA - 1); BA, TPR and TNR are in
the grader's summary). Runs where the HUD CLI is logged in.

  python tools/hud_experiment.py deploy-goal             # adapt and deploy the /solution variants
  python tools/hud_experiment.py rows                    # write the experiment rows
  python tools/hud_experiment.py run --group 3           # launch (prints the job id)
  python tools/hud_experiment.py collect <job id>        # per-rollout results -> runs/<name>/
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.hud_publish import LOCAL, SERVER, adapt_one, deploy          # noqa: E402
from vb.experiment import (CELLS, CROSS, DERIVE, DEV_TASKS, GOAL, INTENT, MUTANTS, PROPERTIES,  # noqa: E402
                           PURPOSE)

HUD = "/Users/anilkumar/privacybench_v2/.venv/bin/hud"
REPO = pathlib.Path(__file__).resolve().parents[1]
BASE = LOCAL / "verifier-bench"
GOALDS = LOCAL / "verifier-bench-goal"
CROSSDS = LOCAL / "verifier-bench-cross"
ROWS = LOCAL / "exp_rows.json"
MODEL = "claude-sonnet-5"


def fetch(task: str) -> pathlib.Path:
    name = f"write-verifier-{task}"
    subprocess.run(["rsync", "-a", "--delete", "--partial", "--timeout=120",
                    f"{SERVER}:/root/vbench/hud_tasks/{name}", str(BASE) + "/"], check=True)
    subprocess.run(["rsync", "-a", "--delete", "--partial", "--timeout=120",
                    f"{SERVER}:/root/tb4/terminal-bench/tasks/{task}/solution/", str(BASE / name / "solution_ref") + "/"],
                   check=True)
    return BASE / name


def deploy_goal(tasks: list[str]) -> None:
    for task in tasks:
        src = fetch(task)
        dst = GOALDS / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--delete", "--exclude", "solution_ref", str(src) + "/", str(dst) + "/"],
                       check=True)
        shutil.copytree(src / "solution_ref", dst / "environment" / "_solution", dirs_exist_ok=True)
        df = dst / "environment" / "Dockerfile"
        text = df.read_text()
        if "COPY _solution /solution" not in text:
            df.write_text(text.rstrip() + "\n\nCOPY _solution /solution\n")
        d, _ = adapt_one(dst)
        deploy(HUD, d)
        print("deployed", task, d.name, flush=True)


def deploy_cross(tasks: list[str]) -> None:
    """The published task with five other models' attempts at /candidates (vb/candidates.py)."""
    for task in tasks:
        src = fetch(task)
        dst = CROSSDS / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--delete", "--exclude", "solution_ref", str(src) + "/", str(dst) + "/"],
                       check=True)
        subprocess.run(["rsync", "-a", "--delete", "--partial", "--timeout=120",
                        f"{SERVER}:/root/vbench/candidates/{task}/files/", str(dst / "environment" / "_candidates") + "/"],
                       check=True)
        df = dst / "environment" / "Dockerfile"
        text = df.read_text()
        if "COPY _candidates /candidates" not in text:
            df.write_text(text.rstrip() + "\n\nCOPY _candidates /candidates\n")
        d, _ = adapt_one(dst)
        deploy(HUD, d)
        print("deployed", task, d.name, flush=True)


def instruction(base: str, cell: str) -> str:
    extra = "".join({"intent": INTENT, "purpose": PURPOSE, "goal": GOAL, "derive": DERIVE, "mutants": MUTANTS,
                     "cross": CROSS, "properties": PROPERTIES}[p]
                    for p in CELLS[cell])
    if not extra:
        return base
    anchor = "## What to write"
    assert anchor in base
    return base.replace(anchor, extra.strip() + "\n\n" + anchor, 1)


def adapted_row(ds: pathlib.Path, task: str) -> dict:
    """The adapted row for a task, with its compose files addressed absolutely: the CLI resolves them relative to
    the rows file, and one rows file here spans eight environments."""
    found = []
    for d in (ds / ".hud-adapt").glob("*"):
        for x in json.loads((d / "tasks.json").read_text()):
            if x["args"]["task"]["id"] == f"write-verifier-{task}":
                found.append((d.stat().st_mtime, d, x))
    _, d, row = max(found, key=lambda f: f[0])
    comp = row["runtime_config"]["compose"]
    comp["document"] = str(d / comp["document"])
    comp["root"] = str((d / comp.get("root", ".")).resolve())
    return row


def rows(tasks: list[str], cells: list[str] | None = None, out_path: pathlib.Path = ROWS) -> None:
    out = []
    for task in tasks:
        for cell, parts in CELLS.items():
            if cells and cell not in cells:
                continue
            ds = GOALDS if "goal" in parts else CROSSDS if "cross" in parts else BASE
            r = json.loads(json.dumps(adapted_row(ds, task)))
            r["args"]["instruction"] = instruction(r["args"]["instruction"], cell)
            r["slug"] = f"{cell}__{task}"
            r["verifier"]["slug"] = f"{cell}__{task}:verify"
            out.append(r)
    out_path.write_text(json.dumps(out, indent=1))
    print(len(out), "rows ->", out_path)


def run(group: int, max_steps: int = 200, rows_file: pathlib.Path = ROWS) -> None:
    # max_tokens: HUD's agent defaults to 16,384 output tokens per turn; writers put a whole test file in one turn,
    # got cut off mid-reply with no tool call, and the rollout ended with nothing written (job 891ab733).
    p = subprocess.run([HUD, "eval", str(rows_file.name), MODEL, "--remote", "--all", "--yes", "--group-size", str(group),
                        "--max-steps", str(max_steps), "--config", "max_tokens=64000"],
                       cwd=rows_file.parent, capture_output=True, text=True)
    out = p.stdout + p.stderr
    job = re.search(r"https://hud.ai/jobs/([a-z0-9-]+)", out)
    print(job.group(1) if job else out[-2000:])


def events_of(trace_id: str) -> list[dict]:
    """Every event of a trace. `hud trace get` returns only the first 100, and the grader's summary comes last."""
    import hud.cli.trace as tr
    c, events, since = tr.PlatformClient.from_settings(), [], None
    while True:
        d = c.get(f"/trace/{trace_id}/events", params={"limit": 200, **({"since_seq": since} if since else {})})
        events += d["events"]
        if not d.get("has_more"):
            return events
        since = d["next_seq"]


def summary_of(trace_id: str) -> dict:
    """The hidden grader's JSON summary (it prints one line: BA, TPR, TNR, verdicts, ...)."""
    for e in reversed(events_of(trace_id)):
        if e.get("name") == "verify" and e.get("kind") == "scenario_evaluate":
            for line in (((e.get("result") or {}).get("info") or {}).get("stdout") or "").splitlines():
                if line.startswith("{"):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        pass
    return {}


def collect(job: str) -> None:
    traces = json.loads(subprocess.run([HUD, "jobs", "get", job, "--json", "-n", "500"], capture_output=True, text=True).stdout)
    out_dir = REPO / "runs" / f"hud_exp_{job[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for t in traces:
        cell, _, task = t["task_slug"].partition("__")
        s = summary_of(t["id"]) if t["status"] == "completed" else {}
        recs.append({"trace": t["id"], "cell": cell, "task": task, "status": t["status"], "reward": t["reward"],
                     "error": (t.get("error") or "")[:300], **{k: s.get(k) for k in (
                         "balanced_accuracy", "tpr", "tnr", "accepted_correct", "rejected_wrong", "verdicts", "labels",
                         "note", "timeouts", "seconds")}})
    (out_dir / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    print(out_dir, len(recs), "rollouts;", sum(r["status"] == "completed" for r in recs), "completed")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "deploy-cross":
        deploy_cross(sys.argv[2].split(",") if len(sys.argv) > 2 else DEV_TASKS)
    elif cmd == "deploy-goal":
        deploy_goal(sys.argv[2].split(",") if len(sys.argv) > 2 else DEV_TASKS)
    elif cmd == "rows":
        rows(sys.argv[2].split(",") if len(sys.argv) > 2 else DEV_TASKS,
             sys.argv[3].split(",") if len(sys.argv) > 3 else None,
             LOCAL / sys.argv[4] if len(sys.argv) > 4 else ROWS)
    elif cmd == "run":
        run(int(sys.argv[sys.argv.index("--group") + 1]) if "--group" in sys.argv else 3,
            rows_file=LOCAL / sys.argv[sys.argv.index("--rows") + 1] if "--rows" in sys.argv else ROWS)
    elif cmd == "collect":
        collect(sys.argv[2])
