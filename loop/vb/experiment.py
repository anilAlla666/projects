"""The intent × goal experiment (docs/research_intent_goal.md).

Four cells, each a generic addition to the HUD writer instruction:
  base    the instruction as published in the HUD taskset
  intent  + list the ways a solver could fake a correct-looking result, one check per item
  goal    + the task's reference solution is on the machine and can be run
  both    intent and goal

Writer tasks are built from the HUD task directories (same environment, same instruction, same grading contract:
the tests run in the official grading image), with only the model gateway reachable, so Harbor can run Claude Code
on the server. Written graders are scored by the HUD grader itself, locally (vb.scorelocal).
"""
from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from pathlib import Path

from .hudtask import HUD_TASKS
from .util import DATA, TB_ROOT, load_env, new_run_dir, write_json
from .writer import ALL_WRITERS, GATEWAY_HOST, WRITERS, dump_toml, run_writer

EXP_TASKS = DATA / "exp_tasks"
EXP_GRADERS = DATA / "exp_graders"
DEV_TASKS = ["production-planning", "uefi-bootkit", "hof-topology-interpenetration", "pretrain-shard-corruption"]
CELLS = {"base": (), "intent": ("intent",), "goal": ("goal",), "both": ("intent", "goal"),
         "purpose": ("purpose",), "purpose_goal": ("purpose", "goal"),
         "derive": ("derive",), "derive_mutants": ("derive", "mutants"), "cross": ("cross",),
         "properties": ("properties",)}
WRITER = "sonnet-claude-code"
KWARGS = ["disallowed_tools=WebFetch,WebSearch", "max_budget_usd=5"]
TIMEOUT_S = 3600.0

INTENT = """
## Before you write any check: how could a wrong attempt look right?

Write `/verifier_out/threats.md` first. List the ways a solver could produce a result that looks correct but is
not: shortcuts, approximations or stand-ins, fabricated or hard-coded results, partial work presented as complete,
and anything the task text forbids or requires about *how* the work is done. For each item, say what your tests
check to catch it. Then write your tests: one check per item, plus checks that the result is actually correct.
Do not reject anything the task allows; a correct solution may differ from yours in every way the task leaves open.
"""

PURPOSE = """
## Before you write any check: the task's intent

Write `/verifier_out/intent.md` first. Answer each of these about the task, in your own words:

1. **Purpose.** What the task's author wants to have happened when the task is done, what the result will be used
   for, and what skill the task tests. Separate the goal from the evidence of it: a metric, a file or an output
   format is evidence that the goal was met, not the goal itself.
2. **What must hold.** The requirements a correct result cannot do without, and for each, why it is there and what
   it protects against.
3. **What may vary.** Everything a correct solution is free to do differently: approach, tools, formatting, ordering,
   naming, extra output, and anything else the task leaves open.
4. **Unstated norms.** What an expert doing this job would take for granted even though the task does not say it.
5. **Priorities.** Which requirements are the point of the task and which are incidental; which failures must make
   an attempt fail.
6. **Ambiguities.** Where the text could be read more than one way, and how the author most likely meant it. Be
   lenient where the author would not care.
7. **What your grader will teach.** An agent will be trained to pass your tests. Which shortcuts would pass a naive
   grader without achieving the purpose, and how your tests make sure they do not pay.

Then write your tests from this document: every check must trace to an item in sections 1, 2, 4 or 7, and no check
may constrain anything section 3 says may vary.
"""

DERIVE = """
## Before you write any check: derive the goal independently, twice

Your checks will depend on concrete things: expected values, how a quantity is computed, what counts as valid.
Do not take them from a single reading of the task. Derive them independently, twice:

1. Solve the task yourself twice, in two separate scratch copies of the machine's files, with two different
   approaches (a different method, algorithm or tool; do not reuse code or intermediate results between them).
2. Write `/verifier_out/derivation.md`: every concrete value, formula, threshold or interpretation your checks will
   rely on, how each of your two solutions arrived at it, and whether they agree.
3. Where the task text can be read more than one way, list each reading, work out what each implies, and use the
   machine's own data, documentation and conventions to decide which one the author meant. If you cannot decide,
   your checks must accept every reading the text allows.
4. Assert only what your two derivations agree on. Where they disagree and you cannot resolve it, do not assert that
   value; check a weaker property that holds under every plausible reading instead.
5. Your tests must accept both of your solutions.
"""

CROSS = """
## Before you write any check: other agents' attempts

`/candidates/A` to `/candidates/E` hold the results of five other agents' attempts at this task, each at the same
paths a solver's results would have. They were produced independently, by different models. They are unverified:
some are correct and some are not, and you are not told which. Use them as independent evidence about what a
correct result looks like:

1. Solve the task yourself as well, in a scratch copy of the machine's files.
2. Write `/verifier_out/derivation.md`: for every concrete value, formula, threshold or interpretation your checks
   will rely on, compare your own result with the five attempts. Where independent attempts converge, that is
   evidence; where they split, find out why from the task text and the machine's own data, documentation and
   conventions, and decide which reading the author meant. Never trust a result only because it matches your own,
   and never follow a majority you can show is wrong.
3. Assert only what you can justify this way. Where you cannot resolve a disagreement, check a weaker property that
   holds under every plausible reading.
4. Your tests must accept every attempt you conclude is correct and reject every attempt you conclude is not.
"""

PROPERTIES = """
## Define correct as properties, and check without solving

Your tests must decide correctness by checking properties that every correct result has, not by comparing the
attempt with an answer you computed yourself.

1. Write `/verifier_out/properties.md` first: the properties a correct result must have. These are the rules the
   task states, relations the result must have with the machine's own data (for example, that it is consistent with
   the inputs it was produced from), invariants that must hold, and what must not have changed. For each, say how
   your tests check it directly from the attempt and the machine's data.
2. Prefer checks that verify without solving: confirm that the attempt's result satisfies the requirements, even
   where producing such a result would be hard. Do not compute your own expected answer and compare against it.
   Where the task has exactly one correct answer, check the attempt's answer against the data it must be consistent
   with, not against a value you worked out.
3. Never constrain what the task leaves open: any result that has every property must pass.
4. A near miss (a result that looks right but breaks one requirement, especially the point of the task) must break
   at least one property. For each kind of near miss you can think of, say which property it breaks.
"""

MUTANTS = """
## Then: prove your tests separate correct from almost-correct

Take one of your solutions and make near misses from it: each breaks exactly one requirement of the task, including
the one that is the point of the task (not only easy rules such as a file name or a format). Your tests must reject
every near miss and accept both of your solutions. If they do not, fix the tests, not the near misses. Write what
you ran and what happened to `/verifier_out/validation.md`.
"""

GOAL = """
## A reference solution

The task's author wrote a solution. It is in `/solution`; `bash /solution/solve.sh` produces a correct result on
this machine (it changes the files under `/app`, so copy `/app` somewhere first if you want the starting state).
Use it to see what a correct result looks like, and check that your tests accept it. Other correct solutions may
differ from it wherever the task allows, and your tests must accept those too.
"""


def make_exp_task(task: str, cell: str, out_root: Path = EXP_TASKS) -> Path:
    src = HUD_TASKS / f"write-verifier-{task}"
    out = Path(out_root) / cell / f"write-verifier-{task}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(src / "environment", out / "environment")
    t = tomllib.loads((src / "task.toml").read_text(encoding="utf-8"))
    env = {k: v for k, v in t["environment"].items() if k != "docker_image"}    # build, so /solution can be added
    parts = CELLS[cell]
    if "goal" in parts:
        shutil.copytree(TB_ROOT / task / "solution", out / "environment" / "_solution")
        df = out / "environment" / "Dockerfile"
        df.write_text(df.read_text(encoding="utf-8").rstrip() + "\n\nCOPY _solution /solution\n", encoding="utf-8")
    text = (src / "instruction.md").read_text(encoding="utf-8")
    anchor = "## What to write"
    extra = "".join({"intent": INTENT, "goal": GOAL}[p] for p in parts)
    if extra:
        assert anchor in text
        text = text.replace(anchor, extra.strip() + "\n\n" + anchor, 1)
    (out / "instruction.md").write_text(text, encoding="utf-8")
    doc = {"schema_version": "2.0", "artifacts": ["/verifier_out/"],
           "task": {"name": f"vbench-exp-{cell}/write-verifier-{task}", "description": ""},
           "metadata": {"source_task": task, "cell": cell},
           "agent": {"timeout_sec": TIMEOUT_S, "network_mode": "allowlist", "allowed_hosts": [GATEWAY_HOST]},
           "verifier": {"timeout_sec": 60.0},
           "environment": env}
    (out / "task.toml").write_text(dump_toml(doc), encoding="utf-8")
    (out / "tests").mkdir()
    (out / "tests" / "test.sh").write_text(
        "#!/bin/bash\nmkdir -p /logs/verifier\n"
        "if [ -s /verifier_out/tests/test.sh ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt\n",
        encoding="utf-8")
    (out / "tests" / "test.sh").chmod(0o755)
    return out


def collect(job_dir: Path, cell: str) -> list[dict]:
    rows = []
    for rj in sorted(Path(job_dir).glob("*/result.json")):
        r = json.loads(rj.read_text(encoding="utf-8"))
        task = r["task_name"].split("/")[-1].removeprefix("write-verifier-")
        out = rj.parent / "artifacts" / "verifier_out"
        dest = EXP_GRADERS / cell / task / rj.parent.name
        ok = (out / "tests" / "test.sh").is_file()
        if dest.exists():
            shutil.rmtree(dest)
        if out.is_dir():
            shutil.copytree(out, dest)
        cost = ((r.get("agent_result") or {}).get("cost_usd"))
        rows.append({"cell": cell, "task": task, "trial": rj.parent.name, "ok": ok, "dest": str(dest),
                     "cost_usd": cost, "exception": (r.get("exception_info") or {}).get("exception_type")})
    write_json(EXP_GRADERS / cell / f"collect_{Path(job_dir).name}.json", rows)
    return rows


def run_cell(cell: str, tasks: list[str], attempts: int, n_concurrent: int = 3) -> list[dict]:
    load_env()
    tasks_dir = EXP_TASKS / cell
    for t in tasks:
        make_exp_task(t, cell)
    w = {**WRITERS[WRITER], "kwargs": KWARGS}
    ALL_WRITERS[f"{WRITER}-exp"] = w
    run_dir = new_run_dir(f"exp_{cell}")
    job = run_writer(f"{WRITER}-exp", tasks_dir, run_dir, attempts, n_concurrent=n_concurrent,
                     include=[f"write-verifier-{t}" for t in tasks])
    return collect(job, cell)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--tasks", default=",".join(DEV_TASKS))
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--make-only", action="store_true")
    a = ap.parse_args()
    tasks = [t for t in a.tasks.split(",") if t]
    if a.make_only:
        for t in tasks:
            print(make_exp_task(t, a.cell))
    else:
        print(json.dumps(run_cell(a.cell, tasks, a.attempts), indent=1))
