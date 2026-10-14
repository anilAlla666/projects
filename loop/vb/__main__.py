"""The verifier, from the command line. Everything runs on a Linux machine with Docker (this one, or VB_HOST's).

    python -m vb setup     --task T             install a task bundle (bundles/T) and build its images
    python -m vb controls  --task T             official tests 1.00; always pass, always fail, cheat, no tests 0
    python -m vb score     --task T --grader D  one tests/ folder scored on the task's hidden pool of real attempts
    python -m vb reproduce [--task T]           every saved step-3 grader, before and after the loop, rescored and
                                                compared with the recorded numbers (free: Docker only, no API key)
    python -m vb requirements --task T [--adapter A]   the trained requirement model's list for a task (needs a GPU)
    python -m vb verify    --task T --grader D [--requirements F]   the verify loop on an agent's grader, with the
                                                requirement model's list if given (needs ANTHROPIC_API_KEY)
    python -m vb write     --task T [--cell base|cross]   an agent writes a grader to verify (needs ANTHROPIC_API_KEY)
    python -m vb make      --task T [--grader D] [--cap-usd 15]   the product, end to end: an agent writes the grader
                                                (or give one), solver agents produce candidate results, the verify loop
                                                (judge v1) repairs the grader, and a report says what each check rests on
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import host
from .util import ROOT

BUNDLES = ROOT / "bundles"
STEP3 = ROOT / "results" / "step3"
CHECKS = ROOT / "templates" / "hud" / "checks"
DOCKER_OPTS = "--cap-add SYS_ADMIN --security-opt seccomp=unconfined --security-opt apparmor=unconfined"


def task_dir(task: str) -> str:
    return host.data("hud_tasks", f"write-verifier-{task}")


def setup(task: str) -> dict:
    """Copy the bundle into the data directory, build the official tests' image (the grading base) and the task's
    grader image on top of it. The base tag is the content hash of the official tests, as the grader's FROM line
    expects (vb.regrade.build_image)."""
    b = BUNDLES / task
    if not b.is_dir():
        raise SystemExit(f"no bundle for {task}: bundled tasks are {sorted(p.name for p in BUNDLES.iterdir())}; "
                         "rebuild others from the leaderboard with vb.pool, vb.hudtask and vb.candidates")
    host.put(b / "hud_task", task_dir(task))
    host.put(b / "candidates", host.data("candidates", task, "files"))
    host.put(b / "base", host.data("bases", task))
    r = host.run(f"cd {host.REPO} && {host.PYTHON} -c \"from pathlib import Path; from vb.regrade import build_image; "
                 f"print(build_image(Path('{host.data('bases', task)}')))\"", timeout=3600)
    base = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    want = host.run(f"grep -m1 ^FROM {task_dir(task)}/tests/Dockerfile").stdout.split()[-1]
    if base != want:
        raise SystemExit(f"{task}: built base {base!r} does not match the grader's {want!r}\n{r.stderr[-800:]}")
    g = host.run(f"docker build -q -t vbhud-{task} {task_dir(task)}/tests", timeout=3600)
    if g.returncode:
        raise SystemExit(f"{task}: grader image failed to build\n{g.stderr[-800:]}")
    return {"task": task, "base_image": base, "grader_image": f"vbhud-{task}"}


def score(task: str, tests: str | None) -> dict:
    """tests: a folder on the Docker machine, or None for no tests at all."""
    mount = f"-v {tests}:/verifier_out/tests:ro" if tests else ""
    r = host.run(f"docker run --rm {DOCKER_OPTS} {mount} vbhud-{task} bash /tests/test.sh 2>/dev/null | tail -1",
                 timeout=10800)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"error": (r.stdout + r.stderr)[-800:]}


def staged(local: Path, name: str) -> str:
    """A local tests/ folder copied to the Docker machine (a no-op copy when it is this machine)."""
    dest = host.data("staged", name)
    host.put(local, dest)
    return dest


def controls(task: str) -> dict:
    out = {"official": score(task, f"{task_dir(task)}/solution/official_tests")}
    for c in ("always_pass", "always_fail", "cheat"):
        out[c] = score(task, staged(CHECKS / c, f"control-{c}"))
    out["none"] = score(task, None)
    ok = (out["official"].get("balanced_accuracy") == 1.0
          and all(out[c].get("reward") == 0.0 for c in ("always_pass", "always_fail", "cheat", "none")))
    return {"task": task, "controls_hold": ok,
            **{k: {x: v.get(x) for x in ("balanced_accuracy", "tpr", "tnr", "reward", "error")} for k, v in out.items()}}


def reproduce(only: str | None) -> dict:
    expected = json.loads((STEP3 / "expected.json").read_text())
    rows, ok = [], True
    for loop, exp in expected.items():
        if only and exp["task"] != only or not (BUNDLES / exp["task"]).is_dir():
            continue
        got = {}
        for side in ("before", "after"):
            s = score(exp["task"], staged(STEP3 / loop / side, f"{loop}-{side}"))
            got[side] = s.get("balanced_accuracy")
        match = all(abs((got[k] or -1) - exp[k]) < 1e-6 for k in ("before", "after"))
        ok &= match
        rows.append({"loop": loop, "task": exp["task"], "recorded": [exp["before"], exp["after"]],
                     "rescored": [got["before"], got["after"]], "match": match})
        print(json.dumps(rows[-1]), flush=True)
    return {"all_match": ok, "rows": len(rows)}


def requirements(task: str, adapter: Path, out: Path) -> dict:
    """The task's text, from its bundle (the <task-instruction> block a writer sees), through the trained model on
    this machine's GPU (gpu/gen_req.py). Writes <out>/<task>.requirements.json."""
    import re
    import subprocess
    from .reqdata import STUDENT, parse
    text = (BUNDLES / task / "hud_task" / "instruction.md").read_text()
    m = re.search(r"<task-instruction>\n?(.*?)\n?</task-instruction>", text, re.S)
    out.mkdir(parents=True, exist_ok=True)
    prompts, gen = out / f"{task}.prompt.jsonl", out / f"{task}.generated.jsonl"
    prompts.write_text(json.dumps({"id": task, "prompt": STUDENT + "<task>\n" + (m.group(1) if m else text).strip() + "\n</task>"}) + "\n")
    subprocess.run([sys.executable, str(ROOT / "gpu" / "gen_req.py"), str(prompts), str(gen), "--adapter", str(adapter)], check=True)
    reqs = parse(json.loads(gen.read_text().splitlines()[0])["text"], salvage=True)
    if not reqs:
        raise SystemExit("the model's reply did not parse as a requirement list; see " + str(gen))
    dest = out / f"{task}.requirements.json"
    dest.write_text(json.dumps(reqs, indent=1, ensure_ascii=False))
    return {"task": task, "requirements": len(reqs), "file": str(dest)}


def make(task: str, grader: Path | None, out: Path, cap: float, rounds: int, loop_cap: float,
         requirements: list | None = None, judge: str = "v2") -> dict:
    """The product: grader in (or written by an agent), corrected grader and its report out. One spend cap over all."""
    from .solve import solve
    from .verify_loop import run
    from .writer_api import Spend, prepare, rollout
    spend, out = Spend(cap), out / task
    if grader is None:
        image, text = prepare(task, "base")
        w = rollout(task, "base", 0, image, text, out / "writer", spend)
        grader = out / "writer" / f"{task}__base__0" / "verifier_out" / "tests"
        if not (grader / "test.sh").is_file():
            raise SystemExit(f"the writer left no tests/test.sh ({w.get('stop')})")
    sol = solve(task, out / "solvers", spend)
    r = run(task, grader, rounds, out / "loop", spend, loop_cap=min(loop_cap, max(cap - spend.usd, 0.5)),
            requirements=requirements, tag="make", judge=judge)
    final = out / "loop" / r["loop"]
    return {"task": task, "usd": round(spend.usd, 2), "solvers": [{k: s.get(k) for k in ("letter", "model", "stop", "usd")}
            for s in sol], "grader_before": str(grader), "report": str(final / "report.md"),
            "grader_after": str(sorted((final / "server").glob("r*/tests"), key=lambda p: int(p.parent.name[1:]))[-1])
            if (final / "server").is_dir() else None,
            "scored_on_hidden_pool": {"before": r["before"].get("balanced_accuracy"),
                                      "after": r["after"].get("balanced_accuracy")}}


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m vb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["setup", "controls", "score", "reproduce", "requirements", "verify", "write",
                                        "make"])
    ap.add_argument("--cap-usd", type=float, default=15.0, help="make: most the whole run may spend, $")
    ap.add_argument("--judge", default="v2", choices=["v0", "v1", "v2", "v3"], help="make: the judge's rules (README)")
    ap.add_argument("--task")
    ap.add_argument("--grader", type=Path, help="a local tests/ folder")
    ap.add_argument("--cell", default="base", choices=["base", "cross"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--loop-cap", type=float, default=9.0, help="most this loop may spend, $")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "cli")
    ap.add_argument("--requirements", type=Path, help="a requirement list (JSON) from `requirements`")
    ap.add_argument("--adapter", type=Path, default=ROOT / "data" / "reqmodel" / "adapter",
                    help="the trained LoRA adapter (Qwen2.5-7B-Instruct base)")
    a = ap.parse_args()
    if a.command != "reproduce" and not a.task:
        ap.error("--task is required")
    if a.command == "setup":
        res = setup(a.task)
    elif a.command == "controls":
        res = controls(a.task)
    elif a.command == "score":
        res = score(a.task, staged(a.grader.resolve(), "score-" + a.grader.resolve().parent.name))
    elif a.command == "reproduce":
        res = reproduce(a.task)
    elif a.command == "requirements":
        res = requirements(a.task, a.adapter, a.out)
    elif a.command == "verify":
        from .verify_loop import run
        from .writer_api import Spend
        reqs = json.loads(a.requirements.read_text()) if a.requirements else None
        r = run(a.task, a.grader.resolve(), a.rounds, a.out, Spend(a.loop_cap), loop_cap=a.loop_cap,
                requirements=reqs, tag="req" if reqs else "")
        res = {k: r[k] for k in ("loop", "usd")} | {"before": r["before"].get("balanced_accuracy"),
                                                     "after": r["after"].get("balanced_accuracy")}
    elif a.command == "make":
        res = make(a.task, a.grader.resolve() if a.grader else None, a.out, a.cap_usd, a.rounds, a.loop_cap,
                   json.loads(a.requirements.read_text()) if a.requirements else None, a.judge)
    else:
        from .writer_api import Spend, prepare, rollout
        image, text = prepare(a.task, a.cell)
        r = rollout(a.task, a.cell, 0, image, text, a.out, Spend(a.loop_cap))
        res = {k: r.get(k) for k in ("task", "cell", "stop", "usd")} | {"grader": str(a.out / f"{a.task}__{a.cell}__0" / "verifier_out" / "tests"),
                                                                     "balanced_accuracy": r.get("grade", {}).get("balanced_accuracy")}
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
