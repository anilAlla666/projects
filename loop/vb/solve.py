"""The product's own evidence: solver agents attempt the task, so the verify loop has candidate results on a task
with no leaderboard. Each solver works in the task's own starting machine (the writer's base-cell image: the task's
environment, no network), from the task's own text; its results at the task's artifact paths become one candidate,
in the layout the loop reads (<DATA>/candidates/<task>/files/<letter>/<artifact path>). Solvers are different models
so that they are less likely to share one mistake; their results are unverified evidence, never labels.

    VB_HOST=root@... python -m vb.solve --task T --cap-usd 6
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
from pathlib import Path

from . import host
from .writer_api import Spend, agent_loop, prepare, ssh

SOLVERS = [("A", "claude-haiku-5-5"), ("B", "claude-sonnet-5"), ("C", "claude-sonnet-5-5")]
SOLVER_CAP_USD, SOLVER_WALL_S, SOLVER_TURNS = 1.5, 1800, 80
BRIEF = ("Do the task below on this machine. Work until the task's results are complete and saved where the task "
         "says. Nobody will check your work before it is graded.\n\n<task>\n{task}\n</task>\n")


def task_text(task: str) -> str:
    """The task as its solver sees it: the <task-instruction> block of the built task's instruction."""
    t = ssh(f"cat {host.DATA_DIR}/hud_tasks/write-verifier-{task}/instruction.md").stdout
    m = re.search(r"<task-instruction>\n?(.*?)\n?</task-instruction>", t, re.S)
    if not m:
        raise RuntimeError(f"{task}: no <task-instruction> block in the built task")
    return m.group(1).strip()


def artifacts(task: str) -> list[str]:
    cfg = ssh(f"cat {host.DATA_DIR}/hud_tasks/write-verifier-{task}/tests/grader/config.json").stdout
    return [a if isinstance(a, str) else a["source"] for a in json.loads(cfg)["artifacts"]]


def solve_one(task: str, letter: str, model: str, image: str, text: str, arts: list[str], out: Path,
              spend: Spend) -> dict:
    rid = f"{task}__solver_{letter}"
    d = out / rid
    if (d / "result.json").is_file():
        return json.loads((d / "result.json").read_text())
    d.mkdir(parents=True, exist_ok=True)
    if spend.usd >= spend.cap:
        return {"letter": letter, "model": model, "skipped": "spend cap reached"}
    container = f"vbs-{rid}"[:120]
    ssh(f"docker rm -f {container} >/dev/null 2>&1; docker run -d --name {container} --network none --cpus 2.5 "
        f"--memory 5g --entrypoint sleep {image} infinity")
    try:
        r = agent_loop(container, BRIEF.format(task=text), d / "transcript.jsonl", spend, model=model,
                       max_turns=SOLVER_TURNS, wall_s=SOLVER_WALL_S, cap_usd=SOLVER_CAP_USD)
    finally:
        dest = f"{host.DATA_DIR}/candidates/{task}/files/{letter}"
        cp = "; ".join(f"mkdir -p {dest}{a.rsplit('/', 1)[0] or '/'} && docker cp {container}:{a} {dest}{a} 2>/dev/null"
                       for a in arts)
        ssh(f"rm -rf {dest}; mkdir -p {dest}; {cp}; docker rm -f {container} >/dev/null 2>&1")
    got = ssh(f"cd {host.DATA_DIR}/candidates/{task}/files/{letter} && find . -type f | head -50").stdout.split()
    res = {"letter": letter, "model": model, **r, "files": got}
    (d / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"task": task, **{k: res[k] for k in ("letter", "model", "stop", "turns", "usd")},
                      "files": len(got)}), flush=True)
    return res


def solve(task: str, out: Path, spend: Spend) -> list[dict]:
    """Every solver on the task, at once. Replaces any earlier candidates for the task."""
    image, _ = prepare(task, "base")
    text, arts = task_text(task), artifacts(task)
    keep = "|".join(letter for letter, _ in SOLVERS)  # results of other origins (leaderboard letters) are removed
    ssh(f"mkdir -p {host.DATA_DIR}/candidates/{task}/files && cd {host.DATA_DIR}/candidates/{task}/files && "
        f"ls | grep -Ev '^({keep})$' | xargs -r rm -rf")
    with cf.ThreadPoolExecutor(len(SOLVERS)) as ex:
        return list(ex.map(lambda s: solve_one(task, s[0], s[1], image, text, arts, out, spend), SOLVERS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--cap-usd", type=float, default=6.0)
    ap.add_argument("--out", type=Path, default=Path("runs/solve"))
    a = ap.parse_args()
    print(json.dumps(solve(a.task, a.out, Spend(a.cap_usd)), indent=1, default=str))
