"""Step 18: score a set of test inputs written for a CodeContests case (tools/cc_build.py) on its hidden pool.

The reference solution gives the expected output for each input; an input on which the reference fails or times out
is invalid and dropped. A hidden solution is accepted if its output matches the reference's on every valid input
(whitespace-insensitive). Balanced accuracy = (accepted correct / correct + rejected incorrect / incorrect) / 2:
no valid input accepts everything and scores 0.5, the same as the loop's measure.

Also: the prompt the model sees and the parser for its answer, so training, scoring and testing share one definition.
"""
from __future__ import annotations

import json
import re

from cc_build import norm, run

MAX_TESTS, MAX_INPUT_CHARS = 10, 20000

INSTRUCTION = """You are checking a grader for a programming problem. A grader here is a set of test inputs: a submitted
program passes if its output matches the reference solution's output on every input. Write test inputs so that every
correct program passes and every incorrect program fails at least one input.

Below are the problem, the reference solution (it is correct), and two submitted programs. Each submitted program may
be correct or incorrect; you are not told which. Your tests will also be run on other submissions you cannot see, so
target the mistakes a wrong program for this problem would make (edge cases, limits, special structure), not only
these two.

Rules: every input must be valid for the problem statement (the reference must run on it). At most {n} inputs, each
under {c} characters. Answer with the inputs as a JSON list of strings between <tests> and </tests>."""


def prompt(case: dict) -> str:
    vis = "\n\n".join(f"### Submitted program {i + 1}\n```python\n{s}\n```" for i, s in enumerate(case["visible"]))
    return (INSTRUCTION.format(n=MAX_TESTS, c=MAX_INPUT_CHARS) + f"\n\n## Problem\n{case['description']}\n\n"
            f"## Reference solution\n```python\n{case['reference']}\n```\n\n{vis}")


def parse(text: str) -> list[str] | None:
    m = re.findall(r"<tests>(.*?)</tests>", text or "", re.S)
    if not m:
        return None
    try:
        xs = json.loads(m[-1].strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(xs, list):
        return None
    xs = [x for x in xs if isinstance(x, str) and 0 < len(x) <= MAX_INPUT_CHARS][:MAX_TESTS]
    return [x if x.endswith("\n") else x + "\n" for x in xs]


def score(case: dict, inputs: list[str] | None) -> dict:
    if not inputs:
        return {"ba": 0.5, "tpr": 1.0, "tnr": 0.0, "valid": 0, "parsed": inputs is not None}
    sols = {"ref": case["reference"]} | {f"c{i}": s for i, s in enumerate(case["hidden_correct"])} \
        | {f"w{i}": s for i, s in enumerate(case["hidden_incorrect"])}
    res = run(sols, inputs, timeout=2.0)
    keep = [i for i, o in enumerate(res["ref"]) if o is not None]
    exp = [norm(res["ref"][i]) for i in keep]

    def accepted(k: str) -> bool:
        return all(norm(res[k][i]) == e for i, e in zip(keep, exp))
    c = [accepted(k) for k in sols if k[0] == "c"]
    w = [accepted(k) for k in sols if k[0] == "w"]
    tpr, tnr = sum(c) / len(c), 1 - sum(w) / len(w)
    return {"ba": round((tpr + tnr) / 2, 4), "tpr": round(tpr, 4), "tnr": round(tnr, 4), "valid": len(keep),
            "parsed": True}
