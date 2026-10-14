"""Training data for the requirement model: task in, the requirements a correct grader must check out (how strictly
each is checked, and a near miss that breaks only it).

The targets are written by a teacher model that sees what the student never will: the official tests (Terminal-Bench)
or the maintainers' test patch (SWE-rebench), and, for Terminal-Bench, real wrong attempts that broke exactly one
official test (vb.requirements). The student learns to produce them from the task text alone. Only training tasks
are used (configs/model_split.json); no evaluation task's tests are ever read here.

    python -m vb.reqdata --swe 700 --cap-usd 25 --out data/reqmodel      # runs where ANTHROPIC_API_KEY is set
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import random
import threading
from pathlib import Path

from .util import ROOT

SRC = ROOT / "data" / "train_src"
TEACHER_MODEL = "claude-sonnet-5-5"
PRICE_IN, PRICE_OUT = 3.0, 15.0
STRICTNESS = {"exact", "tolerance", "threshold", "presence", "absence", "format", "behaviour"}
FIELDS = ("requirement", "source", "strictness", "check", "near_miss")

STUDENT = ("Before any grader is written for the task below, list the requirements a correct grader must check, how "
           "strictly each is checked, and for each a near miss: a result that is correct except for that one "
           "requirement. Reply with only a JSON array of objects with the keys requirement, source, strictness, check "
           "and near_miss. strictness is one of exact, tolerance, threshold, presence, absence, format, behaviour.\n\n")

TEACHER = """You are given a task as its solver saw it, and the official tests that decide whether a solution is correct. \
The solver never saw the tests.{extra} Write the requirements a correct grader for this task must check, as a JSON array.

Use the official tests as ground truth for WHAT is checked and HOW STRICTLY, but write each requirement as something a \
careful grader-writer could derive from the task text and its environment, without the tests.

Each element is an object:
- "requirement": one specific sentence (name the files, fields, values or functions the way the task does)
- "source": the shortest quote from the task text it follows from, or "" if the task does not state it
- "strictness": "exact" (must equal exactly, no tolerance), "tolerance" (numeric closeness within a bound; give the \
bound if known), "threshold" (a score or metric must reach a bound, such as near the best achievable), "presence" \
(must exist or be produced), "absence" (must not exist or must not be done), "format" (structure, schema, naming) or \
"behaviour" (observable behaviour on some input)
- "check": how a grader verifies it from the result, in one sentence
- "near_miss": a concrete result that meets every other requirement and breaks only this one, in the subtlest way a \
real solver might, in one sentence

Order by importance. Include every requirement the official tests check and nothing they do not. Reply with only the \
JSON array.

<task>
{task}
</task>

<official_tests>
{tests}
</official_tests>
{near}"""


def tb_examples() -> list[dict]:
    out = []
    for d in sorted((SRC / "tb_harvest").iterdir()):
        tj = d / "task.json"
        if not tj.is_file():
            continue
        task = json.loads(tj.read_text())
        tests = "\n\n".join(f"# {p}\n{s}" for p, s in task["official_tests"].items() if not p.endswith(("Dockerfile",)))[:40000]
        near = []
        for row in (json.loads(l) for l in (d / "pairs.jsonl").read_text().splitlines() if l.strip()):
            if not row["near_miss"]:
                continue
            att = SRC / "tb_attempts" / d.name / "attempts" / f"{row['trial_id']}.json"
            msg = ""
            if att.is_file():
                a = json.loads(att.read_text())
                for t in a["tests"]:
                    if t.get("status") not in ("passed", "skipped"):
                        tr = [l for l in (t.get("trace") or "").splitlines() if l.strip().startswith(("E ", ">", "assert"))]
                        msg = " | ".join(tr[-4:])[:400] or (t.get("message") or "")[:200]
                        break
            line = f"- broke {row['broken'][0]}: {msg}".strip()
            if line not in near:
                near.append(line)
            if len(near) >= 6:
                break
        nb = ("\n<real_near_misses>\nWrong attempts by real agents that failed exactly one official test:\n"
              + "\n".join(near) + "\n</real_near_misses>\n") if near else ""
        out.append({"id": f"tb:{d.name}", "task": task["instruction"].strip(), "tests": tests,
                    "extra": " You also see real wrong attempts that broke exactly one official test." if near else "",
                    "near": nb})
    return out


def swe_examples(n: int, seed: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in (SRC / "swe_train.jsonl").read_text().splitlines()]
    random.Random(seed).shuffle(rows)
    out, repos = [], {}
    for r in rows:
        repo = r["id"].rsplit("-", 1)[0]
        if repos.get(repo, 0) >= 2:  # spread over repositories
            continue
        cut = r["prompt"].find("\n\nWrite the tests that decide")
        task = (r["prompt"][:cut] if cut > 0 else r["prompt"]).strip()
        if len(task) > 14000 or len(r["completion"]) > 12000:
            continue
        repos[repo] = repos.get(repo, 0) + 1
        out.append({"id": f"swe:{r['id']}", "task": task, "tests": r["completion"], "extra": "", "near": ""})
        if len(out) >= n:
            break
    return out


def complete_prefix(text: str) -> list:
    """The complete objects at the start of a JSON array whose tail is broken (a generation that degenerated into
    repetition and hit its length limit): decoded one at a time, stopping at the first that does not decode."""
    dec, i, out = json.JSONDecoder(), text.find("[") + 1, []
    if i == 0:
        return out
    while True:
        while i < len(text) and text[i] in " \n\r\t,":
            i += 1
        try:
            x, i = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            return out
        out.append(x)


def parse(text: str, salvage: bool = False) -> list[dict] | None:
    """The requirement list in a model's reply, validated; None if it does not parse. With salvage, a reply whose
    array never closes keeps its complete leading requirements instead (used for the model's own lists, never for
    the teacher's training targets)."""
    s, e = text.find("["), text.rfind("]")
    try:
        arr = json.loads(text[s:e + 1]) if 0 <= s < e else None
    except json.JSONDecodeError:
        arr = None
    if arr is None and salvage:
        arr = complete_prefix(text)
    if not isinstance(arr, list) or not arr:
        return None
    clean = []
    for x in arr:
        if not isinstance(x, dict) or not all(isinstance(x.get(k), str) for k in FIELDS):
            return None
        if x["strictness"] not in STRICTNESS or not x["requirement"].strip() or not x["near_miss"].strip():
            return None
        clean.append({k: x[k].strip() for k in FIELDS})
    return clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swe", type=int, default=700)
    ap.add_argument("--tb-repeat", type=int, default=3, help="Terminal-Bench examples count this many times in training")
    ap.add_argument("--cap-usd", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "reqmodel")
    a = ap.parse_args()
    import anthropic
    client = anthropic.Anthropic(max_retries=6)
    cache = a.out / "teacher"
    cache.mkdir(parents=True, exist_ok=True)
    exs = tb_examples() + swe_examples(a.swe)
    spent, lock, stats = [0.0], threading.Lock(), {"ok": 0, "bad": 0, "cached": 0, "skipped_cap": 0}

    def one(ex):
        f = cache / (hashlib.sha256(ex["id"].encode()).hexdigest()[:16] + ".json")
        if f.is_file():
            stats["cached"] += 1
            return json.loads(f.read_text())
        with lock:
            if spent[0] >= a.cap_usd:
                stats["skipped_cap"] += 1
                return None
        msg = TEACHER.format(extra=ex["extra"], task=ex["task"], tests=ex["tests"], near=ex["near"])
        with client.messages.stream(model=TEACHER_MODEL, max_tokens=8000, messages=[{"role": "user", "content": msg}]) as st:
            r = st.get_final_message()
        with lock:
            spent[0] += (r.usage.input_tokens * PRICE_IN + r.usage.output_tokens * PRICE_OUT) / 1e6
        text = "".join(b.text for b in r.content if b.type == "text")
        reqs = parse(text)
        rec = {"id": ex["id"], "task": ex["task"], "requirements": reqs, "raw": None if reqs else text[:2000]}
        f.write_text(json.dumps(rec))
        return rec

    with cf.ThreadPoolExecutor(a.workers) as pool:
        recs = list(pool.map(one, exs))
    train = []
    for rec in recs:
        if not rec:
            continue
        if not rec["requirements"]:
            stats["bad"] += 1
            continue
        stats["ok"] += 1
        row = {"id": rec["id"], "prompt": STUDENT + "<task>\n" + rec["task"] + "\n</task>",
               "completion": json.dumps(rec["requirements"], ensure_ascii=False)}
        train += [row] * (a.tb_repeat if rec["id"].startswith("tb:") else 1)
    random.Random(1).shuffle(train)
    (a.out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    print(json.dumps({**stats, "train_rows": len(train), "tb_tasks": sum(1 for r in recs if r and r["requirements"] and r["id"].startswith("tb:")),
                      "spent_usd": round(spent[0], 2)}, indent=1))


if __name__ == "__main__":
    main()
