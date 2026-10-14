"""The author model, stage A (step 10, docs/research_intent_goal.md): can a model answer, without the official
tests, the questions a grader writer must settle, the way the task's author settled them in those tests?

    fetch    task text and official tests of every task that has them (harvest/ and hud_tasks/), from the server
    ask      a question writer that sees the task text only lists the choices the text leaves open, as
             multiple-choice questions (options shuffled with a fixed seed; each option's original place is kept)
    label    the stand-in author reads the official tests and answers each: an option key, "undecided" (the tests
             do not decide it) or "missing" (the tests decide it and no option is what they check)
    predict  the model, without the tests: zero-shot, and with every training task's decided questions as examples
    score    accuracy on the held-out tasks' decided questions, with controls

Split, fixed before any question exists: held out = the Verifier Bench tasks (hud_tasks/), training = the harvested
tasks that are not among them.

    VB_HOST=root@... VB_DATA=/root/vbench python -m vb.authormodel all --out runs/author --cap-usd 11.5
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import random
import re
from pathlib import Path

from . import host
from .writer_api import Spend, cost, ssh

ASK_MODEL = "claude-opus-5-5"
LABEL_MODEL = "claude-sonnet-5"
PREDICT_MODEL = "claude-opus-5-5"
TESTS_CAP = 120_000  # characters of official tests the stand-in reads; longer suites are flagged as truncated
KEYS = "ABCDE"
LOADED = ("standard", "usual", "typical", "common", "default", "recommended", "conventional", "obvious", "customary")

ASK = """You are about to write a grader (automated tests) for the task below. It must accept every correct result
and reject every wrong one. List the decisions your grader must make that the task text does not settle on its own:
which model, formula or method an answer must follow; tolerances and precision; which output formats or variants are
acceptable; which rules or constraints apply; thresholds; how edge cases are judged. Leave out anything the text
settles. For each decision, write one multiple-choice question with 2 to 5 options, each a concrete reading a grader
could implement. Describe each option by what it is; never call an option standard, usual, typical, common, default,
recommended, conventional, customary or obvious.

Return JSON only: {"questions": [{"question": "...", "options": ["...", "..."]}]}, with 3 to 8 questions.

<task>
{task}
</task>
"""

LABEL = """You wrote the task below and its official tests. Another agent, writing its own grader for the task,
asks how each question below is decided. Answer each from what your official tests check, with exactly one of:
- an option key (A, B, ...), when your tests check that option;
- "undecided", when your tests do not decide the question (every option would get the same verdicts);
- "missing", when your tests do decide it but none of the options is what they check.

Return JSON only: {{"answers": {{"1": "B", "2": "undecided", ...}}}}, one entry per question.

<task>
{task}
</task>

<official_tests>
{tests}
</official_tests>

<questions>
{questions}
</questions>
"""

PREDICT = """Below is a task and questions another agent asked while writing a grader for it. For each question,
answer which option the task's author intended: what the author's own tests check. You do not have the author's
tests. Answer every question with one option key.

Return JSON only: {{"answers": {{"1": "B", ...}}}}, one entry per question.

<task>
{task}
</task>

<questions>
{questions}
</questions>
"""

EXAMPLES = """Questions from other tasks, each with the answer its author's official tests gave:

{examples}

"""

FETCH = r'''
import json, os, re, glob
D = "/root/vbench"
def strip(t): return re.sub(r"^<!-- harbor-canary[^\n]*-->\n?", "", t).strip()
CODE = (".py", ".sh", ".js", ".ts", ".mjs", ".json", ".toml", ".yaml", ".yml", ".txt", ".md", ".rs", ".go", ".c", ".h", ".v", ".lean", ".cfg")
def one(name, text):  # a test's code is read in full; data files (and anything over 300k characters) by name only
    if name.lower().endswith(CODE) and len(text) <= 300000 and "\x00" not in text:
        return "=== %s\n%s\n" % (name, text)
    return "=== %s (%d characters, not shown)\n" % (name, len(text))
out = {}
for d in sorted(glob.glob(D + "/hud_tasks/write-verifier-*")):
    name = os.path.basename(d)[len("write-verifier-"):]
    m = re.search(r"<task-instruction>\n?(.*?)\n?</task-instruction>", open(d + "/instruction.md").read(), re.S)
    tests = ""
    for f in sorted(glob.glob(d + "/solution/official_tests/*")):
        if os.path.isfile(f):
            tests += one(os.path.basename(f), open(f, errors="replace").read())
    out[name] = {"task": name, "split": "held_out", "text": m.group(1).strip(), "tests": tests}
for f in sorted(glob.glob(D + "/harvest/*/task.json")):
    name = os.path.basename(os.path.dirname(f))
    if name in out:
        continue
    t = json.load(open(f))
    ot = t.get("official_tests") or {}
    tests = "".join(one(k, str(v)) for k, v in sorted(ot.items())) if isinstance(ot, dict) else one("tests", str(ot))
    out[name] = {"task": name, "split": "train", "text": strip(t["instruction"]), "tests": tests}
print(json.dumps(list(out.values())))
'''


RAW: list = []  # every reply's stop reason and text, saved beside the results (the first labelling was unreadable)


def call(prompt: str, model: str, spend: Spend, max_tokens: int = 16000, cache_prefix: str | None = None,
         think: bool = True, tag: str = "") -> str:
    """One request. Thinking stays on (it cannot be turned off on Opus 5.5) unless think=False; max_tokens leaves room
    for it, since at 1,000 the stand-in's whole reply was a thinking block (amendment, step 10)."""
    import anthropic
    if spend.usd >= spend.cap:
        raise RuntimeError(f"spend cap ${spend.cap} reached")
    content = ([{"type": "text", "text": cache_prefix, "cache_control": {"type": "ephemeral"}}] if cache_prefix else []) \
        + [{"type": "text", "text": prompt}]
    r = anthropic.Anthropic(max_retries=8).messages.create(
        model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": content}],
        **({} if think else {"thinking": {"type": "disabled"}}))
    spend.add(cost(r.usage, model))
    text = "".join(b.text for b in r.content if b.type == "text")
    RAW.append({"tag": tag, "model": model, "stop": r.stop_reason, "out_tokens": r.usage.output_tokens, "text": text})
    return text


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def shuffled(task: str, i: int, options: list[str]) -> list[dict]:
    """Options in a fixed random order (seeded by task and question), each keeping its place in the writer's list."""
    order = list(range(len(options)))
    random.Random(int(hashlib.sha256(f"{task}/{i}".encode()).hexdigest()[:12], 16)).shuffle(order)
    return [{"key": KEYS[j], "text": options[o], "writer_place": o} for j, o in enumerate(order)]


def show(qs: list[dict]) -> str:
    return "\n\n".join(f"{n}. {q['question']}\n" + "\n".join(f"   {o['key']}) {o['text']}" for o in q["options"])
                       for n, q in enumerate(qs, 1))


def fetch(out: Path) -> list[dict]:
    r = ssh(f"{host.PYTHON} -", FETCH, timeout=300)
    tasks = json.loads(r.stdout)
    for t in tasks:
        t["truncated"] = len(t["tests"]) > TESTS_CAP
    (out / "tasks.json").write_text(json.dumps(tasks))
    return tasks


def ask(t: dict, spend: Spend) -> list[dict]:
    raw = call(ASK.replace("{task}", t["text"]), ASK_MODEL, spend)
    qs = []
    for i, q in enumerate(parse(raw).get("questions") or []):
        opts = [str(o).strip() for o in q.get("options") or [] if str(o).strip()]
        if isinstance(q.get("question"), str) and 2 <= len(opts) <= 5:
            qs.append({"id": f"{t['task']}/{i + 1}", "question": q["question"].strip(), "options": shuffled(t["task"], i, opts)})
    return qs


def label(t: dict, qs: list[dict], spend: Spend) -> dict:
    raw = call(LABEL.format(task=t["text"], tests=t["tests"][:TESTS_CAP], questions=show(qs)), LABEL_MODEL, spend,
               think=False, tag="label/" + t["task"])
    got = parse(raw).get("answers") or {}
    out = {}
    for n, q in enumerate(qs, 1):
        a = str(got.get(str(n), "")).strip().strip(".\"' ")
        keys = [o["key"] for o in q["options"]]
        out[q["id"]] = a.upper() if a.upper() in keys else (a.lower() if a.lower() in ("undecided", "missing") else "invalid")
    return out


def predict(t: dict, qs: list[dict], spend: Spend, examples: str | None) -> dict:
    raw = call(PREDICT.format(task=t["text"], questions=show(qs)), PREDICT_MODEL, spend,
               cache_prefix=examples, tag=("predict_examples/" if examples else "predict_zero/") + t["task"])
    got = parse(raw).get("answers") or {}
    out = {}
    for n, q in enumerate(qs, 1):
        a = str(got.get(str(n), "")).strip().strip(".\"' ").upper()
        out[q["id"]] = a if a in [o["key"] for o in q["options"]] else "invalid"
    return out


def examples_block(train: list[dict], qs: dict, labels: dict) -> str:
    lines = []
    for t in train:
        for q in qs.get(t["task"], []):
            a = labels.get(q["id"])
            if a and a in KEYS:
                lines.append(f"Task {t['task']}: {q['question']}\n" + "\n".join(
                    f"   {o['key']}) {o['text']}" for o in q["options"]) + f"\n   Author's answer: {a}")
    return EXAMPLES.format(examples="\n\n".join(lines))


def wilson(k: int, n: int) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def score(tasks: list[dict], qs: dict, labels: dict, preds: dict) -> dict:
    held = [t for t in tasks if t["split"] == "held_out"]
    rows, per = [], {}
    for t in held:
        for q in qs.get(t["task"], []):
            a = labels.get(q["id"])
            rows.append({"task": t["task"], "truncated": t["truncated"], "label": a, "k": len(q["options"]),
                         "first": next(o["key"] for o in q["options"] if o["writer_place"] == 0),
                         **{arm: preds[arm].get(q["id"]) for arm in preds}})
    decided = [r for r in rows if r["label"] in KEYS]

    def acc(rs, arm):
        k = sum(r[arm] == r["label"] for r in rs)
        return {"correct": k, "n": len(rs), "acc": round(k / len(rs), 3) if rs else None, "ci95": wilson(k, len(rs))}
    res = {"questions": len(rows), "decided": len(decided),
           "undecided": sum(r["label"] == "undecided" for r in rows), "missing": sum(r["label"] == "missing" for r in rows),
           "invalid_labels": sum(r["label"] == "invalid" for r in rows)}
    res["missing_rate"] = round(res["missing"] / max(1, res["decided"] + res["missing"]), 3)
    for arm in preds:
        res[arm] = acc(decided, arm)
        res[arm + "_untruncated"] = acc([r for r in decided if not r["truncated"]], arm)
    res["control_writer_first"] = acc([dict(r, first_arm=r["first"]) for r in decided], "first")
    res["control_chance"] = round(sum(1 / r["k"] for r in decided) / max(1, len(decided)), 3)
    for t in held:
        rs = [r for r in decided if r["task"] == t["task"]]
        per[t["task"]] = {"decided": len(rs), **{arm: sum(r[arm] == r["label"] for r in rs) for arm in preds},
                          "missing": sum(1 for r in rows if r["task"] == t["task"] and r["label"] == "missing"),
                          "truncated": t["truncated"]}
    res["per_task"] = per
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "all"])
    ap.add_argument("--out", type=Path, default=Path("runs/author"))
    ap.add_argument("--cap-usd", type=float, default=11.5)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tasks = fetch(a.out)
    print(json.dumps({"tasks": len(tasks), "held_out": sum(t["split"] == "held_out" for t in tasks),
                      "truncated": [t["task"] for t in tasks if t["truncated"]]}), flush=True)
    if a.stage == "fetch":
        return
    spend = Spend(a.cap_usd)
    if (a.out / "questions.json").is_file():  # resume: the questions are never rewritten once written
        qs = json.loads((a.out / "questions.json").read_text())
    else:
        with cf.ThreadPoolExecutor(4) as ex:
            qs = dict(zip([t["task"] for t in tasks], ex.map(lambda t: ask(t, spend), tasks)))
        (a.out / "questions.json").write_text(json.dumps(qs, indent=1, ensure_ascii=False))
    loaded = [o["text"] for v in qs.values() for q in v for o in q["options"]
              if re.search(r"\b(" + "|".join(LOADED) + r")\b", o["text"], re.I)]
    print(json.dumps({"stage": "ask", "questions": sum(map(len, qs.values())), "loaded_options": len(loaded),
                      "usd": round(spend.usd, 3)}), flush=True)
    if (a.out / "labels.json").is_file():
        labels = json.loads((a.out / "labels.json").read_text())
    else:
        with cf.ThreadPoolExecutor(4) as ex:
            labels = {}
            for d in ex.map(lambda t: label(t, qs[t["task"]], spend) if qs[t["task"]] else {}, tasks):
                labels.update(d)
        (a.out / "labels.json").write_text(json.dumps(labels, indent=1))
        (a.out / "raw_labels.json").write_text(json.dumps(RAW, indent=1, ensure_ascii=False))
    print(json.dumps({"stage": "label", "usd": round(spend.usd, 3)}), flush=True)
    held = [t for t in tasks if t["split"] == "held_out" and qs[t["task"]]]
    ex_block = examples_block([t for t in tasks if t["split"] == "train"], qs, labels)
    preds = {"zero_shot": {}, "with_examples": {}}
    for t in held:  # one after another, so the examples block is cached across tasks
        preds["zero_shot"].update(predict(t, qs[t["task"]], spend, None))
        preds["with_examples"].update(predict(t, qs[t["task"]], spend, ex_block))
    (a.out / "predictions.json").write_text(json.dumps(preds, indent=1))
    (a.out / "raw_all.json").write_text(json.dumps(RAW, indent=1, ensure_ascii=False))
    res = score(tasks, qs, labels, preds)
    res["usd"] = round(spend.usd, 3)
    res["loaded_options"] = len(loaded)
    (a.out / "score.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"stage": "score", **{k: v for k, v in res.items() if k != "per_task"}}), flush=True)


if __name__ == "__main__":
    main()
