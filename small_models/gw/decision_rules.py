"""Decision rules for written graders, chosen on development runs and applied once to the exam. A grader set is one
or more drafts per task; the kept tests are the ones that fail on the unfixed code. Rules:
  all       every kept test of every draft passes (the rule used so far, per single draft)
  frac:t    at least a share t of the kept tests, pooled over the drafts, pass
  vote:m    at least m drafts accept under `all`
A task with no kept tests accepts every attempt. Only agent attempts are scored (the maintainers' fix is excluded).

    python gw/decision_rules.py dev RFT_DIR RUNS_DIR                  # rft4: labels from the instances file
    python gw/decision_rules.py exam REAL_DIR RUNS_DIR RULE           # one rule, on the exam
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def load(runs: Path):
    """task -> draft -> {"kept": [...], "att": {key: outcomes or None}}"""
    out = collections.defaultdict(dict)
    for d in runs.iterdir():
        task = d.name if "." not in d.name else d.name.rsplit(".", 1)[0]
        b = d / "base" / "out" / "result.json"
        if not b.is_file():
            continue
        kept = [k for k, v in json.loads(b.read_text()).get("outcomes", {}).items() if v != "passed"]
        att = {}
        for a in d.iterdir():
            r = a / "out" / "result.json"
            if a.name != "base" and r.is_file():
                x = json.loads(r.read_text())
                att[a.name] = None if "apply_failed" in x else x.get("outcomes", {})
        out[task][d.name] = {"kept": kept, "att": att}
    return out


def decide(drafts: dict, key: str, rule: str):
    live = [d for d in drafts.values() if d["kept"]]
    if not live:
        return 1
    if rule == "all" or rule.startswith("vote"):
        votes = [all((d["att"].get(key) or {}).get(k) == "passed" for k in d["kept"]) for d in live]
        return int(all(votes)) if rule == "all" else int(sum(votes) >= min(int(rule.split(":")[1]), len(live)))
    t = float(rule.split(":")[1])
    tests = [((d["att"].get(key) or {}).get(k) == "passed") for d in live for k in d["kept"]]
    return int(sum(tests) / len(tests) >= t)


def score(rows):
    pos = [p for y, p in rows if y == 1]
    neg = [p for y, p in rows if y == 0]
    ca, wr = sum(pos) / max(1, len(pos)), 1 - sum(neg) / max(1, len(neg))
    return {"ba": round((ca + wr) / 2, 4), "correct_accepted": round(ca, 4), "wrong_rejected": round(wr, 4), "n": len(rows)}


RULES = ["all", "vote:1", "vote:2", "vote:3"] + [f"frac:{t}" for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]


def main():
    mode = sys.argv[1]
    if mode == "dev":
        rft, runs = Path(sys.argv[2]), Path(sys.argv[3])
        insts = {i["instance_id"]: i for i in json.loads((rft / "instances.json").read_text())}
        data = load(runs)
        res = {}
        for rule in RULES:
            rows = []
            for task, drafts in data.items():
                for i, at in enumerate(insts[task]["attempts"]):
                    if at["model"] == "maintainers" or any(f"a{i}" not in d["att"] for d in drafts.values() if d["kept"]):
                        continue
                    rows.append((int(at["target"]), decide(drafts, f"a{i}", rule)))
            res[rule] = score(rows)
        print(json.dumps(res, indent=1))
    else:
        real, runs, rule = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
        ex = [json.loads(l) for l in (real / "examples_realistic.jsonl").read_text().splitlines()]
        data = load(runs)
        rows = []
        for e in ex:
            task, key = e["example_id"].split("/")[0].removeprefix("real:"), e["example_id"].split("/")[1]
            rows.append((e["label"], decide(data.get(task, {}), key, rule)))  # no grader: every attempt accepted
        print(json.dumps({rule: score(rows), "tasks": len(data)}))


if __name__ == "__main__":
    main()
