"""Do the training targets work? For each task: did the grader apply, which of its tests fail on the unfixed code,
and do those tests all pass on the gold fix (attempt a0).

    python gw/val_targets.py RUNS_DIR
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def read(d: Path):
    r = d / "out" / "result.json"
    if r.is_file():
        return json.loads(r.read_text())
    e = d / "out" / "error.txt"
    return {"error": e.read_text()[-300:] if e.is_file() else "missing"}


def main():
    runs = Path(sys.argv[1])
    verdicts, examples = collections.Counter(), collections.defaultdict(list)
    for t in sorted(p for p in runs.iterdir() if p.is_dir()):
        base, gold = read(t / "base"), read(t / "a0")
        if "error" in base or "error" in gold:
            v = "container error"
        elif "apply_failed" in base or "apply_failed" in gold:
            v = "did not apply"
        else:
            kept = [k for k, o in base.get("outcomes", {}).items() if o in ("failed", "error")]
            if not base.get("outcomes"):
                v = "no tests ran"
            elif not kept:
                v = "nothing fails on unfixed code"
            elif all(gold.get("outcomes", {}).get(k) == "passed" for k in kept):
                v = "good: fails unfixed, passes gold"
            else:
                v = "a kept test fails on gold"
        verdicts[v] += 1
        examples[v].append(t.name)
    n = sum(verdicts.values())
    print(json.dumps({"tasks": n, **{k: f"{c} ({c / max(1, n):.0%})" for k, c in verdicts.most_common()},
                      "examples": {k: v[:4] for k, v in examples.items() if not k.startswith("good")}}, indent=1))


if __name__ == "__main__":
    main()
