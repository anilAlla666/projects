"""Does a broken test show itself on the unfixed code? For each kept test (fails on the unfixed code), its category
from the exam runs (broken: fails every fix; good: passes every correct fix and catches a wrong one; other) against
why it failed on the unfixed code: a crash in the test's own use of the code (TypeError, AttributeError, NameError,
an import or fixture error) or an assertion. A rule that drops crash-on-unfixed tests needs no reference fix.

    python gw/crash_rule.py REAL_DIR NAME:EXAM_RUNS:BASE_DIAG_RUNS ...
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

CRASH = ("TypeError", "AttributeError", "NameError", "ImportError", "ModuleNotFoundError", "SyntaxError", "fixture",
         "IndentationError", "UnboundLocalError")


def main():
    real = Path(sys.argv[1])
    ex = [json.loads(l) for l in (real / "examples_realistic.jsonl").read_text().splitlines()]
    bytask = collections.defaultdict(list)
    for e in ex:
        bytask[e["example_id"].split("/")[0].removeprefix("real:")].append((e["example_id"].split("/")[1], e["label"]))
    for spec in sys.argv[2:]:
        name, runs, diag = spec.split(":")
        runs, diag = Path(runs), Path(diag)
        tab = collections.Counter()
        for t, atts in bytask.items():
            b, b2 = runs / t / "base" / "out" / "result.json", diag / t / "base" / "out" / "result.json"
            if not b.is_file() or not b2.is_file():
                continue
            kept = [k for k, v in json.loads(b.read_text()).get("outcomes", {}).items() if v != "passed"]
            why = {}
            for l in json.loads(b2.read_text()).get("failures", []):
                m = re.match(r"FAILED (\S+)(?: - (.*))?", l)
                if m:
                    why[m.group(1).split("[")[0].split("::", 1)[-1]] = m.group(2) or ""
            outs = {k: json.loads((runs / t / k / "out" / "result.json").read_text()).get("outcomes", {})
                    for k, _ in atts if (runs / t / k / "out" / "result.json").is_file()}
            corr = [k for k, y in atts if y == 1 and k in outs]
            wrong = [k for k, y in atts if y == 0 and k in outs]
            for x in kept:
                pc = [outs[k].get(x) == "passed" for k in corr]
                pw = [outs[k].get(x) == "passed" for k in wrong]
                cat = "broken" if not any(pc) and not any(pw) else "good" if all(pc) and not all(pw) else "other"
                w = why.get(x.split("[")[0].split("::", 1)[-1])
                reason = "unknown" if w is None else "crash" if any(c in w for c in CRASH) else "assertion"
                tab[(cat, reason)] += 1
        tot = lambda c, r=None: sum(v for (cc, rr), v in tab.items() if cc == c and (r is None or rr == r))  # noqa: E731
        print(json.dumps({"writer": name, "broken": tot("broken"), "broken_crash_on_unfixed": tot("broken", "crash"),
                          "good": tot("good"), "good_crash_on_unfixed": tot("good", "crash"),
                          "other": tot("other"), "other_crash_on_unfixed": tot("other", "crash"),
                          "unknown_reason": sum(v for (c, r), v in tab.items() if r == "unknown")}))


if __name__ == "__main__":
    main()
