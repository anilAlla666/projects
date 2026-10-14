"""Where a grader disagrees with the official tests, and on which official test. Runs on the server.

For each hidden-pool attempt the grader got wrong: its label, the grader's verdict, and the official tests it failed
(from the attempt's own leaderboard ctrf) or "passes every official test". Grouped by the official test, so a missing
requirement shows up as one test that every leaked wrong attempt fails.

    python tools/diagnose.py TASK VERDICTS          # VERDICTS: the P/F string from a result.json's "after"
"""
import collections
import json
import sys
from pathlib import Path

DATA = Path("/root/vbench")


def official(task: str, entry: str) -> list[str] | None:
    f = DATA / "pools" / task / "job" / entry / "verifier" / "ctrf.json"
    if not f.is_file():
        return None
    tests = json.loads(f.read_text()).get("results", {}).get("tests", [])
    return [t["name"].split("::")[-1] + (": " + (t.get("message") or "")[:160].replace("\n", " ") if t.get("message") else "")
            for t in tests if t.get("status") not in ("passed", "skipped")]


def main():
    task, verdicts = sys.argv[1], sys.argv[2]
    rows = json.loads((DATA / "hud_pool_keys" / f"{task}.json").read_text())
    assert len(rows) == len(verdicts), (len(rows), len(verdicts))
    leaked, rejected = collections.defaultdict(list), []
    for r, v in zip(rows, verdicts):
        if r["label"] == "fail" and v == "P":
            fails = official(task, r["entry_id"]) or ["(no official record: " + r["source"] + ")"]
            for t in fails:
                leaked[t.split(":")[0]].append((r["entry_id"], t))
        if r["label"] == "pass" and v == "F":
            rejected.append(r["entry_id"])
    n_wrong = sum(r["label"] == "fail" for r in rows)
    n_right = len(rows) - n_wrong
    print(f"{task}: wrong attempts the grader PASSES: {sum(1 for r, v in zip(rows, verdicts) if r['label'] == 'fail' and v == 'P')} of {n_wrong}; "
          f"correct attempts it REJECTS: {len(rejected)} of {n_right}")
    print("Leaked wrong attempts, by the official test they fail:")
    for t, xs in sorted(leaked.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(xs)}x {t}")
        for e, full in xs[:2]:
            print(f"      e.g. {e}: {full[:230]}")
    print("Correct attempts rejected:", rejected)


if __name__ == "__main__":
    main()
