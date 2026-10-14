"""Writes ROOT/leaks.json: the example ids whose model input contains an oracle-only test's source or a doctest's
expected output (vm.data.checks.check_no_oracle_leak), so the shared loader drops them.

    python scripts/leaks.py ROOT
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_judge_data as b  # noqa: E402
from vm.data.checks import check_no_oracle_leak  # noqa: E402
from vm.data.load import usable  # noqa: E402
from vm.tasks.task import _test_source  # noqa: E402


def main():
    """python scripts/leaks.py ROOT [SETTING]. In the realistic setting the hidden tests are the task's other
    failing tests, so only their sources count as a leak."""
    root = Path(sys.argv[1])
    setting = sys.argv[2] if len(sys.argv) > 2 else "strict"
    sub_dir = "examples" if setting == "strict" else f"examples_{setting}"
    tasks = {t["task_id"]: t for t in b.all_tasks(root)}
    raw = [e for p in sorted(root.glob(f"*/{sub_dir}/*.jsonl")) for e in map(json.loads, p.read_text().splitlines())]
    exs, _ = usable(raw)
    bad = []
    for e in exs:
        t = tasks[e["example_id"].rsplit("/", 2)[0]]
        src = root / t["library"] / "src"
        if setting == "strict":
            hidden_src, doct = [s for s in (_test_source(src, o) for o in t["oracle_only"]) if s != t["named_test_source"]], b.doctest_outputs(t["original_source"])
        else:
            # a hidden test that shares the named test's source (another parametrization of it) leaks nothing new
            hidden_src = [s for s in (_test_source(src, o) for o in sorted(set(t["failing"]) - {t["named_test"]})) if s != t["named_test_source"]]
            doct = []
        if check_no_oracle_leak(e, hidden_src, doct):
            bad.append(e["example_id"])
    (root / ("leaks.json" if setting == "strict" else f"leaks_{setting}.json")).write_text(json.dumps(sorted(bad)))
    print(json.dumps({"examples": len(exs), "leaking": len(bad)}))


if __name__ == "__main__":
    main()
