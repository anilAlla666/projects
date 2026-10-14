"""The one loader for built examples. Every reader (serialisation, checks, attacks, baselines, evaluation) goes
through it, so they all see the same rows.

Two rules, applied per task:
- A task whose revert (the original, correct code) fails its own oracle, or whose unfixed code with faked test
  output passes it, has an unreliable oracle (flaky or order-dependent tests), so every label on it is suspect:
  the whole task is dropped.
- An example the oracle-leak check flags (ROOT/leaks.json, written by scripts/leaks.py) is dropped.
- Persuasive prose must not predict the verdict. A task keeps its persuasion pair only when the prose-on-the-fix
  attempt is labelled correct (the insertion can break an indented method, which turns it into one more wrong
  attempt with prose); otherwise both halves of the pair are dropped.
"""
from __future__ import annotations

import json
from pathlib import Path


def task_of(example_id: str) -> str:
    return example_id.split("/")[0]


def usable(examples: list) -> tuple[list, dict]:
    by_task = {}
    for e in examples:
        by_task.setdefault(task_of(e["example_id"]), []).append(e)
    out, report = [], {"tasks": len(by_task), "tasks_dropped_unreliable_oracle": 0, "persuasion_pairs_dropped": 0}
    for group in by_task.values():
        if any((e["attempt_kind"] == "revert" and e["label"] == 0) or (e["attempt_kind"] == "fake_output" and e["label"] == 1)
               for e in group):
            report["tasks_dropped_unreliable_oracle"] += 1
            continue
        pair_ok = any(e["attempt_kind"] == "persuasion_correct" and e["label"] == 1 for e in group)
        if not pair_ok and any(e["attempt_kind"] in ("persuasion", "persuasion_correct") for e in group):
            report["persuasion_pairs_dropped"] += 1
        out += [e for e in group if pair_ok or e["attempt_kind"] not in ("persuasion", "persuasion_correct")]
    report["examples"] = len(out)
    return out, report


def leaked(root: Path, setting: str = "strict") -> set:
    p = Path(root) / ("leaks.json" if setting == "strict" else f"leaks_{setting}.json")
    return set(json.loads(p.read_text())) if p.is_file() else set()


def load_examples(root: Path, split: str | None = None, with_report: bool = False, setting: str | None = None):
    """setting: "strict" (the training visibility rule, <lib>/examples) or "realistic" (every existing test visible,
    <lib>/examples_realistic); defaults to $VM_SETTING, else strict."""
    import os
    setting = setting or os.environ.get("VM_SETTING", "strict")
    sub_dir = "examples" if setting == "strict" else f"examples_{setting}"
    raw = [e for p in sorted(Path(root).glob(f"*/{sub_dir}/*.jsonl")) for e in map(json.loads, p.read_text().splitlines())]
    out, report = usable(raw)
    bad = leaked(root, setting)
    report["examples_dropped_oracle_leak"] = sum(e["example_id"] in bad for e in out)
    out = [e for e in out if e["example_id"] not in bad and (split is None or e["split"] == split)]
    return (out, report) if with_report else out
