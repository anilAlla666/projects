"""The four pipeline checks from spec section 7. Each returns a list of violations; an empty list is a pass.
Each has a test that plants a violation and requires it to be reported."""
from __future__ import annotations

import random
from pathlib import Path


def model_input_text(ex: dict) -> str:
    return "\n".join([ex["task_text"], *ex["function_lines"], ex["diff"], *(e["text"] for e in ex["evidence"]), *ex["untrusted"]])


def check_no_oracle_leak(ex: dict, oracle_sources: list, doctest_outputs: list) -> list:
    """No oracle-only test source (40 or more characters) and no doctest expected output reaches the input."""
    text = model_input_text(ex)
    bad = [f"oracle test source: {s[:60]!r}" for s in oracle_sources if len(s) >= 40 and s in text]
    bad += [f"doctest output: {d[:60]!r}" for d in doctest_outputs if len(d) >= 12 and d in text]
    return bad


def check_split(examples: list, heldout: dict) -> list:
    bad, lib_split = [], {}
    for ex in examples:
        prev = lib_split.setdefault(ex["library"], ex["split"])
        if prev != ex["split"]:
            bad.append(f"library {ex['library']} in {prev} and {ex['split']}")
        if ex["split"] != "test" and ex["operator"] == heldout["operator"]:
            bad.append(f"held-out operator in {ex['split']}: {ex['example_id']}")
        if ex["split"] != "test" and ex["attempt_kind"] == heldout["wrong_kind"]:
            bad.append(f"held-out attempt kind in {ex['split']}: {ex['example_id']}")
    return bad


def check_fake_output_fails(examples: list) -> list:
    bad = []
    for ex in examples:
        if ex["attempt_kind"] == "fake_output":
            summary = ex["evidence"][0]["text"]
            n, _, rest = summary.removeprefix("visible tests: ").partition(" of ")
            if ex["label"] != 0 or (n.isdigit() and rest.split(" ")[0].isdigit() and n == rest.split(" ")[0]):
                bad.append(f"fake output not caught: {ex['example_id']}")
    return bad


def check_labels(examples: list, relabel, k: int = 50, seed: int = 0) -> list:
    """relabel(example) -> 0 or 1 from a fresh oracle run; compares on a seeded sample."""
    sample = random.Random(seed).sample(examples, min(k, len(examples)))
    return [f"label mismatch: {ex['example_id']}" for ex in sample if relabel(ex) != ex["label"]]
