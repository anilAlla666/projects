"""Verifier Bench transfer: the Terminal-Bench pools (each attempt's output files, labelled by the official tests)
in the judge's input format, in two versions:

- none: what a grader without tests has, the task instruction and the attempt's files; evidence line 0 says no
  tests are available. This is the measurement.
- control: the same rows with evidence line 0 rebuilt from the attempt's official test counts (pools.json "ctrf").
  That is oracle evidence; it only checks the adapter and is never a reported number.

    python scripts/vb_adapter.py VBENCH_HUD_DIR POOLS.json TOKENIZER OUT_DIR
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vm.corpus.tok import load  # noqa: E402
from vm.data.example import serialise  # noqa: E402
from vm.harness.evidence import diff_shape, split_untrusted, unified_diff  # noqa: E402

TASKS = ["hof-topology-interpenetration", "pretrain-shard-corruption", "production-planning", "react-lead-form", "uefi-bootkit"]
FILE_CAP = 4000


def pid_to_entry(task: str, pool: list) -> dict:
    """Manifest ids were drawn as random.Random(f"{task}:0").getrandbits(64) in the pool's kept order."""
    rng = random.Random(f"{task}:0")
    return {"%016x" % rng.getrandbits(64): e for e in pool if e["kept"]}


def text_of(blob: Path) -> str:
    data = blob.read_bytes()
    try:
        t = data.decode("utf-8")
        if "\x00" in t:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "nul")
    except UnicodeDecodeError:
        return f"(binary file, {len(data)} bytes)\n"
    return t if len(t) <= FILE_CAP else t[:FILE_CAP] + f"\n... ({len(t) - FILE_CAP} more characters)\n"


def main():
    hud, pools_path, tok, out = Path(sys.argv[1]), Path(sys.argv[2]), load(Path(sys.argv[3])), Path(sys.argv[4])
    pools = json.loads(pools_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    rows = {"none": [], "control": []}
    report = {}
    for task in TASKS:
        d = hud / f"write-verifier-{task}"
        instr = (d / "instruction.md").read_text()
        m = re.search(r"<task-instruction>\s*(.*?)\s*</task-instruction>", instr, re.S)
        task_text = f"The task:\n{m.group(1) if m else instr}\n\nThe attempt's output files are shown as the change.\n"
        manifest = json.loads((d / "tests" / "grader" / "pool" / "manifest.json").read_text())
        mapping = pid_to_entry(task, pools[task])
        mapped = sum(x["id"] in mapping for x in manifest)
        agree = sum(1 for x in manifest if x["id"] in mapping and mapping[x["id"]]["label"] == x["label"])
        report[task] = {"attempts": len(manifest), "mapped": mapped, "labels_agree": agree}
        for x in manifest:
            pool = d / "tests" / "grader" / "pool"
            if "files" in x:  # content-addressed layout: path -> [sha256, mode]
                files = {p: text_of(pool / "blobs" / sha) for p, (sha, _) in sorted(x["files"].items())}
            else:  # older layout: one folder per attempt holding its artifact files
                root = pool / x["id"]
                files = {"/" + f.relative_to(root).as_posix(): text_of(f) for f in sorted(root.rglob("*")) if f.is_file()}
            diff = unified_diff({p: "" for p in files}, files)
            shown, moved = split_untrusted(diff)
            label = int(x["label"] == "pass")
            base = {"example_id": f"vb:{task}/{x['id']}", "library": task, "split": "vb", "operator": "vb", "attempt_kind": "vb",
                    "task_text": task_text, "function_lines": [], "bug_line": 0, "diff": shown, "untrusted": list(dict.fromkeys(moved)),
                    "label": label, "hidden_bucket": 0, "pointer": 0}
            tail = ["test files edited: none", "fuzz: not applicable", diff_shape(diff)]
            variants = {"none": ["visible tests: none available for this task"] + tail}
            e = mapping.get(x["id"])
            if e and e.get("ctrf"):
                passed, total = e["ctrf"]
                # the training format names each failing test on its own line after the summary
                fails = [f"official test {i + 1}: failed" for i in range(min(20, total - passed))]
                variants["control"] = [f"visible tests: {passed} of {total} pass"] + fails + tail
            for name, lines in variants.items():
                ex = {**base, "evidence": [{"id": i, "text": t} for i, t in enumerate(lines)]}
                s = serialise(ex, tok)
                rows[name].append({**s, "example_id": ex["example_id"], "attempt_kind": "vb", "operator": "vb", "library": task})
    for name, rs in rows.items():
        (out / f"serialised_vb_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    print(json.dumps({"rows": {k: len(v) for k, v in rows.items()}, "tasks": report}))


if __name__ == "__main__":
    main()
