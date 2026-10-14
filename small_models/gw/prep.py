"""Grader-writer data. Each example: the repository, the issue, and the source files the fix will change (never the
fix itself) as the prompt; the maintainers' test patch as the target. SWE-rebench rows from repositories outside the
real-attempts set train; the real-attempts instances are the held-out evaluation.

    python gw/prep.py RAW_DIR REAL_DIR OUT_DIR [--train-limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from pathlib import Path


def source_files(patch: str) -> list[str]:
    files = []
    for ln in patch.splitlines():
        if ln.startswith("+++ b/"):
            p = ln[6:].strip()
            if p.endswith(".py") and "test" not in p.lower():
                files.append(p)
    return files


def prompt(repo: str, problem: str, files: list[str]) -> str:
    return (f"Repository: {repo}\n\nThe issue a fix must resolve:\n<issue>\n{problem.strip()}\n</issue>\n\n"
            f"Source files the fix will change: {', '.join(files) or 'unknown'}\n\n"
            "Write the tests that decide whether a submitted fix is correct: they must fail on the current code and pass "
            "once the issue is fixed. Reply with a unified diff that adds one new, self-contained test file "
            "(do not edit existing files).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("real", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--train-limit", type=int, default=8000)
    ap.add_argument("--new-files-only", action="store_true", help="train only on test patches that add new files, which are self-contained")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    cols = ["instance_id", "repo", "problem_statement", "test_patch", "patch", "license_name"]
    rows = [r for f in sorted(glob.glob(str(a.raw / "*SWE-rebench*.parquet"))) for r in pq.read_table(f, columns=cols).to_pylist()]
    insts = json.loads((a.real / "instances.json").read_text())
    held_repos = {i["repo"] for i in insts}
    held_ids = {i["instance_id"] for i in insts}
    seen, train, evalr = set(), [], {}
    for r in rows:
        if r["instance_id"] in seen:
            continue
        seen.add(r["instance_id"])
        if r["instance_id"] in held_ids:
            evalr[r["instance_id"]] = r
            continue
        if r["repo"] in held_repos or not r["problem_statement"] or not r["test_patch"]:
            continue
        if not re.search(r"\b(mit|apache|bsd|isc)\b", r["license_name"] or "", re.I):
            continue
        if len(r["test_patch"]) > 10000 or len(r["problem_statement"]) > 6000:
            continue
        if a.new_files_only:
            olds = re.findall(r"^--- (\S+)", r["test_patch"], re.M)
            if not olds or any(o != "/dev/null" for o in olds):
                continue
        train.append({"id": r["instance_id"], "prompt": prompt(r["repo"], r["problem_statement"], source_files(r["patch"])),
                      "completion": r["test_patch"]})
    random.Random("gw-train").shuffle(train)
    train = train[: a.train_limit]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "train.jsonl").write_text("".join(json.dumps(t) + "\n" for t in train))
    ev = [{"id": iid, "prompt": prompt(r["repo"], r["problem_statement"], source_files(r["patch"]))} for iid, r in sorted(evalr.items())]
    (a.out / "eval_prompts.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    print(json.dumps({"train": len(train), "eval": len(ev), "train_repos": len({t["id"].rsplit("-", 1)[0] for t in train})}))


if __name__ == "__main__":
    main()
