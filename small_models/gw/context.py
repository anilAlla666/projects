"""Grader-writer data with repository context. For each task, the repository is cloned once (blobless) and read at the
task's base commit: the prompt shows the issue, the code around where the fix goes (never the fix), and the existing
test files the maintainers extended, so the writer can write tests that apply and use the real API. The target is the
maintainers' test patch. Training tasks come from repositories outside the real-attempts set; the real-attempts
instances are the evaluation.

    python gw/context.py RAW_DIR REAL_DIR OUT_DIR [--train-limit 4000] [--per-repo 4] [--workers 48]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import random
import re
import subprocess
from pathlib import Path

REPOS = Path.home() / "gw" / "repos"
SRC_CAP, TEST_CAP, ISSUE_CAP = 7000, 9000, 5000


def hunks(patch: str) -> dict:
    """path -> [(old_start, old_len)] for every file a patch modifies (new files have no old lines)."""
    out, cur = {}, None
    for ln in patch.splitlines():
        if ln.startswith("--- "):
            p = ln[4:].split("\t")[0].strip()
            cur = None if p == "/dev/null" else p[2:] if p.startswith("a/") else p
        elif ln.startswith("@@") and cur:
            m = re.match(r"@@ -(\d+)(?:,(\d+))?", ln)
            out.setdefault(cur, []).append((int(m.group(1)), int(m.group(2) or 1)))
    return out


def new_files(patch: str) -> list:
    return re.findall(r"^--- /dev/null\n\+\+\+ b/(\S+)", patch, re.M)


def window(text: str, spans: list, around: int, cap: int, head: int = 0) -> str:
    """The file's first `head` lines plus `around` lines either side of each span, with line numbers, capped."""
    lines = text.splitlines()
    keep = set(range(min(head, len(lines))))
    for s, n in spans:
        keep |= set(range(max(0, s - 1 - around), min(len(lines), s - 1 + n + around)))
    out, prev = [], -2
    for i in sorted(keep):
        if i != prev + 1:
            out.append("...")
        out.append(f"{i + 1:5d}  {lines[i]}")
        prev = i
    s = "\n".join(out)
    return s if len(s) <= cap else s[:cap] + "\n... (cut)"


def clone(repo: str) -> Path | None:
    d = REPOS / repo.replace("/", "__")
    if (d / ".git").is_dir():
        return d
    r = subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", f"https://github.com/{repo}.git", str(d)],
                       capture_output=True, text=True, timeout=600)
    return d if r.returncode == 0 else None


def show(d: Path, commit: str, path: str) -> str | None:
    r = subprocess.run(["git", "-C", str(d), "show", f"{commit}:{path}"], capture_output=True, timeout=120)
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None


def pick_test_files(d: Path, commit: str, src_paths: list, k: int = 2) -> list:
    """Evaluation tasks: the test files a person would extend, chosen by name (the module's name in the test file's
    name), never from the hidden tests."""
    r = subprocess.run(["git", "-C", str(d), "ls-tree", "-r", "--name-only", commit], capture_output=True, text=True, timeout=120)
    tests = [p for p in r.stdout.splitlines() if p.endswith(".py") and ("test" in Path(p).name.lower())]
    mods = {Path(p).stem.lower() for p in src_paths} | {Path(p).parent.name.lower() for p in src_paths}
    scored = sorted(tests, key=lambda t: (-sum(int(bool(m) and m in Path(t).stem.lower()) for m in mods), len(t)))
    return [t for t in scored if any(bool(m) and m in Path(t).stem.lower() for m in mods)][:k]


def build(row: dict, eval_mode: bool = False) -> dict | None:
    d = clone(row["repo"])
    if d is None:
        return None
    src_h = {p: s for p, s in hunks(row["patch"]).items() if p.endswith(".py") and "test" not in p.lower()}
    if eval_mode:  # show the start and end of the test files picked by name: no hidden-test locations
        test_h = {}
        for p in pick_test_files(d, row["base_commit"], list(src_h)):
            t = show(d, row["base_commit"], p)
            if t:
                n = len(t.splitlines())
                test_h[p] = [(max(1, n - 30), 30)]
    else:
        test_h = hunks(row["test_patch"])
    parts = []
    for p, spans in list(src_h.items())[:3]:
        t = show(d, row["base_commit"], p)
        if t:
            parts.append(f"### {p} (current code around where the fix goes)\n{window(t, spans, 40, SRC_CAP // max(1, len(src_h)))}")
    tests = []
    for p, spans in list(test_h.items())[:2]:
        t = show(d, row["base_commit"], p)
        if t:
            tests.append(f"### {p} (existing test file)\n{window(t, spans, 30, TEST_CAP // max(1, len(test_h)), head=40)}")
    if not eval_mode:
        for p in new_files(row["test_patch"])[:2]:
            tests.append(f"### {p} does not exist yet; you may create it.")
    if not parts:
        return None
    prompt = (f"Repository: {row['repo']}\n\nThe issue a fix must resolve:\n<issue>\n{row['problem_statement'].strip()[:ISSUE_CAP]}\n</issue>\n\n"
              "Source code (before any fix):\n" + "\n\n".join(parts) + "\n\nTests:\n" + ("\n\n".join(tests) or "(none shown)") +
              "\n\nWrite the tests that decide whether a submitted fix is correct: they must fail on the current code and pass "
              "once the issue is fixed, and they must not depend on details the issue leaves open. Reply with a unified diff "
              "against the test files above, or one that adds a new test file (line numbers are shown for reference; do not include them in the diff).")
    return {"id": row["instance_id"], "prompt": prompt, "completion": row["test_patch"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("real", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--train-limit", type=int, default=4000)
    ap.add_argument("--per-repo", type=int, default=4)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--eval-only", action="store_true", help="rebuild only the exam prompts")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    REPOS.mkdir(parents=True, exist_ok=True)
    cols = ["instance_id", "repo", "problem_statement", "test_patch", "patch", "license_name", "base_commit"]
    rows = [r for f in sorted(glob.glob(str(a.raw / "*SWE-rebench*.parquet"))) for r in pq.read_table(f, columns=cols).to_pylist()]
    insts = json.loads((a.real / "instances.json").read_text())
    held_repos, held_ids = {i["repo"] for i in insts}, {i["instance_id"] for i in insts}
    labelled = {json.loads(l)["example_id"].split("/")[0].removeprefix("real:") for l in (a.real / "examples_realistic.jsonl").read_text().splitlines()}
    seen, train, ev, per = set(), [], {}, {}
    for r in rows:  # every exam task, collected before the training sample stops early
        if r["instance_id"] in held_ids and r["instance_id"] in labelled and r["base_commit"]:
            ev.setdefault(r["instance_id"], r)
    rng = random.Random("gw-context")
    rng.shuffle(rows)
    for r in rows:
        if a.eval_only:
            break
        if r["instance_id"] in seen:
            continue
        seen.add(r["instance_id"])
        if r["instance_id"] in held_ids:
            continue
        if r["repo"] in held_repos or not r["problem_statement"] or not r["test_patch"] or not r["base_commit"]:
            continue
        if not re.search(r"\b(mit|apache|bsd|isc)\b", r["license_name"] or "", re.I) or len(r["test_patch"]) > 9000:
            continue
        if per.get(r["repo"], 0) >= a.per_repo:
            continue
        per[r["repo"]] = per.get(r["repo"], 0) + 1
        train.append(r)
        if len(train) >= a.train_limit * 1.3:
            break
    a.out.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        built = [b for b in ex.map(build, train) if b]
        evb = [b for b in ex.map(lambda r: build(r, eval_mode=True), ev.values()) if b]
    built = built[: a.train_limit]
    if not a.eval_only:
        (a.out / "train.jsonl").write_text("".join(json.dumps(b) + "\n" for b in built))
    (a.out / "eval_prompts_labelled.jsonl").write_text("".join(json.dumps({"id": b["id"], "prompt": b["prompt"]}) + "\n" for b in evb))
    print(json.dumps({"train": len(built), "train_repos": len({b["id"].rsplit("-", 1)[0] for b in built}), "eval": len(evb), "eval_wanted": len(ev)}))


if __name__ == "__main__":
    main()
