"""Standalone grader targets. A maintainer test patch usually edits an existing test file, which a writer can only
reproduce as a diff if it sees that file exactly; a small model's diffs then fail to apply. Here each patched test file
becomes one NEW file beside it: the patched file's imports and helpers plus exactly the tests the patch added or
changed. The prompts are the context prompts with the instruction to add new test files.

    python gw/standalone.py RAW_DIR DATA3_DIR OUT_DIR
"""
from __future__ import annotations

import ast
import concurrent.futures as cf
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context  # noqa: E402

INSTRUCTION_OLD = "Reply with a unified diff against the test files above, or one that adds a new test file"
INSTRUCTION = ("Reply with a unified diff that adds one or more new, self-contained test files next to the existing tests "
               "(with the imports and helpers they need, and only the tests that decide this issue); do not edit existing files")
CAP = 9000


def apply_file_patch(old: str, diff: str) -> str | None:
    """Apply one file's unified-diff hunks to its old text, exactly."""
    old_lines, new, i = old.splitlines(), [], 0
    for h in re.split(r"(?m)^(?=@@ )", diff):
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@.*\n?", h)
        if not m:
            continue
        start = int(m.group(1)) - (0 if m.group(2) == "0" else 1)
        new += old_lines[i:start]
        i = start
        for ln in h[m.end():].splitlines():
            if ln.startswith("\\"):
                continue
            tag, body = ln[:1], ln[1:]
            if tag == " ":
                if i >= len(old_lines) or old_lines[i] != body:
                    return None
                new.append(body)
                i += 1
            elif tag == "-":
                if i >= len(old_lines) or old_lines[i] != body:
                    return None
                i += 1
            elif tag == "+":
                new.append(body)
    return "\n".join(new + old_lines[i:]) + "\n"


def file_diffs(patch: str) -> dict:
    """test path -> (is_new_file, that file's hunks text)."""
    out = {}
    for part in re.split(r"(?m)^(?=diff --git )", patch):
        m = re.search(r"^--- (\S+)\n\+\+\+ b/(\S+)", part, re.M)
        if not m:
            continue
        out[m.group(2)] = (m.group(1) == "/dev/null", part[part.find("@@"):] if "@@" in part else "")
    return out


def standalone(old: str, new: str) -> str | None:
    """Imports, helpers and fixtures of the patched file, plus the tests that are new or changed."""
    try:
        to, tn = ast.parse(old) if old else ast.Module(body=[], type_ignores=[]), ast.parse(new)
    except SyntaxError:
        return None
    def full(n, lines):  # the node with its decorators: an added parametrize case is a changed test
        start = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
        return "\n".join(lines[start - 1:n.end_lineno])
    fns = (ast.FunctionDef, ast.AsyncFunctionDef)
    is_fn = lambda n: isinstance(n, fns) and n.name.startswith("test")  # noqa: E731
    def is_cls(n):  # by name, by a *TestCase base, or by holding test methods
        if not isinstance(n, ast.ClassDef):
            return False
        bases = [ast.unparse(b) for b in n.bases]
        return (n.name.startswith("Test") or n.name.endswith(("Test", "Tests", "TestCase"))
                or any(b.endswith("TestCase") for b in bases) or any(is_fn(m) for m in n.body))
    old_lines, lines = old.splitlines(), new.splitlines()
    old_nodes = {n.name: n for n in to.body if isinstance(n, fns + (ast.ClassDef,))}
    keep, tests, skipped, kept_cls = [], 0, {}, []
    for idx, n in enumerate(tn.body):
        text = full(n, lines)
        o = old_nodes.get(getattr(n, "name", None))
        if is_fn(n):
            if o is None or full(o, old_lines) != text:
                keep.append((idx, text))
                tests += 1
        elif is_cls(n):
            if o is not None and full(o, old_lines) == text:
                skipped[n.name] = (idx, text, n)
                continue
            tests += 1
            kept_cls.append(n)
            first = n.body[0]
            header_end = min([first.lineno] + [d.lineno for d in getattr(first, "decorator_list", [])]) - 1
            start = min([n.lineno] + [d.lineno for d in n.decorator_list])
            if o is None or not isinstance(o, ast.ClassDef) or header_end < start:
                keep.append((idx, text))
                continue
            old_m = {m.name: full(m, old_lines) for m in o.body if isinstance(m, fns)}
            parts, changed = ["\n".join(lines[start - 1:header_end])], 0
            for m in n.body:  # inside a changed class: its setup and helpers, and only the test methods that changed
                t = full(m, lines)
                if is_fn(m):
                    if old_m.get(m.name) != t:
                        parts.append(t)
                        changed += 1
                else:
                    parts.append(t)
            keep.append((idx, parts[0].rstrip() + "\n" + "\n\n".join(parts[1:]) if changed else text))
        else:
            keep.append((idx, text))
    if not tests:
        return None
    while kept_cls:  # an unchanged test class that a kept class inherits from comes back, or the file will not import
        c = kept_cls.pop()
        for b in c.bases:
            name = ast.unparse(b)
            if name in skipped:
                i, t, node = skipped.pop(name)
                keep.append((i, t))
                kept_cls.append(node)
    keep = [t for _, t in sorted(keep, key=lambda x: x[0])]
    body = ""
    for i, k in enumerate(keep):  # consecutive imports stay together, everything else is two blank lines apart
        imp = lambda s: s.startswith(("import ", "from "))  # noqa: E731
        body += ("" if i == 0 else "\n" if imp(k) and imp(keep[i - 1]) else "\n\n\n") + k
    body += "\n"
    return body if len(body) <= CAP else None


def new_file_diff(path: str, content: str) -> str:
    ls = content.rstrip("\n").split("\n")
    return f"diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{len(ls)} @@\n" + "".join("+" + l + "\n" for l in ls)


def target(row: dict) -> str | None:
    d = context.clone(row["repo"])
    if d is None:
        return None
    parts = []
    for path, (is_new, hunks) in file_diffs(row["test_patch"]).items():
        if not path.endswith(".py"):
            continue
        old = "" if is_new else context.show(d, row["base_commit"], path)
        if old is None:
            continue
        new = apply_file_patch(old, hunks)
        if new is None:
            continue
        s = standalone(old, new)
        if s is None:
            continue
        p = Path(path)
        stem = p.stem if p.stem.startswith("test") else "test_" + p.stem
        dest = str(p.parent / (stem + "_grader.py")) if not is_new else path
        parts.append(new_file_diff(dest, s))
    return "".join(parts) if parts else None


def main():
    raw, data3, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    import pyarrow.parquet as pq
    cols = ["instance_id", "repo", "test_patch", "base_commit"]
    rows = {r["instance_id"]: r for f in sorted(glob.glob(str(raw / "*SWE-rebench*.parquet"))) for r in pq.read_table(f, columns=cols).to_pylist()}
    train = [json.loads(l) for l in (data3 / "train.jsonl").read_text().splitlines()]
    with cf.ThreadPoolExecutor(48) as ex:
        targets = list(ex.map(lambda t: target(rows[t["id"]]) if t["id"] in rows else None, train))
    kept = [{"id": t["id"], "prompt": t["prompt"].replace(INSTRUCTION_OLD, INSTRUCTION), "completion": tg}
            for t, tg in zip(train, targets) if tg]
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("".join(json.dumps(k) + "\n" for k in kept))
    ev = [json.loads(l) for l in (data3 / "eval_prompts_labelled.jsonl").read_text().splitlines()]
    (out / "eval_prompts_labelled.jsonl").write_text("".join(json.dumps({"id": e["id"], "prompt": e["prompt"].replace(INSTRUCTION_OLD, INSTRUCTION)}) + "\n" for e in ev))
    print(json.dumps({"train": len(kept), "of": len(train), "eval": len(ev),
                      "instruction_replaced": sum(INSTRUCTION in k["prompt"] for k in kept),
                      "eval_instruction_replaced": sum(INSTRUCTION_OLD in e["prompt"] for e in ev)}))


if __name__ == "__main__":
    main()
