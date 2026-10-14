"""Step 4: pair each harvested attempt with the requirements it met and broke.

A task's requirements are read from its official tests: one per test function (pytest node), with the test's
docstring and its assert lines as the requirement's wording. An attempt's broken requirements are the tests it
failed (ctrf). A wrong attempt that fails exactly one requirement is a real near miss: correct except for one
thing, the case agent-written graders miss most (step 3: "identical" read as close, "minimize" never enforced).

Outputs per task, beside the harvest: requirements.json (the list) and pairs.jsonl (one row per attempt:
label, requirements broken, near-miss flag). Prints the dataset summary.

    python -m vb.requirements                                   # runs on the server, after vb.harvest
"""
from __future__ import annotations

import ast
import json
import re

from .harvest import OUT


def requirements(task_json: dict) -> dict:
    """node id -> {file, test, doc, asserts}; node ids as pytest/ctrf name them (file::Class::test)."""
    reqs = {}
    for path, src in task_json["official_tests"].items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        lines = src.splitlines()
        fname = path.rsplit("/", 1)[-1]

        def visit(nodes, prefix):
            for n in nodes:
                if isinstance(n, ast.ClassDef) and n.name.startswith("Test"):
                    visit(n.body, prefix + [n.name])
                elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test"):
                    asserts = [lines[a.lineno - 1].strip()[:200] for a in ast.walk(n) if isinstance(a, ast.Assert)][:8]
                    reqs["::".join([fname] + prefix + [n.name])] = {
                        "file": path, "test": n.name, "doc": (ast.get_docstring(n) or "").strip()[:600], "asserts": asserts}

        visit(tree.body, [])
    return reqs


def norm(name: str) -> str:
    """ctrf names carry parametrization ([...]) and sometimes a directory; requirements are keyed without them."""
    name = re.sub(r"\[.*\]$", "", name)
    parts = name.split("::")
    parts[0] = parts[0].rsplit("/", 1)[-1]
    return "::".join(parts)


def main():
    summary = {"tasks": 0, "attempts": 0, "pass": 0, "fail": 0, "fail_with_test_detail": 0, "near_miss": 0,
               "unmapped_test_names": 0}
    per_task = {}
    for d in sorted(p for p in OUT.iterdir() if (p / "task.json").is_file()):
        task = json.loads((d / "task.json").read_text())
        reqs = requirements(task)
        (d / "requirements.json").write_text(json.dumps(reqs, indent=1))
        rows = []
        for f in sorted((d / "attempts").glob("*.json")):
            a = json.loads(f.read_text())
            failed = sorted({norm(t["name"]) for t in a["tests"] if t.get("status") not in ("passed", "skipped")})
            names = {norm(t["name"]) for t in a["tests"]}
            summary["unmapped_test_names"] += len([n for n in names if n not in reqs])
            row = {"trial_id": a["trial_id"], "model": a["model"], "label": a["label"], "tests_run": len(names),
                   "broken": failed, "near_miss": a["label"] == "fail" and len(names) > 1 and len(failed) == 1}
            rows.append(row)
            summary["attempts"] += 1
            summary[a["label"]] += 1
            summary["fail_with_test_detail"] += a["label"] == "fail" and bool(failed)
            summary["near_miss"] += row["near_miss"]
        (d / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        per_task[d.name] = {"requirements": len(reqs), "attempts": len(rows),
                            "near_misses": sum(r["near_miss"] for r in rows),
                            "most_broken": sorted({b for r in rows for b in r["broken"]},
                                                  key=lambda b: -sum(b in r["broken"] for r in rows))[:3]}
        summary["tasks"] += 1
    print(json.dumps({"summary": summary, "per_task": per_task}, indent=1))


if __name__ == "__main__":
    main()
