"""A task: one planted bug in one library, one named failing test shown to the judge, everything else hidden."""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from vm.sandbox.runner import run_pytest
from vm.tasks.mutate import mutants, pick_heldout


_PARSED: dict = {}


def _functions_of(file: Path) -> dict:
    """name -> source of the first function with that name in a file; parsed once per process."""
    key = str(file)
    if key not in _PARSED:
        text = file.read_text()
        found = {}
        for n in ast.walk(ast.parse(text)):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name not in found:
                found[n.name] = ast.get_source_segment(text, n)
        _PARSED[key] = found
    return _PARSED[key]


def _test_source(src: Path, node_id: str) -> str:
    path, _, name = node_id.partition("::")
    name = name.split("::")[-1].split("[")[0]
    return _functions_of(src / path).get(name, name)


def _strip_doctests(text: str) -> str:
    out, skip = [], False
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith(">>>") or s.startswith("..."):
            skip = True
            continue
        if skip and s and not s.startswith(">>>"):
            if not s:
                skip = False
            continue
        skip = False
        out.append(ln)
    return "\n".join(out)


def _function_lines(source: str, start: int, end: int) -> list[str]:
    lines = source.splitlines()[start - 1 : end]
    text = _strip_doctests("\n".join(lines))
    return text.splitlines()


def make_tasks(lib: dict, root: Path, max_tasks: int = 12, per_function: int = 2, max_tries: int = 40) -> list[dict]:
    held = pick_heldout()["operator"]
    src = root / lib["name"] / "src"
    venv = root / lib["name"] / "venv"
    test_files = {t.split("::")[0] for t in lib["tests"]}
    candidates = []
    for py in sorted(src.rglob("*.py")):
        rel = str(py.relative_to(src))
        if rel in test_files or "/tests/" in f"/{rel}" or rel.startswith(("test", "docs/", "setup.py")):
            continue
        for m in mutants(py.read_text(), rel):
            if m.operator == held and lib["split"] != "test":
                continue
            candidates.append(m)
    candidates.sort(key=lambda m: hashlib.sha256(f"vm-mutant:{lib['name']}:{m.path}:{m.line}:{m.operator}:{m.new_source[-200:]}".encode()).hexdigest())
    tasks, per_fn, tried = [], {}, 0
    for m in candidates:
        if len(tasks) >= max_tasks or tried >= max_tries:
            break
        key = (m.path, m.function)
        if per_fn.get(key, 0) >= per_function:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "src"
            shutil.copytree(src, work)
            (work / m.path).write_text(m.new_source)
            r = run_pytest(work, venv, node_ids=lib["tests"], timeout=90, maxfail=7)
        tried += 1
        failing = sorted(t for t, o in r.outcomes.items() if o != "passed")
        if not 1 <= len(failing) <= 6:
            continue
        if r.timed_out or len(r.outcomes) < len(lib["tests"]) * 0.9:  # a short run is only trusted when it ran everything
            continue
        failing = sorted(t for t in lib["tests"] if r.outcomes.get(t) != "passed")
        named = sorted(failing, key=lambda t: hashlib.sha256(f"vm-named:{t}".encode()).hexdigest())[0]
        visible = sorted(set(lib["visible"]) - set(failing) | {named})
        oracle_only = sorted(set(lib["tests"]) - set(visible))
        original = (src / m.path).read_text()
        fn_lines = _function_lines(m.new_source, m.function_start, m.function_end)
        orig_lines = _function_lines(original, m.function_start, m.function_end)
        bug_line = next((i for i, (a, b) in enumerate(zip(fn_lines, orig_lines)) if a != b), 0)
        named_src = _test_source(src, named)
        text = (f"In `{m.path}`, the test `{named}` fails.\n\nThe test:\n{named_src}\n\n"
                f"The function `{m.function}` it exercises:\n")
        per_fn[key] = per_fn.get(key, 0) + 1
        tasks.append({"task_id": f"{lib['name']}:{len(tasks)}", "library": lib["name"], "split": lib["split"],
                      "operator": m.operator, "path": m.path, "function": m.function,
                      "function_start": m.function_start, "function_end": m.function_end,
                      "broken_source": m.new_source, "original_source": original, "failing": failing,
                      "named_test": named, "named_test_source": named_src, "visible": visible,
                      "oracle_only": oracle_only, "task_text": text, "function_lines": fn_lines, "bug_line": bug_line})
    return tasks


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1])
    for line in (root / "qualified.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec["ok"]:
            out = root / rec["name"] / "tasks.json"
            if not out.is_file():
                out.write_text(json.dumps(make_tasks(rec, root)))
