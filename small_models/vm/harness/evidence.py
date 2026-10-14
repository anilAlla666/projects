"""Evidence lines, gathered by running code: visible tests, test edits, diff shape and differential fuzzing.
Nothing here reads an attempt's own claims; those go to the untrusted segment."""
from __future__ import annotations

import ast
import difflib
import json
import re
import shutil
import tempfile
from pathlib import Path

from vm.sandbox.runner import run_pytest, run_python


def unified_diff(old_files: dict, new_files: dict) -> str:
    out = []
    for path in sorted(set(old_files) | set(new_files)):
        a, b = old_files.get(path, ""), new_files.get(path, "")
        if a != b:
            out += difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True), f"a/{path}", f"b/{path}", n=2)
    return "".join(out)


_STRING_LINE = re.compile(r"""^\s*(?:[rbuRBU]{0,2})(?:'[^']*'|"[^"]*"|'''.*'''|\"\"\".*\"\"\")\s*$""")


def split_untrusted(diff: str):
    shown, untrusted = [], []
    for ln in diff.splitlines():
        if ln.startswith("+") and not ln.startswith("+++"):
            body = ln[1:]
            if body.strip().startswith("#") or _STRING_LINE.match(body):
                untrusted.append(body.strip())
                shown.append("+" + body[: len(body) - len(body.lstrip())] + "# [moved to untrusted]")
                continue
            if "  #" in body:
                code, _, comment = body.partition("  #")
                untrusted.append("#" + comment)
                shown.append("+" + code + "  # [moved to untrusted]")
                continue
        shown.append(ln)
    return "\n".join(shown) + ("\n" if shown else ""), untrusted


def diff_shape(diff: str) -> str:
    files = {ln[6:] for ln in diff.splitlines() if ln.startswith("+++ b/")}
    added = sum(1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---"))
    return f"diff: {len(files)} file{'s' if len(files) != 1 else ''}, +{added} -{removed} lines"


def touches_tests(files: dict, test_paths: set) -> bool:
    return any(p in test_paths or Path(p).name.startswith("test_") or "/tests/" in f"/{p}" for p in files)


def visible_lines(outcomes: dict, visible: list) -> list[str]:
    passed = sum(outcomes.get(t) == "passed" for t in visible)
    lines = [f"visible tests: {passed} of {len(visible)} pass"]
    lines += [f"{t}: {outcomes.get(t, 'not run')}" for t in visible if outcomes.get(t) != "passed"][:20]
    return lines


FUZZ_RUNNER = r'''
import importlib, json, random, sys, inspect
random.seed(SEED)
mod = importlib.import_module(MODULE)
pool = LITERALS + [0, 1, -1, 2, 3, 10, -5, 0.5, "", "a", "abc", [], [1, 2], (1,), None, True, False, {}]
out = {}
for name in FUNCTIONS:
    f = getattr(mod, name, None)
    if f is None or not callable(f):
        continue
    try:
        n_args = len([p for p in inspect.signature(f).parameters.values()
                      if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    except (TypeError, ValueError):
        continue
    if n_args > 4:
        continue
    res = []
    for _ in range(N):
        args = [random.choice(pool) for _ in range(n_args)]
        try:
            res.append(["ok", repr(f(*args))[:200]])
        except Exception as e:
            res.append(["err", type(e).__name__])
    out[name] = res
print("__FUZZ__" + json.dumps(out))
'''


def fuzz_script(module: str, functions: list, literals: list, n: int = 100, seed: int = 0) -> str:
    return (FUZZ_RUNNER.replace("SEED", str(seed)).replace("MODULE", repr(module))
            .replace("LITERALS", repr(literals)).replace("FUNCTIONS", repr(functions)).replace("N)", f"{n})"))


def fuzz_lines(broken_out: dict, patched_out: dict) -> list[str]:
    lines = []
    for name in sorted(set(broken_out) & set(patched_out)):
        a, b = broken_out[name], patched_out[name]
        k = sum(x != y for x, y in zip(a, b))
        lines.append(f"fuzz {name}: outputs identical on {len(a)} inputs" if k == 0
                     else f"fuzz {name}: outputs differ on {k} of {len(a)} inputs")
    return lines


def _literals(test_sources: list) -> list:
    vals = []
    for s in test_sources:
        try:
            for n in ast.walk(ast.parse(s)):
                if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, str)) and len(repr(n.value)) < 40:
                    vals.append(n.value)
        except SyntaxError:
            pass
    return list(dict.fromkeys(vals))[:50]


def _changed_functions(old: str, new: str) -> list[str]:
    def fns(src):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return {}
        return {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    a, b = fns(old), fns(new)
    return sorted(k for k in a if k in b and a[k] != b[k])


def _run_fuzz(src_root: Path, venv: Path, module: str, functions: list, literals: list) -> dict:
    code, stdout = run_python(src_root, venv, fuzz_script(module, functions, literals), timeout=60)
    for ln in stdout.splitlines():
        if ln.startswith("__FUZZ__"):
            return json.loads(ln[len("__FUZZ__"):])
    return {}


def gather(task: dict, attempt_files: dict, lib_root: Path) -> tuple[list[dict], dict]:
    """Returns (evidence lines, raw visible outcomes). Runs the visible tests on the patched tree and fuzzes the
    changed module-level functions of the task's file, broken against patched. Literals come from the visible
    named test only, never from oracle-only tests."""
    src, venv = lib_root / task["library"] / "src", lib_root / task["library"] / "venv"
    lines = []
    with tempfile.TemporaryDirectory() as tmp:
        broken = Path(tmp) / "broken"
        patched = Path(tmp) / "patched"
        shutil.copytree(src, broken)
        (broken / task["path"]).write_text(task["broken_source"])
        shutil.copytree(broken, patched)
        for p, text in attempt_files.items():
            (patched / p).parent.mkdir(parents=True, exist_ok=True)
            (patched / p).write_text(text)
        r = run_pytest(patched, venv, node_ids=task["visible"], timeout=90)
        lines += visible_lines(r.outcomes, task["visible"]) if not r.timed_out else ["visible tests: timed out"]
        old_files = {task["path"]: task["broken_source"]}
        test_paths = {t.split("::")[0] for t in task["visible"] + task["oracle_only"]}
        if touches_tests(attempt_files, test_paths):
            lines.append("test files edited: " + ", ".join(sorted(p for p in attempt_files if touches_tests({p: ""}, test_paths))))
        else:
            lines.append("test files edited: none")
        new_src = attempt_files.get(task["path"], task["broken_source"])
        fns = _changed_functions(task["broken_source"], new_src)
        if fns and task["path"].endswith(".py"):
            module = task["path"][:-3].replace("/", ".").removeprefix("src.")
            lits = _literals([task["named_test_source"]])
            a = _run_fuzz(broken, venv, module, fns, lits)
            b = _run_fuzz(patched, venv, module, fns, lits)
            lines += fuzz_lines(a, b) or ["fuzz: not applicable"]
        else:
            lines.append("fuzz: not applicable")
    diff = unified_diff({**old_files, **{p: (src / p).read_text() if (src / p).is_file() else "" for p in attempt_files if p != task["path"]}},
                        attempt_files)
    lines.append(diff_shape(diff))
    return [{"id": i, "text": t} for i, t in enumerate(lines)], r.outcomes
