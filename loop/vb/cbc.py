"""Correct-by-construction pilot: tasks made by breaking known-good code with a fixed mechanical rule.

Source: the `portion` library (interval arithmetic, pure Python, 209 passing tests), in /root/vbench/cbc.
Rule, fixed before any task was seen:
  - mutants: every comparison flip (< <-> <=, > <-> >=) and every `and` <-> `or` swap in portion/*.py;
  - keep a mutant only if 1 to 6 existing tests fail on it (subtle, not broken everywhere);
  - the first qualifying mutant in each function, functions in source order; the first 5 functions.
Each task: the broken library at /app, its tests minus the failing ones (FAIL_TO_PASS), and a task text showing one
failing test as the reproduction. The oracle (the separate verifier) runs the library's full original test suite:
an attempt is correct if every FAIL_TO_PASS and PASS_TO_PASS test passes.
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
from pathlib import Path

from .util import DATA, write_json
from .writer import dump_toml

CBC = DATA / "cbc"
SRC = CBC / "env" / "app"
IMAGE = "cbc-portion"
FLIPS = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt}


def mutants(path: Path) -> list[dict]:
    """Every single-site mutant in one file, as (line, col, kind, new source)."""
    src = path.read_text()
    tree = ast.parse(src)
    funcs = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append((n.lineno, n.end_lineno, n.name))
    def owner(line):
        best = None
        for a, b, name in funcs:
            if a <= line <= b and (best is None or a > best[0]):
                best = (a, name)
        return best[1] if best else None
    lines = src.splitlines(keepends=True)
    starts = [0]
    for l in lines:
        starts.append(starts[-1] + len(l))
    def off(line, col):                       # AST columns are byte offsets; portion's sources are ASCII
        return starts[line - 1] + col
    out = []
    def edit(a, b, old, new_tok):
        """Replace the single operator token `old` found between offsets a and b; None if not exactly one."""
        seg = src[a:b]
        import re as _re
        hits = [m for m in _re.finditer(r"(?<![<>=!])" + _re.escape(old) + r"(?![<>=])" if old in ("<", ">") else
                                        r"\b" + old + r"\b" if old in ("and", "or") else _re.escape(old), seg)]
        if len(hits) != 1:
            return None
        h = hits[0]
        return src[:a + h.start()] + new_tok + src[a + h.end():]
    sym = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare):
            prev = n.left
            for op, right in zip(n.ops, n.comparators):
                if type(op) in FLIPS:
                    new = edit(off(prev.end_lineno, prev.end_col_offset), off(right.lineno, right.col_offset),
                               sym[type(op)], sym[FLIPS[type(op)]])
                    if new:
                        out.append({"file": "portion/" + path.name, "line": n.lineno, "col": right.col_offset,
                                    "kind": "compare", "function": owner(n.lineno), "source": new})
                prev = right
        elif isinstance(n, ast.BoolOp):
            word, other = ("and", "or") if isinstance(n.op, ast.And) else ("or", "and")
            a, b = n.values[0], n.values[1]
            new = edit(off(a.end_lineno, a.end_col_offset), off(b.lineno, b.col_offset), word, other)
            if new:
                out.append({"file": "portion/" + path.name, "line": n.lineno, "col": n.col_offset,
                            "kind": "boolop", "function": owner(n.lineno), "source": new})
    return sorted(out, key=lambda m: (m["file"], m["line"], m["col"]))


def run_tests(workdir: Path) -> dict[str, str]:
    """{test id: outcome} for the library at workdir, run in the task image."""
    # Capped: a mutant can make the library loop forever; one such run took the server's 15 GB and SSH with it.
    try:
        r = subprocess.run(["docker", "run", "--rm", "--memory", "1g", "--memory-swap", "1g", "--cpus", "2",
                            "-v", f"{workdir}:/app", IMAGE, "bash", "-c",
                            "cd /app && timeout 90 python -m pytest -q -p no:cacheprovider -rA tests 2>&1 | "
                            "grep -E '^(PASSED|FAILED|ERROR) ' || true"], capture_output=True, text=True, timeout=150)
    except subprocess.TimeoutExpired:
        return {}
    res = {}
    for line in r.stdout.splitlines():
        outcome, _, rest = line.partition(" ")
        res[rest.split(" - ")[0].strip()] = outcome
    return res


def select(n: int = 5, max_fail: int = 6) -> list[dict]:
    base = run_tests(SRC)
    assert base and all(v == "PASSED" for v in base.values()), "the original library must pass its own tests"
    chosen, seen = [], set()
    work = CBC / "_mutwork"
    for f in sorted((SRC / "portion").glob("*.py")):
        for m in mutants(f):
            if m["function"] in seen or m["function"] is None:
                continue
            if work.exists():
                shutil.rmtree(work)
            shutil.copytree(SRC, work)
            (work / m["file"]).write_text(m["source"])
            got = run_tests(work)
            failing = sorted(t for t, v in got.items() if v != "PASSED" and base.get(t) == "PASSED")
            if 1 <= len(failing) <= max_fail and len(got) == len(base):
                chosen.append({**{k: m[k] for k in ("file", "line", "col", "kind", "function")},
                               "fail_to_pass": failing, "source": m["source"]})
                seen.add(m["function"])
                print("kept", m["file"], m["line"], m["function"], m["kind"], len(failing), flush=True)
                if len(chosen) == n:
                    return chosen
    return chosen


INSTRUCTION = """\
The `portion` library in `/app` (interval arithmetic for Python, installed in editable mode) has a bug. This test,
which passes on a correct version of the library, now fails:

```python
{repro}
```

Fix the library so that this test and all other behaviour of the library is correct again. Change only the library
code under `/app/portion`; do not edit or add tests.
"""


def _test_functions(path: Path) -> dict[str, tuple[int, int]]:
    """Map 'file::[Class::]func' to its (first line incl. decorators, last line)."""
    tree = ast.parse(path.read_text())
    out = {}
    def visit(body, prefix):
        for n in body:
            if isinstance(n, ast.ClassDef):
                visit(n.body, prefix + n.name + "::")
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test"):
                first = min([n.lineno] + [d.lineno for d in n.decorator_list])
                out[prefix + n.name] = (first, n.end_lineno)
    visit(tree.body, "")
    return out


def build_tasks(out_root: Path | None = None) -> list[Path]:
    import textwrap
    out_root = out_root or CBC / "tasks"
    ms = json.loads((CBC / "mutants.json").read_text())
    made = []
    for k, m in enumerate(ms, 1):
        name = f"cbc-portion-{k}"
        out = out_root / name
        if out.exists():
            shutil.rmtree(out)
        # the solver's machine: broken library, tests minus the FAIL_TO_PASS ones
        env_app = out / "environment" / "app"
        shutil.copytree(SRC, env_app)
        (env_app / m["file"]).write_text(m["source"])
        by_file: dict[str, list[str]] = {}
        for t in m["fail_to_pass"]:
            f, _, rest = t.partition("::")
            by_file.setdefault(f, []).append(rest)
        repro = None
        for f, names in by_file.items():
            p = env_app / f
            spans = _test_functions(p)
            lines = p.read_text().splitlines(keepends=True)
            drop = set()
            for nm in names:
                nm = nm.split("[")[0]                       # parametrised ids
                a, b = spans[nm]
                if repro is None:
                    repro = textwrap.dedent("".join(lines[a - 1:b])).rstrip()
                drop.update(range(a, b + 1))
            p.write_text("".join(l for i, l in enumerate(lines, 1) if i not in drop))
        shutil.copy2(CBC / "env" / "Dockerfile", out / "environment" / "Dockerfile")
        # the oracle: the library's full original test suite, in its own image
        tests = out / "tests"
        shutil.copytree(SRC, tests / "app")
        (tests / "Dockerfile").write_text((CBC / "env" / "Dockerfile").read_text().replace(
            "COPY app /app", "COPY app /app\nCOPY test.sh /tests/test.sh"))
        (tests / "test.sh").write_text(
            "#!/bin/bash\nmkdir -p /logs/verifier\ncd /app\n"
            "if python -m pytest -q -p no:cacheprovider tests > /logs/verifier/pytest.txt 2>&1; then echo 1; "
            "else echo 0; fi > /logs/verifier/reward.txt\ntail -3 /logs/verifier/pytest.txt\n")
        (tests / "test.sh").chmod(0o755)
        # the reference fix: the original file
        sol = out / "solution"
        sol.mkdir()
        shutil.copy2(SRC / m["file"], sol / Path(m["file"]).name)
        (sol / "solve.sh").write_text(f"#!/bin/bash\ncp /solution/{Path(m['file']).name} /app/{m['file']}\n")
        (sol / "solve.sh").chmod(0o755)
        (out / "instruction.md").write_text(INSTRUCTION.format(repro=repro))
        doc = {"schema_version": "2.0", "artifacts": ["/app/portion/"],
               "task": {"name": f"vbench-cbc/{name}", "description": ""},
               "metadata": {"source": "portion 2.6.2", "mutant_file": m["file"], "mutant_line": m["line"],
                            "mutant_kind": m["kind"], "function": m["function"],
                            "fail_to_pass": len(m["fail_to_pass"])},
               "agent": {"timeout_sec": 1800.0, "network_mode": "allowlist", "allowed_hosts": ["inference.hud.ai"]},
               "verifier": {"timeout_sec": 300.0, "environment_mode": "separate", "network_mode": "no-network"},
               "environment": {"build_timeout_sec": 600.0, "cpus": 2, "memory_mb": 4096}}
        (out / "task.toml").write_text(dump_toml(doc))
        made.append(out)
    return made


SOLVERS = {
    "haiku-cc": {"base": "sonnet-claude-code", "model": "anthropic/claude-haiku-4-5",
                 "kwargs": ["disallowed_tools=WebFetch,WebSearch", "max_budget_usd=1"], "attempts": 4},
    "luna-codex": {"base": "luna-codex", "attempts": 3},
    "luna-terminus": {"base": "luna-terminus", "attempts": 3},
}


def run_solvers() -> dict:
    """Solver attempts on every task, labelled by each trial's own verifier (the full original test suite)."""
    from .util import load_env, new_run_dir
    from .writer import ALL_WRITERS, run_writer
    load_env()
    run_dir = new_run_dir("cbc_solvers")
    jobs = {}
    for name, s in SOLVERS.items():
        w = dict(ALL_WRITERS[s["base"]])
        if "model" in s:
            w["model"] = s["model"]
        if "kwargs" in s:
            w["kwargs"] = s["kwargs"]
        ALL_WRITERS[name] = w
        jobs[name] = str(run_writer(name, CBC / "tasks", run_dir, s["attempts"], n_concurrent=5))
        print("done", name, flush=True)
    write_json(CBC / "solver_jobs.json", jobs)
    return jobs


def near_misses(k: int, per_task: int = 6) -> list[dict]:
    """Synthetic near misses for task k, by rule: every other one-token change (same operator set) to the broken
    file that makes the task's shown failing test pass but still fails the full original suite; the first
    `per_task` in source order. Labelled synthetic wherever they are used."""
    ms = json.loads((CBC / "mutants.json").read_text())
    m = ms[k - 1]
    repro = m["fail_to_pass"][0]
    broken_dir = CBC / "_nm_base"
    if broken_dir.exists():
        shutil.rmtree(broken_dir)
    shutil.copytree(SRC, broken_dir)
    (broken_dir / m["file"]).write_text(m["source"])
    original = (SRC / m["file"]).read_text()
    out, work = [], CBC / "_nm_work"
    # (i) compensating changes: one other token of the broken file; (ii) regressions: the true fix plus one other
    # one-token change in the same file. Both must make the shown test pass and still fail the full suite.
    fixed_dir = CBC / "_nm_fixed"
    if fixed_dir.exists():
        shutil.rmtree(fixed_dir)
    shutil.copytree(SRC, fixed_dir)
    comp = [dict(c, origin="compensating") for c in mutants(broken_dir / m["file"])]
    regr = [dict(c, origin="regression") for c in mutants(fixed_dir / m["file"]) if c["source"] != m["source"]]
    for cand in comp + regr:
        if cand["source"] == original:
            continue                                              # the true fix
        if work.exists():
            shutil.rmtree(work)
        shutil.copytree(SRC, work)                                # full original tests
        (work / m["file"]).write_text(cand["source"])
        got = run_tests(work)
        if len(got) < 200:                                        # hung or crashed: not a usable near miss
            continue
        if got.get(repro) == "PASSED" and any(v != "PASSED" for v in got.values()):
            out.append({"task": k, "file": m["file"], "line": cand["line"], "kind": cand["kind"],
                        "origin": cand["origin"],
                        "failing": sorted(t for t, v in got.items() if v != "PASSED"), "source": cand["source"]})
            print("near miss", k, cand["origin"], cand["line"], cand["kind"], len(out[-1]["failing"]), flush=True)
            if len(out) == per_task:
                break
    return out
