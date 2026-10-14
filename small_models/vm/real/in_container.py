"""Runs inside a SWE-rebench instance image, with /work mounted. Applies one patch the SWE-bench way (the patch,
then the instance's test patch over test files reset to the base commit), runs the oracle tests with the results
plugin, and writes /work/out/result.json. Standard library only: it runs under the instance's own interpreter."""
import ast
import json
import os
import subprocess
import sys

W, T = "/work", "/testbed"
spec = json.load(open(f"{W}/spec.json"))
os.makedirs(f"{W}/out", exist_ok=True)
out = {"mode": spec["mode"]}


def sh(*a, **k):
    return subprocess.run(list(a), cwd=T, capture_output=True, text=True, **k)


def touched(patch):
    ps = []
    for ln in patch.splitlines():
        if ln.startswith("+++ b/") or ln.startswith("--- a/"):
            p = ln[6:].split("\t")[0].strip()
            if p not in ps:
                ps.append(p)
    return ps


def read(p):
    try:
        with open(os.path.join(T, p), encoding="utf-8", errors="replace") as f:
            return f.read()
    except (FileNotFoundError, IsADirectoryError):
        return ""


def enclosing(src, line):
    """The smallest function containing a line, as (name, start, end); start includes decorators."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    best = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno <= line <= n.end_lineno:
            start = min([n.lineno] + [d.lineno for d in n.decorator_list])
            if best is None or n.end_lineno - start < best[2] - best[1]:
                best = (n.name, start, n.end_lineno)
    return best


def finish():
    with open(f"{W}/out/result.json", "w") as f:
        json.dump(out, f)
    os.chmod(f"{W}/out/result.json", 0o666)
    sys.exit(0)


g = spec["gold"]
gsrc = read(g["path"])
fn = enclosing(gsrc, g["line"])
if fn:
    out["gold_function"] = {"path": g["path"], "name": fn[0], "start": fn[1], "end": fn[2],
                            "lines": gsrc.splitlines()[fn[1] - 1:fn[2]], "changed_line": g["line"]}
elif gsrc:  # the reference fix is outside any function: show the surrounding module code instead
    lo, hi = max(1, g["line"] - 12), min(len(gsrc.splitlines()), g["line"] + 12)
    out["gold_function"] = {"path": g["path"], "name": f"module code, lines {lo}-{hi}", "start": lo, "end": hi,
                            "lines": gsrc.splitlines()[lo - 1:hi], "changed_line": g["line"], "module_level": True}

patch = open(f"{W}/attempt.patch").read() if spec["mode"] != "base" else ""
files = touched(patch)
out["files"] = files
out["old"] = {p: read(p) for p in files}
if patch:
    r = sh("git", "apply", "--whitespace=nowarn", f"{W}/attempt.patch")
    if r.returncode:
        r = sh("git", "apply", "--whitespace=nowarn", "--3way", f"{W}/attempt.patch")
    if r.returncode:
        out["apply_failed"] = (r.stderr or r.stdout)[-600:]
        finish()
out["new"] = {p: read(p) for p in files}

test_patch = open(f"{W}/test.patch").read()
for p in touched(test_patch):
    if sh("git", "cat-file", "-e", f"{spec['base_commit']}:{p}").returncode == 0:
        sh("git", "checkout", spec["base_commit"], "--", p)
    elif os.path.exists(os.path.join(T, p)):
        os.remove(os.path.join(T, p))
r = sh("git", "apply", "--whitespace=nowarn", f"{W}/test.patch")
if r.returncode:
    out["test_patch_failed"] = (r.stderr or r.stdout)[-600:]
    finish()

named = spec["named"]
path, *parts = named.split("::")
parts = [x.split("[")[0] for x in parts]
tsrc = read(path)
try:
    for n in ast.walk(ast.parse(tsrc)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and parts and n.name == parts[-1]:
            out["named_source"] = "\n".join(tsrc.splitlines()[n.lineno - 1:n.end_lineno])
            break
except SyntaxError:
    pass

res = f"{W}/out/results.tsv"
env = dict(os.environ, VM_RESULTS=res, PYTHONPATH=W + os.pathsep + os.environ.get("PYTHONPATH", ""))
code = ("import sys, json, pytest; sys.exit(pytest.main(['-p', 'vm_results_plugin', '-p', 'no:cacheprovider', '-q', "
        "'-W', 'ignore::DeprecationWarning'] + json.load(open('/work/ids.json'))))")
try:
    p = subprocess.run([sys.executable, "-c", code], cwd=T, env=env, capture_output=True, text=True, timeout=spec["timeout"])
    out["exit"] = p.returncode
    out["tail"] = (p.stdout + p.stderr)[-1500:]
except subprocess.TimeoutExpired:
    out["timed_out"] = True
outcomes = {}
if os.path.exists(res):
    for ln in open(res):
        if "\t" in ln:
            k, v = ln.rstrip("\n").rsplit("\t", 1)
            if v != "passed" or k not in outcomes:
                outcomes[k] = v
out["outcomes"] = outcomes
finish()
