"""Runs inside a SWE-rebench image with /work mounted: applies the attempt (unless mode is base), resets the test files
the written grader touches to the base commit, applies the grader's test patch, runs those test files, and writes
/work/out/result.json with each test's outcome. Standard library only."""
import json
import os
import subprocess
import sys

W, T = "/work", "/testbed"
spec = json.load(open(f"{W}/spec.json"))
os.makedirs(f"{W}/out", exist_ok=True)
out = {"mode": spec["mode"]}


def sh(*a):
    return subprocess.run(list(a), cwd=T, capture_output=True, text=True)


def finish():
    with open(f"{W}/out/result.json", "w") as f:
        json.dump(out, f)
    os.chmod(f"{W}/out/result.json", 0o666)
    sys.exit(0)


if spec["mode"] != "base":
    r = sh("git", "apply", "--whitespace=nowarn", f"{W}/attempt.patch")
    if r.returncode:
        r = sh("git", "apply", "--whitespace=nowarn", "--3way", f"{W}/attempt.patch")
    if r.returncode:
        out["apply_failed"] = (r.stderr or r.stdout)[-300:]
        finish()
grader = open(f"{W}/grader.patch").read()
# A diff with several files and no `diff --git` lines cannot be applied with --recount: the next file's `--- ` line
# reads as a removed line of the previous hunk. Give every file its header (the same for every writer); a diff that
# already has them is unchanged.
_g, _norm = grader.splitlines(), []
for _i, _l in enumerate(_g):
    if _l.startswith("--- ") and _i + 1 < len(_g) and _g[_i + 1].startswith("+++ b/") and \
            not (_norm and _norm[-1].startswith(("diff --git", "new file mode", "index ", "deleted file mode", "old mode", "new mode"))):
        _p = _g[_i + 1][6:].split("\t")[0].strip()
        _norm.append(f"diff --git a/{_p} b/{_p}")
        if _l.startswith("--- /dev/null"):
            _norm.append("new file mode 100644")
    _norm.append(_l)
if _norm != _g:
    grader = "\n".join(_norm) + "\n"
    out["headers_added"] = True
GP = "/tmp/grader.patch"
open(GP, "w").write(grader)
files = []
for ln in grader.splitlines():
    if ln.startswith("+++ b/"):
        p = ln[6:].split("\t")[0].strip()
        if p not in files:
            files.append(p)
lines = grader.splitlines()
created = {lines[i + 1][6:].split("\t")[0].strip() for i in range(len(lines) - 1)
           if lines[i].startswith("--- /dev/null") and lines[i + 1].startswith("+++ b/")}
for p in files:
    if p in created:  # the grader writes this file whole: what was there before is replaced, not edited
        if os.path.exists(os.path.join(T, p)):
            os.remove(os.path.join(T, p))
    elif sh("git", "cat-file", "-e", f"{spec['base_commit']}:{p}").returncode == 0:
        sh("git", "checkout", spec["base_commit"], "--", p)
    elif os.path.exists(os.path.join(T, p)):
        os.remove(os.path.join(T, p))
r = sh("git", "apply", "--whitespace=nowarn", "--recount", GP)
if r.returncode:
    out["grader_patch_failed"] = (r.stderr or r.stdout)[-300:]
    finish()
targets = [p for p in files if p.endswith(".py") and os.path.exists(os.path.join(T, p))]
if not targets:
    out["no_tests"] = True
    finish()
res = f"{W}/out/results.tsv"
env = dict(os.environ, VM_RESULTS=res, PYTHONPATH=W + os.pathsep + os.environ.get("PYTHONPATH", ""))
code = ("import sys, json, pytest; sys.exit(pytest.main(['-p', 'vm_results_plugin', '-p', 'no:cacheprovider', '-q', '-rf', '--tb=line', "
        "'-W', 'ignore::DeprecationWarning'] + json.load(open('/work/targets.json'))))")
json.dump(targets, open(f"{W}/targets.json", "w"))
try:
    p = subprocess.run([sys.executable, "-c", code], cwd=T, env=env, capture_output=True, text=True, timeout=spec.get("timeout", 300))
    out["exit"] = p.returncode
    out["tail"] = (p.stdout + p.stderr)[-600:]
    out["failures"] = [l[:300] for l in p.stdout.splitlines() if l.startswith(("FAILED ", "E   ", "/")) and ("Error" in l or "assert" in l or l.startswith("FAILED"))][:30]
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
