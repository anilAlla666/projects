"""Runs inside a SWE-rebench instance image with /work mounted: applies the attempt (unless mode is base), drops the
grader's check file into a fresh folder, runs it with pytest, and writes /work/out/result.json with each check's
outcome, telling a failed check from one that errored before it ran. Standard library only."""
import json
import os
import subprocess
import sys

W, T = "/work", "/testbed"
spec = json.load(open(f"{W}/spec.json"))
os.makedirs(f"{W}/out", exist_ok=True)
out = {"mode": spec["mode"]}


def finish():
    with open(f"{W}/out/result.json", "w") as f:
        json.dump(out, f)
    os.chmod(f"{W}/out/result.json", 0o666)
    sys.exit(0)


if spec["mode"] != "base":
    r = subprocess.run(["git", "apply", "--whitespace=nowarn", f"{W}/attempt.patch"], cwd=T, capture_output=True, text=True)
    if r.returncode:
        r = subprocess.run(["git", "apply", "--whitespace=nowarn", "--3way", f"{W}/attempt.patch"], cwd=T, capture_output=True, text=True)
    if r.returncode:
        out["apply_failed"] = (r.stderr or r.stdout)[-400:]
        finish()
os.makedirs(f"{T}/vm_grader_checks", exist_ok=True)
with open(f"{T}/vm_grader_checks/test_vm_grader.py", "w") as f:
    f.write(open(f"{W}/checks.py").read())
plugin = '''
import os
def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        o = report.outcome if report.when == "call" else "error"
        with open(os.environ["VM_RESULTS"], "a") as f:
            f.write(report.nodeid.split("::")[-1] + "\\t" + o + "\\n")
'''
with open(f"{W}/vm_check_plugin.py", "w") as f:
    f.write(plugin)
res = f"{W}/out/results.tsv"
env = dict(os.environ, VM_RESULTS=res, PYTHONPATH=W + os.pathsep + T + os.pathsep + os.environ.get("PYTHONPATH", ""))
code = ("import sys, pytest; sys.exit(pytest.main(['-p', 'vm_check_plugin', '-p', 'no:cacheprovider', '-q', '-x', '--rootdir', '/testbed/vm_grader_checks', "
        "'-o', 'addopts=', '/testbed/vm_grader_checks/test_vm_grader.py']))").replace("'-x', ", "")
try:
    p = subprocess.run([sys.executable, "-c", code], cwd=T, env=env, capture_output=True, text=True, timeout=spec.get("timeout", 120))
    out["exit"] = p.returncode
    out["tail"] = (p.stdout + p.stderr)[-800:]
except subprocess.TimeoutExpired:
    out["timed_out"] = True
outcomes = {}
if os.path.exists(res):
    for ln in open(res):
        if "\t" in ln:
            k, v = ln.rstrip("\n").rsplit("\t", 1)
            outcomes[k] = v
out["outcomes"] = outcomes
finish()
