#!/usr/bin/env python3
"""Track 3 SC3-2 — unit + integration test runner.

Launches sc3_tenant.py under CUDA_INJECTION64_PATH=<SC3 libcipher_rt> for each
scenario, collects the per-tenant JSON verdicts, prints PASS/FAIL, and writes
TRACK_3_SC3_UNIT_TESTS_result.json.

  unit: self_migrate, verify_fault, destroy_fault, optin(yes/no)
  integration (e2e): 2 migratable decode tenants + a COMPACT_MIGRATE controller.
"""
import fcntl
import json
import os
import subprocess
import sys
import threading
import time

HERE = "/home/ubuntu/cipher-fusion-evidence/phase_c/track_3"
RT = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
TENANT = HERE + "/sc3_tenant.py"
sys.path.insert(0, HERE)
import cipher_migrate as cm                                   # noqa: E402

results = {}


def base_env(scenario, out, sm=16, extra=None):
    e = dict(os.environ)
    e["CUDA_INJECTION64_PATH"] = RT
    e["CIPHER_QOS_CLASS"] = "partition"
    e["CIPHER_SM_COUNT"] = str(sm)
    e["SC3_SCENARIO"] = scenario
    e["SC3_OUT"] = out
    if extra:
        e.update(extra)
    return e


def run_tenant(label, scenario, extra=None, timeout=180):
    out = "%s/sc3_%s_result.json" % (HERE, label)
    if os.path.exists(out):
        os.remove(out)
    log = "%s/sc3_%s.log" % (HERE, label)
    env = base_env(scenario, out, extra=extra)
    print("--- %s (scenario=%s) ---" % (label, scenario), flush=True)
    with open(log, "w") as lf:
        p = subprocess.run([sys.executable, TENANT], env=env,
                           stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
    verdict = None
    if os.path.exists(out):
        verdict = json.load(open(out))
    ok = bool(verdict and verdict.get("pass"))
    print("    exit=%d pass=%s" % (p.returncode, ok), flush=True)
    results[label] = {"scenario": scenario, "exit": p.returncode,
                      "pass": ok, "verdict": verdict}
    time.sleep(1.0)                         # let the do_exit reaper free groups
    return ok


# ---- unit scenarios (sequential — each gets a clean ledger) -------------
run_tenant("self_migrate",  "self_migrate")
run_tenant("verify_fault",  "verify_fault",  {"CIPHER_SC3_FAULT": "verify"})
run_tenant("destroy_fault", "destroy_fault", {"CIPHER_SC3_FAULT": "destroy"})
run_tenant("optin_yes",     "optin",         {"CIPHER_MIGRATABLE": "1"})
run_tenant("optin_no",      "optin")

# ---- e2e integration: 2 migratable decode tenants + COMPACT controller --
print("--- e2e (2 migratable decode tenants + COMPACT controller) ---",
      flush=True)
stop = threading.Event()


def compact_controller():
    fd = os.open("/dev/cipher", os.O_RDWR)
    n = 0
    while not stop.is_set():
        try:
            fcntl.ioctl(fd, cm.COMPACT_MIGRATE)
            n += 1
        except OSError:
            pass
        time.sleep(0.15)
    os.close(fd)
    print("    COMPACT controller issued %d COMPACT_MIGRATE calls" % n,
          flush=True)


ctl = threading.Thread(target=compact_controller, daemon=True)
ctl.start()

procs = []
for tag in ("A", "B"):
    out = "%s/sc3_e2e_%s_result.json" % (HERE, tag)
    if os.path.exists(out):
        os.remove(out)
    log = "%s/sc3_e2e_%s.log" % (HERE, tag)
    env = base_env("e2e", out, extra={"CIPHER_MIGRATABLE": "1",
                                      "SC3_TAG": tag, "SC3_ROUNDS": "8"})
    procs.append((tag, out, subprocess.Popen(
        [sys.executable, TENANT], env=env,
        stdout=open(log, "w"), stderr=subprocess.STDOUT)))
    time.sleep(2.0)                         # A registers (low groups) before B

e2e_ok = True
for tag, out, p in procs:
    p.wait()
    v = json.load(open(out)) if os.path.exists(out) else None
    ok = bool(v and v.get("pass"))
    e2e_ok = e2e_ok and ok
    results["e2e_" + tag] = {"scenario": "e2e", "exit": p.returncode,
                             "pass": ok, "verdict": v}
    print("    e2e tenant %s: exit=%d pass=%s migrated=%s"
          % (tag, p.returncode, ok, v.get("migrated") if v else None),
          flush=True)
stop.set()
ctl.join(timeout=5)

# at least one e2e tenant must have actually migrated end-to-end
any_migrated = any(results["e2e_" + t]["verdict"]
                   and results["e2e_" + t]["verdict"].get("migrated")
                   for t in ("A", "B"))
results["e2e_any_migrated"] = any_migrated

# ---- summary ------------------------------------------------------------
unit_pass = all(results[k]["pass"] for k in
                ("self_migrate", "verify_fault", "destroy_fault",
                 "optin_yes", "optin_no"))
overall = unit_pass and e2e_ok and any_migrated
summary = {"unit_pass": unit_pass, "e2e_pass": e2e_ok,
           "e2e_any_migrated": any_migrated, "overall_pass": overall,
           "results": results}
json.dump(summary, open(HERE + "/TRACK_3_SC3_UNIT_TESTS_result.json", "w"),
          indent=2, default=str)

print("\n=== SC3-2 TEST SUMMARY ===")
for k in ("self_migrate", "verify_fault", "destroy_fault", "optin_yes",
          "optin_no", "e2e_A", "e2e_B"):
    print("  %-14s %s" % (k, "PASS" if results[k]["pass"] else "FAIL"))
print("  %-14s %s" % ("e2e_migrated", "PASS" if any_migrated else "FAIL"))
print("OVERALL:", "PASS" if overall else "FAIL")
sys.exit(0 if overall else 1)
