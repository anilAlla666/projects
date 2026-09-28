#!/usr/bin/env python3
"""
Op 22 PULSE — four-test gate.

Uses test-only injection so we do not have to wait for actual hardware drift
or ECC errors. Tests that the alert files are written and contain the
"signal_2_deferred": true marker.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CWD  = os.path.realpath(os.path.join(HERE, ".."))
HOOK = "./libcipher_hook.so"
RT   = "./libcipher_rt.so"
REPORT = "/tmp/cipher_pulse_report.json"
ALERT  = "/tmp/cipher_pulse_alert.json"
LOG    = "/tmp/cipher_pulse.log"

HARNESS = r"""
import torch, ctypes, time, sys, os
scenario = sys.argv[1]

# Clean any prior state — independent test runs share /tmp.
for p in ("/tmp/cipher_pulse_alert.json", "/tmp/cipher_pulse.log",
          "/tmp/cipher_pulse_report.json"):
    if os.path.exists(p): os.remove(p)

rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_pulse_inject_drift_for_shape.argtypes = [
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_double]
rt.cipher_pulse_inject_drift_for_shape.restype = None
rt.cipher_pulse_inject_ecc_delta.argtypes = [ctypes.c_uint64]
rt.cipher_pulse_inject_ecc_delta.restype = None
rt.cipher_pulse_clear_injection.restype = None
rt.cipher_pulse_force_evaluate.restype = None
rt.cipher_pulse_score.restype = ctypes.c_int
rt.cipher_pulse_alert_count.restype = ctypes.c_uint
rt.cipher_pulse_severity.restype = ctypes.c_char_p
rt.cipher_pulse_report.restype = None

# Touch the runtime so PULSE init has fired and rings are alive.
small = torch.randn(1, 4096, dtype=torch.float16, device="cuda")
W     = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
for _ in range(20): torch.mm(small, W)
torch.cuda.synchronize()

if scenario == "drift":
    rt.cipher_pulse_inject_drift_for_shape(4096, 4096, 1, 1.30)
    rt.cipher_pulse_force_evaluate()
elif scenario == "ecc":
    rt.cipher_pulse_inject_ecc_delta(50)
    rt.cipher_pulse_force_evaluate()
elif scenario == "both":
    rt.cipher_pulse_inject_drift_for_shape(4096, 4096, 1, 1.30)
    rt.cipher_pulse_inject_ecc_delta(50)
    rt.cipher_pulse_force_evaluate()
elif scenario == "clean":
    rt.cipher_pulse_force_evaluate()
else:
    raise SystemExit(f"unknown scenario {scenario}")

time.sleep(0.2)
rt.cipher_pulse_report()
print("SCORE", rt.cipher_pulse_score(),
      "ALERT_COUNT", rt.cipher_pulse_alert_count(),
      "SEVERITY", rt.cipher_pulse_severity().decode())
"""


def run(scenario: str) -> dict:
    env = {**os.environ,
           "LD_PRELOAD":          f"{HOOK} {RT}",
           "CIPHER_FORCE_PERMIT": "1",
           "CIPHER_PULSE":        "on"}
    r = subprocess.run(
        ["python3", "-c", HARNESS, scenario],
        capture_output=True, text=True, timeout=180, cwd=CWD, env=env,
    )
    if r.returncode != 0:
        print(f"[{scenario}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {scenario} crashed (exit {r.returncode})")
    out = {
        "stdout":  r.stdout.strip(),
        "report":  None,
        "alert":   None,
        "log":     None,
    }
    if os.path.exists(REPORT):
        with open(REPORT) as f: out["report"] = json.load(f)
    if os.path.exists(ALERT):
        with open(ALERT)  as f: out["alert"]  = json.load(f)
    if os.path.exists(LOG):
        with open(LOG)    as f: out["log"]    = f.read().strip()
    return out


def main():
    pass_count = 0

    # 1. drift only — score 1, INFO log appended, NO alert json
    r = run("drift")
    rep = r["report"] or {}
    ok1 = (rep.get("score") == 1
           and r["alert"] is None
           and r["log"] is not None
           and "signal_2_deferred" in r["log"])
    pass_count += int(ok1)
    print(f"[drift   ] score={rep.get('score')} alert={'Y' if r['alert'] else 'N'} "
          f"log={'Y' if r['log'] else 'N'}  {'PASS' if ok1 else 'FAIL'}")

    # 2. ecc only — score 1, INFO log appended
    r = run("ecc")
    rep = r["report"] or {}
    ok2 = (rep.get("score") == 1
           and r["log"] is not None
           and "signal_2_deferred" in r["log"])
    pass_count += int(ok2)
    print(f"[ecc     ] score={rep.get('score')} log={'Y' if r['log'] else 'N'}  "
          f"{'PASS' if ok2 else 'FAIL'}")

    # 3. both — score 2, WARNING, alert json written with signal_2_deferred=true
    r = run("both")
    rep = r["report"] or {}
    al  = r["alert"]  or {}
    ok3 = (rep.get("score") == 2
           and al.get("severity") == "MEDIUM"
           and al.get("signal_2_deferred") is True
           and al.get("score_ceiling_v1") == 2)
    pass_count += int(ok3)
    print(f"[both    ] score={rep.get('score')} sev={al.get('severity')} "
          f"deferred={al.get('signal_2_deferred')} ceil={al.get('score_ceiling_v1')}  "
          f"{'PASS' if ok3 else 'FAIL'}")

    # 4. clean — no signals, score 0, no alert/log files
    r = run("clean")
    rep = r["report"] or {}
    ok4 = (rep.get("score") == 0
           and r["alert"] is None
           and (r["log"] is None or r["log"] == ""))
    pass_count += int(ok4)
    print(f"[clean   ] score={rep.get('score')} alert={'Y' if r['alert'] else 'N'} "
          f"log={'Y' if r['log'] else 'N'}  {'PASS' if ok4 else 'FAIL'}")

    print(f"\nResult: {pass_count}/4")
    sys.exit(0 if pass_count == 4 else 1)


if __name__ == "__main__":
    main()
