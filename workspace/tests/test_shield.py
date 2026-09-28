#!/usr/bin/env python3
"""
Op 14 SHIELD — three-protection gate.

1. shield-priority: HUMAN_INTERACTIVE workload assigns CIPHER_BAND_PROTECTED
                    via cipher_sm_set_priority.
2. shield-jitter:   bursty timing pattern triggers the >2σ jitter counter.
3. shield-smoothing: P95/P50 > 3 over the rolling window flips the
                    smoothed_session counter.

Each subprocess runs in isolation so SHIELD state is fresh.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CWD  = os.path.realpath(os.path.join(HERE, ".."))
HOOK = "./libcipher_hook.so"
RT   = "./libcipher_rt.so"
REPORT_PATH = "/tmp/cipher_shield_report.json"

# --- Workload templates -----------------------------------------------------

WL_PRIORITY = r"""
import torch, ctypes, time
rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_shield_report.restype = None
big   = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")  # prefill
small = torch.randn(1,    4096, dtype=torch.float16, device="cuda")  # decode
W     = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
torch.mm(big, W); torch.cuda.synchronize()                   # 1 prefill
for _ in range(50):                                          # 50 decode steps
    torch.mm(small, W)
torch.cuda.synchronize()
time.sleep(0.5)
rt.cipher_shield_report()
"""

WL_JITTER = r"""
import torch, ctypes, time
rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_shield_report.restype = None
big   = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
small = torch.randn(1,    4096, dtype=torch.float16, device="cuda")
W     = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
torch.mm(big, W); torch.cuda.synchronize()
# Bursty timing: alternate fast bursts and pauses so per-event ITL variance is high
for k in range(60):
    torch.mm(small, W)
    if k % 5 == 4:
        torch.cuda.synchronize()
        time.sleep(0.020)            # 20 ms gap → spike ITL
torch.cuda.synchronize()
time.sleep(0.5)
rt.cipher_shield_report()
"""

WL_SMOOTHING = r"""
import torch, ctypes, time
rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_shield_report.restype = None
big   = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
small = torch.randn(1,    4096, dtype=torch.float16, device="cuda")
W     = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
torch.mm(big, W); torch.cuda.synchronize()
# Even more pronounced burstiness: pairs of fast/slow.
for k in range(80):
    torch.mm(small, W)
    if k % 2 == 1:
        torch.cuda.synchronize()
        time.sleep(0.030)            # 30 ms gap on every other step
torch.cuda.synchronize()
time.sleep(0.5)
rt.cipher_shield_report()
"""


def run_workload(name: str, code: str) -> dict:
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
    env = {**os.environ,
           "LD_PRELOAD":          f"{HOOK} {RT}",
           "CIPHER_FORCE_PERMIT": "1",
           "CIPHER_SENSE":        "on",
           "CIPHER_SHIELD":       "on"}
    r = subprocess.run(
        ["python3", "-c", code],
        capture_output=True, text=True, timeout=180, cwd=CWD, env=env,
    )
    if r.returncode != 0:
        print(f"[{name}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {name} crashed (exit {r.returncode})")
    if not os.path.exists(REPORT_PATH):
        print(f"[{name}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {name}: no SHIELD report produced")
    with open(REPORT_PATH) as f:
        report = json.load(f)
    return report


def main():
    pass_count = 0

    # 1) Priority — HUMAN session must get CIPHER_BAND_PROTECTED (band==1)
    r = run_workload("priority", WL_PRIORITY)
    sessions = r.get("sessions", [])
    pri_assigned = r.get("priority_assignments", 0)
    bands = [s.get("band", 0) for s in sessions]
    ok1 = (pri_assigned >= 1) and any(b == 1 for b in bands)
    pass_count += int(ok1)
    print(f"[priority ] priority_assignments={pri_assigned}  bands={bands}  "
          f"{'PASS' if ok1 else 'FAIL'}")

    # 2) Jitter — at least one >2σ event flagged
    r = run_workload("jitter", WL_JITTER)
    je = r.get("jitter_events", 0)
    ok2 = (je >= 1)
    pass_count += int(ok2)
    print(f"[jitter   ] jitter_events={je}  {'PASS' if ok2 else 'FAIL'}")

    # 3) Smoothing — P95/P50 > 3 detected on at least one session
    r = run_workload("smoothing", WL_SMOOTHING)
    smoothed = r.get("smoothed_sessions", 0)
    ok3 = (smoothed >= 1)
    pass_count += int(ok3)
    print(f"[smoothing] smoothed_sessions={smoothed}  {'PASS' if ok3 else 'FAIL'}")

    print(f"\nResult: {pass_count}/3")
    sys.exit(0 if pass_count == 3 else 1)


if __name__ == "__main__":
    main()
