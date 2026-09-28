#!/usr/bin/env python3
"""
Op 13 SENSE — three-workload classification gate.

Runs HUMAN, AGENT, BATCH synthetic workloads under LD_PRELOAD with
CIPHER_SENSE=on, then asserts the JSON report classifies each correctly.

Each workload runs in its own subprocess so SENSE state is fresh per case.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CWD  = os.path.realpath(os.path.join(HERE, ".."))
HOOK = "./libcipher_hook.so"
RT   = "./libcipher_rt.so"
REPORT_PATH = "/tmp/cipher_sense_report.json"

WORKLOAD = r"""
import torch, ctypes, time, sys, os
WL = sys.argv[1]
rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_sense_report.restype = None

# Shapes — large M for prefill, M=1 for decode.
K, N = 4096, 4096
big   = torch.randn(1024, K, dtype=torch.float16, device="cuda")  # prefill (M=1024)
small = torch.randn(1,    K, dtype=torch.float16, device="cuda")  # decode  (M=1)
W     = torch.randn(K,    N, dtype=torch.float16, device="cuda")

def prefill():
    torch.mm(big, W); torch.cuda.synchronize()

def decode_burst(n):
    for _ in range(n):
        torch.mm(small, W)
    torch.cuda.synchronize()

if WL == "human":
    # 1 prefill + ~50 decode tokens. Single turn (one session under v1).
    prefill()
    decode_burst(50)
elif WL == "agent":
    # 1 prefill + 1500-token reasoning chain.
    prefill()
    decode_burst(1500)
elif WL == "batch":
    # 4 large prefills back-to-back, no decode chain.
    for _ in range(4):
        prefill()
else:
    raise SystemExit(f"unknown workload {WL}")

# Allow the Stage 1 shadow thread to drain the ring (500us cadence + buffer).
time.sleep(0.5)
rt.cipher_sense_report()
"""


def run_workload(name: str) -> dict:
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
    env = {**os.environ,
           "LD_PRELOAD":         f"{HOOK} {RT}",
           "CIPHER_FORCE_PERMIT": "1",
           "CIPHER_SENSE":        "on"}
    r = subprocess.run(
        ["python3", "-c", WORKLOAD, name],
        capture_output=True, text=True, timeout=120, cwd=CWD, env=env,
    )
    if r.returncode != 0:
        print(f"[{name}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {name} crashed (exit {r.returncode})")
    if not os.path.exists(REPORT_PATH):
        print(f"[{name}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {name}: no SENSE report produced")
    with open(REPORT_PATH) as f:
        report = json.load(f)
    return report


def latest_session(report: dict) -> dict:
    sessions = report.get("sessions", [])
    if not sessions:
        return {}
    return max(sessions, key=lambda s: s["last_seq"])


def main():
    expected = {
        "human": "HUMAN_INTERACTIVE",
        "agent": "AGENT_AUTONOMOUS",
        "batch": "BATCH_BACKGROUND",
    }
    pass_count = 0
    for name, want in expected.items():
        r = run_workload(name)
        s = latest_session(r)
        got = s.get("type", "MISSING")
        ok = (got == want)
        pass_count += int(ok)
        print(f"[{name:5s}] prefill={s.get('prefill_count'):>4} "
              f"decode={s.get('decode_count'):>5}  "
              f"classified={got:<20} expected={want:<20} "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"\nResult: {pass_count}/3")
    sys.exit(0 if pass_count == 3 else 1)


if __name__ == "__main__":
    main()
