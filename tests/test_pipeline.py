#!/usr/bin/env python3
"""
Op 27 PIPELINE — three-workload Jaccard-correlation gate.

Runs correlated, disjoint, single synthetic workloads under LD_PRELOAD with
CIPHER_PIPELINE=on (plus CIPHER_SENSE=on), then asserts the JSON report
shows the expected edge_count.

Gate:
  correlated — edge_count >= 1   (two same-shape sessions; Jaccard >= 0.5)
  disjoint   — edge_count == 0   (two different-shape sessions)
  single     — edge_count == 0   (only one session; no pairs)
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CWD  = os.path.realpath(os.path.join(HERE, ".."))
HOOK = "./libcipher_hook.so"
RT   = "./libcipher_rt.so"
REPORT_PATH = "/tmp/cipher_pipeline_report.json"

WORKLOAD = r"""
import torch, ctypes, time, sys
WL = sys.argv[1]
rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_pipeline_report.restype = None

# Two distinct shape families.
K1, N1 = 4096, 4096
K2, N2 = 2048, 8192
big1   = torch.randn(1024, K1, dtype=torch.float16, device="cuda")
W1     = torch.randn(K1, N1,   dtype=torch.float16, device="cuda")
big2   = torch.randn(1024, K2, dtype=torch.float16, device="cuda")
W2     = torch.randn(K2, N2,   dtype=torch.float16, device="cuda")

def burst_A(n):
    for _ in range(n):
        torch.mm(big1, W1)
    torch.cuda.synchronize()

def burst_B(n):
    for _ in range(n):
        torch.mm(big2, W2)
    torch.cuda.synchronize()

# 300 ms idle > 200 ms PIPELINE boundary -> distinct sessions.
GAP = 0.30

if WL == "correlated":
    burst_A(40); time.sleep(GAP)
    burst_A(40)
elif WL == "disjoint":
    burst_A(40); time.sleep(GAP)
    burst_B(40)
elif WL == "single":
    burst_A(40)
else:
    raise SystemExit(f"unknown workload {WL}")

time.sleep(0.5)
rt.cipher_pipeline_report()
"""


def run_workload(name: str) -> dict:
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
    env = {**os.environ,
           "LD_PRELOAD":          f"{HOOK} {RT}",
           "CIPHER_FORCE_PERMIT": "1",
           "CIPHER_SENSE":        "on",
           "CIPHER_PIPELINE":     "on"}
    r = subprocess.run(
        ["python3", "-c", WORKLOAD, name],
        capture_output=True, text=True, timeout=180, cwd=CWD, env=env,
    )
    if r.returncode != 0:
        print(f"[{name}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {name} crashed (exit {r.returncode})")
    if not os.path.exists(REPORT_PATH):
        print(f"[{name}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {name}: no PIPELINE report produced")
    with open(REPORT_PATH) as f:
        return json.load(f)


def main():
    expected = {
        "correlated": ("ge", 1),
        "disjoint":   ("eq", 0),
        "single":     ("eq", 0),
    }
    pass_count = 0
    for name, (op, thresh) in expected.items():
        r = run_workload(name)
        sessions = r.get("sessions", [])
        edges = r.get("edges", [])
        ec = r.get("edge_count", -1)
        top_j = max((e.get("jaccard", 0.0) for e in edges), default=0.0)
        ok = (ec >= thresh) if op == "ge" else (ec == thresh)
        pass_count += int(ok)
        print(f"[{name:10s}] sessions={len(sessions)} edges={ec} "
              f"top_jaccard={top_j:.3f} "
              f"expect {op} {thresh} {'PASS' if ok else 'FAIL'}")
    print(f"\nResult: {pass_count}/3")
    sys.exit(0 if pass_count == 3 else 1)


if __name__ == "__main__":
    main()
