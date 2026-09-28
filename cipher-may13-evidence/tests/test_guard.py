#!/usr/bin/env python3
"""
Op 16 GUARD — three-workload cross-session residency gate.

Gate:
  isolated — leak_count == 0  (single session, same shape)
  crossed  — leak_count >= 1  (two sessions reusing the same shape)
  distinct — leak_count == 0  (two sessions, disjoint shapes)
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CWD  = os.path.realpath(os.path.join(HERE, ".."))
HOOK = "./libcipher_hook.so"
RT   = "./libcipher_rt.so"
REPORT_PATH = "/tmp/cipher_guard_report.json"

WORKLOAD = r"""
import torch, ctypes, time, sys
WL = sys.argv[1]
rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_guard_report.restype = None

K1, N1 = 4096, 4096
K2, N2 = 2048, 8192
big1 = torch.randn(1024, K1, dtype=torch.float16, device="cuda")
W1   = torch.randn(K1,  N1, dtype=torch.float16, device="cuda")
big2 = torch.randn(1024, K2, dtype=torch.float16, device="cuda")
W2   = torch.randn(K2,  N2, dtype=torch.float16, device="cuda")

def burst_A(n):
    for _ in range(n):
        torch.mm(big1, W1)
    torch.cuda.synchronize()

def burst_B(n):
    for _ in range(n):
        torch.mm(big2, W2)
    torch.cuda.synchronize()

GAP = 0.30   # > 200 ms PIPELINE/session boundary

if WL == "isolated":
    burst_A(40)
elif WL == "crossed":
    burst_A(40); time.sleep(GAP); burst_A(40)
elif WL == "distinct":
    burst_A(40); time.sleep(GAP); burst_B(40)
else:
    raise SystemExit(f"unknown workload {WL}")

time.sleep(0.5)
rt.cipher_guard_report()
"""


def run_workload(name: str) -> dict:
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
    env = {**os.environ,
           "LD_PRELOAD":          f"{HOOK} {RT}",
           "CIPHER_FORCE_PERMIT": "1",
           "CIPHER_SENSE":        "on",
           "CIPHER_GUARD":        "on"}
    r = subprocess.run(
        ["python3", "-c", WORKLOAD, name],
        capture_output=True, text=True, timeout=180, cwd=CWD, env=env,
    )
    if r.returncode != 0:
        print(f"[{name}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {name} crashed (exit {r.returncode})")
    if not os.path.exists(REPORT_PATH):
        print(f"[{name}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {name}: no GUARD report produced")
    with open(REPORT_PATH) as f:
        return json.load(f)


def main():
    expected = {
        "isolated": ("eq", 0),
        "crossed":  ("ge", 1),
        "distinct": ("eq", 0),
    }
    pass_count = 0
    for name, (op, thresh) in expected.items():
        r = run_workload(name)
        lc = r.get("leak_count", -1)
        sc = r.get("session_count", -1)
        sh = r.get("shape_count", -1)
        ls = r.get("leaked_shapes", -1)
        ok = (lc >= thresh) if op == "ge" else (lc == thresh)
        pass_count += int(ok)
        print(f"[{name:9s}] sessions={sc} shapes={sh} leaked_shapes={ls} "
              f"leak_count={lc} expect {op} {thresh} {'PASS' if ok else 'FAIL'}")
    print(f"\nResult: {pass_count}/3")
    sys.exit(0 if pass_count == 3 else 1)


if __name__ == "__main__":
    main()
