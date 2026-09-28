#!/usr/bin/env python3
"""
Op 17 PREDICT — three-workload L2 preload candidate gate.

Gate:
  hot_shape    — candidate_count >= 1   (one shape, 300 dispatches)
  cold_shapes  — candidate_count == 0   (30 distinct shapes, 1x each)
  mixed        — candidate_count == 1   (1 hot + 30 cold)
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CWD  = os.path.realpath(os.path.join(HERE, ".."))
HOOK = "./libcipher_hook.so"
RT   = "./libcipher_rt.so"
REPORT_PATH = "/tmp/cipher_predict_report.json"

WORKLOAD = r"""
import torch, ctypes, time, sys
WL = sys.argv[1]
rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_predict_report.restype = None

# Fixed hot shape.
Kh, Nh = 4096, 4096
big_h = torch.randn(1024, Kh, dtype=torch.float16, device="cuda")
W_h   = torch.randn(Kh,   Nh, dtype=torch.float16, device="cuda")

def hot_burst(n):
    for _ in range(n):
        torch.mm(big_h, W_h)
    torch.cuda.synchronize()

def cold_run(n):
    # n distinct shape tuples by varying K.
    for i in range(n):
        Ki = 256 + i * 32     # 256, 288, ..., all distinct
        N  = 1024
        a = torch.randn(64, Ki, dtype=torch.float16, device="cuda")
        b = torch.randn(Ki, N,  dtype=torch.float16, device="cuda")
        torch.mm(a, b)
    torch.cuda.synchronize()

if WL == "hot_shape":
    hot_burst(300)
elif WL == "cold_shapes":
    cold_run(30)
elif WL == "mixed":
    hot_burst(300)
    cold_run(30)
else:
    raise SystemExit(f"unknown workload {WL}")

time.sleep(0.5)
rt.cipher_predict_report()
"""


def run_workload(name: str) -> dict:
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
    env = {**os.environ,
           "LD_PRELOAD":          f"{HOOK} {RT}",
           "CIPHER_FORCE_PERMIT": "1",
           "CIPHER_PREDICT":      "on"}
    r = subprocess.run(
        ["python3", "-c", WORKLOAD, name],
        capture_output=True, text=True, timeout=180, cwd=CWD, env=env,
    )
    if r.returncode != 0:
        print(f"[{name}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {name} crashed (exit {r.returncode})")
    if not os.path.exists(REPORT_PATH):
        print(f"[{name}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {name}: no PREDICT report produced")
    with open(REPORT_PATH) as f:
        return json.load(f)


def main():
    expected = {
        "hot_shape":   ("ge", 1),
        "cold_shapes": ("eq", 0),
        "mixed":       ("eq", 1),
    }
    pass_count = 0
    for name, (op, thresh) in expected.items():
        r = run_workload(name)
        cc = r.get("candidate_count", -1)
        sc = r.get("shape_count", -1)
        ec = r.get("eviction_count", -1)
        cands = r.get("candidates", [])
        top_count = cands[0].get("count", 0) if cands else 0
        top_ratio = cands[0].get("short_gap_ratio", 0.0) if cands else 0.0
        ok = (cc >= thresh) if op == "ge" else (cc == thresh)
        pass_count += int(ok)
        print(f"[{name:11s}] shapes={sc} candidates={cc} evictions={ec} "
              f"top_count={top_count} top_ratio={top_ratio:.3f} "
              f"expect {op} {thresh} {'PASS' if ok else 'FAIL'}")
    print(f"\nResult: {pass_count}/3")
    sys.exit(0 if pass_count == 3 else 1)


if __name__ == "__main__":
    main()
