#!/usr/bin/env python3
"""
Op 21 DETERMINISM — dispatch-sequence fingerprint gate.

Gate (3 checks):
  reproducible — hash(runA) == hash(runB)    (same script, same shapes)
  distinct     — hash(runA) != hash(runC)    (different shape)
  nonzero      — dispatch_count(runA) > 0
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CWD  = os.path.realpath(os.path.join(HERE, ".."))
HOOK = "./libcipher_hook.so"
RT   = "./libcipher_rt.so"
REPORT_PATH = "/tmp/cipher_determinism_report.json"

WORKLOAD = r"""
import torch, ctypes, time, sys
WL = sys.argv[1]
torch.manual_seed(42)
rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_determinism_report.restype = None

if WL in ("runA", "runB"):
    K, N = 4096, 4096
elif WL == "runC":
    K, N = 2048, 8192
else:
    raise SystemExit(f"unknown {WL}")

a = torch.randn(1024, K, dtype=torch.float16, device="cuda")
w = torch.randn(K,    N, dtype=torch.float16, device="cuda")
for _ in range(100):
    torch.mm(a, w)
torch.cuda.synchronize()
time.sleep(0.5)
rt.cipher_determinism_report()
"""


def run_workload(name: str) -> dict:
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
    env = {**os.environ,
           "LD_PRELOAD":          f"{HOOK} {RT}",
           "CIPHER_FORCE_PERMIT": "1",
           "CIPHER_DETERMINISM":  "on"}
    r = subprocess.run(
        ["python3", "-c", WORKLOAD, name],
        capture_output=True, text=True, timeout=180, cwd=CWD, env=env,
    )
    if r.returncode != 0:
        print(f"[{name}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {name} crashed (exit {r.returncode})")
    if not os.path.exists(REPORT_PATH):
        print(f"[{name}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {name}: no DETERMINISM report produced")
    with open(REPORT_PATH) as f:
        return json.load(f)


def main():
    A = run_workload("runA")
    B = run_workload("runB")
    C = run_workload("runC")
    ha, hb, hc = A["dispatch_hash"], B["dispatch_hash"], C["dispatch_hash"]
    na = A["dispatch_count"]

    reprod  = (ha == hb)
    distinct = (ha != hc)
    nonzero  = (na > 0)

    print(f"[runA ] hash={ha} count={na}")
    print(f"[runB ] hash={hb} count={B['dispatch_count']}")
    print(f"[runC ] hash={hc} count={C['dispatch_count']}")
    print(f"[reprod   ] {'PASS' if reprod else 'FAIL'}  (A==B)")
    print(f"[distinct ] {'PASS' if distinct else 'FAIL'}  (A!=C)")
    print(f"[nonzero  ] {'PASS' if nonzero else 'FAIL'}  (count>0)")
    passed = int(reprod) + int(distinct) + int(nonzero)
    print(f"\nResult: {passed}/3")
    sys.exit(0 if passed == 3 else 1)


if __name__ == "__main__":
    main()
