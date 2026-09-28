#!/usr/bin/env python3
"""
Op 29 COMPLY — compliance artifact bundler gate.

Gate (3 checks):
  structure    — report has keys receipt, carbon, guard, determinism, fairness, topology
  hash_present — determinism.dispatch_hash is a 16-hex string and dispatch_count > 0
  comply_ok    — clean workload => compliance_ok == true
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CWD  = os.path.realpath(os.path.join(HERE, ".."))
HOOK = "./libcipher_hook.so"
RT   = "./libcipher_rt.so"
REPORT_PATH = "/tmp/cipher_comply_report.json"

WORKLOAD = r"""
import torch, ctypes, time
rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_comply_report.restype = None
K, N = 4096, 4096
a = torch.randn(1024, K, dtype=torch.float16, device="cuda")
w = torch.randn(K, N, dtype=torch.float16, device="cuda")
for _ in range(40):
    torch.mm(a, w)
torch.cuda.synchronize()
time.sleep(0.5)
rt.cipher_comply_report()
"""


def run() -> dict:
    if os.path.exists(REPORT_PATH): os.remove(REPORT_PATH)
    env = {**os.environ,
           "LD_PRELOAD":          f"{HOOK} {RT}",
           "CIPHER_FORCE_PERMIT": "1",
           "CIPHER_SENSE":        "on",
           "CIPHER_GUARD":        "on",
           "CIPHER_DETERMINISM":  "on",
           "CIPHER_CARBON":       "on",
           "CIPHER_FAIRNESS":     "on",
           "CIPHER_TOPOLOGY":     "on",
           "CIPHER_RECEIPT":      "on",
           "CIPHER_COMPLY":       "on"}
    r = subprocess.run(
        ["python3", "-c", WORKLOAD],
        capture_output=True, text=True, timeout=180, cwd=CWD, env=env,
    )
    if r.returncode != 0:
        print("STDERR tail:\n" + r.stderr[-2000:])
        raise SystemExit(f"workload crashed (exit {r.returncode})")
    with open(REPORT_PATH) as f:
        return json.load(f)


def main():
    r = run()
    keys_needed = {"receipt","carbon","guard","determinism","fairness","topology","compliance_ok"}
    structure = keys_needed.issubset(set(r.keys()))
    dh = r.get("determinism", {}).get("dispatch_hash", "")
    dc = r.get("determinism", {}).get("dispatch_count", 0)
    hash_present = (bool(re.fullmatch(r"[0-9a-f]{16}", dh)) and dc > 0)
    comply_ok = r.get("compliance_ok") is True

    print(json.dumps(r, indent=2))
    print(f"[structure   ] {'PASS' if structure else 'FAIL'}")
    print(f"[hash_present] {'PASS' if hash_present else 'FAIL'}")
    print(f"[comply_ok   ] {'PASS' if comply_ok else 'FAIL'}")
    passed = int(structure) + int(hash_present) + int(comply_ok)
    print(f"\nResult: {passed}/3")
    sys.exit(0 if passed == 3 else 1)


if __name__ == "__main__":
    main()
