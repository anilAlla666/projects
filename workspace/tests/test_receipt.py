#!/usr/bin/env python3
"""
Op 18 RECEIPT — per-session signed proof of compute gate.

Gate (3 checks):
  present      — one session -> receipt with nonempty 64-char hmac
  two_distinct — two sessions -> two receipts, hmacs differ
  reproducible — same workload + same fixed key -> same hmac across runs
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CWD  = os.path.realpath(os.path.join(HERE, ".."))
HOOK = "./libcipher_hook.so"
RT   = "./libcipher_rt.so"
REPORT_PATH = "/tmp/cipher_receipt_report.json"

FIXED_KEY = "a" * 64

WORKLOAD = r"""
import torch, ctypes, time, sys
WL = sys.argv[1]
torch.manual_seed(7)
rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_receipt_report.restype = None

K, N = 4096, 4096
a = torch.randn(1024, K, dtype=torch.float16, device="cuda")
w = torch.randn(K, N, dtype=torch.float16, device="cuda")

def burst(n):
    for _ in range(n):
        torch.mm(a, w)
    torch.cuda.synchronize()

if WL == "single":
    burst(20)
elif WL == "two":
    burst(20); time.sleep(0.30); burst(20)
else:
    raise SystemExit(WL)
time.sleep(0.5)
rt.cipher_receipt_report()
"""


def run(name: str, key: str = None) -> dict:
    if os.path.exists(REPORT_PATH): os.remove(REPORT_PATH)
    env = {**os.environ,
           "LD_PRELOAD":          f"{HOOK} {RT}",
           "CIPHER_FORCE_PERMIT": "1",
           "CIPHER_SENSE":        "on",
           "CIPHER_RECEIPT":      "on"}
    if key: env["CIPHER_RECEIPT_KEY"] = key
    r = subprocess.run(
        ["python3", "-c", WORKLOAD, name],
        capture_output=True, text=True, timeout=180, cwd=CWD, env=env,
    )
    if r.returncode != 0:
        print(f"[{name}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {name} crashed (exit {r.returncode})")
    with open(REPORT_PATH) as f:
        return json.load(f)


def main():
    s = run("single", FIXED_KEY)
    t = run("two",    FIXED_KEY)
    r2 = run("single", FIXED_KEY)

    s_recs = s.get("receipts", [])
    t_recs = t.get("receipts", [])
    r2_recs = r2.get("receipts", [])

    present = (len(s_recs) >= 1
               and isinstance(s_recs[0].get("hmac"), str)
               and len(s_recs[0]["hmac"]) == 64)

    two_distinct = (len(t_recs) >= 2
                    and t_recs[0]["hmac"] != t_recs[1]["hmac"])

    # Reproducible: same fixed key + same workload + same session fp => same hmac.
    # Session fp depends on SENSE's fingerprint which derives from timing/shapes.
    # Compare sorted set of hmacs.
    reproducible = (sorted(r.get("hmac") for r in s_recs)
                    == sorted(r.get("hmac") for r in r2_recs)) and len(s_recs) == len(r2_recs)

    print(f"[single ] receipts={len(s_recs)} hmac0={s_recs[0]['hmac'][:16] if s_recs else 'none'}...")
    print(f"[two    ] receipts={len(t_recs)}")
    print(f"[rerun  ] receipts={len(r2_recs)} hmac0={r2_recs[0]['hmac'][:16] if r2_recs else 'none'}...")
    print(f"[present     ] {'PASS' if present else 'FAIL'}")
    print(f"[two_distinct] {'PASS' if two_distinct else 'FAIL'}")
    print(f"[reproducible] {'PASS' if reproducible else 'FAIL'}")
    passed = int(present) + int(two_distinct) + int(reproducible)
    print(f"\nResult: {passed}/3")
    sys.exit(0 if passed == 3 else 1)


if __name__ == "__main__":
    main()
