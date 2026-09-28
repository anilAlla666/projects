#!/usr/bin/env python3
"""
Op 24 FAIRNESS — per-tenant work quota gate.

Gate (3 checks):
  under_quota — a single small burst does NOT flip the overrun flag
  over_quota  — a single large burst DOES flip the overrun flag (>= 1)
  two_tenants — session_count == 2 after two separated sessions
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CWD  = os.path.realpath(os.path.join(HERE, ".."))
HOOK = "./libcipher_hook.so"
RT   = "./libcipher_rt.so"
REPORT_PATH = "/tmp/cipher_fairness_report.json"

# Work units = M*K*N per GEMM. 4096^3 ~= 6.87e10 per call.
# 5 calls = 3.4e11 (must stay UNDER), 200 calls = 1.4e13 (must blow past).
SMALL_QUOTA = "1000000000000"   # 1e12 — 5 calls of 4096^3 stay under
LARGE_QUOTA =     "10000000000" # 1e10 — single 4096^3 call blows past

WORKLOAD = r"""
import torch, ctypes, time, sys
WL = sys.argv[1]
rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_fairness_report.restype = None

K, N = 4096, 4096
a = torch.randn(1024, K, dtype=torch.float16, device="cuda")
w = torch.randn(K, N, dtype=torch.float16, device="cuda")

def burst(n):
    for _ in range(n):
        torch.mm(a, w)
    torch.cuda.synchronize()

if WL == "under":
    burst(5)
elif WL == "over":
    burst(200)
elif WL == "two":
    burst(5); time.sleep(0.30); burst(5)
else:
    raise SystemExit(WL)

time.sleep(0.5)
rt.cipher_fairness_report()
"""


def run(name: str, quota: str) -> dict:
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
    env = {**os.environ,
           "LD_PRELOAD":             f"{HOOK} {RT}",
           "CIPHER_FORCE_PERMIT":    "1",
           "CIPHER_SENSE":           "on",
           "CIPHER_FAIRNESS":        "on",
           "CIPHER_FAIRNESS_QUOTA":  quota}
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
    u = run("under", SMALL_QUOTA)
    o = run("over",  LARGE_QUOTA)
    t = run("two",   SMALL_QUOTA)

    under_ok = (u.get("overrun_count", -1) == 0)
    over_ok  = (o.get("overrun_count", -1) >= 1)
    two_ok   = (t.get("tenant_count", -1) == 2)

    print(f"[under] tenants={u.get('tenant_count')} overruns={u.get('overrun_count')}")
    print(f"[over ] tenants={o.get('tenant_count')} overruns={o.get('overrun_count')}")
    print(f"[two  ] tenants={t.get('tenant_count')} overruns={t.get('overrun_count')}")
    print(f"[under_quota] {'PASS' if under_ok else 'FAIL'}")
    print(f"[over_quota ] {'PASS' if over_ok  else 'FAIL'}")
    print(f"[two_tenants] {'PASS' if two_ok   else 'FAIL'}")
    passed = int(under_ok) + int(over_ok) + int(two_ok)
    print(f"\nResult: {passed}/3")
    sys.exit(0 if passed == 3 else 1)


if __name__ == "__main__":
    main()
