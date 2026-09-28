#!/usr/bin/env python3
"""
Op 23 CARBON — per-session carbon certificate gate.

Gate (3 checks):
  nonzero_single — one session -> total_gco2 > 0
  two_sessions   — two sessions -> session_count == 2, both have gco2 > 0
  scales         — large burst has gco2 strictly greater than small burst
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CWD  = os.path.realpath(os.path.join(HERE, ".."))
HOOK = "./libcipher_hook.so"
RT   = "./libcipher_rt.so"
REPORT_PATH = "/tmp/cipher_carbon_report.json"

WORKLOAD = r"""
import torch, ctypes, time, sys
WL = sys.argv[1]
rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_carbon_report.restype = None

K, N = 4096, 4096
a = torch.randn(1024, K, dtype=torch.float16, device="cuda")
w = torch.randn(K, N, dtype=torch.float16, device="cuda")

def burst(n):
    for _ in range(n):
        torch.mm(a, w)
    torch.cuda.synchronize()

if WL == "single_small":
    burst(5)
elif WL == "single_large":
    burst(100)
elif WL == "two":
    burst(5); time.sleep(0.30); burst(5)
else:
    raise SystemExit(WL)
time.sleep(0.5)
rt.cipher_carbon_report()
"""


def run(name: str) -> dict:
    if os.path.exists(REPORT_PATH): os.remove(REPORT_PATH)
    env = {**os.environ,
           "LD_PRELOAD":          f"{HOOK} {RT}",
           "CIPHER_FORCE_PERMIT": "1",
           "CIPHER_SENSE":        "on",
           "CIPHER_CARBON":       "on"}
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
    s = run("single_small")
    l = run("single_large")
    t = run("two")

    nonzero = s.get("total_gco2", 0) > 0
    two_ok = (t.get("session_count") == 2
              and all(x.get("gco2", 0) > 0 for x in t.get("sessions", [])))
    scales = l.get("total_gco2", 0) > s.get("total_gco2", 0)

    print(f"[small ] sessions={s.get('session_count')} total_gco2={s.get('total_gco2'):.3e}")
    print(f"[large ] sessions={l.get('session_count')} total_gco2={l.get('total_gco2'):.3e}")
    print(f"[two   ] sessions={t.get('session_count')} total_gco2={t.get('total_gco2'):.3e}")
    print(f"[nonzero_single] {'PASS' if nonzero else 'FAIL'}")
    print(f"[two_sessions  ] {'PASS' if two_ok   else 'FAIL'}")
    print(f"[scales        ] {'PASS' if scales   else 'FAIL'}")
    passed = int(nonzero) + int(two_ok) + int(scales)
    print(f"\nResult: {passed}/3")
    sys.exit(0 if passed == 3 else 1)


if __name__ == "__main__":
    main()
