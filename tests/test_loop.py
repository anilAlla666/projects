#!/usr/bin/env python3
"""
Op 26 LOOP — three-workload runaway-detection gate.

Runs RUNAWAY, AGENT_NORMAL, HUMAN synthetic workloads under LD_PRELOAD with
CIPHER_LOOP=on (plus CIPHER_SENSE=on), then asserts the JSON report scores
each correctly.

Gate:
  runaway      — score >= 2
  agent_normal — score <  2
  human        — score <  2
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CWD  = os.path.realpath(os.path.join(HERE, ".."))
HOOK = "./libcipher_hook.so"
RT   = "./libcipher_rt.so"
REPORT_PATH = "/tmp/cipher_loop_report.json"

WORKLOAD = r"""
import torch, ctypes, time, sys
WL = sys.argv[1]
rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_loop_report.restype = None

K, N = 4096, 4096
W     = torch.randn(K, N, dtype=torch.float16, device="cuda")
big   = torch.randn(1024, K, dtype=torch.float16, device="cuda")  # prefill (M=1024)
small = torch.randn(1,    K, dtype=torch.float16, device="cuda")  # decode  (M=1)
# A second decode shape for agent_normal so its shape ring has variety.
K2, N2 = 4096, 14336
W2    = torch.randn(K2, N2, dtype=torch.float16, device="cuda")
small2 = torch.randn(1, K2, dtype=torch.float16, device="cuda")

def prefill():
    torch.mm(big, W); torch.cuda.synchronize()

def decode_same(n):
    for _ in range(n):
        torch.mm(small, W)
    torch.cuda.synchronize()

def decode_mixed(n):
    # Alternate two decode shapes + periodic prefill refresh.
    for i in range(n):
        if i and (i % 200) == 0:
            prefill()                    # fresh prefill resets drought
        if (i % 2) == 0:
            torch.mm(small, W)
        else:
            torch.mm(small2, W2)
    torch.cuda.synchronize()

if WL == "runaway":
    # Classic ReAct-style runaway: 1 prefill then thousands of identical
    # decodes, no further prefill — tight cycle, huge burn, long drought.
    prefill()
    decode_same(2000)
elif WL == "agent_normal":
    # Long agent run but with shape variety + periodic prefill refresh.
    prefill()
    decode_mixed(1500)
elif WL == "human":
    # 1 prefill + modest decode response.
    prefill()
    decode_same(50)
else:
    raise SystemExit(f"unknown workload {WL}")

time.sleep(0.5)
rt.cipher_loop_report()
"""


def run_workload(name: str) -> dict:
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
    env = {**os.environ,
           "LD_PRELOAD":         f"{HOOK} {RT}",
           "CIPHER_FORCE_PERMIT": "1",
           "CIPHER_SENSE":        "on",
           "CIPHER_LOOP":         "on"}
    r = subprocess.run(
        ["python3", "-c", WORKLOAD, name],
        capture_output=True, text=True, timeout=180, cwd=CWD, env=env,
    )
    if r.returncode != 0:
        print(f"[{name}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {name} crashed (exit {r.returncode})")
    if not os.path.exists(REPORT_PATH):
        print(f"[{name}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {name}: no LOOP report produced")
    with open(REPORT_PATH) as f:
        return json.load(f)


def max_score(report: dict) -> dict:
    sessions = report.get("sessions", [])
    if not sessions:
        return {"score": -1}
    return max(sessions, key=lambda s: s["score"])


def main():
    expected = {
        "runaway":      ("ge", 2),
        "agent_normal": ("lt", 2),
        "human":        ("lt", 2),
    }
    pass_count = 0
    for name, (op, thresh) in expected.items():
        r = run_workload(name)
        s = max_score(r)
        score = s.get("score", -1)
        ok = (score >= thresh) if op == "ge" else (score < thresh)
        pass_count += int(ok)
        print(f"[{name:12s}] score={score} "
              f"period={s.get('period',0)} decodes={s.get('decodes',0)} "
              f"since_prefill={s.get('since_prefill',0)} "
              f"s1={s.get('s1_cycle',0)} s2={s.get('s2_burn',0)} s3={s.get('s3_drought',0)} "
              f"expect {op} {thresh} {'PASS' if ok else 'FAIL'}")
    print(f"\nResult: {pass_count}/3")
    sys.exit(0 if pass_count == 3 else 1)


if __name__ == "__main__":
    main()
