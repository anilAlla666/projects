#!/usr/bin/env python3
"""
Op 19 CONTINUITY — three-workload manifest gate.

Runs agent_long, human_short, batch synthetic workloads under LD_PRELOAD
with CIPHER_CONTINUITY=on (plus CIPHER_SENSE=on for AGENT classification),
then asserts the JSON report shows the expected manifest_count pattern.

Gate:
  agent_long    — manifest_count >= 1  (long decode chain, classified AGENT)
  human_short   — manifest_count == 0  (short session, HUMAN classified)
  batch         — manifest_count == 0  (prefills only, BATCH classified)
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CWD  = os.path.realpath(os.path.join(HERE, ".."))
HOOK = "./libcipher_hook.so"
RT   = "./libcipher_rt.so"
REPORT_PATH = "/tmp/cipher_continuity_report.json"

WORKLOAD = r"""
import torch, ctypes, time, sys
WL = sys.argv[1]
rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_continuity_report.restype = None

K, N = 4096, 4096
big   = torch.randn(1024, K, dtype=torch.float16, device="cuda")
small = torch.randn(1,    K, dtype=torch.float16, device="cuda")
W     = torch.randn(K,    N, dtype=torch.float16, device="cuda")

def prefill():
    torch.mm(big, W); torch.cuda.synchronize()

def decode_burst(n):
    for _ in range(n):
        torch.mm(small, W)
    torch.cuda.synchronize()

if WL == "agent_long":
    prefill()
    decode_burst(1500)
elif WL == "human_short":
    prefill()
    decode_burst(50)
elif WL == "batch":
    for _ in range(4):
        prefill()
else:
    raise SystemExit(f"unknown workload {WL}")

time.sleep(0.5)
rt.cipher_continuity_report()
"""


def run_workload(name: str) -> dict:
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
    env = {**os.environ,
           "LD_PRELOAD":          f"{HOOK} {RT}",
           "CIPHER_FORCE_PERMIT": "1",
           "CIPHER_SENSE":        "on",
           "CIPHER_CONTINUITY":   "on"}
    r = subprocess.run(
        ["python3", "-c", WORKLOAD, name],
        capture_output=True, text=True, timeout=180, cwd=CWD, env=env,
    )
    if r.returncode != 0:
        print(f"[{name}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {name} crashed (exit {r.returncode})")
    if not os.path.exists(REPORT_PATH):
        print(f"[{name}] STDERR tail:\n{r.stderr[-2000:]}")
        raise SystemExit(f"workload {name}: no CONTINUITY report produced")
    with open(REPORT_PATH) as f:
        return json.load(f)


def main():
    expected = {
        "agent_long":  ("ge", 1),
        "human_short": ("eq", 0),
        "batch":       ("eq", 0),
    }
    pass_count = 0
    for name, (op, thresh) in expected.items():
        r = run_workload(name)
        total = r.get("manifest_count_total", -1)
        sessions = r.get("sessions", [])
        events = sum(s.get("events", 0) for s in sessions)
        attn = sum(s.get("attn_events", 0) for s in sessions)
        types = ",".join(sorted({s.get("sense_type", "?") for s in sessions})) or "-"
        ok = (total >= thresh) if op == "ge" else (total == thresh)
        pass_count += int(ok)
        print(f"[{name:11s}] sessions={len(sessions)} events={events} "
              f"attn={attn} manifests={total} types={types} "
              f"expect {op} {thresh} {'PASS' if ok else 'FAIL'}")
    print(f"\nResult: {pass_count}/3")
    sys.exit(0 if pass_count == 3 else 1)


if __name__ == "__main__":
    main()
