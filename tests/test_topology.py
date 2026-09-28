#!/usr/bin/env python3
"""
Op 25 TOPOLOGY — peer adjacency structural gate.

Gate (3 checks):
  present      — device_count >= 1
  square_matrix — len(adjacency) == device_count AND each row len == device_count
  self_loop_zero — adjacency[i][i] == 0 for all i
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CWD  = os.path.realpath(os.path.join(HERE, ".."))
HOOK = "./libcipher_hook.so"
RT   = "./libcipher_rt.so"
REPORT_PATH = "/tmp/cipher_topology_report.json"

WORKLOAD = r"""
import torch, ctypes, time
rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_topology_report.restype = None
a = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
torch.mm(a, a)
torch.cuda.synchronize()
time.sleep(0.3)
rt.cipher_topology_report()
"""


def run() -> dict:
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
    env = {**os.environ,
           "LD_PRELOAD":          f"{HOOK} {RT}",
           "CIPHER_FORCE_PERMIT": "1",
           "CIPHER_TOPOLOGY":     "on"}
    r = subprocess.run(
        ["python3", "-c", WORKLOAD],
        capture_output=True, text=True, timeout=120, cwd=CWD, env=env,
    )
    if r.returncode != 0:
        print("STDERR tail:\n" + r.stderr[-2000:])
        raise SystemExit(f"workload crashed (exit {r.returncode})")
    if not os.path.exists(REPORT_PATH):
        print("STDERR tail:\n" + r.stderr[-2000:])
        raise SystemExit("no TOPOLOGY report produced")
    with open(REPORT_PATH) as f:
        return json.load(f)


def main():
    r = run()
    n = r.get("device_count", -1)
    adj = r.get("adjacency", None)

    present = (isinstance(n, int) and n >= 1)
    square = (isinstance(adj, list) and len(adj) == n
              and all(isinstance(row, list) and len(row) == n for row in adj))
    self_zero = (square
                 and all(adj[i][i] == 0 for i in range(n)))

    print(f"device_count={n} edge_count={r.get('edge_count')}")
    print(f"[present       ] {'PASS' if present else 'FAIL'}")
    print(f"[square_matrix ] {'PASS' if square else 'FAIL'}")
    print(f"[self_loop_zero] {'PASS' if self_zero else 'FAIL'}")
    passed = int(present) + int(square) + int(self_zero)
    print(f"\nResult: {passed}/3")
    sys.exit(0 if passed == 3 else 1)


if __name__ == "__main__":
    main()
