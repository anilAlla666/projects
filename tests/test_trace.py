#!/usr/bin/env python3
"""
Op 28 TRACE — bounded kernel-trace exporter gate.

Gate (3 checks):
  nonzero — written > 0 after a real workload
  bounded — written <= capacity (drop-on-full behaviour)
  ordered — timestamps in emitted JSONL are monotonic non-decreasing
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CWD  = os.path.realpath(os.path.join(HERE, ".."))
HOOK = "./libcipher_hook.so"
RT   = "./libcipher_rt.so"
REPORT_PATH = "/tmp/cipher_trace_report.json"
JSONL_PATH  = "/tmp/cipher_trace.jsonl"

WORKLOAD = r"""
import torch, ctypes, time
rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
rt.cipher_trace_report.restype = None
a = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
w = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
for _ in range(200):
    torch.mm(a, w)
torch.cuda.synchronize()
time.sleep(0.3)
rt.cipher_trace_report()
"""


def run() -> dict:
    for p in (REPORT_PATH, JSONL_PATH):
        if os.path.exists(p): os.remove(p)
    env = {**os.environ,
           "LD_PRELOAD":          f"{HOOK} {RT}",
           "CIPHER_FORCE_PERMIT": "1",
           "CIPHER_TRACE":        "on"}
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
    rep = run()
    written = rep.get("written", -1)
    capacity = rep.get("capacity", -1)
    dropped = rep.get("dropped", -1)

    nonzero = written > 0
    bounded = written <= capacity

    ordered = True
    prev_ts = 0
    n_lines = 0
    with open(JSONL_PATH) as f:
        for line in f:
            obj = json.loads(line)
            ts = obj["ts_ns"]
            if ts < prev_ts:
                ordered = False
                break
            prev_ts = ts
            n_lines += 1

    print(f"written={written} dropped={dropped} capacity={capacity} jsonl_lines={n_lines}")
    print(f"[nonzero] {'PASS' if nonzero else 'FAIL'}")
    print(f"[bounded] {'PASS' if bounded else 'FAIL'}")
    print(f"[ordered] {'PASS' if ordered else 'FAIL'}")
    passed = int(nonzero) + int(bounded) + int(ordered)
    print(f"\nResult: {passed}/3")
    sys.exit(0 if passed == 3 else 1)


if __name__ == "__main__":
    main()
