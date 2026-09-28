#!/usr/bin/env python3
"""Phase 1.5.2 workload runner.

Runs N+warmup samples of one workload under one condition (with/without
cipher_kmod loaded). Discards warmup runs to avoid cold-cache bias.
Emits one JSON line on stdout describing the samples collected.

Workloads:
  A  TinyLlama Marlin smoke (80-token decode); returns float tps
  B  100 nvidia-smi invocations; returns float wall seconds
  C  Mistral-7B c2_marlin (60s decode); returns float tps
  D  probe_microbench tight ioctl loop; returns float ns/ioctl
"""
import argparse
import json
import re
import subprocess
import sys
import time

SMOKE_A = "/workspace/stress2/smoke_marlin_tinyllama.py"
C2_MARLIN = "/workspace/stress2/c2_marlin.py"
MICROBENCH = "/workspace/cipher_kmod/probe_microbench"


def run_workload_a():
    """smoke_marlin_tinyllama.py -> tps."""
    r = subprocess.run(
        ["python3", SMOKE_A],
        capture_output=True, text=True, timeout=300,
    )
    m = re.search(r"tps\s*=\s*(\d+\.\d+)", r.stdout)
    if not m:
        raise RuntimeError(
            f"workload A: no tps in output (rc={r.returncode}); "
            f"tail: {r.stdout[-1000:]}"
        )
    return float(m.group(1))


def run_workload_b():
    """100 nvidia-smi invocations -> wall seconds for the batch."""
    t0 = time.perf_counter()
    for _ in range(100):
        subprocess.run(["nvidia-smi"], capture_output=True, timeout=10)
    return time.perf_counter() - t0


def run_workload_c():
    """c2_marlin.py -> final tps."""
    r = subprocess.run(
        ["python3", C2_MARLIN],
        capture_output=True, text=True, timeout=300,
        cwd="/workspace/stress2",
    )
    matches = re.findall(r"tps=(\d+\.\d+)", r.stdout)
    if not matches:
        raise RuntimeError(
            f"workload C: no tps in output (rc={r.returncode}); "
            f"tail: {r.stdout[-1000:]}"
        )
    return float(matches[-1])


def run_workload_d():
    """probe_microbench (100k iters default) -> ns/ioctl."""
    r = subprocess.run(
        [MICROBENCH, "100000"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"workload D: probe_microbench failed (rc={r.returncode}); "
            f"stderr: {r.stderr[-500:]}"
        )
    line = r.stdout.strip().splitlines()[-1]
    j = json.loads(line)
    return j["ns_per_ioctl"]


WORKLOADS = {"A": run_workload_a, "B": run_workload_b,
             "C": run_workload_c, "D": run_workload_d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=list(WORKLOADS), required=True)
    ap.add_argument("--condition", choices=["with", "without"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    fn = WORKLOADS[args.workload]
    print(f"[{args.workload}/{args.condition}] starting "
          f"n={args.n} warmup={args.warmup}", file=sys.stderr, flush=True)

    for i in range(args.warmup):
        try:
            v = fn()
            print(f"  warmup {i+1}/{args.warmup} -> {v}",
                  file=sys.stderr, flush=True)
        except Exception as e:
            print(f"  warmup {i+1} failed: {e}",
                  file=sys.stderr, flush=True)

    samples = []
    for i in range(args.n):
        v = fn()
        samples.append(v)
        print(f"  sample {i+1}/{args.n} -> {v}",
              file=sys.stderr, flush=True)

    out = {
        "workload": args.workload,
        "condition": args.condition,
        "samples": samples,
    }
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
