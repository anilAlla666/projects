#!/usr/bin/env python3
"""CP 5.4 Step 1.5B Tier (i) — bare-API green-context create/teardown floor.

Measures torch.cuda.GreenContext.create(K*8) latency in isolation: the lower
bound on green-context churn cost, with NO kmod ioctl, NO %smid self-verify,
NO stream creation. Swept over group count K. This is the floor against which
the real pool-resize path (Tier ii, cp54_s15_resize.py) is compared.

Throwaway Step 1.5 measurement harness — not a campaign anchor, not substrate.
"""
import argparse
import json
import os
import time

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
import torch  # noqa: E402


def pctl(xs, p):
    s = sorted(xs)
    i = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--groups", type=str, default="1,3,5,7,9,11,13,15")
    ap.add_argument("--out", type=str, default="bare_floor.json")
    args = ap.parse_args()

    # A primary context must exist before timing, else the first create() is
    # charged with primary-context bring-up (the CUDAGreenContext.cpp warning).
    torch.cuda.init()
    _ = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

    Ks = [int(x) for x in args.groups.split(",")]
    results = []
    for K in Ks:
        num_sms = K * 8
        warm = torch.cuda.GreenContext.create(num_sms, 0)  # warm this size
        del warm
        lat_ms = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            g = torch.cuda.GreenContext.create(num_sms, 0)
            t1 = time.perf_counter()
            del g                                # teardown (refcount -> __del__)
            lat_ms.append((t1 - t0) * 1e3)
        row = {
            "K": K, "sms": num_sms, "iters": args.iters,
            "mean_ms": sum(lat_ms) / len(lat_ms),
            "p50_ms": pctl(lat_ms, 50),
            "p99_ms": pctl(lat_ms, 99),
            "max_ms": max(lat_ms),
            "min_ms": min(lat_ms),
        }
        results.append(row)
        print("[bare] K=%2d (%3d SM)  mean=%.3f  p50=%.3f  p99=%.3f  max=%.3f ms"
              % (K, num_sms, row["mean_ms"], row["p50_ms"], row["p99_ms"],
                 row["max_ms"]), flush=True)

    with open(args.out, "w") as f:
        json.dump({"tier": "bare_api_floor", "iters": args.iters,
                   "results": results}, f, indent=2)
    print("[bare] wrote " + args.out, flush=True)


if __name__ == "__main__":
    main()
