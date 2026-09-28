#!/usr/bin/env python3
"""CP 5.4 Step 1.5B Tier (ii) — real pool-resize churn cost.

Drives the production resize path: a POOL (this process, via
cp54_pool.PoolBinding) plus a PARTITION (subprocess, cp54_s15_partition_driver)
arriving and departing. Each arrival/departure changes the kmod ledger's pool
group count, so PoolBinding.check_resize() rebuilds the green context — the
full path: ALLOCATE ioctl + GreenContext destroy/create + green-ctx stream +
%smid self-verify. This harness times each resize event.

Per the Step 1.5 design memo §5: the timed window includes a
torch.cuda.synchronize() BEFORE the resize, so pending green-stream work does
not implicitly stall (and mis-attribute) the teardown — the cross-stream
hazard class Step 1.4C paid for.

Throwaway Step 1.5 measurement harness — not a campaign anchor, not substrate.
"""
import argparse
import json
import os
import subprocess
import sys
import time

# cp54_pool.py lives in the Step 1.4 evidence dir.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "step1_4"))
import cp54_pool  # noqa: E402
import torch      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "cp54_s15_partition_driver.py")


def pctl(xs, p):
    s = sorted(xs)
    i = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[i]


def stats(xs):
    return {"n": len(xs), "mean_ms": sum(xs) / len(xs), "p50_ms": pctl(xs, 50),
            "p99_ms": pctl(xs, 99), "max_ms": max(xs), "min_ms": min(xs)}


def spawn_partition(sm_count):
    p = subprocess.Popen([sys.executable, DRIVER, str(sm_count)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True)
    line = p.stdout.readline().strip()
    assert line.startswith("ALLOCATED"), "partition driver said: %r" % line
    return p, line


def free_partition(p):
    p.stdin.write("FREE\n")
    p.stdin.flush()
    line = p.stdout.readline().strip()
    assert line == "FREED", "partition driver free said: %r" % line
    p.wait(timeout=15)


def timed_resize(pool, label, events):
    """Time one check_resize() — the full pool-resize path. The sync before
    the timed window drains pending green-stream work (memo §5)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    changed = pool.check_resize()
    t1 = time.perf_counter()
    dt_ms = (t1 - t0) * 1e3
    events.append({"label": label, "resized": bool(changed), "ms": dt_ms,
                   "pool_groups": pool.grp_count})
    print("[resize] %-26s resized=%-5s %8.3f ms  pool=%2d groups"
          % (label, changed, dt_ms, pool.grp_count), flush=True)
    return dt_ms, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--part-sm", type=int, default=16,   # 16 SM -> 2 groups
                    help="SM count the PARTITION takes (ceil/8 groups)")
    ap.add_argument("--out", type=str, default="resize_path.json")
    args = ap.parse_args()

    pool = cp54_pool.PoolBinding()
    pool.build_green_ctx()                       # initial green ctx (15 groups)
    print("[resize] POOL initial: %d groups / %d SMs"
          % (pool.grp_count, pool.grp_count * 8), flush=True)

    shrink_ms, grow_ms = [], []
    events = []
    ok = True
    for c in range(args.cycles):
        p, alloc = spawn_partition(args.part_sm)
        dt, changed = timed_resize(pool, "cycle%d shrink (part in)" % c, events)
        shrink_ms.append(dt)
        if not changed:
            print("[resize] WARN cycle%d: shrink did not resize (%s)"
                  % (c, alloc), flush=True)
            ok = False
        free_partition(p)
        dt, changed = timed_resize(pool, "cycle%d grow (part out)" % c, events)
        grow_ms.append(dt)
        if not changed:
            print("[resize] WARN cycle%d: grow did not resize" % c, flush=True)
            ok = False

    pool.free()

    summary = {"tier": "real_resize_path", "part_sm": args.part_sm,
               "cycles": args.cycles, "all_resized": ok,
               "shrink": stats(shrink_ms), "grow": stats(grow_ms),
               "events": events}
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print("[resize] shrink mean=%.3f p50=%.3f p99=%.3f max=%.3f ms"
          % (summary["shrink"]["mean_ms"], summary["shrink"]["p50_ms"],
             summary["shrink"]["p99_ms"], summary["shrink"]["max_ms"]))
    print("[resize] grow   mean=%.3f p50=%.3f p99=%.3f max=%.3f ms"
          % (summary["grow"]["mean_ms"], summary["grow"]["p50_ms"],
             summary["grow"]["p99_ms"], summary["grow"]["max_ms"]))
    print("[resize] all_resized=%s — wrote %s" % (ok, args.out))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
