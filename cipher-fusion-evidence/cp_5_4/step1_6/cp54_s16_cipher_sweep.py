#!/usr/bin/env python3
"""CP 5.4 Step 1.6B-4 — Arm-B CIPHER mixed-deployment sweep driver (THROWAWAY).

Drives `cp54_s16_orchestrator.py` (the 1.6B-2/2A mixed-deployment orchestrator,
extended in 1.6B-4 with 3-point FB / NVML power / Marlin-trace hooks) across
the three operating points, and aggregates the Arm-B headline.

Per OP (design memo §5 / advisor guidance):
  - 5 HEADLINE reps — no trace, the CV / throughput population;
  - 1 TRACE rep — `--marlin-trace` on (D4); recorded separately, NOT in the
    headline statistics (the LD_PRELOAD'd CUPTI counter is a characterization
    instrument, kept out of the pristine headline reps).

Between every rep the kmod CP 5.4 ledger is checked via `CIPHER_CP54_QUERY`
(D1 — design memo §4.5): the ledger must be fully drained (0 partition groups,
0 pool groups, 15 free) before the next rep allocates; a fragmented/non-empty
ledger ABORTS the sweep rather than measuring on a polluted state.

Operating points (15-group ledger; PARTITION + POOL = 15 groups):
  OP-2    -> partitions 16,16            POOL 11 grp / 88 SM   (7 logical tenants)
  OP-5    -> partitions 16,16,16,16,16   POOL  5 grp / 40 SM   (10 tenants)
  OP-asym -> partitions 8,16,16,24       POOL  7 grp / 56 SM   (9 tenants)
all with 5 POOL clients.

Output: cp54_cipher_op{2,5,asym}_result.json + cp54_cipher_sweep_summary.json.
"""
import argparse
import fcntl
import json
import os
import random
import statistics
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, "cp54_s16_orchestrator.py")
POOL_CLIENTS = 5

OPS = {
    "op2":    "16,16",
    "op5":    "16,16,16,16,16",
    "opasym": "8,16,16,24",
}

# ---- CP 5.4 ledger ioctl (kmod cipher_ioctl.h nr 15) — D1 clean-ledger check
_CP54_QUERY = (2 << 30) | (32 << 16) | (ord('C') << 8) | 15


def ledger_clean():
    """Return (clean, (n_part,pool_cnt,free_cnt)). Clean = drained ledger."""
    fd = os.open("/dev/cipher", os.O_RDWR)
    try:
        buf = bytearray(32)
        fcntl.ioctl(fd, _CP54_QUERY, buf, True)
        n_part, pool_cnt, free_cnt, _, _ = struct.unpack_from("<IIIII", buf, 0)
        return (n_part == 0 and pool_cnt == 0 and free_cnt == 15,
                (n_part, pool_cnt, free_cnt))
    finally:
        os.close(fd)


def bootstrap_ci(values, n_resamples=10000, ci=0.95):
    """Bootstrap CI of the MEAN of `values` (design memo §11.2 headline test).
    Returns {mean, lo, hi}."""
    if len(values) < 2:
        return {"mean": values[0] if values else None, "lo": None, "hi": None}
    rng = random.Random(0xC1FE)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        s = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    a = (1.0 - ci) / 2.0
    return {"mean": statistics.mean(values),
            "lo": means[int(a * n_resamples)],
            "hi": means[int((1.0 - a) * n_resamples)],
            "n": n}


def run_rep(op, rep_tag, rounds, marlin_trace):
    outdir = os.path.join(HERE, "b4_%s_%s" % (op, rep_tag))
    cmd = [sys.executable, ORCH, "--partitions", OPS[op],
           "--pool-clients", str(POOL_CLIENTS), "--rounds", str(rounds),
           "--tag", "%s_%s" % (op, rep_tag), "--outdir", outdir]
    if marlin_trace:
        cmd.append("--marlin-trace")
    print("\n--- %s %s%s (%d rounds) ---"
          % (op, rep_tag, "  [MARLIN-TRACE]" if marlin_trace else "", rounds),
          flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=HERE).returncode
    rp = os.path.join(outdir, "orchestrator_result.json")
    if not os.path.exists(rp):
        print("  rep produced no result JSON (rc=%d)" % rc, flush=True)
        return None
    res = json.load(open(rp))
    res["_rep_tag"] = rep_tag
    res["_rc"] = rc
    res["_wall_s"] = round(time.time() - t0, 1)
    return res


def assert_clean_ledger(where):
    clean, counts = ledger_clean()
    print("  ledger check %s: n_part=%d pool=%d free=%d -> %s"
          % (where, counts[0], counts[1], counts[2],
             "CLEAN" if clean else "FRAGMENTED/DIRTY"), flush=True)
    if not clean:
        print("  ABORT — ledger not drained before/after a rep (D1). "
              "Refusing to measure on a polluted ledger.", flush=True)
        sys.exit(3)


def sweep_op(op, headline_reps, rounds):
    reps = []
    assert_clean_ledger("pre-%s" % op)
    for i in range(headline_reps):
        r = run_rep(op, "r%d" % i, rounds, marlin_trace=False)
        if r is None or r.get("verdict") != "PASS":
            print("  STOP — %s headline rep %d verdict=%s"
                  % (op, i, r.get("verdict") if r else "NO-JSON"), flush=True)
            sys.exit(1)
        reps.append(r)
        assert_clean_ledger("post-%s-r%d" % (op, i))
    trace = run_rep(op, "trace", rounds, marlin_trace=True)
    if trace is None or trace.get("verdict") != "PASS":
        print("  STOP — %s trace rep verdict=%s"
              % (op, trace.get("verdict") if trace else "NO-JSON"), flush=True)
        sys.exit(1)
    assert_clean_ledger("post-%s-trace" % op)

    # ---- headline aggregation (headline reps only; trace rep excluded) ----
    cv_pop = [t["decode_ms_cv"] for r in reps for t in r["tenant_stats"]
              if t["decode_ms_cv"] is not None]
    agg = {
        "op": op, "partitions_sm": reps[0]["partitions_sm"],
        "headline_reps": len(reps), "rounds": rounds,
        "all_reps_verdict_pass": all(r["verdict"] == "PASS" for r in reps),
        "disjointness_all_pass": all(r["disjointness"]["pass"] for r in reps),
        "kl_all_pass": all(r["correctness"]["all_partition_kl_gate_pass"]
                           for r in reps),
        "partition_cv_population_n": len(cv_pop),
        "partition_cv_bootstrap95": bootstrap_ci(cv_pop),
        "fb_mib": {"idle": [r["fb_mib"]["idle"] for r in reps],
                   "loaded": [r["fb_mib"]["loaded"] for r in reps],
                   "exited": [r["fb_mib"]["exited"] for r in reps]},
        "power_w_mean": [r["power"]["mean_w"] if r["power"] else None
                         for r in reps],
        "pool_agg_tok_s": [r["pool_result"]["agg_tok_s"]
                           if r.get("pool_result") else None for r in reps],
        "headline_reps_detail": reps,
        "trace_rep": {
            "marlin_trace": trace.get("marlin_trace"),
            "verdict": trace["verdict"],
            "partition_cv": [t["decode_ms_cv"]
                             for t in trace["tenant_stats"]],
        },
    }
    outp = os.path.join(HERE, "cp54_cipher_%s_result.json" % op)
    json.dump(agg, open(outp, "w"), indent=2)
    ci = agg["partition_cv_bootstrap95"]
    mt = (trace.get("marlin_trace") or {})
    print("\n%s: %d headline reps  partition-CV mean=%.4f 95%%CI[%.4f,%.4f] "
          "n=%d  disjoint=%s kl=%s  marlin_loads=%s"
          % (op, len(reps), ci["mean"], ci["lo"], ci["hi"],
             agg["partition_cv_population_n"], agg["disjointness_all_pass"],
             agg["kl_all_pass"], mt.get("cumod_load_count")), flush=True)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default="op2,op5,opasym")
    ap.add_argument("--headline-reps", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=10)
    args = ap.parse_args()

    ops = args.ops.split(",")
    print("1.6B-4 Arm-B sweep: ops=%s headline-reps=%d rounds=%d"
          % (ops, args.headline_reps, args.rounds), flush=True)
    sweep = {}
    for op in ops:
        sweep[op] = sweep_op(op, args.headline_reps, args.rounds)
    json.dump(sweep, open(os.path.join(HERE,
              "cp54_cipher_sweep_summary.json"), "w"), indent=2)
    allok = all(s["all_reps_verdict_pass"] and s["disjointness_all_pass"]
                and s["kl_all_pass"] for s in sweep.values())
    print("\n=== 1.6B-4 ARM-B SWEEP %s ==="
          % ("COMPLETE" if allok else "INCOMPLETE/ISSUE"), flush=True)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
