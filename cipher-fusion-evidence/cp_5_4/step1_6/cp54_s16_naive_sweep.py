#!/usr/bin/env python3
"""CP 5.4 Step 1.6B-3 — naive Arm-A measurement sweep driver (THROWAWAY).

Runs the naive orchestrator for the three operating points, 5 reps each, and
aggregates per-OP cross-rep statistics. Tenant counts are the adjudicated
design-memo §2/§4 Arm-A totals (PARTITION-class + 5 POOL-class):

    OP-2  -> 7 tenants     OP-5  -> 10 tenants     OP-asym -> 9 tenants

The CIPHER kmod must be UNLOADED for the whole sweep (true Arm-A). The driver
does not rmmod (needs sudo) — the run procedure unloads it; the orchestrator
records `kmod_loaded` per rep and the driver asserts it is False.

Output: cp54_naive_op2_result.json / _op5_ / _opasym_ — each carries all 5
reps raw plus the cross-rep summary.
"""
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, "cp54_s16_naive_orchestrator.py")
ROUNDS = 10
REPS = 5
OPS = [("op2", 7), ("op5", 10), ("opasym", 9)]


def run_rep(op, n, rep):
    tag = "%s_r%d" % (op, rep)
    outdir = os.path.join(HERE, "b3_" + tag)
    print("\n=== %s rep %d/%d — %d tenants ===" % (op, rep + 1, REPS, n),
          flush=True)
    t0 = time.time()
    rc = subprocess.run(
        [sys.executable, ORCH, "--tenants", str(n), "--rounds", str(ROUNDS),
         "--tag", tag], cwd=HERE).returncode
    dt = time.time() - t0
    rp = os.path.join(outdir, "naive_result.json")
    if rc != 0 or not os.path.exists(rp):
        print("  rep FAILED rc=%d" % rc, flush=True)
        return None
    res = json.load(open(rp))
    res["_rep"] = rep
    res["_wall_s"] = round(dt, 1)
    return res


def xrep(values):
    """Cross-rep summary of a scalar metric."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {"reps": vals,
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals)}


def aggregate(op, n, reps):
    ok = [r for r in reps if r is not None]
    summary = {
        "op": op, "tenants": n, "rounds": ROUNDS,
        "reps_completed": len(ok), "reps_requested": REPS,
        "all_kl_gates_pass": all(r["all_kl_gates_pass"] for r in ok),
        "kmod_unloaded_all_reps": all(not r["kmod_loaded_during_run"]
                                      for r in ok),
        "xrep": {
            "aggregate_tok_s": xrep([r["aggregate_tok_s"] for r in ok]),
            "per_tenant_cv_mean":
                xrep([r["per_tenant_cv_warm"]["mean"] for r in ok]),
            "per_tenant_cv_max":
                xrep([r["per_tenant_cv_warm"]["max"] for r in ok]),
            "fb_loaded_mib": xrep([r["fb_mib"]["loaded"] for r in ok]),
            "fb_exited_mib": xrep([r["fb_mib"]["exited"] for r in ok]),
        },
        "reps": ok,
    }
    return summary


def main():
    sweep = {}
    for op, n in OPS:
        reps = [run_rep(op, n, rep) for rep in range(REPS)]
        agg = aggregate(op, n, reps)
        outp = os.path.join(HERE, "cp54_naive_%s_result.json" % op)
        json.dump(agg, open(outp, "w"), indent=2)
        sweep[op] = agg
        ct = agg["xrep"]["aggregate_tok_s"]
        cv = agg["xrep"]["per_tenant_cv_mean"]
        print("\n%s: %d/%d reps  agg_tok_s=%.1f±%.1f  cv_mean=%.4f±%.4f  "
              "kl_all=%s  kmod_unloaded=%s"
              % (op, agg["reps_completed"], REPS,
                 ct["mean"], ct["std"], cv["mean"], cv["std"],
                 agg["all_kl_gates_pass"], agg["kmod_unloaded_all_reps"]),
              flush=True)

    json.dump(sweep, open(os.path.join(HERE, "cp54_naive_sweep_summary.json"),
                          "w"), indent=2)
    allok = all(s["reps_completed"] == REPS and s["all_kl_gates_pass"]
                and s["kmod_unloaded_all_reps"] for s in sweep.values())
    print("\n=== 1.6B-3 NAIVE SWEEP %s ==="
          % ("COMPLETE" if allok else "INCOMPLETE/ISSUE"), flush=True)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
