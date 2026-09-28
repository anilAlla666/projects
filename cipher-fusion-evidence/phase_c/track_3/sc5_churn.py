#!/usr/bin/env python3
"""Track 3 SC5 — 5-partition randomized-free-order churn orchestrator.

The SC5-3 measurement runner (built in SC5-2, run in SC5-3). For each
(seed × config) it launches 5 migratable PARTITION decode tenants with
staggered lifetimes (the lifetime ordering = the seed's free order), drives
COMPACT_MIGRATE and samples /proc/cipher/migrations, and collects the 6
metric groups (SC5 design memo §3).

config A = migration ON  (cipher_cp54_mig_gap_min_grps = 1)
config B = migration OFF (cipher_cp54_mig_gap_min_grps = 16 — no gap can
           qualify with only 15 groups; pure constrained-grant behaviour)

The compaction trigger is the CIPHER_CP54_FREE ioctl (nr 14), NOT
COMPACT_MIGRATE: FREE runs the kmod's NON-forced compaction pass, which
honours gap_min_grps + sustain. COMPACT_MIGRATE is forced (bypasses the
gates) and would make B migrate too — breaking the single-variable A-vs-B
comparison. A FREE from the orchestrator (a non-registered process) releases
nothing and is a pure repeated non-forced trigger.

Usage:  python3 sc5_churn.py [seed ...]      (default: all 5 seeds, A and B)
        SC5_DESTROY_FAULT=1 python3 sc5_churn.py MIDDLE_OUT   (destroy-fault run)
"""
import fcntl
import json
import os
import subprocess
import sys
import threading
import time

HERE = "/home/ubuntu/cipher-fusion-evidence/phase_c/track_3"
RT = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
TENANT = HERE + "/sc5_tenant.py"
GAP_PARAM = "/sys/module/cipher_kmod/parameters/cipher_cp54_mig_gap_min_grps"
sys.path.insert(0, HERE)
import cipher_migrate as cm                                   # noqa: E402

# CIPHER_CP54_FREE (nr 14, _IO) — from a non-tenant process this releases
# nothing and just runs cp54_eval_migration(false): a non-forced, gap-gate-
# respecting, repeatable eval trigger (see the module docstring).
FREE_IOCTL = cm._IOC(0, 14, 0)

# free order = tenant indices in the order they exit (free their groups)
SEEDS = {
    "LIFO":       [4, 3, 2, 1, 0],
    "FIFO":       [0, 1, 2, 3, 4],
    "MIDDLE_OUT": [2, 1, 3, 0, 4],
    "RAND_A":     [3, 0, 4, 1, 2],
    "RAND_B":     [1, 4, 0, 3, 2],
}
N_TEN = 5
BASE_LIFE_S = 20.0          # first tenant to free lives this long
STEP_LIFE_S = 8.0           # each later free is this much later
SM_PER_TENANT = 8           # 1 group each -> 5 groups used, 10 free


def set_gap(v):
    # the module-param sysfs file is 0644 root-owned — write via sudo
    subprocess.run(["sudo", "sh", "-c", "echo %d > %s" % (v, GAP_PARAM)],
                   check=True)


def quantile(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))]


def read_proc():
    try:
        return open("/proc/cipher/migrations").read()
    except OSError:
        return ""


def parse_proc(txt):
    out = {"stranded": -1, "proposals": 0, "commits": 0,
           "abort_timeout": 0, "abort_nack": 0, "abort_refused": 0}
    for line in txt.splitlines():
        for t in line.split():
            if t.startswith("stranded_groups="):
                out["stranded"] = int(t.split("=")[1])
            elif t.startswith("proposals="):
                out["proposals"] = int(t.split("=")[1])
            elif t.startswith("commits="):
                out["commits"] = int(t.split("=")[1])
            elif t.startswith("timeout="):
                out["abort_timeout"] = int(t.split("=")[1])
            elif t.startswith("tenant_nack="):
                out["abort_nack"] = int(t.split("=")[1])
            elif t.startswith("kmod_refused="):
                out["abort_refused"] = int(t.split("=")[1])
    return out


def run_scenario(seed, config, destroy_fault=False):
    free_order = SEEDS[seed]
    set_gap(1 if config == "A" else 16)
    label = "%s_%s%s" % (seed, config, "_df" if destroy_fault else "")
    print("=== scenario %s (gap_min_grps=%d) ===" %
          (label, 1 if config == "A" else 16), flush=True)

    # lifetime[i] for tenant i: position in free_order -> when it exits
    lifetime = [0.0] * N_TEN
    for pos, ti in enumerate(free_order):
        lifetime[ti] = BASE_LIFE_S + pos * STEP_LIFE_S

    stop = threading.Event()
    samples = []

    def controller():
        fd = os.open("/dev/cipher", os.O_RDWR)
        while not stop.is_set():
            try:
                fcntl.ioctl(fd, FREE_IOCTL)   # non-forced compaction trigger
            except OSError:
                pass
            samples.append({"t": time.time(), **parse_proc(read_proc())})
            time.sleep(0.2)
        os.close(fd)

    ctl = threading.Thread(target=controller, daemon=True)
    ctl.start()

    procs = []
    for ti in range(N_TEN):
        out = "%s/sc5_%s_t%d_result.json" % (HERE, label, ti)
        if os.path.exists(out):
            os.remove(out)
        env = dict(os.environ)
        env.update(CUDA_INJECTION64_PATH=RT, CIPHER_QOS_CLASS="partition",
                   CIPHER_SM_COUNT=str(SM_PER_TENANT), CIPHER_MIGRATABLE="1",
                   SC5_TENANT_ID="t%d" % ti, SC5_OUT=out,
                   SC5_LIFETIME_S=str(lifetime[ti]))
        if destroy_fault and ti < 3:
            # fault tenants 0,1,2; leave 3,4 clean so "other tenants
            # unaffected" is directly testable alongside commit-and-leak.
            env["CIPHER_SC3_FAULT"] = "destroy"
        log = open("%s/sc5_%s_t%d.log" % (HERE, label, ti), "w")
        procs.append((ti, out, subprocess.Popen(
            [sys.executable, TENANT], env=env, stdout=log,
            stderr=subprocess.STDOUT)))
        time.sleep(2.0)            # stagger so tenant i allocates group i

    tenants = {}
    for ti, out, p in procs:
        p.wait()
        tenants["t%d" % ti] = json.load(open(out)) if os.path.exists(out) else None
    stop.set()
    ctl.join(timeout=5)

    # ---- metric groups ----
    strandeds = [s["stranded"] for s in samples if s["stranded"] >= 0]
    all_decode = [d for tv in tenants.values() if tv
                  for d in tv.get("decode_ms", [])]
    kl_ok = all(tv and tv["kl_gate_pass"] for tv in tenants.values())
    n_committed = sum(tv["n_committed"] for tv in tenants.values() if tv)
    res = {
        "seed": seed, "config": config, "destroy_fault": destroy_fault,
        "free_order": free_order, "lifetime_s": lifetime,
        "m1_correctness_kl_gate_all_pass": kl_ok,
        "m2_stranding": {"max": max(strandeds) if strandeds else 0,
                         "mean": (sum(strandeds) / len(strandeds))
                         if strandeds else 0.0,
                         "final": strandeds[-1] if strandeds else 0},
        "m3_latency_ms": {"p50": quantile(all_decode, 0.50),
                          "p95": quantile(all_decode, 0.95),
                          "p99": quantile(all_decode, 0.99),
                          "n": len(all_decode)},
        "m4_migrations": {"committed": n_committed,
                          "proc_final": parse_proc(read_proc())},
        "tenants": tenants,
        "proc_samples": samples,
    }
    json.dump(res, open("%s/sc5_scenario_%s.json" % (HERE, label), "w"),
              default=str)
    print("  %s: kl_gate=%s stranding[max=%d mean=%.2f] "
          "lat[p50=%.1f p95=%.1f p99=%.1f] migrations_committed=%d"
          % (label, kl_ok, res["m2_stranding"]["max"],
             res["m2_stranding"]["mean"], res["m3_latency_ms"]["p50"],
             res["m3_latency_ms"]["p95"], res["m3_latency_ms"]["p99"],
             n_committed), flush=True)
    time.sleep(2.0)                # let the reaper settle before the next run
    return res


def main():
    seeds = sys.argv[1:] or list(SEEDS.keys())
    destroy = os.environ.get("SC5_DESTROY_FAULT") == "1"
    all_res = []
    for seed in seeds:
        for config in ("A", "B"):
            all_res.append(run_scenario(seed, config, destroy_fault=destroy))
    set_gap(1)                     # restore the production default
    out = HERE + "/sc5_churn_results.json"
    json.dump({"scenarios": all_res}, open(out, "w"), default=str)
    print("\nwrote", out, flush=True)


if __name__ == "__main__":
    main()
