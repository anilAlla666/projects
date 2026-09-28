#!/usr/bin/env python3
"""Track 3 SC6 — refined churn orchestrator (within-A + clause-2 + cost).

Per seed (config A only, gap_min_grps=1): 5 barriered PARTITION decode
tenants — 3 migratable (launched first → low groups, the migration
candidates) + 2 pinned (launched last → high groups). Round counts encode
the seed's free order. A barrier coordinator advances all live tenants in
lockstep so the per-round %smid probes are contemporaneous (clause-2). A
controller drives the non-forced compaction trigger + samples
/proc/cipher/migrations.

HARNESS FINDING (SC6-2): the SC6 design-memo §1 POOL process was dropped.
A POOL claims the maximal contiguous low prefix and therefore FILLS the
15-group device when partitions are present — leaving zero free-group
headroom for a count-preserving migration to target. With a POOL no
migration can fire, which defeats the within-A / cost measurements. The
POOL was specified only to fix F2's *stranding-metric* artifact; SC6 does
NOT re-measure A-vs-B stranding (F1 is adjudicated-accepted from SC5), so
the POOL serves no SC6 measurement. SC6 uses the SC5-style headroom
topology (5 partitions, no POOL, free groups available as migration
targets). Surfaced for adjudication.

Computes: within-A per-opt-in-class latency p50/p95/p99; clause-2
(per-round pairwise-disjoint across live tenants); migration cost (per
migrated round, vs the tenant's own non-migrated p50).
"""
import fcntl
import glob
import json
import os
import subprocess
import sys
import threading
import time

HERE = "/home/ubuntu/cipher-fusion-evidence/phase_c/track_3"
RT = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
TENANT = HERE + "/sc6_tenant.py"
POOL = HERE + "/sc6_pool.py"
GAP_PARAM = "/sys/module/cipher_kmod/parameters/cipher_cp54_mig_gap_min_grps"
sys.path.insert(0, HERE)
import cipher_migrate as cm                                   # noqa: E402

FREE_IOCTL = cm._IOC(0, 14, 0)             # non-forced compaction trigger

SEEDS = {
    "LIFO":       [4, 3, 2, 1, 0],
    "FIFO":       [0, 1, 2, 3, 4],
    "MIDDLE_OUT": [2, 1, 3, 0, 4],
    "RAND_A":     [3, 0, 4, 1, 2],
    "RAND_B":     [1, 4, 0, 3, 2],
}
N_TEN = 5
MIGRATABLE_IDS = {0, 1, 2}                 # 3 migratable, tenants 3,4 pinned
BASE_ROUNDS, STEP_ROUNDS = 5, 3            # freed-k-th tenant runs BASE+k*STEP


def set_gap(v):
    subprocess.run(["sudo", "sh", "-c", "echo %d > %s" % (v, GAP_PARAM)],
                   check=True)


def quantile(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))]


def parse_stranded(txt):
    for line in txt.splitlines():
        for t in line.split():
            if t.startswith("stranded_groups="):
                return int(t.split("=")[1])
    return -1


def run_scenario(seed):
    free_order = SEEDS[seed]
    rounds_of = [0] * N_TEN
    for pos, ti in enumerate(free_order):
        rounds_of[ti] = BASE_ROUNDS + pos * STEP_ROUNDS
    max_rounds = max(rounds_of)
    tag = "%s_%d" % (seed, int(time.time()) % 100000)
    print("=== scenario %s  rounds_of=%s ===" % (seed, rounds_of), flush=True)
    for f in glob.glob("/tmp/sc6_%s_*" % tag):
        os.remove(f)

    stop = threading.Event()
    samples = []

    def controller():
        fd = os.open("/dev/cipher", os.O_RDWR)
        while not stop.is_set():
            try:
                fcntl.ioctl(fd, FREE_IOCTL)        # non-forced compaction pass
            except OSError:
                pass
            try:
                samples.append(parse_stranded(
                    open("/proc/cipher/migrations").read()))
            except OSError:
                pass
            time.sleep(0.2)
        os.close(fd)

    def barrier_coord():
        # round r membership = tenants whose round count > r (known schedule)
        for r in range(max_rounds):
            alive = [ti for ti in range(N_TEN) if rounds_of[ti] > r]
            for ti in alive:
                done = "/tmp/sc6_%s_t%d_r%d.done" % (tag, ti, r)
                while not os.path.exists(done) and not stop.is_set():
                    time.sleep(0.02)
            open("/tmp/sc6_%s_all_r%d.go" % (tag, r), "w").close()

    ctl = threading.Thread(target=controller, daemon=True)
    bc = threading.Thread(target=barrier_coord, daemon=True)
    ctl.start()
    bc.start()

    procs = []
    for ti in range(N_TEN):
        out = "%s/sc6_%s_t%d_result.json" % (HERE, tag, ti)
        if os.path.exists(out):
            os.remove(out)
        env = dict(os.environ)
        env.update(CUDA_INJECTION64_PATH=RT, CIPHER_QOS_CLASS="partition",
                   CIPHER_SM_COUNT="8", SC6_TENANT_ID="t%d" % ti,
                   SC6_OUT=out, SC6_ROUNDS=str(rounds_of[ti]),
                   SC6_BARRIER_TAG=tag)
        if ti in MIGRATABLE_IDS:
            env["CIPHER_MIGRATABLE"] = "1"
        else:
            env.pop("CIPHER_MIGRATABLE", None)
        log = open("%s/sc6_%s_t%d.log" % (HERE, tag, ti), "w")
        procs.append((ti, out, subprocess.Popen(
            [sys.executable, TENANT], env=env, stdout=log,
            stderr=subprocess.STDOUT)))
        time.sleep(2.0)                    # stagger -> tenant i takes group i

    tenants = {}
    for ti, out, p in procs:
        p.wait()
        tenants[ti] = json.load(open(out)) if os.path.exists(out) else None
    stop.set()
    ctl.join(timeout=5)
    bc.join(timeout=5)

    # ---- metrics ----
    mig_lat, pin_lat = [], []
    prim_ms, l2_ms = [], []         # migration cost: primitive vs L2 probe
    for ti, tv in tenants.items():
        if not tv:
            continue
        if ti in MIGRATABLE_IDS:
            mig_lat += [d for rd in tv["rounds"] for d in rd["decode_ms"]]
            for rd in tv["rounds"]:
                if rd["migrated"]:
                    if rd.get("primitive_ms") is not None:
                        prim_ms.append(rd["primitive_ms"])
                    if rd.get("l2_ms") is not None:
                        l2_ms.append(rd["l2_ms"])
        else:
            pin_lat += [d for rd in tv["rounds"] for d in rd["decode_ms"]]

    # clause-2: per round, pairwise-disjoint observed SM sets across tenants
    clause2_fails = []
    for r in range(max_rounds):
        sets = {}
        for ti, tv in tenants.items():
            if not tv:
                continue
            for rd in tv["rounds"]:
                if rd["round"] == r:
                    sets[ti] = set(rd["observed_sms"])
        ids = list(sets)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if sets[ids[i]] & sets[ids[j]]:
                    clause2_fails.append({"round": r, "pair": [ids[i], ids[j]],
                                          "overlap": sorted(
                                              sets[ids[i]] & sets[ids[j]])})

    strandeds = [s for s in samples if s >= 0]
    kl_ok = all(tv and tv["kl_gate_pass"] for tv in tenants.values())
    res = {
        "seed": seed, "config": "A", "rounds_of": rounds_of,
        "kl_gate_all_pass": kl_ok,
        "within_A_latency_ms": {
            "migratable": {"p50": quantile(mig_lat, .5),
                           "p95": quantile(mig_lat, .95),
                           "p99": quantile(mig_lat, .99), "n": len(mig_lat)},
            "pinned": {"p50": quantile(pin_lat, .5),
                       "p95": quantile(pin_lat, .95),
                       "p99": quantile(pin_lat, .99), "n": len(pin_lat)}},
        "migration_cost_ms": {
            "n": len(prim_ms),
            "primitive": {"p50": quantile(prim_ms, .5),
                          "p95": quantile(prim_ms, .95),
                          "p99": quantile(prim_ms, .99),
                          "vals": prim_ms},
            "l2_probe": {"p50": quantile(l2_ms, .5),
                         "p95": quantile(l2_ms, .95),
                         "p99": quantile(l2_ms, .99),
                         "vals": l2_ms}},
        "clause2_fails": clause2_fails,
        "stranding": {"max": max(strandeds) if strandeds else 0,
                      "mean": (sum(strandeds) / len(strandeds))
                      if strandeds else 0.0},
    }
    json.dump(res, open("%s/sc6_scenario_%s.json" % (HERE, seed), "w"),
              default=str)
    print("  %s: kl_gate=%s clause2_fails=%d mig_lat[p50=%.1f p99=%.1f] "
          "pin_lat[p50=%.1f p99=%.1f] migr_cost[n=%d primitive p50=%.2f "
          "p99=%.2f | L2 p50=%.2f] stranding_mean=%.2f"
          % (seed, kl_ok, len(clause2_fails),
             res["within_A_latency_ms"]["migratable"]["p50"],
             res["within_A_latency_ms"]["migratable"]["p99"],
             res["within_A_latency_ms"]["pinned"]["p50"],
             res["within_A_latency_ms"]["pinned"]["p99"],
             res["migration_cost_ms"]["n"],
             res["migration_cost_ms"]["primitive"]["p50"],
             res["migration_cost_ms"]["primitive"]["p99"],
             res["migration_cost_ms"]["l2_probe"]["p50"],
             res["stranding"]["mean"]),
          flush=True)
    time.sleep(2.0)
    return res


def main():
    seeds = sys.argv[1:] or list(SEEDS.keys())
    set_gap(1)                              # config A
    allres = [run_scenario(s) for s in seeds]
    json.dump({"scenarios": allres},
              open(HERE + "/sc6_churn_results.json", "w"), default=str)
    print("\nwrote sc6_churn_results.json", flush=True)


if __name__ == "__main__":
    main()
