#!/usr/bin/env python3
"""CIPHER multi-tenant 7-problems POC orchestrator.

Runs 6 experiments and produces the "7 problems solved" table.
"""
import argparse, json, os, signal, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = "/home/ubuntu/models/Llama-3.2-1B"


def gpu_sampler(stop, samples, gpu_id=0):
    while not stop.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_id),
                 "--query-gpu=power.draw,memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1)
            parts = r.stdout.strip().split(", ")
            if len(parts) == 3:
                samples.append((float(parts[0]), float(parts[1]),
                                float(parts[2])))
        except Exception:
            pass
        time.sleep(0.2)


def base_env(load_cipher=True):
    e = dict(os.environ)
    if load_cipher:
        e["LD_PRELOAD"] = (f"{ROOT}/libcipher_hook.so "
                           "/usr/lib/x86_64-linux-gnu/libcuda.so")
        e["CIPHER_FP8_COMPUTE"]   = "on"
        e["CIPHER_SUBSTITUTE_V2"] = "on"
        e["CIPHER_FUSION_KERNELS"]= "on"
        e["CIPHER_NCCL_V4"]       = "on"
        e["CIPHER_CARBON"]        = "on"
        e["CIPHER_FAIRNESS"]      = "on"
        e["CIPHER_SHIELD"]        = "on"
        e["CIPHER_SENSE"]         = "on"
        e["CIPHER_THERMOSTAT"]    = "on"
        e["CIPHER_PULSE"]         = "on"
        e["CIPHER_VOLT"]          = "on"
        e["CIPHER_HIBERNATE"]     = "on"
        e["CIPHER_TRACE"]         = "on"
        e["CIPHER_RECEIPT"]       = "on"
        e["MT_LOAD_RT"]           = "1"
    else:
        e.pop("LD_PRELOAD", None)
        e["MT_LOAD_RT"] = "0"
    e["CUDA_VISIBLE_DEVICES"] = "0"
    e["MT_MODEL_PATH"] = MODEL
    return e


def run_tenants(child_script, n, duration, extra_env=None,
                load_cipher=True, prefix="mt"):
    out_files = [f"/tmp/{prefix}_t{i}.json" for i in range(n)]
    log_files = [f"/tmp/{prefix}_t{i}.log" for i in range(n)]
    for f in out_files:
        if os.path.exists(f): os.remove(f)
    env = base_env(load_cipher)
    env["MT_DURATION_S"] = str(duration)
    if extra_env: env.update(extra_env)

    samples = []
    stop = threading.Event()
    th = threading.Thread(target=gpu_sampler, args=(stop, samples), daemon=True)
    th.start()

    procs = []
    t0 = time.perf_counter()
    for i in range(n):
        e = dict(env)
        e["CIPHER_TENANT_ID"] = str(i)
        e["MT_OUT_PATH"] = out_files[i]
        log = open(log_files[i], "w")
        p = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, child_script)],
            env=e, stdout=log, stderr=subprocess.STDOUT)
        procs.append((p, log))
    rcs = []
    for p, log in procs:
        rc = p.wait()
        log.close()
        rcs.append(rc)
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)

    tenants = []
    for f in out_files:
        if os.path.exists(f):
            tenants.append(json.load(open(f)))
        else:
            tenants.append({"error": "no output"})

    # Steady-state samples (drop first 1/4, last 1/4 to clip startup/teardown).
    nsamp = len(samples)
    if nsamp >= 8:
        steady = samples[nsamp//4:3*nsamp//4]
    else:
        steady = samples or [(0,0,0)]
    pw = [s[0] for s in steady]
    util = [s[2] for s in steady]
    mem = [s[1] for s in steady]
    return dict(
        n=n, elapsed=elapsed, rcs=rcs, tenants=tenants,
        watts_mean=sum(pw)/len(pw),
        util_mean=sum(util)/len(util),
        mem_mean_mb=sum(mem)/len(mem),
    )


def heavy_then_decode(n_decode, duration, with_heavy):
    """Spawn n_decode decode tenants + optionally 1 heavy-prefill tenant."""
    n_total = n_decode + (1 if with_heavy else 0)
    out_files = [f"/tmp/mt_n{i}.json" for i in range(n_total)]
    log_files = [f"/tmp/mt_n{i}.log" for i in range(n_total)]
    for f in out_files:
        if os.path.exists(f): os.remove(f)
    env = base_env(True)
    env["MT_DURATION_S"] = str(duration)
    samples = []; stop = threading.Event()
    th = threading.Thread(target=gpu_sampler, args=(stop, samples), daemon=True)
    th.start()

    procs = []
    t0 = time.perf_counter()
    if with_heavy:
        # Tenant 0 = heavy prefill.
        e = dict(env)
        e["CIPHER_TENANT_ID"] = "0"
        e["MT_OUT_PATH"] = out_files[0]
        log = open(log_files[0], "w")
        p = subprocess.Popen([sys.executable,
            os.path.join(ROOT, "multi_tenant_prefill_child.py")],
            env=e, stdout=log, stderr=subprocess.STDOUT)
        procs.append((p, log))
    # Decode tenants.
    for i in range(n_decode):
        idx = i + (1 if with_heavy else 0)
        e = dict(env)
        e["CIPHER_TENANT_ID"] = str(idx)
        e["MT_OUT_PATH"] = out_files[idx]
        log = open(log_files[idx], "w")
        p = subprocess.Popen([sys.executable,
            os.path.join(ROOT, "multi_tenant_child.py")],
            env=e, stdout=log, stderr=subprocess.STDOUT)
        procs.append((p, log))
    rcs = []
    for p, log in procs:
        rc = p.wait()
        log.close()
        rcs.append(rc)
    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)

    tenants = []
    for f in out_files:
        if os.path.exists(f):
            tenants.append(json.load(open(f)))
        else:
            tenants.append({"error": "no output"})

    nsamp = len(samples)
    steady = samples[nsamp//4:3*nsamp//4] if nsamp >= 8 else (samples or [(0,0,0)])
    pw = [s[0] for s in steady]; util = [s[2] for s in steady]
    mem = [s[1] for s in steady]
    return dict(
        n_decode=n_decode, with_heavy=with_heavy,
        elapsed=elapsed, rcs=rcs, tenants=tenants,
        watts_mean=sum(pw)/len(pw),
        util_mean=sum(util)/len(util),
        mem_mean_mb=sum(mem)/len(mem),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="1,2,3,5,6", help="experiments to run")
    ap.add_argument("--out", default="poc_7problems.json")
    ap.add_argument("--duration", type=float, default=20.0)
    args = ap.parse_args()
    exps = set(args.exp.split(","))
    results = {}

    print("="*100)
    print(" CIPHER 7-PROBLEMS MULTI-TENANT POC")
    print(f" model={MODEL}  duration={args.duration}s per experiment")
    print("="*100)

    if "1" in exps:
        print("\n=== EXPERIMENT 1 — utilization scaling (continuous decode) ===")
        e1 = {}
        for n in [1, 2, 4, 8, 15]:
            r = run_tenants("multi_tenant_child.py", n, args.duration,
                            prefix=f"e1_n{n}")
            e1[n] = r
            valid = [t for t in r["tenants"] if "error" not in t]
            agg_tps = sum(t.get("tps",0) for t in valid)
            tok_w = agg_tps / r["watts_mean"] if r["watts_mean"] > 0 else 0
            print(f"  N={n:>2}  agg_tps={agg_tps:>7.1f}  watts={r['watts_mean']:>5.1f}  "
                  f"tok/W={tok_w:.4f}  util={r['util_mean']:>4.1f}%  "
                  f"mem={r['mem_mean_mb']/1024:>4.1f}GB  rcs={set(r['rcs'])}")
        results["exp1"] = e1

    if "2" in exps:
        print("\n=== EXPERIMENT 2 — bursty agent (Problem 5) ===")
        # Solo baseline.
        print("  [2a] 1 tenant solo bursty baseline...")
        solo = run_tenants("multi_tenant_burst_child.py", 1, 60,
                           prefix="e2_solo")
        # 15 concurrent.
        print("  [2b] 15 tenants concurrent bursty...")
        concur = run_tenants("multi_tenant_burst_child.py", 15, 60,
                             prefix="e2_15")
        results["exp2"] = {"solo": solo, "concur": concur}
        for label, r in [("solo (1)", solo), ("concur (15)", concur)]:
            valid = [t for t in r["tenants"] if "error" not in t]
            if not valid:
                print(f"  {label}: no valid"); continue
            p50 = sorted([t["burst_p50_us"]/1000 for t in valid])
            p95 = sorted([t["burst_p95_us"]/1000 for t in valid])
            p99 = sorted([t["burst_p99_us"]/1000 for t in valid])
            n_bursts = sum(t["n_bursts"] for t in valid)
            print(f"  {label:<14} bursts_total={n_bursts:>4}  "
                  f"P50={min(p50):.1f}-{max(p50):.1f}ms  "
                  f"P95={min(p95):.1f}-{max(p95):.1f}ms  "
                  f"P99={min(p99):.1f}-{max(p99):.1f}ms  "
                  f"watts={r['watts_mean']:.1f}")

    if "3" in exps:
        print("\n=== EXPERIMENT 3 — noisy neighbor / SHIELD (Problem 6) ===")
        print("  [3a] 14 decode + 1 HEAVY PREFILL tenant...")
        with_heavy = heavy_then_decode(14, args.duration, with_heavy=True)
        print("  [3b] 14 decode tenants only (no heavy)...")
        no_heavy = heavy_then_decode(14, args.duration, with_heavy=False)
        results["exp3"] = {"with_heavy": with_heavy, "no_heavy": no_heavy}
        for label, r in [("WITH heavy ", with_heavy), ("WITHOUT heavy", no_heavy)]:
            decode_tenants = [t for t in r["tenants"]
                               if "error" not in t and t.get("mode") != "heavy_prefill"]
            if not decode_tenants:
                print(f"  {label}: no valid decode"); continue
            agg_tps = sum(t.get("tps",0) for t in decode_tenants)
            p50 = sorted([t["lat_p50_us"]/1000 for t in decode_tenants])
            p95 = sorted([t["lat_p95_us"]/1000 for t in decode_tenants])
            p99 = sorted([t["lat_p99_us"]/1000 for t in decode_tenants])
            print(f"  {label}  decode_agg_tps={agg_tps:.1f}  "
                  f"P50={min(p50):.1f}-{max(p50):.1f}ms  "
                  f"P95={min(p95):.1f}-{max(p95):.1f}ms  "
                  f"P99={min(p99):.1f}-{max(p99):.1f}ms  "
                  f"watts={r['watts_mean']:.1f}")

    if "5" in exps:
        print("\n=== EXPERIMENT 5 — moat: WITHOUT CIPHER (Problem 7) ===")
        no_cipher = run_tenants("multi_tenant_child.py", 15, args.duration,
                                 load_cipher=False, prefix="e5_nc15")
        results["exp5"] = {"no_cipher": no_cipher}
        valid = [t for t in no_cipher["tenants"] if "error" not in t]
        agg_tps = sum(t.get("tps",0) for t in valid)
        p50 = sorted([t["lat_p50_us"]/1000 for t in valid]) if valid else [0]
        p99 = sorted([t["lat_p99_us"]/1000 for t in valid]) if valid else [0]
        print(f"  N=15 NO CIPHER  agg_tps={agg_tps:.1f}  watts={no_cipher['watts_mean']:.1f}  "
              f"P50={min(p50):.1f}-{max(p50):.1f}ms  P99={min(p99):.1f}-{max(p99):.1f}ms  "
              f"util={no_cipher['util_mean']:.1f}%")

    if "6" in exps:
        # Carbon: run 15 tenants with CIPHER_CARBON=on for 30s, query stats.
        print("\n=== EXPERIMENT 6 — per-tenant energy (CARBON op) ===")
        e6 = run_tenants("multi_tenant_child.py", 15, 30,
                         extra_env={"CIPHER_CARBON_J_PER_UNIT": "1e-12"},
                         prefix="e6")
        results["exp6"] = e6
        valid = [t for t in e6["tenants"] if "error" not in t]
        # Per-tenant energy = (total wall energy) × (per-tenant tps / agg tps).
        agg_tps = sum(t.get("tps",0) for t in valid)
        wall_J = e6["watts_mean"] * e6["elapsed"]
        print(f"  Total wall energy: {wall_J:.0f} J over {e6['elapsed']:.1f}s "
              f"({e6['watts_mean']:.1f}W avg)")
        print(f"  Aggregate tps: {agg_tps:.1f}, per-tenant proportional split:")
        for t in valid:
            share = t.get("tps",0) / agg_tps if agg_tps > 0 else 0
            tj = wall_J * share
            print(f"    tenant {t['tenant_id']:>2}  tps={t['tps']:>6.1f}  "
                  f"share={share*100:>4.1f}%  J={tj:.1f}  J/tok={tj/(t['tps']*e6['elapsed']):.3f}")

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n[poc] wrote {args.out}")


if __name__ == "__main__":
    main()
