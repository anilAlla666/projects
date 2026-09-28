#!/usr/bin/env python3
"""Multi-tenant POC parent — spawns N child processes on GPU 0 and
collects per-tenant + aggregate stats.

Usage:
    python3 multi_tenant_poc.py --tenants=2 --duration=30
    python3 multi_tenant_poc.py --tenants=1,2 --duration=30 --out=poc_phase1.json
"""
import argparse, json, os, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.abspath(__file__))


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


def run_n_tenants(n, duration_s, model_path, prefill_len, env_extra):
    out_files = [f"/tmp/mt_tenant_{i}.json" for i in range(n)]
    for f in out_files:
        if os.path.exists(f): os.remove(f)

    # Common env across all children.
    base_env = dict(os.environ)
    base_env["LD_PRELOAD"] = (f"{ROOT}/libcipher_hook.so "
                              "/usr/lib/x86_64-linux-gnu/libcuda.so")
    base_env["CUDA_VISIBLE_DEVICES"] = "0"
    base_env["CIPHER_FP8_COMPUTE"] = "on"
    base_env["CIPHER_SUBSTITUTE_V2"] = "on"
    base_env["CIPHER_FUSION_KERNELS"] = "on"
    base_env["CIPHER_NCCL_V4"] = "on"
    base_env["MT_MODEL_PATH"] = model_path
    base_env["MT_DURATION_S"] = str(duration_s)
    base_env["MT_PREFILL"] = str(prefill_len)
    base_env.update(env_extra)

    # Sampler thread.
    samples = []
    stop = threading.Event()
    th = threading.Thread(target=gpu_sampler, args=(stop, samples),
                          daemon=True)

    print(f"\n[poc] spawning {n} tenant(s) on GPU 0, duration={duration_s}s",
          flush=True)
    procs = []
    t0 = time.perf_counter()
    for i in range(n):
        env = dict(base_env)
        env["CIPHER_TENANT_ID"] = str(i)
        env["MT_OUT_PATH"] = out_files[i]
        log_path = f"/tmp/mt_tenant_{i}.log"
        log = open(log_path, "w")
        p = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "multi_tenant_child.py")],
            env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((p, log))

    # Wait until all children have started timed loop, then begin sampler.
    # Coarsest signal: wait ~30s for them to load model + warmup.  Sampler
    # runs concurrently anyway and we slice samples post-hoc.
    th.start()

    rcs = []
    for p, log in procs:
        rc = p.wait()
        log.close()
        rcs.append(rc)

    elapsed = time.perf_counter() - t0
    stop.set(); th.join(timeout=2)
    print(f"[poc] all {n} tenants finished in {elapsed:.1f}s "
          f"(rcs={rcs})", flush=True)

    tenants = []
    for f in out_files:
        if os.path.exists(f):
            tenants.append(json.load(open(f)))
        else:
            tenants.append({"error": "no output", "tenant_id": -1})

    # Use samples from the second half of the run (steady state).
    n_samp = len(samples)
    steady = samples[n_samp//2:] if n_samp >= 4 else samples
    if not steady: steady = [(0,0,0)]
    pw = [s[0] for s in steady]
    mem = [s[1] for s in steady]
    util = [s[2] for s in steady]
    gpu_stats = dict(
        n_samples=len(samples),
        steady_samples=len(steady),
        watts_mean=sum(pw)/len(pw),
        watts_max=max(pw),
        mem_used_mb_mean=sum(mem)/len(mem),
        util_pct_mean=sum(util)/len(util),
    )

    # Aggregate.
    valid = [t for t in tenants if "error" not in t]
    agg = dict(
        n_tenants=n,
        n_valid=len(valid),
        agg_tps=sum(t["tps"] for t in valid),
        elapsed_s=elapsed,
        tenants=tenants,
        gpu=gpu_stats,
        agg_tok_w=(sum(t["tps"] for t in valid) / gpu_stats["watts_mean"]
                   if gpu_stats["watts_mean"] > 0 else 0),
    )
    return agg


def print_row(label, stats):
    g = stats["gpu"]
    valid = [t for t in stats["tenants"] if "error" not in t]
    if not valid:
        print(f"{label}: NO VALID TENANTS"); return
    p50s = sorted([t["lat_p50_us"] for t in valid])
    p95s = sorted([t["lat_p95_us"] for t in valid])
    p99s = sorted([t["lat_p99_us"] for t in valid])
    per_tenant_tps = sorted([t["tps"] for t in valid])
    print(f"{label:<20}  agg_tps={stats['agg_tps']:>7.1f}  "
          f"per_tenant_tps={per_tenant_tps[0]:>5.1f}-{per_tenant_tps[-1]:.1f}  "
          f"watts={g['watts_mean']:>5.1f}  "
          f"agg_tok/W={stats['agg_tok_w']:>5.4f}  "
          f"util={g['util_pct_mean']:>4.1f}%  "
          f"mem={g['mem_used_mb_mean']/1024:>4.1f}GB  "
          f"P50={min(p50s):.0f}-{max(p50s):.0f}us  "
          f"P95={min(p95s):.0f}-{max(p95s):.0f}us  "
          f"P99={min(p99s):.0f}-{max(p99s):.0f}us")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenants", default="1,2",
                    help="comma list of tenant counts to test")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--model", default="/home/ubuntu/models/Llama-3.1-8B")
    ap.add_argument("--prefill", type=int, default=128)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    tenant_counts = [int(x) for x in args.tenants.split(",")]

    print("="*100)
    print(" CIPHER MULTI-TENANT POC")
    print(f" model={args.model}  duration={args.duration}s  prefill={args.prefill}")
    print("="*100)

    results = []
    for n in tenant_counts:
        stats = run_n_tenants(n, args.duration, args.model, args.prefill, {})
        results.append(stats)
        print_row(f"N={n}", stats)

    if args.out:
        json.dump({"runs": results}, open(args.out, "w"), indent=2)
        print(f"\n[poc] wrote {args.out}")

    # Build comparison table.
    print()
    print("="*100)
    print(" SUMMARY (all numbers steady-state)")
    print("="*100)
    print(f"{'tenants':>7}  {'agg tps':>8}  {'per-tenant tps':>14}  "
          f"{'watts':>5}  {'tok/W':>6}  {'util%':>5}  "
          f"{'mem':>5}  {'P50 us':>10}  {'P95 us':>10}  {'P99 us':>10}")
    for r in results:
        valid = [t for t in r["tenants"] if "error" not in t]
        if not valid: continue
        per_tps = sorted([t["tps"] for t in valid])
        p50 = sorted([t["lat_p50_us"] for t in valid])
        p95 = sorted([t["lat_p95_us"] for t in valid])
        p99 = sorted([t["lat_p99_us"] for t in valid])
        g = r["gpu"]
        per_str = (f"{per_tps[0]:.1f}-{per_tps[-1]:.1f}"
                   if len(per_tps) > 1 else f"{per_tps[0]:.1f}")
        p50s = (f"{min(p50):.0f}-{max(p50):.0f}" if len(p50) > 1
                else f"{p50[0]:.0f}")
        p95s = (f"{min(p95):.0f}-{max(p95):.0f}" if len(p95) > 1
                else f"{p95[0]:.0f}")
        p99s = (f"{min(p99):.0f}-{max(p99):.0f}" if len(p99) > 1
                else f"{p99[0]:.0f}")
        print(f"{r['n_tenants']:>7}  {r['agg_tps']:>8.1f}  {per_str:>14}  "
              f"{g['watts_mean']:>5.1f}  {r['agg_tok_w']:>6.4f}  "
              f"{g['util_pct_mean']:>5.1f}  "
              f"{g['mem_used_mb_mean']/1024:>5.1f}  "
              f"{p50s:>10}  {p95s:>10}  {p99s:>10}")


if __name__ == "__main__":
    main()
