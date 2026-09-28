#!/usr/bin/env python3
# D.8 DEEP DIAGNOSIS — is the gate failure real, or a harness artifact?
# Replaces the overhead-bound HF-generate latency probe with a CLEAN
# CUDA-event-timed small-GEMM victim loop (thousands of samples), against a
# genuinely saturating aggressor (separate process). Isolates GPU-queue
# contention from Python/generate() overhead. Sweeps the throttle magnitude so
# we can tell a fundamental mechanism failure from an under-sized sleep.
#
# Modes:
#   victim  : tight CUDA-event-timed small bf16 GEMM loop; prints latency dist.
#   aggressor (delegated to d8_tenant --role noisy --saturate)
#   orchestrate (default): baseline -> +aggressor OFF -> +aggressor ON@throttle sweep
import os, sys, json, time, ctypes, subprocess, statistics

LIB = "/home/ubuntu/cipher_rt_phase4/build_cuda13/libcipher_rt.so.d8_staging"
HERE = os.path.dirname(os.path.abspath(__file__)); TENANT = os.path.join(HERE, "d8_tenant.py")

def victim_probe(n_iter, vdim, think_us, out):
    lib = ctypes.CDLL(LIB, mode=ctypes.RTLD_GLOBAL)
    import torch
    x = torch.randn(vdim, vdim, dtype=torch.bfloat16, device="cuda")
    y = torch.randn(vdim, vdim, dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.Stream()
    ev0 = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ev1 = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    # warmup
    for _ in range(50):
        z = x @ y
    torch.cuda.synchronize()
    lat = []
    for i in range(n_iter):
        ev0[i].record()
        z = x @ y
        ev1[i].record()
        torch.cuda.synchronize()            # measure submit->complete incl. queue wait
        lat.append(ev0[i].elapsed_time(ev1[i]))   # ms (pure GPU time)
        if think_us > 0:
            time.sleep(think_us / 1e6)
    lat.sort()
    def p(q): return lat[min(len(lat)-1, int(q/100*len(lat)))]
    res = {"n": len(lat), "vdim": vdim, "mean_ms": statistics.mean(lat),
           "p50_ms": p(50), "p90_ms": p(90), "p99_ms": p(99), "max_ms": lat[-1], "min_ms": lat[0]}
    json.dump(res, open(out, "w"))
    print(f"[victim vdim={vdim} n={res['n']}] gpu-time ms: min={res['min_ms']:.3f} "
          f"p50={res['p50_ms']:.3f} p90={res['p90_ms']:.3f} p99={res['p99_ms']:.3f} "
          f"max={res['max_ms']:.3f} mean={res['mean_ms']:.3f}", flush=True)
    return res

if len(sys.argv) > 1 and sys.argv[1] == "victim":
    victim_probe(int(os.environ.get("N_ITER","3000")), int(os.environ.get("VDIM","512")),
                 float(os.environ.get("THINK_US","200")), sys.argv[2])
    sys.exit(0)

# ---- orchestrate ----
SECS = os.environ.get("D8_SECONDS","30")
def spawn_aggressor(env_extra):
    env = dict(os.environ); env.update(env_extra)
    return subprocess.Popen([sys.executable, TENANT, "--role","noisy","--lib",LIB,
        "--seconds",SECS,"--saturate",os.environ.get("SAT","8192"),
        "--out","/tmp/diag_agg.json"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
def run_victim(tag):
    env = dict(os.environ)  # victim disarmed (band-0, but we don't arm it)
    for k in ("CIPHER_FAIRNESS","CIPHER_SHIELD","CIPHER_SHIELD_BAND","CIPHER_FAIRNESS_FORCE_YIELD"):
        env.pop(k, None)
    out = f"/tmp/diag_victim_{tag}.json"
    p = subprocess.run([sys.executable, __file__, "victim", out], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    sys.stdout.write(p.stdout.decode(errors="replace"))
    return json.load(open(out))

print("="*70, "\nD.8 DEEP DIAGNOSIS — clean CUDA-event victim probe vs saturating aggressor\n", "="*70)
base = run_victim("baseline")                       # no aggressor

results = {"baseline": base, "off": None, "on": {}}
agg = spawn_aggressor({})                            # aggressor, no enforcement
time.sleep(3)
results["off"] = run_victim("off")
agg.terminate();
try: agg.wait(timeout=10)
except Exception: agg.kill()

for thr in [int(x) for x in os.environ.get("THROTTLE_SWEEP","1000,4000,16000").split(",")]:
    agg = spawn_aggressor({"CIPHER_FAIRNESS":"on","CIPHER_FAIRNESS_THROTTLE_US":str(thr),
                           "CIPHER_FAIRNESS_BURST_MIN":"0"})  # force aggressor to be throttle-eligible
    time.sleep(3)
    results["on"][thr] = run_victim(f"on{thr}")
    try: ag = json.load(open("/tmp/diag_agg.json")); results["on"][thr]["agg_yields"] = ag.get("self_yields")
    except Exception: pass
    agg.terminate()
    try: agg.wait(timeout=10)
    except Exception: agg.kill()

print("\n" + "="*70 + "\nSUMMARY (victim GPU-time p99, ms)")
print(f"  baseline (no aggressor): p50={base['p50_ms']:.3f}  p99={base['p99_ms']:.3f}  max={base['max_ms']:.3f}")
o = results["off"]
print(f"  + aggressor OFF        : p50={o['p50_ms']:.3f}  p99={o['p99_ms']:.3f}  max={o['max_ms']:.3f}  (p99 inflation {o['p99_ms']/base['p99_ms']:.1f}x)")
for thr, r in results["on"].items():
    print(f"  + aggressor ON @{thr}us : p50={r['p50_ms']:.3f}  p99={r['p99_ms']:.3f}  max={r['max_ms']:.3f}  (yields={r.get('agg_yields')})  -> p99 vs OFF {r['p99_ms']/o['p99_ms']:.2f}x, vs base {r['p99_ms']/base['p99_ms']:.1f}x")
print("="*70)
json.dump(results, open("/home/ubuntu/d8_diag_result.json","w"), indent=2)
