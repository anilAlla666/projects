#!/usr/bin/env python3
# D.8 DEEP DIAGNOSIS v2 — fixes the v1 probe flaws:
#  (1) WALL-CLOCK submit->sync latency (captures GPU-queue/time-slice WAIT, the
#      thing v1's CUDA-event kernel-time probe was blind to);
#  (2) a GENUINELY saturating aggressor (deep async matmul queue, ~100% duty;
#      v1's sync-every-8 aggressor was only ~58% duty -> left gaps);
#  (3) DVFS control — the decisive comparison is OFF vs ON (both aggressor-
#      present, same clock state); baseline is a floor reference only.
# Question: under REAL saturation does a latency victim's p99 inflate, and does
# the throttle (swept) bound it? Distinguishes "harness artifact" / "tuning" /
# "fundamental".
import os, sys, json, time, ctypes, subprocess, statistics

LIB = "/home/ubuntu/cipher_rt_phase4/build_cuda13/libcipher_rt.so.d8_staging"

def saturating_aggressor():
    # ~100% GPU duty: keep a DEEP async queue of large matmuls; sync rarely.
    # Participates in the fairness ledger per submit (throttle-eligible).
    lib = ctypes.CDLL(LIB, mode=ctypes.RTLD_GLOBAL)
    lib.cipher_rt_fairness_record_and_maybe_throttle.restype = None
    lib.cipher_rt_fairness_self_yields.restype = ctypes.c_ulonglong
    lib.cipher_rt_fairness_armed.restype = ctypes.c_int
    import torch
    N = int(os.environ.get("SAT", "8192"))
    x = torch.randn(N, N, dtype=torch.bfloat16, device="cuda")
    y = torch.randn(N, N, dtype=torch.bfloat16, device="cuda")
    deadline = time.time() + float(os.environ.get("D8_SECONDS", "40"))
    it = 0
    while time.time() < deadline:
        for _ in range(64):           # deep queue, NO sync -> GPU stays saturated
            z = x @ y
            lib.cipher_rt_fairness_record_and_maybe_throttle()
        torch.cuda.synchronize()       # rare drain
        it += 64
        # write progress EACH loop so a terminate() before the deadline still
        # captures yields (v1 lost them — terminate raced the end-of-run write).
        try: json.dump({"matmul_iters": it, "self_yields": int(lib.cipher_rt_fairness_self_yields()),
                        "armed": int(lib.cipher_rt_fairness_armed())}, open("/tmp/diag2_agg.json", "w"))
        except Exception: pass
    sys.exit(0)

def victim_probe(out):
    # victim is DISARMED (a latency-sensitive tenant we protect, not throttle).
    for k in ("CIPHER_FAIRNESS","CIPHER_SHIELD","CIPHER_SHIELD_BAND","CIPHER_FAIRNESS_FORCE_YIELD"):
        os.environ.pop(k, None)
    ctypes.CDLL(LIB, mode=ctypes.RTLD_GLOBAL)
    import torch
    V = int(os.environ.get("VDIM", "1024"))
    x = torch.randn(V, V, dtype=torch.bfloat16, device="cuda")
    y = torch.randn(V, V, dtype=torch.bfloat16, device="cuda")
    n_iter = int(os.environ.get("N_ITER", "2000"))
    think = float(os.environ.get("THINK_US", "500")) / 1e6
    for _ in range(50):
        z = x @ y
    torch.cuda.synchronize()
    lat = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        z = x @ y
        torch.cuda.synchronize()       # WALL-CLOCK submit->complete incl. queue wait
        lat.append((time.perf_counter() - t0) * 1e3)   # ms
        if think > 0: time.sleep(think)
    lat.sort()
    def p(q): return lat[min(len(lat)-1, int(q/100*len(lat)))]
    res = {"n": len(lat), "vdim": V, "mean_ms": statistics.mean(lat),
           "p50_ms": p(50), "p90_ms": p(90), "p99_ms": p(99), "p999_ms": p(99.9),
           "max_ms": lat[-1], "min_ms": lat[0]}
    json.dump(res, open(out, "w"))
    print(f"[victim V={V} n={res['n']}] wall ms: min={res['min_ms']:.3f} p50={res['p50_ms']:.3f} "
          f"p90={res['p90_ms']:.3f} p99={res['p99_ms']:.3f} p99.9={res['p999_ms']:.3f} max={res['max_ms']:.3f}", flush=True)
    return res

if len(sys.argv) > 1 and sys.argv[1] == "aggressor": saturating_aggressor()
if len(sys.argv) > 1 and sys.argv[1] == "victim": victim_probe(sys.argv[2]); sys.exit(0)

# orchestrate
HERE = os.path.abspath(__file__)
def spawn_agg(env_extra):
    env = dict(os.environ); env.update(env_extra)
    return subprocess.Popen([sys.executable, HERE, "aggressor"], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
def victim(tag):
    p = subprocess.run([sys.executable, HERE, "victim", f"/tmp/diag2_v_{tag}.json"],
                       env=dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    sys.stdout.write(p.stdout.decode(errors="replace")); return json.load(open(f"/tmp/diag2_v_{tag}.json"))

print("="*72, "\nD.8 DEEP DIAGNOSIS v2 — wall-clock victim latency vs SATURATING aggressor\n", "="*72)
base = victim("base")
agg = spawn_agg({}); time.sleep(4); off = victim("off")
agg.terminate()
try: agg.wait(timeout=10)
except Exception: agg.kill()
try: aoff = json.load(open("/tmp/diag2_agg.json")); print(f"[aggressor OFF] matmul_iters={aoff.get('matmul_iters')} (duty proxy)")
except Exception: pass

on = {}
for thr in [int(x) for x in os.environ.get("THROTTLE_SWEEP","2000,8000,30000").split(",")]:
    # FORCE_YIELD: the diagnostic victim is disarmed (n_active=1 would suppress
    # should_yield), so force the aggressor to throttle to isolate the real
    # question — IF the aggressor sleeps, does the victim's p99 recover?
    agg = spawn_agg({"CIPHER_FAIRNESS":"on","CIPHER_FAIRNESS_FORCE_YIELD":"1",
                     "CIPHER_FAIRNESS_THROTTLE_US":str(thr),"CIPHER_FAIRNESS_BURST_MIN":"0"})
    time.sleep(4); on[thr] = victim(f"on{thr}")
    agg.terminate()
    try: agg.wait(timeout=10)
    except Exception: agg.kill()
    try: a = json.load(open("/tmp/diag2_agg.json")); on[thr]["agg_yields"]=a.get("self_yields"); on[thr]["agg_iters"]=a.get("matmul_iters")
    except Exception: pass

print("\n" + "="*72 + "\nSUMMARY — victim wall p99 (ms); OFF-vs-ON is the controlled comparison")
print(f"  baseline (no aggr) : p50={base['p50_ms']:.3f}  p99={base['p99_ms']:.3f}  max={base['max_ms']:.3f}")
print(f"  + aggressor OFF    : p50={off['p50_ms']:.3f}  p99={off['p99_ms']:.3f}  max={off['max_ms']:.3f}   (p99 inflation vs base {off['p99_ms']/max(base['p99_ms'],1e-9):.1f}x)")
for thr, r in on.items():
    print(f"  + aggr ON @{thr}us : p50={r['p50_ms']:.3f}  p99={r['p99_ms']:.3f}  max={r['max_ms']:.3f}   yields={r.get('agg_yields')} iters={r.get('agg_iters')}  p99 vs OFF {r['p99_ms']/max(off['p99_ms'],1e-9):.2f}x")
print("="*72)
json.dump({"baseline":base,"off":off,"on":on}, open("/home/ubuntu/d8_diag2_result.json","w"), indent=2)
