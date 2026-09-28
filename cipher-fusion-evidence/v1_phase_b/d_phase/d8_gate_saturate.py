#!/usr/bin/env python3
# D.8 Gate A under a REAL saturating aggressor (advisor: the prior B=1 "noisy"
# could not starve an H100). Aggressor = continuous large bf16 matmul flood
# (FLOP-heavy, the realistic single-agent hog); victims = B=1 TinyLlama decode.
# Measures: solo victim tok/s (no neighbor) -> OFF (starvation?) -> ON (recovery?).
# Discriminator: aggressor self_yields + whether its big matmul is even visible
# to the cublasGemmEx hook (natural_hook_calls).
import os, sys, json, time, subprocess, tempfile

LIB = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ubuntu/cipher_rt_phase4/build_cuda13/libcipher_rt.so.d8_staging"
PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
TENANT = os.path.join(HERE, "d8_tenant.py")
SECS = float(os.environ.get("D8_SECONDS", "25"))
SAT = os.environ.get("D8_SATURATE_DIM", "8192")
THROTTLE_US = os.environ.get("D8_THROTTLE_US", "500")
NVIC = int(os.environ.get("D8_VICTIMS", "3"))
TMP = tempfile.mkdtemp(prefix="d8sat_")

def spawn(role, out, env_extra, gen_tokens=8, think_ms=10.0, saturate=0):
    env = dict(os.environ); env["CIPHER_FAIRNESS_THROTTLE_US"] = THROTTLE_US; env.update(env_extra)
    cmd = [PY, TENANT, "--role", role, "--lib", LIB, "--seconds", str(SECS),
           "--gen-tokens", str(gen_tokens), "--think-ms", str(think_ms),
           "--saturate", str(saturate), "--out", out]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def run_set(specs):
    procs = [(out, spawn(*args)) for (out, *args) in specs]
    res = []
    for out, p in procs:
        o, _ = p.communicate(); sys.stdout.write(o.decode(errors="replace"))
        try: res.append(json.load(open(out)))
        except Exception as e: res.append({"error": str(e)})
    return res

def victim_tps(res): return sum(r.get("tokens_per_s", 0) for r in res if r.get("role") == "victim")
def noisy(res): return next((r for r in res if r.get("role") == "noisy"), {})

# 1) solo victims (no aggressor) — the unstarved baseline
solo = run_set([(f"{TMP}/v{i}_solo.json", "victim", f"{TMP}/v{i}_solo.json", {}, 8, 10.0, 0)
                for i in range(NVIC)])
solo_tps = victim_tps(solo)

# 2) OFF — aggressor + victims, no enforcement
def with_aggressor(armed):
    env = {"CIPHER_FAIRNESS": "on"} if armed else {}
    specs = [(f"{TMP}/noisy_{armed}.json", "noisy", f"{TMP}/noisy_{armed}.json", env, 8, 0.0, int(SAT))]
    for i in range(NVIC):
        specs.append((f"{TMP}/v{i}_{armed}.json", "victim", f"{TMP}/v{i}_{armed}.json", env, 8, 10.0, 0))
    return run_set(specs)

off = with_aggressor(False)
on  = with_aggressor(True)
off_tps, on_tps = victim_tps(off), victim_tps(on)
n_on = noisy(on)

starved = off_tps < solo_tps * 0.85          # aggressor demonstrably starved victims under OFF
recovered = on_tps > off_tps * 1.10          # enforcement recovered victim throughput
agg_yields = n_on.get("self_yields", 0)
agg_nat = n_on.get("natural_hook_calls_per_3matmul", None)

print("\n" + "=" * 70)
print(f"[SAT] victim aggregate tok/s: SOLO={solo_tps:.1f}  OFF(aggr)={off_tps:.1f}  ON(FAIRNESS)={on_tps:.1f}")
print(f"[SAT] starvation under OFF: {'YES' if starved else 'NO'} (OFF={off_tps:.1f} vs SOLO={solo_tps:.1f}, {100*(off_tps-solo_tps)/max(solo_tps,1e-9):+.1f}%)")
print(f"[SAT] aggressor self_yields ON={agg_yields}; big-matmul natural cublasGemmEx hook calls (per 3 matmul)={agg_nat}")
print(f"[SAT] victim recovery ON-vs-OFF: {'YES' if recovered else 'NO'} ({100*(on_tps-off_tps)/max(off_tps,1e-9):+.1f}%)")
verdict = ("GATE A PASS (starvation demonstrated + throttle fired + victims recovered)"
           if (starved and agg_yields > 0 and recovered) else
           "GATE A FAIL" + (" — no starvation even under saturation (H100 self-arbitrates)" if not starved else
                            (" — aggressor not flagged by call-count throttle (HARD STOP: wrong resource)" if agg_yields == 0 else
                             " — throttle fired but victims did not recover (HARD STOP: throttle can't free FLOP-bound GPU)")))
print(f"[SAT] {verdict}")
print("=" * 70)
json.dump({"solo_tps": solo_tps, "off_tps": off_tps, "on_tps": on_tps,
           "starved": starved, "recovered": recovered, "agg_yields": agg_yields,
           "agg_natural_hook": agg_nat, "verdict": verdict},
          open("/home/ubuntu/d8_gate_saturate_result.json", "w"), indent=2)
