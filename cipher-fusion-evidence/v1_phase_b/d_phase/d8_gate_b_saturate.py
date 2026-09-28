#!/usr/bin/env python3
# D.8 Gate B (SHIELD p99) under the SAME saturating aggressor as Gate A run 2
# (advisor: the prior Gate B leaned on the weak B=1 neighbor). highband =
# latency-sensitive B=1 victim (small frequent gens), aggressor = 8192^3 bf16
# matmul flood. Measures highband p99: solo / OFF(neighbor) / ON(SHIELD band1).
import os, sys, json, subprocess, tempfile
LIB = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ubuntu/cipher_rt_phase4/build_cuda13/libcipher_rt.so.d8_staging"
PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__)); TENANT = os.path.join(HERE, "d8_tenant.py")
SECS = float(os.environ.get("D8_SECONDS", "20")); SAT = os.environ.get("D8_SATURATE_DIM", "8192")
THROTTLE_US = os.environ.get("D8_THROTTLE_US", "500"); TMP = tempfile.mkdtemp(prefix="d8b_")

def spawn(role, out, env_extra, gen_tokens=8, think_ms=30.0, saturate=0):
    env = dict(os.environ); env["CIPHER_FAIRNESS_THROTTLE_US"] = THROTTLE_US; env.update(env_extra)
    return subprocess.Popen([PY, TENANT, "--role", role, "--lib", LIB, "--seconds", str(SECS),
        "--gen-tokens", str(gen_tokens), "--think-ms", str(think_ms), "--saturate", str(saturate),
        "--out", out], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def run_set(specs):
    procs = [(out, spawn(*a)) for (out, *a) in specs]; res = []
    for out, p in procs:
        o, _ = p.communicate(); sys.stdout.write(o.decode(errors="replace"))
        try: res.append(json.load(open(out)))
        except Exception as e: res.append({"error": str(e)})
    return res
def hb_p99(res): return next((r.get("lat_p99_ms", 0) for r in res if r.get("role") == "highband"), 0)

solo = run_set([(f"{TMP}/hb_solo.json", "highband", f"{TMP}/hb_solo.json", {}, 8, 30.0, 0)])
off = run_set([(f"{TMP}/hb_off.json", "highband", f"{TMP}/hb_off.json", {}, 8, 30.0, 0),
               (f"{TMP}/agg_off.json", "noisy", f"{TMP}/agg_off.json", {}, 8, 0.0, int(SAT))])
on = run_set([(f"{TMP}/hb_on.json", "highband", f"{TMP}/hb_on.json",
               {"CIPHER_SHIELD": "on", "CIPHER_SHIELD_BAND": "1"}, 8, 30.0, 0),
              (f"{TMP}/agg_on.json", "noisy", f"{TMP}/agg_on.json",
               {"CIPHER_SHIELD": "on", "CIPHER_SHIELD_BAND": "0"}, 8, 0.0, int(SAT))])
p_solo, p_off, p_on = hb_p99(solo), hb_p99(off), hb_p99(on)
damage = p_off > p_solo * 1.15   # neighbor demonstrably inflated p99 under OFF
print("\n" + "=" * 70)
print(f"[SAT-B] highband p99 (ms): SOLO={p_solo:.1f}  OFF(aggr)={p_off:.1f}  ON(SHIELD)={p_on:.1f}")
print(f"[SAT-B] p99 damage under OFF: {'YES' if damage else 'NO'} (OFF vs SOLO {100*(p_off-p_solo)/max(p_solo,1e-9):+.1f}%)")
verdict = ("GATE B PASS" if (damage and p_on < p_off * 0.85 and p_on <= p_solo * 2.0)
           else "GATE B FAIL — no noisy-neighbor p99 damage to shield (saturating aggressor, latency-bound victim)"
                if not damage else "GATE B FAIL — SHIELD did not bound p99")
print(f"[SAT-B] {verdict}"); print("=" * 70)
json.dump({"p99_solo": p_solo, "p99_off": p_off, "p99_on": p_on, "damage": damage, "verdict": verdict},
          open("/home/ubuntu/d8_gate_b_saturate_result.json", "w"), indent=2)
