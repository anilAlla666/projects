#!/usr/bin/env python3
# D.8 FAIRNESS+SHIELD — Gate A (burst-fairness) + Gate B (SHIELD p99) orchestrator.
# Spawns adversarial multi-tenant sets (separate processes sharing /dev/cipher,
# the cross-tenant kmod ledger) and measures ON-vs-OFF per the D.8 memo §2.
# Requires: kmod 0.7.0 loaded (NR 32 + ledger), the D.8 staging .so, /dev/cipher.
import os, sys, json, time, subprocess, tempfile, statistics

LIB = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ubuntu/cipher_rt_phase4/build_cuda13/libcipher_rt.so.d8_staging"
PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
TENANT = os.path.join(HERE, "d8_tenant.py")
SECS = float(os.environ.get("D8_SECONDS", "30"))
THROTTLE_US = os.environ.get("D8_THROTTLE_US", "300")
TMP = tempfile.mkdtemp(prefix="d8_")

def spawn(role, out, env_extra, gen_tokens=32, think_ms=20.0):
    env = dict(os.environ)
    env["CIPHER_FAIRNESS_THROTTLE_US"] = THROTTLE_US
    env.update(env_extra)
    cmd = [PY, TENANT, "--role", role, "--lib", LIB, "--seconds", str(SECS),
           "--gen-tokens", str(gen_tokens), "--think-ms", str(think_ms), "--out", out]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def run_set(specs):
    procs = []
    for (role, out, env_extra, gt, tm) in specs:
        procs.append((out, spawn(role, out, env_extra, gt, tm)))
        time.sleep(0.5)  # stagger model loads to ease HBM spikes
    results = []
    for out, p in procs:
        o, _ = p.communicate()
        sys.stdout.write(o.decode(errors="replace"))
        try:
            results.append(json.load(open(out)))
        except Exception as e:
            results.append({"error": str(e)})
    return results

print("=" * 70, "\nGATE A — burst-fairness: 1 noisy + 3 victims (TinyLlama bf16)\n", "=" * 70)
def gate_a(armed):
    env = {"CIPHER_FAIRNESS": "on"} if armed else {}
    specs = [("noisy", f"{TMP}/a_noisy_{armed}.json", env, 128, 0.0)]
    for i in range(3):
        specs.append(("victim", f"{TMP}/a_victim{i}_{armed}.json", env, 32, 20.0))
    return run_set(specs)

a_off = gate_a(False)
a_on  = gate_a(True)
def vic_tps(res): return sum(r.get("tokens_per_s", 0) for r in res if r.get("role") == "victim")
def noisy(res):   return next((r for r in res if r.get("role") == "noisy"), {})
voff, von = vic_tps(a_off), vic_tps(a_on)
n_off, n_on = noisy(a_off), noisy(a_on)
# Honest share accounting (advisor): report victim share of total GEMM throughput,
# not just an ON-vs-OFF delta. fair victim share with 3 victims of 4 tenants = 0.75.
noff_tps, non_tps = n_off.get("tokens_per_s", 0), n_on.get("tokens_per_s", 0)
share_off = voff / max(voff + noff_tps, 1e-9)
share_on  = von / max(von + non_tps, 1e-9)
FAIR_VICTIM_SHARE = 3.0 / 4.0
starvation_off = share_off < FAIR_VICTIM_SHARE * 0.9   # victims demonstrably below fair under OFF
# PASS = noisy actually self-throttled AND victims' share moved toward fair.
# If OFF was already fair (no starvation), that is reported as a finding, not tuned.
gateA_pass = (n_on.get("self_yields", 0) > 0) and (n_on.get("armed") == 1) and (share_on > share_off)
print(f"\n[GATE A] victim aggregate tok/s: OFF={voff:.1f}  ON={von:.1f}  (Δ={100*(von-voff)/max(voff,1e-9):+.1f}%)")
print(f"[GATE A] noisy tok/s: OFF={noff_tps:.1f}  ON={non_tps:.1f}   noisy self_yields ON={n_on.get('self_yields',0)} (armed={n_on.get('armed')})")
print(f"[GATE A] victim SHARE of throughput: OFF={share_off:.3f}  ON={share_on:.3f}   (fair={FAIR_VICTIM_SHARE:.2f}; OFF-starvation={'YES' if starvation_off else 'no'})")
print(f"[GATE A] {'PASS' if gateA_pass else 'FAIL'} — noisy throttled + victim share corrected toward fair"
      + ("" if starvation_off else "  [NOTE: OFF showed limited starvation at this scale — report the measured split honestly]"))

print("\n" + "=" * 70, "\nGATE B — SHIELD p99: 1 highband(band1) + 1 noisy neighbor\n", "=" * 70)
def gate_b(mode):  # 'solo' | 'off' | 'on'
    hb_env = {}
    if mode == "on":
        hb_env = {"CIPHER_SHIELD": "on", "CIPHER_SHIELD_BAND": "1"}
    specs = [("highband", f"{TMP}/b_hb_{mode}.json", hb_env, 8, 30.0)]
    if mode != "solo":
        nb_env = {"CIPHER_SHIELD": "on", "CIPHER_SHIELD_BAND": "0"} if mode == "on" else {}
        specs.append(("noisy", f"{TMP}/b_noisy_{mode}.json", nb_env, 128, 0.0))
    return run_set(specs)

b_solo = gate_b("solo"); b_off = gate_b("off"); b_on = gate_b("on")
def hb_p99(res): return next((r.get("lat_p99_ms", 0) for r in res if r.get("role") == "highband"), 0)
p_solo, p_off, p_on = hb_p99(b_solo), hb_p99(b_off), hb_p99(b_on)
# PASS: SHIELD-ON p99 measurably tighter than OFF and near solo (<= ~2x solo)
gateB_pass = (p_off > 0 and p_on > 0 and p_on < p_off * 0.85 and p_on <= p_solo * 2.0)
print(f"\n[GATE B] highband p99 (ms): solo={p_solo:.1f}  OFF(noisy)={p_off:.1f}  ON(SHIELD)={p_on:.1f}")
print(f"[GATE B] {'PASS' if gateB_pass else 'FAIL'} — SHIELD bounds p99 under noisy neighbor (ON<<OFF, ON<=2x solo)")

print("\n" + "=" * 70)
print(f"D8 GATE A: {'PASS' if gateA_pass else 'FAIL'}   D8 GATE B: {'PASS' if gateB_pass else 'FAIL'}")
print(f"(artifacts in {TMP})")
print("=" * 70)
json.dump({"gateA_pass": gateA_pass, "gateB_pass": gateB_pass,
           "victim_tps_off": voff, "victim_tps_on": von,
           "noisy_tps_off": noff_tps, "noisy_tps_on": non_tps,
           "victim_share_off": share_off, "victim_share_on": share_on,
           "fair_victim_share": FAIR_VICTIM_SHARE, "off_starvation": starvation_off,
           "noisy_yields_on": n_on.get("self_yields", 0),
           "hb_p99_solo": p_solo, "hb_p99_off": p_off, "hb_p99_on": p_on},
          open("/home/ubuntu/d8_gates_result.json", "w"), indent=2)
sys.exit(0 if (gateA_pass and gateB_pass) else 1)
