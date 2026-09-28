#!/usr/bin/env python3
# MOCK of the cipher v2.0 deployable's orchestration core (the ONLY new code in
# the design): manifest router + injection-state dispatch + discipline gate +
# Class-D report. Dispatches each job to its regime handler as a subprocess in
# the correct (conflicting) injection state -- exactly as the real router must.
#
# Tested here against mock_handler.py (no GPU). In production these handlers are
# cipher_inc4.py / v0_phaseC_nf4_cofire.py / v0_phaseA_maxmfu.py.
import os, sys, json, re, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

def load(p):
    with open(os.path.join(HERE, p)) as f:
        return json.load(f)

def route(regime, cfg, job):
    spec = cfg["regimes"][regime]
    env = dict(os.environ)
    inject = {k: (cfg["so_path"] if v == "__SO__" else v) for k, v in spec["inject"].items()}
    env.update(inject)
    env["CIPHER_REGIME"] = regime
    inj_keys = [k for k in inject if k in ("CUDA_INJECTION64_PATH", "CIPHER_RT_DISABLE_AUTO_INIT")]
    print(f"  [route] regime={regime:<8} -> {spec['handler']:<16} "
          f"injection={inj_keys or 'default'}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, spec["handler"]), regime, json.dumps(job)],
                       env=env, capture_output=True, text=True, timeout=spec.get("timeout", 30))
    m = re.search(r"CIPHER_MOCK_RESULT (\{.*\})", r.stdout)
    if not m:
        return {"regime": regime, "error": "no result", "stderr": r.stderr[-200:]}
    return json.loads(m.group(1))

def off_byte_identical_gate():
    # MOCK of the discipline gate: real version runs a forward with the .so injected
    # + all actuators OFF and asserts bit-identical (sum+argmax) to a clean baseline.
    baseline = (-1336048.875000, 450)
    off = (-1336048.875000, 450)   # mock: identical by construction
    return baseline == off, baseline, off

def main():
    cfg = load(sys.argv[1] if len(sys.argv) > 1 else "platform.json")
    manifest = load(sys.argv[2] if len(sys.argv) > 2 else "jobs.json")
    print("=== CIPHER v2.0 (MOCK) -- Class-D manifest router ===", flush=True)
    print(f"config: so_path={cfg['so_path']}", flush=True)
    print(f"manifest: {len(manifest['jobs'])} jobs\n--- dispatch ---", flush=True)

    results = [route(j["regime"], cfg, j) for j in manifest["jobs"]]

    print("\n--- discipline gate ---", flush=True)
    ok, base, off = off_byte_identical_gate()
    print(f"  OFF byte-identical: baseline={base} off={off} -> {'PASS' if ok else 'FAIL'}", flush=True)

    # ----- Class-D scorecard (headline) -----
    print("\n=== CLASS-D SCORECARD (MOCK) ===", flush=True)
    agent = next((r for r in results if r.get("regime") == "agent"), {})
    dens  = next((r for r in results if r.get("regime") == "density"), {})
    am, dm = agent.get("metrics", {}), dens.get("metrics", {})
    # p99 may be ms (real handler) or s (mock); render uniformly in ms
    p99s = am.get("p99_short_ms", (am.get("p99_short_s") or 0)*1000 if am.get("p99_short_s") else None)
    p99l = am.get("p99_long_ms",  (am.get("p99_long_s")  or 0)*1000 if am.get("p99_long_s")  else None)
    # density GB/7B: largest per-model footprint (the 7B one) for real, else mock gb_per_7b
    gb7b = dm.get("gb_per_7b")
    if dm.get("per_model"): gb7b = max(m.get("footprint_gb", 0) for m in dm["per_model"])
    real = "REAL" if agent.get("real_engine") else "mock"
    rows = [
        ("agents / GPU",                 am.get("agents_per_gpu"),  f"FAULT={am.get('fault')} exact={am.get('exact')} [{real}]"),
        ("fleet tok/W",                  am.get("fleet_tok_w"),     f"{am.get('thru_tok_s')}tok/s @ {am.get('avg_power_w')}W"),
        ("per-agent p99 (short/long ms)", f"{p99s}/{p99l}",         f"misroute={am.get('misroute_negctrl')}"),
        ("density GB / 7B",              gb7b,                      f"NF4 KL={dm.get('nf4_kl')} [{'REAL' if dens.get('real_engine') else 'mock'}]"),
        ("models co-resident",           dm.get("models_coresident"), f"live_total={dm.get('live_total_gb')}GB" if dm.get('live_total_gb') else ""),
    ]
    for k, v, note in rows:
        print(f"  {k:<32} {str(v):<14} {note}", flush=True)

    # ----- Tier-B ledger (present, not claimed) -----
    comp = next((r for r in results if r.get("regime") == "compute"), {})
    print("\n--- Tier-B ledger (present, NOT a Class-D headline) ---", flush=True)
    print(f"  FP8 compute (secondary):   {comp.get('metrics')}", flush=True)
    print(f"  cross-tenant batching:     SHIPPED, named next-milestone (socket executor)", flush=True)
    print(f"  KV-dedup / weight-share:   SHIPPED on `cipher accelerate` (vLLM) path", flush=True)
    print(f"  Koopman / classifier:      present-but-OFF, documented, never faked", flush=True)

    # ----- injection-state proof -----
    print("\n--- injection-state proof (each regime got the right conflicting state) ---", flush=True)
    for r in results:
        print(f"  {r.get('regime'):<8} saw: {r.get('injection_state_seen')}", flush=True)

    gate_ok = ok and all(r.get("metrics", {}).get("fault", 0) == 0 for r in results if r.get("regime") == "agent")
    out = {"results": results, "off_byte_identical": ok, "gate": "PASS" if gate_ok else "FAIL"}
    with open(os.path.join(HERE, "mock_report.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nreport -> mock_report.json   overall gate: {'PASS' if gate_ok else 'FAIL'}", flush=True)

if __name__ == "__main__":
    main()
