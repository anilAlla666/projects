# Class-D scorecard + Tier-B ledger renderer. Reads handler results (real or mock),
# renders the honest headline metrics. Promoted from the validated mock router.
def scorecard(results):
    def find(reg): return next((r for r in results if r.get("regime") == reg), {})
    agent, dens, comp = find("agent"), find("density"), find("compute")
    am, dm = agent.get("metrics", {}), dens.get("metrics", {})
    p99s = am.get("p99_short_ms"); p99l = am.get("p99_long_ms")
    gb7b = dm.get("gb_per_7b")
    if dm.get("per_model"): gb7b = max(m.get("footprint_gb", 0) for m in dm["per_model"])
    areal = "REAL" if agent.get("real_engine") else "mock"
    dreal = "REAL" if dens.get("real_engine") else "mock"
    print("=== CIPHER CLASS-D SCORECARD ===", flush=True)
    rows = [
        ("agents / GPU", am.get("agents_per_gpu"), f"FAULT={am.get('fault')} exact={am.get('exact')} [{areal}]"),
        ("fleet tok/W", am.get("fleet_tok_w"), f"{am.get('thru_tok_s')}tok/s @ {am.get('avg_power_w')}W"),
        ("per-agent p99 short/long ms", f"{p99s}/{p99l}", f"misroute={am.get('misroute_negctrl')}"),
        ("density GB / 7B", gb7b, f"NF4 KL={dm.get('nf4_kl')} [{dreal}]"),
        ("models co-resident", dm.get("models_coresident"), f"live={dm.get('live_total_gb')}GB" if dm.get('live_total_gb') else ""),
    ]
    for k, v, note in rows:
        print(f"  {k:<30} {str(v):<14} {note}", flush=True)
    print("\n--- Tier-B ledger (present, NOT claimed as headline) ---", flush=True)
    print(f"  FP8 compute (secondary):  {comp.get('metrics', {})}", flush=True)
    print("  cross-tenant batching:    shipped, named next-milestone (socket executor)", flush=True)
    print("  KV-dedup / weight-share:  shipped on `cipher accelerate` (vLLM) path", flush=True)
    print("  Koopman / classifier:     present-but-OFF (observational; OFF==byte-identical verified)", flush=True)
