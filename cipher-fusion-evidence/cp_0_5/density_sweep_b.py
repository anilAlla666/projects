"""CIPHER CP 0.5 — Test B agentic-burst density sweep.

Sweeps N tenants over {8,16,32,64,96,128,192}, one shared Mistral-7B process,
per-tenant StaticCache + CUDA stream. Each tenant repeats (prefill + decode
burst_tokens) then sleeps sleep_s — the agentic tool-use pattern. Re-uses the
canonical Phase-0 density harness (/workspace/stress2/density_harness.py)
verbatim; this driver adds the burst-sweep loop and the Story-B aggregation
that CP 0.5 requires.

Burst is the framing that HOLDS (HARNESS_LIMITATIONS.md §3): tenants are mostly
idle, so they pack into the idle headroom and density × idle_fraction scaling
is valid. Duration is 90 s/step (longer than Test A's 60 s) so each tenant
completes several full burst cycles even under high-N compute contention.

FAIRNESS caveat (HARNESS_LIMITATIONS.md §6): libcipher_rt's FAIRNESS SHM caps
at MAX_TENANTS=64; at N=96/128/192 its observer is partial. The fairness_ratio
reported here is computed at the parent (harness) level from MetricsCollector
per-tenant tps — that metric is valid at every N; the rt observer is only
supplementary.
"""
import sys, os, time, json, traceback

sys.path.insert(0, "/workspace/stress2")

EVID = "/home/ubuntu/cipher-fusion-evidence/cp_0_5"
os.makedirs(EVID, exist_ok=True)

from cipher_metrics import MetricsCollector
import density_harness as dh

N_GRID = [8, 16, 32, 64, 96, 128, 192]
DURATION_S = 90
PREFILL_LEN = 128
BURST_TOKENS = 50
SLEEP_S = 5.0


def make_collector(run_id):
    return MetricsCollector(run_id, gpu_id=0, sample_hz=5.0, out_dir=EVID)


def burst_stats(agg):
    """Story-B per-step roll-up from a receipt aggregate block."""
    tn = agg.get("tenants", [])
    idle = [t.get("idle_fraction", 0.0) for t in tn]
    comp = sum(t.get("compute_s", 0.0) for t in tn)
    bursts = sum(t.get("n_bursts", 0) for t in tn)
    mean_idle = sum(idle) / len(idle) if idle else 0.0
    return dict(
        total_bursts=bursts,
        mean_idle_fraction=round(mean_idle, 4),
        compute_duty=round(1.0 - mean_idle, 4),
        mean_compute_s_per_burst=round(comp / bursts, 4) if bursts else 0.0,
        effective_concurrent=round(len(tn) * (1.0 - mean_idle), 3))


def write_story(rows):
    with open(os.path.join(EVID, "story_b.json"), "w") as f:
        json.dump(dict(
            cp="0.5", test="B_agentic_burst",
            generated=time.strftime("%Y-%m-%d %H:%M:%S"),
            n_grid=N_GRID, duration_s=DURATION_S, prefill_len=PREFILL_LEN,
            burst_tokens=BURST_TOKENS, sleep_s=SLEEP_S,
            model="Mistral-7B-v0.1 fp16 + Marlin INT4",
            harness="/workspace/stress2/density_harness.py",
            harness_note=("burst framing per HARNESS_LIMITATIONS.md §3 — the "
                          "framing that holds; density scales with idle "
                          "headroom. FAIRNESS SHM caps at 64 (§6): "
                          "fairness_ratio computed at parent level, valid at "
                          "all N."),
            rows=rows), f, indent=2)

    L = []
    L.append("# CIPHER CP 0.5 — Test B: agentic-burst density sweep (Story-B)")
    L.append("")
    L.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("")
    L.append("Model: Mistral-7B-v0.1 fp16 + Marlin INT4. Harness: "
             "`/workspace/stress2/density_harness.py` (canonical Phase-0 "
             "density harness, unmodified). Driver: `density_sweep_b.py`. "
             f"Each tenant: repeat(prefill={PREFILL_LEN} + decode "
             f"{BURST_TOKENS} tokens) then sleep {SLEEP_S}s; "
             f"{DURATION_S}s/step.")
    L.append("")
    L.append("**Burst is the framing that holds** (HARNESS_LIMITATIONS.md §3): "
             "tenants are mostly idle and pack into the idle headroom. "
             "`mean_idle_fraction` is the density signal — while it stays "
             "high there is room for more tenants; the N where "
             "`compute_duty` saturates is the headroom limit. "
             "`effective_concurrent = N × compute_duty` is the tenant-"
             "equivalents of continuous compute the single H100 sustains.")
    L.append("")
    L.append("**FAIRNESS caveat** (§6): libcipher_rt FAIRNESS SHM caps at 64 "
             "tenants; at N=96/128/192 its observer is partial. "
             "`fairness_ratio` below is the parent-level "
             "max(tps)/min(tps) from MetricsCollector — valid at every N.")
    L.append("")
    L.append("## Story-B density curve")
    L.append("")
    L.append("| N | agg_tps | total_tokens | bursts | mean_idle_frac | "
             "compute_duty | eff_concurrent | s/burst | MFU % (/989) | "
             "HFU % (/660) | TPW tok/J | $/M-tok | fairness | peak_mem GB | "
             "power W | samples |")
    L.append("|" + "---|" * 16)
    for r in rows:
        if not r.get("ok"):
            L.append(f"| {r['N']} | **FAILED**: {r.get('error', '?')} |"
                     + " |" * 15)
            continue
        L.append(
            f"| {r['N']} | {r['agg_tps']:.2f} | {r['total_tokens']} | "
            f"{r['total_bursts']} | {r['mean_idle_fraction']:.4f} | "
            f"{r['compute_duty']:.4f} | {r['effective_concurrent']:.2f} | "
            f"{r['mean_compute_s_per_burst']:.3f} | {r['mfu_pct']:.4f} | "
            f"{r['hfu_pct']:.4f} | {r['tpw']:.4f} | "
            f"${r['dollar_per_M_tok']:.6f} | {r['fairness_ratio']:.3f} | "
            f"{r['peak_mem_mib'] / 1024:.2f} | {r['mean_power_w']:.0f} | "
            f"{r['n_samples']} |")
    L.append("")
    L.append("## Per-step receipts")
    L.append("")
    for r in rows:
        L.append(f"- N={r['N']}: `{r.get('receipt', '(none)')}`  "
                 f"(step wall {r.get('step_wall_s', 0):.0f}s)")
    L.append("")
    with open(os.path.join(EVID, "STORY_B.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    print("=" * 80)
    print("CIPHER CP 0.5 — Test B agentic-burst density sweep")
    print(f"N grid: {N_GRID}   duration={DURATION_S}s/step  "
          f"prefill={PREFILL_LEN} burst_tokens={BURST_TOKENS} "
          f"sleep={SLEEP_S}s")
    print("=" * 80, flush=True)

    print("\n[setup] loading libcipher_rt...", flush=True)
    rt = dh.setup_rt()

    print("[setup] loading Mistral-7B...", flush=True)
    t0 = time.perf_counter()
    model, tok = dh.load_model_shared()
    print(f"[setup] model loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    import torch
    print(f"[setup] GPU mem after model load: "
          f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB", flush=True)

    print("[setup] patching Marlin INT4 on all linears...", flush=True)
    t0 = time.perf_counter()
    n_c, n_s = dh.patch_model_marlin(rt, model)
    print(f"[setup] Marlin: {n_c} compressed, {n_s} skipped "
          f"in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[setup] GPU mem after Marlin: "
          f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB", flush=True)

    rows = []
    sweep_t0 = time.perf_counter()
    for i, N in enumerate(N_GRID):
        run_id = f"b_n{N}_burst"
        print("\n" + "=" * 80)
        print(f"STEP {i + 1}/{len(N_GRID)} — N={N} tenants  ({run_id})")
        print("=" * 80, flush=True)
        step_t0 = time.perf_counter()
        try:
            r = dh.run_step_burst(
                run_id, model, tok,
                n_tenants=N, duration_s=DURATION_S,
                collector_factory=make_collector,
                prefill_len=PREFILL_LEN, burst_tokens=BURST_TOKENS,
                sleep_s=SLEEP_S)
            a = r["aggregate"]
            bs = burst_stats(a)
            row = dict(
                N=N, run_id=run_id, ok=(a["total_tokens"] > 0),
                receipt=os.path.join(EVID, run_id + ".json"),
                total_tokens=a["total_tokens"], elapsed_s=a["elapsed_s"],
                agg_tps=a["agg_tps"], fairness_ratio=a["fairness_ratio"],
                achieved_TFLOPS=a["achieved_TFLOPS"],
                mfu_pct=a["mfu_pct"], hfu_pct=a["hfu_pct"], tpw=a["tpw"],
                dollar_per_M_tok=a["dollar_per_M_tok_power_only"],
                mean_power_w=a["mean_power_w"], peak_mem_mib=a["peak_mem_mib"],
                mean_util_pct=a["mean_util_pct"],
                mean_clock_mhz=a["mean_clock_mhz"], n_samples=a["n_samples"],
                **bs)
            print(f"  N={N}: agg_tps={a['agg_tps']:.2f}  bursts="
                  f"{bs['total_bursts']}  idle_frac="
                  f"{bs['mean_idle_fraction']:.4f}  duty="
                  f"{bs['compute_duty']:.4f}  eff_concurrent="
                  f"{bs['effective_concurrent']:.2f}  MFU="
                  f"{a['mfu_pct']:.4f}%  TPW={a['tpw']:.4f}  mem="
                  f"{a['peak_mem_mib'] / 1024:.1f}GB", flush=True)
        except Exception as e:
            traceback.print_exc()
            row = dict(N=N, run_id=run_id, ok=False,
                       error=f"{type(e).__name__}: {e}")
            print(f"  N={N}: STEP FAILED — {e}", flush=True)
        row["step_wall_s"] = round(time.perf_counter() - step_t0, 1)
        rows.append(row)
        write_story(rows)
        if i == 0 and not row.get("ok"):
            print("[sweep] first step failed — aborting sweep", flush=True)
            break

    print(f"\n[sweep] total {time.perf_counter() - sweep_t0:.0f}s", flush=True)
    write_story(rows)
    ok = sum(1 for r in rows if r.get("ok"))
    print(f"[sweep] {ok}/{len(N_GRID)} steps OK")
    print("Story-B:", os.path.join(EVID, "STORY_B.md"))
    return 0 if ok == len(N_GRID) else 1


if __name__ == "__main__":
    sys.exit(main())
