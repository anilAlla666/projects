"""CIPHER CP 0.4 — Test A continuous-decode density sweep.

Sweeps N tenants over {4,8,16,24,32,48,64,96,128}, one shared Mistral-7B
process, per-tenant StaticCache + CUDA stream, 60 s timed continuous decode
per step. Re-uses the canonical Phase-0 density harness
(/workspace/stress2/density_harness.py) verbatim — this driver only adds the
sweep loop and the Story-A aggregation that CP 0.4 requires.

Harness reality (DIAGNOSTIC_REPORT_2026_05_13.md / HARNESS_LIMITATIONS.md §3):
the harness is eager-mode and launch-overhead-bound, not silicon-bound. The
curve produced here is a prototype-harness density curve and is reported
honestly as such, not as CIPHER's intrinsic silicon capacity.
"""
import sys, os, time, json, traceback

sys.path.insert(0, "/workspace/stress2")

EVID = "/home/ubuntu/cipher-fusion-evidence/cp_0_4"
os.makedirs(EVID, exist_ok=True)

from cipher_metrics import MetricsCollector
import density_harness as dh

N_GRID = [4, 8, 16, 24, 32, 48, 64, 96, 128]
DURATION_S = 60
PREFILL_LEN = 128
MAX_DECODE = 200


def make_collector(run_id):
    return MetricsCollector(run_id, gpu_id=0, sample_hz=5.0, out_dir=EVID)


def write_story(rows):
    with open(os.path.join(EVID, "story_a.json"), "w") as f:
        json.dump(dict(
            cp="0.4", test="A_continuous_decode",
            generated=time.strftime("%Y-%m-%d %H:%M:%S"),
            n_grid=N_GRID, duration_s=DURATION_S,
            prefill_len=PREFILL_LEN, max_decode=MAX_DECODE,
            model="Mistral-7B-v0.1 fp16 + Marlin INT4",
            harness="/workspace/stress2/density_harness.py",
            harness_note=("eager-mode, launch-overhead-bound prototype harness "
                          "per HARNESS_LIMITATIONS.md §3 — prototype-ceiling "
                          "density curve, not silicon-bound capacity"),
            rows=rows), f, indent=2)

    L = []
    L.append("# CIPHER CP 0.4 — Test A: continuous-decode density sweep (Story-A)")
    L.append("")
    L.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("")
    L.append("Model: Mistral-7B-v0.1 fp16 + Marlin INT4. Harness: "
             "`/workspace/stress2/density_harness.py` (canonical Phase-0 "
             "density harness, unmodified). Driver: `density_sweep_a.py`.")
    L.append("")
    L.append("**Harness reality:** this harness is eager-mode and "
             "launch-overhead-bound (DIAGNOSTIC_REPORT_2026_05_13.md, "
             "HARNESS_LIMITATIONS.md §3). The curve below is a "
             "prototype-harness density curve — honest as such, not a claim "
             "of CIPHER's silicon-bound capacity.")
    L.append("")
    L.append("## Story-A density curve")
    L.append("")
    L.append("| N | agg_tps | tps/tenant | MFU % (/989) | HFU % (/660) | "
             "TPW tok/J | $/M-tok | fairness | peak_mem GB | power W | "
             "clk MHz | util % | samples |")
    L.append("|" + "---|" * 13)
    for r in rows:
        if not r.get("ok"):
            L.append(f"| {r['N']} | **FAILED**: {r.get('error', '?')} |"
                     + " |" * 12)
            continue
        L.append(
            f"| {r['N']} | {r['agg_tps']:.2f} | {r['tps_per_tenant']:.3f} | "
            f"{r['mfu_pct']:.4f} | {r['hfu_pct']:.4f} | {r['tpw']:.4f} | "
            f"${r['dollar_per_M_tok']:.6f} | {r['fairness_ratio']:.3f} | "
            f"{r['peak_mem_mib'] / 1024:.2f} | {r['mean_power_w']:.0f} | "
            f"{r['mean_clock_mhz']:.0f} | {r['mean_util_pct']:.1f} | "
            f"{r['n_samples']} |")
    L.append("")
    L.append("`density` here is the aggregate-throughput curve `agg_tps` vs N "
             "(how densely the single H100 is packed with useful decode work) "
             "together with `tps/tenant` (per-tenant share as N grows).")
    L.append("")
    L.append("## Per-step receipts")
    L.append("")
    for r in rows:
        L.append(f"- N={r['N']}: `{r.get('receipt', '(none)')}`  "
                 f"(step wall {r.get('step_wall_s', 0):.0f}s)")
    L.append("")
    with open(os.path.join(EVID, "STORY_A.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    print("=" * 80)
    print("CIPHER CP 0.4 — Test A continuous-decode density sweep")
    print(f"N grid: {N_GRID}   duration={DURATION_S}s/step  "
          f"prefill={PREFILL_LEN} max_decode={MAX_DECODE}")
    print("=" * 80, flush=True)

    print("\n[setup] loading libcipher_rt...", flush=True)
    rt = dh.setup_rt()

    print("[setup] loading Mistral-7B...", flush=True)
    t0 = time.perf_counter()
    model, tok = dh.load_model_shared()
    print(f"[setup] model loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    import torch
    mem_before = torch.cuda.memory_allocated(0) / 1e9
    print(f"[setup] GPU mem after model load: {mem_before:.2f} GB", flush=True)

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
        run_id = f"a_n{N}_60s"
        print("\n" + "=" * 80)
        print(f"STEP {i + 1}/{len(N_GRID)} — N={N} tenants  ({run_id})")
        print("=" * 80, flush=True)
        step_t0 = time.perf_counter()
        try:
            r = dh.run_step_continuous(
                run_id, model, tok,
                n_tenants=N, duration_s=DURATION_S,
                collector_factory=make_collector,
                prefill_len=PREFILL_LEN, max_decode=MAX_DECODE)
            a = r["aggregate"]
            row = dict(
                N=N, run_id=run_id, ok=(a["total_tokens"] > 0),
                receipt=os.path.join(EVID, run_id + ".json"),
                total_tokens=a["total_tokens"], elapsed_s=a["elapsed_s"],
                agg_tps=a["agg_tps"], tps_per_tenant=round(a["agg_tps"] / N, 4),
                tps_min=a["tps_min"], tps_max=a["tps_max"],
                fairness_ratio=a["fairness_ratio"],
                achieved_TFLOPS=a["achieved_TFLOPS"],
                mfu_pct=a["mfu_pct"], hfu_pct=a["hfu_pct"], tpw=a["tpw"],
                dollar_per_M_tok=a["dollar_per_M_tok_power_only"],
                mean_power_w=a["mean_power_w"], peak_mem_mib=a["peak_mem_mib"],
                mean_util_pct=a["mean_util_pct"],
                mean_clock_mhz=a["mean_clock_mhz"], n_samples=a["n_samples"])
            print(f"  N={N}: agg_tps={a['agg_tps']:.2f}  "
                  f"tps/tenant={row['tps_per_tenant']:.3f}  "
                  f"MFU={a['mfu_pct']:.4f}%  HFU={a['hfu_pct']:.4f}%  "
                  f"TPW={a['tpw']:.4f}  fairness={a['fairness_ratio']:.3f}  "
                  f"mem={a['peak_mem_mib'] / 1024:.1f}GB", flush=True)
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
    print("Story-A:", os.path.join(EVID, "STORY_A.md"))
    return 0 if ok == len(N_GRID) else 1


if __name__ == "__main__":
    sys.exit(main())
