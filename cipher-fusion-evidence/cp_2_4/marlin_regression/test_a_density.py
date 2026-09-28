"""CP 2.4 — Marlin per-stream registry regression, test A (end-to-end).

N-tenant Mistral-7B continuous decode, one shared model, per-tenant
StaticCache + CUDA stream. Marlin routing comes from the v2 libcipher_rt.so
activated via LD_PRELOAD + CUDA_INJECTION64_PATH (see run_test_a.sh) — there
is NO Python _MARLIN_LOCK and NO MarlinLinear monkey-patch. Each tenant's
cuBLAS GEMMs are intercepted by the .symver shim and routed to the per-stream
Marlin registry built in CP 2.4.

Compares aggregate tok/s vs N against the CP 0.4 baseline (~8.5 tok/s flat
plateau for N>=16). Reuses the canonical density harness for model load and
the plain decode worker; it does NOT call setup_rt()/patch_model_marlin().
"""
import sys, os, time, json, traceback, ctypes

sys.path.insert(0, "/workspace/stress2")
EVID = "/home/ubuntu/cipher-fusion-evidence/cp_2_4/marlin_regression"
os.makedirs(EVID, exist_ok=True)

from cipher_metrics import MetricsCollector
import density_harness as dh

N_GRID = [1, 4, 16, 64]
DURATION_S = 30
PREFILL_LEN = 128
MAX_DECODE = 200


def make_collector(run_id):
    return MetricsCollector(run_id, gpu_id=0, sample_hz=5.0, out_dir=EVID)


def main():
    print("=" * 78)
    print("CP 2.4 test A — N-tenant Mistral-7B decode through the v2 Marlin path")
    print(f"N grid: {N_GRID}   duration={DURATION_S}s/step")
    print("LD_PRELOAD =", os.environ.get("LD_PRELOAD", "(unset)"))
    print("CUDA_INJECTION64_PATH =", os.environ.get("CUDA_INJECTION64_PATH", "(unset)"))
    print("CIPHER_MARLIN =", os.environ.get("CIPHER_MARLIN", "(unset)"))
    print("=" * 78, flush=True)

    print("\n[setup] loading Mistral-7B (plain — no MarlinLinear patch)...", flush=True)
    t0 = time.perf_counter()
    model, tok = dh.load_model_shared()
    print(f"[setup] model loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    rows = []
    for i, N in enumerate(N_GRID):
        run_id = f"testA_n{N}_{DURATION_S}s"
        print("\n" + "=" * 78)
        print(f"STEP {i+1}/{len(N_GRID)} — N={N} tenants  ({run_id})")
        print("=" * 78, flush=True)
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
                agg_tps=a["agg_tps"], tps_per_tenant=round(a["agg_tps"]/N, 4),
                mfu_pct=a["mfu_pct"], mean_power_w=a["mean_power_w"],
                peak_mem_gb=round(a["peak_mem_mib"]/1024, 2),
                total_tokens=a["total_tokens"], n_samples=a["n_samples"])
            print(f"  N={N}: agg_tps={a['agg_tps']:.2f}  "
                  f"tps/tenant={row['tps_per_tenant']:.3f}  "
                  f"MFU={a['mfu_pct']:.4f}%  mem={row['peak_mem_gb']:.1f}GB", flush=True)
        except Exception as e:
            traceback.print_exc()
            row = dict(N=N, run_id=run_id, ok=False,
                       error=f"{type(e).__name__}: {e}")
            print(f"  N={N}: STEP FAILED — {e}", flush=True)
        row["step_wall_s"] = round(time.perf_counter()-step_t0, 1)
        rows.append(row)
        with open(os.path.join(EVID, "test_a_result.json"), "w") as f:
            json.dump(dict(cp="2.4", test="A_end_to_end_density",
                           generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                           n_grid=N_GRID, duration_s=DURATION_S,
                           model="Mistral-7B-v0.1 fp16, v2 Marlin path",
                           cp04_baseline_agg_tps={"4": 12.76, "16": 8.86,
                                                  "64": 8.89, "128": 8.82},
                           rows=rows), f, indent=2)

    print("\n" + "=" * 78)
    print("CP 2.4 test A — aggregate tok/s vs N  (CP 0.4 baseline: N4=12.76, "
          "N16=8.86, N64=8.89)")
    for r in rows:
        if r.get("ok"):
            print(f"  N={r['N']:<4} agg_tps={r['agg_tps']:8.2f}  "
                  f"tps/tenant={r['tps_per_tenant']:.3f}  MFU={r['mfu_pct']:.4f}%")
        else:
            print(f"  N={r['N']:<4} FAILED: {r.get('error')}")
    print("result -> test_a_result.json", flush=True)
    return 0 if all(r.get("ok") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
