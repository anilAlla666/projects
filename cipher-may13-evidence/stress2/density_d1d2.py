"""CIPHER density-sweep harness — D1 + D2 validation step.

D1: single-tenant 60s continuous decode (calibration)
D2: 4-tenant 60s continuous decode (multi-thread dry run)

STOP after D2. Reports pass/fail. Does NOT run the full sweep.
"""
import sys, os, time
sys.path.insert(0, "/workspace/stress2")

from cipher_metrics import MetricsCollector
import density_harness as dh


def make_collector(run_id):
    return MetricsCollector(run_id, gpu_id=0, sample_hz=5.0,
                             out_dir="/tmp/cipher_metrics")


def print_step_summary(label, receipt):
    a = receipt["aggregate"]
    print(f"\n[{label}] wall={receipt.get('wall_s', 0):.1f}s  "
          f"elapsed={a['elapsed_s']:.1f}s")
    print(f"  n_tenants        = {a['n_tenants']}")
    print(f"  total_tokens     = {a['total_tokens']}")
    print(f"  agg_tps          = {a['agg_tps']:.2f}")
    print(f"  per-tenant tps   = {a['tps_min']:.2f}  ...  {a['tps_max']:.2f}")
    print(f"  fairness_ratio   = {a['fairness_ratio']:.4f}")
    print(f"  achieved_TFLOPS  = {a['achieved_TFLOPS']:.4f}")
    print(f"  mfu_pct  (/989)  = {a['mfu_pct']:.4f}%   "
          "[industry-comparable]")
    print(f"  hfu_pct  (/660)  = {a['hfu_pct']:.4f}%   "
          "[this pod, 700W cap]")
    print(f"  tpw              = {a['tpw']:.4f}  tok/J")
    print(f"  $/M tok (power)  = ${a['dollar_per_M_tok_power_only']:.6f}")
    print(f"  mean_power_w     = {a['mean_power_w']:.1f}")
    print(f"  peak_mem_mib     = {a['peak_mem_mib']}  "
          f"({a['peak_mem_mib']/1024:.2f} GB)")
    print(f"  mean_util_pct    = {a['mean_util_pct']:.1f}%")
    print(f"  mean_clock_mhz   = {a['mean_clock_mhz']:.0f}")
    print(f"  n_samples        = {a['n_samples']}")


def main():
    print("=" * 80)
    print("CIPHER Density Sweep — D1+D2 validation")
    print("Model: Mistral-7B-v0.1 fp16 + Marlin INT4")
    print("=" * 80, flush=True)

    print("\n[setup] loading libcipher_rt...", flush=True)
    rt = dh.setup_rt()

    print("[setup] loading Mistral-7B...", flush=True)
    t0 = time.perf_counter()
    model, tok = dh.load_model_shared()
    print(f"[setup] model loaded in {time.perf_counter()-t0:.1f}s",
          flush=True)

    import torch
    mem_before = torch.cuda.memory_allocated(0) / 1e9
    print(f"[setup] GPU mem after model load: {mem_before:.2f} GB",
          flush=True)

    print("[setup] patching Marlin INT4 on all linears...", flush=True)
    t0 = time.perf_counter()
    n_c, n_s = dh.patch_model_marlin(rt, model)
    repack_s = time.perf_counter() - t0
    mem_after = torch.cuda.memory_allocated(0) / 1e9
    print(f"[setup] Marlin: {n_c} compressed, {n_s} skipped "
          f"in {repack_s:.1f}s", flush=True)
    print(f"[setup] GPU mem after Marlin compress + orig.weight free: "
          f"{mem_after:.2f} GB  (delta {mem_after - mem_before:+.2f} GB)",
          flush=True)
    print(f"[setup] GPU mem reserved: "
          f"{torch.cuda.memory_reserved(0)/1e9:.2f} GB", flush=True)

    # D1
    print("\n" + "=" * 80)
    print("D1 — single-tenant calibration (60s)")
    print("=" * 80, flush=True)
    r1 = dh.run_step_continuous(
        "d1_single_60s", model, tok,
        n_tenants=1, duration_s=60,
        collector_factory=make_collector,
        prefill_len=128, max_decode=200)
    print_step_summary("D1", r1)

    # D2
    print("\n" + "=" * 80)
    print("D2 — 4-tenant dry run (60s)")
    print("=" * 80, flush=True)
    r2 = dh.run_step_continuous(
        "d2_n4_60s", model, tok,
        n_tenants=4, duration_s=60,
        collector_factory=make_collector,
        prefill_len=128, max_decode=200)
    print_step_summary("D2", r2)

    # Pass/Fail
    print("\n" + "=" * 80)
    print("PASS/FAIL")
    print("=" * 80)
    a1, a2 = r1["aggregate"], r2["aggregate"]

    checks = []
    # D1
    checks.append(("D1: agg_tps in [60, 250]",
                   60 <= a1["agg_tps"] <= 250,
                   f"agg_tps={a1['agg_tps']:.2f}"))
    checks.append(("D1: peak_mem < 20 GB",
                   a1["peak_mem_mib"] < 20 * 1024,
                   f"{a1['peak_mem_mib']/1024:.2f} GB"))
    checks.append(("D1: n_samples >= 250",
                   a1["n_samples"] >= 250,
                   f"{a1['n_samples']}"))
    checks.append(("D1: total_tokens > 0",
                   a1["total_tokens"] > 0,
                   f"{a1['total_tokens']}"))
    # D2
    checks.append(("D2: agg_tps > 1.5x D1",
                   a2["agg_tps"] > 1.5 * a1["agg_tps"],
                   f"D2 {a2['agg_tps']:.2f}  vs  1.5x D1 = "
                   f"{1.5 * a1['agg_tps']:.2f}"))
    checks.append(("D2: fairness < 1.10",
                   a2["fairness_ratio"] < 1.10,
                   f"fairness={a2['fairness_ratio']:.4f}"))
    checks.append(("D2: peak_mem < 22 GB",
                   a2["peak_mem_mib"] < 22 * 1024,
                   f"{a2['peak_mem_mib']/1024:.2f} GB"))
    checks.append(("D2: all 4 tenants produced tokens",
                   all(t["n_tokens"] > 0 for t in a2["tenants"]),
                   f"tenants_with_tokens="
                   f"{sum(1 for t in a2['tenants'] if t['n_tokens']>0)}/4"))

    all_pass = True
    for desc, ok, detail in checks:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {desc}  ({detail})")
        if not ok:
            all_pass = False

    print("\n" + "=" * 80)
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 80)
    print("Receipts: /tmp/cipher_metrics/d1_single_60s.json")
    print("          /tmp/cipher_metrics/d2_n4_60s.json")
    print("\nSTOP — waiting for explicit go on the full sweep.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
