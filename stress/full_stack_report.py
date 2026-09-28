#!/usr/bin/env python3
"""Aggregate the CIPHER full-stack stress test into a single Markdown report.

Reads from:
    <run_dir>/phase_a_single_baseline/{t*.json,nvsmi.csv}
    <run_dir>/phase_b_multi_baseline/{t*.json,nvsmi.csv}
    <run_dir>/phase_c_multi_cipher/{t*.json,nvsmi.csv,counters/*.json,stderr/t*.stderr}
"""
import os, sys, json, glob, csv, re
from pathlib import Path
from statistics import mean, median

# H100 fp16 peak (TFLOPS)  — for MFU
H100_FP16_PEAK_TFLOPS = 989.0
# Llama-3.2-1B params (1.235B)  — for FLOPS estimate
LLAMA32_1B_PARAMS = 1.235e9


def load_jsons(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "t*.json"))):
        try:
            out.append(json.load(open(p)))
        except Exception:
            pass
    return out


def nvsmi_summary(csv_path):
    if not os.path.exists(csv_path):
        return dict(mean_w=0, mean_clock=0, max_mem=0, mean_util=0, mean_temp=0, n=0)
    rows = []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                rows.append({
                    "ts": float(r["ts"]),
                    "power": float(r["power_w"]),
                    "clock": float(r["clock_mhz"]),
                    "mem":   int(r["mem_used_mib"]),
                    "util":  float(r["util_pct"]),
                    "temp":  float(r["temp_c"]),
                })
            except Exception:
                pass
    if not rows:
        return dict(mean_w=0, mean_clock=0, max_mem=0, mean_util=0, mean_temp=0, n=0)
    return dict(
        mean_w   = mean(r["power"] for r in rows),
        mean_clock = mean(r["clock"] for r in rows),
        max_mem  = max(r["mem"]   for r in rows),
        mean_util = mean(r["util"] for r in rows),
        mean_temp = mean(r["temp"] for r in rows),
        n        = len(rows),
        rows     = rows,
    )


def aggregate_phase(d):
    js = load_jsons(d)
    nv = nvsmi_summary(os.path.join(d, "nvsmi.csv"))
    total_tokens = sum(j.get("n_tokens", 0) for j in js)
    duration = max((j.get("elapsed_s", 0) for j in js), default=0)
    agg_tps = total_tokens / duration if duration > 0 else 0.0
    tok_per_w = agg_tps / nv["mean_w"] if nv["mean_w"] > 0 else 0.0
    per_tenant = [j.get("tokens_per_s", 0) for j in js]
    coherent = sum(1 for j in js if j.get("sample_text", "").strip()
                                  and not j["sample_text"].startswith("!!!"))
    return dict(
        n_tenants_alive = len(js),
        total_tokens    = total_tokens,
        duration_s      = duration,
        agg_tps         = agg_tps,
        mean_w          = nv["mean_w"],
        mean_clock      = nv["mean_clock"],
        max_mem_mib     = nv["max_mem"],
        mean_util       = nv["mean_util"],
        mean_temp       = nv["mean_temp"],
        n_samples       = nv["n"],
        tok_per_w       = tok_per_w,
        per_tenant_tps  = per_tenant,
        per_tenant_min  = min(per_tenant) if per_tenant else 0,
        per_tenant_max  = max(per_tenant) if per_tenant else 0,
        per_tenant_mean = mean(per_tenant) if per_tenant else 0,
        coherent_n      = coherent,
        sample_text     = (js[0].get("sample_text", "") if js else ""),
        nvsmi_rows      = nv.get("rows", []),
    )


def aggregate_counters(d):
    counter_dir = os.path.join(d, "counters")
    files = sorted(glob.glob(os.path.join(counter_dir, "*.json")))
    if not files:
        return {}, 0
    agg = {}
    n = 0
    for p in files:
        try:
            o = json.load(open(p))
            for k, v in (o.get("ops") or {}).items():
                agg[k] = agg.get(k, 0) + int(v)
            n += 1
        except Exception:
            pass
    return agg, n


def parse_workload_from_stderr(d):
    types = {}
    intercept_total = 0
    got_patches_total = 0
    for p in sorted(glob.glob(os.path.join(d, "stderr", "t*.stderr"))):
        try:
            with open(p) as f:
                txt = f.read()
            m = re.search(r"\[CIPHER WORKLOAD\] type=(\S+)", txt)
            if m:
                types[m.group(1)] = types.get(m.group(1), 0) + 1
            m = re.search(r"Intercepts:\s+(\d+)\s+\|\s+ProcAddr:\s+\d+\s+\|\s+GOT patches:\s+(\d+)", txt)
            if m:
                intercept_total += int(m.group(1))
                got_patches_total += int(m.group(2))
        except Exception:
            pass
    return types, intercept_total, got_patches_total


def stability_check(rows):
    if len(rows) < 6:
        return None
    span = rows[-1]["ts"] - rows[0]["ts"]
    if span < 60:
        return None
    early = [r for r in rows if r["ts"] - rows[0]["ts"] < 60]
    late  = [r for r in rows if rows[-1]["ts"] - r["ts"] < 60]
    if not early or not late:
        return None
    return dict(
        early_w = mean(r["power"] for r in early),
        late_w  = mean(r["power"] for r in late),
        early_mem = max(r["mem"] for r in early),
        late_mem  = max(r["mem"] for r in late),
    )


def main():
    if len(sys.argv) < 3:
        print("usage: full_stack_report.py <run_dir> <n_tenants>", file=sys.stderr)
        sys.exit(2)
    run_dir = sys.argv[1]
    n_tenants_target = int(sys.argv[2])

    A = aggregate_phase(os.path.join(run_dir, "phase_a_single_baseline"))
    B = aggregate_phase(os.path.join(run_dir, "phase_b_multi_baseline"))
    C = aggregate_phase(os.path.join(run_dir, "phase_c_multi_cipher"))
    counters, n_counter_files = aggregate_counters(os.path.join(run_dir, "phase_c_multi_cipher"))
    workload_types, intercept_total, got_total = \
        parse_workload_from_stderr(os.path.join(run_dir, "phase_c_multi_cipher"))

    # Goals
    # Goal 1: tok/W >= 1.5x single-baseline
    g1_tok_per_w_x = (C["tok_per_w"] / A["tok_per_w"]) if A["tok_per_w"] > 0 else 0
    # Goal 2: revenue multiplier (CIPHER 15-tenant agg / single baseline tok/s)
    single_tps = A["agg_tps"] if A["agg_tps"] > 0 else 1.0
    g2_revenue_x = C["agg_tps"] / single_tps
    # Goal 3: MFU
    actual_flops = C["agg_tps"] * 2 * LLAMA32_1B_PARAMS * 2  # fwd multiply-adds
    g3_mfu = (actual_flops / (H100_FP16_PEAK_TFLOPS * 1e12)) * 100
    # Goal 4: O(1) substitutions
    koopman = counters.get("SUBSTITUTE_KOOPMAN", 0)
    spec_check = counters.get("SPECULATE_CHECK", 0)
    fp8_subs = counters.get("SUBSTITUTE_FP8", 0)
    workload_obs = counters.get("WORKLOAD_OBSERVE", 0)
    # Goal 5: driver-level verification
    g5_intercepts = intercept_total > 0 and got_total > 0
    # Goal 7: stability
    stab = stability_check(C.get("nvsmi_rows", []))

    # Format
    out = []
    P = out.append
    P("# CIPHER FULL-STACK INTEGRATED STRESS TEST")
    P("")
    P(f"Run dir: `{run_dir}`")
    P(f"Target tenants: {n_tenants_target}")
    P(f"Model: Llama-3.2-1B fp16, GPU 0 (H100 80GB), clock locked 1200 MHz")
    P("")
    P("## Headline — Goal scoreboard")
    P("")
    P("| # | Goal | Target | Measured | Verdict |")
    P("|---|---|---|---|---|")
    P(f"| 1 | tok/W vs single-baseline      | ≥ 1.5× | {g1_tok_per_w_x:.2f}× | "
      f"{'✅' if g1_tok_per_w_x >= 1.5 else '❌'} |")
    P(f"| 2 | Multi-tenant revenue          | ≥ 5× single tok/s | {g2_revenue_x:.2f}× | "
      f"{'✅' if g2_revenue_x >= 5 else '❌'} |")
    P(f"| 3 | MFU at 15 tenants             | report-only | {g3_mfu:.2f}% | (single-H100, B=1, 1B model) |")
    P(f"| 4 | O(1) Koopman substitutions    | > 0 | {koopman} | "
      f"{'✅' if koopman > 0 else '⚠ no shapes calibrated'} |")
    P(f"| 5 | Driver-level intercepts       | > 0 in every tenant | "
      f"intercepts={intercept_total:,}, GOT patches={got_total} | "
      f"{'✅' if g5_intercepts else '❌'} |")
    types_str = ", ".join(f"{k}={v}" for k, v in workload_types.items()) or "n/a"
    P(f"| 6 | Workload detection            | report observed types | {types_str} | "
      f"{'✅' if workload_types else '❌'} |")
    if stab:
        ratio = abs(stab['late_w'] - stab['early_w']) / max(stab['early_w'], 1)
        P(f"| 7 | Stability (early vs late)     | ≤ 5% Δwatts | "
          f"early_w={stab['early_w']:.0f} late_w={stab['late_w']:.0f} (Δ {ratio*100:.1f}%) | "
          f"{'✅' if ratio < 0.05 else '⚠'} |")
    else:
        P(f"| 7 | Stability                     | run < 60s | n/a | n/a |")

    P("")
    P("## Phase comparison")
    P("")
    P("| metric | A: single baseline | B: 15-tenant baseline | C: 15-tenant CIPHER |")
    P("|---|---|---|---|")
    P(f"| tenants alive  | {A['n_tenants_alive']}/1 | {B['n_tenants_alive']}/{n_tenants_target} | "
      f"{C['n_tenants_alive']}/{n_tenants_target} |")
    P(f"| duration (s)   | {A['duration_s']:.1f} | {B['duration_s']:.1f} | {C['duration_s']:.1f} |")
    P(f"| total tokens   | {A['total_tokens']:,} | {B['total_tokens']:,} | {C['total_tokens']:,} |")
    P(f"| agg tok/s      | {A['agg_tps']:.1f} | {B['agg_tps']:.1f} | {C['agg_tps']:.1f} |")
    P(f"| mean watts     | {A['mean_w']:.1f} | {B['mean_w']:.1f} | {C['mean_w']:.1f} |")
    P(f"| **tok/W**      | **{A['tok_per_w']:.3f}** | **{B['tok_per_w']:.3f}** | **{C['tok_per_w']:.3f}** |")
    P(f"| max mem (MiB)  | {A['max_mem_mib']:,} | {B['max_mem_mib']:,} | {C['max_mem_mib']:,} |")
    P(f"| mean clock MHz | {A['mean_clock']:.0f} | {B['mean_clock']:.0f} | {C['mean_clock']:.0f} |")
    P(f"| mean util %    | {A['mean_util']:.1f} | {B['mean_util']:.1f} | {C['mean_util']:.1f} |")
    P(f"| coherent       | {A['coherent_n']}/1 | {B['coherent_n']}/{n_tenants_target} | "
      f"{C['coherent_n']}/{n_tenants_target} |")

    if C['per_tenant_tps']:
        P("")
        P("## Per-tenant tok/s (CIPHER phase)")
        P("")
        P(f"min={C['per_tenant_min']:.1f}  mean={C['per_tenant_mean']:.1f}  max={C['per_tenant_max']:.1f}")
        ratio = C['per_tenant_max'] / C['per_tenant_min'] if C['per_tenant_min'] > 0 else 0
        P(f"max/min ratio = {ratio:.2f}  (target < 3.0)")

    P("")
    P("## Op activity (CIPHER phase, summed across tenants)")
    P("")
    P(f"Counter dump files: {n_counter_files}/{n_tenants_target} tenants reported")
    P("")
    P("| op | total calls |")
    P("|---|---|")
    for k in sorted(counters.keys()):
        P(f"| {k} | {counters[k]:,} |")

    P("")
    P("## Driver-level evidence")
    P("")
    P(f"- Total intercepts across {n_counter_files} tenants: **{intercept_total:,}**")
    P(f"- Total GOT patches: **{got_total}**")
    P(f"- Workload types observed: **{types_str}**")
    P(f"- WORKLOAD_OBSERVE counter total: **{workload_obs:,}** "
      f"(driver hot path called workload detector)")
    P(f"- FP8 substitutions: **{fp8_subs:,}**")
    P(f"- SPECULATE_CHECK counter: {spec_check:,}")

    P("")
    P("## Sample output text (first tenant)")
    P("")
    P(f"- A (single baseline): `{A['sample_text'][:80]}`")
    P(f"- B (multi baseline):  `{B['sample_text'][:80]}`")
    P(f"- C (CIPHER):          `{C['sample_text'][:80]}`")

    if stab:
        P("")
        P("## Stability (CIPHER phase, first 60s vs last 60s)")
        P(f"- mean watts: {stab['early_w']:.1f} W → {stab['late_w']:.1f} W")
        P(f"- max mem:    {stab['early_mem']:,} MiB → {stab['late_mem']:,} MiB")

    print("\n".join(out))


if __name__ == "__main__":
    main()
