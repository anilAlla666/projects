"""Compare vanilla vs CIPHER bench_llm.py results.

Usage: compare_runs.py <vanilla_result.json> <cipher_result.json> <outdir>
Produces: comparison_table.md, tps_vs_batch.png, mfu_decode_vs_batch.png,
          tok_per_w_vs_batch.png, hbm_vs_batch.png, significance.json
"""
import json
import math
import random
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bootstrap_ci(a, b, n=2000, seed=7):
    rng = random.Random(seed)
    deltas = []
    for _ in range(n):
        sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
        sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        deltas.append(statistics.mean(sb) - statistics.mean(sa))
    deltas.sort()
    return deltas[int(0.025 * n)], deltas[int(0.975 * n)]


def cohens_d(a, b):
    if len(a) <= 1 or len(b) <= 1:
        return float("nan")
    pv = (statistics.pvariance(a) * (len(a) - 1)
          + statistics.pvariance(b) * (len(b) - 1)) / (len(a) + len(b) - 2)
    psd = math.sqrt(pv) if pv > 0 else 0
    if psd == 0:
        return float("nan")
    return (statistics.mean(b) - statistics.mean(a)) / psd


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    v_path, c_path, outdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    vr = json.load(open(v_path))
    cr = json.load(open(c_path))

    # Side-by-side per-batch table
    rows = []
    significance = {}
    for vpb, cpb in zip(vr["per_batch"], cr["per_batch"]):
        B = vpb["batch_size"]
        va = vpb["aggregate"]
        ca = cpb["aggregate"]
        vri = vpb["iterations_raw"]
        cri = cpb["iterations_raw"]
        sig_metrics = {}
        for metric in ["aggregate_tps", "decode_mfu_pct", "prefill_mfu_pct",
                       "mean_power_w", "energy_j", "tok_per_w", "tok_per_j",
                       "ttft_ms_p50", "ttft_ms_p99",
                       "tpot_ms_p50", "tpot_ms_p99",
                       "peak_hbm_mib"]:
            a = [r[metric] for r in vri
                 if isinstance(r.get(metric), (int, float))
                 and not math.isnan(r[metric])]
            b = [r[metric] for r in cri
                 if isinstance(r.get(metric), (int, float))
                 and not math.isnan(r[metric])]
            if not a or not b:
                continue
            mean_a = statistics.mean(a)
            mean_b = statistics.mean(b)
            delta = mean_b - mean_a
            pct = 100 * delta / mean_a if mean_a != 0 else float("nan")
            cd = cohens_d(a, b)
            lo, hi = bootstrap_ci(a, b)
            sig = (lo > 0) or (hi < 0)
            sig_metrics[metric] = {
                "vanilla_mean": mean_a, "cipher_mean": mean_b,
                "delta": delta, "delta_pct": pct,
                "cohens_d": cd, "ci95_low": lo, "ci95_high": hi,
                "significant": sig,
            }
        rows.append({"batch_size": B, "metrics": sig_metrics})
        significance[B] = sig_metrics

    # Markdown table
    md = ["# Vanilla vs CIPHER — side-by-side", ""]
    md += [f"Vanilla run_id: `{vr['run_id']}`", "",
           f"CIPHER  run_id: `{cr['run_id']}`", ""]
    md += ["Model: " + vr["args"]["model"],
           "Input length: " + vr["args"]["input_length_distribution"],
           "Output tokens: " + str(vr["args"]["output_tokens"]),
           "Iterations: " + str(vr["args"]["iterations"]),
           "Warmup iters: " + str(vr["args"]["warmup_iters"]), ""]
    md += ["## Headline (aggregate_tps, decode_mfu_pct, tok_per_w)"]
    md += ["", "| B | vanilla TPS | CIPHER TPS | Δ% | sig? | vanilla decMFU% | CIPHER decMFU% | Δ% | sig? | vanilla tok/W | CIPHER tok/W | Δ% | sig? |"]
    md += ["|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        B = row["batch_size"]
        m = row["metrics"]
        def cells(k):
            x = m.get(k, {})
            sigm = "✓" if x.get("significant") else "—"
            return [f"{x.get('vanilla_mean', float('nan')):.3f}",
                    f"{x.get('cipher_mean', float('nan')):.3f}",
                    f"{x.get('delta_pct', float('nan')):+.2f}%",
                    sigm]
        md.append(
            f"| {B} | " + " | ".join(cells("aggregate_tps") + cells("decode_mfu_pct") + cells("tok_per_w")) + " |"
        )
    md += ["", "## All metrics with CI"]
    md += ["", "| B | metric | vanilla mean | CIPHER mean | Δ% | Cohen's d | 95% CI of Δ | significant? |"]
    md += ["|---|---|---|---|---|---|---|---|"]
    for row in rows:
        B = row["batch_size"]
        for metric, s in row["metrics"].items():
            md.append(
                f"| {B} | {metric} | "
                f"{s['vanilla_mean']:.4f} | {s['cipher_mean']:.4f} | "
                f"{s['delta_pct']:+.2f}% | {s['cohens_d']:+.2f} | "
                f"[{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}] | "
                f"{'YES' if s['significant'] else 'no'} |"
            )
    (outdir / "comparison_table.md").write_text("\n".join(md) + "\n")
    (outdir / "significance.json").write_text(
        json.dumps(significance, indent=2))

    # Plots
    Bs = [r["batch_size"] for r in rows]

    def plot(metric_keys, ylabel, fname, title):
        plt.figure(figsize=(7, 5))
        v_means = [significance[B][metric_keys]["vanilla_mean"] for B in Bs]
        c_means = [significance[B][metric_keys]["cipher_mean"] for B in Bs]
        plt.plot(Bs, v_means, "o-", label="vanilla", color="tab:blue")
        plt.plot(Bs, c_means, "s--", label="CIPHER", color="tab:orange")
        plt.xlabel("Batch size")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xscale("log", base=2)
        plt.xticks(Bs, [str(b) for b in Bs])
        plt.grid(True, which="both", linestyle=":", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=140)
        plt.close()

    plot("aggregate_tps", "Aggregate TPS (tok/s)",
         "tps_vs_batch.png", "TPS vs batch size — vanilla vs CIPHER")
    plot("decode_mfu_pct", "Decode MFU (%)",
         "mfu_decode_vs_batch.png", "Decode MFU vs batch size — vanilla vs CIPHER")
    plot("tok_per_w", "Tokens per Watt (tok/s/W)",
         "tok_per_w_vs_batch.png", "Energy efficiency vs batch size — vanilla vs CIPHER")
    plot("peak_hbm_mib", "Peak HBM (MiB)",
         "hbm_vs_batch.png", "Peak HBM vs batch size — vanilla vs CIPHER")

    print(f"DONE  outdir: {outdir}")


if __name__ == "__main__":
    main()
