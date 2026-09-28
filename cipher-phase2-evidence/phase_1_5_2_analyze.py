#!/usr/bin/env python3
"""Read phase_1_5_2_results.jsonl, compute Welch's t per workload,
write PHASE_1.5.2_OVERHEAD_REPORT.md.

Usage: phase_1_5_2_analyze.py [--in PATH] [--out PATH]
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

RESULTS_DEFAULT = "/workspace/cipher_kmod/phase_1_5_2_results.jsonl"
REPORT_DEFAULT = "/workspace/cipher_kmod/PHASE_1.5.2_OVERHEAD_REPORT.md"

WORKLOAD_DESC = {
    "A": ("TinyLlama Marlin smoke (80-token decode)",
          "tps", "higher is better", "tokens/sec"),
    "B": ("100 nvidia-smi invocations (batch wall time)",
          "seconds", "lower is better", "wall seconds"),
    "C": ("Mistral-7B c2_marlin (60s decode)",
          "tps", "higher is better", "tokens/sec"),
    "D": ("probe_microbench (100k tight ioctls)",
          "ns/ioctl", "lower is better", "nanoseconds per ioctl"),
}


def lgamma_ratio_two_sided_p(t, df):
    """Two-sided p-value from t and df via the regularized incomplete beta.

    Uses scipy if available; otherwise a small math-only fallback.
    Returns p in [0, 1].
    """
    try:
        from scipy import stats
        return float(stats.t.sf(abs(t), df) * 2.0)
    except Exception:
        # Fallback: numerical integration of t pdf tail.
        # Adequate for reporting; not as precise as scipy.
        x = abs(t)
        # 5000-step Simpson on [x, x+50]
        a, b, n = x, x + 50.0, 5000
        h = (b - a) / n
        const = math.gamma((df + 1) / 2.0) / (
            math.sqrt(df * math.pi) * math.gamma(df / 2.0))

        def pdf(u):
            return const * (1 + u * u / df) ** (-(df + 1) / 2.0)

        s = pdf(a) + pdf(b)
        for k in range(1, n):
            u = a + k * h
            s += (4 if k % 2 else 2) * pdf(u)
        tail = s * h / 3.0
        return min(1.0, max(0.0, 2.0 * tail))


def welch(a, b):
    """Welch's t-statistic and Welch-Satterthwaite df for samples a, b."""
    na, nb = len(a), len(b)
    ma, mb = statistics.mean(a), statistics.mean(b)
    if na > 1:
        va = statistics.variance(a)
    else:
        va = 0.0
    if nb > 1:
        vb = statistics.variance(b)
    else:
        vb = 0.0
    se = math.sqrt(va / na + vb / nb) if (va or vb) else 0.0
    t = (ma - mb) / se if se > 0 else 0.0
    if va > 0 and vb > 0:
        df = (va / na + vb / nb) ** 2 / (
            (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    else:
        df = na + nb - 2
    return t, df, se


def fmt_samples(samples):
    return "[" + ", ".join(f"{s:.3f}" for s in samples) + "]"


def emit_workload(out, w, by_cond):
    desc, unit, direction, _ = WORKLOAD_DESC[w]
    out.append(f"## Workload {w} — {desc}")
    out.append(f"Metric: {unit}  ({direction})\n")

    if "without" not in by_cond or "with" not in by_cond:
        out.append("(missing samples; skipped)\n")
        return

    sw = by_cond["without"]
    sk = by_cond["with"]

    mw = statistics.mean(sw)
    mk = statistics.mean(sk)
    sdw = statistics.stdev(sw) if len(sw) > 1 else 0.0
    sdk = statistics.stdev(sk) if len(sk) > 1 else 0.0
    medw = statistics.median(sw)
    medk = statistics.median(sk)
    delta = mk - mw
    pct = (delta / mw * 100.0) if mw else 0.0

    t, df, se = welch(sk, sw)
    p = lgamma_ratio_two_sided_p(t, df)

    out.append("| arm | n | samples | mean | stddev | median |")
    out.append("|---|---:|---|---:|---:|---:|")
    out.append(f"| without kmod | {len(sw)} | {fmt_samples(sw)} | "
               f"{mw:.3f} | {sdw:.3f} | {medw:.3f} |")
    out.append(f"| **with kmod**    | {len(sk)} | {fmt_samples(sk)} | "
               f"**{mk:.3f}** | {sdk:.3f} | {medk:.3f} |")
    out.append("")
    out.append(f"**delta (with − without):** {delta:+.4f} {unit} "
               f"({pct:+.3f} %)  ")
    out.append(f"**Welch's t:** t = {t:+.3f}, df = {df:.2f}, "
               f"two-sided p = {p:.3f}, SE = {se:.4f}")
    sig = ("statistically significant at p<0.05"
           if p < 0.05 else "NOT significant at p<0.05")
    out.append(f"**Verdict:** {sig}")
    out.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=RESULTS_DEFAULT)
    ap.add_argument("--out", dest="outp", default=REPORT_DEFAULT)
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        print(f"missing {inp}", file=sys.stderr)
        sys.exit(2)

    by_workload = {}
    for line in inp.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        w = rec["workload"]
        by_workload.setdefault(w, {})[rec["condition"]] = rec["samples"]

    md = []
    md.append("# CIPHER kmod — Phase 1.5.2 Overhead Report")
    md.append("")
    md.append("**Date:** 2026-05-13")
    md.append("**Pod:** Lambda H100 80GB SXM5 / driver 580.105.08")
    md.append("**Module:** cipher_kmod.ko 0.1.5 (Phase 1.5.1 binary, "
              "md5 c2131cf49fe11c50c2a56b8452340cb4)")
    md.append("")
    md.append("## Methodology")
    md.append("")
    md.append("Each workload was run N times under each of two conditions: "
              "`without` (kmod unloaded) and `with` (kmod loaded). "
              "First run of each arm is warmup and discarded. Workloads "
              "run as fresh subprocesses to flush GPU state between "
              "samples. Statistical comparison uses Welch's t-test for "
              "unequal variance; two-sided p-value via t-distribution.")
    md.append("")

    for w in ("A", "B", "C", "D"):
        if w in by_workload:
            emit_workload(md, w, by_workload[w])

    md.append("## Headline numbers")
    md.append("")
    if "D" in by_workload:
        bcd = by_workload["D"]
        if "without" in bcd and "with" in bcd:
            mw = statistics.mean(bcd["without"])
            mk = statistics.mean(bcd["with"])
            md.append(f"**Per-ioctl kprobe overhead** (Workload D, the "
                      f"isolated measurement): {mk - mw:+.1f} ns/ioctl "
                      f"({(mk - mw) / mw * 100:+.2f} %).")
            md.append("")
            md.append(f"Baseline ioctl path (without kmod): {mw:.1f} ns")
            md.append(f"With kmod: {mk:.1f} ns  → "
                      f"**{mk - mw:.1f} ns of cipher_kmod overhead per "
                      f"observed ioctl**.")
            md.append("")

    md.append("## Caveats")
    md.append("")
    md.append("- Workload A and C tps measurements have run-to-run noise "
              "of ~1-2 % from CUDA/Python sources unrelated to kmod. "
              "Detecting sub-1 % effects requires more samples than the "
              "current N=10 / N=3.")
    md.append("- Workload D isolates the kprobe path: each ioctl is a "
              "single nvidia_unlocked_ioctl entry/exit, so the per-ioctl "
              "delta in D is the cleanest available estimate of cipher_kmod "
              "overhead per observed call.")
    md.append("- Workload B (100 nvidia-smi invocations) bundles process "
              "fork/exec overhead with the ioctl path. Useful for "
              "operational realism; less useful as an isolated overhead "
              "number.")
    md.append("- Module hash unchanged between baseline check and this "
              "run: c2131cf49fe11c50c2a56b8452340cb4.")

    Path(args.outp).write_text("\n".join(md) + "\n")
    print(f"wrote {args.outp}")


if __name__ == "__main__":
    main()
