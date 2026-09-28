"""Read pair{1..N}_{off,on}/{watts.csv,progress.txt} for one condition,
compute per-pair watts/tok_s/tok_w averaged over the decode window,
emit summary.json with mean ± stddev, 95% CI matched-pair Δ.

Window: trim first 5s (warmup) and last 2s of each watts.csv.
"""
import os, sys, json, glob, math, re

COND = sys.argv[1] if len(sys.argv) > 1 else "C1"
ROOT = f"/home/ubuntu/cipher-phase4-evidence/t4_3_envelope/{COND}"
N = int(os.environ.get("N", "3"))
DUR = int(os.environ.get("DUR", "60"))
TRIM_LEAD = 5  # seconds skipped (warmup)
TRIM_TAIL = 2

PROG_RX = re.compile(r"iters=(\d+).*tokens=(\d+).*tok/s=([\d.]+)")

def parse_progress(prog_path):
    """Return final (iters, tokens, total_secs, tok_s_running)."""
    last = None
    final_line = None
    with open(prog_path) as f:
        for line in f:
            if "tokens=" in line and "iters=" in line:
                last = line.strip()
                if line.startswith("ENV") and " end " in line:
                    final_line = line.strip()
    use = final_line or last
    if not use:
        return None
    # Parse total tokens + iters + elapsed.
    m_tok = re.search(r"tokens=(\d+)", use)
    m_it  = re.search(r"iters=(\d+)", use)
    m_t   = re.search(r"t=([\d.]+)s", use)
    if not (m_tok and m_it and m_t):
        return None
    return {"iters": int(m_it.group(1)),
            "tokens": int(m_tok.group(1)),
            "secs": float(m_t.group(1)),
            "tok_s": int(m_tok.group(1)) / max(float(m_t.group(1)), 1e-6)}

def parse_watts(csv_path):
    rows = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("ts_s"): continue
            parts = line.split(",")
            try:
                rows.append((int(parts[0]), float(parts[1]), int(parts[2])))
            except Exception:
                pass
    if not rows:
        return None
    # Window: trim lead + tail.
    last_ts = max(r[0] for r in rows)
    decode_end = min(last_ts, DUR)
    window = [r for r in rows if TRIM_LEAD <= r[0] <= decode_end - TRIM_TAIL]
    if not window:
        return None
    ws = [r[1] for r in window]
    cs = [r[2] for r in window]
    return {"watts_mean": sum(ws)/len(ws),
            "clock_mean": sum(cs)/len(cs),
            "n_samples": len(window)}

def mean(xs): return sum(xs)/len(xs) if xs else 0
def std(xs):
    if len(xs) < 2: return 0
    m = mean(xs); return math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))

result = {"condition": COND, "n_pairs": N, "duration_s": DUR,
          "per_pair": {"off": [], "on": []}}
for arm in ["off", "on"]:
    for p in range(1, N+1):
        d = f"{ROOT}/pair{p}_{arm}"
        wat = parse_watts(f"{d}/watts.csv")
        prog = parse_progress(f"{d}/progress.txt")
        if not wat or not prog:
            print(f"!! missing pair{p}_{arm} (wat={wat is not None} prog={prog is not None})")
            continue
        tw = prog["tok_s"] / wat["watts_mean"] if wat["watts_mean"] else 0
        result["per_pair"][arm].append({
            "pair": p, "watts": round(wat["watts_mean"], 3),
            "clock_mhz": round(wat["clock_mean"], 1),
            "tok_s": round(prog["tok_s"], 2),
            "tok_w": round(tw, 5),
            "tokens": prog["tokens"], "secs": round(prog["secs"], 2),
            "n_samples": wat["n_samples"],
        })

# Per-arm stats.
for arm in ["off", "on"]:
    pairs = result["per_pair"][arm]
    if not pairs: continue
    stats = {}
    for k in ["watts", "clock_mhz", "tok_s", "tok_w"]:
        vs = [p[k] for p in pairs]
        stats[k] = {"mean": round(mean(vs), 4),
                    "stddev": round(std(vs), 4),
                    "n": len(vs)}
    result.setdefault("summary", {})[arm] = stats

# Matched-pair Δ.
n = min(len(result["per_pair"]["off"]), len(result["per_pair"]["on"]))
if n >= 2:
    deltas = {}
    for k in ["watts", "tok_s", "tok_w"]:
        diffs = [result["per_pair"]["on"][i][k] - result["per_pair"]["off"][i][k]
                 for i in range(n)]
        off_means = [result["per_pair"]["off"][i][k] for i in range(n)]
        m_diff = mean(diffs); s_diff = std(diffs)
        m_off  = mean(off_means)
        pct = 100.0 * m_diff / m_off if m_off else 0
        # 95% CI from t-distribution at small n is conservative; use 1.96 stderr approx
        stderr_diff = s_diff / math.sqrt(n) if n > 0 else 0
        pct_stderr = 100.0 * stderr_diff / m_off if m_off else 0
        ci_lo = pct - 1.96 * pct_stderr
        ci_hi = pct + 1.96 * pct_stderr
        deltas[k] = {"mean_diff": round(m_diff, 4),
                     "stderr_diff": round(stderr_diff, 4),
                     "pct_delta": round(pct, 3),
                     "pct_stderr": round(pct_stderr, 3),
                     "pct_95lo": round(ci_lo, 3),
                     "pct_95hi": round(ci_hi, 3),
                     "n": n}
    result["delta_on_vs_off"] = deltas

out_path = f"{ROOT}/summary.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

# Print a one-liner for the campaign log.
if "delta_on_vs_off" in result:
    d = result["delta_on_vs_off"]
    off = result["summary"]["off"]
    on  = result["summary"]["on"]
    print(f"[{COND}] off: {off['watts']['mean']}W "
          f"{off['tok_s']['mean']}t/s {off['tok_w']['mean']}tok/W "
          f"clk={off['clock_mhz']['mean']:.0f}MHz | "
          f"on: {on['watts']['mean']}W {on['tok_s']['mean']}t/s "
          f"{on['tok_w']['mean']}tok/W clk={on['clock_mhz']['mean']:.0f}MHz | "
          f"Δtok/W={d['tok_w']['pct_delta']:+.2f}% (95% CI "
          f"{d['tok_w']['pct_95lo']:+.2f}..{d['tok_w']['pct_95hi']:+.2f})")
else:
    print(f"[{COND}] INSUFFICIENT DATA")
print(f"[{COND}] wrote {out_path}")
