"""T4.2.4c noisy-neighbor analyzer.

Reads:
  /tmp/nn_victim_latencies.json    (from victim driver)
  /tmp/cipher_tenant_nn_bomb_progress
  /tmp/cipher_tenant_nn_victim_progress
  /tmp/nn_device.csv

Emits a JSON summary on stdout:
  {
    "label": <A|B>,
    "libcipher": <basename>,
    "victim": {tok_total, decode_iters, mean_inter_token_ms, p50, p95, p99, max},
    "bomb":   {prefills, tokens_pseudo, iters_per_s},
    "device": {mean_watts, mean_sm_util_pct, mean_sm_clock_mhz}
  }

Usage:
  python3 nn_analyze.py --label A --libcipher libcipher_rt.so
"""
import json, os, sys, argparse, statistics, csv


def parse_progress_lines(path):
    """run_for_duration progress file; last line has end totals."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [ln.rstrip() for ln in f if ln.strip()]
    if not lines:
        return None
    end_line = next((ln for ln in reversed(lines) if " end " in ln), lines[-1])
    return end_line


def parse_progress_totals(end_line):
    """Extract iters, units, tokens, elapsed from end-line."""
    out = {}
    for tok in end_line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k] = float(v) if "." in v else int(v)
            except ValueError:
                out[k] = v
    return out


def victim_latency_stats(latjson_path):
    with open(latjson_path) as f:
        data = json.load(f)
    rel_ms = data.get("token_arrival_rel_ms", [])
    if len(rel_ms) < 2:
        return {"n_tokens": len(rel_ms), "error": "not enough tokens"}
    inter = [rel_ms[i] - rel_ms[i - 1] for i in range(1, len(rel_ms))]
    inter_sorted = sorted(inter)

    def pct(p):
        idx = max(0, min(len(inter_sorted) - 1, int(round(p * (len(inter_sorted) - 1)))))
        return inter_sorted[idx]

    return {
        "n_tokens": len(rel_ms),
        "n_inter_token": len(inter),
        "mean_inter_token_ms": round(statistics.mean(inter), 3),
        "median_inter_token_ms": round(statistics.median(inter), 3),
        "p50_inter_token_ms": round(pct(0.50), 3),
        "p95_inter_token_ms": round(pct(0.95), 3),
        "p99_inter_token_ms": round(pct(0.99), 3),
        "max_inter_token_ms": round(max(inter), 3),
        "min_inter_token_ms": round(min(inter), 3),
        "stdev_inter_token_ms": round(statistics.pstdev(inter), 3),
    }


def device_stats(csv_path):
    if not os.path.exists(csv_path):
        return {"error": "no device.csv"}
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({
                    "ts": int(row["ts_s"]),
                    "power_w": float(row["power_w"]),
                    "sm_util_pct": float(row["sm_util_pct"]),
                    "sm_clock_mhz": float(row["sm_clock_mhz"]),
                })
            except (ValueError, KeyError):
                continue
    if not rows:
        return {"error": "no rows parsed"}
    return {
        "samples": len(rows),
        "mean_power_w": round(statistics.mean(r["power_w"] for r in rows), 1),
        "mean_sm_util_pct": round(statistics.mean(r["sm_util_pct"] for r in rows), 1),
        "mean_sm_clock_mhz": round(statistics.mean(r["sm_clock_mhz"] for r in rows), 0),
        "max_power_w": round(max(r["power_w"] for r in rows), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--libcipher", default="(unknown)")
    args = ap.parse_args()

    victim_end = parse_progress_lines("/tmp/cipher_tenant_nn_victim_progress")
    bomb_end = parse_progress_lines("/tmp/cipher_tenant_nn_bomb_progress")

    out = {
        "label": args.label,
        "libcipher": args.libcipher,
        "victim_progress": parse_progress_totals(victim_end) if victim_end else None,
        "bomb_progress": parse_progress_totals(bomb_end) if bomb_end else None,
        "victim_latency": victim_latency_stats("/tmp/nn_victim_latencies.json"),
        "device": device_stats("/tmp/nn_device.csv"),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
