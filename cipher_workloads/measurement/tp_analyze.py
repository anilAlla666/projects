"""T4.2.4e two-prefill analyzer.

Reads:
  /tmp/tp_t1_prefill_latencies.json
  /tmp/tp_t2_prefill_latencies.json
  /tmp/cipher_tenant_tp_t1_progress
  /tmp/cipher_tenant_tp_t2_progress
  /tmp/tp_device.csv

Emits JSON summary on stdout.

Usage:
  python3 tp_analyze.py --label A --libcipher libcipher_rt.so
"""
import json, os, argparse, statistics, csv


def parse_progress_lines(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [ln.rstrip() for ln in f if ln.strip()]
    if not lines:
        return None
    end_line = next((ln for ln in reversed(lines) if " end " in ln), lines[-1])
    out = {}
    for token in end_line.split():
        if "=" in token:
            k, v = token.split("=", 1)
            try:
                out[k] = float(v) if "." in v else int(v)
            except ValueError:
                out[k] = v
    return out


def latency_stats(path):
    if not os.path.exists(path):
        return {"error": f"missing {path}"}
    with open(path) as f:
        data = json.load(f)
    lat = data.get("prefill_latency_ms", [])
    if len(lat) < 2:
        return {"n": len(lat), "error": "not enough samples"}
    s = sorted(lat)
    def pct(p):
        idx = max(0, min(len(s)-1, int(round(p*(len(s)-1)))))
        return s[idx]
    return {
        "n": len(lat),
        "mean_ms": round(statistics.mean(lat), 3),
        "median_ms": round(statistics.median(lat), 3),
        "p50_ms": round(pct(0.50), 3),
        "p95_ms": round(pct(0.95), 3),
        "p99_ms": round(pct(0.99), 3),
        "max_ms": round(max(lat), 3),
        "min_ms": round(min(lat), 3),
        "stdev_ms": round(statistics.pstdev(lat), 3),
        # Coefficient of variation — direct proxy for latency variance.
        "cv_pct": round(statistics.pstdev(lat) / statistics.mean(lat) * 100, 2),
    }


def device_stats(path):
    if not os.path.exists(path):
        return {"error": "no device.csv"}
    rows = []
    with open(path) as f:
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
        "max_power_w": round(max(r["power_w"] for r in rows), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--libcipher", default="(unknown)")
    args = ap.parse_args()

    t1_prog = parse_progress_lines("/tmp/cipher_tenant_tp_t1_progress")
    t2_prog = parse_progress_lines("/tmp/cipher_tenant_tp_t2_progress")
    t1_lat = latency_stats("/tmp/tp_t1_prefill_latencies.json")
    t2_lat = latency_stats("/tmp/tp_t2_prefill_latencies.json")
    dev = device_stats("/tmp/tp_device.csv")

    # Total throughput across both tenants (prefills/s aggregate).
    total_prefills_per_sec = None
    if t1_prog and t2_prog:
        t1_pps = t1_prog.get("prefills", 0) / max(1.0, float(str(t1_prog.get("t", "1")).rstrip("s")))
        t2_pps = t2_prog.get("prefills", 0) / max(1.0, float(str(t2_prog.get("t", "1")).rstrip("s")))
        total_prefills_per_sec = round(t1_pps + t2_pps, 2)

    print(json.dumps({
        "label": args.label,
        "libcipher": args.libcipher,
        "tenant_t1": {"progress": t1_prog, "latency": t1_lat},
        "tenant_t2": {"progress": t2_prog, "latency": t2_lat},
        "system_total_prefills_per_sec": total_prefills_per_sec,
        "device": dev,
    }, indent=2))


if __name__ == "__main__":
    main()
