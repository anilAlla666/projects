#!/usr/bin/env python3
"""
straggler_aggregate.py — cross-rank straggler analyzer for CIPHER.

Reads the per-rank JSONL logs produced by Op-Phase3 Straggler when
CIPHER_STRAGGLER=observe|active is set, computes per-rank mean
ns_per_byte, and identifies any rank whose mean exceeds 1.5x the
global mean.

Usage:
  python3 tools/straggler_aggregate.py /tmp/cipher_straggler_rank_*.jsonl
"""
import glob
import json
import sys
from collections import defaultdict

OUTLIER_FACTOR = 1.5


def main():
    paths = []
    for arg in sys.argv[1:]:
        paths.extend(glob.glob(arg))
    if not paths:
        print("usage: straggler_aggregate.py <rank_jsonl_glob>...", file=sys.stderr)
        sys.exit(2)

    per_rank_npb = defaultdict(list)   # rank -> list of ns_per_byte
    per_rank_world = {}
    line_counts = defaultdict(int)
    for p in paths:
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rank = ev.get("rank", -1)
                    npb  = ev.get("ns_per_byte")
                    if npb is None:
                        continue
                    per_rank_npb[rank].append(float(npb))
                    per_rank_world[rank] = ev.get("world_size", 0)
                    line_counts[p] += 1
        except OSError as e:
            print(f"[WARN] could not read {p}: {e}", file=sys.stderr)

    if not per_rank_npb:
        print("[ERR] no usable telemetry rows found", file=sys.stderr)
        sys.exit(1)

    rank_means = {r: (sum(v) / len(v)) for r, v in per_rank_npb.items()}
    global_mean = sum(rank_means.values()) / len(rank_means)

    print(f"== CIPHER straggler aggregate ==")
    print(f"files:           {len(paths)}")
    print(f"ranks observed:  {sorted(rank_means.keys())}")
    print(f"world_size:      {next(iter(per_rank_world.values()), 'unknown')}")
    print(f"global mean ns/byte:  {global_mean:.4f}")
    print(f"outlier threshold:    > {OUTLIER_FACTOR:.1f}x global mean")
    print()
    print(f"{'rank':>5} | {'samples':>7} | {'mean ns/byte':>12} | "
          f"{'ratio':>6} | verdict")
    print("-" * 60)
    outliers = []
    for r in sorted(rank_means.keys()):
        m = rank_means[r]
        ratio = m / global_mean if global_mean > 0 else 0.0
        verdict = "STRAGGLER" if ratio > OUTLIER_FACTOR else "ok"
        if verdict == "STRAGGLER":
            outliers.append(r)
        print(f"{r:>5} | {len(per_rank_npb[r]):>7} | {m:>12.4f} | "
              f"{ratio:>5.2f}x | {verdict}")
    print()
    if outliers:
        print(f"VERDICT: persistent straggler rank(s): {outliers}")
        sys.exit(1)
    else:
        print("VERDICT: no persistent straggler detected")
        sys.exit(0)


if __name__ == "__main__":
    main()
