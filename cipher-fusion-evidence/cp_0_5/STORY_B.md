# CIPHER CP 0.5 — Test B: agentic-burst density sweep (Story-B)

Generated: 2026-05-15 13:01:06

Model: Mistral-7B-v0.1 fp16 + Marlin INT4. Harness: `/workspace/stress2/density_harness.py` (canonical Phase-0 density harness, unmodified). Driver: `density_sweep_b.py`. Each tenant: repeat(prefill=128 + decode 50 tokens) then sleep 5.0s; 90s/step.

**Burst is the framing that holds** (HARNESS_LIMITATIONS.md §3): tenants are mostly idle and pack into the idle headroom. `mean_idle_fraction` is the density signal — while it stays high there is room for more tenants; the N where `compute_duty` saturates is the headroom limit. `effective_concurrent = N × compute_duty` is the tenant-equivalents of continuous compute the single H100 sustains.

**FAIRNESS caveat** (§6): libcipher_rt FAIRNESS SHM caps at 64 tenants; at N=96/128/192 its observer is partial. `fairness_ratio` below is the parent-level max(tps)/min(tps) from MetricsCollector — valid at every N.

## Story-B density curve

| N | agg_tps | total_tokens | bursts | mean_idle_frac | compute_duty | eff_concurrent | s/burst | MFU % (/989) | HFU % (/660) | TPW tok/J | $/M-tok | fairness | peak_mem GB | power W | samples |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 7.45 | 800 | 16 | 0.0931 | 0.9069 | 7.25 | 48.799 | 0.0106 | 0.0159 | 0.0564 | $0.492852 | 1.123 | 39.24 | 130 | 547 |
| 16 | 8.21 | 800 | 16 | 0.0513 | 0.9487 | 15.18 | 92.469 | 0.0118 | 0.0177 | 0.0636 | $0.436604 | 1.009 | 41.00 | 129 | 489 |
| 32 | 8.39 | 1600 | 32 | 0.0263 | 0.9737 | 31.16 | 185.613 | 0.0121 | 0.0182 | 0.0648 | $0.428617 | 1.006 | 42.83 | 129 | 955 |
| 64 | 8.50 | 3200 | 64 | 0.0133 | 0.9867 | 63.15 | 371.333 | 0.0123 | 0.0184 | 0.0652 | $0.425927 | 1.005 | 47.19 | 130 | 1885 |
| 96 | 8.55 | 4800 | 96 | 0.0089 | 0.9911 | 95.14 | 556.655 | 0.0123 | 0.0185 | 0.0659 | $0.421337 | 1.006 | 52.60 | 130 | 2813 |
| 128 | 8.52 | 6400 | 128 | 0.0067 | 0.9933 | 127.15 | 746.062 | 0.0123 | 0.0184 | 0.0657 | $0.422529 | 1.004 | 58.78 | 129 | 3760 |
| 192 | 8.55 | 9600 | 192 | 0.0045 | 0.9955 | 191.14 | 1117.576 | 0.0124 | 0.0185 | 0.0660 | $0.421194 | 1.002 | 69.52 | 130 | 5620 |

## Per-step receipts

- N=8: `/home/ubuntu/cipher-fusion-evidence/cp_0_5/b_n8_burst.json`  (step wall 109s)
- N=16: `/home/ubuntu/cipher-fusion-evidence/cp_0_5/b_n16_burst.json`  (step wall 98s)
- N=32: `/home/ubuntu/cipher-fusion-evidence/cp_0_5/b_n32_burst.json`  (step wall 191s)
- N=64: `/home/ubuntu/cipher-fusion-evidence/cp_0_5/b_n64_burst.json`  (step wall 377s)
- N=96: `/home/ubuntu/cipher-fusion-evidence/cp_0_5/b_n96_burst.json`  (step wall 562s)
- N=128: `/home/ubuntu/cipher-fusion-evidence/cp_0_5/b_n128_burst.json`  (step wall 752s)
- N=192: `/home/ubuntu/cipher-fusion-evidence/cp_0_5/b_n192_burst.json`  (step wall 1124s)

