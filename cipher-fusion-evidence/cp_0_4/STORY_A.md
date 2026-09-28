# CIPHER CP 0.4 — Test A: continuous-decode density sweep (Story-A)

Generated: 2026-05-15 12:05:09

Model: Mistral-7B-v0.1 fp16 + Marlin INT4. Harness: `/workspace/stress2/density_harness.py` (canonical Phase-0 density harness, unmodified). Driver: `density_sweep_a.py`.

**Harness reality:** this harness is eager-mode and launch-overhead-bound (DIAGNOSTIC_REPORT_2026_05_13.md, HARNESS_LIMITATIONS.md §3). The curve below is a prototype-harness density curve — honest as such, not a claim of CIPHER's silicon-bound capacity.

## Story-A density curve

| N | agg_tps | tps/tenant | MFU % (/989) | HFU % (/660) | TPW tok/J | $/M-tok | fairness | peak_mem GB | power W | clk MHz | util % | samples |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 12.76 | 3.190 | 0.0185 | 0.0277 | 0.0971 | $0.285976 | 1.026 | 38.66 | 131 | 1980 | 11.8 | 378 |
| 8 | 9.34 | 1.167 | 0.0135 | 0.0202 | 0.0725 | $0.383168 | 1.092 | 39.76 | 128 | 1980 | 9.0 | 327 |
| 16 | 8.86 | 0.554 | 0.0127 | 0.0190 | 0.0673 | $0.412923 | 1.027 | 41.95 | 130 | 1980 | 8.5 | 357 |
| 24 | 8.91 | 0.371 | 0.0128 | 0.0191 | 0.0678 | $0.409605 | 1.027 | 43.12 | 130 | 1980 | 8.5 | 389 |
| 32 | 8.84 | 0.276 | 0.0127 | 0.0191 | 0.0679 | $0.409002 | 1.014 | 44.44 | 129 | 1980 | 8.6 | 415 |
| 48 | 8.93 | 0.186 | 0.0127 | 0.0191 | 0.0682 | $0.407330 | 1.027 | 47.70 | 129 | 1980 | 8.6 | 483 |
| 64 | 8.89 | 0.139 | 0.0128 | 0.0192 | 0.0675 | $0.411497 | 1.019 | 51.12 | 131 | 1980 | 8.6 | 539 |
| 96 | 8.90 | 0.093 | 0.0127 | 0.0191 | 0.0673 | $0.412810 | 1.030 | 57.55 | 131 | 1980 | 8.5 | 647 |
| 128 | 8.82 | 0.069 | 0.0126 | 0.0189 | 0.0668 | $0.416100 | 1.034 | 64.42 | 131 | 1980 | 8.4 | 794 |

`density` here is the aggregate-throughput curve `agg_tps` vs N (how densely the single H100 is packed with useful decode work) together with `tps/tenant` (per-tenant share as N grows).

## Per-step receipts

- N=4: `/home/ubuntu/cipher-fusion-evidence/cp_0_4/a_n4_60s.json`  (step wall 76s)
- N=8: `/home/ubuntu/cipher-fusion-evidence/cp_0_4/a_n8_60s.json`  (step wall 65s)
- N=16: `/home/ubuntu/cipher-fusion-evidence/cp_0_4/a_n16_60s.json`  (step wall 71s)
- N=24: `/home/ubuntu/cipher-fusion-evidence/cp_0_4/a_n24_60s.json`  (step wall 78s)
- N=32: `/home/ubuntu/cipher-fusion-evidence/cp_0_4/a_n32_60s.json`  (step wall 83s)
- N=48: `/home/ubuntu/cipher-fusion-evidence/cp_0_4/a_n48_60s.json`  (step wall 97s)
- N=64: `/home/ubuntu/cipher-fusion-evidence/cp_0_4/a_n64_60s.json`  (step wall 108s)
- N=96: `/home/ubuntu/cipher-fusion-evidence/cp_0_4/a_n96_60s.json`  (step wall 129s)
- N=128: `/home/ubuntu/cipher-fusion-evidence/cp_0_4/a_n128_60s.json`  (step wall 159s)

