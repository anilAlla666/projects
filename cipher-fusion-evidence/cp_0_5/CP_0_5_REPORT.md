# CP 0.5 — Test B agentic-burst density sweep — REPORT

**Date:** 2026-05-15. **Status:** deliverable produced; CP 0.5 has no pass/fail
gate — it requires Story-B numbers (MFU/HFU/TPW/density) per N step. All seven
N steps completed and produced receipts.

## What was missing, what was built

Audit (phase_audit/phase0/audit.md, CP 0.5): the burst harness functions
`run_step_burst` / `tenant_burst_worker` existed in the canonical Phase-0
density harness, but **no burst-sweep driver and zero burst receipts** existed
— CP 0.5 had never been run once. Built: `density_sweep_b.py` — the sweep loop
+ Story-B aggregation only; the harness itself
(`/workspace/stress2/density_harness.py`) was run unmodified.

## Artifacts on disk

| Artifact | Path |
|---|---|
| Driver | `cipher-fusion-evidence/cp_0_5/density_sweep_b.py` |
| Story-B table (md) | `cipher-fusion-evidence/cp_0_5/STORY_B.md` |
| Story-B data (json) | `cipher-fusion-evidence/cp_0_5/story_b.json` |
| Run log | `cipher-fusion-evidence/cp_0_5/sweep_b.log` |
| Per-step receipts (7) | `cp_0_5/b_n{8,16,32,64,96,128,192}_burst.json` |

Config: Mistral-7B-v0.1 fp16 + Marlin INT4. Each tenant repeats
(prefill=128 + decode 50 tokens) then sleeps 5.0 s, for 90 s/step.
N grid {8,16,32,64,96,128,192}. 7/7 steps OK. Sweep wall 3213 s.

## Story-B density curve

| N | agg_tps | bursts | idle_frac | compute_duty | eff_concurrent | s/burst | MFU % | HFU % | TPW | $/M-tok | fairness | mem GB | power W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 7.45 | 16 | 0.0931 | 0.9069 | 7.25 | 48.8 | 0.0106 | 0.0159 | 0.0564 | $0.4929 | 1.123 | 39.2 | 130 |
| 16 | 8.21 | 16 | 0.0513 | 0.9487 | 15.18 | 92.5 | 0.0118 | 0.0177 | 0.0636 | $0.4366 | 1.009 | 41.0 | 129 |
| 32 | 8.39 | 32 | 0.0263 | 0.9737 | 31.16 | 185.6 | 0.0121 | 0.0182 | 0.0648 | $0.4286 | 1.006 | 42.8 | 129 |
| 64 | 8.50 | 64 | 0.0133 | 0.9867 | 63.15 | 371.3 | 0.0123 | 0.0184 | 0.0652 | $0.4259 | 1.005 | 47.2 | 130 |
| 96 | 8.55 | 96 | 0.0089 | 0.9911 | 95.14 | 556.7 | 0.0123 | 0.0185 | 0.0659 | $0.4213 | 1.006 | 52.6 | 130 |
| 128 | 8.52 | 128 | 0.0067 | 0.9933 | 127.15 | 746.1 | 0.0123 | 0.0184 | 0.0657 | $0.4225 | 1.004 | 58.8 | 129 |
| 192 | 8.55 | 192 | 0.0045 | 0.9955 | 191.14 | 1117.6 | 0.0124 | 0.0185 | 0.0660 | $0.4212 | 1.002 | 69.5 | 130 |

## Honest finding — the Marlin-lock saturation

The burst framing predicted (HARNESS_LIMITATIONS.md §3) that idle tenants pack
into idle headroom, so `mean_idle_fraction` stays high and density scales.
**The data refutes that on this harness.**

- `mean_idle_fraction` **collapses monotonically**: 0.093 (N=8) → 0.026 (N=32)
  → 0.0045 (N=192). `compute_duty` rises to 0.9955 — the idle headroom is
  essentially gone by N=192, and is already thin (> 0.97) by N=32.
- `s/burst` rises **almost perfectly linearly with N**: 48.8 s (N=8) →
  1117.6 s (N=192) — i.e. ~5.8–6.1 s of burst-wall per tenant, constant slope.
  A tenant's burst does not get more expensive; it spends linearly longer
  *waiting in a queue*.
- `agg_tps` is **flat at ~8.5 tok/s** across the entire N=8…192 range — the
  identical plateau Test A (CP 0.4) hit. Adding tenants moves no more tokens.
- Therefore `effective_concurrent ≈ N` is **an artifact, not a result**: duty
  approaches 1 because every tenant is blocked on the shared Marlin queue, not
  because the H100 is absorbing 192 tenants' worth of concurrent compute.

Root cause is the harness's Marlin-workspace serialization
(HARNESS_LIMITATIONS.md §1): `cipher_weight_compress_marlin_gemm` uses one
module-scope workspace, so the harness funnels **all** tenant Marlin GEMMs
through a single `_MARLIN_LOCK` + shared `_MARLIN_STREAM`. Every tenant's
compute is one serial GPU queue. Burst does not escape the prototype ceiling —
it hits the same ~8.5 tok/s wall as continuous decode, by the same mechanism.

Memory was never the constraint (39→69.5 GB, no OOM, consistent with
HARNESS_LIMITATIONS.md §4). Fairness stayed tight (1.002–1.123) — the serial
queue is at least shared evenly. The libcipher_rt FAIRNESS SHM caps at 64
tenants (§6); `fairness_ratio` here is the parent-level max/min tps from
MetricsCollector and is valid at every N.

## Bottom line for CP 0.5

CP 0.5 is delivered: the burst sweep ran end-to-end, all 7 N-steps produced
receipts, Story-B is on disk with MFU/HFU/TPW/density per step. The honest
headline: **on this eager-mode prototype harness the burst workload saturates
at the same ~8.5 tok/s aggregate as the continuous workload**, because the
shared Marlin lock serializes all tenant compute. Peak MFU is 0.0124%. This is
a true characterisation of the harness ceiling, reported as such — not a
silicon-capacity claim. The launch/serialization ceiling is the quantity the
kernel-module track is built to remove.
