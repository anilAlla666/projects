# WL01 Multi-Tenant Aggregate — Phase A Pillar 3

**Date:** 2026-05-18. Substrate `a7ac8e97`. Workload WL01 = TinyLlama-1.1B,
B=1, greedy, 128-token decode, 5 prompts. Single H100, sequential arms.

**Verdict: the WL01 4-tenant operating point does NOT meet the 3.6× target,
and the CIPHER substrate contributes none of the lift it does show.**

## Measurement

Four arms. Correctness: per-step logit-KL(gold‖substrate); vanilla self-KL is
the sanity anchor, 1-tenant marlin→gold KL the honest-INT4-drift anchor.

| arm | agg tok/s | full-GPU W | **tok/W** | MFU% | KL mean | SMs used |
|---|---|---|---|---|---|---|
| 1-tenant vanilla | 72.2 | 147.8 | **0.488** | 0.020 | 0.000 | 132 (shared) |
| 1-tenant marlin | 73.1 | 145.2 | 0.503 | 0.020 | 0.085 | 132 |
| 4-tenant vanilla | 197.9 | 199.2 | **0.994** | 0.055 | 0.000 | 132 (shared) |
| 4-tenant all-on | 86.7 | 86.9 | **0.998** | 0.024 | 0.085 | 32 (4×8 green) |

KL sanity: vanilla self-KL = **0.00000** on all 5 prompts — harness valid.
KL drift: marlin/all-on KL_mean **0.071–0.101** (per-prompt), identical
single- and 4-tenant — partition isolation does not corrupt output. At the
KL ≤ 0.1 gate, 4/5 prompts pass (prompt 0 = 0.101, marginal). Teacher-forced
≥99% gate: 0/5 (top-1 argmax flips — honest INT4 drift, same as CP 5.6 P2).

## The decomposition (this is the load-bearing result)

Total lift of 4-tenant all-on over the 1-tenant vanilla baseline:
**0.998 / 0.488 = 2.04×.** It decomposes:

```
total 2.04x  =  concurrency factor 2.04x   x   substrate factor 1.00x
                (4t-vanilla / 1t-vanilla)      (4t-all-on / 4t-vanilla)
                = 0.994 / 0.488                = 0.998 / 0.994
```

- **Concurrency factor — 2.04×.** Running 4 processes instead of 1 doubles
  tok/W. This is a property of the H100 (fixed power overhead amortised over
  more work) — obtainable with plain vanilla processes, MPS, or MIG. **Not
  CIPHER.**
- **Substrate factor — 1.00×.** 4-tenant all-on tok/W (0.998) vs 4-tenant
  vanilla tok/W (0.994): identical within noise. **CIPHER's green
  partitioning + DVFS add zero tok/W at this operating point.**

And the substrate does not merely fail to help — it reaches the same tok/W a
*worse* way: 4-tenant all-on delivers **86.7 tok/s**, 4-tenant vanilla
**197.9 tok/s**. The substrate is **2.3× slower in absolute throughput**; its
DVFS clock-lock cuts power 2.3× (199 W → 87 W) and throughput 2.3× in
lockstep — a 1:1 trade, zero net efficiency. Plain vanilla concurrency
realises the same 2.0× tok/W at 2.3× the speed.

## Why — SM starvation

All-on binds each tenant to an 8-SM green partition: 4 tenants = **32 of 132
SMs**, 100 SMs idle. The operating point is SM-starved by design. Vanilla's
4 processes share all 132 SMs and run far faster. Filling the GPU via the
substrate's 8-SM partitions needs ~16 tenants (16×8=128) — but a 16-tenant
vanilla run would fill it too, so the substrate-vs-vanilla question stands at
every tenant count.

## Against the engineering target

- 4-tenant aggregate tok/W = **2.04×** the 1-tenant vanilla baseline
  (concurrency-inclusive) — **below 3.6×**.
- Substrate-attributable = **1.00×** — CIPHER meets none of the target here.
- This is the third independent measurement (CP 5.6 P2 single-tenant 1.54×;
  WL01 PASS 1; this) in which the substrate does not deliver a ≥3.6× tok/W
  lift on honest, correctness-gated output. The 3.6× is not currently a real
  driver-level operating point — single-tenant or multi-tenant.

## Phase B implication

The multi-tenant pillar's gap is the **whole** 3.6×: concurrency gives 2.0×
for free, the substrate currently adds 0×. Green partitioning today is an
*isolation* mechanism, not an efficiency lever; DVFS trades throughput for
power 1:1. Phase B has to make the substrate earn a real factor — e.g.
partitioning that raises aggregate MFU (pack tenants so idle SMs do work),
or DVFS that finds a genuinely super-linear tok/W point, not a 1:1 trade.

## Honest-disclosure note

CP 5.6 P2 already retracted the 3.617× single-tenant headline (loop
artifact; honest single-tenant = 1.54×). This multi-tenant measurement is a
*different* claim and lands at 2.04× concurrency-inclusive / 1.00×
substrate-attributable. Neither number re-validates the 3.617× sent to
investors. If anything investor-facing is updated, it should disclose the
retraction and the corrected operating points — not present a concurrency
multiplier as a CIPHER substrate result.

## Artefacts

`phase_a/WL01/mt{1,4}_{vanilla,marlin,allon}/` — `tenant*.json`, `power.csv`,
`aggregate.json` per config; `gold.json` + `gold_logits.pt`. Harness:
`pillar_driver.py`, `phase_a/run_multitenant.sh`, `analyze_multitenant.py`.

6- and 8-tenant scaling not yet run — the substrate-factor≈1.00× finding does
not depend on tenant count, so it is reported now rather than after the
scaling curve.
