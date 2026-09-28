# R-I1 SHIELD (noisy-neighbor p99 isolation) — CLOSE

**Date:** 2026-05-29. **CLOSED — lever VALIDATED, measurement-only, NO substrate built, anchor
UNCHANGED.** bf16. Deployment integration folded to V.1 (rationale below).

## Outcome

R-I1's goal — bound a latency-sensitive tenant's p99 under a saturating noisy neighbor on one
H100 — is **achievable**, via **CUDA MPS + the existing CP54 green-ctx partition** (validated in
`R_I1_MPS_VERDICT.md`):

- victim p99 **0.130 ms** under a saturating aggressor with a disjoint green-ctx partition **under
  MPS** ≈ the **0.114 ms** 8-SM partition floor (C2), recovered from **1.959 ms (40×)** without MPS
  and **1.148 ms (22×)** with MPS but no partition. Aggressor kept **81%** of its work (productive
  isolation, not the D.8 throttle's ~1% kill-switch).
- The lever is the **combination**: MPS provides cross-process concurrency, green-ctx CU masks
  provide spatial confinement. Neither alone isolates. **No new wiring** — the built CP54 green-ctx
  composes with MPS as-is.

The two prior negatives are both explained and superseded: D.8 throttle-band (a contention-
frequency reducer, p99 collision-bound) and bare green-ctx (no cross-process concurrency without
MPS) each failed; **MPS + green-ctx is the composition that works.**

## Honest scope (bounded-subset, NOT 100-way)

There are only **15 × 8-SM groups** on this H100. This does **not** give per-tenant isolation to
100 agents (≥85 must share; MPS thread-% can't rescue it — 100 clients ≈ 1.3 SMs each, below the
8-SM green-ctx floor). What is validated: **a bounded subset (≤ ~14) of designated latency-sensitive
tenants can hold dedicated groups with real p99 isolation; the bulk share.** This reconciles with
D.8 §4 (at ~2% duty only ~2 agents are active at once) — Goal-1's "each agent thinks it owns the
GPU" is met by **hardware self-arbitration at realistic duty (D.8) + dedicated-group protection for
the latency-critical few (this result)**, not by 100 disjoint partitions.

## Why deployment integration → V.1 (not designed now)

The integration is an **admission-control / packing policy** (who is band ≥1; how to allocate 15
groups when more than ~14 want protection) — and its **load-bearing parameter is the real band-≥1
population and the live contention shape**, which are **unknown until the 100-agent soak**.
Designing the policy now would be guessing at the inputs. It is therefore folded into V.1 scoping
(below), where the soak measures the actual inputs and proves the integration rather than assuming
it.

## Anchors — UNCHANGED (measurement only)

cipher_rt_phase4 `ed130e7` / `01d4effb` / tag `d7-rh1-close`; kmod canonical `02fc2d1` (0.6.6),
loaded `0.7.0` (inert); bridge `5a3db034`. No rotation. KL moot (pure isolation — partition changes
only which SMs run, not output; no substitution path engaged).

## V.1 SCOPING CARRY (pre-registered isolation sub-test — PROVE, don't assume)

The V.1 100-agent capstone soak MUST include the R-I1 in-situ isolation sub-test:
- Run the serving GPU **under CUDA MPS**.
- Bind **≥1 latency-band tenant** to a **dedicated green-ctx group**; leave the bulk on the shared
  remainder.
- **MEASURE** the dedicated tenant's p99 under the bursty 100-agent mix vs the shared bulk.
- **PASS = dedicated-tenant p99 near its partition floor while the bulk shares.**
This converts "lever validated in a 2-tenant microbench" into "isolation demonstrated in situ at
100-agent scale" — the real Goal-1 isolation evidence.

**R-I1 CLOSED.** D.8 FAIRNESS+SHIELD track complete: R-D5 closed-as-moot (decode overhead-bound),
R-I1 closed lever-validated (MPS+green-ctx, bounded subset, integration→V.1). No anchor rotation
across the whole D.8/R-I1 arc (the throttle-band substrate built remains inert/additive/default-OFF).
