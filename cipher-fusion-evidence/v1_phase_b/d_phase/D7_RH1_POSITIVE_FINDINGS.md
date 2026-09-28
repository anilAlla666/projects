# D.7 R-H1 positive gate (bf16) — co-residence isolation DEMONSTRATED; full regression remains

**Date:** 2026-05-29. **NOT closed — anchor NOT rotated** (`9fe23143`). Positive correctness
demonstrated on the clean co-residence test; the Mem #16 full no-regression (the close
prerequisite) is NOT yet run. bf16 (real serving dtype) per the locked directive.

## Positive correctness — DEMONSTRATED (clean co-residence, no churn)

4 distinct families bf16 co-resident in one process, each bound to its `model_uuid` via
`cipher_rt_marlin_engine_bind_model`, d10 staging (LT_ROUTE off — Marlin works via the
existing cublasGemmEx interception):

| family | marlin_sub | KL=0 vs solo (exact greedy match) |
|---|---|---|
| Mistral-7B | 3600 | **6/6 ✓** |
| Qwen2-7B   | 1792 | **6/6 ✓** |
| TinyLlama  | 1776 | ran (Marlin engaged); clean solo ref not captured this run |
| phi-2      | 0 | Marlin **correctly declines** (shapes fail N%64/K%128 gate) → vanilla, sane |

- **No crash; ≥3 Marlin-engaged families co-resident; per-family routing isolated** (Mistral &
  Qwen2 match their solo refs exactly → no cross-family kit contamination). The
  `(model_id, w_ptr)` re-key isolation is confirmed (and the mis-route BLOCK was independently
  proven, `D7_KEYING_PASS`).
- **Marlin bf16 substitution is correct per-family in isolation:** Qwen2 bf16 vanilla
  `[2714,300,11,1817,31871,553,264,…]` vs Marlin `[…,553,2155,…]` — 6-token match then a benign
  near-tie greedy flip (Marlin INT4 rounding; argmax-stable). NOT corruption.

## The earlier "FAIL" was a harness load/free-churn confound (NOT the re-key, NOT co-residence)

The first harness (Phase A loaded+**freed** each family solo, then Phase B co-resident) produced
Qwen2 all-zeros + a co-resident illegal-memory-access crash. Root cause: **Marlin's persistent
`g_weights` cache holds a freed model's kits (dangling device pointers) across model free/reload**
— Phase A's frees polluted Phase B. The clean test (load all once, no free/reload) is correct.
**This is NOT Goal-1's production pattern** (one model per tenant process, loaded once), so it is
a robustness note (Marlin cache GC-on-free), not a V.1 blocker. Recorded, not forced past.

## Honest residue (before a rigorous close)

1. Clean solo refs captured for only Mistral/Qwen2; TinyLlama/phi-2 refs not re-run clean.
   Tighten: separate-process solo refs for all ≥3 Marlin-engaged families, 16-token compare.
2. Mis-route BLOCK proven separately (`D7_KEYING_PASS`), not re-run in this co-resident process.
3. **FULL no-regression NOT run** (Mem #16 HARD GATE): W7–W11 microbenches + 30-min N=128 soak
   + every-prior-model KL = ZERO degradation vs `8b5e928`/`9fe23143`. **D.7 cannot close without it.**

## Disposition (STOP for Anil)

- **Positive R-H1 correctness DEMONSTRATED:** ≥3 bf16 families co-resident, per-`model_uuid`
  routing isolated (KL=0, Marlin engages), no cross-contamination — the dtype misdiagnosis and
  the harness-churn confound are cleared.
- **D.7 does NOT close yet:** the Mem #16 full no-regression remains (multi-hour), plus the
  tightened positive refs (all-family clean solo + mis-route re-run).
- **Marlin cache-GC-on-free** robustness item noted (non-production churn; future hardening).
- Anchor unrotated. **Options:** (1) tighten positive refs (all-4 clean) + run the full
  no-regression → close D.7 (rotate anchor, tag `d7-rh1-close`); (2) bank the positive milestone
  and sequence the regression next; (3) proceed to the broader V.1 measurement at bf16 now that
  the actuators are confirmed engaging+correct in-container.
