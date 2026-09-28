# CP 5.4 — Step 1.4 (confined-pool batch-lift curve) — REPORT

**Date:** 2026-05-19. **Status: COMPLETE — full curve measured, K = 15…5
(120…40 SMs).** The K ≤ 7 correctness failure in the earlier STOP report was
root-caused to a **measurement-harness race in the executor's Phase-2 TFGATE**
(not a substrate or green-context defect); the executor was fixed and the
K = 7, 5 points re-swept clean. The lift curve is **flat across the entire
120 → 40 SM range**. Anchors unchanged. Awaiting Step 1.4 close adjudication.

---

## Phase 1.4A — design

Adjudicated: pool sizes {15,13,11,9,7,5 groups = 120/104/88/72/56/40 SMs};
cap-knob method (`CIPHER_POOL_MAX_GROUPS` in `cp54_pool.py`, a documented
measurement instrument). Memo: `CP_5_4_STEP_1_4_DESIGN_MEMO.md`.

## Phase 1.4B — sweep (first pass) and the K ≤ 7 stop

The first 18-run sweep produced a clean, flat curve for K = 15…9 (120…72 SMs)
but hit `TFGATE KL = 32.56` (gate ≤ 0.1) at K = 7 and K = 5, triggering the
directed hard stop. The throughput numbers at K = 7, 5 *looked* normal
(~490 tok/s) but were void — the gate said the logits were wrong.

## Phase 1.4C — root cause of the K ≤ 7 failure

**Root cause: a cross-stream read race in the executor's Phase-2 TFGATE — a
measurement-harness bug, not CIPHER and not the green context.**

### Bisection (`cp54_s14_bisect.py`, throwaway harness)

A re-implementation of the TFGATE under progressively more of the executor's
context — bare `torch.cuda.GreenContext`, + `libcipher_v2` CUPTI injection,
+ the 5 prior `generate()` rounds, + the `cp54_pool` green-context path — was
run at K = 9, 7, 5 (logs `bisect_*.log`). **Every arm was clean (KL ~1e-5),
including the full "gen + pool + injection" replication at K = 7.** The
bisection harness could not reproduce the failure — because it already had a
`torch.cuda.synchronize()` after the green-stream forward, which the executor
did not. That *was* the differentiator.

### Decisive diagnostic (`arm3_s14diag_g7_rep1`)

The executor's own Phase-2 was instrumented (`TFDIAG`) to dump `tf.logits`
stats at K = 7. It showed the failure directly:

| round | prompt | tf.logits absmax | tf.logits mean | gold mean | KL |
|---|---|---|---|---|---|
| 0 | 0 | 24.312 | −3.875 | −3.870 | 0.00005 |
| 1 | 1 | 30.047 | −3.483 | −3.459 | 0.00003 |
| **2** | **2** | **0.000** | **0.00000** | −2.564 | **10.37** |
| 3 | 3 | 23.594 | −3.834 | −3.906 | 0.00002 |
| **4** | **4** | **29.219** | **−2.648** | −3.583 | **32.56** |

Round 2's logit tensor was **all zeros**; round 4's was **partially written**
(mean off). The Phase-2 forward runs on the green-context stream
(`_pool.gc_stream`); `tf.logits` is then consumed (`log_softmax`, slicing) on
the default stream **with no synchronization between them**. PyTorch does not
insert a compute dependency across that stream boundary. At ≥ 72 SMs the green
forward finishes before the default-stream consumer reaches the read, so the
race is won silently; at ≤ 56 SMs the forward is throttled enough that the
consumer overtakes it and reads zero/partial memory — finite-but-wrong logits,
deterministic, content-independent. The 72↔56-SM boundary is a timing
threshold, not a correctness threshold.

This is consistent with everything the STOP report observed (deterministic,
finite-not-NaN, KL identical across reps, green-ctx self-verify PASS) and with
the prior isolation test (torch green contexts proven bit-identical at 40 SMs
— because that harness, too, synchronized).

### Fix and verification

One line added to the executor's Phase-2 TFGATE — `torch.cuda.synchronize()`
immediately after the green-stream forward, before the default-stream consumer
(`cipher_batch_executor_gen.py:130`). Phase-1's free-running generate already
synchronized each round; only Phase-2's teacher-forced forward was missing it.

Fix-verification run `arm3_s14fix_g7_rep1` (K = 7): all 5 TFGATE rounds clean,
`tf.logits` stats now match gold on every round, max KL = 0.000055.

**The fix touches only `cipher_batch_executor_gen.py`, a Phase B
measurement/test harness — NOT a campaign anchor.** No anchor rotates.
Pre-fix executor preserved (`.pre_cp5_4_step1_3b`, and the buggy
pool-but-pre-fix version at
`cp_5_4/step1_3b/cipher_batch_executor_gen.py.cp5_4_step1_3b`);
diagnostic version `.cp5_4_step1_4_diag`; final clean version `.cp5_4_step1_4`.

## Phase 1.4D — re-sweep of K = 7, 5 on the fixed executor

K = 7 and K = 5 re-swept, 3 reps each, fixed executor (TFDIAG removed), tag
`step1_4b_g{7,5}`:

| K | SMs | tok/s reps | mean tok/s | mean tok/W | worst KL | correctness |
|---|---|---|---|---|---|---|
| 7 | 56 | 494.82 / 496.94 / 491.68 | 494.48 | 3.5497 | 0.000055 | **PASS** |
| 5 | 40 | 496.88 / 476.16 / 470.70 | 481.25 | 3.4360 | 0.000055 | **PASS** |

K = 15…9 are **not** re-swept: at ≥ 72 SMs the race was won silently, so those
points were always valid; the fixed executor reproduces their TFGATE KLs
bit-identically — every one of `g15`/`g13`/`g11`/`g9` shows the identical
`0.000045 / 0.000030 / 0.000055 / 0.000020 / 0.000031` round sequence as the
fixed `s14fix_g7` run. Their rows stand, empirically, for every K.

## The completed lift curve (`step1_4_lift_curve.csv`)

| K (groups) | SMs | mean tok/s | Δ vs 120-SM | mean tok/W | correctness |
|---|---|---|---|---|---|
| 15 | 120 | 491.75 | anchor | 3.5447 | PASS |
| 13 | 104 | 492.41 | +0.13% | 3.5424 | PASS |
| 11 |  88 | 487.79 | −0.81% | 3.5104 | PASS |
|  9 |  72 | 494.98 | +0.66% | 3.5574 | PASS |
|  7 |  56 | 494.48 | +0.55% | 3.5497 | PASS |
|  5 |  40 | 481.25 | −2.14% | 3.4360 | PASS |

**The curve is flat from 120 SMs down to 56 SMs** — every point inside the
workload's ~±3% rep-to-rep noise. At **40 SMs** tok/s is −2.14% (within ±3%)
and tok/W −3.07% (at the noise gate); the K = 5 reps are noisy (470–497 tok/s,
~5% spread). **No knee, no sub-linear collapse anywhere in 120 → 40 SMs** —
§8's "collapse past the 64-SM point" stop-condition is not triggered. The
wider K = 5 spread (~5% vs ~2–3% at K ≥ 7) is a soft signal of *approaching*
but not crossing the §8 knee; Step 1.5 may want to probe past 40 SMs.

**Interpretation:** confining the Phase B N=8 TinyLlama pool from 120 SMs to as
few as 40 SMs costs no significant throughput. B=8 decode is overhead /
memory-bound, not SM-bound, in this range — consistent with the Phase-A
finding. The lift curve anchors at the 120-SM (15-group) top point per scope
memo §8 / Step 1.3b' closure; it does not use Phase B's full-132-SM 3.69×.

## What is solid

- The confined-pool lift curve is **measured and flat across 120 → 40 SMs**
  (6 sizes × 3 reps, all PASS, teacher-forced KL ≤ 1e-4).
- The K ≤ 7 "failure" was a measurement-harness cross-stream race, root-caused,
  fixed in the executor, and re-verified clean. It was never a substrate or
  green-context correctness problem — torch green contexts and the CP 5.4 pool
  path are clean at 40 SMs.
- Green-ctx self-verify PASSED at every size (SM placement always correct).

## Anchors

Unchanged — measurement step. kmod `8d777dfb`, libcipher_rt `ebc0baaa`,
libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d`. Source touches, both in
Phase B test/measurement files (not anchors):
1. `cp54_pool.py` — `CIPHER_POOL_MAX_GROUPS` measurement knob (Step 1.4A;
   preserved `.pre_cp5_4_step1_4`).
2. `cipher_batch_executor_gen.py` — Phase-2 TFGATE cross-stream sync fix
   (Step 1.4C; preserved `.cp5_4_step1_4`, diag `.cp5_4_step1_4_diag`).

## Adjudication ask

Phase 1.4 is measurement-complete: the curve is flat 120 → 40 SMs, the K ≤ 7
blocker is resolved (harness bug, fixed, re-verified). Recommend **close
Step 1.4** on the flat curve. Remaining CP 5.4 steps: 1.5–1.8.
