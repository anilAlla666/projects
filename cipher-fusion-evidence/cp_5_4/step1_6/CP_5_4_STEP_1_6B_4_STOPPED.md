# CP 5.4 — Step 1.6B-4 — STOPPED (deliberate reprioritization)

**Date:** 2026-05-19. **Status: Arm-B sweep STOPPED mid-run by operator
decision — a strategic reprioritization, not a failure.** Anchors unchanged.

---

## What ran before the stop

The Phase 5 Arm-B sweep was launched (OP-2/OP-5/OP-asym, 5 headline + 1 trace
rep each). Completed before the stop:

- **OP-2 — all 6 reps PASS** (5 headline + 1 trace). Disjointness clause1/2 = 0
  every rep; all partition KL gates PASS.
- OP-5 rep 0 was in progress; OP-asym not started.

OP-2 headline result (`cp54_cipher_op2_result.json`, n=10 partition tenants):

| arm | per-tenant decode CV |
|---|---|
| Arm-A naive (OP-2, from 1.6B-3) | **0.160** |
| Arm-B CIPHER (OP-2) | **0.256** — bootstrap 95 % CI [0.252, 0.260] |

**At OP-2, CIPHER SM-partitioned tenants show *higher* latency variance than
naive multi-process** — the CIs do not overlap, so this is a real result, not
noise. The Step 1.6 design-memo §3 premise ("PARTITION tenants under CIPHER
show tighter per-tenant latency variance") is **falsified at the
low-contention operating point.**

**D4 (Marlin × partition) — resolved:** the OP-2 trace rep's CUPTI cuModuleLoad
trace shows `marlin_engine_initialised=False` — Marlin is **out of the measured
path** (B=1 decode never reaches Marlin's B≥8 regime), exactly as the design
memo §4.5 predicted.

## Why it was stopped

A deliberate operator reprioritization, reasoned through explicitly:

1. **SM partitioning is not a tok/W lever** — established earlier in the
   campaign ([[cipher-lift-framing]]). 1.6B-4 was never going to produce an
   efficiency number.
2. **The partition path is already functionally verified** — the 1.6B-2A
   re-validation passed on current anchors (`CP_5_4_STEP_1_6B_2A_REVALIDATION.md`).
   "Does partitioning work?" → yes. 1.6B-4 only characterizes *how much it
   helps variance* — and OP-2 answered: it does not.
3. **CP 5.5 (the 100-tenant benchmark) does not gate on 1.6B-4** — at N=100 the
   deployment is POOL-batched (≈95 batched + a few partitioned); it needs the
   partition path to *work*, not to be variance-characterized.
4. **Latency isolation is a real concern — but belongs elsewhere.** Per-tenant
   latency predictability genuinely matters for the 100-agent goal (agentic
   workloads chain many model calls; jitter compounds). But the right place to
   measure it is as a **latency axis of the CP 5.5 POOL-batched benchmark** —
   on the architecture that will actually ship — not as a standalone
   SM-partitioning study. Real serving stacks deliver latency isolation via
   scheduling/batching policy, not SM fencing.

Finishing OP-5/OP-asym would only have mapped how negative the isolation
result is across more operating points — refining a writeup, changing no
downstream decision.

## CP 5.4 status

**CP 5.4 is FUNCTIONALLY COMPLETE; latency-variance characterization
DEFERRED.** The substrate is built and verified — the kmod CP 5.4 group
ledger, the 15×8-SM-group allocator, green-context partition allocation, the
do_exit reaper, disjointness — all pass (isolation test 15/15; 1.6B-2A
mixed-deployment re-validation PASS). What is *not* done is the 1.6B latency-
variance sweep, deliberately deferred as not goal-critical.

## Anchors

Unchanged — `cp54_s16_*` are throwaway harnesses, no substrate source touched:
kmod `008b3c66`, libcipher_rt `83afd1ca`, cipher_kv_bridge `c04b0c39`,
libcipher_v2 `cc0479b8`.

## Carried forward

"Per-agent latency predictability" → a **measured axis of the CP 5.5
100-tenant POOL-batched benchmark** (p99 / inter-token-latency per agent),
alongside aggregate tok/W. Not a standalone SM-partitioning measurement.
