# Phase B — Session 2 Step 4: Heterogeneous Batch — Finding (STOP / ADJUDICATE)

**Date:** 2026-05-18. Workload WL01 (TinyLlama-1.1B). Anchor `a7ac8e97`
unchanged. **Result is materially below projection — stopping per the
Session-2 discipline ("if >20% off: stop and adjudicate; don't optimize
through a fundamental bottleneck").**

## What was built

The batched-decode executor's free-run path was rewritten from a monolithic
`generate()` into a **continuous-batch decode loop**: prefill B=N, then a
manual step loop where each tenant has its own `gen_len`; a row that reaches
its `gen_len` (or EOS) is **evicted** from the batch and the KV cache
(`past.layers[i].keys/values` reindexed). `cipher_batch_executor.py`
(`.pre_step4` preserves the Step-3 version).

## Measurement — N=4, heterogeneous vs equal-length

Both runs on the executor; correctness teacher-forced (KL ≤ 0.1).

| run | gen_lens | useful tok | wall | tok/W | batch_eff |
|---|---|---|---|---|---|
| equal-length | 128,128,128,128 | 2112 | 10.27 s | **1.471** | 1.000 |
| heterogeneous | 32,64,128,256 | 1953 | 22.66 s | **0.622** | 1.000 |

**Padding/drain cost = 1 − 0.622/1.471 = 57.7 %** — target was < 15 %.
Correctness PASS (KL_max 5.5e-5).

## Diagnosis — eviction works; the *drain* is the cost

`batch_efficiency` (useful_tokens / token_steps) = **1.000** for the
heterogeneous run: eviction is correct — **zero wasted compute**, the batch
never computes a finished row. So the 57.7 % is *not* a padding/masking
inefficiency.

The cost is the **drain**. A round's wall time is gated by the *longest*
tenant: gen_lens [32,64,128,256] → the round runs **256 decode steps**. After
step 32 tenant-1 evicts, after 64 tenant-2, after 128 tenant-3 — so for steps
129–256 only **one** row is active. Each decode step costs ~the same wall time
regardless of batch size (decode is launch/overhead-bound — established in
Diagnostic 1; confirmed: B=4 and B=16 rounds had near-identical wall). So the
executor pays 256 steps of step-time to deliver 480 useful tokens, while the
equal-length case delivers 512 in 128 steps. The GPU runs near-empty for the
tail of every heterogeneous round.

This is fundamental to **eviction-only** batching: a batch that starts at N
and only drains can do no better than its longest member's step count.

## The fix is admission — i.e. Step 5

Refilling drained slots with new work keeps the batch full: when tenant-1
finishes at step 32, **admit** a new request into its slot; the batch stays
≈N; 480 useful tokens take ≈120 steps, not 256; the drain cost collapses to
near zero. That is exactly the **dynamic admission scheduler** scoped as
Step 5.

**Steps 4 and 5 are one mechanism.** The heterogeneous-batch target (< 15 %
overhead) is not reachable by eviction alone — it requires admission. The
continuous-batch loop built for Step 4 (eviction proven, 0 % compute waste)
is the substrate; admission is the missing half. Trying to "optimize through"
the drain without admission would be optimizing through a fundamental
bottleneck — which the discipline says not to do.

## Recommendation (for adjudication)

**Merge Steps 4 and 5 into one build:** the continuous-batching scheduler =
the eviction loop (done) + admission control (the remaining work). Then
re-measure: heterogeneous [32,64,128,256] *with admission* — projected
padding/drain cost well under 15 % because the batch no longer drains.
Step 5's other targets (tail-latency p99 ≤ 1.5× single-tenant, aggregate
tok/W within 10 % of the static-barrier result) are measured on the same
combined build.

Eviction is verified and correct; correctness gate passes. The only honest
path to the Step-4 target runs through Step-5 admission. Adjudicate: proceed
to the combined Step 4+5 build, or revise the Step-4 target.

## Artefacts

`phase_b/session1/` — `cipher_batch_executor.py` (+`.pre_step4`),
`run_batch_het.sh`, `equal/`, `het/` (executor results, client JSONs, power),
`step4_result.json`.
