# Phase B — Step 4+5 Sub-component 2 (KV-splice) — Finding (STOP / ADJUDICATE)

**Date:** 2026-05-18. Anchor `a7ac8e97` unchanged. **Throughput is
materially below projection — stopping per the Session-2 discipline.**

## What was built

`cipher_cb_executor.py` sub-component 2: KV-splice admission. Survivors' KV is
kept in place; only a newly-admitted request's prompt is prefilled (B=1) and
spliced into the running batch, left-padded to a common length. Invariant
held: every row's real KV content stays right-aligned; mask = trailing
`real_len` ones; a new token's position_id == `real_len`.

## Verification — N_SLOTS=4, 12 mixed-gen_len requests

| metric | sub-component 1 (re-prefill) | sub-component 2 (KV-splice) |
|---|---|---|
| completed | 12/12 | 12/12 |
| correctness — teacher-forced KL_max | 5.5e-5 (PASS) | **5.5e-5 (PASS)** |
| aggregate tok/s | ~51 | **52.2** |
| decode steps | 432 | 427 |

**The KV-splice is correct** — admission works, the left-padding invariant
holds, the correctness gate passes. **But throughput did not move** — 52 vs
51 tok/s. The target was ≥ 80 % of static-N=4 batched (≥ ~160 tok/s);
delivered ~25 %. That is > 20 % off projection → stop.

## What this rules in and out

The splice removed the re-prefill-of-survivors cost — and throughput was
**unchanged**. Therefore the bottleneck is **not** the rebatch strategy. Both
sub-components run at ~53 ms per decode step; the static batched executor runs
at ~15 ms. The regression is in the **per-step cost of the manual
continuous-batch decode loop itself**.

## Diagnosis — honestly incomplete

I hypothesised that the left-padded ragged batch forces an explicit dense
attention mask, taking SDPA off the FlashAttention path. A micro-benchmark to
confirm it was **inconclusive** — warmup-dominated noise (the *same*
explicit-mask decode step measured 77 ms, then 19 ms on a later run). A
*warm* decode step is ~19 ms even with the explicit mask, yet the executor
averages ~53 ms/step. The gap is real but **not cleanly pinned**. Candidate
causes, none confirmed: per-step Python overhead in the manual loop; KV length
`L_phys` ratcheting up (never cropped) so late steps attend over a long KV;
CUDA kernel re-selection as the batch tensors are replaced every rebatch;
warm-up. Pinning it needs proper profiling, not another guess.

## The honest bigger picture

This is the point the prototype was always going to reach. Cross-tenant
batching at **static** batch sizes is verified and strong — Session 2
Steps 1-3 (N=8, 3.69× substrate-attributable) and the Mistral-7B N=4
verification (3.26×), both correctness-gated. Those used `m.generate()` /
equal-length batches on the Flash path and hit ~15 ms/step.

**Dynamic continuous batching** — admission under streaming heterogeneous load
at *competitive* throughput — is a different magnitude of work. Making it fast
is precisely what paged-attention serving engines (vLLM, SGLang) exist to
solve, with years of kernel engineering (paged-KV Flash kernels that handle
ragged lengths without a dense mask). A left-padding prototype is correct but
will not match a static-batch / paged engine on throughput. "Make continuous
batching production-competitive" is not a sub-component — it is a serving
engine.

## Recommendation — adjudicate among three

1. **Close CP 5.6 on the verified static result.** The defensible,
   correctness-gated CIPHER multi-tenant numbers already exist: N=8 3.69×,
   Mistral-7B N=4 3.26×, substrate-attributable. Scope *dynamic* continuous
   batching (admission, streaming, tail latency) as separate future work.
   CP 5.6's TPW question is answered by the static result.
2. **Commit to a real paged-attention continuous batcher.** A major build
   (effectively adopting vLLM's paged-KV core). Weeks, not days. Only worth it
   if the dynamic operating point is a hard product requirement.
3. **Integrate vLLM for the dynamic regime.** Position CIPHER's substrate as
   the transparent *cross-process* layer (the architectural distinction from
   the methodology memo) on top of vLLM's intra-process paged batching, rather
   than reimplementing it.

My recommendation: **option 1** for CP 5.6 closure. The static multi-tenant
result is real, verified, and meets the ≥3.6× target; the dynamic-regime
prototype has demonstrated *correctness* (admission + eviction + splice all
work, gated) but its throughput is a serving-engine problem, not a prototype
fix. Closing CP 5.6 on the static result is honest and defensible;
continuous-batching-at-throughput is its own scoped project.

## Status

Sub-component 2 (KV-splice): mechanism built, **correctness verified**,
**throughput target missed**, root cause not pinned. Sub-components 3–7
(scheduler, TTFT/ITL, stream, burst, naive baselines) are blocked on this
throughput question — building them on a loop running at 25 % of static would
measure the prototype's overhead, not CIPHER. Anchor `a7ac8e97` unchanged.

## Artefacts

`phase_b/session1/cipher_cb_executor.py` (+`.subcomp1`), `cb_subcomp2_result.json`,
`cb_subcomp1_result.json`, `cb_workload_test.json`.
