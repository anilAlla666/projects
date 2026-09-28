# Phase B — Mistral-7B Cross-Tenant Batched N=4 Verification

**Date:** 2026-05-18. Model **Mistral-7B-v0.1** (the model behind CP 2.4's
retracted 3.617× and CP 5.6 P2's honest 1.54×). Session 2 Step-1 methodology:
generate-based batched executor, N=4, 5 prompts × 128-tok greedy, B=1/tenant.
Anchor `a7ac8e97` unchanged.

**Verdict: the cross-tenant batching lever generalizes to Mistral-7B —
substrate-attributable 3.26× at N=4, above the 1.5× pass bar. Proceed to the
merged Step 4+5 build.**

## Result

| arm | agg tok/s | power | agg tok/W | correctness |
|---|---|---|---|---|
| batched executor N=4 | 175.7 | 204.2 W | **0.861** | TF KL_max 3.2e-5 |
| naive 4-concurrent | 85.9 | 324.9 W | 0.264 | — |

**Substrate-attributable = 0.861 / 0.264 = 3.26×.** Correctness gate
(teacher-forced logit-KL ≤ 0.1) PASS — KL_max 3.2e-5.

Memory: the batched executor holds **one** 14 GB Mistral copy; the naive
control runs 4 independent ~14 GB copies (~56 GB, fits 80 GB). N=8 naive does
not fit without weight-sharing (Track 2) — N=4 is the no-weight-sharing
ceiling, as scoped.

## Reading — and an honest note on why 3.26× exceeds the ~1.75× projection

The projection (~1.75×) was the TinyLlama N=4 figure. Mistral lands higher,
and the reason is instructive: **the naive baseline is far worse for the 7B
model.** Naive 4-concurrent Mistral delivers only 85.9 tok/s (≈1.8× a single
~48 tok/s tenant) at **324.9 W** — four independent 14 GB weight streams
thrash HBM bandwidth and the processes contend hard, burning power for little
aggregate gain. The batched executor sidesteps it: one 14 GB weight copy, one
efficient B=4 kernel stream — 2.0× the throughput at 0.63× the power.

So the substrate-attributable ratio is large partly because the realistic
alternative (naive concurrency) genuinely degrades on big models. That is a
real, honest result — the comparison is batched vs the realistic naive
baseline, both measured — and it means **the substrate's value grows with
model size**: the bigger the model, the worse naive multi-tenancy behaves,
the more cross-tenant batching wins. Mistral-class is the production target,
so this is the favourable direction.

Cross-model comparison, N=4:

| model | batched tok/W | naive tok/W | substrate-attributable |
|---|---|---|---|
| TinyLlama-1.1B | 1.750 | 0.994 | 1.76× |
| Mistral-7B | 0.861 | 0.264 | **3.26×** |

Absolute tok/W is lower for Mistral (7× the model) but substrate-attributable
is higher (naive degrades more). The launch-overhead bottleneck Diagnostic 1
identified does generalize — batching still amortises it — and the
concurrency-contention penalty it competes against is steeper at 7B.

## The loop on Mistral-7B

Same model, three measurements:
- **CP 2.4: 3.617× tok/W** — retracted (CP 5.6 P2: a degenerate-loop artifact).
- **CP 5.6 P2: 1.54× tok/W** — honest single-tenant composed lift.
- **Phase B (here): 3.26× substrate-attributable tok/W** — cross-tenant N=4
  batched, correctness-gated, measured vs naive concurrency.

The 3.26× is a **different claim** from the retracted 3.617× — it is
multi-tenant aggregate efficiency from cross-tenant batching, not a
single-tenant composed lift. It should be disclosed as such: a new, verified,
correctness-gated operating point on Mistral-7B — not a reinstatement of the
old number.

## Honest scope boundary

- **N=4 only** — N=8+ on Mistral needs weight-sharing (Track 2) for memory.
- **Equal-length prompts, generate-based static executor** — heterogeneous
  lengths + admission are the merged Step 4+5 build.
- **Vanilla FP16** — batching alone earns this; Marlin/DVFS not composed in.

## Next

Pass bar met (3.26× ≥ 1.5×) → the lever generalizes. Per the plan, proceed to
the **merged Step 4+5 build** (continuous-batching scheduler: eviction —
done — plus admission control), then CP 5.6 closes.

## Artefacts

`phase_b/session1/mistral_n4/` (executor result, client JSONs, power),
`phase_b/session1/cipher_batch_executor_gen.py`, `run_mistral_verify.sh`;
Mistral gold `phase_a/WL_MISTRAL/`; naive control `phase_a/WL_MISTRAL/mt4_vanilla/`.
