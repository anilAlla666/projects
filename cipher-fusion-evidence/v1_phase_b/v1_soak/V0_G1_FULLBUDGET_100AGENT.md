# G-O1 at full budget — the residency/memory side imposes NO constraint on 100 agents; dispatch is 100% of the gap (and it's the deferred engine, not the pager)

**Date:** 2026-06-01. **Type:** measurement (NO source change; built pager, eager decode — the dispatch engine is
the deferred build). Harness: `pager_g1_fullbudget_100agents.py`. **Anchors UNCHANGED: deployed `1f305ce6`, staging
`2edba0d2`.** Policy moot here (full budget → no eviction → LFU-DA ≡ LRU; ran once). Correctness: per-agent KL=0 5/5.

**The real product claim, measured: at FULL budget the residency/memory side accommodates 100 agents with zero
paging — so it is NOT the binding constraint. The entire gap to a real 100-agent deployment is DISPATCH throughput,
which is the deferred engine (vLLM-regime), not the pager.**

## Structural fact (binding, not a harness choice)
This pod has exactly **5 distinct causal-LM int4 models** (TinyLlama-1.1B, Llama-3.2-1B + Mistral-7B, Qwen2-7B,
Llama-3.1-8B; the other assets — MiniLM/CLIP/LLaVA/SDXL/Whisper — are not causal LMs the pager serves). Their reserves
sum to **24.5 GiB ≪ 75 GiB budget.** Two consequences the literal task can't escape on this pod:
- **"Models-per-GPU toward 100" has no answer here** — only 5 distinct models exist; *model availability* is the
  ceiling, not the 75 GiB budget.
- **At full budget there is nothing to page** — all 5 stay resident, so the pager is correctly *inactive*. The
  pager's demonstrated value is at *constrained* budget (the earlier Zipfian / hot≠big work); at full budget the
  memory fit is **arithmetic**, not the pager doing clever work.

## Residency / memory side — imposes NO constraint at full budget
| metric | result |
|---|---|
| models resident @ 75 GiB | **5/5** (all distinct fit; reserves 24.5 GiB) |
| cold-miss, 100-agent Zipfian stream, burstiness 1.0/0.5/0.2 | **0 / 0 / 0** (256/327/452 reqs) |
| KV @ 1K ctx, 100 agents | TinyLlama 2.1 GiB, 7-8B models 12.5 GiB — all fit in 50.5 GiB headroom |
| per-agent KL=0 across swap | **5/5** |

So 100 agents' weights (24.5 GiB) + KV (~2–12.5 GiB depending on model) fit comfortably in 75 GiB; every agent's
model is a HIT (cold-miss 0). **The memory side does not constrain 100 agents at full budget.** This also confirms the
earlier finding that the 18 GiB budget was a *stress knob* — at the card's real budget, residency is a non-issue for
this model count.

## Dispatch side — 100% of the gap (eager floor is a CONTAMINATED directional lower bound; lean on prior clean numbers)
The single in-order eager server saturates catastrophically: per-burst (prefill 512 + 16 decode) measured 1.1–2.1 s,
100-agent eager-stream p99 = 221–368 s, agents-per-GPU at interactive SLO (500/1000/2000 ms) = **0** (even one agent's
single burst exceeds the SLO).
- **These eager numbers are an OVERHEAD-CONTAMINATED directional floor, NOT a clean measurement.** The tell: per-model
  latency is **non-monotonic in size** — TinyLlama-1.1B (1747 ms) is *slower* than Llama-3.1-8B (1152 ms), impossible
  if decode-bound. So the 1–2 s is dominated by fixed per-`generate()` Python overhead + first-call CUDA warmup
  (TinyLlama is measured first), not by decode throughput or quantization. Do not read precise eager latencies here.
- **The clean dispatch evidence is the PRIOR measured numbers:** eager ~26 tok/s vs **graph-decode 167 tok/s/agent**
  vs same-model **continuous batching 8808 tok/s** (N=64). The engine lifts dispatch by **1–2 orders of magnitude** —
  that magnitude is already established; this run only re-confirms the *direction* (eager single-thread is unusable
  for bursty multi-agent serving).
- **Why "agents=0" is NOT a pager/residency failure:** eager single-thread in-order serving has *no batching* — it is
  the wrong architecture for bursty multi-agent, which is exactly *why the dispatch engine exists*. This is a statement
  about the missing dispatch layer, not about residency (cold-miss is 0) or KV (fits).

## The decomposition — the engine is two already-measured halves composed
A 100-agent / 5-distinct-model deployment is **same-model continuous batching** (vLLM, measured 8808 tok/s — most of
100 agents share the hot small models) **⊕ 5-way distinct-model weight co-residence** (the built pager, cold-miss 0).
The "engine" is the *orchestration* that composes them **in one process** — a real build (vLLM's CuMemAllocator
singleton won't compose in-process, per the earlier probe → CIPHER must own the multi-model mux), but **not**
from-scratch decode.

## Verdict — decides the engine on a proven foundation, still Anil's call
- **Residency/memory imposes no constraint on 100 agents at full budget** (cold-miss 0, fits, KL=0). The pager's part
  is not the blocker.
- **Dispatch is 100% of the remaining gap**, and it is the deferred engine (batching ⊕ co-residence), whose magnitude
  is already measured (1–2 orders over eager). The literal "agents-per-GPU near 100" is reachable only with that
  engine; on eager it is ~0 at interactive SLO (architecture, not residency).
- **What is unmeasurable on this pod:** real batched dispatch composed with the pager (= the engine build itself), and
  M=15+ distinct models under pressure (only 5 exist). Another on-pod eager measurement won't move the decision.

**This confirms the residency foundation is sufficient for 100 agents at full budget → the engine (orchestration) is
justified on a proven foundation. It remains the same product call Anil already holds (months; vLLM-regime dispatch +
the in-process composition only CIPHER can do). STOP for the engine build decision.**
[[cipher-freq-residency-build]], [[cipher-zipfian-k-fleet-quality]], [[cipher-go1-held-regime-map]], [[cipher-go1-inprocess-composition]].
