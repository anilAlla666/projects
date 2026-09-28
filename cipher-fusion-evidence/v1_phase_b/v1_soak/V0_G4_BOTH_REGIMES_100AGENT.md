# V0 G-O1 ENGINE INCREMENT 4: both-regimes 100-agent real-workload gate, CIPHER end-to-end (CLOSING gate)

**2026-06-03. CORRECTNESS PASS (FAULT=0) under the Anil-ratified teacher-forced oracle, both configs. NO `.so`
change; inc-1+inc-2 modules BYTE-IDENTICAL + gates PASS; inc-3 M=3 wall PASS; deployed anchor 1f305ce6 (May-27) +
staging (Jun-1) UNCHANGED. `CIPHER_RT_DISABLE_AUTO_INIT=1`.** Artifacts: `cipher_inc4.py` (engine + scheduler +
teacher-forced verify), `pager_inc4_{diag,logitdelta,graphtf}.py` (the oracle-ratification probes).

## What ran

The full engine end-to-end: multi-model pager residency (inc-3, swap-on-miss) + batched lockstep coalescing (inc-2
WaveServer, re-captures per wave => inc-3b swap fix satisfied for free) + graph-decode over pager (inc-1). CIPHER
owns allocator+dispatch+graphs; NO vLLM in the serving path. Reproducible fixed trace (NOT live-asyncio): **100
agents = 50 short-burst (10-80 tok) + 50 long-burst (128-160 tok)**, Zipfian model-popularity over the
CAPTURE-CORRECT set {Qwen2-7B, Llama-3.1-8B, Llama-3.2-1B, TinyLlama} (Mistral EXCLUDED -- SWA static-KV capture is a
separate open binding term), gamma bursty arrivals, growing context, coalesced only within (model, gen-bucket)
cohorts (never short+long together -- that would inflate short P99 to the long-wave time).

## The correctness oracle was REDEFINED mid-increment (Anil-ratified) -- and that's the real story

The first K=4 run flagged **9/100 FAULT under free-running token-match-vs-batch-1-solo**. Probe-first
root-causing showed the engine is CLEAN and the GATE was wrong:
- **Graph teacher-forced (the decisive test):** feed an agent its solo tokens, free-run its 7 co-tenants, capture
  the batched GRAPH's per-step logits -> within **0.039** (the batch-shape delta) of solo at EVERY step ->
  zero contamination, correct logits.
- The 9 "FAULTs" are **benign near-tie CASCADES**: a single sub-delta near-tie flip early (e.g. eager fd=27, solo
  margin **0.016** < delta) yields a valid-but-different greedy continuation over 48-160 tokens; the flips are
  **non-deterministic** (didn't reproduce run-to-run = FP near-tie signature). co-tenant invariance: an agent's
  output is identical across two B=8 compositions (`pager_inc4_logitdelta.py`).
- **Anil ratified** (AskUserQuestion, exactly as for the inc-2 B>1 gate): correctness = **teacher-forced per-step
  logit/argmax closeness** (TF kills the cascade -> each mismatch is an ISOLATED step; benign iff solo top1-2
  margin < TFTHRESH=0.10, real if margin >> the ~0.04 batch delta) **+ co-tenant invariance**. Free-running
  token-match over long sequences is the wrong oracle for batched decode.

## Result under the ratified oracle (LIVE per-agent, ALL 100, not sampled)

| config | served | exact | near-tie | **FAULT** | misroute-nc | swaps | HBM free |
|---|---|---|---|---|---|---|---|
| **K=4 (product, swap INACTIVE -- all 4 fit)** | 100/100 | 92 | 8 | **0** | DETECTED | 0 | 40.4GB |
| **K=2 (forced budget, swap ACTIVE -- debt stress)** | 100/100 | 94 | 6 | **0** | DETECTED | 20 | 45.7GB |

**The carried inc-3b swap debt did NOT re-emerge** under the K=2 20-swap stress (re-capture-per-wave held); the live
safeguard confirmed 0 FAULT, not silently passed.

**Product numbers:** 100 agents, 4 distinct models, ONE H100, 0 correctness FAULT. P99 latency: **short
p50=1.70s/p99=3.49s | long p50=5.39s/p99=10.21s** (≈identical K=4 vs K=2 -- swap cost is small vs decode).
COALESCING 27 waves, mean **3.70/8** (modest, realistic under Zipfian+gamma+mixed -- the lockstep-alignment limit).
COST: re-capture ~56ms/wave, decode ~865ms/wave, lockstep waste 13%.

## DRIVER-LEVEL split (no overclaim)

Per-model pager region reserve/map/unmap (cuMemMap) = DRIVER-BOUNDARY; per-model batched graph capture+replay =
IN-PROCESS; CIPHER owns the allocator + the graphs; zero vLLM in the serving path. The singleton wall (inc-3) is
what makes the in-process multi-model mux possible.

## Honest CP-5.5 read -- what this proves and what's still gated

PROVES (toward "100 agents, 50 short + 50 long, distinct models, one H100, KL=0"): the **multi-model fp16 serving
SUBSTRATE** works end-to-end -- 100 both-regimes agents across 4 distinct models in one process, per-agent
correctness (teacher-forced, 0 FAULT), zero contamination, swap-on-miss correctness-held.

STILL GATED (state plainly):
1. **CIPHER's COMPUTE STACK IS OFF.** Runs under `CIPHER_RT_DISABLE_AUTO_INIT=1` -> Marlin/FP8/fusion/DVFS-tok/W
   all disabled (legacy actuators are not torch.cuda.graph-capture-safe). So inc-4 is the SUBSTRATE, **not the
   integrated product**; engine ⊕ compute-actuators do NOT yet compose. This is the single biggest gap.
2. **4-model availability ceiling, not the 14-family vision.** Only 4 capture-correct models exist (Mistral's SWA
   excluded). And 4 fp16 7-8B+1B fit in 35GB < 80GB -> **in the product config (K=4) the pager/SWAP is INACTIVE**;
   M>capacity does not occur at current model availability. The pager is a solution to M>capacity; swap (the
   highest-debt component) only runs in the artificial K<4 stress config. The "5-distinct-model availability
   ceiling" wall stands.
3. **Swap mechanism still UNCHARACTERIZED** (inc-3b, 5 refuted candidates). Did not re-emerge under K=2 stress, but
   it is un-rooted debt guarded only by the live teacher-forced safeguard.
4. **P99 is queue+decode-bound, host-overhead-limited.** Single-GPU serial decode of 100 agents + per-step
   host-overhead (inc-2, ~not bandwidth-floored). short P99 3.5s exceeds a typical sub-second tool-call SLA -> the
   binding term is single-GPU serial throughput + host-loop overhead, NOT engine correctness. No scope-down: that
   is the honest number; lifting it needs GPU-side sampling/pipelining + more GPUs, not capping agents.
5. **G-O3 (MFU, compute-bound regime) UNTOUCHED** -- a separate axis.

## Gate status (Mem #11) + non-regression

Correctness FIRST: per-agent teacher-forced KL-closeness 0 FAULT across ALL 100 agents both configs + misroute
DETECTED. Product numbers reported honestly. NON-REGRESSION: inc-1+inc-2 modules byte-identical + gates PASS; inc-3
M=3 wall PASS; OFF byte-identical; anchor 1f305ce6 UNCHANGED. Subprocesses os._exit-reaped, GPU->0.

## STOP -- the G-O1 engine is DONE (substrate)

Increments 1-4 closed+tagged: graph-decode over pager (KL=0, physical evict), batched coalescing (zero
contamination), singleton wall DISSOLVED (the moat), swap correctness sustained, both-regimes 100-agent substrate
(0 FAULT). The G-O1 multi-model fp16 SUBSTRATE is proven. The integrated PRODUCT is gated on: composing the compute
stack (Marlin/FP8/DVFS) capture-safely into the engine path (#1), more capture-correct/distinct models (#2), and
rooting the swap mechanism (#3). Next axes per the plan: G-O3 (MFU, compute-bound) + V.0/V.1 assembly. Anil's call.
