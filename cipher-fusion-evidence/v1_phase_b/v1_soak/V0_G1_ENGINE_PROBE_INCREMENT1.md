# G-O1 engine: PROBE (overlap dead → graph-decode is the latency lever) + INCREMENT-1 (composed async router) — sub-second by composition for SHORT bursts; the engine's justification RELOCATES to longer bursts / higher concurrency / sub-200ms / M>capacity

**Date:** 2026-06-01. **Type:** PROBE-FIRST + increment-1 measurement (NO CIPHER source change; CLEAN vLLM, the
`.so` is not loaded). Harnesses: `g1_overlap_bandwidth_probe.py`, `pager_g1_async_router.py`,
`pager_g1_async_router_ctrl.py`. **Anchors UNCHANGED: deployed `1f305ce6`, staging `2edba0d2`.** **vLLM = per-model
engine (validator); CIPHER = the cross-model async ROUTER (the dispatch loop CIPHER owns — COMPOSED over vLLM
AsyncLLMEngine, NOT a reimplemented batching kernel).** Correctness: per-agent **greedy-match KL=0 3/3**.

## PROBE (load-bearing, before any months-build): is the latency floor removable by cross-engine OVERLAP?
Graph-replayed GEMV microbench sized to weight bytes (true GPU time — NOT an eager forward, which is dispatch/GIL-bound):
| regime | single-decode bw | concurrent 2-model speedup | aggregate bw |
|---|---|---|---|
| int4 (4 GiB/model) | 2.48 TB/s = **74% peak** | **1.01×** | 2.51 TB/s = 75% peak |
| fp16 (14 GiB/model) | 2.49 TB/s = **74% peak** | **1.00×** | 2.50 TB/s = 75% peak |

**A single model's clean weight-streaming already saturates HBM (~74% peak), so two concurrent decodes share the one
~2.5 TB/s effective pipe → no speedup. Cross-engine overlap is DEAD in the gapless regime.** Stated precisely (not
"overlap does nothing"): real *eager* decode is gappy (~20% peak, 167 tok/s) with spare bandwidth, so overlap *could*
fill gaps — but graph-decode (remove the gaps → low per-request latency) is strictly preferable to overlap (fill gaps
→ same ~75% ceiling, worse latency), and once decode is gapless overlap dies. **So overlap is NOT the latency lever;
graph-decode is.** Same-model continuous batching (prior 13.77×, amortizes one weight-read across the batch) stands
regardless. Principled (not measured) target: a 5-model wave's bandwidth floor ≈ 5×4GiB×16tok / 2.5TB/s ≈ ~130 ms →
sub-second is *reachable in principle* (real decode adds attention/KV, lands above the microbench).

## INCREMENT-1: composed async router (vLLM AsyncLLMEngine ×3 in-process, eager + CIPHER asyncio router)
3 engines = 1 small-hot (TinyLlama) + 2 big tail (Mistral, Llama-3.1); Zipfian affinity, duty-cycle burst+idle,
growing ctx, REAL-TIME async replay. The payoff is the real p50/p99 split my earlier conservative sum-over-engines
wave server (1.4–1.9 s) understated.

| config | p50 | p99 | per-engine p50/p99 (TinyLlama / Mistral / Llama-3.1) |
|---|---|---|---|
| 100 agents, 16-tok burst, **prefix-cached** (favorable artifact) | 171 ms | 479 ms | 167/189 · 356/389 · 445/488 |
| 100 agents, 16-tok burst, **prefix OFF + distinct ctx** (controlled) | 246 ms | **817 ms** | 208/344 · 471/849 · 493/817 |
| 100 agents, **128-tok** burst, prefix OFF + distinct ctx | 1803 ms | **5436 ms** | 1728/1835 · 4259/5528 · 3613/4242 |

**Controlled for the favorable artifact (advisor): prompts were initially `"data "*ctx` (identical → vLLM
`enable_prefix_caching=True` elided prefill). Re-run with prefix-caching OFF + distinct random per-request contexts
(the pessimistic bound — no within-agent reuse either). Verdict: at the realistic 16-token burst, p99 = 817 ms —
SUB-SECOND survives** with full prefill. Real agents have within-agent prefix reuse → land between 817 ms (control)
and 479 ms (cached). **But the 16-token-burst assumption is load-bearing: at 128-token bursts (realistic
response-generating turns) p99 = 5.4 s — NOT sub-second, decode-bound (latency ~linear in burst length, 16→128 ≈ 7×).**

## What this is (framing — confirms-and-quantifies, NOT a reversal)
The earlier wave-server number (1.4–1.9 s) was explicitly labeled a *conservative lower-bound-on-goodness*; the async
result **confirms that label and quantifies the real floor** — it does not overturn a careful prior claim. The
decomposition is unchanged: residency non-binding, throughput non-binding, **latency is the axis** — what moves is
that the latency axis is *already low for the short-burst regime*.

## The engine's justification RELOCATES (not "not needed")
**Within the measured envelope — ≤3 async engines (async ×5 UNTESTED — fragile; see operational note), 16-token
bursts, ≤1024 ctx, full budget (pager inactive), 100 agents — composing vLLM AsyncLLMEngine + a thin CIPHER async
router serves SUB-SECOND (p99 817 ms) TODAY, eager, no graph-decode, no months-build.** The months graph-decode engine
(CIPHER-owned, bypassing vLLM's cudagraph-monitor singleton) is justified for the regimes *outside* this envelope,
which this run shows are real:
- **longer decode bursts** (128-tok → 5.4 s p99, MEASURED; 500-tok turns would be ~20 s eager) — the dominant one;
- higher concurrency, sub-200 ms SLOs, and the **M>capacity pager case** (M=15, hardware-gated).
So: **composition covers the short-burst regime; the engine is for the harder ones.** "Composition wins, engine
pointless" would be the over-claim.

## Operational note (guard the artifact)
vLLM V1 spawns `VLLM::EngineCore` processes even at `VLLM_ENABLE_V1_MULTIPROCESSING=0`; `os._exit()` orphans them
(observed 3 holding ~48 GiB → the next run OOM'd at engine #3). Same class as the 44 GiB zombie earlier. The composed
engine must reap engine subprocesses on shutdown. 5-engine async-one-event-loop was not attempted (the flagged
fragility); 3 stood up cleanly. gpu_memory_utilization is PER-ENGINE footprint (own weights+KV)/total, NOT cumulative.

## Two honesty guards on the sub-second headline (the per-engine tail + conjunctive envelope)
- **The survivor is the MIX, not the tail.** The 817 ms aggregate stays sub-second only because Zipfian sends ~60%
  of traffic to TinyLlama (208/344 ms); the **big-model tail is already at/over 1 s at 16 tokens** (Mistral 471/**849**,
  Llama-3.1 493/**817**). So it is "sub-second for the *small-model-dominated mix*" — a fleet with a higher big-model
  share crosses 1 s even at 16-token bursts. The tail is the story (same lesson as hot≠big), not the blended number.
- **The envelope is CONJUNCTIVE; the corners compound.** Measured: 16-tok ∧ 3-engines ∧ ≤1024 ctx ∧ 100 agents →
  817 ms; 128-tok → 5.4 s. NOT measured: 128-tok ∧ 500-agents ∧ 5-engines (these degrade *together*, not
  independently). Honest line: **sub-second holds at the CONJUNCTION of favorable-corner assumptions; relax ANY one
  knob — burst length, big-model share, concurrency, context — and it moves toward the engine's regime.** Burst length
  is not the only knob (16→128 tok = 7× is just the one sensitivity we bounded directly).

## Verdict — decision-ready, the commitment is Anil's
PROBE: overlap refuted as the latency lever → graph-decode is the lever. INCREMENT-1: composed async (eager) is
sub-second for short bursts TODAY (no months-build), and the engine's justification relocates to longer bursts /
higher concurrency / sub-200 ms / M>capacity (longer-burst multi-second floor MEASURED). The decomposition is now
fully characterized in both directions. **STOP — the engine-build commitment (months) is Anil's; everything on this
pod that bears on it is measured.** No source change; clean vLLM; anchors unchanged.
[[cipher-go1-dispatch-router]], [[cipher-go1-fullbudget-100agent]], [[cipher-go1-inprocess-composition]], [[cipher-go1-dispatch-mechanism]].
