# G-O1 dispatch-side at full budget — throughput and residency are both NON-binding; the gap is per-request latency under multi-model routing = the engine (cross-engine overlap + async batching; graph-decode secondary)

**Date:** 2026-06-01. **Type:** measurement (NO substrate source change; the CIPHER .so is NOT loaded — clean vLLM
via `VLLM_PLUGINS=""`). Harness: `pager_g1_router_eager.py`. **Anchors UNCHANGED: deployed `1f305ce6`, staging
`2edba0d2`.** **vLLM = per-model decode harness (5 in-process engines, continuous batching = the dispatch lever);
CIPHER = the multi-model router across them (this script).** Correctness: per-agent **greedy-match KL=0 5/5** (solo
vs batched — continuous batching does not perturb routing).

**Replaces the CONTAMINATED eager-HF capstone number (96c7e8c: 1–2s/burst, p99 221–368 s) with the clean dispatch
number on real continuous batching. Result: at full budget, throughput and residency are BOTH non-binding for 100
agents; the only remaining gap is per-request latency under multi-model routing — which is the deferred engine.**

## Structural notes (binding)
- **At full budget the PAGER is inactive** (5 models fit; prior capstone). This measures **vLLM-per-model ⊕
  CIPHER-routing**, NOT the pager. The pager's differentiation is the M>capacity case (M=15, hardware-gated).
- **graph-decode 5-engine in-process is BLOCKED by a vLLM global-singleton:** all 5 engines coexist with
  `enforce_eager=False` (65.2 GiB), but the first decode raises `CUDA graph capturing detected at an inappropriate
  time. This operation is currently disabled` — vLLM's process-global cudagraph-capture monitor. **Same class of wall
  as the CuMemAllocator singleton (earlier probe): vLLM's globals do not compose across N in-process graph engines.
  This independently proves "the engine" is a real in-process build, not "run 5 vLLM engines."** Per advisor,
  eager-vLLM answers the same question (continuous batching is the lever; graph-decode is a ~10–30% decode
  increment, second-order given ~20× throughput headroom). Ran eager.
- **The server model is CONSERVATIVE:** 5 engines share ONE GPU, processed sum-over-engines per wave with **zero
  cross-engine overlap** → a lower-bound-on-goodness (the real async engine overlaps and does better).

## What is ROBUST (model-independent): throughput is NOT the binding term
Under the 100-agent load the engines run at **~1000–1300 tok/s (hot) down to ~165–380 (cold)** vs the measured
same-model peak **8808 tok/s** — i.e. far below saturation, because bursty load keeps per-engine batches small. This
does not depend on the wave model's quirks: **the engines are mostly idle; dispatch capacity is not the limit.**
(Consistent with the throughput arithmetic: ~100 bursty agents ≈ a few hundred tok/s aggregate ≪ per-engine capacity.)

## Numbers (eager, conservative single-GPU wave server; burstiness 0.2 = correlated, hardest)
| N agents | reqs | p50 | p99 |
|---:|---:|---:|---:|
| 25 | 161 | 403 ms | 1433 ms |
| 50 | 375 | 773 ms | 1738 ms |
| 100 | 833 | 1248 ms | 1902 ms |
| 150 | 1270 | 1467 ms | 2051 ms |
| 200 | 1662 | 1528 ms | 2178 ms |

N=100 across burstiness 1.0/0.5/0.2: p99 = 1771 / 1762 / 1908 ms.

**Honest reading (not the confirmation-bias one):** p99 is **~N-independent** (~1.4–1.9 s, already 1433 ms at N=25),
but **p50 is NOT flat — it triples 403→1528 ms from N=25→200**, so there *is* real queue accumulation (same
conservative model produces both; I do not credit only the favorable metric). The sharp, SLO-independent finding is
therefore: **agent count is not the binding variable — the per-burst latency FLOOR (~1.4–1.9 s) is.** The arbitrary
SLO line, not the agent count, picks any "agents-per-GPU" number: ~100–150 at a (lax) 2 s line, **0 at sub-1 s even at
N=25.** So "agents-per-GPU @ P99<2000ms ≈ 100–150" is a *secondary, SLO-line-dependent, conservative-lower-bound* data
point — not the headline.

## What the ~1.4–1.9 s floor is (don't misattribute)
Mostly the **conservative single-GPU sum-over-engines serialization (zero cross-engine overlap)** + eager per-burst
service (prefill of growing ctx + 16 decode). Graph-decode is the *secondary* (~10–30%) factor. So **sub-second is
reachable by the engine, and the engine's win is mostly genuine cross-engine overlap + async continuous batching, with
graph-decode secondary** — not by "turning on graphs."

## The decomposition is now CLOSED (this is the stop signal)
- **residency / memory:** not the limit (prior capstone 96c7e8c — 5 models fit, cold-miss 0, KV fits, KL=0).
- **throughput:** not the limit (this run — engines far below the 8808 peak; robust, model-independent).
- **per-request latency under multi-model routing:** *is* the gap → **cross-engine overlap + async continuous
  batching (+ graph-decode secondary) = the engine.**
- the cudagraph-monitor singleton wall independently shows the engine is a real **in-process** build (CIPHER owns the
  multi-model mux), not "run N vLLM engines."

**Nothing further is measurable on this pod:** graph-decode in-process is singleton-blocked, and the async-overlap path
*is* the engine (the deferred months-build). Chasing sub-second here means building it. Correctness bar
(greedy-match KL=0 5/5) passed. **STOP for Anil's engine-build call.**
[[cipher-go1-fullbudget-100agent]], [[cipher-go1-inprocess-composition]], [[cipher-go1-held-regime-map]], [[cipher-freq-residency-build]].
