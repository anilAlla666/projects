# G-O3 scheduler STEP 0 (probe-first) — the 85% MFU headline is PHYSICALLY UNREACHABLE for the stated regime

**Date:** 2026-06-01. **Type:** READ-ONLY probe (roofline + measured; NO build, NO commit; deployed `1f305ce6`,
staging `9c318ac6` UNCHANGED, all tags stand). **STEP 0 verdict: BLOCKED-DIAGNOSED — "85% MFU via distinct-model
bursty multiplexing" conflates OCCUPANCY with MFU and is unreachable by ~60× against the HBM roofline. The build is
held; the metric is refuted, not the multiplexer.**

## The roofline (physics — the hard bound)
Decode at B=1 is HBM-bandwidth-bound: it must read all weights per token. 4-bit 7B = 3.5 GB/tok ÷ 3.35 TB/s →
**~957 tok/s ceiling = 1.35% MFU** (957 × 2·7B FLOP ÷ 990 TFLOP). 85% MFU needs **~60,000 tok/s = 63× over a hard
physical limit** — not a tuning gap, two orders of magnitude against the roofline. **Distinct models cannot batch**
(different weights → no weight-read amortization), so interleaving N of them fills occupancy and *contends for the
same 3.35 TB/s* → aggregate stays bandwidth-bound → MFU stays ~1-2%.

## Measured (converting the refutation from reasoned to measured — `pager_mfu_probe.py`, 3 distinct 4-bit 7-8B)
| | aggregate tok/s | occupancy (util%) | DERIVED MFU (tok/s·2·params/peak) | correctness |
|---|---:|---:|---:|---|
| SERIAL back-to-back | 36 | 32% | **0.05%** | — |
| CONCURRENT multiplex | 20 | 19% | **0.03%** | concurrent==solo ✓ (KL=0) |

Two findings: (1) **derived MFU 0.03-0.05%** — even below the 1.35% roofline (naive transformers `generate()` B=1 is
Python/launch-overhead-bound, far from even the bandwidth ceiling); (2) **concurrent multiplex is SLOWER than serial**
(20<36 tok/s, 19%<32% occupancy) — Python-threaded `generate()` is GIL/launch-serialized, corroborating the prior
[[cipher-smpack-sizing]] "concurrent-stream packing = 1.0×". Occupancy is reported **separately from MFU on purpose**:
occupancy can be raised (GPU busy reading weights), but MFU (compute) cannot — that gap IS the refutation. A perfect
scheduler reaches the 1.35% roofline; it is still 60× short of 85%.

## The crisp reason (advisor): distinct-model density and high MFU are OPPOSITE regimes
- **High MFU ⟺ few models × many requests each** — batchable, compute-bound, weight-read amortized = **vLLM's
  existing continuous batching**.
- **Distinct-model density ⟺ many models × few requests each** — un-batchable, bandwidth-bound, **low MFU** = what
  the pager (G-O2) bought.
**G-O3-as-specified asks for BOTH ends of the tradeoff at once** (many distinct models AND 85% compute MFU). That is
why it can't be built — not a scheduler shortcoming. The only exception: **prefill/compute-dominated** bursts (long
prompt, short output — RAG/rerank/long-context) ARE compute-bound and CAN hit high MFU; but the stated regime
("burst ~200ms then idle", generation) is decode = bandwidth-bound.

## The honest alternative metric (the reachable claim)
The bursty-agent win is NOT compute efficiency — it's **consolidation**: **N idle-heavy agents served per GPU at
correct outputs and bounded P99, where dedicated deployment needs ~N GPUs.** The metric is **agents-per-GPU at
acceptable P99** (or $/agent, or aggregate tok/s/GPU), **bandwidth-and-capacity-bound** — exactly what the pager arc
built toward. This is consistent with [[cipher-density-axis-cargo]]. **It still has to beat N orchestrated
sleep-mode vLLM instances** — the delta question left open in [[cipher-pager-delta-assessment]].

## STOP — held the build (don't-build-cargo); the decision is the metric, and it's Anil's
The MFU *framing* is refuted; the *multiplexer* may still be worth building — but for the reachable metric
(agents-per-GPU / bounded P99 / aggregate tok/s), NOT against the 85% MFU premise. Building a scheduler and reporting
"0.03% MFU" or quietly relabeling occupancy as MFU would be exactly the cargo this arc has repeatedly refused
(fusion / partial-layer / compaction). Anchors unchanged (read-only). NEXT = Anil picks the corrected metric +
whether the multiplexer is worth it given it must beat N-sleep-instances.
