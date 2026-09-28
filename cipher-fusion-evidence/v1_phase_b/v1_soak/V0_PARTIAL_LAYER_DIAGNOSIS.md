# DIAGNOSIS (probe-first, build HELD) — partial-layer residency is BACKWARDS for multi-token decode density

**Date:** 2026-06-01. **Type:** READ-ONLY diagnosis (probes + arithmetic on measured numbers; NO build, NO commit;
deployed `1f305ce6`, staging `d6dfd5a2`, all tags stand). **The probe-first discipline that held this whole arc
surfaced a fundamental problem with the partial-layer build's PREMISE before spending the build — surfacing to
Anil per "diagnose, don't just proceed."**

## The build premise ("fewer bytes per swap, fixes both walls, makes 100 reachable") is FALSE for multi-token serving
Autoregressive **decode re-reads EVERY weight on EVERY token.** With K<N layers resident, the (N−K) evicted layers
must re-page **per token**. Quantified with this session's measured numbers (Mistral-7B, 32 layers, per-layer
page_in pinned 9.6 ms / mmap-warm 51 ms; whole-model-resident B=1 decode ~30 ms/token):

| K resident | cold/tok | paged GiB/tok | +ms/tok (pinned) | +ms/tok (mmap) | decode slowdown (pinned) |
|-----------:|---------:|--------------:|-----------------:|---------------:|-------------------------:|
| 32 (whole) | 0 | 0.0 | 0 | 0 | 1.0× |
| 24 | 8 | 3.4 | 77 | 410 | 3.6× |
| 16 (N/2) | 16 | 6.8 | 154 | 819 | 6.1× |
| 8 | 24 | 10.1 | 231 | 1229 | 8.7× |

- Whole-model resident: **1× model over PCIe per REQUEST** (load once, serve all tokens at HBM ~3 TB/s).
- Partial-layer K=N/2, 20-token response: **~10× model over PCIe per request** (re-paged per token).
- **The deepest form:** B=1 decode is memory-bandwidth-bound (mostly weight reads, little compute). Serving
  cold-layer weights from PCIe (~50 GB/s) instead of HBM (~3 TB/s) is a **~60× slower weight read** for those
  layers. Prefetch can't hide it — per-layer page_in (~10 ms) ≫ per-layer B=1 compute (~sub-ms), so the forward is
  PCIe-bound regardless of overlap. PCIe is THE binding constraint ([[cipher-density-axis-cargo]], [[cipher-pager-delta-assessment]]);
  partial-layer trades 1× → many× PCIe to shrink peak HBM — **the wrong direction for the density goal.**

## Why the originally-specified gate would have passed GREEN while hiding this
Gates 1-2 (bit-identical single forward) and gate 3 ("bytes-per-eviction-decision: 1 layer = 0.42 GiB vs 13.5 GiB")
are all TRUE per decision — but a forward does (N−K) evict+pagein, and decode does a forward **per token**. The
single-forward gates pass; decode throughput silently craters. The honest gate had to measure **paged-bytes-per-TOKEN
and tokens/sec over a real generation** — which predicts the collapse above.

## Where partial-layer genuinely wins (so the mechanism isn't wrongly dismissed — just correctly scoped)
- **Single-forward / prefill-only models**: embeddings, rerankers, classifiers, safety guards (1 forward, no decode
  re-reads) — partial-layer + peak-HBM headroom is a real win there.
- **Cold-start TTFT** via prefetch-overlap (first token only; you'd want whole-model resident for the decode that
  follows).
- **Peak-HBM headroom** during a cold model's first forward.
- **NOT** multi-token decode density — which is the "100 bursty agents" goal (agents generate, i.e. decode).

## The actual decode-density lever: INT4 (correcting the prior "INT4 + partial-layer" co-equal framing)
INT4 weight quantization reduces resident bytes **~4× PERMANENTLY**, weights stay in **HBM** (full bandwidth), with
**ZERO per-token re-paging** — it attacks the same RAM-ceiling AND PCIe walls without partial-layer's decode penalty.
CIPHER already has the INT4 GEMM path (Marlin, `w2-marlin-bf16-machete-full` / `graph-gate-step1`). **INT4 weights +
the proven co-residence pager = ~4× more models resident, no decode penalty = the real "toward 100."** Partial-layer
is situational (prefill/cold-start); INT4 is the lever. The prompt's "prefetch, then INT4" sequencing is likely
backwards for the density goal.

## Mechanism soundness (so the held build is a known-good option, not a dead end)
The partial-layer mechanism itself is sound and cheap to build IF a prefill/cold-start use-case is the target:
reuse the co-residence manager with per-layer sub-regions; torch forward-hooks (pre: serve_demand, post: serve_end)
drive paging (torch API, substrate-line-OK like MemPool); embed/lm_head pinned via a permanent ref; cuCtxSync-per-
evict is correct-but-serializing; mmap-warm is **bit-identical at 10.3 GB/s** (probed, `pager_mmap_probe.c`).
Caveat: forward-hooks don't fire inside CUDA-graph-captured forwards (eager transformers is the vehicle).

## RECOMMENDATION (Anil decides the sequencing)
For the **100-bursty-agent decode-density goal**: build **INT4-weights-in-the-pager** next (4× density, no decode
penalty), NOT partial-layer. Build partial-layer only if a **prefill/embedding/cold-start** workload is the target,
and only with the corrected decode-cost gate. Held the build to surface this rather than spend it on a misframed
premise.
