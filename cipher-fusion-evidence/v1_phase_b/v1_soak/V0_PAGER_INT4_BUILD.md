# BUILD — INT4 weights in the pager (ORCHESTRATE G-O2, the decode-density lever, pager STEP 3)

**Date:** 2026-06-01. **Type:** BUILD via integration (NO new substrate — the pager is precision-agnostic, pages
bytes). deployed `1f305ce6` UNCHANGED; staging `.so` **UNCHANGED at `d6dfd5a2`** (no cipher_rt_phase4 source
change); cipher_rt_phase4 stands at `dea0fd9`. Builds on `pager-step2arch-coresidence`. **Result (lead with the durable, non-contingent win): INT4 weights stay
HBM-resident with ZERO per-token re-paging — the thing partial-layer structurally could NOT do (it re-paged 6-28×
per decode). 3 distinct 7-8B models, 4-bit (bnb NF4), co-resident behind the proven pager; pager-correctness KL=0
(byte-identical round-trip + forward bit-identical, all 3). Density: CURRENT (as-implemented) ~10-12 models in 80 GiB
(1.6-2.2× physical); 2.8× / ~15 is the post-compaction TARGET. NF4 quality indistinguishable from fp16 on short
prompts (KL 0.0085 nats); tail/long-context quality UNMEASURED.**

**Does NOT close the delta-vs-N-sleep question:** N sleep-mode vLLM instances can also run INT4 — INT4 is an
orthogonal density multiplier BOTH sides get. The pager's edge over N-sleep stays co-residence-in-one-process +
copy-free evict ([[cipher-pager-delta-assessment]]); this build does not re-open that as a moat.

## The key structural result: INT4 needed NO new pager code
The pager pages bytes; 4-bit weights are just 4× smaller bytes. The co-residence manager + per-region eviction-
during-use coherence + MemPool routing + reclaim all work unchanged. This build is integration + measurement, which
is *why INT4 is the right lever* — 4× density on machinery already proven, not new substrate. The `.so` is byte-
identical (`d6dfd5a2`), so OFF byte-identical holds by construction.

## Probe-first (binding): do bnb 4-bit weights land in the MemPool?
`pager_int4_probe.py`: TinyLlama 4-bit inside `use_mem_pool` → 1.13 GiB captured (≥ 0.75 GiB packed), forward works
from pool-resident 4-bit weights → **bnb Params4bit allocates through the torch allocator MemPool captures.** Yes.

## Two walls diagnosed + FIXED (no scope-down)
1. **Quantize-on-load fp32 transient × bump-allocator → region OOM.** transformers' `_initialize_missing_keys` does
   `init.normal_(weight.float())` — a 2 GiB fp32 temp for vocab-heavy embeds — and the pager's bump allocator
   doesn't reclaim it, so the region bump overflowed the reserve (Qwen2 OOM). **Fix:** pre-quantize + `save_pretrained`
   once, load the COMPLETE 4-bit checkpoint (no missing-key init) → bump tracks the real 4-bit footprint. Also the
   production-realistic path (ship quantized checkpoints, don't quantize-on-every-load).
2. **transformers 5.8.1 threaded materialize × thread-local `use_mem_pool` → params escaped (residency 0/291).**
   `core_model_loading.py:1286` materializes via `ThreadPoolExecutor`; `use_mem_pool` is thread-local, so worker-
   thread allocations miss the pool. **Fix:** `HF_DEACTIVATE_ASYNC_LOAD=1` (`core_model_loading.py:1280`) → main-
   thread materialize → params route to the region (100% residency). (+ vocab-embed `.float()` transient is now
   in-pool on the main thread → reserve +3 GiB headroom.)

## CORRECTNESS GATES (`pager_int4_serve.py`) — ALL PASS
- **G-cores:** 3 distinct 4-bit models 100% params in-region (291/291, 339/339, 291/291), disjoint VAs
  (`0x302000000` / `0x50e000000` / `0x782000000`).
- **G-correct (pager-correctness, KL=0):** per model, 4-bit weights **byte-identical across evict→pagein**
  (live_cksum match) + forward **bit-identical** (max_abs_diff 0.00, det baseline first). The pager preserves the
  quantized weights exactly.
- **G-reclaim:** `page_out` frees **8.18 GiB physically** (mem_get_info; the mapped region = the reserve).
- **Co-residence churn (budget<sum):** every served forward bit-identical, wasted=0 — eviction at INT4 coherent.

## INT4 QUALITY (gate 2, INDICATIVE — `pager_int4_quality.py`, lossy-bounded, NOT a KL=0 claim)
NF4-runtime, Mistral-7B: **mean per-token KL(fp16‖nf4) = 0.0085 nats** (per-prompt 0.006-0.011); **perplexity
fp16 13.36 / nf4 13.11 (0.98×)** — essentially indistinguishable on these prompts. **Hardened caveat (advisor):
INDICATIVE on 4 SHORT prompts only — 4-bit damage concentrates in the long tail (rare tokens, long context,
reasoning chains) which this does NOT surface; PPL dropping 0.98× is noise, not improvement.** Labeled NF4-runtime;
the production number is **CIPHER-Marlin-format quality**, and **bnb NF4 ≠ Marlin format** → "INT4-in-pager proven
with bnb" does NOT transfer the SERVING-KERNEL path to Marlin for free (the pager holds either — just bytes; the
Marlin serving + quality eval is the open integration).

## DENSITY MEASURED (honest, two numbers)
- **Tight 4-bit (ckpt footprint): 5.07 GiB avg vs fp16 14.2 GiB = 2.8× smaller → ~15 models fit in 80 GiB vs 5
  (fp16); 2.8× less PCIe per swap; ZERO per-token re-paging.** The toward-100 step (INT4 alone; <4× because vocab-
  heavy embeds/lm_head stay fp16 — Mistral 3.3×, Qwen2/Llama ~2.6× on 152k/128k vocabs).
- **As-implemented region (reserve physical): ~6.6-8.2 GiB (≈2.2× / 1.6×)** — diluted by (a) the no-reclaim bump
  allocator keeping the load-transient, (b) the +3 GiB reserve headroom for that transient. The gap to tight is an
  allocator artifact, not fundamental.

## Anchors / next op
deployed `1f305ce6` UNCHANGED; staging `d6dfd5a2` UNCHANGED; cipher_rt_phase4 `dea0fd9` (no source change — INT4 =
integration). Substrate-line: cuMemMap region + MemPool routing, no framework source patch (the load tweaks are env
+ pre-quant, not source edits). **NEXT OP:** (a) **reclaim/compaction** — a free-list or post-load region-shrink to
close the as-implemented→tight gap (the bump allocator's STEP-1.5 simplicity now limits density); (b) **CIPHER-Marlin
-native 4-bit format** (the production serving kernel + the real quality number); (c) **PCIe predictive prefetch**.
[[cipher-pager-coresidence-build]], [[cipher-partial-layer-diagnosis]], [[cipher-pager-delta-assessment]].
