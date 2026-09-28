# STEP 4 — pager reclaim/compaction: INVESTIGATED → compaction substrate NOT required (don't-build-cargo)

**Date:** 2026-06-01. **Type:** probe-first investigation → load-level fix. **What STEP 4 actually ships: the diagnostic instrumentation
that PROVED the compaction unnecessary (a substrate change), + a load-level reserve fix. The compaction MECHANISM
itself was investigated and REJECTED — no compaction/reclaim logic was written.** deployed `1f305ce6` UNCHANGED;
staging `.so` `d6dfd5a2`→**`9c318ac6`** (the cur_live/peak_live diagnostic — its standing value: it discriminates
holes=0 and lets a future two-pass reserve read peak_live); cipher_rt_phase4 `dea0fd9`→new. **Result: the density gap was
NOT an allocator flaw — it was my own over-reservation (a load-level reserve formula), reclaimed with ZERO
substrate-compaction and ZERO coherence risk: ~8-9 → ~11 models/80GiB. A free-list / copy+remap compaction would
have been cargo.**

## Probe-first refuted the compaction premise (the central finding)
Added live-tracking instrumentation (`cur_live`/`peak_live`, additive, OFF byte-identical) to cipher_rt_pager.c
and measured a real 4-bit load: **`cur_live == bump == peak_live`, holes = 0** for both Mistral and Qwen2. So:
- There are **no freed-allocation holes** within `[0,bump)` for cipher to reclaim → a free-list/reclaiming allocator
  buys **nothing**. (Confirmed structurally: torch's CUDACachingAllocator caches freed blocks ABOVE the pluggable
  allocator, so `cipher_pager_malloc` only ever sees torch's net peak-simultaneous-live demand → bump == peak-live
  regardless of any cipher free-list.)
- The only cipher-reclaimable waste was the **unmapped padding** (`reserve − bump`), which was **mostly my own
  `+3 GiB` reserve formula**, not a structural flaw. `bump = ckpt + the fp32 embed-init transient (vocab·hidden·4)`
  — verified to nail both models (Mistral 4.15+0.49=4.64 vs measured 4.65; Qwen2 5.45+2.03=7.48 vs 7.52).

## The mechanisms — ruled out, with reasons (advisor + a 5-agent adversarial workflow)
- **copy+remap tail-shrink (the prompt's recommended-candidate): REJECTED as cargo.** It reaches the exact bump
  vs a computed tight reserve's bump+margin — a **~6% edge** — at the cost of a substrate change to the one module
  whose eviction-during-use coherence took five builds to prove. Not worth it. (It is load-phase-safe and would
  work; it's just unnecessary.)
- **free-list: REJECTED (useless).** torch caches above the pluggable allocator (see above).
- **shrink-to-live / sparse per-granule unmap: REJECTED.** It converts the single coherence-proven
  `cuMemUnmap(va, size)` at the PAGING_OUT boundary into selective per-granule unmaps — modifies the exact code
  proven under 700 evicts × 11147 reads. Coherence outranks the marginal density.
- **load-avoid (kill the embed `.float()` transient): the ONLY thing that reaches the ~15 target, but it requires a
  transformers init patch** (`modeling_utils.py:2374`; there is NO config flag / `_fast_init` / `assign` mode that
  skips it) → **VIOLATES substrate-line (Mem #24, no framework source patch).** Out of scope.

## The FIX delivered (load-level, zero substrate-compaction): computed tight reserve
`reserve = ckpt_size + vocab·hidden·4 (the fp32 embed transient) + 384 MiB`, replacing the `+3 GiB` pad. This hugs
the bump → reclaims the padding upfront (no tail-shrink, no copy, no remap, no coherence interaction). Measured:
**physical region (reserve) 6.94 GiB avg vs fp16 14.2 GiB = 2.0× → ~10-11 models/80GiB, MODEL-MIX-DEPENDENT
(Mistral 4.65 GiB ≈ 17 individually; Qwen2/Llama ~7.6 GiB ≈ 10) — not 11-of-a-kind; was ~8-9 at the +3GiB pad.
reserve hugs bump 6.61 GiB (padding reclaimed).**

## Correctness + full regression (Mem #11) — ALL PASS (the substrate touch is diagnostic-only)
- **INT4 gate (pager_int4_serve.py):** G-cores 100% residency (291/291, 339/339, 291/291) disjoint VAs; G-correct
  KL=0 (4-bit weights byte-identical across evict+pagein + forward bit-identical max_abs_diff 0.00, all 3);
  G-reclaim frees 5.01 GiB physical (tight, vs 8.18 padded); co-residence churn bit-identical wasted=0.
- **Regression (instrumentation touched malloc/free + the struct):** test_pager GATE1/2 PASS; test_residency GATE-A
  PASS (coherent/pressured/budget_ok/livelock_free/no_phys_leak) + livelock negative-control DETECTED.
- **OFF byte-identical:** Marlin graph-gate vs `9c318ac6` passes identically (instrumentation additive).

## Honest residual + the real ceiling (surfaced for Anil)
Reserve→ckpt gap (~11 → ~15) = the fp32 embed-init transient — **NOT a bump-allocator hole** (holes=0 measured);
closing it needs a transformers init patch (violates substrate-line) OR a two-pass/measured reserve. **Deeper
ceiling (advisor):** the vocab-heavy embed/lm_head stays **fp16** (not quantized), which is *also* why density caps
at ~2.6× (ckpt-tight) not 4× — **quantizing or sharing the embed is a separate lever bigger than the transient**,
and it's the real path toward higher N for vocab-heavy models. Neither is compaction.

## Anchors / next op
deployed `1f305ce6` UNCHANGED; staging `9c318ac6` (live-tracking diagnostic only — no compaction logic);
cipher_rt_phase4 new commit (instrumentation). Substrate-line held (no framework patch; the fix is a load-level
reserve formula).

**The two residuals are ONE root cause (advisor): the unquantized fp16 vocab embed.** It is both (i) the reserve→ckpt
transient (its fp32 init temp) AND (ii) the 2.6×-not-4× ceiling (it stays fp16 at rest). A quantized/shared embed
shrinks BOTH (smaller at rest AND smaller init temp) → it subsumes the transient question. So frame the next-op as
**one decision about the embed**, not three scattered options. The two-pass reserve is ~1 model of polish; the embed
lever is the structural one. **And note: "toward 100" on 7-8B models is gated by (the embed lever) + (the
delta-vs-N-sleep question left open in [[cipher-pager-delta-assessment]]) — NOT by more pager substrate.** Other
next-ops: PCIe predictive prefetch; CIPHER-Marlin-native 4-bit serving kernel. [[cipher-pager-int4-build]],
[[cipher-pager-coresidence-build]].
