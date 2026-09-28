---
name: Instance 4 build plan — Graph-captured block at fp16 (CURRENT PRIORITY)
description: Phase-by-phase build plan for Instance 4 (graph-captured block) at the stable envelope — batch 32–256 fp16 on H100; this is the load-bearing instance after the 2026-04-05 plan stabilization
type: project
---

**This is the active plan as of 2026-04-05 end of session.** Instance 4 (graph-captured block, fp16) is the primary lever for reaching 85% MFU at batch=256 on H100. Build order: **4 → 3 → 1 (keep), optional 2' sparsity**.

## Why Instance 4 first (stable rationale)

At batch=256 fp16 on H100, the theoretical MFU ceiling is 86.7% (physics). Stock PyTorch is predicted to hit ~55% — a ~32 pp gap composed mostly of launch overhead, kernel scheduling gaps, and framework dispatch. Instance 4's graph capture directly collapses 15–25 pp of that gap by replacing ~900 individual kernel launches per decode step with a single `cuGraphLaunch`. No other instance attacks that specific cost.

The Phase 4.0 pool-observer finding (allocator interceptor doesn't help at batch=1 because PyTorch's caching allocator internalizes decode-step demand) is **much less constraining** at batch=256 because:
- Fewer small kernels fire per token when the batch is large
- Address variance matters less when we re-capture per-step boundary
- At batch=256 the cost of re-capture is amortized across 256 sessions

## Phase structure with gates

### Phase 4.0 — Continuous-batching measurement harness (prerequisite, zero C++)
**File**: `tests/bench_continuous_batching.py` (NEW)

- PyTorch harness simulating continuous batching at batches 32, 64, 128, 192, 256
- Measures: tokens/sec, **MFU against fp16 peak**, HBM bandwidth utilized, launch-overhead fraction, per-GEMM-shape time breakdown
- Runs stock PyTorch (no CIPHER active) to establish baselines
- **Gate**: baseline table produced. Predicts ~55% MFU at batch=256 and ~10–15% at batch=32. If measured baseline already >80% at batch=256, the plan changes (unlikely); if ~55%, commit to Instance 4.
- Scope: ~150 lines Python, ~5 min runtime, zero C++ changes.

### Phase 4.1 — Persist-detector + fingerprint-aligned capture trigger
**Files**: extend `src/cipher_persist.cpp` (already has fingerprint detection + tandem-repeat detector)

- Reuse the existing stable-block detection (promote on 10 repeated tandem repeats at L ∈ {4, 8, 16})
- Extend `StableBlock` struct with `CUgraphExec exec` + `bool captured` fields
- On promotion, attempt `cuStreamBeginCapture` → `cuGraphInstantiate` → attach to StableBlock
- On subsequent fingerprint hits, call `cuGraphLaunch` once instead of N individual launches
- **Gate**: persist detector correctly identifies the TinyLlama decode step signature under continuous batching at batch=256; capture succeeds on ≥80% of promoted blocks.

### Phase 4.2 — Capture safety: verification + fallback
**Files**: extend `src/cipher_dispatch.cpp`, new `src/cipher_graph_verify.cpp`

- On first capture of a block, run one eager replay in parallel, bit-compare output tensors against captured replay
- Sentinel cadence: every N=100 captured replays, re-run one step eagerly and bit-compare
- On any mismatch: invalidate the StableBlock, revert to per-kernel fast path, never retry capture on that signature
- **Gate**: zero correctness failures over 10,000-replay stress run; token IDs bit-match eager across 500 decode steps.

### Phase 4.3 — End-to-end MFU at batch=256 (THE INTERMEDIATE GATE)
Run `tests/bench_continuous_batching.py` with CIPHER + Instance 4 active on TinyLlama H100 at batch=256.

**Gates**:
- **Gate A (primary)**: MFU ≥ **75%** at batch=256 fp16 (Instance 4 alone closes ~2/3 of the overhead gap)
- **Gate B (accuracy)**: token ID overlap ≥99% vs eager fp16 over 500-token rollout
- **Gate C (regression)**: `tests/test_hw_validation.py` 7/7 PASS; `cipher_demo.py` M=1 ≥ 1.46×
- **Gate D (scaling)**: MFU tracks theoretical ceiling within 7 pp across batches 32, 64, 128, 192, 256

If Phase 4.3 passes → Instance 4 is shipped; proceed to Instance 3.
If Phase 4.3 fails → hard stop, diagnose, no weakening.

### Phase 4.4 — Handoff to Instance 3
Instance 3 (fused megakernel) picks up the remaining ~10 pp gap to reach 85%. Reuses Instance 4's captured-block infrastructure as the substrate — megakernels are inserted into the same captured graph for attention+MLP+norm fusion.

## What stays from earlier work

All Phase 4.0 observation-mode code on disk is REUSED by Instance 4, not discarded:

| File | Role in Instance 4 |
|---|---|
| `src/cipher_pool.cpp` (observation-only) | Measurement infrastructure for capture-time tensor addresses |
| `include/cipher_pool.h` | Public API (still observation-only in Instance 4) |
| `tests/test_pool_observation.py` | Debug tool for diagnosing capture failures |
| `tests/test_pool_interposition_check.py` | Sanity probe |
| `tests/analyze_pool_observation.py` | Gate analyzer pattern (copy for new Phase 4.3 gate) |

None of the 2026-04-05 pool code needs to change. The allocator-interceptor hypothesis was wrong but the instrumentation is useful.

## After Instance 4 lands

- **Instance 3** (fused megakernel) — second priority, closes the remaining gap to 85%
- **Instance 1** (Koopman) — kept as-is for structured shapes
- **Optional Instance 2'** (2:4 sparsity) — only if user approves; unlocks batch=128 gate point
- **Deferred Instance 2** (quantization tiers) — post-session-7 expansion, never load-bearing

## Open items pending user approval before build

1. Go on Phase 4.0 build (the measurement harness). Zero C++, ~150 lines Python.
2. 2:4 sparsity approval (affects later phases, not Phase 4.0 itself).
3. Nothing else blocks.

## How to apply when resuming

- Start at Phase 4.0 — measurement harness first. Every subsequent phase depends on the baseline it produces.
- Do NOT skip to Phase 4.1 without the baseline table in hand.
- Do NOT re-introduce quantization into the load-bearing path.
- The envelope is STABLE — do not propose another reset unless the user explicitly challenges it.
