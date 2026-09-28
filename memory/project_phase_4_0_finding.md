---
name: Phase 4.0 pool-observer finding (historical) — allocator interceptor hypothesis falsified at batch=1
description: The 2026-04-05 Phase 4.0 observation-mode result that killed the allocator-interceptor approach under the original batch=1 envelope; code on disk is reused by the new Instance 4 plan, not discarded
type: project
---

## Status as of 2026-04-05 end of session

**Historical finding, still valid, no longer blocking.** Instance 4 (graph-captured block) has returned to FIRST priority under the corrected envelope (batch range 32–256 fp16 on H100), but via a different mechanism than the original plan. See `project_instance_4_plan.md` for the active plan.

## What Phase 4.0 measured

Observation-mode allocator interceptor hooks on `cudaMalloc / cudaFree / cudaMallocAsync / cudaFreeAsync`. Ran 20 TinyLlama decode steps on H100 at **literal batch=1**.

**Interposition works correctly**: weight-load sanity check produced 24 cudaMalloc entries (1 × 2.2 GB arena + 23 × 2 MB caching segments).

**During the 20-step decode window at batch=1**: zero cudaMalloc, zero cudaFree, zero cudaMallocAsync, zero cudaFreeAsync. Only step-marker entries. PyTorch's `c10::cuda::CUDACachingAllocator` serves 100% of decode-step tensor demand from pre-warmed 2 MB segments. The libcudart boundary is never crossed during decode.

## Interpretation (still valid)

The 27.7% params[0] address variance observed in the earlier Change 6 diagnostic does NOT originate at libcudart. It originates inside `c10::cuda::CUDACachingAllocator` — suballocation of offsets within the 2 MB caching segments, handed out by LIFO/LRU logic in libtorch's C++ code.

An LD_PRELOAD interceptor on libcudart cannot see, control, or fix this. The original allocator-interceptor hypothesis was wrong **at batch=1**.

## Why this no longer blocks Instance 4 under the new envelope

At batch=256 fp16 (the stable primary gate point):
- Fewer small kernels fire per forward pass per effective token (because the GEMMs are big enough to dominate)
- Graph capture can re-capture per-request boundary without high overhead
- Address variance matters proportionally less — 256 users amortize one re-capture
- The Phase 4.0 finding (PyTorch's allocator internalizes decode-step demand) is actually HELPFUL at batch=256 because it means the allocator is deterministic at the PyTorch level for stable request sizes, even if the underlying pointers drift

The Instance 4 plan in `project_instance_4_plan.md` uses **fingerprint-detected persistent kernel sequences + cuStreamBeginCapture at block boundaries**, NOT a custom pool underneath PyTorch. The allocator interceptor is observation-only and reused as measurement infrastructure.

## Code state on disk (unchanged, observation-only, zero regression, REUSED)

| File | Role in the new Instance 4 plan |
|---|---|
| `include/cipher_pool.h` | Public observation API (unchanged) |
| `src/cipher_pool.cpp` (~170 lines) | Ring buffer + dump (unchanged) |
| `src/cipher_intercept_cudart.cpp` (+80 lines) | cudaFree/cudaMallocAsync/cudaFreeAsync shims + resolve_allocator_reals() |
| `exports.map` (+12 symbols) | cudaMalloc/cudaFree/async + cipher_pool_* |
| `Makefile` (+4 lines) | HOOK_SRC/HOOK_OBJ wiring |
| `tests/test_pool_observation.py` | Debug harness for capture diagnostics |
| `tests/test_pool_interposition_check.py` | Interposition sanity probe (confirmed working) |
| `tests/analyze_pool_observation.py` | Gate analyzer pattern — copy for Phase 4.3 |
| `/tmp/pool_obs.log` | Historical evidence (21 step-marker entries, 0 alloc/free during batch=1 decode) |
| `/tmp/pool_interp_check.log` | Historical evidence (24 weight-load entries confirming interposition) |

Build: `libcipher_hook.so` at 58K, clean link, zero errors, zero regressions. TinyLlama decode runs to completion with coherent output when CIPHER_POOL_OBSERVE=1.

## How to apply when resuming

- Do NOT delete Phase 4.0 code. It's reused by Instance 4.
- Do NOT re-run the Phase 4.0 falsification at batch=1 — it's already done and the result is known.
- DO re-run similar observation mode at batch=256 if Phase 4.1 capture runs into allocator-related issues (a different measurement, not a re-falsification).
- The new Instance 4 plan is in `project_instance_4_plan.md`. Start at Phase 4.0 of that plan — the continuous-batching measurement harness — which is a different Phase 4.0 than this one.
