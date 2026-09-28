# CP 2.4 — Marlin hang: root cause + fix (audit trail)

**Date:** 2026-05-15 → 2026-05-16. **Status:** root cause confirmed by
instrumentation; **Fix A applied and verified.**

When kmod 0.4.8 restored `/dev/cipher` access (the CP 3.3 regression fix —
`../REGRESSION_AUDIT_2026_05_15.md`), every `CIPHER_MARLIN=on` workload began
to hang. This is the diagnosis trail and the fix.

---

## 1. Symptom

Mistral-7B B=1 decode, `CIPHER_MARLIN=on`, substrate active: the workload
hung at the 3rd weight Marlin quantized. gdb showed the main thread in a
synchronizing CUDA runtime call inside `cipher_rt_marlin_engine_quantize_repack`
(`cuMemcpyDtoH_v2`, later `cuMemFree_v2` after a partial fix); GPU pegged at
100 %; decode loop never reached. CP 2.4 test A had shipped Marlin at
1.5–1.9× — but test A ran **pre**-kmod-fix, when `/dev/cipher` was EPERM.

## 2. Three hypotheses — two disproved, one confirmed

**H1 — Marlin × v2 substrate (GREEN/ARB) interaction.** Inferred from one
isolation point (`MARLIN=off`+substrate-on completes; `MARLIN=on`+substrate-on
hangs). **DISPROVED:** `MARLIN=on` + `CIPHER_SUBSTRATE=off` still hung, and the
green context initialises independently of the ARB/SMP/PR substrate gate.

**H2 — cuBLAS-shim reentrancy.** Marlin's quant kernels re-entering the
`.symver` cuBLAS shim → recursive Marlin-handling. **DISPROVED:** the gdb
backtrace shows a *single* `cipher_rt_matmul_dispatch` frame, and
`cipher_rt_marlin_engine.cpp` shows the quant path
(`quantize_fp16_to_int4_groupwise_gpu`) launches NVRTC-compiled quant kernels
via `cuLaunchKernel` — it never calls cuBLAS.

**H3 — Marlin GEMM kernel in the green context.** **CONFIRMED by an
instrumented build** (`b1a3424c`, diagnostic; see
`MARLIN_HANG_INSTRUMENTATION.md`). Per-launch logging showed:

```
GREEN: green context bound ... with 8 SMs (CUcontext=0x587b98689cf0)
marlin.gemm M=1 N=4096 K=4096  grid=(132,1,1) ... ctx=0x587b98689cf0  in_marlin_quant=0
```

The Marlin GEMM kernel — `grid = SM count = 132`, persistent-style, with
inter-CTA split-K coordination through the `locks` buffer — was launched into
the **green context (8 SMs)**. 132 blocks × 96 KiB shared mem ⇒ only ~16
co-resident; 116 cannot schedule; the inter-CTA `locks` protocol deadlocks
(resident CTAs spin on signals from CTAs that can never run). GPU pegs 100 %
forever; the next device-wide synchronizing call hangs behind it.

**Mechanism — why the kmod fix exposed it.** Pre-kmod-fix `/dev/cipher` was
EPERM → CUPTI could not subscribe → its per-launch callback never ran → the
T4.2.4d green-context enforcement (`green_ctx_make_current`) never fired →
`ctx_swaps_to_green = 0` → Marlin's kernels ran in the primary context on all
132 SMs. test A passed for exactly that reason. kmod 0.4.8 (correct, adjudicated)
let CUPTI subscribe; its callback now makes the green context current on
every `cuLaunchKernel`, so the Marlin GEMM lands in the 8-SM partition.

This is an **architecture-level incompatibility**, not a coding bug — see
`PHASE_5_MARLIN_PARTITION_CONSTRAINT.md`.

## 3. Fix

The defect: Marlin's engine GPU work (quant kernels via driver-API
`cuLaunchKernel` + the GEMM kernel) ran in whatever context the CUPTI callback
last made current — the green context. The fix pins it to the primary context.

- **Fix (i)** (`caae0bb8`) — `PrimaryCtxGuard` (RAII) wrapping the *quant*
  path (`ensure_weight_quantized_repacked`): saves the caller's context, makes
  the device-0 primary context current, restores on exit; a thread-local
  `cipher_rt_in_marlin_quant` flag makes the CUPTI callback skip
  green-enforcement for Marlin's own launches. **Incomplete** — it pinned the
  quant kernels (instrumentation confirmed `ctx=<primary>`) but not the GEMM
  dispatch; the hang moved (`cudaMemcpy`→`cudaFree`) but persisted.
- **Fix A** (final) — extend the *same* `PrimaryCtxGuard` to
  `cipher_rt_marlin_engine_dispatch`, so the GEMM launch is pinned too. All
  Marlin engine GPU work now runs in the primary context (132 SMs). Single
  mechanism, scope widened. Files: `cipher_rt_marlin_engine.cpp` (guard +
  dlsyms for `cuCtxGetCurrent`/`cuCtxSetCurrent`/`cuDevicePrimaryCtxRetain`),
  `cipher_cupti.c` (the `cipher_rt_in_marlin_quant` check).

## 4. Verification (Fix A, libcipher_rt `5e304549`)

| Check | Result |
|---|---|
| Smoke — `CIPHER_MARLIN=on`, substrate ON | **PASS** — decode completes; instrumentation confirmed `marlin.gemm ... ctx=<primary> in_marlin_quant=1`; 225 weights quantized |
| Post-kmod-reload Fix A smoke | **PASS** — decode completes, 0 ENOSPC, GEMM in primary ctx |
| Clean rebuild (instrumentation stripped) smoke | **PASS** — `5e304549`, decode completes, 48.8 tok/s |
| CP 3.3 gate (post-reload) | **PASS 4/4** (r_util 0.9808, ratio 1.162) |
| Regression Checks 1/3 (post-reload) | **PASS** — kvdedup (a)(b)(d), alloc_unit_test 14/14 |

test A's multi-tenant N-sweep retest did **not** cleanly complete — it hit an
unrelated kmod SM-partition-allocator exhaustion (`ENOSPC`) after this session's
25+ substrate-on runs; tracked in `../PHASE_5_ALLOCATOR_DIAGNOSTIC.md`. The
N-sweep was a sanity check, not a gate criterion — the CP 2.4 gate is
single-tenant. Fix A is verified by the single-tenant smokes above.

## 5. libcipher_rt anchor chain

| md5 | role |
|---|---|
| `a0d6cddacb116f2d51ad1d1f867ef564` | original (test A baseline, pre-kmod-fix); rollback |
| `55e2e3233c3b92506d1dd2f573b852c4` | substrate-gate diagnostic (H1, Option B — failed) |
| `caae0bb88650c1121b9f92b1eda00e22` | fix (i) — quant-path pin only (failed; hang moved) |
| `b1a3424c…` | instrumented diagnostic (H3 confirmation; log in `MARLIN_HANG_INSTRUMENTATION.md`) |
| `c63b6abc4c1b2fcce7641c11b01c9e40` | instrumented Fix A (quant + dispatch guard) |
| **`5e304549a7e8a82b39072d11a85e71fb`** | **clean Fix A — new campaign anchor** |

All preserved as `libcipher_rt.so.*` except `b1a3424c` (its diagnostic value
is the captured log, not the binary). kmod 0.4.8 `e2f50452` and libcipher_v2
`86618c30` held throughout (kmod reloaded once for allocator hygiene — same md5).

## 6. Bottom line

Root cause: the Marlin GEMM kernel is structurally full-GPU and deadlocks in a
green-context SM partition; the kmod 0.4.8 `/dev/cipher` fix exposed it by
activating the CUPTI-driven green-context enforcement. Fix A pins all Marlin
engine GPU work to the primary context. **The Marlin sub-task is closed
unconditionally** for the single-tenant CP 2.4 scope. Marlin + SM-partitioning
is a Phase 5 architectural problem — `PHASE_5_MARLIN_PARTITION_CONSTRAINT.md`.
