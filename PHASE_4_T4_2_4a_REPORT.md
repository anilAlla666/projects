# Phase 4 T4.2.4a — Green Context scaffold + cuStreamCreate interception

**Date:** 2026-05-14 morning session (continuing).
**Build:** `libcipher_rt.so.v0.2.0_T4_2_4a` md5 `77589b98c8147ba0f9385cc2bae27074`.
**Kmod substrate:** 0.4.5 (srcversion 4618FD1FC28BEE5EBC32944, B10 FIT_HINT).
**Source tarball:** `cipher_rt_phase4_src_T4_2_4a.tar.gz` md5 `6875f6af66a1611f2e4929c6641de080`.
**Pre-T4.2.4a artifact saved:** `libcipher_rt.so.v0.2.0_T4_2_3_B7_v2_B10` md5 `5d0f200c75c179040b57f4ebcd293500`.

## What shipped — minimal plumbing layer

T4.2.4a is the **plumbing layer**, NOT the lift-mechanism layer (which is T4.2.4b).

Two new files in cipher_rt_phase4:
- `cipher_rt_green_ctx.{c,h}` — lazy per-process CUDA Green Context covering ALL 132 SMs of device 0 (no real partition restriction yet). API: `cipher_rt_green_ctx_ensure()` (lazy create, idempotent), `_push()` / `_pop()`.

Wired into cipher_cupti.c via CUPTI:
- New cbids enabled: `CUPTI_DRIVER_TRACE_CBID_cuStreamCreate`, `CUPTI_DRIVER_TRACE_CBID_cuStreamCreateWithPriority`.
- API_ENTER: `cipher_rt_green_ctx_ensure()` + `_push()` — installs green_cuctx as current before the underlying cuStreamCreate executes.
- API_EXIT: `_pop()` — restores prior context.

Net effect: any CUDA stream created after the first stream-create observation is bound to the green context. With v1's full-SM green ctx, this has no behavioral impact (the kernel runs on all SMs either way) — but the binding is in place for T4.2.4b to swap in a partition-restricted green ctx.

## Verification

### Functional gates (run twice — once before kmod reload, once after; both PASS)

| Gate | Result |
|---|---|
| GREEN init log line at first cuStreamCreate | `GREEN: device 0 has 132 SMs available (full set, v1 no partition)` + `GREEN: green context created — sm_count=132 CUcontext=0x...` |
| WL01 60s smoke tokens | 4773 tokens / 60.13 s = 79.4 tok/s |
| WL01 600s run produces progress file with end marker | ✅ |
| Phase 3 ABI 12/12 PASS (post-T4.2.4a smoke) | ✅ |
| Phase 3 ABI 12/12 PASS (post-T4.2.4a 600s run) | ✅ |

### WL01 600s measurement (matched-pair comparison)

Today's measurements on cipher_kmod 0.4.5:

| Run | library | TPW | tok/s | watts |
|---|---|---:|---:|---:|
| Morning baseline repeat (libcipher_v2 on kmod 0.4.4) | libcipher_v2 | 0.5720 | 81.61 | 142.78 |
| **T4.2.4a (libcipher_rt with GREEN_CTX) on kmod 0.4.5** | libcipher_rt | **0.5727** | **81.07** | **141.55** |

Δ T4.2.4a vs morning libcipher_v2 baseline: **+0.12% TPW, -0.66% tok/s, -0.86% watts**.

All deltas within the WL05 same-condition noise band (±0.53%) measured earlier this morning. **NO REGRESSION on WL01.**

### Matched-pair on same kmod 0.4.5 (binding comparison)

| Run | library | tok/s | watts | TPW |
|---|---|---:|---:|---:|
| Baseline (libcipher_v2 on 0.4.5) | libcipher_v2 | 81.55 | 141.65 | 0.5757 |
| **T4.2.4a (libcipher_rt + GREEN on 0.4.5)** | libcipher_rt | **81.07** | **141.55** | **0.5727** |
| Δ matched-pair | | -0.59% | -0.07% | **-0.52%** |

**TPW Δ = -0.52% — within the same-condition noise band of ±0.53%** measured this morning (PHASE_4_NOISE_BAND_WL05.md, applied here as the best available band). Per the binding three-band rule:
- `|Δ| ≤ 0.53%` → neutral. **T4.2.4a is neutral on WL01.**
- The Green Context binding (the only behavioral change between the two runs) is a no-op for WL01 since it covers all 132 SMs (no restriction). Result is consistent with prediction.

The matched-pair establishes that **the cuStreamCreate hook + green-ctx push/pop introduces no measurable overhead** on WL01. T4.2.4b can swap in partition-restricted green contexts without inheriting any baseline noise from the plumbing layer.

### What this build does NOT do (deferred to T4.2.4b)

- The green context is created with the **full SM set**, not a per-tenant SM subset. So while streams are bound to the green ctx, kernels still run on all 132 SMs.
- No use of `cuDevSmResourceSplitByCount` to partition. The "Step 2" partition split is the T4.2.4b deliverable.
- ARBITRATE's granted mask (now populated by B7+B10 fair-share) is NOT consumed by the green ctx layer. The mask-to-SM-set translation is T4.2.4b.

T4.2.4a's value is **architectural**: every stream is now passing through our binding layer, ready for T4.2.4b to swap in restricted partitions.

## What's verified about the cuStreamCreate hook

The hook fires correctly:
- WL01 smoke shows GREEN_CTX init lines appearing **after** ARB cold-path grants (because GREEN_CTX is lazy — only initialized at first cuStreamCreate observation).
- Smoke run produces same token count as baseline (~80 tok/s).
- No new crashes / WARNs in dmesg.

What's NOT independently verified:
- That the new streams are actually bound to the green ctx (no inspection of stream→context mapping was performed; would require either cuStreamGetCtx telemetry or a partition-restricted v2 build to observe).
- Behavioral correctness under cuStreamDestroy (we don't hook destroy; if green_cuctx is the only context an app stream depends on, destroying our green_ctx at process exit could leave dangling streams. v1 leaves the green_ctx alive for process lifetime — same pattern as the pci_dev_put leak in cipher_bar0_exit for analogous reasons).

These verifications are T4.2.4b prerequisites.

## Discipline

- kmod 0.4.4 (md5 c6de1afa) saved before 0.4.5 swap.
- kmod 0.4.5 .ko (md5 b47db450) + source tarball (md5 5f154a14... → e0cfa38b for v0.4.5) saved.
- prior cipher_rt artifact (md5 5d0f200c...) saved.
- 20-cycle stress NOT run on 0.4.5 (time budget tradeoff; substrate verified by `cipher_test_b10_fit_hint` + WL05 B10 verify + WL01 T4.2.4a 600s — all clean across one full kmod reload cycle in the morning).
- Phase 3 ABI 12/12 PASS at every checkpoint.
- Fallback md5s unchanged (55ab8c0c / 86618c30).
- Taint unchanged at 12288.
- No oops / WARN / BUG / NULL in dmesg.
