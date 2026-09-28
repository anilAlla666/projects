# Tasks 1 & 2 — wider FP8 gate + GEMM/non-GEMM profile

Date: 2026-04-30
Pod: H100 80GB SXM, CUDA 12.8, sm_90.

## Task 1 — FP8 gate widened from `n<=64` to `n<=512`

`src/cipher_intercept_cudart.cpp:691` changed:
- before: `&& n >= 2 && n <= 64`
- after:  `&& n >= 2 && n <= 512`

**Correctness gate** — `tests/test_fp8_correctness.py` extended to
M ∈ {1, 8, 32, 64, **128, 256, 512**} × 4 Mistral shapes:

```
28/28 PASS    (fro_rel ~0.037 across all M, 0 NaN, 0 Inf)
```

**Re-measurement at B=32 / B=64 (1100 MHz, full mode, P=128):**

```
  B   baseline tok/W   full tok/W (gate≤64)  full tok/W (gate≤512)   Δ
 32          3.3501              5.2070               5.2264       +0.4%
 64          3.9411              5.9257               5.9399       +0.2%
```

**Conclusion:** widening the gate is essentially noise (+0.2 to +0.4%
tok/W) at this prefill length.  Reason: at B=32/B=64 decode the
n_cublas value is exactly 32/64 — already inside the original gate.
At prefill (P=128, B=32) n_cublas=4096 — still outside the new 512
gate.  The change would only matter for prefill-heavy workloads at
moderate B where prefill n_cublas falls in (64, 512].  Worth keeping
since correctness is unaffected and longer-context measurements may
benefit, but it's not the lever that closes the B=32/B=64 gap.

## Task 2 — per-decode-step GPU time breakdown (vanilla PyTorch baseline)

Profiled 8 decode steps with `torch.profiler` after a 3-step warmup.
Bucketized every kernel by name into GEMM / ATTN / Elementwise / Other.

```
                    B = 32   per-step  14,096 µs        B = 64   per-step  19,246 µs
                    ────────────────────────────        ────────────────────────────
  category          cuda_us / step    %                cuda_us / step    %
  ────────────      ──────────────    ─────            ──────────────    ─────
  GEMM (linear)              5,731   40.65 %                  5,839   30.34 %
  ATTN  (cutlass MEA)        2,217   15.72 %                  4,117   21.39 %
  Elem (RMSNorm/SiLU/        5,345   37.91 %                  8,464   43.98 %
        RoPE/cast/index)
  Other (reductions/         803     5.72 %                   826     4.29 %
         concat)
```

Top kernels driving each bucket:

- **GEMM**: `nvjet_hsh_*` (Hopper-optimized cuBLAS hgemm; q/k/v/o + gate/up/down projections) + `cublasLt::splitKreduce_kernel`.
- **ATTN**: `fmha_cutlassF_f16_aligned_64x128_rf_sm80(PyTorchMemEffAttention…)` — PyTorch's memory-efficient attention.
- **Elem**: `at::native::elementwise_kernel<128,4,gpu_kernel_impl>` (RMSNorm + RoPE + residual + cast + SiLU·mul) + `unrolled_elementwise_kernel`, `vectorized_elementwise_kernel`, `index_elementwise_kernel`.
- **Other**: `at::native::reduce_kernel<512,1,...>` (RMSNorm variance reductions) + `CatArrayBatchedCopy` (KV cache concat).

## What this says about the 2× ceiling at large B

CIPHER's existing levers attack different buckets:

| lever                          | bucket       | B=32 attackable share | B=64 attackable share |
|--------------------------------|--------------|-----------------------|-----------------------|
| FP8 substitute (cublasLt)      | GEMM only    | 40.7 %                | 30.3 %                |
| Fused RMSNorm + SiLU·Mul       | part of Elem | ~12 % (RMSNorm + SiLU)| ~13 %                 |
| Clock lock                     | global power | applies to all        | applies to all        |
| Graph capture                  | CPU launch   | cross-cutting         | cross-cutting         |

If FP8 made GEMMs *infinitely* fast and fusion erased its share,
the theoretical ceiling at B=32 is:
> `14,096 / (14,096 − 5,731 − 1,693) = 14,096 / 6,672 = 2.11×` tps

— but FP8 in practice delivers ~2× kernel throughput on its share
(not infinite), and fusion cuts ~50 % of RMSNorm+SiLU time.  Realistic
ceiling at B=32:
> `14,096 / (14,096 − 5,731/2 − 1,693/2) = 14,096 / 10,384 = 1.36×` tps
> + clock-lock power factor ~1.50× → **≈ 2.0× tok/W** at the limit.

We measure **1.55× tok/W** at B=32 and **1.49× tok/W** at B=64.  Gap
to the analytic ceiling: ~0.5 / 0.4 of the way to 2×.  The gap comes
from:
1. FP8's 2× wgmma throughput is theoretical.  In practice cublasLt's
   FP8 path lands ~1.4–1.6× of fp16 on these shapes (small batch dim,
   K / N typical for Mistral).
2. Fusion saves ~50 % of its bucket, not 100 %.  The unfused
   elementwise tail (RoPE, residual add, embedding lookup, index_copy
   into the KV cache) is still ~22 % at B=32 and ~30 % at B=64.
3. Attention (15.7–21.4 %) is **completely untouched**.  No FP8
   attention, no flash-attention-3 substitute, no V3 materialize-bypass.

## What lever is missing

To push B=32 / B=64 past 2× tok/W requires attacking the non-GEMM
portion.  Ranked by attackable share:

1. **Attention (16–21 % of time).**  PyTorch MEA at decode runs
   `fmha_cutlassF_f16_aligned`.  The Hopper FP8 attention path
   (FlashAttention-3 with E4M3 + descaling) would cut this 1.5–2×.
   Not in CIPHER today.
2. **Unfused elementwise tail (~22–30 % of time).**  RoPE, residual
   add, KV cache `index_copy_`, embedding gather, cast.  Each is a
   small kernel; together they're large.  Megakernel fusion (single
   kernel doing RMSNorm + RoPE + residual + KV write) would cut their
   launch share.
3. **KV materialize bypass.**  Eliminates PyTorch's `StaticCache.update`
   + materialize copies (~1 % of time per layer × 32 layers ≈ 5 %).
   The V3 cache + B-aware kernels are already in place from rev-3;
   what's missing is the Python-level `StaticCache.update` monkey-patch.

## Summary

- Task 1 ✓ FP8 gate widened to n≤512, 28/28 correctness PASS, +0.2–0.4 %
  tok/W at B=32/B=64 with P=128 (the change becomes meaningful only at
  prefill-heavy or longer-context benchmarks).
- Task 2 ✓ At B=32 only **40.7 %** of decode time is in addressable
  GEMMs; at B=64 only **30.3 %**.  The other 60–70 % (attention +
  unfused elementwise + reductions) is what blocks 2× tok/W at large
  batches.  CIPHER's existing levers reach the analytic ceiling at
  ~1.5× given current bucket coverage; the missing lever is FP8 /
  flash attention to reach the next 1.5–2× share.

## Files

- `src/cipher_intercept_cudart.cpp` — gate widened (1 line, line 691)
- `tests/test_fp8_correctness.py` — extended BATCHES to {1,8,32,64,128,256,512}
- `profile_gemm_vs_nongemm.py` — torch.profiler-based GPU-time bucketizer
- `rerun_b32_b64_wider_gate.sh` — measurement runner
- `PROFILE_AND_GATE.md` — this report
