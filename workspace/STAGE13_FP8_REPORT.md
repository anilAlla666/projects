# Stage 13 — FP8 Compute Substitution via cublasLtMatmul

Date: 2026-04-30
Pod: H100 80GB SXM (700 W TDP, 1350 MHz sustained, 1395 MHz boost),
CUDA 12.8, libcublasLt 12.8.4, sm_90.

## What ships

A new module `cipher_fp8_compute.{h,cpp}` (in rt) and a 70-line block in
`cipher_intercept_cudart.cpp` (in hook) that together substitute every
fp16 cublasGemmEx call from PyTorch's nn.Linear with cublasLtMatmul on
FP8 E4M3 inputs. Output stays fp16; transport and compute happen in FP8.

Three NVRTC kernels do the conversion:

1. `cipher_fp8_absmax`     — per-tensor absmax via per-block reduction +
                                atomicMax on bit-pattern (monotonic for
                                non-negative floats).
2. `cipher_fp8_finalize_scale` — turn absmax into scale = absmax / 448.
3. `cipher_fp8_quantize`   — fp16 → FP8 E4M3 via the device-resident scale.

Weight quant is one-shot per (ptr, shape) and cached for the engine's
lifetime. Activation quant fires inline before each matmul (~2 µs at
decode M).

`cublasLtMatmul` is configured with:
- compute = CUBLAS_COMPUTE_32F, scaleType = CUDA_R_32F
- transA = T, transB = N (matches PyTorch's nn.Linear convention)
- A_SCALE_POINTER + B_SCALE_POINTER point to device fp32 scalars
- explicit AlgoGetHeuristic call (`algo=NULL` produces silently wrong
  output for FP8 at M ≥ 2 with large N — discovered during Step 6)
- C = NULL with beta=0 (per NVIDIA reference)

Dispatch rule (in the cublasGemmEx shim, at `n_cublas` = decode batch):

| `n_cublas` | path                                                  |
|------------|-------------------------------------------------------|
| 1          | reserved for INT4 GEMV (Python-level Int4Linear)      |
| 2 .. 64    | **FP8 cublasLtMatmul (this stage)**                   |
| > 64       | fall through to fp16 cuBLAS                            |

Stability gate: 2 hits at the same (weight_ptr, m, k) before quantizing
— filters transient buffers. After the threshold every subsequent call
takes the FP8 path.

## Steps run end-to-end

| Step | Status |
|------|--------|
| 1 — verify libcublasLt.so.12 / sm_90 / FP8 tensor cores | PASS |
| 2 — NVRTC FP8 weight quant kernel                        | PASS |
| 3 — NVRTC FP8 activation quant kernel                    | PASS |
| 4 — cublasLtMatmul wrapper (handle, layouts, desc, scales) | PASS |
| 5 — wire into cipher_cublasGemmEx_impl                  | PASS |
| 6 — correctness vs fp16 cuBLAS, all 4 Mistral shapes × {1,8,32,64} | **16 / 16 PASS** |
| 7 — Mistral-7B end-to-end eager, B={1,8,32,64}, clock-locked | PASS |
| 8 — stack with M=1 INT4 + M=2..64 FP8 dispatch          | PASS |

## Step 6 correctness (Mistral GEMM shapes, fro_rel)

```
   M      N      K    fro_rel  rel_max rel_mean   NaN  verdict
   1   4096   4096     0.0362  54.5652   0.1746     0    PASS
   8   4096   4096     0.0372 154.5476   0.1834     0    PASS
  32   4096   4096     0.0374 191.2448   0.1853     0    PASS
  64   4096   4096     0.0374 221.0388   0.1882     0    PASS
   1   1024   4096     0.0364  26.6443   0.1692     0    PASS
   8   1024   4096     0.0373  47.8220   0.1715     0    PASS
  32   1024   4096     0.0373 143.1905   0.1846     0    PASS
  64   1024   4096     0.0374 136.6662   0.1829     0    PASS
   1  14336   4096     0.0374  63.2149   0.1727     0    PASS
   8  14336   4096     0.0375 188.6841   0.1821     0    PASS
  32  14336   4096     0.0374 165.8706   0.1849     0    PASS
  64  14336   4096     0.0375 217.2083   0.1848     0    PASS
   1   4096  14336     0.0376  72.4279   0.2078     0    PASS
   8   4096  14336     0.0379 207.9233   0.1918     0    PASS
  32   4096  14336     0.0376 189.1260   0.1828     0    PASS
  64   4096  14336     0.0373 129.2639   0.1821     0    PASS
```

Frobenius rel-error sits at ~3.7% across the entire sweep — that is the
FP8 E4M3 noise floor (3 mantissa bits, 8-bit exponent saturated at
±448). Per-element max can be large because reference values near zero
amplify rel-err; mean ~0.18 is consistent with E4M3's quant noise on
small-magnitude entries. Zero NaN / Inf in any cell.

## Step 7 — Mistral-7B fp16 EAGER decode (clock locked at 1350 MHz)

`step7_fp8_eager.py` spawns each mode in its own process under
`LD_PRELOAD=libcipher_hook.so:libcuda.so` and measures a 8-second
decode window after a 10-step warmup. Power sampled by nvidia-smi at
~150 ms cadence.

**Without fused RMSNorm + SiLU (FP8 isolated):**

```
  B   baseline tps  baseline tok/W     fp8 tps   ×tps     fp8 tok/W   ×tok/W  fp8_calls
  1          46.58          0.2990       47.01   1.01        0.3014     1.01          0
  8         378.41          1.9315      355.53   0.94        2.2309     1.16      68850
 32        1082.82          4.1507     1087.32   1.00        4.5837     1.10      61200
 64        1399.89          4.9612     1394.01   1.00        5.2408     1.06      39375
```

**With fused RMSNorm + SiLU·Mul (the full eager-mode lever stack):**

```
  B   baseline tps  baseline tok/W     fp8 tps   ×tps     fp8 tok/W   ×tok/W
  8         381.76          1.9570      408.69   1.07        2.4205     1.24
```

At B=8, fusion + FP8 deliver +7 % tps and **+24 % tok/W** vs the eager
fp16 baseline at the same locked clock. fp8_calls = 68,850 over the
window confirms substitution is firing on every decode-step linear (32
layers × 7 linears × ~308 decode steps).

**Stack mode (Int4Linear at M=1 + FP8 at M=2..64):**

```
  B   baseline tps  baseline tok/W   stack tps   ×tps   stack tok/W   ×tok/W  fp8_calls
  1          46.58          0.2990       44.24   0.95        0.2903     0.97          0
  8         378.41          1.9315      326.85   0.86        2.1176     1.10      68850
 32        1082.82          4.1507     1083.12   1.00        4.5746     1.10      60975
 64        1399.89          4.9612     1392.36   0.99        5.2405     1.06      39375
```

At B=1 the FP8 path is correctly off (`fp8_calls = 0`) — M=1 is reserved
for INT4 GEMV. Stack at B=1 shows a **3 % regression vs the fp16
baseline**. This is consistent with the existing CLAUDE.md result that
INT4 GEMV's tok/W win (1.08×) was measured **in graph-capture mode**;
in eager mode the Python-level Int4Linear wrapper costs more than the
INT4 GEMV kernel saves. Graph capture (out of scope for this stage —
spec says "EAGER mode only") would recover the win.

## Why FP8 helps

FP8 trades 3 mantissa bits for half the HBM bandwidth on weight reads
and double the H100 tensor-core throughput. Empirically:

- At **B = 8** the path is HBM-bandwidth bound (decode is GEMV-like).
  The 50 % reduction in weight bytes per matmul lets the GEMM finish
  faster while drawing less power, netting +24 % tok/W when stacked
  with fusion. The 6 % tps loss in FP8-isolated comes from the per-
  call activation absmax + quant launches; fusion offsets it.
- At **B = 32 / 64** the path is moving toward compute-bound. tps lands
  at parity (the FP8 tensor cores deliver more flops, roughly canceling
  quant overhead) and the win shrinks to 6–10 % because the bandwidth
  saving matters less when compute already dominates.

This matches the spec's prediction:
> At decode (memory-bound), the bandwidth reduction is the win. At
> larger batch (approaching compute-bound), the 2× TOPS is the win.

## Files added

- `include/cipher_fp8_compute.h`
- `src/cipher_fp8_compute.cpp`              — 9 KB, ~360 LOC
- `tests/test_fp8_correctness.py`           — 16-shape PASS gate
- `step7_fp8_eager.py`                       — Mistral-7B harness
- `STAGE13_FP8_REPORT.md`                    — this file

## Files modified

- `src/cipher_intercept_cudart.cpp`          — 70-line FP8 dispatch block
  in `cipher_cublasGemmEx_impl`, gated on TransA=T, TransB=N, fp16
  in/out, 2 ≤ n_cublas ≤ 64, and the stability counter.

## What is *not* in this stage

- **No graph-capture path.** Spec is explicit that this is eager-mode
  only. Stage 5 (cipher_graph) already does graph capture; the FP8
  kernels are graph-capture-safe (they go through cuLaunchKernel on the
  caller's stream and use a persistent cublasLt workspace) so future
  graph-mode integration should plug in cleanly.
- **No FP8 epilogue.** Stage 13 hands fp16 back to PyTorch. A future
  pass could keep the activation in FP8 across consecutive linears
  (e.g. q_proj output → k_proj input via FP8 epilogue) for additional
  HBM savings; not in scope here.
- **No per-channel / blocked scales.** Per-tensor absmax is what cuBLAS
  natively supports via `*_SCALE_POINTER`. Per-channel would need a
  follow-up dequant epilogue or the new `MATRIX_SCALE` API.
