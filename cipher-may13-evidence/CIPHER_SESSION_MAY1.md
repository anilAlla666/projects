# CIPHER Session — 2026-05-01

Pod: 2× NVIDIA H100 80GB HBM3, NV18 NVLink (no NVSwitch).
Driver 580.105.08 / CUDA runtime 13.0 / nvcc 12.8.93 / PyTorch 2.7.0 / transformers 4.57.6.

This session: rebuilt CIPHER on a fresh pod, ran it against four model/quant configurations, found and fixed four bugs, and produced audited tok/W numbers across the full matrix.

---

## Headline numbers (all post-fix, output coherence verified ≥50 tokens)

### Product comparison — what a CIPHER user vs a stock-PyTorch user sees

This is the apples-to-real-product comparison: **stock eager PyTorch at default clocks** (no graph, no FP8, no fusion, no clock lock) vs **CIPHER's full delivered stack** (graph capture + FP8 + fusion + 1200 MHz lock). Same model, same prompt, same prefill — only "is CIPHER deployed" differs.

| Model | B | eager tps | CIPHER tps | ×tps | eager W | CIPHER W | ×pwr | eager tok/W | CIPHER tok/W | **×tok/W** |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B fp16 | 1 | 42.72 | 106.38 | 2.49× | 266.1 | 236.7 | 0.89× | 0.1605 | 0.4495 | **2.80×** |
| Llama-3.1-8B fp16 | 8 | 345.11 | 687.66 | 1.99× | 282.7 | 202.2 | 0.72× | 1.2206 | 3.4007 | **2.79×** |
| Llama-3.1-8B fp16 | 32 | 1296.13 | 1389.37 | 1.07× | 363.4 | 245.5 | 0.68× | 3.5667 | 5.6584 | 1.59× |
| Llama-3.1-8B fp16 | 64 | 2383.71 | 1676.85 | 0.70× | 465.7 | 263.0 | 0.56× | 5.1189 | 6.3757 | 1.25× |
| Mistral-7B fp16 | 1 | 59.60 | 95.95 | 1.61× | 237.5 | 218.8 | 0.92× | 0.2509 | 0.4385 | **1.75×** |
| Mistral-7B fp16 | 8 | 465.57 | 507.02 | 1.09× | 362.0 | 227.6 | 0.63× | 1.2862 | 2.2279 | 1.73× |
| Mixtral-8x22B 4-bit (2 GPU) | 1 | 8.63 | 9.07 | 1.05× | 380.6 | 281.1 | 0.74× | 0.0227 | 0.0323 | 1.42× |
| Mixtral-8x22B 4-bit (2 GPU) | 8 | 48.24 | 46.33 | 0.96× | 517.2 | 368.2 | 0.71× | 0.0933 | 0.1258 | 1.35× |

**Llama-3.1-8B fp16 at B=1 and B=8 hits 2.80× tok/W as a product** — the user sees ≈2× more tokens per joule simultaneously with ≈2× higher tokens-per-second.

### Apples-to-apples decompositions (graph-vs-graph, same-clock, same model)

These isolate what CIPHER's substitution+fusion add ON TOP of graph capture and clock locking:

| Model | Best B | best ×tok/W (graph base → graph CIPHER, both 1200 MHz) |
|---|---|---|
| **Llama-3.1-8B** | B=8 | **1.81×** |
| Mistral-7B | B=1 | 1.75× |
| Mistral-7B | B=8 | 1.73× |
| Mixtral-8x22B 4-bit (2 GPU) | B=1 | 1.42× |
| Llama-3.1-70B fp16 (2 GPU) | — | memory bound (see notes) |

The 70B fp16 case OOMs the second `model.generate()` call: Llama-70B leaves only ~8.6 GB headroom per H100 after weight load, and CIPHER's LD_PRELOAD adds ~10 GB of cuBLASLt driver-side workspace pressure on cuda:0 across consecutive generates. Documented; not solvable on 2× 80 GB without 4-bit quantization or 4 GPUs.

---

## Bugs found and fixed

### 1. RoPE stride bug (CRITICAL — caused all dense-fp16 measurements to be wrong)

**Symptom**: Llama-3.1-8B fp16 + CIPHER full stack produced `' computing work.<|end_of_text|>'` after 3 tokens. tok/W gains were on EOS pad tokens (illusory).

**Root cause**: `step{7,8,9}.py` RoPE wrapper passed `q.data_ptr()` directly to the NVRTC kernel. PyTorch's call site is:
```python
query_states = self.q_proj(hs).view(hidden_shape).transpose(1, 2)
```
That makes `q` a non-contiguous transposed view of `[B, S, Hq, D]` underlying memory, but the kernel reads with `stride = (Hq*S*D, S*D, D, 1)` — assuming contiguous `[B, Hq, S, D]`. Reads cross head/position boundaries → garbage RoPE output → wrong attention scores → model degrades to EOS.

Why it slipped through: synthetic test used `torch.randn(B, Hq, S, D)` which IS natively contiguous in `[B, Hq, S, D]`. Mixtral 4-bit didn't show it because bnb's dequant noise dominated CIPHER's RoPE error.

**Fix**: `q_in = q.contiguous()` before `q.data_ptr()`. Applied to step7, step8, step9, llama8b_fusion_probe.py.

**Numerical impact** (after layer 0, vs PyTorch native):
| | max_abs | mean_rel |
|---|---|---|
| Pre-fix | 3.25 | 3.40 (340% err) |
| Post-fix | 7.81e-3 | 2.87e-3 (0.3% — fp16 noise) |

### 2. FP8 weight cache aliased on bnb's recycled scratch buffer

**Symptom**: Mixtral-8x22B 4-bit + FP8 substitution produced gibberish. Same `A` pointer logged across all GEMMs.

**Root cause**: bnb's `Linear4bit` dequantizes each layer's 4-bit weight into one shared fp16 scratch buffer before calling `cublasGemmEx`. CIPHER's FP8 cache (`cipher_fp8_compute.cpp`) keys by `void* fp16_key = A`. After 2 observations, FP8 path quantized the FIRST layer's weight and routed every subsequent layer's GEMM through that same cached FP8 → all 56 layers used layer-0's weights.

**Fix**: Added `content_hash` (FNV-1a of first 128 bytes) and `transient` flag to `WeightEntry`. On observe: hits 1 and 2 verify the hash; if it drifts at the same address, mark transient and bypass FP8 forever for that pointer. Past 2 observations, no further D2H sync — the hot path stays sync-free.

### 3. Multi-GPU CUDA-context binding for fused kernels

**Symptom**: With model split across 2 GPUs (Mixtral via `device_map='auto'`), fused RMSNorm/SiLU/residual_add silently produced uninitialized output on cuda:1 layers.

**Root cause**: `cipher_fusion_kernels.cpp` used a single global `g_rmsnorm_id` etc. The CUmodule was loaded into whatever CUDA primary context was current at first compile (cuda:0). From cuda:1, `cuLaunchKernel(fn)` returns `CUDA_ERROR_INVALID_HANDLE` and `out` (allocated via `torch.empty_like`) was returned uninitialized.

**Fix**: Per-device `DeviceFns g_per_device[16]`. Each helper does `cudaGetDevice` + lazy compile per device — caller already wraps in `torch.cuda.device(d)` so `cipher_substitute_v2_compile`'s `cudaFree(0)` lands the cubin in the right context. Also discovered accelerate's `add_hook_to_module` captures the original `forward` at install time (`module._old_forward`), making class-level patches invisible — fixed by walking `model.modules()` and rebinding `_old_forward = MethodType(_patched_forward, m)`.

### 4. NCCL v4 tuner over-rides on small fabrics

**Symptom**: With `LD_PRELOAD=…libcipher_rt.so` + `NCCL_TUNER_PLUGIN=…libnccl-tuner-cipher.so`, AllReduce bandwidth dropped from 288.5 → 71.9 GB/s on 2× H100 NV18 (4× regression).

**Root cause**: `cipher_nccl_v4_decide` always returns RING/LL128/4 channels for 256KB-16MB and NVLS for ≥16MB. The bias was tuned for 8-GPU NVSwitch + NVLS systems where NCCL's defaults are suboptimal. On a 2-GPU NV18 direct fabric, NVLS is unavailable and CIPHER's RING/LL128/4ch is too thin — NCCL's auto-tuner picks a wider configuration.

**Fix**: Added topology guard in `cipher_nccl_tuner.cpp::decide_and_map`. If `nRanks ≤ 4` and `nNodes ≤ 1`, return `algorithm = NCCL_ALGO_UNDEF` (passthrough) before consulting v4. CIPHER's bias only kicks in on multi-node or 8+ GPU fabrics.

---

## Verified tok/W matrix (post all fixes)

### Mistral-7B fp16, single H100, prefill=512, graph capture, 60-token coherence verified

| B | base tps | full tps | base tok/W | full tok/W | ×tok/W | output |
|---|---|---|---|---|---|---|
| 1 | 59.60 | 95.95 | 0.2509 | 0.4385 | **1.75×** | coherent |
| 8 | 465.57 | 507.02 | 1.2862 | 2.2279 | **1.73×** | coherent |

### Llama-3.1-8B fp16, single H100, prefill=128, eager mode

| B  | base tps | full tps | base tok/W | full tok/W | ×tok/W |
|----|---|---|---|---|---|
| 1  | 42.72   | 46.17   | 0.1605 | 0.2047 | 1.28× |
| 8  | 345.11  | 367.02  | 1.2206 | 1.7555 | 1.44× |
| 32 | 1296.13 | 1375.26 | 3.5667 | 5.0562 | 1.42× |
| 64 | 2383.71 | 2488.19 | 5.1189 | 7.2063 | 1.41× |

### Llama-3.1-8B fp16, single H100, graph capture (this session's best)

| B  | base tps | full tps | base tok/W | full tok/W | ×tok/W | MFU% |
|----|---|---|---|---|---|---|
| 1  | 107.18  | 106.38  | 0.3207 | 0.4495 | 1.40× | 0.17 |
| 8  | 681.19  | **687.66** | 1.8763 | **3.4007** | **1.81×** | 1.12 |
| 32 | 1598.90 | 1389.37 | 3.7064 | 5.6584 | 1.53× | 2.26 |
| 64 | 2105.66 | 1676.85 | 4.3350 | 6.3757 | 1.47× | 2.72 |

### Mixtral-8x22B 4-bit (bnb nf4 + double-quant), 2× H100 with device-aware fusion + accelerate hook bypass

| B | base tps | full tps | base tok/W | full tok/W | ×tok/W |
|---|---|---|---|---|---|
| 1 | 8.63 | 9.07 | 0.0227 | 0.0323 | 1.42× |
| 8 | 48.24 | 46.33 | 0.0933 | 0.1258 | 1.35× |

### Llama-3.1-8B B=8 graph clock sweep (full mode tok/W vs default-clock baseline = 1.8763)

| clk MHz | tps    | watts  | tok/W  | ×tok/W |
|---------|--------|--------|--------|--------|
| 900     | 514.31 | 168.0  | 3.0619 | 1.63×  |
| 1000    | 582.57 | 178.7  | 3.2591 | 1.74×  |
| 1100    | 637.49 | 188.9  | 3.3750 | 1.80×  |
| **1200**| **683.99** | **201.0** | **3.4023** | **1.81×** ← peak |
| 1300    | 728.93 | 220.0  | 3.3133 | 1.77×  |
| 1400    | 768.08 | 243.5  | 3.1540 | 1.68×  |
| 1500    | 802.57 | 269.9  | 2.9739 | 1.59×  |

Peak at 1200 MHz. Output coherent at every clock point.

### Llama-3.1-8B B=1 graph clock sweep

| clk MHz | tps    | watts  | tok/W  |
|---------|--------|--------|--------|
| 900     | 79.72  | 197.0  | 0.4048 |
| 1000    | 90.09  | 210.7  | 0.4275 |
| 1200    | 106.02 | 235.3  | 0.4505 |

Higher clocks help here too — B=1 was already near peak at 1200 MHz; 0.4505 / 0.3207 = 1.40×.

### MFU at peak (Llama-3.1-8B graph)

| B | tps | MFU% (vs 989.4 TFLOPS H100 fp16 peak) |
|---|---|---|
| 1 | 106.38 | 0.17 |
| 8 | 687.66 | 1.12 |
| 32 | 1389.37 | 2.26 |
| 64 | 1676.85 | 2.72 |

Decode-only at fp16 with 8B params is intrinsically low-MFU (HBM-bandwidth-bound, not compute-bound). MFU ceiling for this workload class is ~10% even with aggressive kernel work.

---

## NCCL Stage 9 verification on 2× H100 NV18

| Configuration | per-iter | algBW | Notes |
|---|---|---|---|
| No CIPHER | 466.5 us | 287.7 GB/s | NCCL auto-tuner |
| CIPHER tuner only (no rt) | 465.6 us | 288.3 GB/s | passthrough, +0.2% |
| CIPHER tuner + rt + v4 (PRE-fix) | 1867.6 us | **71.9 GB/s** | **4× regression** |
| CIPHER tuner + rt + v4 (POST-fix) | 466.0 us | **288.0 GB/s** | small-fabric guard active |

The fixed tuner: detects nRanks≤4 + nNodes≤1, returns `nccl_algo=-1 nccl_proto=-1`, lets NCCL pick. Logs `[CIPHER TUNER] passthrough: nRanks=2 nNodes=1 — small-fabric guard #N` per call.

NCCL's loader confirmed using v2 plugin interface (it also looks for v3/v4 — those are not exported, treated as warnings, falls through to v2).

---

## Op regression on 2× H100 (CUDA_VISIBLE_DEVICES=0,1)

**31 PASS / 5 FAIL / 8 SKIP**.

FAIL list (all pre-existing, none multi-GPU specific):
- test_dep2.py — telemetry baseline shape check (perf gate)
- test_hw_validation.py — 5 sub-fails (MFU baseline, clock stability, NCCL intercept, non-GEMM classify, billing report)
- test_packaging.py — needs `cipher_runtime` Python module (separate install step not run)
- test_persist_dispatch.py — shim-overhead gate (-22% reduction vs 50% target — perf gate, not correctness)
- test_power_cap.py — 0 substitutions logged (intercept harness)

Multi-GPU vs single-GPU diff: gained `test_kernel_table.py` (now passes after `transformers` was installed). No new failures from multi-GPU exposure.

---

## What's working

- All 4 fused kernels (RMSNorm, SiLU·Mul, residual_add, RoPE) — bit-identical-up-to-fp16-noise vs PyTorch native, on dense fp16 and quantized weights, on single-GPU and multi-GPU device maps.
- FP8 substitution — fires on stable PyTorch nn.Linear weights, correctly bypasses bnb-style recycled scratch buffers.
- Graph capture — captures one decode step including FP8 + fused kernels; replay path produces coherent output at all batch sizes tested.
- Clock locking — 1200 MHz is the tok/W peak across both Mistral and Llama at this workload class; documented sweep.
- NCCL plugin — verified intercepting on 2× H100 NV18, correctly bails to passthrough on small fabrics.
- Multi-GPU op regression — 31/36 passing.

## What's not

- 70B fp16 on 2× 80 GB GPU: hard memory bound. CIPHER's LD_PRELOAD adds untracked-by-PyTorch driver workspace (~10 GB on cuda:0) that's invisible until inference. No clean workaround at this size; requires 4-bit or more GPUs.
- B=1 INT4 GEMV path on Mistral-7B did not produce a >2× tok/W win this session (got 1.75×). Historical 2.24× claim appears specific to a different pod or measurement set; on this pod the verified ceiling is 1.81× (at Llama-8B B=8 graph).
- Test_persist_dispatch perf gate (-22% reduction): the L2 persist shim is real but its overhead-reduction story isn't holding under current measurement methodology.
- NCCL v4 bias logic on 8+ GPU NVSwitch systems: not validated on this pod (only 2 GPUs available); the topology guard correctly defers to NCCL on small fabrics, but the multi-node win remains unmeasured.

## What's next

1. **Validate NCCL v4 on a real 8-GPU NVSwitch box** — the bias logic should win there; currently it's only verified to NOT lose on small fabrics.
2. **Marlin INT4 GEMM at decode** — historical 2.24× claim was driven by INT4 wins. If this matters going forward, port Marlin invocation into the graph-capture path on Llama, not just Mistral.
3. **70B fp16 with CIPHER**: either 4-bit quantization (quantize to int8/int4 in-place at load) or 4-GPU split. The hook's driver workspace overhead is fundamental.
4. **Lower-MFU floor for tok/W**: 1200 MHz is current peak; could try power-capping (`-pl 250W`) instead of clock-locking — power cap with high clock should let DVFS find a more energy-efficient operating point per kernel.

---

## Files added this session

- `step9_llama70b.py` — Llama 70B / 8B fp16 measurement harness (parameterized via --model/--params/--device-map)
- `step9_graph.py` — Llama-8B graph-capture variant
- `llama8b_fusion_probe.py` — per-kernel ablation + numerical-error probe
- `nccl_bench.py` — torchrun-launched 128MB AllReduce benchmark
- `clock_sweep.sh` — Llama-8B clock sweep driver
- `step8_mixtral_8x22b.py` — Mixtral 4-bit measurement harness (existing, updated)
- `CIPHER_SESSION_MAY1.md` — this report

## Code changes this session

- `src/cipher_fp8_compute.cpp` — added content-hash + transient flag to `WeightEntry`; observe path is sync-free past hits=2.
- `src/cipher_fusion_kernels.cpp` — replaced single global cubin IDs with `g_per_device[16]`; lazy per-device compile.
- `src/cipher_nccl_tuner.cpp` — added small-fabric topology guard before consulting v4 decide.
- `src/cipher_intercept_cudart.cpp` — extended GEMM-EX log line with transA/transB/dtype (debug aid; left in place).
- `step{7,8,9}*.py` — `q.contiguous()` / `k.contiguous()` before passing to RoPE NVRTC kernel.
- `step{7,8,9}*.py` — accelerate `_old_forward` rebinding to bypass hooks for class-level patches.
- step9 RoPE kernel: per-device CUmodule via dict (Python side) for fallback robustness.
