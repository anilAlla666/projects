# LANE 2 KERNEL CONTRACT — the quality-passing FP8 recipe (2026-06-10)

Extracted read-only from the D.9 record; vLLM-source items labeled separately. Every line is **RECORD (file:line)**
or **ASSUMPTION-FROM-VLLM-SOURCE (file:line)** — never blurred. 5-agent extraction + adversarial verify (confirmed).

## ⚠️ Three honesty flags the contract carries (read before building)

1. **The "+0.37 % PPL @ 88–98 % MFU" pairing is NOT one measured config.** Quality (+0.37 %) = HF eager
   per-channel via `torch._scaled_mm`. MFU (88–98 %) = **vLLM online per-TENSOR** prefill, then derated by a
   measured per-channel/per-tensor GEMM ratio 0.886. The two were never co-measured; the record's own panel
   calls this the "single-engine-rule violation" (`d9_build_verdict_panel.json:27`). **No single config has
   demonstrated ≥85 % MFU at ≤0.3 % PPL.**
2. **+0.37 % FAILS the LOCKED 0.3 % bar** (`d9_hf_quality_result.json:15` `WITHIN_BAR:false`). It "passes" only
   the 0.37 % amendment pre-authorized for Path B (basis arXiv 2411.02355). The config that passes the locked
   0.3 % bar *unamended* is **per-channel drop_downproj** (192 linears, +0.2465 % PPL, modeled projMFU ~80–86 %).
3. **Path B (the CIPHER-native CUTLASS rowwise kernel) was NEVER BUILT** (`D9_FP8_ACTUATOR_BUILD_REPORT.md:214`).
   Its quality is *inherited* from the reference HF run. Building it correctly is exactly Lane 2.

## The recipe (RECORD — the scheme behind +0.37 %)

| Element | Value | Provenance (file:line) |
|---|---|---|
| Scheme | per-output-channel weight scale + per-token dynamic activation scale, rowwise FP8 **E4M3** | `d9_hf_quality_result.json:2` |
| Weight quant (static, one-time) | `w_scale[oc] = amax(|W[oc,:]|, dim=in)/448.0`, `clamp(min=1e-12)`, fp32; `W8 = (W/w_scale)→float8_e4m3fn`; scale_b passed transposed-contiguous | `d9_hf_quality.py:15-16` |
| Activation quant (dynamic, per call) | flatten x→[-1,K] contiguous; `a_scale[row] = amax(|x_row|)/448.0`, `clamp(min=1e-12)`, fp32; `x8 = (x/a_scale)→float8_e4m3fn` | `d9_hf_quality.py:18-20` |
| GEMM | `torch._scaled_mm(x8, W8.t(), scale_a=a_scale, scale_b=w_scaleᵀ, out_dtype=torch.float16, use_fast_accum=True)`; try/except fallback retries WITHOUT use_fast_accum | `d9_hf_quality.py:21-22` |
| **Accumulation dtype** | **UNRESOLVED** — fast-accum requested, but whether it was active (vs the except-branch fallback) and the actual accumulator precision are not logged. Lane 2 must pin this (fast-accum = fp22-ish partial sums; the quality number may depend on which branch ran) | `d9_hf_quality.py:21-22` (no record of which branch executed) |
| Bias / output | bias added AFTER GEMM in fp16; output reshaped to input leading dims; base fp16 | `d9_hf_quality.py:23-24,29` |
| Projections quantized | **224 = 32 × {q,k,v,o,gate,up,down}**; **lm_head EXCLUDED** by name filter (`"lm_head" not in n`). lm_head inclusion is disproportionately costly: Path A with lm_head = +0.567 % / −1.6 pp MMLU FAIL | `d9_hf_quality.py:66`; `D9_FP8_ACTUATOR_BUILD_REPORT.md:145-150` |
| 448.0 | E4M3 max representable (the scale normaliser) | `d9_hf_quality.py:18` |

## Quality gate (the reproducible Lane-2 acceptance test)

| Element | Value | Provenance |
|---|---|---|
| PPL result | fp16 5.352497 → fp8 5.372299 = **+0.37 %** | `d9_hf_quality_result.json:5,9,12` |
| PPL dataset/protocol | **WikiText-2-raw-v1 test**, non-empty lines joined `"\n\n"`, first **50 non-overlapping 2048-tok windows** (stride 2048, 102,400 tok); teacher-forced next-token NLL, fp32 log-softmax, first token/window unscored (2047 targets/window); `PPL = exp(ΣNLL/Σtargets)` | `d9_hf_quality.py:49-52,61`; `D9_BUILD_PREREG.md:26-28` |
| MMLU | 0-shot, **500 q** = first 100 each of {abstract_algebra, anatomy, astronomy, college_computer_science, high_school_mathematics} (cais/mmlu test); argmax over " A".." D" last-token logits → **−0.4 pp** (45.4→45.0) PASS | `d9_hf_quality.py:11,35-43,56-60` |
| KL (diagnostic) | true full-vocab per-token KL(p_fp16‖q_fp8), mean, first 10 of the 50 windows → **0.00357 nats** (≤0.01 diag) | `d9_hf_quality.py:53,72-76` |
| Locked bar | PPL ≤0.3 % HARD, MMLU ≤0.5 pp HARD, KL ≤0.01 diag; **+0.37 % is over the 0.3 % HARD bar** → amendment-only pass | `D9_BUILD_PREREG.md:15`; `d9_hf_quality_result.json:15` |
| Unamended-pass config | per-channel **drop_downproj**: 192 linears (q,k,v,o,gate,up), 73 % GEMM-time coverage, **+0.2465 % PPL PASS**, MMLU 45.6, projMFU 85.9 % (harmonic-blend MODEL; per-channel-corrected band ~77–86 %) | `d9_perlayer_backoff_result.json:79-92`; `D9_BUILD_REPORT.md:24-26` |

## MFU half (what 88–98 % actually is)

| Element | Value | Provenance |
|---|---|---|
| Measured | vLLM **online per-tensor** FP8 prefill: 98.3 % (B8/S2048, 972.4 TFLOP/s, 1485 MHz, 647 W) … 91.7 % (B16/S4096) **vs 989 TFLOP/s bf16 peak**; = 45.8–49.1 % vs FP8-native 1979 | `d9_build_mfu_result.json:9-51` |
| Config | `quantization="fp8"` (online), dtype fp16, **enforce_eager=False** (cudagraph), max_num_seqs 16, max_model_len 4200, prefix-cache OFF, distinct prompts | `d9_build_mfu.py:23-24` |
| "88 %" low end | DERIVED: measured per-tensor 98.3 % × (per-channel 1265 / per-tensor 1427 TFLOP/s = 0.886) ⇒ ~88 % per-channel-adjusted; **per-channel never run end-to-end in vLLM** (llm-compressor↔vLLM `_C.abi3.so` env-block) | `d9_build_mfu_result.json:3-7`; `D9_CIPHER_DELIVERY_REPORT.md:46-47` |
| FLOP model | `2·P_lin·B·S + 2·L·B·H·S²·HD`, P_lin=7.241e9, L=32, H=32, HD=128; denom 989e12 | `d9_build_mfu.py:9-10` |

## vLLM-source defaults (ASSUMPTION-FROM-VLLM-SOURCE — fill only where record silent)

| Element | Value | Provenance |
|---|---|---|
| Online fp8 scheme | per-tensor static symmetric weight (`kFp8StaticTensorSym`, `ops.scaled_fp8_quant(weight, scale=None)`) + dynamic per-token activation (`kFp8DynamicTokenSym` when CUTLASS fp8 supported) | `vllm/model_executor/layers/quantization/fp8.py:303-310,538` |
| lm_head | left **unquantized** (`get_quant_method` returns FP8 only for LinearBase; ParallelLMHead is a VocabParallelEmbedding) | `vllm/model_executor/layers/quantization/fp8.py:172-204` |
| _scaled_mm fallback | vLLM's torch `_scaled_mm` fallback passes **no** `use_fast_accum` | (vLLM fp8 utils) |
| Single-impl route (blocked) | llm-compressor `QuantizationModifier(scheme="FP8_DYNAMIC", ignore=["lm_head"])` → compressed-tensors checkpoint vLLM loads natively (per-channel W + per-token A in ONE engine) — **env-blocked in the 2026-05 container**; the natural Lane-2 single-impl if unblocked | `d9_quantize_fp8_dynamic.py:14`; `D9_BUILD_REPORT.md:35-37` |

## Lane-2 build directive (synthesis)

Build the per-channel-W + per-token-A rowwise E4M3 kernel (CUTLASS or `torch._scaled_mm`-backed) that (a) reproduces
the quality gate above as a HARD ≤0.3 % test (target the **drop_downproj unamended-pass** point, or earn the 0.37 %
amendment explicitly), (b) **pins the accumulator precision** (the one UNRESOLVED element), and (c) is **co-measured
for quality AND MFU in ONE engine config** — the gap the entire D.9 record left open. Per Lane 1b, that engine should
be **eager vLLM prefill** (cublasGemmEx-covered, hosting tax ~0.9 %) or torch prefill, NOT cudagraph vLLM (where the
88–98 % was measured but CIPHER cannot intercept).
