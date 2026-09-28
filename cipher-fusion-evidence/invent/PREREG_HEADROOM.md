# PRE-REGISTRATION — headroom-to-2x session (2026-06-11, before results read)

Continues the per-model bundle (PER_MODEL_BUNDLE_RESULT.md). All runs: H100, vLLM, cudagraph ON,
NREQ=32 saturated raw-prompt harness (s24_harness.py), VLLM_PLUGINS= (substrate NOT loaded),
VLLM_DEEP_GEMM_WARMUP=skip, best-of-3, power sampled at 10 Hz. Anchor 2edba0d2 untouched.

## Incident already logged (before these predictions)
First cell attempt ran WITHOUT the env protocol: vLLM auto-loaded the cipher_v2 KV plugin and init
died on deep_gemm warmup. Verified yesterday's headline runs are clean (0 "[cipher" lines in their
stderr). Contaminated artifacts: first s24_a..d.{json,log,err} (overwritten by the fixed re-run).

## Predictions

**P1 (cell A vs B, sparse bf16 vs dense bf16):** vLLM likely serves the bf16-only 2of4 checkpoint
with DENSE kernels (compressed-tensors sparse-only support is the FP8-2of4 path; the FP8 variant
404'd). Predict tok/s ratio A/B in 0.95–1.05 (no engaged sparse kernel). If instead vLLM engages a
sparse kernel (quant field != None), predict 1.1–1.4x.

**P2 (cell C vs D, sparse+fp8-dynamic vs dense+fp8-dynamic):** on-the-fly fp8 wraps linears with
dense FP8 GEMM, blind to pruned zeros. Predict C/D in 0.95–1.05 — i.e. 2:4 as-served adds NOTHING
without a sparse-kernel export. The 2:4 throughput lever then = per-model pipeline work (produce a
compressed-tensors 2:4(+FP8) artifact), not a serving flag.

**P3 (cell E, sparse + Instruct-trained EAGLE-3 head):** drafter was trained on Llama-3.1-8B-
INSTRUCT; target here is the pruned, recovery-trained BASE model. Predict acceptance collapse →
tok/s ≤ cell A (0.6–1.0x of A). Either outcome supports the per-model thesis: heads must be
co-trained per model; they do not transfer.

**P4 (dense bf16 @300W, chat harness, vs tpw_cap300 bundle 20.41 tok/W):** DVFS helps any model.
Predict dense@300W tok/W ≈ 12.5–15 (≈1.4–1.6x its uncapped 9.02), so the honest split of the
bundle@300W 2.26x total is ≈1.4–1.6x (DVFS-on-anything) ⊗ ≈1.4–1.6x (bundle), multiplicative.

**P5 (coverage probe, rvcount.so):** the FP8 target linears run on cutlass kernels invisible to
cuBLAS-API interposition. Predict: bundle run shows gemmex/ltmatmul counts dominated by the bf16
DRAFTER + (maybe) lm_head only — target FP8 linears ABSENT (no m∈{6144,28672} k=4096 fp8 shapes
via GemmEx). Dense control shows the full Llama linear set (m=6144,28672,4096 k=4096,14336,
lm_head m=128256). Consequence if confirmed: the detection family (GEMM-output recompute at the
cuBLAS seam) has ~0 coverage on the FP8 bundle as-built → detection on co-designed FP8 models
needs a different seam (cutlass epilogue / vLLM-coop at the quant-linear layer). First-class
product finding, not a footnote.

## Decision rule
No post-hoc re-scoping of P1–P5; misses get reported as misses.
