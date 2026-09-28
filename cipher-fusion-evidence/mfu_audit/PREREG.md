# PREREG — MFU INVENTORY + ENGAGED MEASUREMENT (2026-06-10)

Written BEFORE any Part-2 measurement. Anchor 2edba0d2136f8ede4713d90a8f7cd55f (entry verified).
Clocks DEFAULT this run (verified unlocked at entry: 345 MHz idle, GpuIdle DVFS); achieved clock reported per lane.

## MFU definition (used everywhere)
MFU = achieved model-FLOPs/s ÷ 989.5 TFLOP/s (H100 SXM dense fp16/bf16 peak, NVIDIA datasheet, no sparsity).
Model FLOPs counted from the analytic per-token model regardless of executed precision (FP8 speedup ⇒ higher MFU
on the same denominator). Prefill: 2·N_params·T (N=7,241,732,096 exact). LoRA train: 4·N_params/token (frozen base).
Decode MFU is physics-bound low and is NOT a lift target here (Part 2 measures prefill/train only).

## Pre-registered expectations

| Lane | Expectation |
|---|---|
| 2a vLLM prefill vanilla (8×2048×10, eager, VLLM_PLUGINS="") | MFU 55–65 % (yesterday 60.6 % at locked clocks; default clocks should land similar — SwPowerCap governs compute-bound work either way) |
| 2a LoRA train vanilla (wi1 recipe, B=1 seq512) | MFU 9–12 % under the 4N model (yesterday 10.7 %) |
| 2a torch-eager HF prefill vanilla (FP8 lane's matched baseline) | MFU 35–60 % (HF eager less optimized than vLLM prefill; wide band, first measurement) |
| 2b E.3 RMSNorm+SiLU fusion | **WALL expected**: prior V.1 Tier-B rejected non-GEMM driver substitution (opaque ATen args at the dispatch boundary); the ledger's ~13 % e2e / 65–72 % ceiling is expected to resolve as a PLAN/projection number, not a measurement. If engagement is possible dispatch-boundary-legal, expected fused delta ≤ a few % e2e (non-GEMM time share at prefill is small) — i.e., the 13 % claim is expected NOT to reproduce |
| 2c FP8 actuator | **ENGAGE expected on the torch path** (V.0 precedent: FP8 routed=1125 via CUDA_INJECTION64_PATH on eager torch GEMMs; LD_PRELOAD ordering faults documented). Expected: handled-counter > 0; throughput/MFU effect on a compute-bound fp16 GEMM workload +0–20 % (prior: 1.05× iso-clock M=1 → 1.19× M=64 throughput; PPL +0.72 % quality caveat carried). Expected NOT engageable in vLLM/production (documented block — to be quoted from the attempt/inventory). If the torch-path engagement also fails this run, that IS the headline wall |
| 2d Marlin | bytes/token lever (W4A16); MFU effect expected ≈neutral-to-negative on fp16 compute-bound prefill; engagement only meaningful on quantized models — expect built+engaged-in-V0-demo, NOT an MFU lift mechanism |
| 2d Koopman | cite-only: shipped r=64 kernel 20–25× SLOWER, OOD gate never fires (refuted 2026-06-04); no measurement lane |

## Surprise rule
Anything outside these bands, or any engagement state contradicting the expectation, is reported as a surprise
with mechanism — including a SUCCESSFUL E.3 engagement (would contradict the Tier-B wall).

## Part-1 expectation (for completeness; resolved from source not memory)
Op-count question "21 vs 33": no pre-commitment; the dispatch table is 16 OpClass slots (already read), so both
21 and 33 likely count something else (exported counters / registered mechanisms / scorecard rows). The source
count with file:line wins.
