# 3-Arm Cross-Process Benchmark — Arm 2 Setup (vLLM intra-process baseline)

**Date:** 2026-05-18. Foundational setup per `FUTURE_SCOPE/C` — not the full
3-arm run. Goal: install vLLM, verify it serves TinyLlama-1.1B, capture a
single-tenant baseline (tok/s, tok/W).

## vLLM install

- **vLLM 0.21.0**, installed via `pip` into an **isolated venv**
  `/home/ubuntu/vllm_env` (not the system environment).
- **Why isolated:** the brief said "pip install vllm in the ubuntu
  environment", but vLLM pins heavy deps (its own torch, CUDA libs). Installing
  into the system env risked clobbering the torch 2.11/cu130 the CIPHER
  substrate, `pillar_driver`, and the Phase B executors are built against. The
  isolated venv protects the CIPHER environment — Arm-2 runs use
  `vllm_env/bin/python`; everything else is untouched. (Verified: vLLM's venv
  brought torch **2.11.0+cu130** — same version, runs on the pod's 580 driver.)
- Install clean (`VLLM_INSTALL_DONE`, exit 0). Log: `/home/ubuntu/vllm_install.log`.

## Verification — vLLM serves TinyLlama-1.1B

vLLM offline `LLM` API loaded `/home/ubuntu/models/TinyLlama-1.1B` and generated
all 5 Phase-B prompts successfully. Config:
- model: TinyLlama-1.1B (local), `dtype="float16"`
- `gpu_memory_utilization=0.5`, default CUDA-graph capture, paged attention
- `SamplingParams(max_tokens=128, temperature=0.0)` — greedy, matching Phase B

## Single-tenant baseline

5-prompt workload submitted to one vLLM instance (vLLM batches the 5 via
intra-process continuous batching, B ≤ 5), 150 iterations, 32.4 s sustained
window, 100 power samples:

| metric | value |
|---|---|
| aggregate tok/s | **2447.4** |
| mean full-GPU power | **303.1 W** (p50 306, settled ~306–313) |
| SM clock | **1980 MHz** (full boost) |
| **tok/W** | **8.07** |

`total_tok` 79 200 over 150 iters = 528 useful tokens/iter — matches the WL01
gold (prompt 3 EOS at 16: 128+128+128+16+128). Artefacts:
`phase_c/vllm_arm2_baseline.py`, `/tmp/vllm_arm2_power.csv`.

## Observations (relevant to the full 3-arm comparison)

1. **vLLM runs the GPU at full power.** 303 W, SM clock 1980 MHz — vLLM pushes
   full clocks. CIPHER's Phase B batched executor ran at ~150 W (and the all-on
   arm applies a DVFS clock-lock). `tok/W = tok/s ÷ power`, so vLLM's high
   power is a real factor — and CIPHER's substrate carries DVFS as an actuator
   vLLM does not. The 3-arm benchmark must hold this in view.
2. **This baseline is "vLLM serving a 5-prompt batch", not "vLLM with a B=1
   stream".** In the 3-arm benchmark's Arm 2 — *N separate tenant processes,
   each B=1 decode* — a single-stream vLLM instance has nothing to batch
   intra-process, so Arm 2 is expected ≈ Arm 1 for that topology (per
   `FUTURE_SCOPE/C` and `INDUSTRY_METHODOLOGY_ALIGNMENT.md` — that ≈ is the
   point: it shows where vLLM structurally cannot help). This foundational
   number is a vLLM throughput/power *reference*, not the Arm-2 result itself.

## Status

**Arm-2 infrastructure ready.** vLLM installed (isolated), verified serving
TinyLlama, baseline captured. The **full** Arm-2 measurement — N concurrent
single-stream vLLM tenant processes under the methodology-memo arrival pattern
— is run as part of the 3-arm benchmark (`FUTURE_SCOPE/C`), which needs
`FUTURE_SCOPE/A` (CIPHER + vLLM composed architecture) for Arm 3.

Anchor `a7ac8e97` unchanged; CIPHER system environment unchanged (vLLM isolated).
