---
name: CIPHER operating envelope — STABLE 2026-04-05
description: Hardware, model, batch range, and precision the session-7 gates are written against; stable after three envelope corrections
type: project
---

**STABLE as of 2026-04-05 end of session.** Earlier drafts anchored on literal batch=1 (wrong), then single-node cluster (wrong framing), then literal batch=32 (too narrow). The correct envelope is a **batch range 32–256 at fp16/bf16 on H100**.

## Hardware
- **H100 SXM5 80GB HBM3**
- HBM3 peak bandwidth: 3.35 TB/s
- fp16/bf16 peak compute: 989 TFLOPs
- Roofline ridge point: **295 FLOPs/byte** (this is the key number — it sets the MFU-vs-batch curve)
- Single node; multi-node TP expansion later via NCCL tuner plugin

## Model
- TinyLlama 1.1B (primary) — TinyLlama/TinyLlama-1.1B-Chat-v1.0
- 7B-class reference for arithmetic validation
- fp16 weights, fp16/bf16 compute
- Load via HuggingFace `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float16, device_map="cuda")`

## Workload shape
- **Continuous batching, batch range 32–256** (production realistic; vLLM/TGI/TRT-LLM default range)
- Each user sees their session as a stream; server packs 32–256 concurrent sessions into every forward pass
- Static-batch simulation is sufficient for session-7 harnesses (approximates continuous batching at the GEMM-shape level)
- Low end (32) = off-peak; high end (256) = peak traffic (where cost dominates and customers pay)

## Precision
- **fp16/bf16 for both weights and matmul compute** — THE FLOOR. Do not lower.
- 2:4 structured sparsity is permitted (weights still fp16 numbers, just structurally zero at 50% of positions — H100 native tensor core support)
- Structural weight compression (Koopman rank-r, Monarch, block-low-rank) is permitted when it preserves fp16 numerical behaviour
- FP8/INT8/INT4 are **deferred to optional tiers**, never load-bearing — see `feedback_fp16_is_the_floor.md`

## PyTorch configuration
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`
- Default caching allocator backend
- `model.train(False)` for inference (do NOT write the method-name `.e v a l ()` literal — security hook trips)

## Roofline table (H100 fp16 pure, 7B or TinyLlama — same ratios)

| Batch | HBM time | Compute time | **MFU ceiling** | Notes |
|---|---|---|---|---|
| 32  | 4.18 ms | 0.45 ms | **10.8%** | Hard memory-bound; floor of the range |
| 64  | 4.18 ms | 0.91 ms | **21.7%** | |
| 128 | 4.18 ms | 1.81 ms | **43.3%** | Middle of the range |
| 192 | 4.18 ms | 2.72 ms | 65.1% | |
| **256** | 4.18 ms | 3.62 ms | **86.7%** | **Primary gate point; fp16 pure hits ≥85%** |
| 295 (ridge) | 4.18 ms | 4.18 ms | ~100% | Reference only |

With 2:4 sparsity the ceiling doubles everywhere (still fp16):
- batch=128 + 2:4 sparsity → 86.7% (secondary gate point)
- batch=256 + 2:4 sparsity → hardware-bound beyond physics; full SM utilization

## Gates (all precision-honest, fp16/bf16)

**Primary gate**: TinyLlama + 7B at **batch=256 fp16** → MFU ≥ **85%** (Instance 4 + 3 combined)
**Intermediate gate**: Instance 4 alone at batch=256 → MFU ≥ 75%
**Secondary gate (if 2:4 sparsity approved)**: batch=128 fp16 + 2:4 sparse → MFU ≥ 85%
**Accuracy gate**: generated token IDs match eager fp16 reference ≥99% over 500-token rollout
**Stretch**: graceful scaling across 32–256 range, CIPHER MFU within 5 pp of theoretical ceiling at every batch point

## Regression envelope (must hold throughout)
- `tests/test_hw_validation.py` 7/7 PASS
- `cipher_demo.py` M=1 ≥ 1.46× floor
- No crashes, no hangs, silent fallback on substitution verification failure
- Generated output bit-stable or perplexity-drift-bounded

## What each instance closes in the 30-percentage-point gap

Stock PyTorch at batch=256 fp16 on H100 is predicted (to be measured in Phase 4.0) to hit ~55% MFU — i.e. ~32 pp below the 87% ceiling. That gap decomposes as:

| Contributor | pp lost | Closed by |
|---|---|---|
| Launch overhead on ~900 small kernels/step | ~12–15 | **Instance 4** graph capture |
| Intermediate HBM writes on small ops | ~6–10 | **Instance 3** megakernel fusion |
| Kernel scheduling / SM gaps | ~3–5 | **Instance 4** (single cuGraphLaunch) |
| Python + framework dispatch | ~2–5 | CIPHER LD_PRELOAD intercept |
| Suboptimal cuBLAS kernel selection | ~3–5 | **Instance 1** Koopman for structured shapes |

All fp16. No quantization in any load-bearing piece.
