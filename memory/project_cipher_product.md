---
name: CIPHER product definition
description: What CIPHER is and the four-instance substitution library plan — stable as of 2026-04-05 after three envelope corrections
type: project
---

CIPHER is a **runtime operator substitution primitive** at the CUDA interception layer, delivered as an LD_PRELOAD shim.

> At cudaLaunchKernel / cublasGemm time, recognize the op via fingerprint, swap for a cheaper mathematically-equivalent op from a registered library, verify correctness via held-out snapshot, silently fall back if verification fails. O(1) at call site. No framework cooperation.

## Architecture (session 7 lock-in)

- `libcipher_hook.so` — LD_PRELOAD shim (intercept, fingerprint, persist detector, pool observer)
- `libcipher_rt.so` — substitution engine (Koopman registry, EDMD live calibration, verification gate)
- `libnccl-tuner-cipher.so` — NCCL tuner plugin (multi-node TP expansion, later)

## The substitution library — four instances, one primitive

One interception point, one fingerprint table, one verification gate, one fallback path. **Order STABLE as of 2026-04-05** after the batch-range correction (32–256, not literal batch=1; fp16/bf16 only, no quantization):

| Order | Instance | Status | Attacks | Gate |
|---|---|---|---|---|
| **1st** | **#4 Graph-captured block (fp16)** | **QUEUED — next to build** | Launch overhead + kernel scheduling gaps | Batch=256 MFU ≥ 75% (intermediate) |
| **2nd** | **#3 Fused megakernel (fp16)** | Queued | Intermediate HBM writes, non-GEMM small kernels | Combined with #4: batch=256 MFU ≥ **85%** |
| **3rd** | **#1 Koopman rank-r** | Shipped (LM head 7.4×) | Structured shapes | Already validated |
| **Optional** | **#2' — 2:4 sparsity offline preprocessor** | Pending user approval | HBM weight bandwidth (2× free, fp16-preserving) | Batch=128 MFU ≥ 85% |
| **Deferred** | #2 — Quantization tiers (FP8/INT8/INT4) | Shelved | Extreme-throughput opt-in only | Not part of session 7 scope |

## Why this order is stable

At the corrected envelope (continuous batching, batch range 32–256 on H100, fp16/bf16), the physics ceiling at batch=256 is **86.7% MFU** (arithmetic: 256 FLOPs/byte vs H100 ridge point 295 FLOPs/byte). **No quantization is needed to reach the 85% gate** — stock PyTorch leaves ~30 percentage points on the table via launch overhead, kernel scheduling gaps, and intermediate HBM writes. Instance 4 (graph capture) + Instance 3 (megakernel fusion) close that gap directly.

Batch=128 also reaches the gate if 2:4 structured sparsity is approved (adds 2× HBM relief, still fp16 weights, no reputation hit). Below batch=128 the theoretical ceiling drops; CIPHER tracks the ceiling within 5 pp across the whole 32–256 range.

## Three product goals (non-negotiable, MET under this plan)

1. **New primitive** — MET (runtime op substitution with online verification at CUDA intercept)
2. **O(1) substitution** — MET (fingerprint → hash → pointer swap)
3. **MFU 85%+** — REACHABLE at batch=256 fp16 pure via Instances 4+3, confirmed by arithmetic

## How to apply

- Build order is **4 → 3 → 1 (keep), optional 2' (sparsity)**. Not 2 → 4 → 3. Not 4 → 2 → 3.
- fp16/bf16 is the compute precision floor — see `feedback_fp16_is_the_floor.md`. Weight STORAGE tricks (sparsity, structural compression) are permitted; weight COMPUTE precision is not.
- Quantization (FP8/INT8/INT4) is deferred — see `project_instance_2_deferred.md`. Never the load-bearing path.
- Primary gate is at batch=256. Below that, we deliver ceiling-tracking MFU and the product value is "near-hardware-limit at any batch the customer runs".
- Cheap falsification first (see `feedback_cheap_falsification.md`) — Phase 4.0 measurement harness runs before any C++ work on Instance 4.
