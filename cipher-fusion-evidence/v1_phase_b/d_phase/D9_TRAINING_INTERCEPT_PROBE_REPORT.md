# D.9 TRAINING-PATH INTERCEPT PROBE — VERDICT: FIRES + ACTUATABLE (driver-level, torch path)

**Date:** 2026-05-29. **Measure-only; no actuator built; no monkeypatch (Mem #24); anchors
UNCHANGED.** The load-bearing question: does CIPHER's driver-boundary GOT-patch FIRE on a real
torch training / large-batch step (torch `nn.Linear` → public cuBLAS), or is the address-caching /
nvjet bypass (which gave `calls=0` on vLLM) **torch-wide**? **Answer: it FIRES, with actuatable
arguments — the bypass is vLLM/matmul-specific, NOT torch-wide.**

## Setup
CIPHER ACTIVE via the same CDI/injection that armed the GOT-patches on the vLLM run
(`CUDA_INJECTION64_PATH=/usr/lib/cipher/libcipher_rt.so`, 17/17 LT trampolines armed, cublasGemmEx
resolved). Same CIPHER build (`01d4effb`/`d7-rh1-close`, kmod `0.7.0`). Different CALLER: plain
torch/HF, NOT vLLM. Workloads: (A) real TinyLlama-1.1B training step (fwd+bwd+AdamW, B=8×S=512,
3 steps); (B) Mistral-7B B=64×S=512 large-batch forward (the MFU-relevant big-GEMM shape).

## Decisive counters (CIPHER's own teardown telemetry)
| run | cublasGemmEx `calls / handled / passthrough` | attn `tramp_calls` |
|---|---|---|
| **vLLM prefill (prior, `d9_feasibility.log`)** | **0 / 0 / 0** | 0 |
| Training + B=64 forward (`d9_training_probe.log`) | **2070 / 0 / 2070** | 162 |
| **Mistral B=64 forward ×15 (`d9_bigGEMM_probe.log`)** | **3375 / 0 / 3375** | — |

**The 3375 is exact and airtight:** 15 forwards × 32 layers × **7 Linear** (q,k,v,o,gate,up,down) +
15 lm_head = **3375**. ⇒ **EVERY linear GEMM — including the big FFN ones (gate/up N=14336, down
K=14336) and attn proj (N=4096) — went through the public `cublasGemmEx` that CIPHER patches.** Had
the 3 big FFN linears bypassed (to cublasLt/nvjet), the count would be ~1935, not 3375.

## Verdict: **FIRES + ACTUATABLE**
- **FIRES:** CIPHER's driver-boundary `cublasGemmEx` GOT-patch is invoked on plain-torch training and
  large-batch forward — 2070 and 3375 calls. The vLLM `calls=0` bypass does **NOT** reproduce on the
  torch caller path. **The wall is vLLM-specific (its fused linear / nvjet entry), not torch-wide.**
- **ACTUATABLE:** these are **public `cublasGemmEx` calls with explicit, clean arguments**
  (m, n, k, A/Atype, B/Btype, C/Ctype) — the classic API CIPHER's shim already decodes and that
  Marlin already actuates for M≤64. `handled=0` here only because Marlin's **M≤64 decode gate**
  declines the large-M training/forward GEMMs (M = B·S = thousands) — they passthrough, but they are
  **at the actuatable hook**, not opaque-LT. An FP8 actuator at this intercept (without the M≤64
  gate) would engage them. NOT the D.10 LT-reconstruct problem.

**Reconciliation with §1b** ("torch big matmul → cublasLt"): §1b used raw `torch.matmul`/`_scaled_mm`,
which route to cublasLt/nvjet. **HF `nn.Linear` (`at::addmm`) routes through the PUBLIC `cublasGemmEx`**
— libcublas dispatches nvjet *internally*, but CIPHER catches the public call *before* that internal
dispatch, with clean args. So the substitution point is intact for the `nn.Linear` caller path.

## Consequence (per the pre-registered gate)
**FIRES + ACTUATABLE → driver-level transparent FP8 delivery is VIABLE on the torch-training /
HF-large-batch path** — the highest-value MFU target — at a public symbol CIPHER patches, with clean
actuatable tensors, **Mem #24 intact (no monkeypatch, no framework coupling), nvjet-interception not
needed.** The vLLM-prefill delivery stays deferred to v1.5 (its fused path bypasses; D.9_DELIVERY
workarounds), but the **training/large-batch MFU regime is reachable at the driver boundary now.**

**Next (if approved, NOT done here):** build the FP8 actuator on the `cublasGemmEx` intercept —
per-channel-weight + per-token-activation FP8 E4M3 (ba873f44 lineage), engaged for large-M GEMMs
(lift Marlin's M≤64 gate for this actuator), KL=0-on-output / bounded-quality per-layer policy
(down_proj→fp16). Measure CIPHER-engaged training/large-batch MFU + quality, one config, at the
driver boundary.

## Discipline
Measure-only at the driver boundary; **no framework-layer monkeypatch** (the vLLM LinearMethod patch
stays REJECTED, Mem #24); no torch/vLLM source touched; anchors UNCHANGED, no rotation. No FP8
engaged in this probe (intercept-telemetry only) → KL N/A. Minor residual: the classifier's `max_n`
summary did not emit on these short runs, so "big GEMMs included" rests on the exact call-count
identity (3375 = all 7 linears × 32 × 15 + lm_head) rather than a direct `max_n` read — the count is
a hard identity, not an estimate. Evidence: `d9_training_probe.log`, `d9_bigGEMM_probe.log`,
`d9_feasibility.log` (the vLLM `calls=0` contrast).
