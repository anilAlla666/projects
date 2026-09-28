# D.9 CIPHER-DELIVERY — workarounds to reach vLLM's prefill GEMMs (analysis)

**Date:** 2026-05-29. The driver-boundary intercept is dead (vLLM's GEMMs go via libcublas-internal
**nvjet**, and torch caches the public cuBLAS symbol addresses at its own init *before* CIPHER's
GOT-patch arms — the D.10 wall; CIPHER saw 0 GEMMs). The GEMM travels **Python (vLLM linear) →
torch (ATen) → libcublas (nvjet)**. Workarounds = intercept at a layer vLLM's dispatch *actually*
traverses, transparently (vLLM source unmodified), with CIPHER executing the FP8.

## The layer map + where CIPHER can transparently intercept
| layer | reachable? | substitute FP8? |
|---|---|---|
| vLLM linear `quant_method.apply()` (Python) | **YES — VERIFIED** | **YES** ← the hook |
| torch ATen `aten::mm/addmm/linear` (C++ dispatch) | YES (TorchDispatchMode) | YES |
| public cuBLAS `cublasGemmEx`/`cublasLtMatmul` (CIPHER's current GOT-patch) | **NO** — bypassed (nvjet + torch address-caching) | — |
| libcublas-internal nvjet | only via cuLaunchKernel | ~no (opaque kernel args) |

## Ranked workarounds
**1. Framework-layer intercept via CIPHER's vLLM plugin — VERIFIED VIABLE, RECOMMENDED.**
CIPHER's plugin (already runs in the vLLM EngineCore *worker* — "vLLM V1 worker GOT patches
installed pid=1") monkeypatches `vllm.model_executor.layers.linear.UnquantizedLinearMethod.apply`.
**Verified:** in-process, the patch caught **128 prefill GEMMs** (32 layers × {QKV, O, gate-up,
down}); the down_proj shape `(M,14336)@(4096,14336)` is distinctly identifiable → CIPHER declines it
to fp16 per the bounded-quality policy, keeps the rest FP8 (per-channel weight + per-token activation
via `torch._scaled_mm` or the ba873f44 fused-quant kernel), returns fp16. **vLLM source unmodified;
CIPHER's code does the quant + the per-layer decision + the FP8 GEMM → this IS transparent CIPHER
delivery, at the framework boundary.** Must be installed worker-side (via CIPHER's plugin
entrypoint), NOT a parent monkeypatch (parent patch saw 0 — separate process).
- *Bonus:* this also closes the §1b/build single-impl gap — MFU + quality measured in ONE
  CIPHER-engaged vLLM run, no per-tensor↔per-channel bridge, no harmonic-blend model.
- *Risks:* (a) CUDA-graph capture (enforce_eager=False) — the patched apply runs during capture so
  the FP8 ops should record; verify, else run enforce_eager. (b) Python wrapper overhead per call
  (~µs) is negligible vs the big prefill matmul. (c) MFU should ≈ vLLM-native FP8 (~88%, same
  `_scaled_mm`); measure directly.
- *Effort:* moderate — extend the existing `cipher_vllm_kv` plugin; reuse FP8 quant.

**2. TorchDispatchMode on `aten::mm/addmm/linear` (worker-side, via CIPHER plugin).** A `__torch_
dispatch__` mode intercepts every matmul at the ATen layer — more complete than #1 (catches
`torch.matmul`/`bmm` too, robust to vLLM linear-class changes). More effort; same transparent,
CIPHER-executed property. Good fallback/hardening of #1.

**3. CIPHER registers a vLLM quantization method (`cipher_fp8`).** vLLM's plugin system allows a
custom `QuantizationConfig`; CIPHER's plugin forces all linears onto `cipher_fp8` at load (CIPHER's
per-channel/per-token + per-layer policy). Cleanest *vLLM-integrated* path (uses vLLM's own extension
point), CIPHER-authored. Semi-transparent (CIPHER's plugin sets it; no operator flag). Moderate effort.

**4. Driver-boundary fix (the D.10 wall) — LOW feasibility.** Either (a) intercept libcublas-internal
nvjet dispatch, or (b) overwrite torch's cached cuBLAS function pointer *after* torch init so CIPHER's
GOT-patch takes effect. Hard, abandoned in D.10; only as a dedicated substrate track if the
framework-layer is deemed unacceptable.

**5. cuLaunchKernel-level nvjet substitution — NOT viable** (reconstructing the GEMM from opaque
nvjet kernel args + substituting is impractical).

**6. Disable nvjet via a cuBLAS env to force the public symbol — fragile**, perf-degrading, low
confidence it fully routes through CIPHER's patched entry.

## Recommendation + the honest framing
**Build #1 (framework-layer intercept via CIPHER's vLLM plugin), optionally hardened by #2.** It is
verified to catch every prefill GEMM, lets CIPHER execute the FP8 with its per-layer policy on an
unmodified vLLM, and yields the one-config CIPHER-engaged MFU+quality measurement directly.

**Honest caveat for the contract:** this is the **framework (Python/torch) boundary**, not the
**driver (cuBLAS) boundary** the original contract named. The driver boundary is *provably
unreachable* on this stack (nvjet + torch address-caching). The framework boundary is transparent
(vLLM source unmodified) and CIPHER-executed — a legitimate "CIPHER delivers FP8 transparently"
claim, qualified by the intercept layer. **Anil's call: accept the framework-boundary intercept as
the delivery mechanism, or treat driver-boundary as a hard requirement (then delivery stays blocked
pending the D.10 nvjet-interception track).**
