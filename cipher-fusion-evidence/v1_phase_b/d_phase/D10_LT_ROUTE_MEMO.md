# D.10 — cublasLt variant actuator-routing (LT-ROUTE) — DESIGN MEMO

**Date:** 2026-05-29. **Status:** DESIGN MEMO (after step-0 diagnostic) — STOP for Anil
approval before any fix code. **This gates ALL in-container actuators — D.7's positive gate
AND every V.1 goal's actuator — so it sequences BEFORE re-attempting D.7-positive and before
V.1.** **Anchors UNCHANGED** (cipher_rt_phase4 `8b5e928`/`9fe23143`, kmod `02fc2d1`, bridge
`5a3db034`; evidence HEAD `39d489c`). Staging container build `eab01676`.

## §0 — Provenance

D.7's in-container test found Marlin never engages (`marlin_engage=0`, `max_n=56`) despite
real fp16 forwards. Step-0 diagnostic (this memo) traced why, in-container with the staging
`.so` bind-mounted over the CDI path.

## §1 — The exact interception gap (ground-truthed)

**Torch's hot-path matmuls route through the typed cublasLt variants, not the hooked paths.**
Authoritative evidence (cuBLAS's OWN logging, `CUBLASLT_LOG_LEVEL=4`, no CIPHER hooks):
- Big fp16 Linear projections execute via **`cublasLtHSHMatmul`** (half/single/half typed
  variant): `A=[rows=4096 cols=4096] type=R_16F`, `[rows=4096 cols=1024]`, etc.
- Tiny GEMMs: `cublasLtSSSMatmul` (R_32F rows=7) and 901 `cublasGemmEx` (small).

**CIPHER counters (staging `.so`, B=8 Mistral):** `cublasGemmEx_calls=9000` (all small-n →
`max_n=56`), **`cublasLtMatmul_calls=0`** (the umbrella is bypassed). The big fp16 GEMMs go
through `cublasLtHSHMatmul`.

**CIPHER already knows this and trampolines all 17 typed variants — but telemetry-only.**
`cipher_rt_cublaslt_variants.c` (F-B.3.6.1) header documents it verbatim: torch routes
fp32/fp16/bf16/tf32 through the 17 specialized entries (ACC…HSH…ZZZ), "the umbrella
cublasLtMatmul intercept is bypassed," telemetry is per-variant, and **"actuator routing for
variants requires decoding opaque `cublasLtMatrixLayout_t` descriptors (separate file
`cipher_rt_cublaslt_layout.c` + variant-specific full C shims) — [NOT BUILT]."**

**So the gap is NOT a missing hook — it's that the typed-variant trampolines are
pass-through telemetry with NO M/N/K extraction and NO actuator routing.** The big GEMMs are
*seen* (per-variant counters) but never reach `cipher_rt_matmul_dispatch` → no actuator
substitutes.

## §2 — Which goals are affected

**Every matmul-routed actuator is blind in-container on the default fp16/bf16 workload:**
- **Marlin (Goal 2 INT4 tok/W lever):** blind — `marlin_engage=0` (confirmed). The big
  projections never reach the actuator.
- **Koopman (Goal 4), Machete, any matmul substitution:** blind on the cublasLt-variant path
  (they route via `cipher_rt_matmul_dispatch`, which the variants don't call).
- **D.7 R-H1 positive gate:** blocked (can't show per-family INT4 KL when Marlin never fires).
- **Likely still firing (separate paths, NOT affected):** VOLT (NVML clock lock, no matmul
  dependency) and the attn/SDPA actuator (its own dispatcher). To be confirmed in the close.

This **gates V.1 entirely** — Goals 2/3 (tok/W, MFU) depend on actuators that are blind
in-container. Without this fix, a V.1 soak measures vanilla-with-overhead, not CIPHER.

## §3 — The fix (substrate-line only)

Build the documented-but-unbuilt F-B.3.6.1 follow-up:
1. **`cipher_rt_cublaslt_layout.c` — descriptor decode.** Extract M/N/K, dtype, transpose,
   batchCount from the opaque `cublasLtMatmulDesc_t` + `cublasLtMatrixLayout_t` via
   `cublasLtMatmulDescGetAttribute` / `cublasLtMatrixLayoutGetAttribute` (the public cublasLt
   introspection API — substrate-line, no torch patch).
2. **Upgrade the hot typed-variant trampolines to full shims** (start with **HSH** [fp16] +
   the bf16/tf32 equivalents torch actually uses; enumerate from the per-variant telemetry
   counters): decode dims (step 1) → build a `cipher_rt_matmul_call` → route through
   `cipher_rt_matmul_dispatch(&call, real_variant_fn)` (the same path `cublasGemmEx` uses at
   `cipher_rt_cublas_shim.c:221`) so Marlin/Koopman/actuators engage; passthrough to the real
   variant otherwise.
3. **Substrate-line (Mem #24):** GOT/dlsym-hook trampoline + the public cublasLt descriptor
   API only. **No torch/vLLM source patch (HARD STOP).**
4. **Default-OFF / additive (Mem #16):** the variant→actuator routing ships behind a gate
   (e.g. `CIPHER_LT_ROUTE=1`); OFF ⇒ the variants stay telemetry-only pass-through = current
   behavior = vanilla. Worst case unhooked == vanilla.

## §4 — Close gate

- **Interception proven:** on a real in-container fp16 forward, the big GEMMs (N=4096) are
  **OBSERVED through the cublasLt-variant path** — the classifier's `max_n` reflects ≥4096
  (not 56), and per-variant routed-counters > 0.
- **Actuator engages:** **`marlin_engage>0`** (Marlin actually substitutes) on a real
  in-container forward — the thing that is `0` today.
- **Correctness (Mem #11, HARD STOP):** every newly-engaging actuator is **KL=0 vs that
  model standalone**; any divergence = HARD STOP.
- **FULL no-regression (Mem #16 HARD GATE):** W7–W11 microbenches + 30-min N=128 soak +
  every prior model's KL = ZERO degradation vs `8b5e928`/`9fe23143`.
- Only then: rotate anchor + tag + close report.

## §5 — Discipline

Substrate-line only (Mem #24; cublasLt descriptor API + trampoline, no torch patch);
KL=0 every engaging actuator (Mem #11); FULL prior-regression before close + additive
default-OFF (Mem #16); anchor rotates only after the close gate passes; commit as Anil,
no co-author. One track at a time.

## §6 — One-line ask

Approve D.10 (build `cipher_rt_cublaslt_layout.c` + upgrade the HSH/bf16/tf32 variant
trampolines to dim-decode + actuator-route, default-OFF) — or adjust the variant set / gate.
On close, re-run the D.7-positive gate (Marlin now engaging) then V.1. No fix code until
approval.
