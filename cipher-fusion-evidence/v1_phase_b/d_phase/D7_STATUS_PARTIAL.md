# D.7 R-H1 — STATUS: re-key BUILT + mis-route PROVEN; positive inference-KL PARTIAL

**Date:** 2026-05-29. **NOT a close — anchor NOT rotated.** Honest status of the approved
D.7 build, stopped at the gate per discipline. **Anchors UNCHANGED** (cipher_rt_phase4
`8b5e928`/`9fe23143`, kmod `02fc2d1`, bridge `5a3db034`). Substrate source edit committed
at `7f9f98c` (not built into the anchor; staging only).

## PROVEN

- **Step 1 (docker) + Step 2 (container build):** DONE. The D.7 re-key (commit `7f9f98c`)
  rebuilt INSIDE the vllm-openai:v0.21.0 CUDA-13 container →
  `build_cuda13/libcipher_rt.so.d7_staging` md5 **`eab01676`**, `bind_model` exported,
  `make rc=0`. Anchor `9fe23143` preserved (staging is a separate file).
- **Step 4 mis-route BLOCK + family isolation: PASS** (`D7_KEYING_PASS`, in-container via
  ctypes on the staging `.so`). The `(model_id, w_ptr)` re-key isolates families and blocks
  mis-route — exactly Anil's spec ("register family-A uuid, feed family-B w_ptr → must NOT
  serve A's cache"): same `w_ptr` rebound A→B gives a FRESH slot (no bleed); A preserved
  after the B interleave (two-way isolation); `model_id=0` default isolated. This is the
  load-bearing R-H1 fix (prevents the audit-§3.1-G4 cross-family cache-collision corruption).

## PARTIAL — positive inference-KL gate cannot be met in this container (exact gap)

**Gate:** "each tenant KL=0 vs its family standalone" requires real per-family Marlin INT4
substitution during a forward. **Marlin does NOT engage on real model forwards in the
vllm-openai container:**
- B=8 Mistral-7B fp16 decode, staging substrate, `CIPHER_MARLIN=1`: **18,000 GEMMs observed
  but `max_n=56`** — the model's big Linear projections (N=4096 / 14336) are **never observed**
  by the actuator. `marlin_engage=0`, `marlin_bf16_obs=0`, `marlin_bf16_sub=0`.
- **Root cause:** the big Linear GEMMs bypass the GOT-patched `cublasGemmEx` path the shim
  hooks (they go through cublasLt / a fused path the shim doesn't intercept in the container's
  cu13 torch). Only tiny GEMMs (N≤56) reach the actuator → no eligible GEMM → Marlin never
  substitutes.
- This is **pre-existing and orthogonal to the D.7 re-key** (the re-key is correct and proven;
  this is a substrate↔container-torch interception gap that would affect the baseline anchor
  too). It is NOT forced past and NOT faked — a passthrough KL=0 (with `marlin_bf16_sub=0`)
  would be a false pass and is explicitly rejected.

**Therefore D.7 does not close:** the positive multi-family inference-KL with real Marlin
substitution is **un-demonstrable on this stack** until the cuBLAS-shim interception of the
container torch's big GEMMs is resolved (a separate substrate substep: extend the shim to
cublasLt / the fused matmul path — est. from the W6 cuBLAS-shim work, a few ED; net-new).

## NOT done (close prerequisites, blocked by the above)

- Step 3/4-full multi-family inference-KL (blocked: Marlin non-engagement above).
- Step 5 full no-regression (W7–W11 + 30-min N=128 soak + every-prior-model KL) — not run
  (close not reached; would only matter at close/anchor-rotation).
- Step 6 anchor rotation + tag + close report — NOT done (anchor stays `9fe23143`).

## Forward flag for V.1 (one line, not actioned here)

Even were Marlin engaging, its eligible regime (small-M INT4 GEMMs) vs Goal-1's **bursty
decode** workload is a V.1 question — D.7's INT4 residence may not be the actuator that
serves Goal-1's actual workload. Worth a sentence in the V.1 plan; not a D.7 detour.

## Disposition (for Anil)

- **R-H1 re-key: BUILT + correctness PROVEN** (isolation + mis-route BLOCK) in the validated
  container — the cross-family cache-collision risk is mitigated.
- **Positive inference-KL: PARTIAL** — blocked by the pre-existing cuBLAS-shim↔container-torch
  big-GEMM interception gap (Marlin never substitutes; `max_n=56`/`marlin_engage=0` evidence).
  Named, not forced, not faked.
- **Options:** (1) authorize a shim-interception substep (extend the cuBLAS shim to cublasLt /
  fused matmul so Marlin observes the big GEMMs) — net-new substrate, then re-run the D.7
  positive gate; (2) accept D.7 as re-key-proven / inference-PARTIAL and decide whether the
  Marlin INT4 path is even on Goal-1's critical path (the V.1 flag above) before investing.
Anchor unrotated; awaiting adjudication.
