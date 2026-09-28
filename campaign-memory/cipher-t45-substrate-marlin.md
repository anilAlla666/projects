---
name: cipher-t45-substrate-marlin
description: T4.5 ships matmul-routing substrate (cuBLAS LD_PRELOAD via .symver) + Marlin actuator on it. Substrate clean; Marlin regime requires Mistral-7B B≥8 validation (TinyLlama B=1 regresses outside designed envelope).
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d28504c-5084-4f22-8df4-2a4f96d437ce
---

T4.5 (2026-05-14 evening, ~7h work) shipped the matmul-routing substrate
for CIPHER's compute-path moat. Marlin INT4 landed as the first actuator
on the substrate.

**Substrate (T4.5.1) — clean ship:**
- `libcipher_rt.so.v0.2.0_T4_5_substrate` md5 `2f8218abaccc43150aa9edfca472e61b`
- `.symver`-tagged cuBLAS interception (50 LOC + 4-line linker version
  script) — supersedes the ~3000 LOC of GOT-patching the prior
  cipher-may13-evidence shim used. PyTorch 2.11/cu13 lookups for
  `cublasGemmEx@libcublas.so.13` route to our shim.
- Substrate registers up to 16 priority-ordered actuators; each
  actuator's `maybe_handle()` returns HANDLED / PASSTHROUGH / ERROR.
- Verified: 155/155 TinyLlama cublasGemmEx calls dispatched cleanly;
  outputs byte-identical to baseline; overhead −1.69% ± 1.17% (CI
  straddles 0). Operator-context preserved: LD_PRELOAD + CUDA_INJECTION
  both point at same .so.

**Marlin actuator (T4.5.2) — landed but regime gap on TinyLlama:**
- `libcipher_rt.so.v0.2.0_T4_5` md5 `9ee93ec8958ab0de868c0bc0506c1cfa`
- 770-line IST-DASLab Marlin kernel ported via NVRTC (sm_90, 18.4s cold compile);
  GPU-side quantize+transpose; host-side Marlin XOR-swizzle repack
- Three architectural learnings:
  1. Sync heavy work on hot path is fatal. CPU quant draft = 80 min for
     full model; GPU kernels brought it to ~10s.
  2. PyTorch's cuBLAS A operand is the weight (col-major (out,in)).
     Marlin wants (K_in, N_out) row-major — added GPU transpose.
  3. Marlin grid=132 deadlocks on small N/K. Gate at N≥1024, K≥1024;
     ~45% of TinyLlama matmuls qualify.
- Accuracy isolated Linear 2048×2048: cos=0.997 rel_err=0.073 (within
  INT4 noise floor; PASS).
- Accuracy full TinyLlama logits: cos=0.94 rel_err=0.34, argmax FLIPS.
  Quantization error compounds across 22 layers + lm_head on a small
  model (TinyLlama 1.1B less redundant than Mistral 7B).
- Throughput TinyLlama B=1 decode (3 matched pairs, 60s):
  tok/s 67.0 → 45.3 (**−32.4%**), tok/W 0.444 → 0.325 (**−26.9%**).

**Why the regression: Marlin operating outside designed envelope.**
Prior cipher-may13-evidence 1.62× tok/W was on Mistral-7B at B=8. On
Mistral-7B at B=1 the prior measurement was 1.03× tok/s (already
marginal). TinyLlama at M=10 is GEMV territory which Marlin doesn't
optimize for. **The substrate did its job — routed cleanly, gated
honestly, surfaced the regression. The regression is in Marlin's
target-regime mismatch, not in the substrate.**

**How to apply:**
1. The substrate is shippable today; future actuators (FP8, fusion,
   spec-decode, per-tenant) plug in via `cipher_rt_matmul_register_actuator`.
2. **Do NOT enable CIPHER_MARLIN=on in production until Mistral-7B
   B≥8 regime is re-validated on current dispatch.** Marlin's product
   number must come from its designed regime.
3. Next session priority: Phase 4.6 KV dedup (multi-tenant moat).
   Separately, ~2h to validate Marlin on Mistral-7B B=8 to either
   confirm the prior 1.38× tok/s or document the perf gap honestly.

**Substrate composition story (the strategic anchor):**
Phase 4.2 GREEN_CTX (substrate, validated capability) +
Phase 4.3 VOLT (efficiency moat, +57% tok/W shipped) +
Phase 4.5 matmul-routing substrate (deployment moat, Marlin first actuator) +
Phase 4.6 KV dedup (multi-tenant moat, next) +
Phase 4.7 fusion + agentic +
Phase 4.8 composed 24h soak = the marvel demo.

Linked: [[cipher-phase-discipline]], [[cipher-lift-framing]],
[[cipher-t431-volt-shipped]], [[cipher-t432-kmod-volt-ioctl]].

**Path B Mistral-7B validation attempt (2026-05-14 evening):**
FP16 baseline on Mistral-7B verified through substrate (B=1 39ms/652 tok/s,
B=8 84ms/2391 tok/s; 450 cublasGemmEx routed cleanly). Marlin smoke runs
on Mistral-7B HUNG at first eligible decode iteration — terminated at
15-min timeout. Same root cause as TinyLlama scaled larger: synchronous
quant pipeline + GREEN_CTX 8-SM constraint + 225 sequential weights =
infeasible first-call latency.

**Three durable substrate-contract findings (for next-session re-port):**
1. Actuator contract should mandate µs maybe_handle() budget OR async
   warmup. Synchronous heavy work on cuBLAS hot path is fatal.
2. Setup kernels should push primary context (not inherit GREEN_CTX 8-SM).
3. Pre-deployed weight kit (offline `cipher_marlin_prequant /path/to/model`)
   is the operator-deployment-correct pattern.

**Recommended next-session path: Option B (offline pre-quant kit, ~5h).**
Decouples Marlin port from PyTorch's lazy weight loading; zero hot-path
quant latency forever; cleanest deployment story.

Phase 4.5 final close: substrate ships clean as the architectural deliverable.
Marlin's product number deferred to fresh session with concrete scope.
Tarball v2: `cipher_rt_phase4_src_T4_5_v2.tar.gz` md5 `526e068153133bd2298c9b2189c6c864`.
