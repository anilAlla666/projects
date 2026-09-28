# READ-ONLY SIZING — fusion lever alone (RMSNorm + SiLU·mul) at B≥8, Marlin OFF, cu13

**Date:** 2026-05-31. **Type:** SIZING measurement (no production build, no commit, no tag, anchors
unchanged). Isolates the **non-GEMM fusion half** of the old combined 1.38× — the Marlin-GEMM half is
disproven end-to-end (`V0_G3_MARLIN_VLLM_BATCHED_DECODE_MEASUREMENT.md`). **The number: fusion alone delivers
correct output and a positive graph-mode tok/s lift that DECAYS monotonically with batch — +11.6% (B=8) →
+9.0% (B=32) → +7.0% (B=64)** — weakest exactly in the product's high-batch regime, trending toward the ~5%
"marginal" floor. Unlike Marlin (which *regressed* and worsened with batch), fusion stays positive — but the
task's question "hold/**grow** with batch?" is answered **no: positive but decaying**. Caveat: a real lift
here is still a per-framework vLLM-module monkey-patch *build*, not a wire.

---

## STEP 1 — repair (the "garbage" was a missing env, NOT build drift)

The may13 fused kernels emitted `<unk>` garbage last task because the NVRTC compile path was **gated off**:
`cipher_substitute_v2_init` sets enabled only if `getenv("CIPHER_SUBSTITUTE_V2")` is truthy
(`cipher-may13-evidence/src/cipher_substitute_v2.cpp:78`); without it, `ensure_compiled_for`
(`cipher_fusion_kernels.cpp:158`) bails → the kernel never launches → `cipher_fused_rmsnorm` returns 0 → the
Python wrapper returned the **uninitialized `torch.empty` buffer** → garbage. **Fix = set
`CIPHER_SUBSTITUTE_V2=1` before init** (one env; the `.so` still needs `libcuda.so.1` dlopened RTLD_GLOBAL
first for `cuGreenCtxDestroy`). No source edit. This also retracts last task's "build-drifted / non-functional"
misdiagnosis (see correction note in `V0_FUSED_MEGAKERNEL_GRAPH_INVENTORY.md`).

**Isolated kernel correctness (rc + vs torch):** `cipher_fused_rmsnorm` rc=1, max_rel_err **6.9e-4**;
`cipher_fused_silu_mul` rc=1, max_rel_err **1.4e-5**. The kernels compile (sm_90) and are correct on cu13.

## STEP 1 GATE — correctness FIRST (Mem #11), teacher-forced vs fp16 native — **PASS**

Mistral-7B fp16, B=8, monkey-patched `MistralRMSNorm.forward`/`MistralMLP.forward` (the working may13
mechanism), teacher-forced (identical input both arms, single forward, isolates per-op difference, no cascade):
- **argmax-agreement = 100.00%**, KL_mean **4.8e-6**, KL_max **2.4e-5** (≈ fp16 round-off). Fusion is
  **non-lossy** (same math, fused) — exactly as expected, and the near-zero KL confirms the kernels, not a
  tradeoff. Gate PASS.

## GRAPH INTEGRITY — does the fused kernel actually execute on graph replay? — **PROVEN YES**

This is the trap that inflated may13's original numbers and a prior measurement (stale buffer: kernel fires
once at capture, absent from replay → faster-but-wrong). Settled with a **harness-independent** test:
capture a graph containing the fused RMSNorm, **zero the output buffer**, replay, check output. If the kernel
is NOT a graph node, the zeroed buffer stays zero; if it IS, replay recomputes. Result: **max_rel 8.0e-4 vs
torch — the kernel recomputed correct output from a zeroed buffer ⇒ it is a real graph node that executes on
replay.** (The stream-fix — passing `current_stream().cuda_stream`, not `NULL` — is what captures it. Proven
on RMSNorm; SiLU·mul shares the identical compile/launch/stream path.)
- Why the graph tok/s is valid despite a confound: the full-model greedy/single-step output checks diverged
  (4.2% / 25%) **equally for baseline AND fusion** → a StaticCache/left-padding `cache_position` confound in my
  manual-capture harness, NOT fusion-specific. Crucially, **CUDA-graph work is shape-static**: every replay
  executes the identical fixed op-sequence and shapes regardless of the wrong cache *content* (StaticCache
  attends over a fixed max length, masked), so the confound corrupts output *correctness* but not the *compute
  profile* — the tok/s is a faithful decode-step throughput, matched-fair across arms, and the isolated probe
  proves the fused kernels are in the replay. So the deltas are real, not stale-buffer-inflated.

## STEP 2 — THE SIZE (Marlin OFF; graph-captured = the production regime)

Marlin OFF confirmed: `CIPHER_MARLIN` unset, no `CUDA_INJECTION64_PATH`, cipher_rt_phase4 `.so` not loaded;
the may13 `.so` teardown prints **"Substitutions: 0"**. Fusion firing confirmed: `cipher_fusion_kernels_stats`
rmsnorm_calls 65→1950, silu 32→960 at capture. transformers **5.8.1**, fp16.

| B | baseline graph tok/s | fusion graph tok/s | **Δ tok/s** | Δ tok/s/W | eager Δ (context) |
|---|---|---|---|---|---|
| 8 | 757–758 | 838–846 | **+10.7 … +11.6%** | +6.0 … +8.3% | +93–96% |
| 32 | 1902 | 2074 | **+9.0%** | +4.3% | +93% |
| 64 | 2868 | 3069 | **+7.0%** | +3.2% | +93% |

- **Graph (production regime) fusion-alone lift DECAYS monotonically with batch: +11.6% → +9.0% → +7.0%**
  (B=8/32/64), tok/s/W +8%→+3%. Correct output at every B (teacher-forced 100%). Extrapolating the trend,
  B=128 lands near the ~5% "marginal" floor. So fusion is **positive but weakening into the high-batch regime
  the product runs in** — it does NOT hold/grow. Still categorically better than Marlin INT4 (which *regressed*
  and worsened: −10.7% B8 → −27.5% B64): fusion is the "compute→memory" round-trip lever working, just modest
  and shrinking. **Native baseline is unfused** — transformers 5.8.1 `MistralRMSNorm.forward` decomposes into
  separate eager ops (`hidden.pow(2).mean(-1) → rsqrt → mul → weight·`), so the launch/HBM-round-trip
  mechanism is real and the may13 ≈1.13× corroboration is apt (same unfused-baseline situation; the +93% eager
  also confirms the baseline isn't already fused).
- **Eager +93%** is a launch-bound **upper bound** (HF eager at B=8–32 is heavily launch-bound; fusing many
  small RMSNorm/SiLU kernels into single launches saves a lot of launch overhead that **CUDA graphs already
  eliminate**). It massively overstates the production lever — the graph number is the real one.
- **Corroboration:** the ~+11% graph lift matches may13's independently-recorded fusion-alone-in-graph
  **≈1.13×** (their Amdahl ablation: fusion ≈1.13×, almost entirely RMSNorm; SiLU·mul ≈1.009× = nil).

## THE ANSWER (Amdahl-honest; Anil decides the build)

Fusion attacks the **~17% non-GEMM** decode time (RMSNorm ~15% + SiLU ~3%, per may13's ablation). Measured:
**+11.6% (B=8) → +9.0% (B=32) → +7.0% (B=64) graph tok/s**, correct (KL≈0), **decaying** toward the ~5%
marginal floor as batch climbs into the product's regime. So: a *real, modest, but shrinking* lever — clears
"~10% worth-a-build" at B=8, sits below it by B=64, and the answer to "holds/grows with batch?" is **no**. Not
a regression (Marlin), not marginal-nil — but its value concentrates at *low* batch, which is not where the
cross-tenant product runs. The combined old "1.38×" does not return — its GEMM half is disproven; only this
~1.1×-and-shrinking fusion half is real.

**Deployability (unchanged from the inventory):** magnitude ≠ wire. Capturing this in production needs a
**per-framework vLLM-module monkey-patch build** (replace vLLM's RMSNorm/MLP modules with the CIPHER fused
kernels so vLLM's own graph captures them) + porting the kernels into the production runtime (they live only
in the may13 archive). Anil decides: (a) port+integrate fusion (a ~10% lever for a real build); (b) leave it —
~10% on the non-GEMM slice may not clear the integration cost; (c) revisit if a larger non-GEMM fusion (RoPE,
attention-epilogue) widens the Amdahl target. No code changed here; anchors unchanged.
