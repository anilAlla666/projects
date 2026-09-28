# FINDING — fusion is NOT Phase-1 cargo for vLLM: the build's premise collapses on audit (DO NOT BUILD)

**Date:** 2026-05-31. **Type:** READ-ONLY finding that REFUSES the requested fusion-port build, with cited
evidence (no source change this turn; the Marlin graph gate `bb0d42b`/`graph-gate-step1` stands; anchors
unchanged). **Verdict: do not port fusion through the gate. The task's three premises each fail on contact
with the evidence; the honest next op is an actuator that can actually beat vLLM's already-optimized production
kernels — a strategic fork for Anil, not a port.**

---

## The three premises, each refuted

**1. "Push fusion through the *capture-safe* gate" — there is NO capture-safety work to do.** Audit
(`fusion-graph-gate-audit` workflow): `cipher_fused_rmsnorm` (`cipher_fusion_kernels.cpp:226`) and
`cipher_fused_silu_mul` (`:243`) are **already pure kernel-launch on the passed stream** — no per-call
`cudaMalloc`/`cudaFree`/`cuCtxSetCurrent`/`cuEventCreate`/`cudaStreamSynchronize`; output is caller(torch)-
allocated; NVRTC compiles **once per device** and is cached (`ensure_compiled_for`, `:152-185`). Unlike Marlin
(which needed async-alloc + ctx-guard-skip + workspace-reuse surgery), fusion has **zero capture-hostile ops**.
The verb "make capture-safe" has no referent. (Confirmed already-graph-captured in `V0_FUSION_LEVER_SIZING.md`
via the zero-buffer-replay test.)

**2. "Fusion DELIVERS ~9% in-graph" — that lift does NOT exist in vLLM, because vLLM already fuses these ops.**
Direct from vLLM 0.20.2 source:
- RMSNorm → `ops.fused_add_rms_norm` (`vllm/model_executor/layers/layernorm.py:56-64`), `@CustomOp.register("rms_norm")`
  (`:102`) — a single fused custom CUDA kernel (and it fuses residual-add too).
- SiLU·mul → `torch.ops._C.silu_and_mul` (`vllm/model_executor/layers/activation.py:133`),
  `@CustomOp.register("silu_and_mul")` (`:117`) — already fused.

CIPHER's fused RMSNorm/SiLU fuse the **same** ops → replacing vLLM's hand-tuned `_C` kernels with NVRTC fp16
equivalents = **parity at best, likely slightly worse**. The ~9% (B=8) → 7% (B=64) I measured in
`V0_FUSION_LEVER_SIZING.md` was vs **unfused HF-eager** (torch-HF Mistral 5.8.1: RMSNorm decomposed into
`pow.mean→rsqrt→mul→mul`) — an unfused-baseline artifact, not the production stack. The fusion sizing's own
advisor caveat ("if the native baseline is already fused, the lift is coincidental") is now **confirmed** for
vLLM.

**3. "Port it into the production runtime" — that produces DEAD CODE.** cipher_rt_phase4 has no fusion (OBJS
grep=0) and CAN host it (reusable `nvrtc_compile_to_module`, no may13 dependency). But fusion is **monkey-patch,
not substrate-line** (RMSNorm/SiLU decompose into N kernels — no symbol to GOT-patch; the documented wall,
`V0_FUSION_LEVER_SIZING.md:90`). So a ported kernel is callable only by a vLLM-module monkey-patch that does not
exist — ~150 LOC added to the production `.so` that nothing invokes, re-proving an already-green test, 0% closer
to production. Per the standing discipline (don't call unwired things delivered), this is busywork, not cargo.

## The real state of Phase-1 cargo — scoped precisely (the strategic finding)
The graph gate **mechanism** is proven (Marlin, `graph-gate-step1-marlin-capture-safe`). The refuted actuators
are all **COMPUTE-KERNEL** levers — exactly the category vLLM has already optimized:
- **Marlin INT4** — substitutes but **regresses** end-to-end (−32%→−53%, `V0_G3_…`) + fails KL=0; vLLM also has
  Marlin-for-quantized.
- **Single-op fusion (RMSNorm, SiLU·mul)** — **redundant**: vLLM ships `fused_add_rms_norm` + `silu_and_mul`.
  ~0 lift. *Caveat:* vLLM fuses INDIVIDUAL ops, NOT **cross-op megakernels** (RMSNorm+RoPE+QKV, or the may13
  `int4_silu_mul` GEMV+act). The only fusion with headroom is the **cross-op megakernel** — which is the
  build-drifted, never-ran-on-cu13 may13 work (`V0_FUSED_MEGAKERNEL_GRAPH_INVENTORY.md`): unvalidated, high-risk.
- **FP8** — quality-failed (PPL +0.567%, `D9_FP8_ACTUATOR_BUILD_REPORT.md`); AND vLLM supports fp8 natively, so
  it must beat **vLLM's fp8**, not just bf16. Partially redundant.

**This is NOT "CIPHER has no cargo" — it is "the compute-kernel actuator thesis is exhausted against vLLM."**
vLLM optimizes **single-model compute** (fused norm/act, fp8, paged attention, continuous batching, its own
CUDA graphs). It does **NOT** subsume CIPHER's **memory-capacity / multi-model-density axis** — KV-dedup,
cross-tenant weight-sharing, N-distinct-model residence (the 100-heterogeneous-model goal; Track 2/3; B2-A
same-model full lift). That is a different axis vLLM gives you nothing on, and it is where CIPHER's prior
evidence actually pointed.

## Decision (for Anil — "moving forward we do architectural decisions")
- **DO NOT build the fusion port.** Over-determined: (1) no capture-safety work exists (audit), (2) dead code
  (no invoker), (3) ~0 vLLM lift (vLLM already fuses). Don't even build to "confirm ~0" — that's the busywork.
- **Re-aim Phase 1 off the compute-kernel axis (exhausted vs vLLM) onto the memory/density axis** vLLM does not
  subsume: N-distinct-model residence, KV-dedup, cross-tenant weight-sharing. The graph gate stays useful there
  too (those levers also run under vLLM's graphs).
- Narrow compute-kernel angles that remain, both hard: (a) a **cross-op megakernel** that beats vLLM's
  per-op-fused kernels (may13 build-drifted, unvalidated); (b) **FP8 that beats vLLM's fp8** at acceptable
  quality (separately gated on the PPL fix). Neither is a safe Phase-1 default.
- The port is trivial if Anil wants to *stage* kernels production-side — but it stages for an integration that
  itself delivers ~0, so it is staging-for-nothing; not recommended.
- No code changed this turn; the Marlin graph-gate commit/tag stands; deployed anchor `1f305ce6` unchanged.
