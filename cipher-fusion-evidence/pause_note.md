# CIPHER fusion — PAUSE NOTE

**Date:** 2026-05-15. Fusion paused at op #2. Build pivoting to T4.6.3
KV dedup (a different prompt). This is the document to read when fusion
resumes.

---

## (a) Op #1 — S2b allocator integration — SHIPPED

`cipher_rt_kv_alloc.{h,c}` (VMM 2 MiB page allocator, tagged pages) +
`cipher_kv_bridge` (libtorch `from_blob` extension, md5
`2cf82c06f3a7a4732ba59abb58f93b9c`) + `cipher_kv_cache.py`
(`CipherStaticLayer` / `CipherStaticSlidingWindowLayer`, operator-injected).

Gate passed: 3 prompts × 32 tokens byte-identical baseline vs CIPHER,
both the sliding-window and plain `StaticLayer` paths; allocator unit
test 14/14; page tags read back correct; `memory_allocated` KV delta
4.9 MiB → 0.0 MiB (KV is CIPHER-VMM-owned).

**What it gates / enables:** CIPHER owns and per-page-tags the KV cache
memory. This is the foundation T4.6.3 (in-process dedup) and T4.6.4
(cuIpc cross-tenant pool) build on. **Crucially — S2b owns KV memory by
*being the cache class*, not by intercepting calls.** It is therefore
NOT subject to the coverage problem in (c): the CIPHER cache holds the
KV tensors whether or not the decode is compiled. T4.6.3 built on S2b is
coverage-immune. This is why pivoting to T4.6.3 now is sound.

Requires `cache_implementation=static` (operator-policy default; customer
code unchanged). Per-tenant opt-out `CIPHER_KV_ALLOC=0`.

## (b) Op #2 — Op 9 AUDIT — DIAGNOSED, NOT SHIPPED

AUDIT was built (`cipher_rt_audit.{h,c}`, HMAC-SHA256 tamper-evident
chain, registers priority-0 observer actuators on the matmul + attn
substrate registries). Two problems surfaced at the gate:

1. **FakeTensor crash (fixed).** The T4.6.1 attention trampolines called
   `tensor.data_ptr()` unconditionally. Under `StaticCache` (op #1's
   requirement) transformers runs `torch.compile`; dynamo traces SDPA
   with FakeTensors; `data_ptr()` on a FakeTensor throws → host crash.
   Pre-existing T4.6.1 bug — its own smoke used `DynamicCache` (no
   compile) so the path was never exercised. **Fixed:** FakeTensor guard
   in `cipher_rt_attn_dispatch.cpp` (dispatch-key check + try/catch
   safety net); traced calls pass straight through. Re-run: no crash
   (exit 0), 3-prompt output byte-identical to baseline, HMAC chain
   independently re-verified over all 2346 entries (recomputed head ==
   live head). `libcipher_rt.so` md5 `4f5cf543439d64a34b18742faef2ce93`.

2. **Coverage gap (the blocking finding — see (c)).** AUDIT registered
   and recorded correctly, but the substrate it observes sees only a
   fraction of the workload. AUDIT's chain is honest over *what it sees*;
   what it sees is the problem.

**AUDIT status — restated, do not conflate with "broken".** AUDIT the
component is **functional and verified**: it registered on both substrate
registries, recorded 2346 events, and the HMAC-SHA256 chain was
independently recomputed and matched the live head exactly. The 43% is
the **coverage of the substrate AUDIT sits on**, not a defect in AUDIT.
AUDIT is shippable, with this honest scope label:

> "AUDIT records substrate-visible dispatches with HMAC tamper-evidence.
> Substrate visibility under `StaticCache`+`torch.compile` is 43% (eager
> prefill only); the coverage gap is upstream and tracked separately."

What is *not* done is fusion proceeding to op #3 — that is gated on the
coverage fix, not on AUDIT. "Op #2 paused" means the fusion sequence is
paused at op #2, not that AUDIT is wrong.

## (c) Coverage measurement

Method: raw atomic counters in the attn trampolines (`g_tramp_calls`,
`g_tramp_fake`) + the existing matmul substrate counter. Workload:
3 prompts × 6 new tokens = 3 × (1 prefill + 6 decode) = 21 forwards.
Expected substrate dispatch events at full coverage ≈ 21 × 257 ≈ 5400.

| Substrate | Observed | Expected (full) | Coverage |
|---|---|---|---|
| matmul (cuBLAS shim) | 2250 | 4725 | **48%** |
| attention (SDPA `.symver`) | 96 | 672 | **14%** |
| **total** | **2346** | **5397** | **43%** |

attn `tramp_calls=192` = 96 real (the 3 eager prefills × 32 layers) +
96 fake-exempted (dynamo's compile-time trace pass). **Zero of the 18
compiled-decode forwards' attention reached the trampoline.** matmul
fares better (~half) but still loses ~half of decode.

**Root cause:** the T4.5.1 / T4.6.1 substrate interposes ATen op-call
symbols (`.symver` on `cublasGemmEx`, `_scaled_dot_product_*::call`).
That layer sits **above `torch.compile`**. Compiled / cudagraph-replayed
decode does not re-enter those symbols — attention catastrophically
(14%), matmul badly (48%). The substrate is substantially blind to
compiled decode.

This is "case 5" from the op #2 plan. It is **upstream of fusion**:
porting more ops (Op 3 SUBSTITUTE, etc.) onto a 14–48%-visible substrate
means every actuator silently operates on a fraction of the workload.
Better found at op #2 than at op #7.

## (d) Recommended resume condition

**Fusion of interception-based actuators (Op 3 SUBSTITUTE and beyond)
must not resume until the substrate observes ≥95% of decode dispatch
events.** The aten-op `.symver` layer structurally cannot reach that —
measured, not assumed.

Paths, in recommended order — a coverage investigation must pick one
before fusion resumes:

- **Path A (recommended): hook at the driver kernel-launch layer.**
  Interpose `cuLaunchKernelEx` + handle `cuGraphLaunch` via graph-node
  inspection — **below** `torch.compile`. This is exactly what
  op31-prod's `libcipher_hook.so` already does, and op31-prod measured
  real decode gains (1.38× B=8) at that layer. Implication: the correct
  fusion is op31-prod's *hook layer* as the substrate, with the Phase 4
  `.symver` layer kept only for non-compiled / framework-direct paths.
  The Phase 4 aten-op substrate is not the right primary for compiled
  inference.
- **Path B: operator bootstrap forces eager execution.** Restores 100%
  coverage for the Phase 4 substrate but disables the customer's
  `torch.compile` decode speedup — a real performance cost. Likely
  unacceptable as a default.
- **Path C: register CIPHER interception as opaque torch custom ops** so
  the compiled graph keeps calling them (the FakeTensor error message
  itself suggests this). Larger rearchitecture; survives compile by
  construction.

**Note:** this coverage problem affects *interception-based* actuators
only. Op #1 S2b and T4.6.3 KV dedup own KV memory by being the cache
class — they are coverage-immune. The pivot to T4.6.3 correctly builds
on the part of the foundation that is not broken.

## Architectural distinction — interception vs data-ownership layers

The coverage finding splits every fusion port into two layers. Each op's
per-op PROPOSE must declare which layer it is in; the resume criterion
differs.

**Interception-layer ports** — observe or substitute *someone else's*
calls (they hook a dispatch path they do not own). Examples: AUDIT,
Op 3 SUBSTITUTE, Op 1 CLASSIFY, RING_WRITE, GENERATE, the Marlin
actuator, the cuBLAS shim, the SDPA substrate, the overlay observers.
- **Coverage-bound.** Effectiveness == fraction of the workload the hook
  sees. The `.symver` ATen-op layer sits above `torch.compile` and
  measured 43% (14% attention). 
- **Resume criterion:** an interception-layer op may only ship once it
  runs on a **cuLaunchKernel-layer hook** (op31-prod's
  `libcipher_hook.so` path — `cuLaunchKernelEx` + `cuGraphLaunch`
  inspection, *below* `torch.compile`), **not** on the `.symver`
  ATen-op layer. Ship gate additionally requires measured decode
  coverage ≥95%.

**Data-ownership-layer ports** — own a resource directly (they *are* the
allocation / the state, they do not hook a call). Examples: op #1 S2b
KV allocator + cache class, T4.6.3 L1 KV dedup, T4.6.4 L3 cuIpc pool,
KV compression, the VMM pool, REMEMBER (CfC h-state).
- **Coverage-immune.** They hold the data whether or not the decode is
  compiled — there is no call to miss.
- **Resume criterion:** standard ship gate (three-indicator binding
  diagnostic + byte-identical/documented-tolerance). **No coverage
  constraint.**

This is why T4.6.3 can proceed now and op #3 cannot: T4.6.3 is
data-ownership (builds on S2b's cache class); Op 3 SUBSTITUTE is
interception and is blocked until the cuLaunchKernel-layer hook exists.

## Discipline state

- Fallback anchors `55ab8c0c…` / `86618c30…` — unchanged. Taint 12288.
- `libcipher_rt.so` current md5 `4f5cf543439d64a34b18742faef2ce93`
  (pre-op2 `libcipher_rt.so.pre_op2_audit` retained).
- Op #2 artifacts: `cipher_rt_audit.{h,c}`, `audit_verify.py`,
  `op_2_AUDIT.md`. AUDIT code is sound and re-usable once coverage is
  fixed — it is the *substrate* under it that is paused, not AUDIT.
