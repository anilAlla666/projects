# ============================================================================
# ARC CLOSED — 2026-06-13. See SESSION_CLOSEOUT_2026-06-12.md for the full close-out.
# ============================================================================
# CAPABILITY PROVEN: zero-plugin FP8 SDC detection works under DEFAULT config (MP + cudagraph ON),
#   injection-only (no pip/plugin). Always-on in-graph EXACT RECOMPUTE = validated: 100% catch, 0 FP,
#   default cudagraph (Add.4; cg2gate_on.json). This is a real, differentiated, deployable capability.
# COST FLOOR: always-on = -51.9% on decode (MEASURED, Add.4). The faithful-EVERY-TOKEN floor is ~2x
#   (dual-modular-redundancy) — matches the SDC literature (Google/Meta/NVIDIA/OCP) and proven on our
#   stack 5 ways below.
# ALL CHEAP-DETECTION PATHS EXHAUSTED:
#   - Periodic dual-graph:        WALLED  — vLLM caches one compiled forward (Add.5).
#   - Freivalds checksum:         WALLED  — sqrt-N dilution; misses single-element SDC (mantissa 0%,
#                                           bit-14 75%); +44.7% under cudagraph (Add.6).
#   - ABFT separate-kernel:       WALLED  — at M=1 decode the column-checksum IS a full recompute
#                                           (+928% fp32 / ~+100% fp8), strictly dominated (Add.7).
#   - Out-of-band periodic:       COST OK (-8.9% @ N=64 measured; ~1% projected @ N=512 via free bf16
#                                           copies, +0.1% measured) but CATCH unvalidated under cudagraph
#                                           4x (Add.9/10/12 + Add.13 v2/v3). LOGIC validated in EAGER
#                                           (Add.13 v4: 0 FP, caught, localized). Lone unblocked test =
#                                           a CUDA .so corrupting out's DEVICE memory directly.
#   - Fused-ABFT (own the kernel):KILLED at GATE 1 (Add.11) — ownable fusable Triton FP8 GEMM is 3-4x
#                                           slower than CUTLASS on every shape; CUTLASS/cuBLASLt match
#                                           but are non-fusable black boxes.
# STRATEGIC VERDICT: SDC detection is an OPTIONAL/insurance readout for the neocloud buyer; nobody pays
#   ~52% throughput for optional insurance (a 1% optional feature and a 52% optional feature share the
#   same flaw: optional). We hardened the hardest-to-sell readout first. The DRIVER-LEVEL POSITION is
#   intact; the detector being non-sellable says nothing about the position. Detector CLOSED as
#   "works, not the product." NEXT: point the driver position at NON-OPTIONAL readouts — TPW (clock
#   actuator, +57.28% tok/W measured single-GPU matched-pairs; power is a mandatory neocloud
#   constraint) and/or multi-tenant multiplexing (margin lever, biggest prize, least validated).
#   Decided by a buyer conversation (Devang / Nebius / one neocloud), not another pod experiment.
# ============================================================================

# FP8 DETECTION SEAM — keystone build, 2026-06-12

Goal: make the SDC detector see FP8-served linears (audit 2026-06-12 found ~93% of FP8 target
weight-FLOPs blind at the cuBLAS GemmEx seam, HEADROOM_2X_RESULT.md:104-140). Anchor
`2edba0d2136f8ede4713d90a8f7cd55f` at start; re-checked at end (NO rebuild — this is all
launcher-side Python, the .so is untouched). Files under cipher-fusion-evidence/fp8seam/.

## Phase 0 — recon + A/B/C transparency fork

Target checkpoint (already on pod, NOT downloaded): `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8`,
compressed-tensors naive-quantized FP8 e4m3, per-tensor static act+weight, lm_head ignored
(config.json quantization_config). On sm90 the FP8 linears route
`Fp8LinearMethod.apply` -> `CutlassFP8ScaledMMLinearKernel` -> `ops.cutlass_scaled_mm`
(_custom_ops.py:875-923) -> `torch.ops._C.cutlass_scaled_mm` (vllm._C cutlass), NOT cuBLAS.

| where FP8 linears dispatch | evidence | seam sees it? |
|---|---|---|
| cuBLAS GemmEx (the existing detector's seam) | rvcount.so LD_PRELOAD on this checkpoint: 0 target-linear GemmEx calls (rvc_fp8probe*.* all gemmex:0); audit cov_bundle: target qkv [6144,*,4096] ENTIRELY ABSENT | NO (~0%, the blindness) |
| `vllm._custom_ops.cutlass_scaled_mm` | launcher monkeypatch captured 2048 calls / 8 shapes in 16 decode tokens = all 4 fused linears (qkv 6144x4096, o 4096x4096, gate_up 28672x4096, down 4096x14336), prefill+decode (fp8probe2.json) | YES (100% of the FP8 linear path) |

**A/B/C verdict (evidence-based, not convenience):**
- **A (pure C-ABI GOT injection, founding substrate): DEAD.** The FP8 matmul is a torch-dispatcher
  custom op inside vllm._C; it never resolves cublasGemmEx/cublasLt through the caller's GOT, so
  the existing GOT hooks see 0 (rvcount: 0). Proven, not asserted.
- **B (coop-hook-class launcher monkeypatch of cutlass_scaled_mm, ZERO vLLM source edits): WORKS.**
  Captured 2048/2048 linears. SAME mechanism class and SAME operating condition as the already-
  approved -3.9% cudagraph coop hook: requires `VLLM_ENABLE_V1_MULTIPROCESSING=0` (engine
  in-process). With default multiprocessing ON, the launcher patch is in the parent and the worker
  is a spawn child -> 0 captured (fp8probe.json, the negative control). This is the in-process
  proviso, identical to ra_coophook.
- **C (formal vllm.general_plugins entry point / source edit): not needed for the prototype.** It
  is the route to reach the worker under DEFAULT multiprocessing; flagged as the deployment
  blocker below.

Resolved to **B**. Proceeded per the gate (stop only on C/ambiguous).

## Phase 1 — the seam (cipher_fp8_seam.py)

Wraps `cutlass_scaled_mm`, reusing the detector contract verbatim from detector_shim.c:99-117:
shape predicate k in {4096,14336} && m,n != vocab (det:81); GPS=128 linears/step (4 fused x 32
layers); every N steps independently recompute D' and compare residual `(D'-out).norm()`; clean
residual exactly 0 at T=0, any nonzero on a clean linear = FALSE POSITIVE (det:115); persistent
output bit-perturbation injection on a chosen ordinal (det:89-97). Contract honored
(cipher_rt_matmul_dispatch.h:72,82-87): check work is PERIODIC; non-check steps add only a counter
increment, no sync, no GPU work on the hot path.

## Phase 2 — the gate (real FP8 checkpoint, eager, in-process)

| metric | result | artifact |
|---|---|---|
| **COVERAGE** | FP8 linear FLOPs seen: **~0% baseline -> 100%** of the decoder linear path. 49,152 covered_linears == 49,152 csmm_total (every cutlass_scaled_mm call IS a target linear); all 4 fused projections, 32 layers, prefill+decode | seam_clean.json shapes_seen |
| **CORRECTNESS (0-FP)** | **0 false positives over 6,144 clean comparisons**; max_clean_resid exactly 0.0 (clean run) AND 0.0 on the 6,096 non-target checks during the inject run | seam_clean.json, seam_inject.json |
| **FAULT CATCH** | injected persistent output perturbation on ordinal 3 (a previously-blind FP8 linear): caught at **first check step, latency 0 within N=8**; 48/48 check steps detected, max_detect_resid 0.505, 0 FP on the co-checked clean linears | seam_inject.json |
| **OVERHEAD eager N=8** | OFF 66.99 -> ON 58.3 tok/s = **-12.97%** (256 tok) | cell_off.json, cell_on.json |
| **OVERHEAD eager N=45** | OFF 66.99 -> ON 62.76 tok/s = **-6.31%** (256 tok), 0 FP / 2,304 clean | cell_on45.json |
| **CUDAGRAPH** | **WALL (mechanism identified).** enforce_eager=False crashes in `torch._dynamo`: "Could not guard on data-dependent expression zuf0 > 0.0" — the seam's host-side `resid > 0.0` branch + `.item()` is not dynamo/inductor-traceable inside vLLM's compiled region. This is the EXACT limitation the 2026-06-12 audit predicted (HEADROOM §4: detector ops must be fully dynamo-traceable, no host branches). The eager seam cannot run under torch.compile/cudagraph as-built. | cell_cg.err |

**Transparency verdict:** ran with ZERO vLLM source-file changes (launcher monkeypatch only), but
required `VLLM_ENABLE_V1_MULTIPROCESSING=0` (the in-process condition the existing coop hook also
uses). Not a vLLM fork; not the default multiprocessing config.

## Bottom line

FP8 detection is **DEPLOYABLE-WITH-CAVEAT today in EAGER mode**, transparently (zero vLLM source
edits, in-process): coverage moved **0% -> 100%** of the FP8 decoder linear path, **0 false
positives over 6,144 clean comparisons**, fault on a previously-blind FP8 linear **caught at the
first check step**, **6.3% overhead at N=45** (eager). The keystone claim "the SDC detector can see
FP8-served linears" is now MEASURED-true, not projected.

Two gaps stand between this and a default-config neocloud deployment, and they compound into ONE
next blocker:
1. **Multiprocessing:** the transparent launcher monkeypatch only reaches the worker with
   `VLLM_ENABLE_V1_MULTIPROCESSING=0` (same condition the existing coop hook uses); default
   vLLM spawns the engine in a child, where the patch is absent (proven negative control,
   fp8probe.json). Reaching the default-MP worker needs a `vllm.general_plugins` entry point.
2. **Cudagraph:** neoclouds run cudagraph ON for throughput; the eager seam hits the dynamo
   data-dependent-guard wall there. The fix is the existing ra_coophook dual-full-graph
   capture-time design (recompute into a GPU buffer, read residual out-of-band, NO host branch in
   the compiled region) ported to the cutlass_scaled_mm seam.

**SINGLE next blocker:** a dynamo-traceable FP8 detector (recompute-to-buffer, out-of-band read)
shipped as a `vllm.general_plugins` entry point — that one artifact reaches the default-MP worker
AND survives torch.compile, closing both gaps. This is path C, the deployability decision Anil
flagged for adjudication: the transparent (no-plugin) version works ONLY in the eager + in-process
operating mode demonstrated here.

Anchor at end: `2edba0d2136f8ede4713d90a8f7cd55f` — unchanged (no rebuild; launcher-side Python
only).

---
## ADDENDUM — INJECTION-ONLY worker reach (zero plugin), Phase-0 recon, 2026-06-12

Constraint (Anil, non-negotiable): NO vllm.general_plugins entry point, NO pip install, NO
customer-side install. Sole delivery = the injected .so in CUDA_INJECTION64_PATH / LD_PRELOAD (the
surface the cuBLAS GemmEx hook already uses). Question: can that .so make the detector see FP8
cutlass_scaled_mm linears in the DEFAULT-MULTIPROCESSING spawn worker? Anchor 2edba0d2 entry==exit;
probe is scratch inj_pyhook.so, anchor .so untouched.

### 0a worker process model + .so reach
The existing substrate already reaches the worker by this surface: cipher_inject.c:116
(InitializeInjection2, driver-called at cuInit) + :131-134 (__attribute__((constructor)) fires in
every forked/spawned process incl. the V1 EngineCore worker). The comment at cipher_inject.c:124-129
states this exact Goal-5 contract ("customer set only LD_PRELOAD... no plugin install").
MEASURED: default-MP run, EngineCore worker pid 3559312 (parent=launcher 3559195); injected .so's
trigger+installer ran in BOTH (injrun2.log; injpy.*.installed for both pids).

### 0b interceptable boundary
| option | reachable from injected .so in worker? | evidence |
|---|---|---|
| A pure-C GOT/PLT interpose of the op | NO | `nm -D vllm/_C.abi3.so | grep cutlass_scaled_mm` = empty: the op is a TORCH_LIBRARY dispatcher op called `torch.ops._C.cutlass_scaled_mm(...)` (_custom_ops.py:921) via the dispatcher's fn-ptr table, not a PLT slot. Only C-reachable layer below is generic cuLaunchKernel (already may13-patched) with no semantic a/b/scales/out -> useless for residual recompute |
| B injected .so drives in-worker Python patch | YES (PROVEN) | CPython C-API (PyGILState_Ensure/PyRun_SimpleString/Py_IsInitialized) resolves via dlsym(RTLD_DEFAULT) in the worker; inj_pyhook.c installer thread acquires the GIL and patches vllm._custom_ops.cutlass_scaled_mm in-process. MEASURED: injpy.3559312 = **2304 cutlass_scaled_mm calls captured in the worker** (last_mnk [1,4096,14336]), engine healthy, coherent output |
| D neither (needs plugin/install) | RULED OUT | B works with the .so as the sole artifact |

First run crashed `TypeError: _w() got an unexpected keyword argument 'out_dtype'` at cutlass.py:177
-- that crash PROVES interception (the worker's real qkv_proj forward called the injected wrapper);
it was a probe signature bug (positional vs keyword), fixed by forwarding *args/**kwargs verbatim.

### 0c MECHANISM VERDICT (recon complete, STOP for adjudication)
**B** -- injected .so drives an in-worker Python patch, no plugin, no install, default-MP. A is dead
(dispatcher-routed, no GOT symbol). D ruled out. Worker reach + FP8-linear interception MEASURED
under default multiprocessing (2304 calls in the spawn worker).

SCOPE / honesty: this recon proves WORKER-REACH + interception in EAGER mode. It does NOT yet prove
the full default-config gate -- the cudagraph traceability wall (eager seam crashed dynamo on a
host-side resid>0 branch) is a SEPARATE Phase-1b problem: the detector must use the ra_coophook
out-of-band design (coop_driver.py:97-119 -- checks emitted inline during capture, residual read
from a GPU buffer on the replay path, host branch OUTSIDE the compiled region). Whether that design
installs cleanly via the same injected-.so-driven in-worker patch under torch.compile ON is the
gating Phase-1 risk, untested here. The injection mechanism for REACH is proven; the cudagraph-safe
DETECTOR on top of it is the next build. Anil adjudicates B before Phase 1.

---
## ADDENDUM 2 — INTERCEPTION-LAYER recon (which symbol to hook), 2026-06-12

Question: lowest-vLLM-dependence layer with enough semantic info (a,b,scales,out) for residual
detection, from the injected .so in the default-MP worker. Anchor 2edba0d2 entry==exit; scratch
probes (inj_torchhook.so, fwgen_test.py); anchor .so untouched.

### Option 1 — TORCH DISPATCHER LEVEL (torch.ops._C.cutlass_scaled_mm), MEASURED
- Op identity/schema: `_C::cutlass_scaled_mm(Tensor!out, Tensor a, Tensor b, Tensor a_scales,
  Tensor b_scales, Tensor? bias) -> ()` — in-place; ALL operands present at the torch.ops layer.
  Registered by vLLM's _C extension (import vllm._custom_ops required). `torch.ops._C.cutlass_scaled_mm`
  attribute is SETTABLE (verified) so the injected .so can overwrite it in-worker; vLLM calls it via
  fresh torch.ops attribute lookup each call (_custom_ops.py:921), so the wrap catches every call.
- 1b WORKER TEST (inj_torchhook.so, default-MP, real FP8 ckpt): worker pid 3560535, **2304 calls
  captured at layer torch.ops._C** (th.3560535), operands visible: out[16384,6144]bf16,
  a[16384,4096]fp8_e4m3fn, b[4096,6144]fp8_e4m3fn, a_scales/b_scales f32 scalars (th.*.operands);
  generation clean (coherent output). MEASURED.
- 1c FRAMEWORK-GENERAL TEST (fwgen_test.py, NO vLLM caller): hook fired on a DIRECT
  torch.ops._C.cutlass_scaled_mm call = 2; recompute_clean_residual 0.0 (deterministic);
  direct_out_vs_worker_out_residual 0.0 (captured operands reproduce the real output EXACTLY).
  MEASURED: the hook is CALLER-INDEPENDENT and operands suffice for recompute.
- COUPLING: the MECHANISM is PyTorch-dispatcher-ABI coupled (torch.ops handle / torch.library),
  NOT vLLM-Python-internal coupled -> survives vLLM Python refactors that would break Option B, and
  catches the op regardless of which wrapper invokes it (proven). NOT zero-vLLM: the op
  `_C::cutlass_scaled_mm` is vLLM-registered. True cross-framework FP8 (non-vLLM) uses a DIFFERENT
  op (core-torch `aten::_scaled_mm`, present; torchao `aten::_cslt_sparse_mm`); the SAME technique
  hooks those too -> framework-general by hooking the op SET. torch.library.Library.impl("CUDA")
  kernel-override is an untested [PROJECTED] alternative that would also catch C++-side dispatcher
  calls (handle-wrap catches Python torch.ops callers, which is what vLLM is).

### Option 2 — PURE-C KERNEL-LAUNCH LEVEL (cuLaunchKernel), [tested-negative], DEAD for recompute
- Operands are PHYSICALLY present in the opaque cutlass Params struct passed via kernelParams, but
  NOT semantically recoverable without per-kernel, per-cutlass-version reverse-engineering of that
  struct layout. The substrate's own may13 layer treats kernelParams as an opaque fixed-layout
  buffer: identifies kernels by NAME/op_class (cipher_intercept_cudart.cpp:2761), "never mutates the
  kernelParams pointers" (:3300-3301); the ONLY operand extraction is a debug-flag probe logging
  "FlashAttention's first 6 kernelParams as 8-byte values" (:2764-2766) = raw pointers, hand-RE'd,
  no shapes/dtypes/scale semantics. Definitive: no clean (a,b,scale_a,scale_b,out,m,n,k) recovery ->
  residual recompute not drivable from this layer. Confirms the prior finding.

### RANKED LAYER TABLE
| layer | coupling | operands for recompute? | reaches default-MP worker? | framework-general? | status |
|---|---|---|---|---|---|
| **Opt 1 torch.ops._C dispatcher** | PyTorch-ABI (op is vLLM-registered) | YES (out,a,b,a_scales,b_scales) | YES (2304 in worker) | technique YES (extends to aten::_scaled_mm); op-identity per-backend | MEASURED |
| Opt B vllm._custom_ops wrapper | vLLM Python internal symbol | YES (proven earlier, 2304) | YES (proven earlier) | NO (vLLM-only) | MEASURED (prior) |
| Opt 2 cuLaunchKernel pure-C | none (driver) | NO (opaque param struct) | YES (may13 already hooks) | YES if it worked | [tested-negative] |

### RECOMMENDATION (stop for adjudication)
Hook at **Option 1 (torch dispatcher, torch.ops._C.cutlass_scaled_mm handle-wrap)** via the proven
in-worker injection. Same .so-only delivery, full operands, deterministic recompute, and it
DECOUPLES from vLLM's Python wrapper (caller-independent, survives vLLM refactors) — coupling drops
from "vLLM internal Python symbol" (Opt B) to "vLLM-registered torch op schema," and the identical
technique extends to aten::_scaled_mm for non-vLLM FP8. Option B is the proven fallback. Option 2 is
dead for recompute. NOTE (forward, not this recon): the handle-wrap is still a Python-level wrap, so
the cudagraph traceability question is unchanged; torch.library.impl("CUDA") kernel-override is the
candidate that may also help there and is worth testing in Phase 1. Anchor end: 2edba0d2136f8ede4713d90a8f7cd55f.

---
## ADDENDUM 3 — CUDAGRAPH-SAFE detector attempt (default config), 2026-06-12

Goal: clear BOTH walls in one build — intercept FP8 linears so detection survives torch.compile +
cudagraph, injected-.so-only, default config (MP ON, cudagraph ON), on neuralmagic FP8. Anchor
2edba0d2 entry==exit; scratch probes (inj_cgseam.so, micro_cudagraph_test.py); anchor .so untouched.

### Phase 1A (impl('CUDA') override) — INFEASIBLE, MEASURED
micro_cudagraph_test.py: registering a CUDA-key override and calling the original RECURSES
(recursed=True, depth=3, can-call-original=False) — the override BECOMES the kernel, PyTorch gives
no handle to the displaced kernel. The 1a-ii requirement ("call the ORIGINAL, bit-identical") cannot
be met. Killed.

### Phase 1B (functional handle-wrap + out-of-band residual) — built; 3 walls CLEARED, 1 WALL hit
Built the injected-.so cudagraph seam (inj_cgseam.c): in-worker wrap of vllm._custom_ops.cutlass_scaled_mm
(NOT the torch.ops packet), real compute + device inject flag + independent recompute + residual via a
custom op cipher::detect_sink (mutates_args) into static buffers, + out-of-band reader thread. Walls
hit and cleared IN SEQUENCE (each MEASURED from the worker traceback):
1. torch.ops-packet overwrite -> dynamo fullgraph AOT crash `'function' has no attribute 'default'`.
   FIX: wrap the vLLM python helper (dynamo inlines it), leave the op packet intact. CLEARED.
2. inject dtype promotion (float32 INJ -> out becomes float) -> `mat1/mat2 dtype` crash.
   FIX: INJ.to(out.dtype). CLEARED.
3. out-of-band reader .item() during init cudagraph CAPTURE -> `CUDA driver error: operation failed
   during capture`. FIX: gate reader on a READY file set after init capture. CLEARED.
4. **THE WALL: detection side-channel does not escape the compiled region.** After 1-3, the engine
   runs clean under cudagraph AND the INJECT works (inj_out = garbage "_<?_<?..." -> the wrap EXECUTES
   IN the captured graph), but covered_calls=0, res_max=0, NOT caught: inductor keeps the inject (an
   external-tensor READ that feeds the output) but the residual/counter WRITES to my host RES/CNT do
   not surface. Tried in-place write, a mutates_args custom op (cipher::detect_sink), and
   torch._dynamo.mark_static_address on the buffers — ALL four read cnt=0/res_max=0 (cggate_on.json).
   tok/s under cudagraph ~122 (vs ~129 unarmed -> ~5% but MOOT, detection inert).

### EAGER CONTRAST (same injected .so, cudagraph OFF) — isolates the wall, MEASURED
Identical inj_cgseam.so, S_EAGER=1 (cudagraph OFF), default MP: **covered_calls=98,688,
res_max=0.516, CAUGHT=True** (cggate_eager.json). The detector LOGIC is correct and the injected .so
works fully WITHOUT cudagraph. The cudagraph run differs in exactly one way -- the side-channel
readout (cnt/res) does not surface from the compiled region -- proving the wall is specifically
in-compiled-region value extraction, not the detector.

### WHY 1B's ra_coophook fallback CANNOT port to FP8 (structural)
ra_coophook's cudagraph-safe detection works because it interposes cublasGemmEx at the C ABI
(rv_coop_shim.c:3,119 "when the call's stream IS CAPTURING"): the compiled artifact still calls that
dynamically-linked C symbol during capture, so the C shim emits recompute into the SAME graph and
manages a host-readable buffer (rv_read, coop_driver.py:106). FP8 has NO C-ABI GEMM seam (the
original blindness: cutlass_scaled_mm is a dispatcher op, no GOT symbol). The only FP8 seam is the
Python op/wrapper, which inductor consumes at compile time, and a value can only leave the compiled
forward as a GRAPH OUTPUT — which requires editing vLLM's forward or a cooperative hook, NOT
injection-only.

### VERDICT
- Mechanism shipped: 1B functional wrap (1A killed). Under default config it INSTALLS in the spawn
  worker, EXECUTES in the captured cudagraph (inject proven), and does NOT crash compile — but the
  DETECTION READOUT is WALLED: no injection-only side-channel out of vLLM's compiled+cudagraph'd
  forward, and FP8 (unlike cuBLAS) has no C-ABI seam to host-manage a buffer during capture.
- Coverage off~0 -> [under cudagraph: interception in-graph YES, readout NO]; 0-FP: clean residual 0
  but inert; fault catch UNDER CUDAGRAPH: **NOT achieved injection-only**.
- ONE LINE: zero-plugin FP8 detection is DEPLOYABLE in EAGER mode (prior FP8_SEAM_RESULT: 100% cov,
  0 FP, caught, 6.3%@N=45) but BLOCKED under default cudagraph. Single blocker: extracting the
  detection result from vLLM's compiled forward needs a graph-output / cooperative hook (not
  injection-only) OR a C-ABI FP8 GEMM seam (absent). Anil adjudicates: accept eager-only injection,
  or allow a cooperative hook for the cudagraph path.
Anchor end: 2edba0d2136f8ede4713d90a8f7cd55f.

---
## ADDENDUM 4 — FP8 detection UNDER CUDAGRAPH SOLVED via MODULE-BUFFER input-mutation, 2026-06-12

Reverses Addendum 3's wall. The prior failure used CLOSED-OVER GLOBAL residual buffers (inductor
DCE'd the writes). Fix: route the residual through a buffer that is part of the compiled region's
INPUT contract — a MODULE BUFFER (dynamo lifts module buffers/params as graph inputs; vLLM's KV
cache is the same in-place-mutation pattern). Anchor 2edba0d2 entry==exit; scratch probes only
(inj_cgseam2.so, cgmod_inproc.py, phase0*); anchor .so untouched.

### Phase 0 micro (no vLLM) — mechanism settled
- **P0 declared-INPUT buffer, strict cudagraph (reduce-overhead+fullgraph):** clean cnt=3/res=0,
  inject -> res=0.508, CAUGHT=True. `torch._inductor.config.triton.cudagraph_support_input_mutation`
  (config.py:1534, default True) honors it.
- **P0B closed-over GLOBAL buffer:** CUDA "operation not permitted when stream is capturing" — globals
  FAIL (matches Addendum 3's DCE under vLLM). The buffer MUST be in the I/O contract.
- **P0C MODULE BUFFER + manual torch.cuda.graph (EXACTLY vLLM's capture style):** clean cnt=3/res=0,
  inject -> res=0.508, CAUGHT=True. Module buffers survive vLLM-style capture.
- **P0-CSE:** perturbing the recompute's OPERAND gives residual 0.172 -> the two cutlass calls are
  independent kernels (not CSE-folded).

### Why module buffers (option ii), not the config flag (option iii)
vLLM does its OWN manual `torch.cuda.graph` capture (compilation/cuda_graph.py:308) and "does not
store persistent buffers" (:161) — it is NOT torch's cudagraph-trees, so the inductor input-mutation
flag is not the governing mechanism. A value persists out of vLLM's captured forward only if it is a
mutated INPUT of the compiled region. Module buffers ARE lifted as inputs by dynamo -> the channel.

### In-process proof (cgmod_inproc.py), cudagraph ON
Patched CompressedTensorsW8A8Fp8.create_weights to attach cipher_res/cnt/inj buffers per FP8 layer +
apply_weights to recompute->residual into those buffers. **128 layers, cnt=16,384, clean_res=0.0,
inject one layer -> res=0.5, CAUGHT under cudagraph = True.**

### THE GATE — injected-.so-only, DEFAULT config (MP ON + cudagraph ON), real FP8 model
inj_cgseam2.so patches the FP8 scheme IN the spawn worker (gated on sys.modules to avoid an import
race; 80 ms poll wins the create_weights race -> all layers instrumented). cg2gate_on.json:

| gate item | result | artifact |
|---|---|---|
| **2a COVERAGE** | **128 FP8 layers, cnt=32,768** under cudagraph = ~0% baseline -> 100% of the FP8 decoder linear path | cg2gate_on.json worker |
| **2b 0-FP** | **clean_res=0.0** over 32,768 comparisons | cg2gate_on.json |
| **2c FAULT CATCH UNDER CUDAGRAPH** | inject one layer -> res_max=0.5, **CAUGHT=True** (first_detect_wall set). THE cell that read 0 in Addendum 3 | cg2gate_on.json |
| **2d CSE-INDEPENDENCE** | operand-perturb probe (no fault): **res_max=6.0** while clean=0.0 -> recompute is a genuine 2nd kernel, not folded (+ P0-CSE op-level) | cg2gate_cse.json (cg2c.*) |
| **2e OVERHEAD** | OFF 250.82 -> ON 120.68 tok/s = **-51.9% (ALWAYS-ON:** recompute every covered linear every step). Periodic (every N) = the optimization, not built | cg2gate_off/on.json |
| **2e TRANSPARENCY** | injected .so SOLE artifact (CUDA_INJECTION64_PATH/LD_PRELOAD), default vLLM config, no plugin/install/source-edit. Customer changes NOTHING | inj_cgseam2.so |

### VERDICT
**Zero-plugin FP8 SDC detection is DEPLOYABLE under DEFAULT config (multiprocessing + cudagraph ON),
injected-.so-only.** The Addendum-3 wall (side-channel can't escape the compiled region) is CLEARED
by routing the residual through MODULE BUFFERS that dynamo lifts into the forward's input-mutation
contract — the same channel vLLM uses for its KV cache. Coverage 0->100%, 0 FP / 32,768, fault caught
UNDER CUDAGRAPH, recompute proven CSE-independent. ONE remaining item is COST, not capability:
always-on doubles the covered matmuls (-51.9%); periodic dual-graph checking (the ra_coophook A3
pattern, every N steps) is the overhead optimization to reach single digits — buildable, not blocking.
ONE LINE: yes, deployable under default cudagraph injection-only; the only open work is periodic
checking to cut the always-on 52% to ~single digits. Anchor end: 2edba0d2136f8ede4713d90a8f7cd55f.

---
## ADDENDUM 5 — PERIODIC checking to cut the 51.9% always-on cost: mechanism PROVEN, vLLM
## detection-integration WALLED (prove-or-characterize), 2026-06-12

Goal: periodic recompute (every N steps, not every step) to bring the Addendum-4 always-on -51.9%
to single digits, keeping catch+0-FP+CSE, injected-.so-only under default cudagraph. Anchor 2edba0d2
entry==exit; scratch probes; anchor .so untouched.

### Phase 0 micro (no vLLM) — periodic MECHANISM settled
- **DUAL CAPTURED GRAPHS saves cost (MEASURED, phase0_periodic.py):** unchecked step 0.0236 ms vs
  checked 0.0529 ms (2.24x). Host-alternated N=8 schedule = 0.0273 ms/step = **1.155x unchecked
  (15.5% overhead)**; persistent fault injected at step 20 CAUGHT at step 23 (within N). Projected
  single-digit at N>=16 (overhead ~ 1 + 1.24/N: 7.7%@16, 3.9%@32).
- **DEVICE-GATE does NOT save (MEASURED):** recompute * device 0/1 flag = 0.0518 ms = 2.19x
  unchecked — the kernels still launch under cudagraph; the gate only masks the result. Killed.
- **Host-flag yields a real checked variant in raw torch.compile (MEASURED):** flag False -> RES 0.0,
  flag True -> RES 0.5 (dynamo recompiles, sink fires). The flag mechanism itself is sound.

### Phase 1 vLLM — COST structure works, DETECTION does not surface
inj_cgseam2-style module-buffer detector + a CUDAGraphWrapper dual-graph hook (ra_coophook A3 port):
capture a checked + an unchecked graph, alternate host-side every N.
- **Cost saving CONFIRMED:** N=32 -> 239.85 tok/s = **-4.4%**; N=8 -> 221.75 = -11.6% (vs OFF 250.82,
  always-on 120.68/-51.9%). The dual-graph alternation DOES cut cost to single digits.
- **Detection BROKEN:** sink_cnt_total=0, caught=False at every N (incl. N=1). The checked cudagraph
  does not contain the recompute/sink, so nothing is detected.

### ROOT CAUSE (diagnosed, MEASURED)
vLLM CACHES a single compiled forward (@support_torch_compile). The host-flag dual-variant trick that
works in raw torch.compile (DYN test) is DEFEATED by vLLM's caching: at cudagraph-capture time, fl
ipping _CHK does NOT re-dispatch dynamo (the compiled callable is cached), so both captured graphs are
the SAME variant — whichever was compiled first (unchecked at profile_run) -> both-unchecked -> cnt=0.
I cannot produce checked AND unchecked compiled forwards inside vLLM from injection. ra_coophook
solved the cuBLAS case by adding recompute at CAPTURE time via a C cublasGemmEx shim (below dynamo);
FP8 has NO C-ABI GEMM seam, so there is no capture-time injection point for the recompute.

### VERDICT (prove-or-characterize)
- Periodic MECHANISM: PROVEN (dual-graph saves cost to single digits; device-gate doesn't).
- vLLM periodic DETECTION: WALLED injection-only — vLLM's single-compiled-forward caching prevents
  making both checked+unchecked variants, and FP8 has no capture-time recompute seam (unlike cuBLAS).
- DEPLOYABLE TODAY: always-on (Addendum 4) = full catch + 0 FP + CSE-clean at -51.9%. Periodic cuts
  cost to single digits but loses detection under vLLM's caching.
- PATHS TO single-digit WITH detection (not built, ranked): (a) a cheap ALWAYS-ON check that needs no
  dual-graph — e.g. a Freivalds/random-projection checksum (out@v ?= a@(b@v), two tiny mat-vecs,
  single graph, full probabilistic coverage) — most promising, sidesteps this entire wall but changes
  detection from exact-recompute to probabilistic (note: a prior Freivalds attempt at the cuBLAS
  prefill seam was DO-NOT-PROCEED for a DIFFERENT reason; the decode module-buffer context is new);
  (b) force vLLM to compile two variants (deep vLLM internals / two compile keys); (c) a vLLM-
  cooperative hook that exposes the checked forward (not injection-only).
- ONE LINE: dual-graph cuts cost to single digits (proven), but injection-only periodic DETECTION is
  walled by vLLM's single-compiled-forward caching + FP8's lack of a capture-time recompute seam; best
  deployable today is always-on at -51.9% (full catch). Anchor end: 2edba0d2136f8ede4713d90a8f7cd55f.

---
## ADDENDUM 6 — FREIVALDS checksum for cheap detection: FAILS BOTH cost AND detection for FP8
## decode (prove-or-characterize, STOP-for-adjudication), 2026-06-12

Hypothesis: replace exact-recompute (2x GEMM, Addendum 4) with a Freivalds randomized checksum
(verify out=a@b via out@v ?= a@(b@v), two mat-vecs) for single-digit overhead, single graph, no
dual-graph. RESULT: fails on BOTH axes for decode. Anchor 2edba0d2 entry==exit; scratch micros
(phase0_freivalds.py); anchor .so untouched. Phase 0 is decisive -> stopped before the full gate.

### 0b COST — Freivalds does NOT save for decode (M=1)
Decode shape M=1, K=4096, N=14336 (down/up-proj scale), R=4 projections:
- **fresh-v Freivalds = 6.16x the GEMM** (MEASURED, eager): b@v for a fresh v is a FULL-WEIGHT pass
  (K*N) = the decode GEMM's own cost, + fp8->float conversions. Fresh v (needed for temporal
  miss-decay) costs MORE than exact recompute's 2x. No saving.
- **fixed-V (b@V precomputed) under CUDAGRAPH = +44.7%** (MEASURED, cgcost: bare 0.0236 ms ->
  freivalds 0.0341 ms). NOT single-digit. Reason: the decode GEMM is tiny/fast (memory-bound on the
  56 MB weight, ~23 us); the check is ~6 small kernels (o.float, o@V, a.float, a@bV, sub, abs) whose
  FIXED per-kernel execution overhead (~10 us) is ~45% of the GEMM. Tiny-op overhead dominates, not
  the (negligible) FLOPs. The "mat-vec << mat-mat" premise is FALSE for M=1 decode.

### 0c CLEAN TOLERANCE (principled) — fp8 noise floor
Over 200 fresh clean operands: relative residual mean 0.0023, **max 0.0068**. Principled threshold =
3x clean max = **2.05% relative** (separates fp8 rounding from faults). abs residual mean 0.21, max
0.48 (scale-dependent; relative is the right metric).

### 0d/0e DETECTION — probabilistic checksum MISSES single-element SDC (the floor)
Catch rate per check (fixed V, R=4, rel tol 2.05%), MEASURED over 300 trials:
- gross faults caught: single-elem +max-magnitude 0.927, row-scale x1.02 0.37.
- **localized faults MISSED: single-elem +0.5 = 0.027, +0.05 = 0.003, row-scale x1.001 = 0.0,
  tiny 1e-3-all-elem = 0.003.** Single-element additive floor: ~0 below magnitude 0.5, 24.5% at 1.0,
  86.5% at 2.0.
- **REALISTIC bit-flip on one output element (the actual SDC model, the exact detector catches
  100%):** bit15/sign 0.28, **bit14/exp (the R.A fault) 0.75**, bit13 0.06, bit10 0.09, bit7 0.02,
  **mantissa bits 5/2/0 = 0.00**. Even the gross bit-14 exponent flip is caught only 75%/check; all
  mantissa flips and most exponent flips are MISSED.
- CSE guard PASSES (resid 0.12 clean -> 6.65 operand-perturbed, responds=True).
- ROOT CAUSE: the checksum compresses N=14336 outputs into R values, diluting a single-element fault
  by ~sqrt(N) below the fp8-noise tolerance. Fundamental to randomized checksums; more projections
  (R) only helps as sqrt(ln R), cannot overcome sqrt(N) dilution. Fresh-v would give temporal decay
  but costs 6x (above); fixed-V is cheap-ish but NON-compounding (a null-space miss is permanent).

### VERDICT (prove-or-characterize, STOP)
**Freivalds is NOT deployable for FP8 decode SDC detection — it fails BOTH the cost target and the
detection floor.** (1) Cost: fixed-V +44.7% under cudagraph (tiny-kernel-overhead-bound; not single
digit), fresh-v 6.16x (worse than exact's 2x). (2) Detection: misses most single-element SDC
(mantissa 0%, most exponent <10%, gross bit-14 only 75%/check) — strictly weaker than exact-recompute
(100%). For M=1 decode the premise "mat-vec << mat-mat" is false and the checksum's sqrt(N) dilution
makes it blind to the localized-bit-flip SDC class the detector exists to catch.
- BEST DEPLOYABLE remains always-on EXACT recompute (Addendum 4): -51.9%, 100% catch, 0 FP, CSE-clean,
  injected-.so-only, default cudagraph. Reducing its cost stays OPEN: periodic dual-graph WALLED
  (Addendum 5, vLLM single-forward caching), Freivalds INADEQUATE (this addendum).
- Candidate not yet tried: ABFT row+column checksums catch single-element errors at O(M+N) extra
  (unlike Freivalds), but the project's STEP-A memory flagged ABFT overhead at the cuBLAS seam; the
  decode module-buffer context would need its own evaluation. Anil's call.
- ONE LINE: no — zero-plugin Freivalds is not deployable at single-digit-with-adequate-detection for
  FP8 decode; it fails cost (+44.7% cudagraph / 6x fresh) AND detection (misses most single-element
  SDC); exact-recompute always-on (-51.9%, full catch) remains best. Anchor end: 2edba0d2136f8ede4713d90a8f7cd55f.

---
## ADDENDUM 7 — ABFT row+col checksum (Phase-0 micro ONLY): detection BEATS Freivalds, but cost for
## DECODE is strictly dominated by exact-recompute (M=1 degeneration). prove-or-kill, STOP. 2026-06-12

Last unexplored cost-reduction option. ABFT keeps a checksum per ROW and per COLUMN, so a single
error shows as a row AND a column mismatch at O(M+N) — NO sqrt(N) dilution (Freivalds' killer).
Question: does that better detection come cheap under cudagraph on the fast decode GEMM? Anchor
2edba0d2 entry==exit; scratch micro (phase0_abft.py); anchor .so untouched. Phase-0 only, no integration.

### 0a DETECTION — single-element bit-flip catch, head-to-head (DECODE M=1, K=4096, N=14336)
Same realistic faults as Freivalds (Addendum 6), 300 trials, principled 3x-clean-max relative tol:
| bit (flipped) | exact (Add.4) | Freivalds (Add.6) | ABFT col | ABFT row |
|---|---|---|---|---|
| sign | 100% | 28% | **93.7%** | 0% |
| bit14 exp (R.A) | 100% | 75% | **88.7%** | 0% |
| bit13 exp | 100% | 6% | **91.0%** | 0% |
| bit10 exp | 100% | 9% | **90.0%** | 0% |
| bit7 exp | 100% | 2% | **80.3%** | 0% |
| bit5 mant | 100% | 0% | **43.7%** | 0% |
| bit2 mant | 100% | 0% | 0% | 0% |
| bit0 mant | 100% | 0% | 0% | 0% |
**ABFT DETECTION BEATS Freivalds decisively:** the COLUMN checksum catches single-element exp/sign at
80-94% and mid-mantissa (bit5) 44%, vs Freivalds 0-75%. Reason: for M=1 the column checksum is
per-element (one row per column -> NO dilution). Only the lowest mantissa bits (2,0) are missed — those
flips change the value below the fp8 rounding noise (genuinely sub-detectable; exact-recompute catches
them only because it is bit-exact). The ROW checksum is USELESS for M=1 (one sum over N; relative tol
27.5 = 2750%, catches nothing) — it dilutes exactly like Freivalds.

### 0b CLEAN / 0-FP — principled tolerance
DECODE col_tol = 2.8% relative (3x clean-max); row_tol 2750% (useless, M=1). PREFILL row/col_tol ~0.9%.
No false mismatch in the clean trials at these tolerances. CSE guard PASSES (resid 0.009 clean ->
0.108 operand-perturbed, responds=True).

### 0c COST under cudagraph — the decisive axis (MEASURED)
| shape | bare ms | row-only (cheap path) | full ABFT (row+col, the detecting path) |
|---|---|---|---|
| DECODE M=1 | 0.0236 | 0.0490 = **+107%** (catches NOTHING) | 0.243 = **+928%** |
| PREFILL M=512 | 0.0545 | 0.0864 = +58.7% | 0.303 = +455% |
**DECODE cost is the kill.** The DETECTING path (column checksum) is +928% — because for M=1 the
column reference `(1_M @ a) @ b = a@b` IS the full recompute, done here in fp32 (+ a 56 MB fp8->float
weight conversion). The CHEAP path (row checksum, +107%) hits the same tiny-kernel-overhead floor that
killed Freivalds AND catches nothing for M=1. [PROJECTED] an efficient fp8-AUGMENTED ABFT (append 1
checksum row -> GEMM grows M=1->2) would be ~+100% for decode = doubling the rows = SAME as exact
recompute, still NOT single-digit. Either way decode ABFT is >= exact-recompute cost.

### VERDICT (Phase-0, STOP)
Head-to-head (DECODE, the cost-critical case):
| approach | single-elem catch | clean 0-FP | decode cost (cudagraph) | verdict |
|---|---|---|---|---|
| exact-recompute (Add.4) | 100% all bits | yes | full-model -51.9% (~+100%/linear) | DEPLOYABLE, costly |
| Freivalds (Add.6) | mantissa 0%, bit14 75% | yes | +44.7%/linear, blind | DEAD (both axes) |
| **ABFT (this)** | exp/sign 80-94%, mant5 44%, mant2/0 0% | yes | detecting path +928% (=recompute); cheap path +107% catches nothing | **DEAD for decode** |

ABFT is "detection-yes, cost-no": it catches single-element SDC FAR better than Freivalds (the column
checksum has no dilution at M=1), but for DECODE it is STRICTLY DOMINATED by exact-recompute — the
detecting column checksum IS a full recompute when M=1 (no row dimension to compress), so it costs
~+100% (best case) to +928% (measured fp32) while catching LESS than exact (misses low mantissa). The
cheap row-checksum path is +107% (kernel-overhead floor) and blind. Same fundamental M=1 root cause as
Freivalds: decode has a single row, so any single-element-sensitive checksum carries the full output =
recompute. [Note vs prior STEP-A ABFT-at-cuBLAS-seam overhead concern: here the decode overhead is
MORE fundamental — M=1 degeneration, not just kernel/seam overhead.] For PREFILL (M large) augmented
ABFT would be cheap and catch single-elem moderately — but prefill was never the cost problem.
ONE LINE: NO — ABFT does not achieve single-digit cost for FP8 DECODE (detecting path = full recompute
at M=1, +928% measured / ~+100% projected-efficient; cheap path +107% & blind); it beats Freivalds on
detection but is strictly dominated by exact-recompute on the decode cost axis. Exact-recompute
always-on (-51.9%, 100% catch) remains the best deployable; cutting its cost on decode is now
THREE-WAYS-walled (periodic dual-graph, Freivalds, ABFT). Anchor end: 2edba0d2136f8ede4713d90a8f7cd55f.

---
## ADDENDUM 8 — PERIODIC checking via OUT-OF-BAND recompute: VIABLE, projected SINGLE-DIGIT (the
## first cost-reduction that ISN'T walled). Phase-0 micro, 2026-06-12

Frequency axis (not per-check cost, which is recompute-floored for M=1). Goal: recompute every Nth
step, not every step. Wall to route around (Add.5): periodic-via-dual-graph needed two compiled
forwards; vLLM caches one. So reduce FREQUENCY without a second compiled variant. Anchor 2edba0d2
entry==exit; scratch micros; anchor .so untouched. Phase-0 found a SURVIVING mechanism.

### Mechanism survey (which frequency-reducer survives vLLM's single forward)
- **(A) device-flag skip inside the op: DEAD** (Add.5 confirmed — a captured graph replays ALL its
  kernels; a device flag masks the result but cannot remove kernels).
- **(B)/(C) move the recompute OUT of the captured forward; expose per-layer I/O via cheap in-graph
  COPIES; recompute host-side every Nth step on the buffered I/O. SURVIVES.** Unchecked steps cost
  only the copies; the check is a plain host-side branch (frequency = host decision, legal).

### The decisive measurements
- **Exposure floor (copies) is ~FREE at the model level (MEASURED, the key result):** a definite
  non-elidable copy of each FP8 layer's output (`empty_like().copy_()`, 128 layers) = **+0.1%** tok/s
  (off 251.59 -> 251.27). 1-copy clone = -0.1%. Plain bf16 activation copies overlap the
  weight-memory-bound decode and cost nothing. (The earlier float-convert proxy showed -37.8% -> that
  cost was the FLOAT CONVERSION, NOT the copy; always-on checks must avoid per-layer float-converts.)
- **(C) standalone check graph DETECTS EXACTLY + is cheap to replay (MEASURED, micro):** a separate
  cudagraph that recomputes scaled_mm from the layer's buffered input and compares to the buffered
  output, replayed every Nth step, reads the CURRENT static-address values: clean residual 0.0;
  catches single-element bit-flips incl. mantissa bit-0 (resid 0.0078) -> full exact-recompute
  fidelity preserved. Per-linear amortized extra = check_graph_ms/N: 7.5%@N=16, 3.8%@N=32, 2.7%@N=45.
- CSE/fold guard: the check responds to operand perturbation (exact recompute; same as Add.4).

### PROJECTED N-sweep (model level) = free copies + amortized recompute
Always-on exact recompute = -51.9% (every step). Periodic = recompute every Nth step -> ~51.9%/N
amortized + ~0% copies:
| N | projected model overhead |
|---|---|
| 8  | ~6.5% (single-digit) |
| 16 | ~3.2% |
| 32 | ~1.6% |
| 45 | ~1.2% (≈ the cuBLAS detector's ~3% regime) |
[PROJECTED from measured components: copies +0.1% MEASURED; amortized recompute = measured always-on
-51.9% / N. Full end-to-end integration not yet run — that's the gate to confirm.]

### VERDICT (Phase-0, POSITIVE — recommend integration)
**Frequency-reduction is VIABLE for FP8 decode — the FIRST cost-reduction approach that is NOT walled.**
A surviving mechanism (cheap free copies to expose per-layer I/O + out-of-band periodic recompute via a
separate check graph) sidesteps every prior wall: it uses ONE compiled forward (not two -> beats Add.5
dual-graph), keeps EXACT detection (not probabilistic -> beats Add.6 Freivalds / Add.7 ABFT), and moves
the check OUT of the graph (not in-graph device-gate -> beats Add.5 device-gate). Copies are free
(+0.1% MEASURED), exact detection preserved (MEASURED), amortized recompute single-digit at N>=8
(PROJECTED ~6.5%@8 / ~3%@16). This reaches the cuBLAS detector's overhead regime for FP8.
- HONEST GAPS: (i) full end-to-end integration (copies in apply_weights + out-of-band recompute at the
  step boundary, all 128 layers, injected-.so default-MP) NOT yet built/measured -> the gate number is
  PROJECTED, not measured; (ii) the out-of-band recompute needs per-layer buffered x AND out (2 copies;
  measured 1-copy free, 2-copy expected ~free); (iii) honest transient bound: faults shorter than N
  steps between checks CAN be missed (same disclosure as the cuBLAS detector); persistent faults caught
  within N.
- ONE LINE: YES (projected) — zero-plugin FP8 decode detection is reachable at single-digit overhead
  (~6.5%@N=8, ~3%@N=16) via free-copy I/O exposure + out-of-band periodic exact recompute; this is the
  first non-walled path. Recommend proceeding to integration + the default-config gate to convert the
  projection to a measured number. Anchor end: 2edba0d2136f8ede4713d90a8f7cd55f.

---
## ADDENDUM 9 — PERIODIC out-of-band recompute, INTEGRATED + GATED: single-digit overhead REACHED
## but DETECTION FAILS in integration (exposure-not-captured). prove-or-characterize. 2026-06-12

Converted the Addendum-8 projection to a MEASURED gate. Built into the in-process module-buffer path
(delivery proven Add.4): apply_weights wrap does PLAIN bf16 copies of x,out into per-layer static
buffers (batch-guarded narrow, profile-batch skipped); a host-side EAGER exact recompute runs at the
CUDAGraphWrapper boundary every Nth step on the buffered I/O. Anchor 2edba0d2 entry==exit; scratch
(cgperiodic_oob.py); anchor .so untouched.

### 2a OVERHEAD — MEASURED end-to-end (orchestration live), N-sweep vs OFF 251.5
| N | tok/s | overhead | 0-FP (clean_max_resid) |
|---|---|---|---|
| 8  | 137-140 | -44% | 0.0 |
| 16 | 181.6 | -27.8% | 0.0 |
| 32 | 207.7 | -17.4% | 0.0 |
| 64 | 229.2 | **-8.9% (single-digit)** | 0.0 |
Single-digit IS reached, at N=64 (MEASURED). The Phase-8 projection (6.5%@N=8) was OPTIMISTIC: the
host-side EAGER recompute of 128 layers + orchestration costs ~5 steps PER CHECK, so amortization
needs N=64 not N=8. A captured standalone check graph (option C) would lower the per-check cost
(in-graph recompute ~1 step) and reach single-digit nearer N=16 -- NOT built/measured here.

### 2b/2d CATCH — UNVALIDATED (test-method limitation, NOT a proven failure)
Persistent fault injected into the target layer's output. It DID corrupt live generation (inj_out
diverged: "the Roman army" vs clean "founding of Rome in 753 BC"), but the EXPOSED BUFFER obuf read
CLEAN (target_direct_resid=0, caught_within_N=False) for BOTH a functional-add inject AND an in-place
mutates_args custom-op inject. HONEST READING: both of my in-process injects are INDUCTOR-VISIBLE
(a torch op / functional add), and inductor can compute the side-channel `obuf.copy_(out)` from an SSA
value preceding the inject -> the copy is reordered around my fault. A REAL hardware SDC is a PHYSICAL
memory corruption inductor cannot see; the copy reads out's physical memory at copy time and would
capture it. So I CANNOT conclude the detector fails on real SDC -- my in-process injection method
cannot simulate a physical SDC (it gets reordered). Validating the out-of-band catch needs a
PHYSICAL-memory fault injected outside inductor's view (a CUDA-level shim corrupting the output buffer,
i.e. the .so path / the cuBLAS-detector R.A method) -- NOT DONE here. Catch correctness of the
out-of-band path is therefore OPEN, not failed.

### 2c 0-FP — holds (clean_max_resid 0.0 over all checks, every N).

### VERDICT (prove-or-characterize)
**The out-of-band periodic approach reaches single-digit overhead (N=64, -8.9% MEASURED, 0 FP), but
its DETECTION is UNVALIDATED in this integration** (in-process inductor-visible fault injection is
reordered around the side-channel copy -> can't test it; physical-SDC validation not done).
- Overhead axis: single-digit at N=64 MEASURED (Phase-8's 6.5%@N=8 was optimistic -- host-side eager
  recompute is ~5 steps/check, so amortization needs N=64; a captured standalone check graph would
  lower per-check cost toward the projection, unbuilt). 0-FP holds at all N.
- Detection axis: OPEN. To validate, inject a PHYSICAL SDC via a CUDA-level shim (the .so path) and
  confirm the in-graph copy captures it; if it does, this is a deployable single-digit detector.
- BEST PROVEN-DEPLOYABLE remains always-on IN-GRAPH exact recompute (Add.4): -51.9%, 100% reliable
  catch (validated), 0 FP, CSE-clean, injection-only, default cudagraph -- there the faulted out and
  the recompute are compared INLINE (no side-channel copy to reorder around).
- NEXT (concrete, not walled): port the out-of-band detector to the .so + inject a PHYSICAL SDC to
  validate catch at N=64; and/or build the standalone check graph to push single-digit to lower N.
- ONE LINE: PARTIAL -- out-of-band periodic FP8 detection MEASURES single-digit overhead (N=64, -8.9%,
  0 FP) but its catch is UNVALIDATED (in-process inductor-visible injection can't test the side-channel
  exposure; needs a physical-SDC/.so test); always-on in-graph recompute (-51.9%, validated catch)
  remains the best PROVEN-deployable. Anchor end: 2edba0d2136f8ede4713d90a8f7cd55f.

---
## ADDENDUM 10 — PHYSICAL-SDC catch validation: INCONCLUSIVE on this FP8 stack (honest). 2026-06-12

Open question (Add.9): does the in-graph plain-copy of the FP8 output capture a PHYSICAL SDC (so the
out-of-band recompute catches it)? The Add-9 inject was inductor-VISIBLE (reorderable) -> couldn't
test it. Tried a genuinely physical fault. Anchor 2edba0d2 entry==exit; scratch (phase0_physical.py);
anchor .so untouched.

### Phase 0 — physical fault injection (weight corruption in device memory, host-side between generates)
- Mechanism: corrupt the target QKVParallelLinear's weight ([4096,6144] fp8) IN-PLACE host-side
  AFTER compile/capture, BETWEEN generates. Invisible to inductor (graph already compiled; replay
  reads the corrupted memory; the GEMM produces a genuinely wrong output with NO clean SSA version).
- A weak corruption (x1.5 on 64 of 4096 rows) was a NO-OP (generation unchanged). An aggressive one
  (zero the whole weight) DID diverge generation (clean "founding of Rome in 753 BC" -> corrupt
  "founding of the city of Rome"), CONFIRMING layer.weight feeds the GEMM and a real physical fault
  is injectable. (Divergence is modest -- zeroing one QKV layer degrades but doesn't garbage output.)

### Phase 1 — the discriminator FAILED TO RESOLVE (the honest finding)
Intended discriminator on the exposed obuf: resid_corrupt=|obuf - recompute(xbuf, LIVE weight)| (~0 if
faithful) vs resid_clean=|obuf - recompute(xbuf, CLEAN weight)| (>tol if obuf=corrupted). Result: BOTH
0. The CLEAN-WEIGHT SWAP does not affect the eager recompute -- after `layer.weight.data.copy_(cleanw)`
the recompute output is unchanged (resid_clean=0 even though generation had diverged). vLLM's
compressed-tensors FP8 eager apply path uses a CACHED/PROCESSED weight that does not track in-place
modifications of layer.weight.data, so a clean-reference recompute cannot be built this way. With the
swap inert, resid_corrupt=0 is AMBIGUOUS: it equals 0 whether obuf captured the corrupted output (and
the recompute used the corrupted weight) OR obuf is a clean snapshot (and the recompute used a cached
clean weight). The discriminator cannot distinguish the two cases on this stack.

### VERDICT (prove-or-kill -> INCONCLUSIVE; not validated, not killed)
**I could NOT cleanly determine whether the out-of-band copy captures a physical SDC on this FP8
stack within budget.** A real physical fault IS injectable (weight-zeroing diverges generation), but
vLLM's processed/cached FP8 weight prevents a reliable clean-reference recompute, so the binary
discriminator (corrupted-vs-clean) is inert. Honest status: OPEN. Resolving it needs either (a) a
second independent CLEAN model instance to recompute the reference (not a weight swap on the live
model), or (b) corrupting the OUTPUT tensor's memory directly via a CUDA shim timed between the GEMM
and the copy (the .so path) -- both are real next steps, not done here.
- What IS established this whole investigation: PROVEN-DEPLOYABLE = always-on IN-GRAPH exact recompute
  (Add.4): -51.9%, 100% VALIDATED catch (inline compare, no side-channel to reorder/cache around), 0
  FP, CSE-clean, injection-only, default cudagraph. The cheap out-of-band path has MEASURED
  single-digit overhead (N=64, -8.9%, 0 FP, Add.9) but its catch is UNVALIDATED (Add.9 reorder issue;
  Add.10 discriminator inert) -- it is the honest OPEN item, not a shipped capability.
- ONE LINE: INCONCLUSIVE -- physical-SDC capture by the out-of-band copy could not be cleanly tested
  on this FP8 stack (weight-corruption discriminator inert due to vLLM's cached processed weight);
  the cheap path's catch stays OPEN; always-on in-graph recompute (-51.9%, validated) remains the only
  PROVEN-deployable FP8 detector. Anchor end: 2edba0d2136f8ede4713d90a8f7cd55f.

---
## ADDENDUM 11 — FUSED-ABFT FP8 GEMM: GATE 1 (base perf) FAILS for the ownable kernel -> KILL.
## prove-or-kill, STOPPED at GATE 1 per honesty gate. 2026-06-12

Idea (field: TurboFFT/ATTNChecker/FT-Kmeans): fuse the row+col ABFT checksum into the GEMM EPILOGUE
-> cheap faithful SDC detection, no side-channel, substituted transparently for cutlass_scaled_mm via
the dispatcher .so. GATE 1 = our fusable base GEMM (checksum OFF) must MATCH cutlass_scaled_mm, else
fusion is a net loss. Anchor 2edba0d2 entry==exit; scratch (gate1_triton.py); anchor .so untouched.

### GATE 1 — base FP8 GEMM perf vs cutlass_scaled_mm (MEASURED, decode + prefill)
- **The bar IS matchable by a tuned VENDOR kernel:** torch._scaled_mm (cuBLASLt FP8) vs
  cutlass_scaled_mm: decode qkv -3.4%, o -0.8%, gate_up +1.7%, down +6.5%, bigN -6.6%; prefill
  +3.9-9.4% (numeric rel <=1.1%). So matching CUTLASS is POSSIBLE -- but cuBLASLt is a vendor
  BLACK BOX, NOT fusable (cannot add an ABFT epilogue).
- **The OWNABLE/FUSABLE kernel (Triton FP8 GEMM, autotuned) is 2.2-4.2x SLOWER (GATE 1 FAIL):**
  decode bigN +224%, decode down +233%, decode qkv +316%, prefill gate_up +320% (numeric rel
  <=0.85%, correct). For decode M=1 the tiled GEMM wastes its M-block (1 of 16-32 rows used); for
  prefill it still trails CUTLASS's sm90 WGMMA/TMA warp-specialized pipelines 4x. A basic Triton
  kernel does not approach hand-tuned CUTLASS FP8.

### VERDICT (KILL at GATE 1)
**Fused-ABFT FP8 is NOT viable via the ownable/fusable kernel: GATE 1 fails -- the Triton base GEMM
is 3-4x slower than CUTLASS, so fusion is a net loss vs just using CUTLASS + out-of-band (or
always-on in-graph recompute).** STOPPED before fusion per the honesty gate -- did NOT build the
checksum on a slow base and call the combined number a win.
- The ONLY kernels that match CUTLASS base perf are CUTLASS itself and cuBLASLt -- both NON-fusable
  (we cannot add an ABFT epilogue to a vendor library kernel). 
- STRUCTURAL reason the field's fusion works but ours doesn't: TurboFFT/ATTNChecker fuse into THEIR
  serving kernel (they own it). CIPHER would have to REPLACE vLLM's already-tuned CUTLASS kernel with
  our own kernel; the only fusable kernel we can author (Triton) is 3-4x slower, and matching vLLM's
  per-shape-tuned CUTLASS would require building+tuning a custom CUTLASS FP8 GEMM with an epilogue
  visitor (a major C++/nvcc effort) AND matching its tuning -- uncertain, not pursued. The
  cheap-faithful-fused-detector advantage assumes you OWN a CUTLASS-class kernel; we do not, and
  authoring one to vLLM-tuned parity is the unproven hard part.
- BEST PROVEN-DEPLOYABLE remains always-on IN-GRAPH exact recompute (Add.4): -51.9%, 100% validated
  catch, 0 FP, injection-only, default cudagraph. (Cheap out-of-band: single-digit overhead measured
  but catch unvalidated, Add.9-10.)
- ONE LINE: KILL -- fused-ABFT FP8 fails GATE 1 (our fusable Triton base GEMM is 3-4x slower than
  CUTLASS; the only CUTLASS-matching kernels are non-fusable); fusion would need a vLLM-tuned-parity
  custom CUTLASS kernel we have not built; always-on in-graph recompute (-51.9%, validated) remains
  the best deployable FP8 detector. Anchor end: 2edba0d2136f8ede4713d90a8f7cd55f.

---
## ADDENDUM 12 — OUT-OF-BAND catch at N=512, GATE B (catch): INCONCLUSIVE 3rd time (FP8 weight
## indirection blocks fault injection into the exposed GEMM). GATE A skipped. 2026-06-12

Fixed discriminator (avoids Add-10's cached-recompute trap): compare exposed obuf after a
corrupted-weight forward vs SAVED clean obuf for the SAME input (corrupt ONLY the target layer ->
its input unchanged -> direct same-input tensor compare, no recompute). In-process, default config.
Anchor 2edba0d2 entry==exit; scratch (gateB_physical.py); anchor .so untouched.

### GATE B result — physical fault does NOT reach the exposed GEMM
| corruption | generation_diverged | resid_xbuf (input) | resid_obuf (exposed output) |
|---|---|---|---|
| target weight x4 on 256/4096 rows | True | 0.0 | **0.0** |
| target weight ZEROED (entire) | True | 0.0 | **0.0** (bit-identical) |
Both: generation DIVERGED (something uses layer.weight.data) but the exposed obuf is BIT-IDENTICAL to
clean (resid_obuf=0 exactly), for the same input (resid_xbuf=0). A ZEROED weight would force the
GEMM output to ~0; if the EXPOSED decode GEMM read layer.weight.data, obuf would be hugely different.
Bit-identical -> **the exposed decode GEMM does NOT read layer.weight.data** -- it reads a processed/
repacked FP8 weight (the same indirection that made Add.10 inconclusive). My corruption never reaches
the GEMM whose output I expose. Generation diverges via a DIFFERENT use of layer.weight.data (eager
prefill / a different reference).

### Why this is INCONCLUSIVE, not a KILL (honesty gate)
resid_obuf=0 does NOT prove the copy is blind -- it proves the exposed GEMM used a CLEAN weight (my
fault missed it). The copy could be faithful (correctly showing the clean-weight output) OR lossy; I
CANNOT distinguish, because I could not inject a fault into the exposed GEMM. This is the THIRD
obstruction of the out-of-band catch validation: Add.9 (inductor reordered a visible inject), Add.10
(cached-weight recompute reference inert), Add.12 (weight corruption can't reach the processed-weight
GEMM). All three are TEST-INJECTION obstacles, not proofs the detector fails.

### VERDICT
- GATE B: UNRESOLVED on this stack via weight corruption (FP8 processed-weight indirection). GATE A
  (cost at N=512) SKIPPED -- B did not pass, so cost is moot.
- The ONLY remaining clean validation: a CUDA-level shim (.so) that corrupts the OUTPUT tensor's
  device memory directly (bypasses the weight indirection entirely; the fault is in `out`, which the
  copy reads), timed on the same stream after the GEMM. That is the .so delivery path and is the
  concrete unbuilt next step to settle out-of-band catch once and for all.
- BEST PROVEN-DEPLOYABLE remains always-on IN-GRAPH exact recompute (Add.4): -51.9%, 100% VALIDATED
  catch (inline compare -- no side-channel/weight-indirection to obstruct), 0 FP, injection-only,
  default cudagraph. This is the honest shippable FP8 detector.
- ONE LINE: out-of-band catch at N=512 is UNRESOLVED (weight-corruption fault can't reach the exposed
  processed-weight GEMM; 3rd injection obstruction) -> not deployable on validated grounds; the
  always-on in-graph detector (-51.9%, validated) is the answer unless a CUDA output-memory shim
  (.so) is built to settle the out-of-band catch. Anchor end: 2edba0d2136f8ede4713d90a8f7cd55f.

---
## ADDENDUM 13 — OUT-OF-BAND catch, 3 more cudagraph reformulations + an EAGER control. 2026-06-13.
## Catch LOGIC validated in eager (copy + discriminator correct, 0 FP); cudagraph block ISOLATED to
## inductor buffer-realization (not a detector flaw). The .so output-memory shim remains the one
## clean cudagraph test. Anchor 2edba0d2 entry==exit; scratch (cgperiodic_oob_v{2,3,4eager}.py); .so untouched.

Continuation of Add.9/10/12. Goal: settle whether the in-graph copy `obuf.copy_(out)` faithfully
captures a corrupted `out` so the out-of-band recompute catches it. Three new in-graph fault models
plus an eager control, all on real `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8`, default cudagraph
unless noted. (Env note this session: vLLM auto-loaded an installed `cipher_vllm_kv` general plugin
that crashed KV init, and DeepGEMM warmup was required-but-absent — both are ENVIRONMENT drift, not
the experiment; cleared with `VLLM_PLUGINS="" VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip`, which
restores the parked-substrate cutlass path the prior addenda measured.)

### v2 — FUNCTIONAL fault feeding BOTH copy and downstream (cgperiodic_oob_v2.py, oobv2_n8.json)
Hypothesis: Add.9's in-place fault was reorderable because the copy was a separate reader; make the
fault a functional op returning a corrupted tensor and feed THAT to both the copy and the return, so
a single data dependency forces all readers to the same value. RESULT (N=8, default cudagraph):
generation DIVERGED (inj_out != clean -> downstream consumed the corrupted value) but obuf stayed
CLEAN (inj_max_resid 0.0, caught_within_N false, target_direct_resid 0.0). Inductor still realized a
clean source buffer for the copy. **In-place vs functional is irrelevant — the split between the
corruption and a separately-traced copy is what inductor exploits.**

### v3 — FUSED op: corruption + both copies INSIDE one opaque custom op (cgperiodic_oob_v3.py, oobv3_n8.json)
Hypothesis: remove inductor's scheduling freedom entirely — do `out.add_(flag*5)` then `obuf.copy_(out)`
inside ONE custom op (`cexpose::expose`, schema `Tensor(a!) out ... -> Tensor(a!)`), so the copy
provably reads post-corruption memory atomically. RESULT (N=8, default cudagraph): generation did NOT
even diverge (inj_diverged false) and obuf clean (resid 0). The in-place mutation of the GRAPH-INTERNAL
`out` buffer did not propagate at all — consistent with Add.4's finding that only MODULE-BUFFER
mutations are lifted as graph inputs and persist through capture; `out` is not a module buffer, so a
custom-op in-place write to it is dropped under cudagraph. Third independent in-graph reformulation,
third distinct failure mode.

### v4 — EAGER control (enforce_eager=True; cgperiodic_oob_v4eager.py, oobv4_eager.json) — CAUGHT
With inductor and cudagraph OFF, apply_weights runs in strict program order: in-place fault on `out`
then `obuf.copy_(out)`. RESULT: **clean_max_resid 0.0 (0 FP) ; inj_max_resid 0.29538 with worst_layer
== target_layer == 64 ; caught=True.** (inj_diverged false — the +5.0 on one element of one layer was
sub-threshold for greedy token flips over 64 tokens, but the residual check caught it exactly at the
faulted layer.) **This isolates the failure: the copy + out-of-band discriminator LOGIC is correct
(faithful capture, 0 FP, exact localization) whenever ordering is respected. The 3x cudagraph block
(Add.9/v2/v3) is specifically inductor's freedom over buffer realization — NOT a detector flaw.**

### VERDICT (Add.13)
- Out-of-band catch is now VALIDATED IN EAGER (logic sound: 0 FP, caught, localized to the faulted
  layer). Under DEFAULT cudagraph it remains UNVALIDATED after 3 more honest attempts (v2 functional,
  v3 fused) — every in-graph traced injection is defeated by inductor buffer-realization, exactly as
  Add.12 predicted.
- The ONLY remaining clean cudagraph validation is unchanged and now triply-confirmed as necessary: a
  CUDA `.so` that corrupts `out`'s DEVICE MEMORY directly, outside inductor's SSA model (a real SDC is
  not a traced op). Not built — it is the single open thread if cheap detection is ever revisited.
- BEST PROVEN-DEPLOYABLE remains always-on IN-GRAPH exact recompute (Add.4): -51.9%, 100% validated
  catch, 0 FP, injection-only, default cudagraph.
- ONE LINE: out-of-band catch LOGIC validated in eager (0 FP, caught, localized); default-cudagraph
  catch still walled by inductor buffer-realization after 3 reformulations; the .so output-memory shim
  is the lone unblocked test. Anchor end: 2edba0d2136f8ede4713d90a8f7cd55f.
