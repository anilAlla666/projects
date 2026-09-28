# V1 capability audit + product architecture re-anchor

**Substep:** B.6''.9.X (codebase capability audit). **Date:** 2026-05-27.
**Status:** READ-ONLY audit complete; HOLD for Anil review of substep sequence
and Section E memory anchor proposal.

**Anchors at audit close (unchanged):**
- `cipher_rt_phase4` `959a6f5`
- `cipher_kmod` `8c643fc`
- `cipher-platform v2.0 rev6` md5 `7c7068ca`
- `cipher-fusion-evidence` — `3ae4043` HEAD; this memo + `.8.5b.4` Framing Y memo both pending commit (auto-mode classifier blocked Anil-identity commits this session; the .8.5b.4 memo is staged with md5 `b60062c3…`; this memo will be staged adjacent)

---

## Headline finding

The keystone gap for the workload-adaptive product is **the Workload
Classifier**. It does not exist as code today:
- No `cipher_workload_detect.cpp` file in the tree.
- No `CipherProfile` struct, no signal-capture surface for process count /
  NCCL presence / capture mode / dtype / batch size / backward-pass detection.
- Classification happens inline in `cipher_dispatch.cpp` via `classify_launch()`
  using kernel geometry only (grid, block, shared) — no workload-level
  signals composed into a profile.

Without it, "customer installs .deb → CIPHER auto-activates appropriate
capability subset" is structurally impossible. Every other gap is downstream
of this one.

The substrate beneath the missing keystone is mostly built: kernel intercept
(L1) is robust; substitution machinery (Marlin, Koopman, EDMD pipeline) is
present; multi-tenant primitives (KV-dedup, weight-sharing, green-ctx,
arena, NR 27) are functional. They are individually env-gated OFF or
gate-restricted (fp16-only) such that on a stock customer deployment, the
Classifier-shaped opt-in surface is missing, and so nothing engages.

Two further architectural findings worth surfacing before substep proposals:

1. **Two parallel dispatch systems exist and are not wired together**: the
   may13-era `cipher_dispatch()` (cipher_rt_phase4/src/may13/cipher_dispatch.cpp:339-517)
   operates at the cuLaunchKernel level and contains the EDMD→Koopman pipeline
   end-to-end; the v1 substrate `cipher_rt_matmul_dispatch` operates at the
   cublasGemmEx-shim level and contains the registered-actuator model
   (Marlin, Koopman engine). They classify and route independently and do not
   share state. The v1 `cipher_rt_classify_substrate.cpp:11` explicitly notes
   "Step 6 wires it through the F1 cuLaunchKernel intercept" — that wiring
   step never landed.

2. **NR 27 REGISTER_MODEL is functional in kmod**
   (cipher_kmod/cipher_model_registry.c:107-161, 256-bucket FNV-32 hashtable,
   spinlock-safe) but is **gated off in the CDI patch userspace path**
   (`/usr/lib/cipher/cipher_cdi_patch.py:254` `CIPHER_REGISTER_MODEL=0`) due
   to the B.6''.9.1 vLLM plugin crash. The kmod surface is solid; the
   userspace consumer (`cipher_vllm_kv.py` REGISTER_MODEL path) is the bug.
   Downstream consumers (G3 KV dedup model-keying, G4 Marlin model-keying,
   G12 Koopman model-keying) all expect this gate to be on.

---

## Section A — Capability matrix

Status legend: **BUILT** = production-ready, **PARTIAL** = exists with
known gap, **SPEC-ONLY** = designed/scaffolded but no functional code,
**MISSING** = absent.

| # | Capability | Status | Default engagement | Anchor file:line | Notes |
|---|---|---|---|---|---|
| 1 | Driver-API intercept (cuLaunchKernel + variants + cudaLaunchKernel + cuGraph* + cuModule* + cuLibrary*) | **BUILT** | ON (auto-init) | `src/may13_intercept/cipher_intercept_cudart.cpp:3034-3095` (runtime API), `:2096-2100` (driver API), `:1832-1958` (module/library) | All hot paths shimmed. Driver-API hooks present but several are stubbed-passthrough (no rewrite). |
| 2 | Layer 3 Substitutor classify_route + dispatch | **PARTIAL** | ON in may13 path; OFF in v1 substrate path | may13: `src/may13/cipher_dispatch.cpp:339-517`. v1: `cipher_rt_classify_substrate.cpp:50-68`. | Two parallel dispatch systems not wired together (see Headline #1). |
| 3 | Layer 2 Orchestrator (SM packing + Green Ctx + NCCL + fusion + prefetch) | **PARTIAL** | observation-only | `cipher_rt_sm_packer.c:33-117`, `cipher_rt_partition_router.c:105-131`, `include/may13/cipher_liquid_state.h:14,107-118` | Liquid state fields present; no active fusion/routing kernel. SM packer counts streaks but no persistent kernel to act on detection. |
| 4 | Layer 1 Generator (EDMD + Koopman runtime registry) | **BUILT** | `CIPHER_EDMD_LIVE` default ON | `src/may13/cipher_edmd.cpp:1-150`, `src/may13/cipher_edmd_live.cpp:1-150`, `src/may13/cipher_block_sub_kernel.cu:57-122` | TARGET_ROWS=2000, KR_RANK=64, POWER_ITERS=4. Pipeline runs; convergence-time is the .8.5b.3 gap (~60s under realistic prefill+decode load). |
| 5 | Workload Classifier (cipher_workload_detect.cpp + CipherProfile + dispatcher) | **MISSING** | — | (no file in tree) | Classification inline only via `classify_launch()` (geometry-only). No CipherProfile struct, no signal capture for NCCL/dtype/process-count/capture-mode. **KEYSTONE GAP.** |
| 6 | Marlin actuator (cuBLAS-fp16 INT4 substitution) | **PARTIAL** | `CIPHER_MARLIN=on` required (OFF default) | `cipher_rt_marlin_actuator.c:120-146`, `cipher_rt_marlin_engine.cpp:1051-1115` | fp16-only dtype gate (rejects bf16). cuBLAS-path only (bypassed by Machete/AWQ). Phase 5 partition constraint unresolved. |
| 7 | Koopman actuator (narrow-domain O(1) substitution + shape registry) | **PARTIAL** | `CIPHER_KOOPMAN=1` required (OFF default) | `cipher_rt_koopman_engine.cpp:109-201` | fp16-only dtype gate. Shape registry empty by default (calibration must accrue 2000 snapshots × ≥8 distinct ptrs × 16 shapes max). Auto-cal works but takes ~60s on real traffic. |
| 8 | VOLT actuator (NVML DVFS + adaptive clock lock) | **BUILT** | `CIPHER_VOLT=on + CIPHER_VOLT_BATCH` required (OFF default) | `cipher_rt_volt.c:253-330` | NVML + kmod ioctl paths; +57% TinyLlama-1.1B B=1, regression on Mistral-7B per `cipher-t43-envelope`. Bandwidth-bound regime detection NOT in code (claim from clock-lock only). |
| 9 | KV-prefix dedup (cross-process prefix sharing + TTL) | **PARTIAL** | always-on if `/dev/cipher_kvdedup` accessible | `cipher_rt_kv_alloc.c:505-696`, `cipher_kmod/cipher_kvdedup.c` | cuMemExportToShareableHandle + xxhash64-keyed + model_uuid mixed. Refcount-based eviction; **NO TTL** for tool-call-pause prefix retention (TTL is spec-only). |
| 10 | Weight-sharing arena (Track 2 cross-tenant residency) | **BUILT** | NR 21-24 ioctls available; userspace registration required | `cipher_kmod/cipher_arena.c`, `cipher_vllm_kv.py` (sc4 path) | SC6 anchor 76% saving Mistral-7B N=4 verified `track3-pre-step5` 2026-05-23. Same-model only; cross-model is Track 2.5/3 future. |
| 11 | Cross-tenant SM packer | **PARTIAL** | observation-only | `cipher_rt_sm_packer.c:33-117`, `cipher_rt_partition_router.c:114-131` | Observes per-stream streaks; no persistent kernel to actuate substitution. Sync-domain partitioning explicit no-op pending per-tenant memory-access telemetry. |
| 12 | Per-tenant routing (REGISTER_STREAMS NR 29 + stream→tenant resolver) | **BUILT** | always-on per-process | `cipher_kmod/cipher_dev.c` NR 29 case + `cipher_rt_phase4` resolver (W9 Step 5 close memory `week9-complete`) | tgid-keyed view-slot, vmalloc_user'd 8192-slot open-addressing, .mmap multiplexed; N=128 soak passed. |
| 13 | POOL substrate (cross-tenant batching for B=1 agent decode) | **PARTIAL** | `CIPHER_QOS_CLASS=POOL` required | `cipher_kmod/cipher_cp54_sched.c:17,266-349` (kmod scheduler), `cipher_rt_green_ctx.c:23-25` (userspace) | Kmod allocates POOL maximal contiguous SM groups. **Executor batching wiring NOT shipped** ("Step 1.3b'" stub). Cross-tenant B=1→B=N batching is not implemented in v1. |
| 14 | FlashAttention intercept (.8.6 scope target) | **BUILT** | always-on at init | `cipher_rt_attn_dispatch.cpp:1-86` | T4.6.1 SDPA shim on libtorch 2.11.0+cu130; three backend mangled symbols (Flash/Efficient/cuDNN). **Substrate built; HANDLED/REDIRECTED actuators not yet registered → attn_calls=0 measurement is wiring gap, not substrate gap.** |
| 15 | cuBLAS GOT-patch (cublasGemmEx + cuBLAS-Lt variants + 17 trampolines + dlsym hook) | **BUILT** | always-on at init | `cipher_inject.c:82-85` + `cipher_rt_got_patch.c:212-222` + LT-VARIANT log line at 17/17 armed | 25 slots across 28 modules; 7 targets registered per .8.5b.3 init log. |
| 16 | CUPTI launch observation | **BUILT** | always-on at init | `cipher_cupti.c:172` (SM packer observe hook), `cipher_v2_cupti_init` (cipher_inject.c:54) | Subscribed; kernel launch callbacks active; flush every 256 launches per init log. |
| 17 | CDI deployment patch + cipher-platform-watch daemon | **BUILT** | systemd watcher on /var/run/cdi/nvidia.yaml | `/usr/lib/cipher/cipher_cdi_patch.py:249-255` + `/lib/systemd/system/cipher-platform-watch.{service,path}` | Injects CUDA_INJECTION64_PATH, LD_LIBRARY_PATH, TORCH_CUBLASLT_DISABLE=1, CIPHER_REGISTER_MODEL=0. **NO actuator-opt-in env vars** (no CIPHER_VOLT/KOOPMAN/MARLIN/REMEMBER). |
| 18 | cipher_kmod (NRs 21-29 + audit ring + tenant snapshot) | **BUILT** | always-on once kmod loaded | `cipher_kmod/cipher_dev.c`, `cipher_audit_chain.c:1-150`, ko `0.6.5` ABI | NR 21-24 ARENA, 27 REGISTER_MODEL (kmod fine; userspace path crashes), 28 OBSERVE_PUBLISH, 29 REGISTER_STREAMS. 4096-entry / 256 KiB per-tenant HMAC-SHA256 audit chain. |
| 19 | FLOP telemetry + HMAC billing receipts | **PARTIAL** | `CIPHER_RECEIPT=on` for receipts; `/proc/cipher/flops` always | `cipher_kmod/cipher_flops.c:1-99` (FLOPS counter via attribution from device sample), `src/may13/cipher_receipt.cpp:116-210` (HMAC-SHA256 receipts) | FLOPS attribution proportional (no per-context HW counters); RECEIPT chains FNV-64 dispatch + launch counter. **FLOP counts NOT emitted into audit chain** — telemetry and audit are decoupled. |
| 20 | NCCL tuner plugin | **SPEC-ONLY** | — | `include/may13/cipher_liquid_state.h:107-118` + `:229` | Data struct only. NCCL collective interception explicitly de-scoped at `src/may13_intercept/cipher_intercept_cudart.cpp:1528-1529`. **Zero coverage of AllReduce / training overlap.** |
| 21 | Persistent kernel dispatch | **MISSING** | — | (no file in tree) | Zero matches for `persistent_kernel`/`persistent_dispatch`/`cipher_persistent`/`persistent_grid` in cipher_rt_phase4 or cipher_kmod. Pre-training and large-batch arithmetic intensity have no actuator. |
| 22 | Speculative decode integration | **BUILT** | `CIPHER_SPEC=1` default ON | `cipher_spec_decode.py:1-34`, `spec_measure_driver.py`, `spec_verify.py`, `spec_smoke.py` | AdaptiveK + NgramDraft + ModelDraft + spec_generate. Draft-stream overlap deprecated (cuDNN attention hang per CUDNN_ATTN_MARLIN_HANG.md). |
| 23 | CfC / LNN models (Layer 3 substitutor + Layer 2 NCCL policy) | **BUILT** | `CIPHER_REMEMBER=1` required | `include/may13/cipher_lnn.h:1-150`, `src/may13/cipher_lnn.cpp:1-200+` | 48-dim input × 64-dim CfC × 12-dim output, analytical (not learned) weights. Read by REMEMBER consumer; no Layer 2 NCCL active policy. |
| 24 | Auto-repatch + GOT walker | **BUILT** | always-on, triggered on dlopen events | `src/may13_intercept/cipher_intercept_cudart.cpp:2396-2410`, hooks at lines 1773,1859,1885,1948,2221,2325,2436 | Handles vllm/_C.abi3.so, flashinfer, bitsandbytes loaded post-init. `CIPHER_HOOK_PASSIVE=1` opt-out. |
| 25 | Per-launch atomicity (`cipher_rt_commit_observe_and_publish`) | **BUILT** | always-on | `cipher_rt_commit.c` (W7-9 Step 2 commit primitive close), `cipher_rt_commit_observe_and_publish` symbol per `commit-primitive` memory | 4/4 test atomicity PASS; p99 49 ns. Used by overlay-op `_report()` ports (19/23 ops). |

**Roll-up by status:**
- **BUILT** (12): driver-API intercept, L1 generator, weight-sharing arena, per-tenant routing, FlashAttention substrate, cuBLAS GOT-patch, CUPTI, CDI deploy, kmod, speculative decode, CfC LNN, auto-repatch, commit primitive
- **PARTIAL** (8): L3 substitutor (wiring gap), L2 orchestrator (no active routing kernel), Marlin (dtype+path gates), Koopman (dtype+registry gates), KV-prefix dedup (no TTL), SM packer (observation only), POOL substrate (no batching), FLOP telemetry (not in audit chain)
- **SPEC-ONLY** (1): NCCL tuner
- **MISSING** (2): Workload Classifier (KEYSTONE), Persistent kernel dispatch

**Total v1 coverage estimate:** ~55-60% of capability list is build-complete; the missing keystone (Classifier) is what holds back the assembly into a product.

---

## Section B — Workload-to-capability mapping

Each workload class lists the capabilities the Classifier must auto-activate
to deliver Memory #1 goals. Status from Section A in parentheses.

### B.1 Pre-training (large batch, NCCL-heavy, mixed precision)

**Auto-activate signal:** backward-pass kernels detected + NCCL collective
ops observed + large allocation sum + single-process-per-rank topology +
`cuStreamBeginCapture` for training-step graph.

| Capability | Status | v1 deliverable on this workload |
|---|---|---|
| VOLT (training power) | BUILT (env-off) | YES if Classifier sets CIPHER_VOLT |
| NCCL tuner | SPEC-ONLY | **NO — capability does not exist** |
| Layer 2 NCCL overlap | PARTIAL (no active routing) | **NO** |
| cuBLAS GOT (training GEMMs) | BUILT | YES |
| FLOP telemetry | PARTIAL | partial: counts attributed but not billing-grade |
| Persistent kernel dispatch | MISSING | **NO** |
| Marlin if INT4 weights | PARTIAL (fp16-only gate) | NO for bf16 training; NO for INT4 weight-only QAT |
| cuGraph hooks (training step) | BUILT (read-only) | observability only |

**Honest v1 ship on pre-training: VOLT-only.** No NCCL, no persistent
kernel, no INT4 path. The pre-training surface is essentially uncovered by
v1 capabilities beyond clock lock.

### B.2 Fine-tuning (medium batch, LoRA-shared backbone)

**Auto-activate signal:** backward-pass + smaller batch than pre-training +
LoRA-style adapter weight pattern (small N matrices alongside large) +
single-process or DDP.

| Capability | Status | v1 deliverable |
|---|---|---|
| VOLT | BUILT (env-off) | YES with Classifier |
| Weight-sharing arena (LoRA shared backbone) | BUILT | YES |
| Layer 2 NCCL overlap | PARTIAL | NO |
| cuBLAS GOT | BUILT | YES |
| FLOP telemetry | PARTIAL | partial |
| Marlin if INT4 | PARTIAL (fp16-only) | NO for bf16 |

**Honest v1 ship on fine-tuning: VOLT + weight-sharing arena for shared
backbone.** Modest coverage.

### B.3 Agent inference multi-tenant (3-100 concurrent agents, B=1, long ctx, tool calls)

**Auto-activate signal:** N≥2 CUDA contexts observed + B=1 decode pattern
+ same-model loaded across contexts + long context detected.

| Capability | Status | v1 deliverable |
|---|---|---|
| KV-prefix dedup (cross-process) | PARTIAL (no TTL) | YES for prefix sharing across same-model contexts; no TTL retention during tool-call pauses |
| Weight-sharing arena | BUILT | YES |
| POOL substrate (B=1→B=N) | PARTIAL (no batching) | **NO — kmod has scheduling but no executor batching** |
| Per-tenant routing (NR 29) | BUILT | YES |
| Cross-tenant SM packer | PARTIAL (obs-only) | observability only |
| Koopman (long-ctx attention) | PARTIAL (fp16-only + registry empty) | NO out-of-box; YES after .8.6c calibration on fp16-explicit models |
| FlashAttention intercept | BUILT (passthrough actuators only) | observability only; substitution actuator missing |
| VOLT | BUILT (env-off) | YES with Classifier |
| FLOP telemetry per-agent | PARTIAL | partial |

**Honest v1 ship on agents: KV-dedup + arena + per-tenant routing.**
POOL batching (the headline 3-6× density lever) is NOT shipped. This is the
single biggest gap relative to the Goal 1 "100 agents per H100" claim.

### B.4 Continuous batched serving (one vLLM, many sessions, mixed prefill+decode)

**Auto-activate signal:** one CUDA context + high stream count +
`cuStreamBeginCapture` detected + continuous-batching pattern (mixed
batch_size at each forward).

| Capability | Status | v1 deliverable |
|---|---|---|
| Per-tenant routing | BUILT | YES (tenant = logical session) |
| POOL substrate | PARTIAL | NO batching layer |
| Koopman (prefill attention) | PARTIAL | NO out-of-box; conditional after calibration |
| FlashAttention intercept | BUILT (passthrough) | observability only |
| VOLT | BUILT (env-off) | YES with Classifier |
| cuGraph hooks (vLLM capture/replay) | BUILT (read-only) | observability only |
| FLOP telemetry | PARTIAL | partial |

**Honest v1 ship on continuous batched serving: VOLT + observability.**

### B.5 Batch inference (large prompt batch, prefill-heavy)

**Auto-activate signal:** high batch size + prefill-dominant pattern +
single-tenant.

| Capability | Status | v1 deliverable |
|---|---|---|
| Marlin INT4 (if applicable) | PARTIAL (fp16-only + cuBLAS-path) | NO for AWQ INT4 (Machete bypass); NO for bf16; conditional for fp16 |
| cuBLAS GOT | BUILT | YES |
| Persistent kernel dispatch | MISSING | NO |
| VOLT | BUILT (env-off) | YES with Classifier |

**Honest v1 ship on batch inference: VOLT-only on the realistic dtype/quant
configs that customers actually run.**

### B.6 RAG / long-context (retrieval prefill-heavy attention)

**Auto-activate signal:** long context (>32K tokens) + prefill-heavy pattern.

| Capability | Status | v1 deliverable |
|---|---|---|
| Koopman (long-ctx attention O(1)) | PARTIAL | NO out-of-box |
| KV-prefix dedup (retrieved context cache) | PARTIAL (no TTL) | YES for repeated retrieval prefix; no TTL |
| FlashAttention intercept | BUILT (passthrough) | observability only |
| Weight-sharing arena | BUILT | YES if multi-tenant |

**Honest v1 ship on RAG: KV-dedup of retrieved-prefix cache; arena if multi-tenant.**

### B.7 Single-tenant streaming (one process, B=1, short ctx)

**Auto-activate signal:** single CUDA context + B=1 + decode-dominant +
short context.

| Capability | Status | v1 deliverable |
|---|---|---|
| VOLT (bandwidth-bound decode) | BUILT (env-off) | YES with Classifier |
| Koopman | PARTIAL | NO out-of-box |

**Honest v1 ship on single-tenant streaming: VOLT-only (+55% if memory-bandwidth-bound regime, regression risk at 7B+ scale per `cipher-t43-envelope`).**

---

## Section C — Per-goal × per-workload gap analysis

### Goal 1 — 100 concurrent agents per H100

**Required capability stack:** KV-prefix dedup + weight-sharing arena +
POOL substrate (batching) + cross-tenant SM packer + per-tenant routing.

**v1 status:** 3 of 5 BUILT (dedup, arena, routing). 2 of 5 PARTIAL
without the cross-tenant lever (POOL has no batching executor; SM packer is
observation-only).

**Path to engagement on agent workload:** POOL substrate needs the
executor batching wired (Step 1.3b' deferred per `cipher-cp54-step1-3b`
memory). Without it, 100-agent claim relies on memory savings alone, not
throughput density. SC6 anchor 76% memory saving Mistral-7B N=4 already
verified; the throughput-side anchor (cross-tenant batching 3-6×) was
measured on synthetic stress per `cipher-phase-a-multitenant`, not under
real multi-process executor batching.

**Recommended next step:** **ship POOL executor batching for B=1
multi-tenant decode** before claiming Goal 1 on agent workload.

### Goal 2 — 2× tok/W via stacked actuators

**Required capability stack per workload (covered in Section B):**

| Workload | Stack | Currently engaging out-of-box on stock vLLM? |
|---|---|---|
| Single-tenant fp16 | VOLT + Marlin + Koopman | NO (env off; registry empty) |
| Single-tenant bf16 (most modern models) | VOLT only | NO (env off; Marlin/Koopman dtype-reject) |
| AWQ INT4 | VOLT only (Marlin can't intercept Machete) | NO |
| Multi-tenant | VOLT + cross-tenant batching (POOL) | NO (POOL batching not shipped) |

**v1 status:** all four currently 0% engagement out-of-box because (a) CIPHER_ENV
doesn't inject opt-in vars, (b) Marlin/Koopman fp16-only, (c) POOL batching not
shipped, (d) VOLT magnitude unverified on rev6 substrate.

**Path:** Workload Classifier sets actuator env per workload class; Marlin
gets bf16 path OR Machete intercept; Koopman gets pre-shipped registry or
auto-cal at deploy. Per .8.5b.4: ~15-25 ED for fp16 coverage; ~20-25 ED for
bf16 coverage too.

### Goal 3 — 85%+ MFU cross-tenant

**Required capability stack:** SM packer (cross-tenant) + per-tenant routing
+ POOL substrate + FLOP telemetry (for measurement).

**v1 status:** routing BUILT; SM packer PARTIAL (observation); POOL PARTIAL;
FLOP telemetry PARTIAL.

**Path:** "85%+ MFU" needs the substrate to drive arithmetic intensity, not
just observe it. Persistent kernel dispatch (MISSING) would be the
arithmetic-intensity lever for batch inference. For multi-tenant agent
mixes, MFU rises with cross-tenant batching (POOL executor batching).
**Persistent kernel dispatch + POOL executor are the two missing pieces
gating Goal 3.**

### Goal 4 — O(1) Koopman substitution

**Required capability stack:** EDMD pipeline (L1) + Koopman engine
(actuator) + shape registry pre-population or auto-cal + dtype coverage.

**v1 status:** EDMD pipeline BUILT (functional end-to-end); Koopman engine
PARTIAL (fp16-only + empty registry by default); LM head 7.43× provenance
not located.

**Path:** ship calibration tooling (.8.6c per .8.5b.4 audit) and/or bf16
coverage; Workload Classifier triggers calibration warm-up on first
deployment for long-context-prefill workloads.

### Goal 5 — Transparent .deb deployment

**Required capability:** Workload Classifier-driven auto-activation.

**v1 status:** MISSING (keystone gap). CDI patch infrastructure BUILT;
auto-repatch BUILT; the wiring is "Classifier writes per-process env to
enable actuator opt-ins" — the Classifier itself doesn't exist.

**Path:** build the Classifier. Without it, Goal 5 is impossible regardless
of any other capability.

### Cross-goal roll-up

| Goal | Capabilities BUILT | Capabilities PARTIAL with known gap | Capabilities SPEC-ONLY or MISSING |
|---|---|---|---|
| 1 (100 agents) | 3 (dedup, arena, routing) | 2 (POOL, SM packer) | 0 |
| 2 (2× tok/W) | 1 (VOLT substrate) | 3 (Marlin, Koopman, env-injection) | 0 |
| 3 (85% MFU) | 1 (routing) | 3 (POOL, SM packer, FLOP telemetry) | 1 (persistent kernel) |
| 4 (O(1) substitution) | 1 (L1 EDMD) | 1 (Koopman engine — dtype + registry) | 0 |
| 5 (transparent deploy) | 2 (CDI, auto-repatch) | 0 | **1 (Workload Classifier — KEYSTONE)** |

---

## Section D — Substep sequence to v1 ship

Each substep declares: **capability advanced**, **goal × workload advanced**,
**customer-journey delta**, **closing gate** (which must be a stock-workload
goal-engagement measurement, not a substrate counter).

### D.1 Architectural keystones first

**SUBSTEP K.1 — Build Workload Classifier (`cipher_workload_detect.cpp`)** ⟶ **gates everything downstream**

- Capability: #5 (Workload Classifier).
- Signals to capture (framework-agnostic per Memory #9): process count
  on /dev/cipher devnode + observed CUDA contexts (`cuCtxGetCurrent` walk),
  GEMM shape histogram (from existing matmul-dispatch observer), kernel
  name patterns (cuFuncGetName on existing intercept), batch-dim inference
  from grid geometry, dtype from cuBLAS shim Atype/Btype, NCCL presence
  (`cuMemAlloc` + bind pattern via libnccl symbol presence in dlopen
  events), memcpy pattern (cudaMemcpy hooks), capture mode
  (cuStreamBeginCapture hook), backward-pass detection (loss.backward
  kernel patterns from existing kernel registry).
- CipherProfile struct: `{workload_class, dtype, model_family, multi_tenant,
  long_context, captured_graph, backward, batch_size_estimate, …}`.
- Dispatcher: writes per-process actuator env (via /proc/self/environ
  manipulation NOT viable at runtime; instead via a SHM-side capability
  toggle that each actuator's init reads OR via cipher-platform-watch
  re-injecting on detection).
- Customer-journey delta: customer's `vllm serve` is observed; profile
  emitted in cipher-platform status; nothing engages yet (separate
  substeps wire actuators to the profile).
- Closing gate: **on three stock workloads (Llama-3-8B bf16, Mistral-7B
  default bf16, TinyLlama-AWQ), the cipher-platform CLI reports the
  correct profile within 30 seconds of `vllm serve` start.** No tok/W
  claim yet; this is the Classifier-only gate.
- ED: ~5-7.

**SUBSTEP K.2 — Wire the two dispatch systems together**

- Capability: #2 (Layer 3 Substitutor — unify may13 cipher_dispatch and
  v1 matmul-dispatch substrate).
- The may13 cipher_dispatch should call into the v1 cipher_rt_classify_route,
  and cipher_rt_matmul_dispatch should consume the may13 classifier's
  op_class instead of just dtype/shape. Per
  `cipher_rt_classify_substrate.cpp:11` this is the "Step 6" referenced
  but never landed.
- Customer-journey delta: invisible to customer; eliminates the parallel
  dispatch path divergence that's been a recurring source of "engagement
  zero" findings.
- Closing gate: **cipher_rt_matmul_dispatch_route returns Koopman-engaged
  HANDLED for a calibrated shape AND cipher_dispatch (may13) returns
  CIPHER_HANDLED for the same call — single dispatch decision, single
  routing path.**
- ED: ~3-4.

### D.2 Wire existing BUILT capabilities into Classifier dispatch

**SUBSTEP W.1 — VOLT auto-activation by workload class**

- Capability: #8 (VOLT) wired to Classifier #5.
- When Classifier emits `bandwidth_bound_decode=true`, dispatcher sets
  VOLT clock-lock to the appropriate batch-conditional target (existing
  `batch_to_mhz` table in `cipher_rt_volt.c:286`). Otherwise VOLT
  remains off.
- Customer-journey delta: stock `vllm serve TinyLlama` sees clock locked
  to memory-bandwidth-bound point; tok/W rises measurably.
- Closing gate: **stock-config TinyLlama-1.1B fp16 single-tenant `vllm serve`
  shows ≥+30% tok/W vs vanilla, no env vars set by customer.** (Lowered
  from historical +55% to honest reproducible target; .8.5b.4 noted +55%
  unverified on rev6.)
- ED: ~2-3 (including the rev6 substrate re-verification deferred at .8.5b.4).

**SUBSTEP W.2 — Marlin via Machete symbol intercept (Option iv)**

- Capability: #6 (Marlin) + new actuator on a parallel
  `cipher_rt_machete_dispatch` substrate.
- GOT-patch `_ZN7machete11mm_dispatchENS_6MMArgsE` in `vllm/_C.abi3.so`.
  Classifier triggers `CIPHER_MACHETE=on` when AWQ-INT4 workload class
  detected. v1 ship target: observability + counter parity (no
  substitution yet).
- Customer-journey delta: customer's AWQ INT4 vLLM is *observed* by CIPHER;
  Machete call counter present in cipher-platform status. No tok/W lift
  claim yet — this is the substrate to enable future substitution.
- Closing gate: **on TinyLlama-AWQ stock `vllm serve`, Machete call
  counter is non-zero and matches CUPTI launch count for MacheteCollectiveMma
  kernels within 1% over a 5-min decode run.**
- ED: ~3-4.

**SUBSTEP W.3 — Koopman calibration tooling (.8.6c)**

- Capability: #7 (Koopman) — pre-ship registries + cipher-platform CLI for
  per-model calibration.
- New deliverable `/usr/lib/cipher/registries/<model_uuid>.npz` for common
  fp16 model families. New CLI subcommand `cipher-platform calibrate
  --model PATH --workload synth|user-trace`.
- Customer-journey delta: customer installs .deb on fp16 model and Koopman
  engages on calibrated shapes from minute one. Runs `cipher-platform
  calibrate` for own model.
- Closing gate: **on stock `vllm serve TinyLlama-1.1B-fp16`, Koopman
  handled_count > 0 within 30 seconds of first inference; KL preservation
  ≤ ε on validation prompts.**
- ED: ~7-9.

**SUBSTEP W.4 — POOL executor batching for B=1 multi-tenant decode**

- Capability: #13 (POOL substrate) — the deferred Step 1.3b' wiring.
- Build the executor that aggregates B=1 decode launches from N tenants
  on the POOL green-ctx into a single B=N forward, scheduled per the
  existing kmod POOL scheduler.
- Customer-journey delta: N concurrent `vllm serve` instances on same
  model see ~N× density at unchanged tok/W; ~3-6× density at
  modest tok/W cost depending on N.
- Closing gate: **4 concurrent `vllm serve Mistral-7B-Instruct` on same
  H100 sustain combined throughput ≥ 2.5× a single-instance baseline;
  per-tenant KL=0 vs vanilla; per-tenant tail latency degradation <2×.**
- ED: ~8-12. **This is the Goal 1 lever.**

**SUBSTEP W.5 — FlashAttention substitution actuator (Goal 4 on attention)**

- Capability: #14 (FlashAttention intercept) — register HANDLED/REDIRECTED
  actuators on the SDPA dispatcher.
- Long-context prefill attention → Koopman-style narrow-domain surrogate
  OR cross-tenant KV-block re-use. Closes the attn_calls=0 measurement
  gap from B.6''.9.4.
- Customer-journey delta: long-context prefill workloads (RAG, agent
  context > 32K) see tok/s lift on prefill.
- Closing gate: **on a 32K-context RAG decode of TinyLlama-fp16, attn_calls
  handled fraction ≥ 50% with KL preservation ≤ ε.**
- ED: ~5-8.

**SUBSTEP W.6 — CIPHER_ENV CDI patch update + NR 27 userspace fix**

- Capabilities: #17 (CDI) + dependency-resolution for #18 (kmod NR 27).
- Inject Classifier-driven opt-in env vars at CDI level; OR build the
  Classifier-driven SHM-toggle path (alternative architecture). Fix the
  vLLM plugin REGISTER_MODEL crash (B.6''.9.1 root cause is
  kmod-mmap-interaction outside ASan coverage per
  `WEEK_13_14_SCOPE_LOCK.md` and `pause_note.md`).
- Customer-journey delta: customer's `vllm serve` calls REGISTER_MODEL
  ioctl without crash; downstream G3/G4/G12 model-keying engages.
- Closing gate: **stock `vllm serve Mistral-7B` runs for 30 minutes with
  CIPHER_REGISTER_MODEL=1; zero crashes; model_uuid emitted in CIPHER
  status.**
- ED: ~3-5 (the kmod-mmap bug is the unknown ED component).

### D.3 PARTIAL fixes (known gaps)

**SUBSTEP P.1 — KV-prefix dedup TTL for tool-call pauses**

- Capability: #9 KV-prefix dedup TTL.
- Add timer-based retention for cuIpc-shared prefix pages during tool-call
  pauses; current refcount-only model evicts immediately on session end.
- Customer-journey delta: agent workloads with tool-call interruptions
  retain shared prefix cache across pause; subsequent decode reuses it.
- Closing gate: **on 100 concurrent agents with tool-call interruptions
  averaging 5s, prefix cache hit rate ≥80% after 10-min steady-state.**
- ED: ~3-4.

**SUBSTEP P.2 — FLOP telemetry → audit chain integration**

- Capability: #19 FLOP telemetry + #18 audit chain.
- Per-op FLOP counts emitted into HMAC chain. Enables billing-grade
  receipts.
- Customer-journey delta: customer runs cipher-platform audit-verify and
  gets cryptographically signed billing ledger.
- Closing gate: **30-min agent inference workload produces audit chain
  with FLOP counts matching ground-truth (vLLM accounting) within ±1%.**
- ED: ~4-5.

### D.4 SPEC-ONLY / MISSING gaps (v1 or v1.x decision)

**SUBSTEP M.1 — NCCL tuner (PRE-TRAINING coverage)**

- Capability: #20 NCCL tuner.
- Build LD_PRELOAD shim on `ncclAllReduce` / `ncclBroadcast` + the
  liquid-state-fed Layer 2 NCCL overlap routing.
- Customer-journey delta: pre-training workloads see AllReduce overlap.
- Closing gate: **on a 2-rank Mistral-7B pre-training step, AllReduce
  duration p99 reduces ≥20% with correctness preserved.**
- ED: ~10-15. **v1 vs v1.x decision: defer to v1.x unless pre-training is
  a v1 ship surface.**

**SUBSTEP M.2 — Persistent kernel dispatch (BATCH INFERENCE coverage)**

- Capability: #21 Persistent kernel dispatch.
- Build a resident persistent kernel that consumes work-items from a
  per-tenant ring buffer; SM packer streak detection triggers dispatch
  redirect to it.
- Customer-journey delta: batch inference at high arithmetic intensity
  sees MFU rise.
- Closing gate: **single-tenant batch=64 Mistral-7B prefill sustains
  ≥80% MFU.**
- ED: ~10-15. **v1 vs v1.x decision: defer to v1.x unless batch inference
  is a v1 ship surface; v1 batch inference would ship VOLT-only.**

### D.5 End-to-end goal validation (Phase E CP 5.5)

**SUBSTEP V.1 — CP 5.5 stock-customer-mix soak**

- All 5 Memory #1 goals validated on a 100-mixed-workload soak as defined
  in v1.2.3 CP 5.5 scope.
- Workload mix per Section B: 30 agent-inference + 30 continuous-batched-serving
  + 20 single-tenant streaming + 10 batch-inference + 10 RAG. Mixed
  dtypes (bf16, fp16, AWQ INT4); mixed models (Llama-3-8B, Mistral-7B,
  TinyLlama, Qwen-2.5).
- Closing gates (ALL must pass; substep does not close unless all gates
  pass on stock-config workloads with zero customer env intervention):
  - Goal 1: ≥10 concurrent agents per H100 sustained at KL=0 (interim
    target; "100 agents" is post-v1 once SM packer arbitration lands)
  - Goal 2: ≥1.5× tok/W on weighted mix vs vanilla (interim; 2× is post-v1
    if bf16 path lands)
  - Goal 3: ≥70% weighted-mean MFU (interim; 85% post-v1 with persistent
    kernel)
  - Goal 4: Koopman handled_count > 0 on long-context prefill class with
    KL preservation
  - Goal 5: cipher-platform status reports correct profile + activated
    capability set within 30s of `vllm serve` start, no customer env
- ED: ~3 (the soak itself is short; the deliverable is the engagement gate
  report).

### D.6 Ordered substep sequence with calendar estimate

**Phase B remaining + Phase E:**

| Substep | Description | ED | Gates |
|---|---|---|---|
| K.1 | Build Workload Classifier | 5-7 | Profile correct on 3 stock workloads |
| K.2 | Wire dispatch systems | 3-4 | Single-routing-path test |
| W.6 | CIPHER_ENV CDI + NR 27 userspace fix | 3-5 | REGISTER_MODEL stable for 30 min |
| W.1 | VOLT auto-activate | 2-3 | TinyLlama-fp16 +30% tok/W stock |
| W.3 | Koopman calibration tooling | 7-9 | Koopman handled > 0 on stock |
| W.2 | Marlin via Machete intercept (observability) | 3-4 | Machete counter parity |
| W.4 | POOL executor batching | 8-12 | 4× Mistral-7B 2.5× combined |
| W.5 | FlashAttention substitution actuator | 5-8 | attn handled ≥50% on 32K RAG |
| P.1 | KV-prefix dedup TTL | 3-4 | Agent prefix-cache hit 80% |
| P.2 | FLOP → audit chain | 4-5 | Audit ledger ±1% |
| V.1 | CP 5.5 stock-mix soak | 3 | 5/5 interim goals on stock config |

**Sub-total Phase B remaining + Phase E ship: ~46-64 ED.** At 1 ED/day:
**10-13 calendar weeks** until v1 ships with all 5 goals delivering on the
stock customer surface (with interim per-goal targets, not the Memory #1
maxima).

**v1.x scope (deferred from v1):**
- M.1 NCCL tuner (~10-15 ED)
- M.2 Persistent kernel dispatch (~10-15 ED)
- bf16 Marlin coverage (~5-7 ED per .8.5b.4)
- Memory #1 maxima (100 agents, 2×, 85% MFU) lift from interim targets

**v1 honest pitch:**
> CIPHER v1 delivers measurable tok/W lift on stock OSS vLLM inference
> workloads via the Workload Classifier-driven actuator stack:
>  - Single-tenant fp16: +30%+ via VOLT (TinyLlama scale; 7B+ degradation
>    risk acknowledged)
>  - Multi-tenant same-model: ~2.5×+ density via KV-dedup + arena + POOL
>    executor batching
>  - Long-context prefill: Koopman substitution available for calibrated
>    shapes
>  - Cryptographic audit ledger with HMAC-signed FLOP-billed receipts
> v1.x adds bf16 actuator coverage, NCCL training overlap, and persistent
> kernel dispatch for batch inference; Memory #1 maxima (100 agents,
> 2× tok/W universal, 85% MFU) are the v1.x ship targets.

---

## Section E — Memory anchor proposal: replacing Memory #11 "deferred to Phase B" pattern

### E.1 Why Memory #11's honest-residue pattern keeps misfiring

Memory #11 (`cipher-regression-discipline`) and the broader "honest residue"
practice was intended to PREVENT the substrate-keeps-shipping-product-keeps-not-engaging
trap. In practice, "honest residue" became a release valve: every step
closeout could record "engagement deferred to future" and the step still
closed. The .8.5b.3 and .8.5b.4 audits both surface this exact pattern.

### E.2 Proposed locked rule (NEW memory anchor)

**Memory `cipher-product-engagement-gate`** (proposed):

> **No substep closes unless its CLOSING GATE is a stock-config customer-workload
> measurement showing the substep's contribution survives on at least one of:
> {Llama-3-8B bf16 default, Mistral-7B-Instruct bf16 default, TinyLlama-AWQ
> INT4}.**
>
> **If a substep's contribution is zero on all three workloads under their
> default configurations, the substep ships as "substrate forward-compatible
> — does not affect default-config customer" in writing, and downstream
> substeps cannot claim aggregate Goal lift until at least one substep
> demonstrably moves the needle on at least one default config.**
>
> **The "honest residue" practice continues to RECORD limitations
> transparently. The locked rule is that recording the residue does NOT
> close the substep if engagement is zero across all three reference
> workloads.**
>
> **Why:** Both the may13 audit (mid-Phase 4) and .8.5b.4 (this session)
> surfaced the same pattern: substrate-level SCs passing while
> customer-visible delta stays zero. The discipline gap is at substep
> closure, not at any single phase. This rule binds closure to product
> reality.
>
> **How to apply:** every substep prompt should declare which of the three
> reference workloads it advances and what the engagement measurement is.
> Future audit substeps verify the measurement against actual stock
> deployment, not against substrate-internal counters.

### E.3 Companion: revise `cipher-fusion-campaign` Goal 2/3 framings

Update Memory #1 (`cipher-fusion-campaign`) Goal 2 + Goal 3 framings:

> **Goal 2 (revised, v1 interim):** ≥1.5× tok/W on weighted mix of stock OSS
> vLLM workloads (single-tenant fp16 + multi-tenant same-model bf16 +
> RAG fp16). 2× target lifted to v1.x once bf16 Marlin coverage + persistent
> kernel dispatch land.
>
> **Goal 3 (revised, v1 interim):** ≥70% weighted-mean MFU on the same mix.
> 85% target lifted to v1.x with persistent kernel dispatch.
>
> **Goal 1 (revised, v1 interim):** ≥10 concurrent same-model agents per H100
> sustained at KL=0 via KV-dedup + arena + POOL executor batching. "100
> agents" lifted to v1.x with cross-tenant SM packer arbitration kernel.

Memory `cipher-lift-framing` already establishes "ship on MFU gates not
TPW lift" — that discipline composes well; this revision narrows what
"MFU gates" and "TPW lift" must be measured on.

### E.4 Memory entries proposed for revision or deletion (Anil decision)

- **REVISE** `cipher-fusion-campaign` (Memory #1): incorporate the interim
  v1 targets per E.3.
- **NO CHANGE** `cipher-regression-discipline` (Memory #11): the regression
  discipline stays; only the substep-closure rule is new.
- **ADD** `cipher-product-engagement-gate` (NEW per E.2).
- **NO CHANGE** to Memory #13 LM head 7.43× provenance: per .8.5b.4
  Section B.5, no memory entry needs deletion — the framing lives only in
  session prompts and W13-14 task briefs.

---

## Section F — Working context update for future substep prompts

Every future substep prompt should declare:

```
WORKLOAD CLASS: { pre-training | fine-tuning | agent inference | continuous
                  batched serving | batch inference | RAG | single-tenant
                  streaming | substrate-internal }

CAPABILITY ADVANCED: { capability # from Section A }

GOAL × WORKLOAD ADVANCED: { Goal N on Workload W from Section B }

ENGAGEMENT GATE (closing condition): { customer-workload measurement on
                  one of three reference configs: Llama-3-8B bf16 default,
                  Mistral-7B-Instruct bf16 default, TinyLlama-AWQ INT4 }
                  
                  If substrate-internal: declare "substrate-only — does
                  not affect default-config customer in this substep; the
                  downstream consumer that moves the needle is substep X."
```

**Substep titles henceforth follow the pattern:** `<phase>.<step>.<capability-tag>
— <workload-class> <engagement-target>`. Example: "W.4 POOL executor
batching — multi-tenant same-model 2.5× density." Or
"K.1 Workload Classifier — substrate-internal (downstream gate is W.1
VOLT auto-activate on stock TinyLlama)."

This format makes the dependency chain visible at the prompt level —
substrate substeps cannot claim product impact, only product-tier substeps
can, and each product-tier substep cites its substrate-substep parent. The
intent is to make it structurally hard for a substrate-shaped prompt to
close while pretending to deliver product impact.

---

## HOLD point

**Decisions for Anil review (Section E primarily):**

1. **Adopt `cipher-product-engagement-gate` as a locked memory rule** per
   E.2, or modify the wording?
2. **Revise Memory #1 Goal targets** per E.3 to v1 interim values?
3. **Adopt the Section F substep-prompt template** for all future substeps?
4. **Approve the Section D substep sequence** (~46-64 ED Phase B remaining
   + Phase E), or reshape based on customer-surface priorities?
5. **v1 vs v1.x scoping**: confirm NCCL tuner + persistent kernel dispatch
   are v1.x (not v1)? confirm bf16 Marlin coverage is v1.x (not v1)?

No substrate touches in this audit. Memory #20 / #11 / #12 / #9 disciplines
preserved.

---

**Anchors at audit close (UNCHANGED — READ-ONLY):**
- `cipher_rt_phase4` `959a6f5`
- `cipher_kmod` `8c643fc`
- `cipher-platform v2.0 rev6` md5 `7c7068ca` (installed)
- `cipher-fusion-evidence` `3ae4043` HEAD; this memo pending commit (auto-mode
  classifier blocked Anil-identity commits this session)
