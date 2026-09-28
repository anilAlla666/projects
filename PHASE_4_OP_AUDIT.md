# Phase 4 — cipher_rt Op Audit (T4.0.1) — FULL

**Date:** 2026-05-13
**Source root:** `/home/ubuntu/cipher-may13-evidence/` (identical mirror at `/workspace/`)
**Binary inspected:** `libcipher_rt.so` (restored from `.preroadmap`, md5 `95284eb1c028382deef061cf679dcc7c`, 785352 B)
**Pre/post precondition check:** kmod fallback `55ab8c0c…`, libcipher_v2 fallback `86618c30…`, kmod loaded, taint 12288, /proc/cipher entries readable — all intact.

## Caveat — build broken on this pod

`make clean && make` failed: `src/cipher_10ops_impl.cpp` requires `openssl/hmac.h`; `libssl-dev` is not installed on this pod (only the pycparser fake stub at `/usr/share/python3-pycparser/fake_libc_include/openssl/hmac.h` exists). `make clean` deleted the session-start `libcipher_rt.so`; restored from `.preroadmap` for symbol inspection.

**Consequence:** every "working" claim in the State column is sourced from binary symbol presence (nm), source-file structure, and BUILD_STATE.md documentation, **not** from a fresh build on this pod. Cross-validation requires `apt-get install libssl-dev` + rebuild (T4.0.6 cut/keep call, not done here).

## Op universe

64 distinct `cipher_*_init` T symbols exported by `libcipher_rt.so`. Plus 10 sub-ops named inside the `cipher_10ops` bundle (Stage 0 ops 1–6 + Stage 1 ops 7–10) + 2 Stage 2 ops (ADAPT, ARBITRATE).

Audit covers all 64 + 12 sub-ops = **76 rows total**.

## Classification scheme

| Class | Meaning |
|---|---|
| **A** | Wants per-tenant state; rebuild against `cipher_tenant_snapshot` contract from T4.0.2 |
| **B** | Tenant-invariant by design; keep as-is, no fusion needed |
| **C** | Stub, dead, or no measurable lift — cut from Phase 4 |
| **D** | Deferred (NCCL multi-node, Koopman advanced) — out of Phase 4 scope |
| **INFRA** | Substrate the ops sit on; T4.1 may extend |

---

## Section A — Class A ops (fusion targets, rebuild for per-tenant state)

These ops currently consume global / session-fingerprint state and have a clear per-tenant analogue. They are the substantive Phase 4 work.

| # | Name | Source | State | Top-3 Inputs | Output Effect | MFU | TPW | Notes / Fusion blockers |
|---|---|---|---|---|---|---|---|---|
| 13 | SENSE | `cipher_sense.cpp` | working v1 | dispatch ring, env, kernel-class history | Session-classifier (HUMAN/AGENT/BATCH) → `session_fp` 64-bit hash | low | low | **Identity bridge needed:** SENSE's `session_fp` is the tenant key for FAIRNESS/CARBON/RECEIPT downstream. Phase 4 must add a `tenant_id`-string → `session_fp` mapping (or vice versa) so Phase 3's `CIPHER_TENANT_ID` env-stamped identity reaches these ops. |
| 14 | SHIELD | `cipher_shield.cpp` | working v1 | session band class (from SENSE), latency budget, kernel-class | Latency-protection priority hints + ARBITRATE band scan | mid | low | Already session-aware; rebuild = add tenant SLO targets from tenant_snapshot. |
| 15 | SUSTAIN | `cipher_sustain.cpp` | working v1 | KV pressure slope, dispatch rate, session class | KV-pressure flag → SUBSTITUTE compression hints | mid | low | Tenant-specific KV pressure differs per workload. |
| 16 | GUARD | `cipher_guard.cpp` | working v1 obs | KV cache pointer table, session_fp, leak signal | KV cache privacy enforcement (observer, hint-only) | n/a | n/a | Per-tenant leak detection is exactly what this should be — currently global. Fusion = enforce on tenant boundary. |
| 17 | PREDICT | `cipher_predict.cpp` | working v1 obs + v2 wiring | params_hash table, hot-pointer counter, session_fp | Proactive L2 preload candidate JSON; wires to `persist_engine_register` when ptrs go hot | **5–10%** | mid | **High leverage.** P4.4 target. Per-tenant hot regions trivially derivable. |
| 19 | CONTINUITY | `cipher_continuity.cpp` | working v1 obs | session_fp, KV region map, dispatch count | Per-session checkpoint manifests at 500-event boundaries | low | low | Already session-keyed; rebuild = use tenant_id as primary, session_fp as join. |
| 20 | THERMOSTAT | `cipher_thermostat.cpp` | working (Path B) | NVML temp, clock_sustained, power_draw_watts | Writes silicon dynamic fields (clock/headroom/power); upstream of THERMAL_FEEDBACK | n/a | mid | Today device-wide. Phase 4 wants per-tenant thermal accounting — easy via tenant_snapshot.thermal_headroom_pct. |
| 22 | PULSE | `cipher_pulse.cpp` | working v1 | NVML ECC errors, retired pages, throttle reasons | Hardware-fault early warning (score 0..2) | n/a | n/a | Device-wide signal. Tenants only care about the signal level; light fusion. |
| 23 | CARBON | `cipher_carbon.cpp` | working v1 | session_fp, grid×block work, env J/unit + gCO2/kWh | Per-session carbon estimate JSON | n/a | direct | **TPW directly:** rebuilds to per-tenant carbon attribution for the Prometheus exporter. |
| 24 | FAIRNESS | `cipher_fairness.cpp` (184 lines, fully read) | working v1 obs, hint-only | session_fp (mandatory), grid×block FLOP proxy, env quota | Per-session work-quota overrun flag | mid | mid | **Already tenant-aware** via session_fp (open-address table, MAX_TENANTS=256). Fusion = expose tenant_id mapping + add enforcement (currently hint-only). |
| 25 | TOPOLOGY | header-only in `cipher_dispatch.cpp` | working init-time | cudaDeviceCanAccessPeer matrix | NVLink/PCIe adjacency JSON | n/a | n/a | Single-GPU pod = no-op (n=1, 0 edges per MEMORY.md). Becomes Class A on multi-GPU. **Single-pod: effectively Class B today.** |
| 26 | LOOP | `cipher_loop.cpp` | working v1 hint | shape ring, decode/prefill ratio, prefill drought | Agentic runaway flag (score ≥ 2) | low | low | Per-tenant runaway differs by workload. |
| 27 | PIPELINE | `cipher_pipeline.cpp` | working v1 obs | session shape set, Jaccard pairs, session_fp | Multi-agent pipeline graph JSON | low | low | Sister to PIPELINE in fusion sense; already session-keyed. |
| 28 | TRACE | header-only in `cipher_dispatch.cpp` | working v1 | dispatch ring → 8192-record JSONL | Per-tenant trace export | n/a | n/a | Light fusion = add tenant_id label to JSONL records. |
| 30 | VOLT | `cipher_volt.cpp` (525 lines) | working (Path B, pod-degraded) | arithmetic intensity classifier, NVML clock-set support, calibration JSON | Per-shape SM-clock steering via `nvmlDeviceSetGpuLockedClocks` | **direct** | **3–5%** | **High leverage. P4.3 target.** Pod-degraded path is fine — bare-metal pods get real lift. |
| 31 | HIBERNATE | `cipher_hibernate.cpp` | working (Path B, pod-degraded) | dispatch idle gap, NVML powerMgmtLimit support | 5–50ms idle SM power gating | low | **2–4%** | P4.3 secondary. Bare-metal: real TPW lift. |
| — | STRAGGLER | `cipher_straggler.cpp` | working v1 hint | local kernel timing, NCCL algorithm enum | Slowdown signal + algo hint | low | low | Per-tenant straggler detection trivially adds via tenant_id rollup. |
| — | THERMAL_FEEDBACK | `cipher_thermal_feedback.cpp` | working v1 | silicon.clock_sustained, headroom, power_w | Aggressiveness score [0,1] for SUBSTITUTE | n/a | mid | Consumer of THERMOSTAT outputs. Tenant fusion = per-tenant aggressiveness. |
| — | PARTITION_ROUTER | `cipher_partition_router.cpp` (203 lines, fully read) | working v1 | pointer hash, silicon sm_count, env CIPHER_PARTITION_COUNT=2 | L2 partition binding {ptr→partition}; affects cuStreamCreate priority | **direct** | n/a | **P4.2 substrate.** Today: pointer-hash bound. Phase 4: tenant-bound (each tenant's hot regions → its assigned partition). |
| — | SM_PACKER | `cipher_sm_packer.cpp` (45 lines, fully read) | working decision-logic only | liquid.device.hw.sm_idle_fraction, idle EMA, idle threshold 5% | PACK/HOLD signal — **no actuation, ARBITRATE consumes** | mid | n/a | Decision exists; ARBITRATE makes the call. Phase 4: per-tenant pack decisions feed P4.2 SM partition allocator. |
| — | GREEN_CTX | `cipher_green_ctx.cu` (333 lines, fully read) | working via real CUDA 12.4+ driver API | cuDeviceGetDevResource, CIPHER_SM_COUNT, driver version | Reserves N SMs (default 8) for CIPHER's LNNs, splits rest for user | n/a | n/a | **NOT a stub** — uses `cuGreenCtxCreate`, `cuDevSmResourceSplitByCount`. Phase 5 critical path. **For P4: ARBITRATE may consume to rebalance per tenant.** |
| — | L2_PERSIST | `cipher_l2_persist.cu` | working with caveats | cudaAccessPropertyPersisting, silicon.l2_persist_max=31MB | Pins LNN weights in L2 cache | n/a | low | Direct L2 pin. P4.4: per-tenant L2 pin via PERSIST_ENGINE. |
| — | PERSIST_ENGINE | `cipher_persist_engine.cpp` (362 lines, fully read) | working v1 | {ptr, bytes, freq_score} table (max 64), silicon.l2_persist_max | Fractional knapsack admission → `cudaAccessPolicyWindow` per region | **5–10%** | n/a | **P4.4 substrate.** Per-tenant freq_score from CUPTI launches; admit tenant's hot regions. |
| — | WEIGHT_SHARE | `cipher_weight_share.cpp` (541 lines, fully read) | working — real cross-process IPC | content_hash, CUDA IPC handles, `/dev/shm/cipher_ws_…` files | Cross-process weight dedup via `cuIpcGetMemHandle` / `cuIpcOpenMemHandle` | n/a | huge | **Phase 5 critical path confirmed real.** Cross-tenant dedup of identical weights (LLM base models shared across tenants). Not currently keyed by tenant_id (keyed by content_hash) — leave that and **add tenant accounting on top** rather than re-key. |
| — | WEIGHT_COMPRESS | `cipher_weight_compress.cpp` (1426 lines) | working | weight tensor, compression dict, env CIPHER_WEIGHT_COMPRESS | INT4 / 2-bit weight quantization on detected hot weights | mid | mid | Per-tenant compression budget. |
| — | KV_COMPRESS | `cipher_kv_compress.cpp` (354 lines) | working v1 (Mistral-shaped) | KV region table (max 256), residual window, allowlist | KV cache 2-bit quantization (LLM-specific) | mid | low | LLM inference tenants only. P4.6 candidate. |
| — | KV_REDIRECT | `cipher_kv_redirect.cpp` (862 lines) | working V1 (memcpy) + V3 persistent cache | KV layer ID (max 32), 256MB scratch per layer, batch/token dims | KV-cache buffer redirection + persistent compressed cache | mid | low | Pairs with KV_COMPRESS. Mistral-7B-shaped today (P=2048, B=8); other tenants need re-tune. |
| — | FAIRNESS_SHM | `cipher_fairness_shm.cpp` (229 lines, fully read) | working — cross-process | POSIX shm `/cipher_fairness`, MAX_TENANTS=64 slots, tenant_id uint32 + gemm_calls | Cross-process tenant gemm-call accounting + yield signals | n/a | n/a | **Already cross-tenant aware** with explicit tenant_id field (32-bit). Phase 4 = bridge to Phase 3's tenant_id string. Either keep 32-bit hash, or extend slot struct. |
| — | FUSION | `cipher_fusion.cpp` (79 lines) | working orchestrator | kernel class, oracle hit, dispatch decision | Fuses adjacent kernels into one launch | mid | low | Tenant-aware = group fusion candidates per tenant (don't fuse across tenant boundary). |
| — | FUSION_KERNELS | `cipher_fusion_kernels.cpp` (275 lines) | working | fused kernel templates, substitute_v2 compile | Actually emits fused PTX/SASS via substitute_v2 | mid | low | Companion to FUSION; same per-tenant constraint. |
| — | ATTN_KOOPMAN | `cipher_attn_koopman.cpp` (583 lines) | working — partial (Koopman wires open) | attention kernel signatures, Koopman state | Attention-specific Koopman fusion (one TODO noted) | low | n/a | Speculative-decode budget hook lives here? P4.6 candidate; needs deeper read. **audit-incomplete:** Q: is the TODO at attn_koopman.cpp:583 a blocker for fusion or cosmetic? |
| — | SUBSTITUTE_V2 | `cipher_substitute_v2.cpp` (389 lines) | working (19 `return 0;` are legitimate enabled-checks) | NVRTC, kernel source string, substitute env enable | Runtime PTX compile + dispatch substitution | mid | mid | Per-tenant substitution policies (e.g., tenant A keeps cuBLAS, tenant B uses fused kernels). |
| — | FLOW_PATTERNS | `cipher_flow_patterns.cpp` | working v1 | dispatch ring, kernel-fingerprint chain | Detects stable repeating kernel sequences | low | low | Per-tenant flow recognition. |
| — | FLOW_RECORDER | `cipher_flow_recorder.cpp` | working v1 | dispatch ring → bounded record buffer | Records dispatch sequences for replay/analysis | low | low | Tenant-tagged flows enable per-tenant replay. |
| — | FLOW_SUBSTITUTE | `cipher_flow_substitute.cpp` (524 lines) | working v1 | flow recognition result, recipe table | Replaces recognized flows with optimized recipes | mid | mid | Per-tenant recipe inventory. |
| — | RECIPES | `cipher_recipes.cpp` (522 lines) | working | recipe table → flow_substitute | Recipe registry (Marlin INT4, FP8 paths, etc.) | mid | mid | Recipe selection per tenant SLA. |
| — | GRAPH | `cipher_graph.cpp` (293 lines) | working v1 | CUDA Graph capture state, kernel chain | CUDA Graph instantiation observer + decision | low | n/a | Per-tenant graph capture trigger. |
| — | GRAPH_INSPECT | `cipher_graph_inspect.cpp` (287 lines) | working | CUgraph node params, cuGraphInstantiate_v2 | Graph internal structure inspector | n/a | n/a | Diagnostic only — but per-tenant graph fingerprint potentially useful. |
| — | OVERLAP | in `cipher_layer2.cpp` (and `cipher_nccl_neural.cpp`) | working v1 | NCCL communicator, compute stream, overlap budget | Compute/comm overlap orchestration | mid | mid | Multi-tenant: don't overlap A's compute with B's comm (privacy). |
| — | OPB / OVERLAP_B | (within OVERLAP) | wired | — | (same) | (same) | (same) | rolled into OVERLAP row above |
| — | MEM_LAYOUT | `cipher_mem_layout.cpp` | working init-time | cudaDeviceProp, env CIPHER_MEM_LAYOUT | Memory layout recommendations (channel binding) | low | low | Per-tenant memory binding hints. |

**Class A row count: 35** (includes the OVERLAP roll-up note above; effective distinct symbols 34).

---

## Section B — Class B ops (tenant-invariant, keep as-is)

| # | Name | Source | State | Top-3 Inputs | Output Effect | MFU | TPW | Notes |
|---|---|---|---|---|---|---|---|---|
| 18 | RECEIPT | `cipher_receipt.cpp` | working v1 | session_fp, params_hash chain, env HMAC key | Per-session signed proof JSON | n/a | n/a | Per-session already; identity-bridge to tenant_id is cosmetic. Operates below tenant boundary semantically (per-session proof). |
| 21 | DETERMINISM | `cipher_determinism.cpp` | working v1 | dispatch sequence, params_hash | Reproducibility fingerprint | n/a | n/a | Identity invariant — fingerprint is the contract. |
| 29 | COMPLY | `cipher_comply.cpp` (82 lines) | working aggregator | RECEIPT/CARBON/GUARD/DETERMINISM/FAIRNESS/TOPOLOGY state | Compliance JSON bundle | n/a | n/a | Pure aggregator. Tenant-awareness flows from upstream ops. |
| — | EDMD | `cipher_edmd.cpp` (863 lines) | working | (input, output) op snapshots, env CIPHER_EDMD | EDMD snapshot collection for Koopman solver | n/a | n/a | Used by ADAPT; tenant-invariant (operates per op-class). |
| — | EDMD_LIVE | `cipher_edmd_live.cpp` (578 lines) | working | live EDMD state | Online Koopman update | n/a | n/a | Same as EDMD. |
| — | KOOPMAN_RUNTIME / `cipher_kr_init` | `cipher_koopman_runtime.cpp` (409 lines) | working | Koopman dictionary, EDMD output | Koopman matrix solve, weight swap into ADAPT | n/a | n/a | Substrate for ADAPT; per-op-class, not per-tenant. |
| — | LIQUID_STATE | `cipher_liquid_state.cu` (330 lines) | working | LNN state, SM idle fraction, env | Liquid-state manager: hw context for CIPHER's LNNs | n/a | n/a | Internal to CIPHER's NN, not workload-facing. |
| — | LNN | `cipher_lnn.cpp` (477 lines) | working | CfC weights, input vector | Liquid NN forward pass | n/a | n/a | CIPHER-internal NN; not workload-facing. |
| — | LAYER2 | `cipher_layer2.cpp` (85 lines) | working orchestrator | layer 3 output, AMD-mode flag | Layer 2 (orchestration LNN) wiring | n/a | n/a | NN substrate. |
| — | LAYER3 | in `cipher_dispatch.cpp` | working orchestrator | env, kernel class | Layer 3 (substitution LNN) wiring | n/a | n/a | NN substrate. |
| — | ORACLE | `cipher_oracle.cpp` (451 lines) | working | kernel signature hash, dispatch decision history | Oracle lookup for SUBSTITUTE / structure_lookup pre-stage | n/a | n/a | Per-shape, not per-tenant. |
| — | STRUCT_LOOKUP / `cipher_struct_lookup_init` | `cipher_oracle.cpp` + `cipher_structural_lookup.cpp` | working | kernel layout fingerprint, struct probe | Look up kernel by structural fingerprint | n/a | n/a | Identity invariant. |
| — | REGISTRY / `cipher_registry_init` | in `cipher_dispatch.cpp` + `cipher_recipes.cpp` | working | recipe id, kernel class | Registry of recipes/kernels | n/a | n/a | Pure registry. |
| — | INTERCEPT | `cipher_intercept.cpp` (445 lines) + `cipher_intercept_cudart.cpp` (3006 lines) | working | dlsym, GOT/PLT slots, env | The Phase 2 cuBLAS/cudart interceptor itself | n/a | n/a | Substrate. Per-tenant routing is decided downstream of the intercept call, not here. **INFRA-leaning but kept as B because it's the entrypoint surface.** |
| — | KERNEL_TABLE | `cipher_kernel_table.cpp` (317 lines) | working | dladdr on host stubs, function pointer | Kernel identification table | n/a | n/a | Per-shape, not per-tenant. |
| — | TELEMETRY | `cipher_telemetry.cpp` (410 lines) | working | NVML poll | NVML telemetry adapter (cipher_rt's own NVML poll, predates Phase 3 daemon) | n/a | n/a | **Note:** redundant with Phase 3's `cipher-gpustate` daemon. Consider cut in T4.0.6 if functionality fully overlaps. |
| — | SILICON | `cipher_silicon.cpp` (145 lines) | working | cudaDeviceProp + NVML init | Silicon model: sm_count, l2_persist_max, peak FP16, etc. | n/a | n/a | **INFRA, see Section E.** Listed here because it's referenced by many B-class ops; classified as INFRA. |
| — | HW_DESC | `cipher_hw_desc.cpp` | working init-time | cudaDeviceProp | Hardware-descriptor table | n/a | n/a | INFRA. |
| — | TOPOLOGY | (see Op 25 in Section A) | — | — | — | — | — | Single-GPU pod = effectively B; multi-GPU = A. Counted once in Section A. |
| — | POWER_CAP | `cipher_power_cap.cpp` (288 lines) | working v1 | NVML power-cap support, env CIPHER_POWER_CAP | Sets device-wide power cap | n/a | direct | **Operates below tenant** — power cap is one device, one number. Per-tenant power accounting comes from CARBON, not here. |
| — | PARAM_RECOVERY | `cipher_param_recovery.cpp` (668 lines) | working | params_hash, fallback param table | Recovers GEMM params when launch metadata is degraded | n/a | n/a | Per-kernel, not per-tenant. |
| — | VMM | `cipher_vmm.cpp` (327 lines) | working | env CIPHER_VMM, cuMemAddressReserve | Virtual memory manager pool | n/a | n/a | Per-process, below tenant. |
| — | MARLIN_SRC | `cipher_marlin_src.cpp` (773 lines) | working — recipe source | Marlin INT4 templates | Compiled-in Marlin INT4 kernel source for substitute_v2 | n/a | n/a | Source code, not a runtime op. Companion to WEIGHT_COMPRESS / RECIPES. |
| — | BLOCK_SUB / BLOCK_SUB_GPU | `cipher_edmd.cpp` + `cipher_block_sub_kernel.cu` | working | EDMD dict size, Chebyshev coefficients | Block-substitute kernel + Chebyshev auto-tuner | n/a | n/a | EDMD substrate; per-op-class. |
| — | CHEBYSHEV_AUTO | `cipher_block_sub_kernel.cu` | working | calibration data → chebyshev coefficients | Chebyshev polynomial auto-tune for block sub | n/a | n/a | Same. |

**Class B row count: 22.**

---

## Section C — Class C ops (cut from Phase 4)

| # | Name | Source | State | Inputs | Output | MFU | TPW | Notes |
|---|---|---|---|---|---|---|---|---|
| — | FP8_COMPUTE | `cipher_fp8_compute.cpp` (929 lines, 19 `return 0;`) | **stub-heavy** | NVRTC compile path, FP8 template | Compiles FP8 paths via substitute_v2; 19 early-return-0 in source | n/a | n/a | **audit-incomplete:** the 19 `return 0;` are mostly disabled-when-not-enabled guards but the volume suggests this op is mostly compile-machinery, no measurable lift today. Recommend cut unless a specific FP8 lift is measured. Sub-question: does it actually produce TFLOPS today? |

**Class C row count: 1** (conservative — most C candidates are kept as B with "no measurable lift" note rather than cut, pending T4.0.6 confirmation).

---

## Section D — Deferred ops (out of Phase 4 scope)

| # | Name | Source | State | Inputs | Output | MFU | TPW | Notes |
|---|---|---|---|---|---|---|---|---|
| — | NCCL | `cipher_nccl.cpp` (369 lines) | working — local only | NCCL communicator, ring/tree algo | NCCL orchestrator (single-node) | n/a | n/a | DEFERRED per MEMORY.md: "NCCL P2P CPU Proxy DEFERRED — requires libibverbs + multi-node, not buildable here." |
| — | NCCL_BPF | `cipher_nccl_bpf.cpp` (200 lines, 5 stubs) | partial | BPF program, NCCL hook | BPF-injected NCCL telemetry | n/a | n/a | DEFERRED — multi-node + libbpf-dev required. |
| — | NCCL_NEURAL | `cipher_nccl_neural.cpp` (341 lines) | working — neural-driven tuner | NCCL profile, NN tuner | NN-driven NCCL algo selection | n/a | n/a | DEFERRED — single-pod NCCL is no-op. |
| — | NCCL_V4 | `cipher_nccl_v4.cpp` (114 lines) | wired | NCCL tuner ABI | New NCCL tuner ABI adapter | n/a | n/a | DEFERRED — same reason. |

**Class D row count: 4.**

---

## Section E — INFRA (substrate; T4.1 may extend or read-only consume)

| Name | Source | State | Inputs | Output | Notes |
|---|---|---|---|---|---|
| `cipher_10ops_init` (orchestrator) | `cipher_10ops_impl.cpp` (1080 lines) | working | all 20+ external op state, ring buffer, env | Wires Stage 0 ring → Stage 1 shadow → Stage 2 background | **The runtime spine.** Sub-op breakdown below. |
| `cipher_silicon_init` | `cipher_silicon.cpp` | working | cudaDeviceProp, NVML | Read-only silicon model (sm=132, l2=50MB, peak=989TF, etc.) | INFRA. T4.1 must extend this with `cipher_tenant_snapshot` static fields. |
| `cipher_hw_desc_init` | `cipher_hw_desc.cpp` | working | cudaDeviceProp | HW description table | INFRA. |
| `cipher_lnn_init` | `cipher_lnn.cpp` | working | CfC weights | CIPHER-internal LNN runtime | INFRA. |
| `cipher_edmd_init` / `cipher_kr_init` / `cipher_liquid_state_init` | as named | working | NN substrate | EDMD pipeline + Koopman runtime + liquid state | INFRA. |
| `cipher_intercept_init` | `cipher_intercept.cpp` + `cipher_intercept_cudart.cpp` | working | dlsym targets, GOT/PLT, env | The Phase 2 hook surface | INFRA. T4.1 will extend with tenant-snapshot lookup at intercept time. |
| `cipher_kernel_table_init` | `cipher_kernel_table.cpp` | working | dladdr stubs | Kernel-ID table | INFRA. |
| `cipher_registry_init` | `cipher_dispatch.cpp` | working | recipe/op table | Op + recipe registry | INFRA. |
| `cipher_struct_lookup_init` | `cipher_oracle.cpp` + `cipher_structural_lookup.cpp` | working | kernel layout DB | Structural lookup table | INFRA. |
| `cipher_layer2_init` / `cipher_layer3_init` | `cipher_layer2.cpp` / `cipher_dispatch.cpp` | working | NN layer config | LNN layer wiring | INFRA. |

**INFRA row count: 10** (excluding ops also listed in Section A/B). These count toward the 64-symbol total but are not "ops" in the Phase 4 fusion sense — they're the substrate the ops sit on.

---

## Section F — cipher_10ops bundle: 12 sub-ops (Stage 0/1/2)

`cipher_10ops_init` orchestrates 12 distinct sub-ops. 10 inline in `cipher_10ops_impl.cpp`, 2 in Stage 2 thread. Each gets full 7-column treatment.

| # | Sub-op | Stage | Source location | State | Inputs | Output | Class | MFU | TPW | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CLASSIFY | 0 | `cipher_intercept_cudart.cpp` (within dispatch_and_log) | working | kernel sig, params_hash | kernel_class enum (0..N) | INFRA | n/a | n/a | Pre-dispatch classifier; runs every launch. |
| 2 | ORACLE | 0 | `cipher_oracle.cpp` | working | kernel_class, recipe table | Substitution candidate or null | B | n/a | n/a | Per-shape lookup; ~ns. |
| 3 | SUBSTITUTE | 0 | `cipher_substitute_v2.cpp` + `cipher_flow_substitute.cpp` | working | oracle result, env enables, recipe id | Replaced kernel launch (Marlin INT4 / FP8 / fused) | A | mid | mid | Per-tenant policy: which substitutions are allowed for which tenant. |
| 4 | COMMIT | 0 | `cipher_dispatch.cpp` | working | substitution decision, ring entry | Final kernel launch | INFRA | n/a | n/a | Dispatch finalizer. |
| 5 | SAMPLE | 0 | `cipher_intercept_cudart.cpp` (~line 446) | working | ring write completed, atomic seq | Ring entry timestamped | INFRA | n/a | n/a | The Stage-0 ring entry emit. |
| 6 | RING_WRITE | 0 | inline `cipher_10ops.h` `cipher_ring_write()` | working | ring buffer state, write seq | SPMC publication via `release` store | INFRA | n/a | n/a | The 10-ns critical-path primitive. |
| 7 | REMEMBER | 1 | `cipher_10ops_impl.cpp:446` | working | ring read, CfC hidden state | Updated hidden state in shadow CfC | B | n/a | n/a | LNN forward on shadow thread; per-op-class. |
| 8 | VALIDATE | 1 | `cipher_10ops_impl.cpp` | working | Welford stats per kernel class | 3-σ anomaly flag | B | n/a | n/a | Internal anomaly detection. |
| 9 | AUDIT | 1 | `cipher_10ops_impl.cpp:392` | working (when libssl-dev) | HMAC chain, ring entry params_hash | Tamper-evident SHA-256 hash chain | B | n/a | n/a | Identity invariant. **Build-broken without libssl-dev.** |
| 10 | SPECULATE | 1 | `cipher_10ops_impl.cpp:493` | working | REMEMBER hidden state, CfC | Predicted next kernel class → look-aside | B | n/a | n/a | <2ns SPECULATE check on Stage 0 hot path. |
| 11 | ADAPT | 2 | `cipher_10ops_impl.cpp:582` | working | EDMD snapshots per op-class, Koopman solve | LNN weight swap when Koopman converges | B | n/a | n/a | Stage 2 background; per-op-class. |
| 12 | ARBITRATE | 2 | `cipher_10ops_impl.cpp` | working | SHIELD band hints, SM_PACKER signal, GREEN_CTX state | SM partition rebalance via Green Context streams | **A** | mid | mid | **Phase 4 fusion natural fit.** Currently uses SHIELD's session_fp band; rebuild to consume `cipher_tenant_snapshot.sm_partition_mask`. |

**Sub-op classes: A=2 (SUBSTITUTE, ARBITRATE), B=6, INFRA=4.**

---

## Final counts

| Class | Count | Includes |
|---|---|---|
| **A** | **35** | (Section A's 34 distinct symbols + ARBITRATE sub-op + SUBSTITUTE sub-op — note ARBITRATE/SUBSTITUTE roll up but are listed here as the fusion-relevant Class-A pivot points) |
| **B** | **22** | Section B |
| **C** | **1** | FP8_COMPUTE (audit-incomplete, recommend cut pending T4.0.6 measurement) |
| **D** | **4** | All NCCL variants (multi-node deferred) |
| **INFRA** | **10** | Section E |
| **audit-incomplete** | **2** | FP8_COMPUTE state; ATTN_KOOPMAN TODO |

**Total init symbols audited:** 64. Plus 12 sub-ops of cipher_10ops. **Total rows: 76.**

(Section A claims 35; the underlying init-symbol count for Class A in Sections A is 34. The +1 is ARBITRATE which is a sub-op inside `cipher_10ops_init`, not a separate exported symbol. SUBSTITUTE is also a sub-op listed under cipher_10ops; the corresponding init symbol is `cipher_substitute_v2_init` which is counted in Section A's distinct symbols, but functionally SUBSTITUTE is multiple ops (`substitute_v2`, `flow_substitute`, `flow_patterns`, `recipes`) that together implement the Stage 0 op. Counted as one Class-A pivot.)

## Aggregate lift projection (if all Class A ops fuse cleanly)

This is projection, not measurement. T4.0.5 measurement harness + T4.2-T4.6 implementations replace these.

**MFU lift toward G1 (15-25% per tenant from 5% baseline):**
- P4.2 PARTITION_ROUTER + SM_PACKER + ARBITRATE per-tenant SM partition: +8-12%
- P4.4 PREDICT + PERSIST_ENGINE + L2_PERSIST per-tenant L2 pin: +5-10%
- P4.5 TMA via SUBSTITUTE/FUSION per-tenant fusion: +2-5%
- P4.6 KV_COMPRESS + KV_REDIRECT per-tenant decode budget: +1-3%
- **Aggregate: +16-30% MFU** → baseline 5% → 21-35% per tenant. G1 (15-25%) achievable.

**TPW lift toward G2 (5-7× over baseline, current 2.96×):**
- P4.3 VOLT + HIBERNATE + THERMAL_FEEDBACK per-tenant DVFS bucketing: ×1.4-1.8 (bare-metal pod required)
- P4.4 L2 pin reduces wasted DRAM bandwidth: ×1.1-1.2
- P4.2 partition routing reduces cross-tenant cache thrash: ×1.1-1.2
- CARBON gives Prometheus attribution (no direct TPW lift)
- **Aggregate: 2.96× × 1.4-1.8 × 1.1-1.2 × 1.1-1.2 ≈ 5.0-7.7× TPW.** G2 (5-7×) on the dot.

**Caveats:**
- VOLT and HIBERNATE return DEGRADED on Lambda's containerized pod (NVML clock-set returns NOT_SUPPORTED). G2 measurement requires either a bare-metal pod or fallback paths.
- WEIGHT_SHARE (cross-tenant dedup) is a 5× memory-density multiplier, not a TPW multiplier. It enables G3 (30+ tenants) by reducing per-tenant memory footprint.

## cipher_10ops bundle: 12 sub-ops breakdown — done above in Section F

## cipher_weight_share semantics — CONFIRMED real cross-tenant

- 541 lines of CUDA IPC + /dev/shm code (not a stub)
- Protocol: content-hash-keyed slots, `cuIpcGetMemHandle` / `cuIpcOpenMemHandle`
- Process A publishes weight; Process B picks up if content matches
- Per-slot refcount via flock
- **Phase 5 critical-path primitive confirmed.** Not currently tenant-aware (keyed on content hash, not tenant ID) — but tenant accounting can be added on top without touching the IPC protocol.

## cipher_fairness tenant logic — CONFIRMED, but uses different identity convention

- `cipher_fairness.cpp` (single-process): keyed on `session_fp` (64-bit hash from SENSE), MAX_TENANTS=256 slots, open-address atomic table
- `cipher_fairness_shm.cpp` (cross-process): POSIX shm `/cipher_fairness`, MAX_TENANTS=64 slots, explicit `tenant_id` field as `uint32_t`
- **Phase 4 identity-bridge work required:** Phase 3 uses `char tenant_id[64]` string; FAIRNESS uses 32-bit and 64-bit numeric. Three options:
  1. SENSE consumes Phase 3 tenant_id string and produces session_fp = FNV-64(tenant_id). Fairness already correct.
  2. Extend Phase 3 ABI with a numeric tenant_handle alongside the string.
  3. Refactor FAIRNESS to consume tenant_id string directly.
  Recommend option 1 (least invasive, preserves both ABIs).

## Source-file caveats encountered

| Symbol | Issue | Resolution |
|---|---|---|
| `cipher_block_sub_gpu_init` | No `src/cipher_block_sub_gpu.{cpp,cu}` | Lives in `src/cipher_edmd.cpp` + `cipher_block_sub_kernel.cu` (naming mismatch) |
| `cipher_block_sub_init` | Same | Same |
| `cipher_chebyshev_auto_init` | No `src/cipher_chebyshev_auto.{cpp,cu}` | Lives in `src/cipher_block_sub_kernel.cu` |
| `cipher_kr_init` | No `src/cipher_kr.cpp` | Lives in `src/cipher_koopman_runtime.cpp` (abbreviation: kr = koopman_runtime) |
| `cipher_layer3_init` | No `src/cipher_layer3.cpp` | Lives in `src/cipher_dispatch.cpp` |
| `cipher_overlap_init` | No `src/cipher_overlap.cpp` | Lives in `src/cipher_nccl_neural.cpp` + `src/cipher_layer2.cpp` |
| `cipher_registry_init` | No `src/cipher_registry.cpp` | Lives in `src/cipher_dispatch.cpp` + `src/cipher_recipes.cpp` |
| `cipher_struct_lookup_init` | No `src/cipher_struct_lookup.cpp` | Lives in `src/cipher_oracle.cpp` + `src/cipher_structural_lookup.cpp` (naming mismatch) |

**None of the 8 "missing" symbols are genuinely missing.** All resolve to existing source under a different filename or via inline definitions. Audit fully covers all 64.

## audit-incomplete rows requiring follow-up

| Op | Question | Estimated time to resolve |
|---|---|---|
| FP8_COMPUTE | Are the 19 `return 0;` legitimate enabled-checks, or genuine stubs? Does this op produce TFLOPS today on FP8 paths? | 30 min, requires reading full 929 lines + running an FP8 benchmark |
| ATTN_KOOPMAN | The `// Would run fused kernel here (TODO: wire up once...)` at line ~583: blocker or cosmetic? Does the attention path currently dispatch the fused Koopman or the cuBLAS attention? | 20 min, requires reading the dispatch path |

Both can be deferred to T4.8 (Stage 2 overlay ops promotion) without blocking T4.0.2–T4.0.7.

## Build state

**Build broken.** `make clean && make` fails on `cipher_10ops_impl.cpp:71` — `openssl/hmac.h` not found. `libssl-dev` not installed (`dpkg -l libssl-dev` returns nothing). All "working" claims sourced from binary (`libcipher_rt.so.preroadmap` md5 `95284eb1...`) + source structure + BUILD_STATE.md. Cross-validation requires libssl-dev install in T4.0.6 (decided not done in audit).

**Phase 3 substrate intact (post-audit):**
- Fallback `cipher_kmod.ko.v0.2.0` md5 `55ab8c0cd8309ca7cc0fc40fe556aa19` ✓
- Fallback `libcipher_v2.so.v0.2.0` md5 `86618c30896470b642fcc6985d8dc632` ✓
- cipher_kmod 0.3.1 loaded, srcversion `B1AF5E2A...` ✓
- Taint 12288 stable ✓
- /proc/cipher/{stats,bar0_state,gpu_state} all readable ✓

## Final report — the requested specifics

- **Total symbols audited:** 64 init symbols + 12 cipher_10ops sub-ops = **76 rows**
- **Class A (rebuild for fusion):** **35** rows. Comfortably in your "Class A 12+ → more fusion surface than expected. P4 sequence needs more sub-phases" zone.
- **Class B (tenant-invariant, keep):** **22**
- **Class C (cut):** **1** (FP8_COMPUTE, audit-incomplete)
- **Class D (deferred — NCCL multi-node):** **4**
- **INFRA:** **10**
- **audit-incomplete:** **2** (FP8_COMPUTE, ATTN_KOOPMAN TODO)
- **cipher_10ops bundle:** 12 sub-ops detailed in Section F. Of these, **2 are Class A** (SUBSTITUTE, ARBITRATE); the rest are B/INFRA.
- **cipher_weight_share semantics:** CONFIRMED real cross-process / cross-tenant CUDA IPC. Phase 5 substrate; not tenant-keyed today but tenant accounting easy to layer.
- **cipher_fairness tenant logic:** CONFIRMED already tenant-aware (open-address `session_fp` table) + cross-process FAIRNESS_SHM with explicit `tenant_id` field. Identity bridge to Phase 3's tenant_id string is the Phase 4 work — see recommendations above.
- **Aggregate MFU lift if all Class A fuse cleanly:** **+16-30%** (baseline 5% → 21-35% per tenant); G1 (15-25%) achievable.
- **Aggregate TPW lift if all Class A fuse cleanly:** **5.0-7.7×** over baseline (current 2.96×); G2 (5-7×) achievable, requires bare-metal pod for VOLT/HIBERNATE to deliver.
- **Source-file caveats:** 8 init symbols had naming mismatches; all resolved, no genuine missing source.
- **Build state:** broken (libssl-dev missing). Documented. T4.0.6 decides install/cut.

## Blockers / surprises during audit

1. **Class A=35 is significantly larger than "5-7 fusion actuators"** the Phase 4 spec envisioned. The original P4.2-P4.6 sequence (5 sub-phases) cannot cover all 35 in any reasonable schedule. **Architecture-impact flag:** the P4 sub-phase plan should be revisited at T4.0.7 — recommend grouping the 35 into ~8 thematic clusters (SM-partition cluster, DVFS cluster, L2 cluster, KV cluster, weight-share cluster, observe-only-to-tenant cluster, agentic cluster, compliance cluster) and assigning each cluster to a P4.x sub-phase. This may extend Phase 4 by 1-2 sub-phases beyond the original 4.7.
2. **Identity-bridge gap.** Phase 3 uses `char tenant_id[64]` string; cipher_rt uses `session_fp` (64-bit) and `tenant_id_u32` (32-bit). T4.0.2's `cipher_tenant_snapshot` contract MUST resolve this — recommend the snapshot struct hold all three (string, session_fp, u32 handle) so each subsystem reads what it needs.
3. **WEIGHT_SHARE is more valuable than the Phase 4 spec credits.** It's the Phase 5 density multiplier already implemented. Phase 4 should ensure it isn't broken by P4.2-P4.6 fusion work.
4. **cipher_telemetry redundancy with cipher-gpustate.** `cipher_telemetry.cpp` (410 lines) is cipher_rt's own NVML poll, predating Phase 3's daemon. Functionality overlaps. Cut or keep is a T4.0.6 question.
5. **POWER_CAP (Section B) is a single-device knob.** Per-tenant power cap is structurally impossible on H100 (no per-partition power limit). Per-tenant power *attribution* via CARBON is what we have.
6. **GREEN_CTX is solid Phase 5 substrate, not stub.** Real `cuGreenCtxCreate` implementation. Phase 4 can build on it.

## Holding for review

Per [[cipher-phase-discipline]]: not proceeding to T4.0.2 (kernel-internal contract design) without explicit approval. The Class-A count of 35 in particular warrants architectural re-discussion before fixing the contract spec.

Audit document: `/home/ubuntu/PHASE_4_OP_AUDIT.md` — ~12 KB markdown.
Source-of-truth data: `/tmp/cipher_audit_headers.txt` (2122 lines), `/tmp/cipher_audit_stubs.txt` (73 lines).
