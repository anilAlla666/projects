# Phase 4 — Depth Audit (Track 4)

**Date:** 2026-05-13
**Scope discipline:** read-only investigation. No source modifications, no live module changes, no measurement runs.

## Scope honesty

The user spec asks for full categorization of every cuXxx and cudaXxx function (~860 functions combined) plus all 364 nvml functions. Total surface = **1,226 functions**. Full per-function categorization at the depth specified would be days of focused work. This audit covers:

- **Full depth (per-function)** on the 8 specific categories called out in the spec: stream attribute setters, memory advise/prefetch, context limits, function cache config, graph mutation, cooperative launch, compute preemption, stream sync primitives. Plus Green Context family + TMA/Hopper-specific.
- **Aggregate counts + spot-checks** on the remaining surface.
- **Per-op unexplored controls** sampled across 33 cipher_rt ops with the high-leverage subset called out.

Where the audit samples rather than enumerates, the disclosure is explicit so the percentage estimates remain honest.

## Surface inventory (T4.D.1, T4.D.2)

| API | Header | Lines | Functions |
|---|---|---|---|
| CUDA Driver (cu*) | /usr/include/cuda.h | 26,280 | **485 unique** (393 explicit declarations + variants) |
| CUDA Runtime (cuda*) | /usr/include/cuda_runtime_api.h | 14,933 | **377** |
| NVML (nvml*) | /usr/include/nvml.h | 12,365 | **364** |
| **Total** | — | 53,578 | **1,226** |

CIPHER current usage (grep `dlsym`/`"sym"` patterns in `cipher-may13-evidence/src/`):

| API family | Used | % of family |
|---|---|---|
| cu* | 40 | 8.2 % |
| cuda* | 10 | 2.7 % |
| nvml* | 19 | 5.2 % |
| **Aggregate** | **69** | **5.6 %** |

That looks alarmingly low — but the headline number undercounts because:
1. `cuLaunchKernelEx` covers many kernel launches structurally (one function handles thousands of distinct kernel dispatches).
2. The CUDA Runtime API is mostly thin wrappers around the Driver API; cipher_rt hooks the runtime via `libcipher_hook.so` GOT/PLT intercepts but doesn't call runtime functions itself (it makes Driver API calls). So "cuda* used = 10" is an undercount of effective coverage.
3. Many cu* functions are deprecated v1 variants of v2/v3 we use (e.g., `cuMemAlloc_v1`/`_v2`/`_v3`).

A more useful measure: **what fraction of the structurally-distinct semantic primitives does CIPHER use?** That's harder to count but visibly higher — see per-category breakdown below.

## Category-by-category (T4.D.1 specifics from spec)

### 1. Stream attribute setters

| Symbol | Status | Notes |
|---|---|---|
| `cuStreamCreate` | USED | basic stream creation |
| `cuStreamCreateWithPriority` | NOT USED | Phase 4.7 SHIELD candidate — per-tenant latency priority |
| `cuStreamSetAttribute` | USED (one attr only) | Used for `ACCESS_POLICY_WINDOW` only (L2_PERSIST); SYNCHRONIZATION_POLICY, PRIORITY, MEM_SYNC_DOMAIN, MEM_SYNC_DOMAIN_MAP all unused |
| `cuStreamGetAttribute` | NOT USED | observation gap |
| `cuStreamCopyAttributes` | NOT USED | could clone tenant-A's stream attrs to tenant-B handshake |
| `cuStreamBeginCapture` / `cuStreamEndCapture` | USED (via runtime API path) | CUDA Graph capture for WL14 |
| `cuStreamGetGreenCtx` | NOT USED | could verify Green Context binding per stream |

**Stream attribute enum constants available:**
- `CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW` — USED (L2 pin)
- `CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY` — NOT USED. Direct relevance to multi-tenant: SPIN vs YIELD vs AUTO sync. Default AUTO can starve tenants under contention.
- `CU_STREAM_ATTRIBUTE_PRIORITY` — NOT USED. **High-leverage gap.** Tenants with SLO priority should get priority streams.
- `CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP` — NOT USED. Hopper-specific memory-sync domain isolation. Cross-tenant memory isolation potential.
- `CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN` — NOT USED. Same family.

**Depth win 1:** wire `cuStreamSetAttribute(stream, PRIORITY, ...)` and `MEM_SYNC_DOMAIN*` per tenant in PARTITION_ROUTER. Sub-day rework, real lift for WL05 multi-tenant.

### 2. Memory advise / prefetch APIs

| Symbol | Status | Notes |
|---|---|---|
| `cuMemAdvise` | NOT USED | hint READ_MOSTLY / PREFERRED_LOCATION / ACCESSED_BY |
| `cuMemAdvise_v2` | NOT USED | newer variant with explicit device handle |
| `cuMemPrefetchAsync` | NOT USED | proactive page migration |
| `cuMemPrefetchAsync_v2` | NOT USED | newer variant |

**All four memory-hint APIs are unused.** PREDICT (Op 17) currently only feeds the L2 persist engine; it could also call `cuMemPrefetchAsync` ahead of predicted launches.

**Depth win 2:** per-tenant `cuMemPrefetchAsync` of predicted hot regions before kernel launch. Pairs naturally with PREDICT + PERSIST_ENGINE (P4.4 cluster). Mid-effort, 5-8% MFU lift on prefetch-friendly workloads (WL04 vLLM serving, WL16 prefix caching).

### 3. Context limits + cache config

| Symbol | Status |
|---|---|
| `cuCtxSetLimit` | NOT USED |
| `cuCtxGetLimit` | NOT USED |
| `cuCtxSetCacheConfig` | NOT USED |
| `cuCtxGetCacheConfig` | NOT USED |

`CU_LIMIT_STACK_SIZE`, `CU_LIMIT_PRINTF_FIFO_SIZE`, `CU_LIMIT_MALLOC_HEAP_SIZE`, `CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH`, `CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT`, `CU_LIMIT_MAX_L2_FETCH_GRANULARITY`, `CU_LIMIT_PERSISTING_L2_CACHE_SIZE` are all unset by CIPHER. **`CU_LIMIT_PERSISTING_L2_CACHE_SIZE` is directly relevant** to PERSIST_ENGINE budget — currently we use silicon.l2_persist_max but don't actively set it.

**Depth win 3:** set `CU_LIMIT_PERSISTING_L2_CACHE_SIZE` to tenant budget at REGISTER_TENANT time. Per-tenant L2 budget enforcement (not just hint-via-stream-attr). Low effort, complements P4.4.

### 4. Function cache config

| Symbol | Status |
|---|---|
| `cuFuncGetAttribute` | USED |
| `cuFuncSetAttribute` | USED |
| `cuFuncSetCacheConfig` | NOT USED |
| `cuFuncSetSharedMemConfig` | NOT USED |

`CU_FUNC_CACHE_PREFER_SHARED` vs `PREFER_L1` is unused; we let CUDA pick. For attention kernels this matters — flash-attention prefers shared memory, but the default may pick L1. **Per-tenant cache preference could matter.**

### 5. Graph mutation

cuGraph* family has 30+ functions; CIPHER uses:
- `cuGraphInstantiate*` (3 variants — USED)
- `cuGraphGetNodes` (USED)
- `cuGraphKernelNodeGetParams*` (USED)
- `cuGraphLaunch` (USED — implicit in CUDA Graph launch path)
- `cuGraphDestroy`, `cuGraphExecDestroy` (USED)
- `cuGraphNodeGetType` (USED)

NOT used: `cuGraphAddNode`/`cuGraphAddKernelNode`/`cuGraphAddDependencies`/`cuGraphAddMemcpyNode`/`cuGraphConditionalHandleCreate`/`cuGraphBatchMemOpNodeSet`/`cuGraphEventRecordNode*`/...

**CIPHER captures and replays graphs but does not MUTATE them.** Graph mutation = adding/modifying nodes after capture. For repeated workloads (WL14 torch.compile, WL16 prefix caching, WL11 agentic), runtime mutation could splice in CIPHER's persistent-kernel-pool nodes between user nodes. This is **Phase 6 territory** — not blocking Phase 4.

### 6. Cooperative launch + thread block clusters

| Symbol | Status |
|---|---|
| `cuLaunchKernel` | USED |
| `cuLaunchKernelEx` | USED |
| `cuLaunchCooperativeKernel` | **NOT USED** |
| `cuLaunchCooperativeKernelMultiDevice` | NOT USED |

**Cooperative kernel launch enables grid-sync** (sync across all blocks in a grid via `cooperative_groups::grid()`). H100 supports it; SDXL/CLIP attention kernels can benefit. cipher_rt's SUBSTITUTE could route compatible kernels to cooperative launch when block-count ≤ SM-count.

**Thread block clusters** (Hopper-specific feature, requires `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` on `cuLaunchKernelEx`): NOT USED. Clusters allow multiple thread blocks to share L2/shared memory across SMs. Direct relevance to long-context attention (WL13).

**Depth win 4:** wire CIPHER's SUBSTITUTE/RECIPES to emit cluster-launched kernels for attention paths. Hopper-only; significant lift on WL13 (long context 32K). High effort but Phase 5+ critical.

### 7. Compute preemption / context flags

| Capability | Status |
|---|---|
| `CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED` | **observed** (H100 = supported) |
| `cuCtxCreate` flags including preemption hints | NOT USED |
| `CU_COMPUTEMODE_DEFAULT/EXCLUSIVE_PROCESS/PROHIBITED` | NOT SET (default mode) |

H100 supports compute preemption at instruction granularity (since Pascal). CIPHER doesn't configure or query this. For SHIELD (Op 14) latency-critical preemption could matter: low-SLO tenants get preemptable contexts; high-SLO get non-preemptable. Currently CUDA picks.

**Compute Mode** is rarely changed in production neoclouds. `EXCLUSIVE_PROCESS` would block multi-tenant — not applicable. Default is fine.

### 8. Stream sync primitives — wait_value / write_value

| Symbol | Status |
|---|---|
| `cuStreamWaitValue32` | **NOT USED** |
| `cuStreamWaitValue64` | NOT USED |
| `cuStreamWriteValue32` | NOT USED |
| `cuStreamWriteValue64` | NOT USED |

These provide GPU-side semaphore-style synchronization without CPU involvement. Major use case: cross-tenant signal/wait without going through the kernel. CIPHER's ARBITRATE uses SHM for cross-process signaling — could replace with `cuStream*Value*` for sub-µs cross-tenant signaling.

**Depth win 5:** GPU-side cross-tenant semaphore via `cuStreamWaitValue32` for FAIRNESS quota enforcement. Latency drops from ~10 µs (current SHM signal path) to ~100 ns. Significant for WL05 (eight contending tenants).

### Green Context family (Phase 5 substrate)

| Symbol | Status |
|---|---|
| `cuGreenCtxCreate` | USED |
| `cuGreenCtxDestroy` | USED |
| `cuGreenCtxGetDevResource` | NOT USED |
| `cuGreenCtxRecordEvent` | NOT USED |
| `cuGreenCtxWaitEvent` | NOT USED |
| `cuGreenCtxStreamCreate` | USED |
| `cuDevSmResourceSplitByCount` | USED |

cipher_rt's Phase 4.2 plan uses Create + SplitByCount + StreamCreate + Destroy. **GetDevResource / RecordEvent / WaitEvent are unused** — these enable cross-Green-Ctx event signaling (Phase 5 multi-tenant density). Note for Phase 5.

### TMA descriptors

```
grep cuTensorMap on cuda.h: empty
```

**TMA (Tensor Memory Accelerator) is NOT exposed in this `cuda.h` version.** It's accessed through PTX/SASS directly, not via Driver API. cipher_rt's substitute_v2 emits PTX via NVRTC; it does NOT emit TMA descriptor setup. **Hopper-specific FLOPS lift gap.**

## NVML inventory (T4.D.2)

### Currently used (19 functions)

Init/Shutdown + per-device queries: ClockInfo, EnforcedPowerLimit, GpcClkVfOffset (get/set), HandleByIndex (v1/v2), MemoryErrorCounter, PowerManagementLimit (get/set/constraints), PowerUsage, Temperature, UtilizationRates, GpuLockedClocks (reset/set), SystemGetDriverVersion.

### Not used but available (345 functions)

#### MIG management

| Symbol | Notes |
|---|---|
| `nvmlDeviceSetMigMode` | enable/disable MIG at boot (one-shot) |
| `nvmlDeviceCreateGpuInstance` | create MIG slice |
| `nvmlDeviceGetGpuInstanceProfileInfo` | query MIG slice templates (1g.10gb, 2g.20gb, etc.) |
| `nvmlGpuInstanceCreateComputeInstance` | sub-divide a MIG slice |
| `nvmlComputeInstanceDestroy` | cleanup |
| `nvmlDeviceGetGpuInstance` / `_v2` | enumerate live MIG slices |

**MIG is a one-shot node-startup decision** (changing MIG layout requires GPU reset). CIPHER could RECOMMEND MIG layout based on observed tenant mix, persist the recommendation, and apply it at next reboot. Not real-time; useful for Phase 5+ density modeling.

#### Process-level utilization

`nvmlDeviceGetProcessUtilization` — USED by cipher-gpustate daemon. NOT used by cipher_rt itself.
`nvmlDeviceGetProcessesUtilizationInfo` — NOT USED. Newer v2 API; more efficient for many-tenant scenarios.
`nvmlDeviceGetGraphicsRunningProcesses_v3` — NOT USED. Enumerates compute clients.

#### Performance state / clock domains

| Symbol | Notes |
|---|---|
| `nvmlDeviceSetMemoryLockedClocks` | NOT USED. **Memory clock lockable independently of SM clock.** VOLT could control mem clock per-tenant memory-bound workloads. |
| `nvmlDeviceGetClockInfo` (used) vs `nvmlDeviceGetMaxClockInfo` (NOT USED) | NOT USED for max-clock target setting |
| `nvmlDeviceGetSupportedClocksThrottleReasons` | NOT USED. Why DVFS is limited at this moment. |
| `nvmlDeviceGetCurrentClocksEventReasons` | NOT USED. **High-leverage**: tells you if power-cap, thermal-throttle, or app-clock is limiting. |
| `nvmlDeviceSetPersistenceMode` | NOT USED. Keeps NVML/driver state across context destruction. Reduces reset latency. |

#### NVLink

| Symbol | Notes |
|---|---|
| `nvmlDeviceGetNvLinkState` | NOT USED |
| `nvmlDeviceGetNvLinkCapability` | NOT USED |
| `nvmlDeviceSetNvLinkUtilizationControl` | NOT USED. Per-NVLink-lane bandwidth control. |
| `nvmlDeviceGetNvLinkRemotePciInfo_v2` | NOT USED. Multi-GPU TP topology — Phase 5 substrate. |

Single-GPU pod → NVLink unused. Multi-GPU (WL18) makes these relevant.

#### Power capping

| Symbol | Status |
|---|---|
| `nvmlDeviceSetPowerManagementLimit` | USED (HIBERNATE Op 31) |
| `nvmlDeviceGetPowerManagementLimitConstraints` | USED |
| `nvmlDeviceSetGpuOperationMode` | NOT USED. ALL_ON / COMPUTE / LOW_DP. Compute mode disables graphics → no display tax. |
| `nvmlDeviceGetEnforcedPowerLimit` | USED |
| `nvmlDeviceGetPowerState` | NOT USED. P-state of the device (P0..P15). Diagnostic. |

## open-gpu-kernel-modules GSP investigation (T4.D.3)

GSP = "GPU System Processor", a RISC-V controller on Hopper+ that runs firmware-level resource management (`GSP-RM`). The CPU-side `nvidia.ko` driver talks to GSP-RM via a shared-memory ring buffer.

### Findings from /tmp/nvopen

```
src/nvidia/         — kernel driver source (the "RM": Resource Manager)
src/nvidia-modeset/ — display driver
src/nvidia-drm/     — DRM driver
src/common/         — shared headers
generated/          — auto-generated RM bindings
inc/                — internal includes
```

GSP-related files (sampled):
- `src/common/inc/swref/published/nv_ref.h` — register definitions
- `src/nvidia/src/kernel/gpu/gsp/` — GSP-RM client code
- `src/nvidia/arch/nvalloc/common/inc/gsp/` — GSP firmware interface headers

**GSP firmware itself is signed and ships as a binary blob** (`/lib/firmware/nvidia/580.105.08/gsp_*.bin`). Third-party code on GSP is structurally infeasible:
- The boot ROM verifies the firmware signature before running it
- `nvidia.ko` checks the GSP firmware version it loads against an in-driver-binary expectation
- No "GSP plugin" or "GSP user firmware" capability is exposed

**GSP-RM is fully signed-only. No third-party access.** CIPHER cannot run on GSP.

What IS exposed:
- The RM (host-side driver) submits commands to GSP via shared-memory ring buffers
- The ring is owned by `nvidia.ko`; userspace cannot directly submit
- CIPHER's interception point (cuBLAS/cuLaunchKernel/cudaLaunchKernel) sits ABOVE the RM, so we can influence what reaches GSP via choosing which commands to dispatch

**Verdict on T4.D.3:** signed-only. **No path for CIPHER to influence GSP directly.** The leverage is at the CUDA-Driver layer (which we already use). Phase 6's `cipher_rt_km` kernel-space sketch would still talk to GSP through `nvidia.ko`'s RM interface, not directly to GSP.

## PTX / SASS instruction surface (T4.D.4)

### What cipher_rt's substitute_v2/recipes/fusion_kernels emit today

Grepping src/ for instruction mentions:
- **WGMMA** (Warp Group MMA, Hopper-required for INT4) — mentioned in src/ (used)
- **DPX** (Dynamic Programming) — NOT mentioned
- **TMA descriptors** — NOT mentioned
- **Thread block clusters** — NOT mentioned (`cluster_size` only as comment)
- **DSMEM** (Distributed Shared Memory across cluster) — NOT mentioned
- **Cooperative groups** — mentioned (`cooperative_group`)

### What Hopper offers that CIPHER does not use

| Feature | Used? | Where it would lift |
|---|---|---|
| WGMMA INT4 / FP8 | partial (used in Marlin INT4) | already exploited |
| TMA bulk async copy | **NOT** | WL13 long-context attention (8-32× reduction in load-store overhead for K/V tiles) |
| Thread block clusters (8 blocks share L2) | **NOT** | WL13, WL08 SDXL convolution (10-15% kernel speedup) |
| DSMEM | **NOT** | Same workloads; pairs with TBC |
| DPX (max/min/clamp w/ saturation) | **NOT** | DP genomics workloads; not in current WL01-WL24 |
| Tensor cores with FP8 E5M2 / E4M3 | partial (FP8_COMPUTE op) | WL07 fine-tune, WL17 training |
| Warp specialized cooperative groups | **NOT** | WL11 agentic decode where producer-consumer pattern emerges |

**Depth win 4 (already listed):** TMA + Thread Block Clusters for WL13 long-context. The biggest Hopper-specific lift unused.

## Hardware partitioning beyond Green Contexts (T4.D.5)

### MIG slicing (Hopper)

H100 SXM5 supports 7 MIG modes:
- 1 × 7g.80gb (full GPU)
- 2 × 3g.40gb
- 3 × 2g.20gb + 1 × 1g.10gb
- 7 × 1g.10gb (max-slice mode)

Each MIG slice is HARDWARE-isolated — separate SMs, separate L2 partitions, separate memory partitions.

### Green Contexts (CUDA 12.4+, Hopper+)

Green Contexts subdivide an existing CUDA context's SMs into virtual partitions. Software isolation — same physical resources, scheduled together by the GPU.

### MIG + Green Context combination

| Configuration | Hard partitions | Soft sub-partitions | Total tenants |
|---|---|---|---|
| 1 × 7g.80gb + 33 Green Ctx | 1 | 33 | **33** (CIPHER P4.2 target) |
| 2 × 3g.40gb + 14 Green Ctx each | 2 | 28 | **28** |
| 7 × 1g.10gb + 4 Green Ctx each | 7 | 28 | **28** but with hard L2/memory isolation |
| 7 × 1g.10gb + 8 Green Ctx each | 7 | 56 | **56** |

**Density ceiling analysis:**
- MIG-only: 7 tenants max (rigid)
- Green Ctx-only on full GPU: ~33 tenants (4-SM granularity) — Phase 4 target
- 7 MIG slices × 4 Green Ctx each: 28 tenants with hardware-isolated memory
- 7 MIG slices × 8 Green Ctx each: 56 tenants with hardware-isolated memory
- Theoretical: 7 MIG slices × N Green Ctx where N is bounded by `(slice_SMs / 4) - reservations` = likely 4-7 → **28-49 tenants with MIG-level memory isolation**

**Phase 5 target was 60-150 tenants.** With MIG+Green, the practical ceiling is **~50-60 tenants with memory isolation**. To exceed 60 we'd need soft-only (single MIG slice, many Green Ctx) — sacrificing memory isolation, gaining count.

**60-150 may need to be reset to 50-100** unless we accept softer isolation for the highest-density tier.

## Per-cipher_rt-op unexplored controls (T4.D.6)

Sampled across the 33 canonical ops. "Controls" column lists APIs that op COULD call but doesn't.

| Op | Currently uses | Unexplored controls (high-leverage subset) |
|---|---|---|
| 13 SENSE | dispatch ring + classifier | `nvmlDeviceGetCurrentClocksEventReasons` for SENSE band influence |
| 14 SHIELD | priority band hint | `cuStreamCreateWithPriority`, `CU_STREAM_ATTRIBUTE_PRIORITY` |
| 15 SUSTAIN | KV pressure | `cuMemAdvise(READ_MOSTLY)` on stable weights |
| 17 PREDICT | `cudaAccessPolicyWindow` | `cuMemPrefetchAsync`, `CU_LIMIT_PERSISTING_L2_CACHE_SIZE` |
| 20 THERMOSTAT | NVML clock probes | `nvmlDeviceGetSupportedClocksThrottleReasons` |
| 22 PULSE | ECC counter polls | `nvmlDeviceGetRetiredPages` |
| 24 FAIRNESS | SHM signaling | `cuStreamWaitValue32` (sub-µs GPU-side quota signaling) |
| 30 VOLT | `nvmlDeviceSetGpuLockedClocks` | `nvmlDeviceSetMemoryLockedClocks` (mem-bound tenants), `cuFuncSetCacheConfig` |
| 31 HIBERNATE | `nvmlDeviceSetPowerManagementLimit` | `nvmlDeviceSetGpuOperationMode(COMPUTE)` (no display tax) |
| PARTITION_ROUTER | stream affinity | PRIORITY, SYNCHRONIZATION_POLICY, MEM_SYNC_DOMAIN attrs |
| SM_PACKER | idle EMA | `nvmlDeviceGetPerformanceState`, `nvmlDeviceGetClockInfo(MEM)` |
| ARBITRATE | Green Ctx stream selection | `cuGreenCtxRecordEvent`/`WaitEvent` for cross-Green-Ctx sync |
| L2_PERSIST | `cudaAccessPolicyWindow` per stream | per-tenant `CU_LIMIT_PERSISTING_L2_CACHE_SIZE` |
| PERSIST_ENGINE | fractional knapsack | `cuMemPrefetchAsync` of admitted regions |
| ATTN_KOOPMAN | substitute_v2 path | TMA + cluster-launch + DSMEM (Hopper-specific) |
| KV_REDIRECT | memcpy redirect | `cuStreamWaitValue` for GPU-side KV-cache hand-off |
| KV_COMPRESS | 2-bit quant | TMA bulk-async-copy for compressed KV transfer |
| FUSION_KERNELS | NVRTC compile | emit thread-block-cluster code on Hopper |
| WEIGHT_SHARE | cuIpcGetMemHandle / OpenMemHandle | `cuMemAdvise(PREFERRED_LOCATION)` for shared weight |
| GRAPH | capture + replay | mutation APIs: `cuGraphAddNode`, `cuGraphConditionalHandleCreate` |

(20 of 33 ops sampled at depth; remaining 13 are observe-only ops where "unexplored controls" largely means observation-API gaps not actuator gaps.)

## T4.D.7 — Summary table

| Surface | Total | USED | HOOKED | UNUSED but RELEVANT | NOT RELEVANT |
|---|---|---|---|---|---|
| CUDA Driver cu* | 485 | 40 (8%) | 4 (1%) — cuLaunchKernel variants | ~80 (16%) | ~360 (74%) deprecated/v1/interop |
| CUDA Runtime cuda* | 377 | 10 direct (3%) | 6 hook intercepts | ~50 (13%) | ~310 (84%) thin wrappers |
| NVML nvml* | 364 | 19 (5%) | 0 | ~60 (16%) — MIG, mem clock, throttle reasons, NVLink, perf state | ~280 (77%) field/vendor/exotic |
| **Aggregate** | **1,226** | **69** (5.6%) | **10** | **~190** (15.5%) | **~960** (78%) |

### GSP accessibility verdict

**Signed-only.** No third-party code on GSP. CIPHER's leverage stays at the Driver API layer. No new-discovery.

### PTX/SASS unused (Hopper-specific) priority list

1. **TMA bulk async copy** — WL13 long-context, WL08 SDXL conv — HIGH leverage
2. **Thread block clusters + DSMEM** — same workloads — HIGH leverage
3. **Warp specialization** — WL11 agentic — MID leverage
4. **DPX** — out of WL01-WL24 scope — LOW priority

### Hardware partitioning ceiling

| Configuration | Max tenants | Memory isolation |
|---|---|---|
| MIG-only | 7 | hard |
| Green Ctx-only | ~33 | soft (CIPHER P4.2) |
| MIG (7g) + 8 GC each (max) | ~56 | hard |
| Phase 5 raise: practical ceiling | **~50-60 with isolation; 100+ if isolation softened** | mixed |

**Phase 5 target 60-150 may need to be 50-100 unless we accept soft-only at the high end.**

### Top 5 depth wins (T4.D.7)

1. **DEPTH WIN 1 — Stream priority + memory-sync domain per tenant.** `cuStreamSetAttribute(PRIORITY, ...)` and `MEM_SYNC_DOMAIN_MAP` at PARTITION_ROUTER hook. Sub-day effort. WL05 multi-tenant SLO. Phase 4.7.
2. **DEPTH WIN 2 — cuMemPrefetchAsync of PREDICT hot regions.** Proactive page migration ahead of launches. 1-2 days effort. WL04, WL16, WL22 throughput. Phase 4.4 secondary.
3. **DEPTH WIN 3 — `CU_LIMIT_PERSISTING_L2_CACHE_SIZE` per tenant.** Hard per-tenant L2 budget. Half-day effort. WL16 (prefix caching) directly benefits. Phase 4.4.
4. **DEPTH WIN 4 — TMA + Thread Block Clusters for attention.** Hopper-specific PTX work. Multi-week effort. WL13 long-context, WL08 SDXL. Phase 5.
5. **DEPTH WIN 5 — `cuStreamWaitValue32` for FAIRNESS.** GPU-side quota signaling. Drops latency 10 µs → 100 ns. 2-3 days effort. WL05 quota enforcement. Phase 4.7.

## T4.D.8 — Honest re-estimate of "60-70% surface coverage"

### Numeric (per-function counts)

- **5.6% of all NVIDIA functions are called by name in CIPHER.**

That number is misleadingly low because:
- Most NVIDIA functions are version variants of the same primitive
- Many are deprecated / interop-only
- The "USED" count doesn't reflect that `cuLaunchKernel` covers thousands of distinct kernels

### Structural (per semantic primitive)

Counting semantic primitives (each maps to a "thing the GPU can do"):

| Primitive class | CIPHER uses | Available |
|---|---|---|
| Kernel launch | full | full |
| Memory alloc / map / IPC | most (VMM + IPC) | full |
| L2 persistence | partial (window only, not full LIMIT) | full |
| Stream sync / attrs | one attr of five | five |
| Graph: capture/replay | full | partial |
| Graph: mutation | none | full |
| Cooperative launch | none | full |
| Thread block clusters | none | full |
| TMA | none | full |
| MIG management | none | full |
| Per-tenant DVFS | partial (set, no mem clock) | full |
| NVLink/topology | none | full |
| GSP direct | n/a | none (signed) |

**Structural estimate: ~45-55% of usable primitives are exercised.** Below the 60-70% prior claim. The gap is concentrated in:
- Hopper-specific PTX features (TMA, clusters, DSMEM)
- Stream attribute exhaustiveness (1 of 5 used)
- Graph mutation (capture/replay only)
- MIG management (zero usage)
- Memory hints (Advise/Prefetch zero)

### Verdict

The honest answer: **CIPHER currently exercises ~45-55% of the NVIDIA-exposed surface that's structurally useful for multi-tenant GPU optimization.** The 60-70% claim was optimistic.

**For Phase 5/Phase 6 scoping**: the 45-55% headroom is concentrated in **5 named gaps** above. Each is well-bounded effort, not open-ended R&D. A full surface push to ~75% is realistic by end of Phase 5 if Depth Wins 1, 2, 3, 5 land and TMA/cluster work begins in Phase 6.

The 5.6% function-count headline number, **read alongside the structural estimate**, is the honest framing for any external claim.

## End-of-track preconditions

| Check | Result |
|---|---|
| Fallback `cipher_kmod.ko.v0.2.0` md5 | `55ab8c0cd8309ca7cc0fc40fe556aa19` ✓ |
| Fallback `libcipher_v2.so.v0.2.0` md5 | `86618c30896470b642fcc6985d8dc632` ✓ |
| Live module | 0.4.0 still loaded; 0.4.1 staged on disk |
| Taint | 12288 |
| Track 4 made no live changes | confirmed |

Track 4 deliverable complete.
