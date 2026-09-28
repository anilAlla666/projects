# Phase 4 — Architecture (T4.0.7, binding)

**Date:** 2026-05-13
**Status:** Binding spec for P4.1–P4.8. Changes require explicit approval.
**Phase 3 baseline:** cipher_kmod 0.3.1 (srcversion `B1AF5E2A...`), Phase 3 telemetry substrate shipped 2026-05-13.

## Executive summary

CIPHER's Phase 3 shipped a per-tenant telemetry substrate: `/dev/cipher` ioctls 1/5/6/7, cipher-gpustate daemon polling NVML at 250 ms, libcipher_v2 with CUPTI launch counters, cipher-exporter on :9402. Every per-tenant metric (SM%, MEM%, LAUNCHES) reaches Prometheus end-to-end. Validation: 2-tenant TinyLlama-style workload at GPU saturation, live SM% splits visible in /metrics.

Phase 4 fuses cipher_rt's **35 Class A actuators** with that substrate. Each actuator currently reads global GPU state; Phase 4 rebuilds them against `struct cipher_tenant_snapshot` — a 336-byte 6-cache-line per-tenant view, RCU-protected, sub-200ns hot-path lookup, exported via the new ioctl nr 8 (CIPHER_GET_TENANT_SNAPSHOT).

The thesis: H100 hardware + per-tenant actuator fusion = B200-equivalent throughput-per-watt under multi-tenant load.

## The four success criteria (G1–G4) — workload-based

| Gate | Target | Measurement |
|---|---|---|
| **G1** | **85%+ MFU on each of WL01–WL24 individually, sustained 10+ minutes per workload** | `mfu_per_tenant.sh` per workload; sustained-window check |
| **G2** | **90%+ MFU on WL05 (Multi-tenant ×8 Llama-3.2-1B) specifically** | `mfu_per_tenant.sh` with WL05 launcher; aggregate device MFU |
| **G3** | **≥30 concurrent tenants in WL05 scenario** | `density_pack.sh` against WL05 launcher; per-tenant non-trivial SM time |
| **G4** | **Zero kernel oops/WARN, taint bits added ≤ 1 across 24-hour soak rotating through WL01–WL24** | `soak_24h.sh` extended for WL rotation; dmesg + taint deltas |

**All four must pass simultaneously, reproducibly, on real silicon.** Partial pass on three of four is FAIL.

## The 24-workload taxonomy (canonical, binding)

The four gates are evaluated against this workload set. Each WL has a reproducible launcher (to be specified in T4.0.5 measurement scripts as per-WL drivers).

| Code | Workload | Reference model |
|---|---|---|
| WL01 | LLM Decode B=1 | Llama-3.2-1B |
| WL02 | LLM Decode B=8 | Llama-3.2-1B |
| WL03 | LLM Prefill B=8 | Llama-3.1-8B |
| WL04 | vLLM Serving | (latest) |
| WL05 | Multi-tenant ×8 | Llama-3.2-1B (G2 + G3 gate workload) |
| WL06 | Embeddings | MiniLM-L6-v2 |
| WL07 | LoRA Fine-tuning | (Llama-class) |
| WL08 | Diffusion | SDXL |
| WL09 | Speech | Whisper |
| WL10 | Speculative Decoding | (Llama-class) |
| WL11 | Agentic Multi-turn | (Llama-class) |
| WL12 | Batch Processing | (mixed) |
| WL13 | Long Context 32K | (Llama-class) |
| WL14 | torch.compile | (any) |
| WL15 | MoE Models | (Mixtral-class) |
| WL16 | Prefix Caching | (Llama-class) |
| WL17 | Training Full | (any) |
| WL18 | Multi-GPU TP | (any) |
| WL19 | Vision | CLIP |
| WL20 | Multimodal | LLaVA |
| WL21 | Code Generation | (Llama-class) |
| WL22 | RAG Pipeline | (mixed) |
| WL23 | Model Switch Stress | (rapid weight churn) |
| WL24 | Quantized Native | AWQ |

WL01–WL24 is the **binding workload set**. P4.2–P4.7 cluster checkpoints reference the **declared-workloads-set** (workloads where any of that cluster's ops are A or s in `PHASE_4_OP_WORKLOAD_MATRIX.md`) and the **complement set** (the remaining WLs).

## Architecture diagram

```
+-------------------------------------------------------------------------+
| Phase 3 substrate (shipped 2026-05-13 — UNTOUCHED through Phase 4)      |
|                                                                         |
|   cipher_gpustate daemon ---NVML 250ms--->  ioctl 5 GPU_STATE  --+      |
|                                              ioctl 6 PROC_UTIL --+      |
|   libcipher_v2 (CUPTI in workload) --256x--->  ioctl 7 LAUNCHES --+     |
|                                              ioctl 1 REGISTER_TENANT    |
|                                                                  |      |
|                                  +-------------------------------v---+  |
|                                  | cipher_kmod 0.3.1                  | |
|                                  |   cipher_pid_table[] (Phase 3)     | |
|                                  |   cipher_gpu_state (Phase 3)       | |
|                                  +------------------+-----------------+ |
+--------------------------------------------------- |---------------------+
                                                     |
                  Phase 4 additions (cipher_kmod 0.4.0) |
                                                     |
+----------------------------------------------------v--------------------+
| cipher_kmod 0.4.0                                                       |
|                                                                         |
|   cipher_state_updater kthread (1 kHz)  <-- gpu_state + pid_table       |
|                                              writes derived fields on   |
|                                              cipher_pid_stats via       |
|                                              WRITE_ONCE                  |
|                                                                         |
|   struct cipher_tenant_snapshot  <-- assembled view, 336 B               |
|                                                                         |
|   cipher_get_current_tenant_snapshot()  ---> sub-200ns RCU lookup       |
|   cipher_get_tenant_snapshot_by_id()    ---> O(N) scan                  |
|   cipher_enumerate_tenants()            ---> snapshot copy              |
|                                                                         |
|   ioctl 8 CIPHER_GET_TENANT_SNAPSHOT    ---> cross-process query        |
|   ioctl 9 CIPHER_REQUEST_SM_PARTITION   ---> P4.2 allocator              |
|                                                                         |
|   EXPORT_SYMBOL_GPL for the three lookup functions (Phase 6 substrate)  |
+--------------------------------+----------------------------------------+
                                 |
+--------------------------------v----------------------------------------+
| cipher_rt actuators (REBUILT against contract — 35 Class A ops)         |
|                                                                         |
|   Mode 1 ioctl (~5 µs)    --- per-tenant query for ARBITRATE etc.      |
|   Mode 2 /proc poll (100ms) -- batched view for Stage 2 threads        |
|   Mode 3 TLS cache (sub-µs) -- per-launch decisions on hot path        |
|                                                                         |
|   Refreshed by Stage 0 ring-write hook on each kernel launch            |
+-------------------------------------------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------------+
| Phase 3 Prometheus exporter (already running)                           |
|                                                                         |
|   Extended in P4.8 with new metrics:                                    |
|     cipher_tenant_mfu_pct{tenant=X}                                     |
|     cipher_tenant_tpw_x{tenant=X}                                       |
|     cipher_tenant_partition_mask{tenant=X}                              |
|     cipher_tenant_l2_residency_kb{tenant=X}                             |
|     cipher_tenant_weight_dedup_savings_mb{tenant=X}                     |
+-------------------------------------------------------------------------+
```

## Sub-phase plan — 8 thematic clusters

The audit found **35 Class A ops** vs the 5-actuator P4.2–P4.6 plan. Grouped into 8 thematic clusters; sub-phases run sequentially with checkpoints.

| Phase | Cluster | Ops (key members) | Days |
|---|---|---|---|
| **P4.1** | Kernel ABI extension | cipher_kmod 0.4.0 + state_updater + ioctl 8/9 | 4 |
| **P4.2** | SM partition | PARTITION_ROUTER, SM_PACKER, ARBITRATE, PERSIST_ENGINE pool, GREEN_CTX integration | 10 |
| **P4.3** | DVFS / thermal | VOLT, HIBERNATE, THERMOSTAT, THERMAL_FEEDBACK, PULSE, CARBON, SUSTAIN | 8 |
| **P4.4** | L2 cluster | PREDICT, PERSIST_ENGINE admission, L2_PERSIST, MEM_LAYOUT | 7 |
| **P4.5** | Weight cluster | WEIGHT_SHARE accounting, WEIGHT_COMPRESS, SUBSTITUTE, RECIPES | 6 |
| **P4.6** | KV / attention | KV_COMPRESS, KV_REDIRECT, ATTN_KOOPMAN, GRAPH | 6 |
| **P4.7** | Fusion + agentic + compliance | FUSION, FUSION_KERNELS, FLOW_*, SENSE, SHIELD, GUARD, LOOP, PIPELINE, CONTINUITY, FAIRNESS, FAIRNESS_SHM, STRAGGLER, OVERLAP, GRAPH_INSPECT, TRACE | 7 |
| **P4.8** | Integration + density + soak | All-clusters demo, G3 density push, G4 24h soak, evidence tarball | 10 |

**Total estimate:** 58 calendar days. Compressible to ~45 days if P4.5 and P4.6 run in parallel (independent clusters, separate measurement gates).

## Cluster checkpoint gates — workload-set rule

Each P4.2–P4.7 checkpoint is evaluated as:

> **"Cluster N ops lift MFU on declared-workloads-set; no regression on complement set."**

Where:
- **declared-workloads-set** = the union of WL01–WL24 where any op in the cluster is marked `A` or `s` in `PHASE_4_OP_WORKLOAD_MATRIX.md`
- **complement set** = the remaining WLs (where this cluster's ops are marked `.`)
- **lift** = sustained MFU improvement vs the prior-cluster baseline on the declared set
- **no regression** = MFU on complement set within ±5% of prior baseline

Final P4.8 ship gate is the four G1–G4 criteria above, measured simultaneously on real silicon under WL05 multi-tenant load.

## Depth wins integrated into cluster sub-phases

Five depth wins identified in `PHASE_4_DEPTH_AUDIT.md` are integrated into Phase 4 sub-phases as opportunistic enhancements, not as new sub-phases:

- **Stream PRIORITY + MEM_SYNC_DOMAIN**: P4.2 (SM partition cluster) — wire `cuStreamSetAttribute(PRIORITY, ...)` and `MEM_SYNC_DOMAIN_MAP` in PARTITION_ROUTER's per-tenant stream setup
- **cuMemPrefetchAsync**: P4.4 (L2 cluster) — primary mechanism for WL13, WL15, WL17 hitting 85% MFU; proactive page migration of PREDICT's hot regions ahead of launches
- **CU_LIMIT_PERSISTING_L2_CACHE_SIZE per tenant**: P4.4 (L2 cluster) — hard L2 budget enforcement via `cuCtxSetLimit` at REGISTER_TENANT time; complements the existing `cudaAccessPolicyWindow` per-stream window
- **TMA descriptors + thread block clusters**: P4.5 (Weight cluster) — emit cluster-launched attention kernels via SUBSTITUTE/RECIPES PTX path; Hopper-only; biggest WL13 (long-context) and WL08 (SDXL) lift
- **cuStreamWaitValue32 / WriteValue32**: P4.7 (Fusion+agentic cluster) — primary mechanism for WL14 torch.compile lift; replaces SHM-based FAIRNESS quota signaling with GPU-side semaphores (sub-µs vs ~10 µs)

Each depth win is folded into its cluster's actuator rewrite, measured against the affected workloads' baseline JSONs.

## Kernel ABI contract (T4.0.2)

See `PHASE_4_CONTRACT.md` for the full spec. Key points:

- `struct cipher_tenant_snapshot` (336 B target, fits 6 cache lines)
- Three-identity bridge: `tenant_id_str[64]` + `tenant_session_fp` (u64) + `tenant_handle_u32`
- Lookup functions: `cipher_get_current_tenant_snapshot()` (< 200 ns target), `_by_id()` (O(N)), `_enumerate_tenants()`
- New ioctls: nr 8 GET_TENANT_SNAPSHOT (cross-process query), nr 9 REQUEST_SM_PARTITION (P4.2)
- `EXPORT_SYMBOL_GPL` on the three lookups (Phase 6 cipher_rt_km substrate)
- Phase 3 ioctls 1/5/6/7 frozen; reserved 2/3/4 still `-ENOSYS`

## Actuator API (T4.0.4)

`cipher_rt_tenant.h` + `.cpp` staged at `/home/ubuntu/cipher_rt_phase4_draft/`.

Three access modes:
- Mode 1: ioctl nr 8 (~5 µs)
- Mode 2: `/proc/cipher/stats` poll (100 ms cadence for Stage 2 threads)
- Mode 3: `__thread` TLS cache (sub-µs, refreshed on Stage 0 ring write)

### Binding deployment requirement: per-thread `/dev/cipher` fd (T4.0.9.D)

**Every actuator path that issues `/dev/cipher` ioctls from N parallel
threads (or N CUDA streams, or N worker processes) MUST hold its own fd to
`/dev/cipher`. Sharing a single process-wide fd across threads is
prohibited at the architectural level.**

Why this is binding:

Phase 4.2 T4.2.1 contention measurement under cipher_kmod 0.4.3 (post-reboot
2026-05-13) ran three sweeps against `CIPHER_REQUEST_SM_PARTITION`:

| Sweep | Setup | 33-thread p99 | ratio | ops/s |
|---|---|---:|---:|---:|
| Shared fd, hint=8                | 10,430 ns | 44.2× |  8.5 M |
| Shared fd, hint=1 (33×1 = 33 supply) | 11,892 ns | 50.4× |  8.1 M |
| **Per-thread fd, hint=1**        | **322 ns**  | **1.4×** | **75.9 M** |

The shared-fd ops/s flatlined at ~8 M from 4 threads through 33 — that is
kernel-side single-`struct file` throughput cap (VFS-level serialization),
not allocator contention. With per-thread fds the lock-free allocator
delivers near-linear scaling: 75.9 M ops/s at 33 threads and a 1.4× p99
ratio against the single-thread baseline.

A shared-fd deployment forfeits the lock-free allocator's design payoff and
re-introduces a ~50× p99 amplification under contention. **This is not an
allocator deficiency; it is a deployment misuse.**

Concrete constraints for any consumer of the actuator API:

1. **cipher_rt_tenant.h** consumers: open `/dev/cipher` per worker thread,
   not in module-init / process-init code. The `g_cipher_fd` pattern in the
   current `cipher_rt_phase4_draft/cipher_rt_tenant.cpp` is a known gap to
   close (tracked in PHASE_4_BACKLOG.md).
2. **Per-CUDA-stream actuators** (Phase 5 PARTITION_ROUTER, SM_PACKER,
   GREEN_CTX, REBALANCER): one fd per stream; the actuator's hot-path
   ioctl issues from the stream's CPU-side callback / launch hook and
   must reuse the per-stream fd, not the process-wide one.
3. **Multi-process tenants** (e.g. vLLM model-shard worker procs):
   each worker process opens its own fd — already correct under process
   isolation, called out here so it does NOT get refactored into a
   shared-fd "optimization".

Implementation-level note: the underlying `struct file *` carries position
state and file_lock that the kernel serializes on ioctl entry/exit. This is
a property of the VFS, not of cipher_kmod, so it cannot be fixed kernel-side
without rewriting kernel infrastructure. The cheapest enforcement is the
per-thread fd. Each open of `/dev/cipher` is `O(1)` and the file allocates
no resources in cipher_kmod beyond a `struct file` — open-cost is not a
hot-path concern.

Verification: anyone refactoring an actuator should re-run
`/tmp/cipher_test_phase4_partition_contention 33 30000 1` with and without
`CIPHER_PER_THREAD_FD=1` and confirm the ratio collapses below 5× in the
per-thread case. The test source lives at
`/home/ubuntu/cipher_phase4_tests/cipher_test_phase4_partition_contention.c`.

References: full discriminator data in PHASE_4_NOTES.md "Phase 4.2
contention scaling sweep" and in `/home/ubuntu/cipher-phase4-evidence/`
(`contention_sweep.log`, `contention_sweep_hint1.log`,
`contention_sweep_perthread_fd.log`).

## Measurement plan (T4.0.5)

`/home/ubuntu/cipher_measurement/`:
- `mfu_per_tenant.sh` — G1 measurement
- `tpw_per_tenant.sh` — G2 measurement
- `density_pack.sh` — G3 measurement (2 → 5 → 10 → 20 → 30 → 50 → 100)
- `soak_24h.sh` — G4 measurement (24-hour mixed workload, taint + WARN tracking)

All scripts read from cipher-exporter `/metrics`. No new instrumentation; bash + curl + awk.

## Cut list (T4.0.6 — applied in T4.1.x atomic with cipher_kmod 0.4.0 bump)

See `PHASE_4_CUTS.md` for rationale. Net:

| Op | Action |
|---|---|
| cipher_telemetry | Cut (redundant with Phase 3 daemon) → `#ifdef CIPHER_LEGACY_TELEMETRY` |
| cipher_nccl_bpf | Cut → `#ifdef CIPHER_HAVE_LIBBPF` |
| cipher_nccl_v4 | Cut → `#ifdef CIPHER_NCCL_MULTINODE` |
| cipher_nccl_neural | Source kept, build excluded → `#ifdef CIPHER_NCCL_MULTINODE` |
| cipher_nccl | Source kept, build excluded → `#ifdef CIPHER_NCCL_MULTINODE` |
| FP8_COMPUTE | **Keep** (audit-incomplete resolved: 19 returns are enabled-checks, not stubs) |

Audit-incomplete: 2 → 0. FP8_COMPUTE reclassified C→B; ATTN_KOOPMAN TODO deferred to P4.8 follow-up.

## Lambda-pod pod-degradation register

| Op | Lambda containerized | Workaround |
|---|---|---|
| VOLT (`nvmlDeviceSetGpuLockedClocks`) | `NVML_NOT_SUPPORTED` | Classifier-only mode; G2 measurement requires bare-metal |
| HIBERNATE (`nvmlDeviceSetPowerManagementLimit`) | `NVML_NOT_SUPPORTED` | Idle-gate-flag only (no actuation); G2 measurement requires bare-metal |
| NCCL multi-node | libibverbs absent | Phase 5 territory anyway |

**Phase 4 G2 gate:** if Lambda pod blocks measurement, document the projected lift from bare-metal characterization and the actual measurement gap. G1, G3, G4 measurable on Lambda.

## Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Class A count 35 >> 5: sub-phase plan undersized | high | Re-plan complete; 8 clusters across P4.2–P4.7 explicit above |
| Lambda pod blocks VOLT/HIBERNATE (NVML NOT_SUPPORTED) | medium | Bare-metal characterization needed; document projected-vs-measured |
| cipher_kmod 0.4.0 hashtable scaling to 150+ tenants | medium | Phase 4.8 stress test; fallback to deeper bucket hashing if contention |
| 24-hour soak: any kernel WARN/oops invalidates G4 | high | Per-checkpoint rmmod/insmod cycle proves clean module load; soak validates sustained stability |
| Phase 3 substrate regression during cipher_kmod 0.4.0 build | high | Additive-only changes; Phase 3 ABI nrs 1/5/6/7 frozen; per-checkpoint regression test (cipher_test_happy/root/negative — 17/17) |
| Identity-bridge inconsistency between three tenant IDs | medium | Contract holds all three; SENSE computes session_fp + handle_u32 from tenant_id_str at REGISTER_TENANT |
| WEIGHT_SHARE break during cluster rewrites | high | Phase 5 critical path; per-checkpoint accounting-layer test; **never modify the IPC protocol** |
| Build dependency (libssl-dev) regression | low | Now installed via `apt`; document in environment notes |
| ATTN_KOOPMAN fused-fast-path TODO not wired | low | Falls through to revert (correct output, slower); not a Phase 4 blocker |
| cipher_state_updater kthread CPU saturation at 150 tenants | medium | 1 kHz × 150 = 150k field writes/sec; well within budget; monitor in P4.8 |

## Sub-phase entry/exit checks

Every sub-phase boundary must verify:

1. **Phase 3 substrate intact:**
   - `cipher_kmod.ko.v0.2.0` md5 = `55ab8c0cd8309ca7cc0fc40fe556aa19`
   - `libcipher_v2.so.v0.2.0` md5 = `86618c30896470b642fcc6985d8dc632`
   - cipher_kmod ≥ 0.3.1 loaded
   - /proc/cipher/{stats,bar0_state,gpu_state} readable
   - Phase 3 stage 5b /metrics demo reproducible

2. **Refinement 2 (no writes to nvidia.ko region):**
   - `nm` cipher_kmod.ko shows no `iowrite32`, `__raw_writel`, `memcpy_toio`

3. **Taint stable** or only incremented per documented event

4. **All prior actuator measurements reproducible** (rerun mfu_per_tenant.sh / tpw_per_tenant.sh with same config gets numbers within ±5% noise)

If any precondition fails: STOP, roll back to last green state, do NOT fix forward.

## Execution rules (binding)

1. Sub-phases run sequentially. P4.5 and P4.6 may parallelize at the discretion of the implementer (independent clusters, separate measurement gates).
2. Each sub-phase ends with a checkpoint report: time elapsed, gate measurements, regression check, fallback md5s, taint state.
3. Failing a non-negotiable gate: STOP. No fix-forward.
4. New ioctls take fresh nrs (8, 9, 10...). Nrs 1/5/6/7 frozen. Nrs 2/3/4 reserved.
5. Live source modifications batched at T4.1 cipher_kmod 0.4.0 bump (`#ifdef` cuts applied atomically). Before T4.1: staging-only at `phase4_draft/` and `cipher_rt_phase4_draft/`.

## Phase 4 close-out evidence bundle

At P4.8 close:
- `PHASE_4_NOTES.md` (close-out)
- `PHASE_4_OP_AUDIT.md` (T4.0.1, updated with T4.0.6 resolutions)
- `PHASE_4_CONTRACT.md` (T4.0.2)
- `PHASE_4_CUTS.md` (T4.0.6)
- `PHASE_4_ARCHITECTURE.md` (this doc)
- `cipher-phase4-evidence.tar.gz` (sources + binaries + fallbacks + tests + measurement scripts + 24h soak log + density CSVs + per-checkpoint /metrics snapshots + this doc)
- `cipher-phase4-evidence.tar.gz.sha256`

## What this document does not do

- Specify the cipher_rt source-level changes per Class A op (that's P4.2–P4.7 per-op work, not architectural)
- Predict exact MFU/TPW numbers (those come from measurement, not projection)
- Decide whether to run on bare-metal vs Lambda containerized (operational decision; G2 measurement may require bare-metal)
- Cover Phase 5 (Green Contexts density — see updated tiered target below) or Phase 6 (kernel-space cipher_rt_km) — both pre-figured here as future consumers of the contract

## Phase 5 density target (revised post-depth-audit)

The original "60-150 tenants per GPU" target was aspirational. Per `PHASE_4_DEPTH_AUDIT.md` (T4.D.5), the practical hardware-partitioning ceiling on H100 SXM5 is:

- **Hardware-isolated tier: 50-60 tenants per GPU**
  MIG (max 7 slices) × Green Contexts (~8 per slice) = ~56 tenants with hardware memory isolation. Recommended for production serving where cross-tenant data isolation is required.

- **Soft-isolated tier: up to 100 tenants per GPU**
  Single MIG slice (full GPU) × Green Contexts (~33 with 4-SM granularity) extended via finer Green Context subdivision = ~100 tenants. Suitable for batch / training / internal workloads where memory isolation is not required.

Phase 5 target: **50-100 with tiering**, grounded in the depth audit's hardware partitioning math. The 60-150 target is retired.

## Positioning language

CIPHER operates at the deepest layer NVIDIA exposes to third parties on Hopper hardware:
- Userspace symbol interception (GOT/PLT) in `libcipher_hook.so`
- CUDA driver injection (`CUDA_INJECTION64_PATH`) via `libcipher_v2.so`
- Kernel-level ioctl observation (kprobes on `nvidia_unlocked_ioctl`) in `cipher_kmod`

GSP firmware is signed and not accessible to any third party (verified in `PHASE_4_GSP_INVESTIGATION.md` against `kernel_gsp_booter.c` + libspdm). CIPHER currently uses **45-55% of the available capability at this layer**, with a documented path (5 depth wins above) to 80%+ by Phase 6.

## Sign-off check

End-of-T4.0.7 preconditions to be verified at the P4.0 checkpoint:
- Fallback md5s unchanged ✓ (verified at start of each sub-phase)
- cipher_kmod 0.3.1 still loaded ✓
- Taint 12288 stable ✓
- Phase 3 /proc/cipher entries readable ✓
- No code applied to live tree yet (all P4.0 deliverables are spec/script/staging files) ✓

Phase 4.0 checkpoint: PASS if all five preconditions plus all seven P4.0 deliverables (T4.0.1 through T4.0.7) are complete.
