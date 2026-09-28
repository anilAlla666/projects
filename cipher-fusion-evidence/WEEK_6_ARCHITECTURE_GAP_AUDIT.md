# WEEK_6_ARCHITECTURE_GAP_AUDIT.md

**Date:** 2026-05-23
**Scope:** Read-only research audit. V1.2.2 substrate vs the product goal "**100 concurrent agents per H100, each potentially running a different model (Mistral / Qwen / Llama / SLMs), driver-level multiplexing via LD_PRELOAD, zero application code changes.**"
**Pre-condition anchors at audit time:**

- `cipher-fusion-evidence` HEAD = `cc913a6` (Week 6 bench harness landed)
- `cipher_rt_phase4` HEAD = `ec0e005` (week-5-complete)
- `cipher_kmod` HEAD = `2fc70c3` (week-5-complete)
- `libcipher_rt.so` md5 = `259ac994aead2da8289fc84d6116fbe9`

No source edits, no builds, no GPU runs. The only artifact produced is this document.

---

## 0. Executive answer

> **The product goal is partially supported by the substrate that ships today. The structural blockers are not the items v1.2.2 §7 schedules; they are an orthogonal set.**
>
> Of the canonical 33 ops, ~25 are present in tree (the brain ports in `cipher_rt_phase4/src/may13/` are compiled, the hot-path wiring is staged into v1.2.2 W2-W10). What is *missing* for the 100-agent heterogeneous-model goal is six load-bearing items:
>
> 1. **CP 5.4 allocation table is sized at 64 (HARD STOP)** — cannot register a 65th tenant SM partition.
> 2. **Weight-arena slot cap is 16** — at 100 different-model agents, every agent wants its own arena.
> 3. **KV-dedup hash key is content-only** — at heterogeneous models the same 2 MiB content can land on incompatible tensor layouts; this is a *correctness* defect, not a perf defect.
> 4. **Marlin INT4 weight kit is process-global, single-model** — at 100 different models inside one LD_PRELOAD process scope, Marlin cannot be tenant-scoped.
> 5. **VA pool is 80 GiB per-process** — at N=100 separate vLLM processes the substrate wants 8 TB of host VA.
> 6. **AUDIT / RECEIPT / CARBON chains are not present in the kmod** — billing primitives for the 100-agent neocloud product are unbuilt.
>
> v1.2.2 §7 W7-W14 (COMMIT, RING_WRITE, Koopman tier, CP 5.5) does not close items 1, 2, 3, 4, 5, 6 directly. Items 1 and 2 are mentioned in §8 R-C2 ("bump arena slots from 16 to 32 (kmod 0.5)") but the bump is to 32, not to ≥100, and the CP 5.4 allocation table is a separate cap not bundled in that mitigation. Items 3 and 4 are not in the v1.2.2 risk register. Item 5 is not in the v1.2.2 risk register. Item 6 is enumerated in v1.2 §2 (COMMIT/AUDIT/RECEIPT/CARBON are canonical-33 ops) but the *kmod-resident state* for those ops is not present.
>
> The May 13 POC (15 tenants × Llama-3.2-1B) was run against the **op31-prod** dispatch surface, which has been superseded by the Week-5 consolidation surface. The POC source exists on disk; "reproduction on V1.2.2 substrate" is a **reconstruction** task, not a verification task. Eng cost estimate is in §6.
>
> Strategic read (full §7.1 below): **v1.2.2 §7 is adjacent to the 100-agent heterogeneous-model goal, not on its critical path.** The right next-prompts (full §7.2 below) re-sequence the order: close the structural caps first, the dispatch correctness defects second, the billing chains third, then the v1.2.2-scheduled work lands productively on top.

---

## 1. Product requirement decomposition (Part A)

Each requirement gets a stable ID. Used in §3 for the gap matrix.

### 1.1 Density (R-D*)

| ID    | Requirement |
|-------|-------------|
| R-D1  | 100 concurrent tenant processes per H100, each with its own LD_PRELOAD interception scope |
| R-D2  | Per-tenant state structures (tenant id, partition assignment, weight binding, KV cache scope, AUDIT chain position) scale to 100 |
| R-D3  | AUDIT chain write rate at 100 tenants × decode hot path must not stall the LD_PRELOAD intercept hot path (sub-µs budget per R-W9.2) |
| R-D4  | TelemetrySampler scope (10 ms NVML samples per metric) aggregates across 100 tenants without sample loss |
| R-D5  | Scheduler overhead for 100-tenant ARBITRATE/FAIRNESS decisions fits the actuator-contract budget |

### 1.2 Model heterogeneity (R-H*)

| ID    | Requirement |
|-------|-------------|
| R-H1  | Weight kit is tenant-scoped, not process-global. Different tenants → different models → different INT4 weight residence |
| R-H2  | Actuator routing is model-aware. Marlin variants for Mistral-7B and Qwen-7B differ (shape distributions). Koopman recipe registry keys on (model, layer, shape), not just shape |
| R-H3  | KV-cache scope is model-aware. Cross-tenant KV-dedup is valid only within identical (model, layer, head) triples |
| R-H4  | Per-tenant tokenizer state. CIPHER plugin must not assume a single tokenizer |
| R-H5  | Weight-residence strategy at 100 different models (140 GB if FP16 7B × 10, exceeds 80 GB). Tiered (hot HBM / warm host RAM / cold NVMe), aggressive INT4 (~15 7B-INT4 models fit), or model-pooling (small set of unique models, agents bind to one) |

### 1.3 Isolation (R-I*)

| ID    | Requirement |
|-------|-------------|
| R-I1  | Per-tenant p99 bounded under noisy-neighbor adversarial workloads (SHIELD's role) |
| R-I2  | Fair scheduling — one bursting agent does not starve idle agents' eventual bursts (FAIRNESS's role) |
| R-I3  | Dynamic SM partition reassignment as agents burst and idle (PARTITION_ROUTER + ARBITRATE's role) |
| R-I4  | Per-tenant cryptographic billing receipts (AUDIT + CARBON + RECEIPT's role) |

### 1.4 Deployment (R-Dep*)

| ID     | Requirement |
|--------|-------------|
| R-Dep1 | LD_PRELOAD only at install. Customer applications run unchanged |
| R-Dep2 | Tenant-identity propagation — tenant A starts a process, CIPHER associates that process with tenant id A for its lifetime |
| R-Dep3 | Per-tenant model registration — CIPHER knows which model each tenant uses, to select the right actuator routing |
| R-Dep4 | HMAC-chained per-tenant billing receipts, exportable to neocloud operator for invoicing |

---

## 2. Current substrate inventory (Part B)

Findings from parallel read-only investigation of `cipher_kmod/`, `cipher_rt_phase4/`, `cipher_vllm_plugin/`, `cipher-may13-evidence/`, and the v1.2.2 plan + supporting docs. Every claim cites file:line.

### 2.1 LD_PRELOAD interception layer (B.1)

The injection entrypoint is `cipher_rt_phase4/cipher_inject.c:64–76`. It exports both `InitializeInjection` and `InitializeInjection2` (the modern CUDA debug-injection ABI). Both route to a single `cipher_v2_init_body()` via `pthread_once` (`cipher_inject.c:32, 67`). The body sequence (`cipher_inject.c:35–61`) does *not* open `/dev/cipher` at injection time; instead it registers per-subsystem init hooks and calls `cipher_v2_tenant_register()` (`cipher_inject.c:37`) — the Phase 2 tenant-register ioctl path.

Env vars read at injection time: `CIPHER_VOLT_ENABLED` and clock-lock config in `cipher_rt_volt.c:203–204`; `CIPHER_MARLIN` in `cipher_rt_marlin_actuator.c:203`; `CIPHER_KV_ALLOC`, `CIPHER_KVDEDUP`, `CIPHER_OFFLOAD`, `CIPHER_TENANT_NUM`, `CIPHER_KV_VA_POOL_GIB`, `CIPHER_RT_DIR` in the cipher_vllm_plugin tier. The `/dev/cipher` file descriptor is opened *lazily* on first tenant query (`cipher_rt_tenant.cpp:83–94, 109–116`), not at `InitializeInjection2`.

### 2.2 Tenant identity at the hook layer (B.1 cont.)

**Tenant is not first-class at the hook/actuator layer; it is process-global with thread-local caching.** The current tenant id is determined by `getpid()` / `gettid()` and propagated to the kmod via the `CIPHER_GET_TENANT_SNAPSHOT` ioctl (nr 8) keyed by the caller's tid (`cipher_rt_tenant.cpp:225–226`, `my_tid()` = `gettid()` syscall at `cipher_rt_tenant.cpp:78`). The kmod side resolves tid → tenant_id by hashtable lookup. Each thread holds a `__thread` cached snapshot (`cipher_rt_tenant.cpp:65`), refreshed lazily on first query (`cipher_rt_tenant.cpp:221–232`). Actuators call `cipher_rt_tenant_cached()` (`cipher_rt_tenant.cpp:235`).

vLLM-side tenant identity comes from the `CIPHER_TENANT_NUM` env var (`cipher_vllm_plugin/cipher_vllm_kvdedup.py:62`, `cipher_vllm_plugin/cipher_vllm_kv.py:46`), defaulting to `0` if unset. This is the *value* the plugin tells the bridge to use; there is no kmod-side guard that two processes can't both register tenant 0.

### 2.3 Actuator substrate (B.2)

- **Marlin INT4 actuator** (`cipher_rt_phase4/cipher_rt_marlin_actuator.c`): the actuator registers at priority 10 (`cipher_rt_marlin_actuator.c:197`), gated by `CIPHER_MARLIN=on` (`cipher_rt_marlin_actuator.c:203–214`). The INT4 weight residence path observes a `weight_ptr` and, after `STABILITY_THRESHOLD` (4 observations, `cipher_rt_marlin_actuator.c:51`) on the same pointer, calls `cipher_rt_marlin_engine_quantize_repack(weight_ptr, marlin_K, marlin_N)` (`cipher_rt_marlin_actuator.c:157–158`). The engine's weight cache is keyed by the **`weight_ptr` alone** — there is no `(model_id, layer_id, shape)` tuple. **The kit is hard-coded to one implicit model per process.** Gate constraints: `M ≤ 64`, FP16 dtype, `K % 128 == 0`, `N % 64 == 0` (`cipher_rt_marlin_actuator.c:131–146`).

- **VOLT/DVFS** (`cipher_rt_phase4/cipher_rt_volt.c`): one `nvmlDevice_t` and one `locked_mhz` per process (`cipher_rt_volt.c:65–81`). NVML handle is `dlopen`'d once (`cipher_rt_volt.c:83–102`). Assumes a single device (GPU 0); batch-to-MHz lookup is hard-coded (`cipher_rt_volt.c:54–63`). **VOLT is GPU-global, not per-tenant** — one clock policy for the whole device (confirmed in `NEOCLOUD_SUBSTRATE_AUDIT.md:321`).

- **cuBLAS dispatcher** (`cipher_rt_phase4/cipher_rt_matmul_dispatch.c`): process-global registry, 16-slot array, mutex-protected, priority-sorted, append-only (`cipher_rt_matmul_dispatch.c:15–29, 52–82`). **No tenant scoping**; passes M/N/K/dtype opaquely.

- **SDPA dispatcher** (`cipher_rt_phase4/cipher_rt_attn_dispatch.cpp`): process-global registry, 16-slot max (`cipher_rt_attn_dispatch.cpp:57–68`); priority-sorted (`cipher_rt_attn_dispatch.cpp:218–226`). Currently registered: `test_actuator` (priority 0) if `CIPHER_ATTN_TEST=on`. AUDIT actuator registers priority 0 (`cipher_rt_audit.c:132–137`).

- **KV-bridge** (`cipher_rt_phase4/cipher_kv_bridge.cpp`): per-slab tenant tagging via `struct cipher_rt_kv_page_tag` with `uint32_t tenant_id` (`cipher_kv_bridge.cpp:43, 51–64`). **KV per-slab is tenant-aware.** Slab allocation is guarded by a per-process mutex (`cipher_rt_kv_alloc.c:54`). Single VA pool per process, 2 MiB page granularity, 80 GiB default (`cipher_vllm_plugin/cipher_vllm_kv.py:50–62`, `va_gib=80` at line 58).

### 2.4 Classifier brain ports — `cipher_rt_phase4/src/may13/` (B.3)

21 ported `.cpp` files, ~5,688 LOC. Operation map (every file corresponds to one of the canonical-33 ops; numbers in parentheses are approximate LOC):

| Op | File | LOC | Status |
|----|------|-----|--------|
| CLASSIFY | `cipher_dispatch.cpp` | 543 | ported; hot-path wiring scheduled W2 |
| ORACLE | `cipher_oracle.cpp` | 539 | ported; CPU stub, no hot-path actuator wiring |
| SENSE | `cipher_sense.cpp` | 340 | ported; session-boundary inference firing |
| PREDICT (recipes) | `cipher_recipes.cpp` | 522 | ported; weight/KV prediction table |
| TOPOLOGY | `cipher_topology.cpp` | ~100 | ported; NVLink/PCIe adjacency, no-op observer |
| TRACE | `cipher_trace.cpp` | 116 | ported; JSONL ring exporter |
| DETERMINISM | `cipher_determinism.cpp` | 88 | ported; dispatch-fingerprint accumulator |
| FAIRNESS | `cipher_fairness.cpp` + `cipher_fairness_shm.cpp` | 184 + 229 | ported; per-session grid·block quota + SHM variant |
| CARBON | `cipher_carbon.cpp` | 169 | ported; v1 per-tenant carbon estimate (userspace only) |
| RECEIPT | `cipher_receipt.cpp` | 225 | ported; per-session signed compute proof (userspace only) |
| GUARD | `cipher_guard.cpp` | 207 | ported; cross-session params_hash residency detector |
| COMPLY | `cipher_comply.cpp` | 82 | ported |
| LOOP | `cipher_loop.cpp` | 286 | ported (stage 1 only) |
| PIPELINE | `cipher_pipeline.cpp` | 239 | ported |
| PULSE | `cipher_pulse.cpp` | 416 | ported; per-(M,K,N) Welford + ITL fault warning |
| CONTINUITY | `cipher_continuity.cpp` | 236 | ported; v1 attention-region checkpointing observer |
| STRUCTURAL_LOOKUP | `cipher_structural_lookup.cpp` | 300 | ported |
| KERNEL_TABLE | `cipher_kernel_table.cpp` | 317 | ported |
| TELEMETRY | `cipher_telemetry.cpp` | 410 | ported |
| L2_PERSIST | `cipher_l2_persist.cpp` | ~250 | ported |
| RUNTIME | `cipher_runtime.cpp` | ~250 | ported |
| LIQUID_STATE | `cipher_liquid_state.cpp` | ~300 | ported |

**Pending / not yet ported:** STRAGGLER (in op31 build but not in may13/), SAMPLE (not found), full RING_WRITE substrate (deferred to v1.2.2 W9-10), COMMIT atomic primitive (deferred to v1.2.2 W7-8), AUDIT *kmod-resident* HMAC chain (no kmod-side implementation; userspace AUDIT in `cipher_rt_audit.c` works as observer but has no per-tenant chain head). **Koopman EDMD runtime is built (`cipher-may13-evidence/src/cipher_edmd_live.cpp`) but has no caller** — `cipher_rt_phase4/src/may13/cipher_dispatch.cpp:38` declares a weak reference to `cipher_edmd_live_collect` that is never called (Koopman tier wiring is v1.2.2 W11-12).

Per the user's "25 of 33" framing: 21 may13/ ports + 4 actuators already wired in Tree B (Marlin, VOLT, cuBLAS shim, SDPA dispatcher) = 25. The remaining 8 are COMMIT, RING_WRITE, the 3 Koopman tier ops, STRAGGLER, SAMPLE, and the kmod-resident audit/billing chain.

### 2.5 Multi-tenant primitives in kmod (B.4) — **HARD STOPS LIVE HERE**

**Slot caps:**

| Cap | Value | Source | Implication for 100 agents |
|-----|-------|--------|----------------------------|
| `CIPHER_CP54_MAX_ALLOCS` (per-PID partition allocation table) | **64** | `cipher_cp54_sched.c:123` | **HARD STOP** — `REQUEST_SM_PARTITION`-style allocation returns -ENOMEM at the 65th tenant |
| `CIPHER_WA_MAX_ARENAS` (weight-arena slot cap, **global**) | **16** | `cipher_weight_arena.c` (via `cipher_ioctl.h:481`) | At 100 different-model agents, no agent can register a 17th unique-model arena |
| PID hashtable buckets | 1024 | `cipher_internal.h:73–74` | At 100 agents this is ~10% load — functional but RCU lookup latency grows under collision |
| `cipher_pid_stats` per-slot array | 24 NV_ESC slots + 2 aggregate | `cipher_internal.h:32–35` | Adequate; does not gate density |
| KV-dedup per-tenant pages | 8192 | `CIPHER_KVDEDUP_MAX_PER_TENANT`, `cipher_kvdedup.h:82` | Adequate for typical agentic decode loads |
| KV-dedup global entries | 2²⁰ | `cipher_kvdedup.h` (per content) | Adequate |
| CP 5.4 SM groups (hardware) | 15 × 8-SM | `cipher_cp54_sched.c:118`; measured on this pod per `cipher-cp54-15groups` memory | At 100 agents we already cannot give every agent a *partition*; the design assumes POOL coexistence (CP 5.4 Step 1.3b' partial-close) |

**Per-tenant state:** `struct cipher_pid_stats` (`cipher_internal.h:172–234`), allocated `GFP_ATOMIC` in `cipher_pid_get_or_create` (`cipher_probe.c:41–76`), ~1.7 KB per entry, identified by LWP PID. Lifecycle: inserted on first nvidia ioctl observation, reaped at `do_exit` via kprobe (`cipher_probe.c:184–221`) with RCU grace-period defer (`kfree_rcu`). Locking: `cipher_pid_insert_lock` spinlock guards hashtable mutations; per-field `WRITE_ONCE` on derived telemetry (readers tolerate torn reads per `cipher_internal.h:207–233`).

**Tenant registration:** `CIPHER_REGISTER_TENANT` ioctl nr 1 (`cipher_dev.c:58–96`). Anti-spoof: `payload.pid == current->pid && payload.tgid == current->tgid` enforced at `cipher_dev.c:70–71`. Upserts hashtable entry with `tenant_id[64]` string, FNV-64 session fingerprint, murmur32 handle (`cipher_dev.c:85–93`). **No inheritance on fork** — child processes must re-ioctl.

**Ioctl surface (26 NRs):** nr 1 REGISTER_TENANT, nrs 2-4 reserved (-ENOSYS), nr 5 SUBMIT_GPU_STATE (CAP_SYS_ADMIN), nr 6 SUBMIT_PROCESS_UTIL, nr 7 SUBMIT_LAUNCH_STATS, nr 8 GET_TENANT_SNAPSHOT, nr 9 REQUEST_SM_PARTITION (deactivated → -ENOSYS, replaced by CP 5.4), nr 10 SET_CLOCK_MHZ (VOLT), nr 11 SUBMIT_FLOP_SAMPLE, nr 12 QUERY_FLOPS, nrs 13–15 CP54 ALLOCATE/FREE/QUERY, nrs 16–20 CP54 DSM FSM (SUBSCRIBE/POLL/START/ACK/COMPACT), nrs 21–24 weight-arena fd custodian (REGISTER/IMPORT/LEAVE/QUERY), nr 25 PUSH_CLASSIFY_STATS, nr 26 DSM_PROPOSE. (All NRs documented in `cipher_ioctl.h`; dispatch in `cipher_dev.c:244–297`.)

**Cross-tenant communication that is wired:**

- Weight-arena VMM POSIX-fd custodian: implemented (`cipher_weight_arena.c:157–214`); 16-slot global cap; producer/consumer fd lifecycle managed via 5 s reaper (`cipher_weight_arena.c:53, 128–152`).
- KV-dedup arena: implemented as a separate device `/dev/cipher_kvdedup` (`cipher_kvdedup.c:1–35`); per-request xxhash64 lookup; refcount table + cuIpc fd refs held in kmod; PUT/CONFIRM/RELEASE ioctls (`cipher_kvdedup.c:60+`).

**Cross-tenant primitives that are NOT wired:**

- **AUDIT chain shm — NOT IMPLEMENTED.** No kmod-side HMAC accumulator. No cross-tenant audit-only shared memory. Zero grep hits for "AUDIT" in `cipher_kmod/`.
- **CARBON / RECEIPT kmod-side state — NOT REFERENCED.** Zero hits in `cipher_kmod/` for either string.
- **FAIRNESS table — STUB ONLY.** `fairness_quota_remaining_pct` (`cipher_internal.h:230`) is a `u32` field defaulted to 100 in `cipher_state_updater.c:132–133`, with a comment "leave at 100 (default to full quota)" pending the FAIRNESS table integration. No quota enforcement, no per-tenant ledger.

**SM-arbitration (CP 5.4) ledger:** state in `cipher_cp54_sched.c:118–139`. 15-element atomic group array (`cipher_cp54_sched.c:118`) encodes free/OWNED/RESERVED bits with embedded PID (`cipher_cp54_sched.c:74–80`). Per-PID metadata in 64-slot table `cipher_cp54_allocs[64]` (`cipher_cp54_sched.c:124–139`): pid, qos_class (PARTITION/SHARED/POOL), in_use flag, migration state (IDLE/PROPOSED/MIGRATING), target_mask, timeouts. Locking: ALLOCATE under `cipher_cp54_lock` mutex (`cipher_cp54_sched.c:158`); FREE / release(pid) lock-free via `atomic_cmpxchg`; QUERY lock-free.

**Track 3 DSM FSM:** states + transitions in `cipher_cp54_sched.c:365–414` (FSM comment block) + handler nrs 16–20 (`cipher_cp54_sched.c:702–804`). Timeout 30 s (`CIPHER_CP54_MIG_TIMEOUT_NS`, `cipher_cp54_sched.c:93`). Commit and abort paths: `cp54_commit_migration` (`cipher_cp54_sched.c:383–398`), `cp54_abort_migration` (`cipher_cp54_sched.c:402–414`). Evaluator: `cp54_eval_migration` (`cipher_cp54_sched.c:445+`).

### 2.6 Per-tenant billing infrastructure (B.5)

- **AUDIT chain:** userspace `cipher_rt_audit.c` records a tamper-evident HMAC-SHA256 chain over every op + decision (`cipher_rt_audit.h:1–20`). Entries include `timestamp_delta` (`cipher_rt_audit.h:46`) but **no `tenant_id`** in the entry struct (`cipher_rt_audit.h:43–52`). **No kmod-resident chain head**; AUDIT is process-wide only. To produce per-tenant billing receipts the chain head must be tenant-scoped *and* kmod-resident so it survives any single tenant crashing.
- **CARBON:** ported to `cipher_rt_phase4/src/may13/cipher_carbon.cpp` (169 LOC, userspace). No kmod-side state.
- **RECEIPT:** ported to `cipher_rt_phase4/src/may13/cipher_receipt.cpp` (225 LOC, userspace). No kmod-side state.
- **HMAC-chained per-tenant billing receipts exportable to neocloud operator (R-Dep4):** the data structures exist (userspace), the export protocol is unwritten.

### 2.7 KV-cache management substrate (B.6) — **HARD STOP LIVES HERE**

- **`cipher_vllm_kvdedup.py`** (vLLM plugin): dedups KV pages by **content hash alone** — `cipher_rt_kv_dedup_alias(page_devptr)` (`cipher_vllm_kvdedup.py:138–150`) calls the kmod, which reads page content via DtoH copy and xxhash64s the 2 MiB body (`cipher_rt_kv_alloc.h:112`). **The hash key has no per-tenant, per-model, per-layer, or per-head scoping.** Per-tenant `tenant_id` field exists in `struct cipher_rt_kv_page_tag` (`cipher_rt_kv_alloc.h:43`) but is **only in the page-tag metadata** — it is never used in the hash lookup. **Heterogeneous-model implication:** at two different-model tenants (Llama-3.1-8B with 32 kv_heads, head_dim=128 vs. Mistral-7B with 8 kv_heads, head_dim=128), if both tenants run identical 2 MiB system-prompt content, **the kmod will issue tenant B a dup'd fd onto tenant A's page layout**, producing silent attention errors at decode time. This is a correctness defect, not a performance defect. NEOCLOUD_SUBSTRATE_AUDIT.md confirms (`NEOCLOUD_SUBSTRATE_AUDIT.md:508–509`) that real-world cross-process KV-dedup hit-rate at B2-B (different-model) is "~0% true content hits — different models have different KV layouts and content," which is the same finding observed from a different angle.
- **`cipher_kv_offload.py`** (CP 5.2): the `bind_kv_caches` method identifies the block dim by scanning for `shape[d] == num_blocks` (`cipher_kv_offload.py:72–102`). Gather/scatter uses `torch.index_select(kv, self.block_dim, idx)` (`cipher_kv_offload.py:127`). Assumes a fixed paged layout where all layers have identical block-dim position — valid for *one* model. At preempt/resume across mixed-model contexts, the gather shape is per-layer incompatible; `cipher_kv_offload.py:160` checks `if len(new_block_ids) < n` but does **not** validate shape compatibility across model swaps.
- **`cipher_vllm_kv.py`** (CP 5.1 buffer ownership): `cipher_kv_bridge.init(va_gib * (1 << 30))` (`cipher_vllm_kv.py:50–62`), `va_gib` defaults to 80 (line 58). **VA reservation is per-process, not per-tenant.** At N=100 tenant *processes*, each reserves 80 GiB host VA → 8 TB demand against typical 1 TB host VA → exhaustion at N ≈ 12. (This is host-VA, not GPU HBM; mitigation is to size the VA pool per actual model size.)

### 2.8 Telemetry + observability scope (B.7)

The Week-6 bench harness's `TelemetrySampler` reads NVML at **device level** — `nvmlDeviceGetPowerUsage`, `nvmlDeviceGetMemoryInfo`, `nvmlDeviceGetUtilizationRates`, `nvmlDeviceGetClockInfo(SM)` (per `WEEK_6_BENCH_HARNESS_REWRITE.md` §1, §2). The known-limitations section (`WEEK_6_BENCH_HARNESS_REWRITE.md:177`) states: *"GPU-only power. We measure `nvmlDeviceGetPowerUsage` at the GPU package level."* **No per-tenant attribution surface exists at the harness level.** Per-tenant TPS/TPOT/TTFT attribution lives in vLLM streaming hooks (one per process); the harness does not aggregate across 100 processes today. Per-tenant *power* attribution from a single GPU is **not exposed by NVML at the 10 ms cadence** — `nvidia-smi pmon` provides per-process GPU breakdown at 1 Hz only (an environmental limit, not a CIPHER bug). At 100 tenants × 10 ms sample × per-tenant attribution, no NVIDIA-supported path exists; tenants must be cross-correlated via CIPHER's CUPTI launch counts + per-tenant kernel-time accumulators on the kmod side.

### 2.9 What v1.2.2 plan calls "100-tenant heterogeneous" (cross-reference)

CP 5.5 in `CIPHER_REENGINEERING_PLAN.md` at line 1515 is the "full 100-tenant heterogeneous benchmark." Workload distribution at line 1521: "100-tenant launcher mixed-mode (5 prefill + 80 decode + 15 burst)." Verification gates at line 1532 cover COMMIT atomicity at N=100, RING_WRITE throughput at N=100, per-tenant FAIRNESS, `/proc/cipher/op_status` enumeration. **The plan's "heterogeneous" means heterogeneous workload classes (prefill / decode / burst mix), not heterogeneous model architectures.** Confirmed by `NEOCLOUD_SUBSTRATE_AUDIT.md` (line 482–520), which decomposes:

- **B1** (single-process, ~100 agents, line 12): substrate green; KV-dedup needs operator trigger; not a correctness blocker.
- **B2-A** (same-model two-customer, line 13 + 482–500): substrate ready; cross-process KV-dedup + weight sharing + isolation + fairness all PASS.
- **B2-B** (different-model two-customer, line 13 + 501–520): **only SM-partition isolation + fairness ledger fire**; cross-tenant weight-share and KV-dedup hit-rate are ~0% (line 508–509, 517–518).
- **B3** (5-customer, 5-model, dynamic load, line 14 + 547–563): **NOT v1; deferred to CP 5.5 / Weeks 13–14**; missing COMMIT, RING_WRITE, 24 h soak, cold-start optimization, SLA-aware power management.

`CIPHER_WORKLOAD_ARCHITECTURE.md` defines Class D (line 38) by **agent count + workload shape**, not by model heterogeneity. The word "heterogeneous" appears at line 62 to describe "bursty, heterogeneous control flow" — control-flow heterogeneity, not model heterogeneity.

**Implication:** the 100-agent heterogeneous-*model* product framing is **NOT what the v1.2.2 plan's headline CP 5.5 measures**. CP 5.5 measures workload-class heterogeneity on (likely) one model architecture. Heterogeneous *model* multi-tenant is B2-B / B3 in NEOCLOUD vocabulary, and is partially recognized as a substrate-ready / partially-substrate-ready / not-substrate-ready trichotomy in the NEOCLOUD audit.

---

## 3. Gap classification matrix (Part C)

Classification key:

- **SUPPORTED** — code exists, tested, ready at the 100-agent heterogeneous scale
- **PARTIAL** — code exists at lower scale; extension is mechanical
- **BUG** — code exists but has a known defect blocking the requirement
- **MISSING** — no code; new substrate work required
- **ENVIRONMENT** — hardware / deployment constraint, not engineering work
- **N/A** — does not apply to this cell

Component columns: **HOOK** = libcipher_rt LD_PRELOAD + tenant identity (B.1, B.7); **ACTU** = actuator substrate (B.2); **KMOD** = kmod multi-tenant primitives (B.4); **KV** = KV-cache plugins (B.6); **BILL** = AUDIT/CARBON/RECEIPT (B.5); **TELE** = telemetry/observability (B.7).

| Req      | HOOK         | ACTU                                 | KMOD                                  | KV                                    | BILL                          | TELE                                  |
|----------|--------------|--------------------------------------|---------------------------------------|---------------------------------------|-------------------------------|---------------------------------------|
| **R-D1** 100 procs/H100, each LD_PRELOAD | SUPPORTED   | N/A                                  | PARTIAL (hashtable 1024 buckets @ ~10% load) | N/A                                | N/A                           | ENVIRONMENT (host VA 8 TB at 80 GiB×100) |
| **R-D2** Per-tenant state scales to 100 | PARTIAL (process-global, thread-cache) | PARTIAL (process-global registries 16-slot ok) | **BUG** (`CIPHER_CP54_MAX_ALLOCS=64`); PARTIAL (WA cap 16, KV-dedup 8 K pages adequate) | PARTIAL (process per tenant; VA waste) | MISSING (no per-tenant chain head) | MISSING |
| **R-D3** AUDIT sub-µs at 100 hot path  | N/A          | MISSING (AUDIT op registered priority 0 obs-only) | MISSING (no kmod-resident chain)     | N/A                                   | MISSING                       | N/A                                   |
| **R-D4** Telemetry across 100 no loss   | N/A          | N/A                                  | PARTIAL (per-PID utilization ioctl nr 6 exists, 10 Hz daemon) | N/A                              | N/A                           | **MISSING** (NVML is device-aggregate; no per-tenant power) |
| **R-D5** ARBITRATE/FAIRNESS fits budget | N/A          | N/A                                  | **BUG** (FAIRNESS stub only; no quota enforcement) | N/A                            | N/A                           | N/A                                   |
| **R-H1** Tenant-scoped weight kit       | N/A          | **BUG** (Marlin keyed by `weight_ptr` only; process-global) | PARTIAL (WA cap 16 < 100 unique models)         | N/A                                | N/A                           | N/A                                   |
| **R-H2** Model-aware actuator routing   | N/A          | MISSING (CLASSIFY ported, not hot-path wired; Koopman registry unkeyed) | N/A                                | N/A                                   | N/A                           | N/A                                   |
| **R-H3** Model-aware KV scope           | N/A          | N/A                                  | N/A                                   | **BUG** (KV-dedup hash content-only, no `(model, layer, head)` key) | N/A | N/A                                   |
| **R-H4** Per-tenant tokenizer           | N/A          | N/A                                  | N/A                                   | SUPPORTED (lives above CIPHER in vLLM; plugin doesn't assume) | N/A | N/A                                   |
| **R-H5** Weight-residence @ 100 models  | N/A          | MISSING (Marlin INT4 kit single-model; no tiered HBM/RAM/NVMe; no pooling) | PARTIAL (WA fd-custodian works ≤16)             | N/A                                  | N/A                           | N/A                                   |
| **R-I1** Per-tenant p99 vs noisy-neighbor | N/A         | N/A                                  | PARTIAL (SM-partition isolation 15-group; SHIELD not wired) | N/A                              | N/A                           | N/A                                   |
| **R-I2** Fair scheduling                | N/A          | N/A                                  | **BUG** (FAIRNESS field stub)         | N/A                                   | N/A                           | N/A                                   |
| **R-I3** Dynamic partition reassignment | N/A          | N/A                                  | PARTIAL (Track 3 DSM closed; capped at 64 allocs by CP 5.4) | N/A                          | N/A                           | N/A                                   |
| **R-I4** Per-tenant crypto receipts     | N/A          | MISSING (HMAC chain entries no `tenant_id` field) | MISSING (no kmod chain head)         | N/A                                   | MISSING                       | N/A                                   |
| **R-Dep1** LD_PRELOAD only at install   | SUPPORTED    | N/A                                  | SUPPORTED                             | SUPPORTED                             | N/A                           | N/A                                   |
| **R-Dep2** Tenant id propagation        | PARTIAL (env var `CIPHER_TENANT_NUM` only; no anti-collision guard) | N/A | PARTIAL (REGISTER_TENANT ioctl exists; no fork inheritance) | PARTIAL (env-var-driven) | N/A | N/A                                   |
| **R-Dep3** Per-tenant model registration | N/A         | MISSING (no model_id/model_hash registry) | MISSING (no kmod model-binding ABI) | N/A                                  | N/A                           | N/A                                   |
| **R-Dep4** HMAC chain export            | N/A          | N/A                                  | MISSING                               | N/A                                   | MISSING (export protocol unwritten) | N/A                            |

### 3.1 Engineering sketches for each BUG / MISSING cell

**G1 — CP 5.4 ALLOCATE table size (R-D2 × KMOD: BUG).**
Raise `CIPHER_CP54_MAX_ALLOCS` from 64 to ≥128 in `cipher_cp54_sched.c:123`. Audit the lookup loops (`cipher_cp54_sched.c:124–139` and the migration FSM body for any O(N) walks that need to remain ≤µs at the new size). Re-run CP 5.4 step 1.3b' POOL regression + Track 3 DSM SC1–SC6 at N=128. **LOC ~50, eng time ~1 day.** This is the single hard-stop.

**G2 — Weight-arena slot count (R-D2 × KMOD: PARTIAL; R-H5 × KMOD: PARTIAL).**
Raise `CIPHER_WA_MAX_ARENAS` (currently 16, per `cipher_weight_arena.c` via `cipher_ioctl.h:481`) to ≥100. v1.2.2 §8 R-C2 mentioned raising it to 32 (kmod 0.5); we need ≥100 for the 100-different-models product framing. Audit reaper performance at 100 arenas × 5 s cycle. **LOC ~30, eng time ~half day.** Pairs with G1.

**G3 — KV-dedup hash key scoping (R-H3 × KV: BUG).**
Re-key the kmod's xxhash64 lookup so the dedup index is `(model_uuid, layer_id, head_dim, dtype, content_hash)` instead of `content_hash` alone. The plugin (`cipher_vllm_kvdedup.py:138-150`) and the kmod side (`cipher_rt_kv_alloc.h:112`) both touch. The `tenant_id` field in `struct cipher_rt_kv_page_tag` (line 43) already exists and can be reused, plus we add `model_uuid` (`uuid_t` or 128-bit fnv) carried per-page. **LOC ~200 across kmod + plugin, eng time ~3 days.** Includes a teacher-forced KL gate on a synthetic 2-different-model N=2 test before re-enabling cross-process dedup at heterogeneous N>1. **Without this fix, B2-B different-model multi-tenant has a silent-correctness defect, not a perf issue.**

**G4 — Marlin tenant-scoped weight kit (R-H1 × ACTU: BUG; R-H5 × ACTU: MISSING).**
Refactor `cipher_rt_marlin_engine.cpp` weight cache from `weight_ptr` key to `(model_uuid, layer_id, K, N)`. Add a model-registration ABI (a new ioctl on `/dev/cipher` — call it `CIPHER_REGISTER_MODEL`, returns `model_uuid` — additive per the `cipher-abi-rule`). Threading: the cache must scale to ≥100 keys without contention; per-key shard or RCU. **LOC ~500 across engine + actuator + kmod + plugin, eng time ~5 days.** Risk: cubin recompile per (model, K, N) shape — at 100 distinct (model, K, N) tuples this stresses JIT cache; budget JIT amortization across `model_uuid` if shapes repeat (they will: e.g., all 7B Mistral-class share K=4096, N=4096 for o_proj). The CP 5.3 anchor `cipher-marlin-primary-ctx-pin` memory remains in force — Marlin is full-GPU-only; this gap does not change that constraint.

**G5 — VA pool per-tenant sizing (R-D2 × KV: PARTIAL).**
Either (a) shrink the default `va_gib=80` in `cipher_vllm_plugin/cipher_vllm_kv.py:58` to the per-model footprint (e.g., 8 GiB for 7B-class) and let the plugin compute from `hf_config`, or (b) move the VA reservation to the kmod weight-arena fd custodian (already implemented, just re-aim it as the VA backing store). Option (b) is the architecturally correct move — the kmod becomes the authoritative VA broker, the plugin requests only what it needs per model. **LOC ~150 (plugin) + ~80 (kmod broker ABI), eng time ~3 days.** Without this, N ≈ 12 is the practical concurrent-process ceiling on a 1 TB host.

**G6 — Kmod-resident AUDIT chain (R-D3 × KMOD: MISSING; R-I4 × KMOD: MISSING; R-Dep4 × KMOD: MISSING).**
Add a kmod-side per-tenant HMAC-SHA256 chain head. Storage: append-only per-tenant ring buffer in shared memory (mmap'd to userspace at registration time), HMAC accumulator in `struct cipher_pid_stats` (~64 B addition). Hot-path write rate at 100 tenants × decode (~80 tok/s/tenant × 8 actuator calls/token = 64 000 writes/s aggregate): each write is a sub-µs RING_WRITE (already scheduled in v1.2.2 W9-10 — this overlaps with that work). Per-tenant chain head with FNV-prefixed HMAC seeded at REGISTER_TENANT. **LOC ~400 across kmod + libcipher_rt audit shim, eng time ~5 days.** Pairs with v1.2.2 W7-W10 (COMMIT + RING_WRITE) — there is real *synergy* here, this is not a parallel track. The 100-agent goal needs the chain head, the v1.2.2 plan needs RING_WRITE; both close together.

**G7 — Per-tenant power attribution (R-D4 × TELE: MISSING).**
NVML does not expose per-tenant power at the 10 ms cadence. Three options: (a) live with NVML aggregate + per-tenant kernel-time accumulators (CIPHER kmod already provides via SUBMIT_LAUNCH_STATS / QUERY_FLOPS ioctls 7+12) and *attribute* power proportionally — the standard approach for shared-GPU billing; (b) sample `nvidia-smi pmon` at 1 Hz alongside NVML 100 Hz and back-fill (heuristic); (c) RAPL + per-tenant CPU energy + GPU split — out of scope. Recommend (a). **LOC ~250 in the bench harness's TelemetrySampler + reporter, eng time ~2 days.** Touches `bench_llm.py` at the harness level, not the substrate.

**G8 — CARBON / RECEIPT kmod-side state (R-I4 × KMOD: MISSING; R-Dep4 × BILL: MISSING).**
The may13 ports (`cipher_carbon.cpp`, `cipher_receipt.cpp`) are userspace and process-scoped; they need a kmod-resident store so per-tenant accumulators survive process crashes. Architecture: piggyback on the AUDIT chain shm (G6) — every CARBON / RECEIPT entry becomes a typed event in the ring buffer, the kmod just owns the ring. **LOC ~200 on top of G6, eng time ~2 days.** Not independent of G6; budget together.

**G9 — Fork inheritance for tenant identity (R-Dep2 × KMOD: PARTIAL).**
Child processes don't inherit `tenant_id` today. Fix: kmod kprobe on `do_fork` (or modern equivalent) that copies parent's `tenant_id` slot to child on fork. Caveat: only do this when parent has registered with anti-spoof clean; child still must call REGISTER_TENANT to attest. **LOC ~80, eng time ~half day.**

**G10 — Model-registration ABI (R-Dep3 × KMOD: MISSING).**
Add ioctl `CIPHER_REGISTER_MODEL(model_path, hash) → model_uuid`. Pairs with G4 (Marlin keying) and G3 (KV-dedup keying). **LOC ~150 in kmod + ~50 in plugin, eng time ~2 days.** Budget alongside G3 + G4.

**G11 — FAIRNESS table (R-D5 × KMOD: BUG; R-I2 × KMOD: BUG).**
Replace the `fairness_quota_remaining_pct` stub (`cipher_internal.h:230`) with a per-tenant quota ledger. Per-tenant SHM mmap'd to userspace for fast hot-path reads; quota refill via FAIRNESS kthread on a slow cadence (e.g., 10 ms). **LOC ~400, eng time ~4 days.** This is the FAIRNESS port from `cipher_rt_phase4/src/may13/cipher_fairness.cpp` (184 LOC) + `cipher_fairness_shm.cpp` (229 LOC) becoming hot-path-wired. v1.2.2 W2 / W4 nominally covers this — confirm.

**G12 — CLASSIFY / Koopman registry model-keying (R-H2 × ACTU: MISSING).**
The Koopman recipe registry currently keys on shape. Add model + layer to the key. Mostly mechanical: every cache lookup in `cipher_recipes.cpp` (522 LOC) and `cipher_kernel_table.cpp` (317 LOC) takes a model_uuid prefix. **LOC ~300, eng time ~3 days.** Pairs with G4, G10.

**G13 — Tiered weight residence (R-H5 × ACTU: MISSING).**
Decide product strategy: aggressive INT4 (15 7B models fit in HBM), tiered HBM/RAM/NVMe (more models, latency cost), or model-pooling (only K unique models, agents bind). The substrate has none of these wired today. INT4 aggressive is the lowest-risk first step (Marlin is already INT4 for prefill; extend to weight residence). **LOC ~600 + per-product design memo, eng time ~10 days minimum.** This is product strategy, not pure engineering.

---

## 4. Critical-path analysis (Part D)

### 4.1 Dependency DAG (BUG + MISSING cells)

```
G1 (CP54 64→128)  ──┐
                    ├──> R-D2 unblocked at 100
G2 (WA 16→100)  ────┘

G10 (RegisterModel ABI) ──┐
                          ├──> G3 (KV-dedup keying) ──┐
                          ├──> G4 (Marlin keying)    ─┤
                          └──> G12 (Recipes keying)  ─┴──> R-H1, R-H2, R-H3, R-H5 unblocked

G6 (kmod AUDIT chain) ──┬──> G8 (CARBON/RECEIPT)  ──> R-I4, R-Dep4 unblocked
                        └──> R-D3 unblocked

G5 (VA pool per-tenant) ──> R-D2 (host-VA dimension) unblocked

G7 (Per-tenant power attribution, harness-side) ──> R-D4 unblocked

G9 (Fork inheritance) ──> R-Dep2 fully closed

G11 (FAIRNESS hot-path) ──> R-D5, R-I2 unblocked

G13 (Weight residence strategy) ──> R-H5 fully resolved (product decision required first)
```

### 4.2 The blocking gap (D.2)

**The single blocking gap is G1 — raise `CIPHER_CP54_MAX_ALLOCS` from 64 to ≥128.**

Justification: every other gap can be worked around with single-tenant or 2-tenant demos. G1 is the only gap where the **65th LD_PRELOAD process** receives `-ENOMEM` from the kmod and cannot proceed. There is no userspace workaround; the kmod ledger is the cross-process broker and there is exactly one in the system. Until G1 closes, a 100-agent demonstration is *structurally infeasible* — you cannot stand up the 65th tenant at all.

(Adjacent observation: G1 was *not* in the v1.2.2 §8 risk register. §8 R-C2 mentions raising `weight_arena_slots` from 16 to 32 [G2], which is a different cap. G1 must be filed as a new entry in any v1.2.2 amendment.)

### 4.3 Fan-out gaps (D.3) — leverage points

- **G10 (CIPHER_REGISTER_MODEL ABI)** unlocks G3 + G4 + G12 simultaneously. Closing it first means three downstream BUG/MISSING items become straightforward keying changes rather than ABI work. **High leverage.**
- **G6 (kmod AUDIT chain)** unlocks G8 (CARBON/RECEIPT export piggyback) and closes R-D3 + R-I4 + R-Dep4. Also overlaps cleanly with v1.2.2 W9-10 RING_WRITE substrate (same hot-path budget, same shm-mmap pattern). **High leverage when paired with v1.2.2 work.**
- **G1 + G2 together** close R-D1 + R-D2 (KMOD column). Cheap (1.5 eng-days) and they share the kmod-ABI-bump pattern. Bundle.

### 4.4 Deferable gaps (D.4)

- **G9 (fork inheritance)** — most neocloud agentic deployments run one process per tenant without fork; inheritance is nice-to-have but not on the demo path. Defer to v1.5.
- **G13 (tiered weight residence)** — until product picks a strategy (aggressive INT4 vs tiered vs pooling), this is design work, not engineering. Defer until product decision. For the v1 100-agent demo, pick a small set of unique models (~5–10) per the pooling strategy, which is implementable on the substrate today without G13.
- **G11 (FAIRNESS hot-path)** — required for "100-agent fair-scheduling under bursting," but a v1 demo can run a homogeneous-load workload (no burst), which masks the gap. Surface but allow staged delivery.
- **G7 (per-tenant power attribution)** — required for billing-grade tok/W reporting, but a v1 *technical* demo can rely on aggregate tok/W and per-tenant kernel-time proxies (already available via SUBMIT_LAUNCH_STATS). Defer the harness-side correctness work.

**Not deferable:** G1, G2, G3 (correctness defect on the dispatch hot path), G4 (Marlin would error on cross-model weight cache), G5 (host-VA exhaustion is a hard limit), G6 (no billing receipts → no neocloud product), G10 (every other heterogeneity fix depends on it), G12 (CLASSIFY hot-path is meaningless at 100 different models without model keying).

---

## 5. V1.2.2 §7 timeline reconciliation (Part E)

### 5.1 Map of MISSING / BUG gaps → v1.2.2 W1-W14 schedule

V1.2.2 §7 schedule (confirmed by line citations in `CIPHER_REENGINEERING_PLAN.md`):

| Week | Planned deliverable | Line ref |
|------|---------------------|----------|
| W1 | Classifier port; LP-7 struct-collision fix; snapshot reserved-tail bump | L1217 |
| W2 | Hot-path classifier wire-up; LP-2 attn-trampoline refactor | L1218 |
| W3 | Dispatch routing goes live | L1219 |
| W4 | Observability integration | L1220 |
| W5 | KV-dedup live wire + v1 substrate consolidation | L1221 (per v1.2.2 amendment line 1209) |
| W6 | Reserved (TBD per Week 5 close) | L1222 |
| W7–8 | COMMIT primitive | L1223 |
| W9–10 | RING_WRITE substrate | L1224 |
| W11–12 | Koopman tier integration | L1225 |
| W13–14 | CP 5.5 headline benchmark | L1226 |

Gap-to-week mapping:

| Gap | v1.2.2 week | Closes / partially closes / not addressed |
|-----|-------------|-------------------------------------------|
| **G1** CP54 cap 64→128 | not scheduled | **NOT ADDRESSED** in v1.2.2 §7. §8 R-C2 (line 1548) addresses weight-arena slots only |
| **G2** WA cap 16→32 | implicitly W4-W5 per §8 R-C2 | partially closed (target is 32, product needs ≥100) |
| **G3** KV-dedup keying | not scheduled | **NOT ADDRESSED**. v1.2.2 plan assumes same-model dedup |
| **G4** Marlin tenant scoping | not scheduled | **NOT ADDRESSED**. R-C3 (line 1549) restricts Marlin to full-GPU lane; doesn't address multi-model |
| **G5** Per-tenant VA pool | not scheduled | **NOT ADDRESSED** |
| **G6** Kmod-resident AUDIT chain | overlaps W9-W10 (RING_WRITE) | **PARTIALLY CLOSED** — RING_WRITE provides the shm pattern; chain head is separate work bundled into the same shm-mmap |
| **G7** Per-tenant power attribution | not scheduled | **NOT ADDRESSED** (harness-side) |
| **G8** CARBON/RECEIPT kmod state | partially W4 (observability integration) | **PARTIALLY CLOSED** — userspace ports land, kmod store is separate |
| **G9** Fork inheritance | not scheduled | **NOT ADDRESSED** |
| **G10** Model-registration ABI | not scheduled | **NOT ADDRESSED** |
| **G11** FAIRNESS hot-path | overlaps W4 | **PARTIALLY CLOSED** in W4 observability integration |
| **G12** Koopman registry model-keying | W11-12 | **PARTIALLY CLOSED** if scoped to include model key; the plan text does not specify model-keying explicitly |
| **G13** Tiered weight residence | not scheduled | **NOT ADDRESSED** |

### 5.2 100-agent-heterogeneous gaps NOT in v1.2.2

**Explicitly unscheduled by v1.2.2 §7:** G1, G3, G4, G5, G7, G9, G10, G13. That is **8 of 13** gaps for the 100-different-models product framing. Several are 1- to 3-day items; the cumulative effort is on the order of **~25-30 engineering days** for the structural + correctness gaps (G1+G2+G3+G4+G5+G10), plus product decisions (G13) and observability polish (G7, G9, G11).

### 5.3 V1.2.2-scheduled work that does NOT directly serve the 100-agent-heterogeneous-model goal

This is not "wasted work" — it serves other goals and the wider re-engineering plan. The point is to surface which scheduled work is *orthogonal* to the 100-agent product so the user can decide on re-sequencing.

- **W11-12 Koopman tier (3 ops: REMEMBER, SPECULATE, ADAPT)** — gates the **O(1) substitution** goal (the dispatch tier's learning-based recipe selection). Critical for v2; does **not** gate any 100-agent heterogeneous-model gap directly. Without G12 (model-keying) it doesn't even fire correctly for the heterogeneous case.
- **W13-14 CP 5.5 benchmark** — verifies the *v1.2.2 plan's* headline (workload-class heterogeneous), not the 100-different-models product headline. Useful as a substrate certification; not the same demonstration.
- Parts of W4 observability integration — STRAGGLER, SAMPLE, full TRACE export are valuable for v1 production observability but do not gate the 100-agent demo.

### 5.4 Timeline-adjustment recommendation

Three options. **Recommended is (b).**

**(a) Keep §7 unchanged, accept 100-agent demo post-CP-5.5.**
- Weeks added: 0
- Engineering risk: low for §7 deliverables; the 100-agent demo lands after week 14 + ~25-30 days of structural-gap work = effectively week 19-20.
- Headline at end of §7: workload-class heterogeneous 100-tenant benchmark on (likely) one model.
- **Verdict:** safe but the 100-agent-different-models product story stays unsupported through W14.

**(b) Re-sequence §7 to front-load structural gaps, defer Koopman tier.**
- Replace W6 (reserved) with G1+G2 (cap bumps, 1.5 days).
- Move G10 (model-registration ABI) into W7 alongside COMMIT primitive; they share the kmod-ABI-bump pattern.
- Move G3 + G4 (KV-dedup keying + Marlin tenant scoping) into W9-W10 alongside RING_WRITE; they piggyback on the shm-mmap work.
- Move G5 (per-tenant VA pool) into W11 in place of Koopman tier *start*. Defer Koopman tier ops to v2 (per the §2 tier classification, they were already marked "v2 deferred" in v1.2 §2.4 — bringing them into v1 was the v1.2.2 ADJUDICATION A1 move, which is reversible).
- W12 closes G11 (FAIRNESS hot-path) — already on the v1.2.2 plan, just reframed as part of the 100-agent path.
- W13-14 becomes the **100-tenant + 5-models heterogeneous benchmark** (a hybrid of v1.2.2's CP 5.5 workload-class heterogeneity *and* B2-B's different-model heterogeneity).
- Weeks added: 0 (re-sequence, not extend)
- Engineering risk: low — every replacement task is documented sketch + 1-5 days; Koopman defer is reversible.
- Headline at end of W14: **100 tenants × 5 different models × workload-class heterogeneous mix on the unified runtime.**

**(c) Augment §7 with new tracks, keep existing tracks intact.**
- Add weeks W15-W17 to absorb G1, G2, G3, G4, G5, G6, G10, G12 (the unscheduled structural + correctness items).
- Keep Koopman tier at W11-12.
- Keep CP 5.5 at W13-14.
- Weeks added: 3
- Engineering risk: moderate — pushing the headline demo two phases out is calendar-expensive.
- Headline at end of W17: same as (b) but two weeks later.
- **Verdict:** preserves the full v1.2.2 §7 unchanged, at the cost of three weeks of headline delay.

**Recommendation: (b).** Koopman tier was already a v2 candidate before the v1.2.2 ADJUDICATION A1 pulled it in; reverting it for the 100-agent product story is consistent with the architectural classification, not a retreat. The re-sequence costs zero weeks and unblocks the heterogeneous-model demo within the existing 14-week envelope.

---

## 6. May 13 POC reproduction question (Part F)

### 6.1 Does the May 13 POC reproduce on V1.2.2 substrate today?

**No — not without a reconstruction effort.** Citations:

- The May 13 POC's entry point is `/home/ubuntu/cipher-may13-evidence/multi_tenant_poc.py:31–100`. It spawns N child processes (`multi_tenant_poc.py:60–68`), each running `multi_tenant_child.py` with `LD_PRELOAD=libcipher_hook.so` and `CIPHER_TENANT_ID=i` env.
- The env vars set: `CIPHER_FP8_COMPUTE`, `CIPHER_SUBSTITUTE_V2`, `CIPHER_FUSION_KERNELS`, `CIPHER_NCCL_V4` (all `=on`).
- The dispatch surface used: **op31-prod** (`CIPHER_SUBSTITUTE_V2` + fusion kernels + NCCL v4, per the inline notes in `CIPHER_MULTITENANT_POC.md:72`).
- The Week-5 consolidation supersedes that dispatch path. The current substrate uses `CIPHER_KV_ALLOC`, `CIPHER_KVDEDUP`, `CIPHER_OFFLOAD`, `CIPHER_TENANT_NUM`, `CIPHER_VOLT_ENABLED`, `CIPHER_MARLIN` env vars. The op31-prod env vars are **not consumed** by the current `libcipher_rt.so` (md5 `259ac994`).
- The library name change: POC loads `libcipher_hook.so` (op31-prod era); current name is `libcipher_rt.so`. Direct `LD_PRELOAD=libcipher_hook.so` from the POC will load nothing CIPHER (or fail-open if a stale `libcipher_hook.so` is still on disk from a prior pod state).
- The dispatch route: POC's `cublasGemmEx` interception is preserved (the GOT-patch pattern survived the Week-5 consolidation, `cipher_inject.c:50–56`). But the actuators that fired in POC (FP8 compute, fusion kernels, NCCL v4) are **not registered** in the current matmul / attention registries — see B.2 / §2.3.

**Net:** the *interception surface* is intact (LD_PRELOAD into `cublasGemmEx` still works), but the *actuator behavior* the POC depended on has been replaced. The POC's headline ("7 problems addressed at 15 tenants on Llama-3.2-1B") cannot reproduce by re-running `multi_tenant_poc.py` against today's substrate — the actuators that produced those numbers are not in `libcipher_rt.so` `259ac994`.

### 6.2 Cost of reproduction on V1.2.2 substrate

This is a **reconstruction**, not a verification. Required work:

1. Port the POC's actuator behaviors (FP8 compute, fusion kernels, NCCL v4 collective patches) into the current `cipher_rt_phase4/` registries. **LOC: ~800-1500** depending on whether the POC actuators land as new entries in the existing 16-slot matmul/attention registries (the simpler option) or as a parallel substrate (matched to v1.2.2 §4's classifier-substrate addition).
2. Map the POC's env vars to the current substrate's env-var contract (CIPHER_FP8_COMPUTE → CIPHER_MARLIN-style env-gate). **LOC: ~50.**
3. Re-write the multi-tenant launcher (`multi_tenant_poc.py`) to set the new env vars and use the new library name. **LOC: ~100.**
4. Re-run the 7-problem evaluation suite on the v1.2.2 substrate at 15 tenants × Llama-3.2-1B. **GPU-time: ~30-60 min.**
5. Decide which of the POC's 7 problems are still on the v1.2.2 critical path and which were *replaced* by structural primitives (e.g., the POC's per-tenant KV-cache trick is replaced by W5's KV-dedup + buffer-ownership hook; that section reproduces *better*, not the same).

**Total reconstruction time estimate: 6-10 engineering days** + benchmark GPU time. Not trivial; cannot be the first sub-task of the implementation sequence unless the user wants a substrate-equivalence checkpoint before pushing further. Recommendation in §7.

---

## 7. Recommendation (Part G)

### 7.1 Strategic read — is v1.2.2 on the critical path?

**V1.2.2 §7 is adjacent to the 100-agent heterogeneous-model product, not on its critical path.** The substrate that the v1.2.2 plan builds (COMMIT + RING_WRITE + Koopman tier + workload-class-heterogeneous CP 5.5) is structurally compatible with the 100-different-models goal but does not directly close any of the structural + correctness gaps that gate that goal (G1, G3, G4, G5, G10 in particular). Of the eight unscheduled gaps in §5.2, six (G1, G3, G4, G5, G6 partial, G10) are **prerequisites** for a credible 100-different-models demo. Without them the substrate ships an architecturally elegant 100-tenant workload-class-heterogeneous demonstration that cannot honestly call itself "100 different models."

The v1.2.2 plan is internally coherent — it builds the right substrate for the right reasons. It just was not framed against the "100 agents × 100 different models" product story. The NEOCLOUD audit (B2-A vs B2-B vs B3) recognizes this implicitly; the v1.2.2 plan does not foreground it.

The blocking item is **G1 (CP 5.4 ALLOCATE table size = 64)**. Until that flips, nothing in the 100-tenant story stands up.

### 7.2 Recommended next 2-3 prompts

In execution order:

**Prompt 1 — Close G1 + G2 (the kmod-cap bumps).**
*"Raise `CIPHER_CP54_MAX_ALLOCS` from 64 to 128 in `cipher_kmod/cipher_cp54_sched.c:123`. Raise `CIPHER_WA_MAX_ARENAS` from 16 to 100 in `cipher_kmod/cipher_weight_arena.c` (and the matching constant in `cipher_ioctl.h:481`). Audit every `for (i = 0; i < CIPHER_CP54_MAX_ALLOCS; i++)` loop body for µs-level cost at the new size. Re-run CP 5.4 Step 1.3b' POOL regression at N=128 and Track 3 DSM SC1-SC6 at the new cap. Land as a kmod 0.5 bump (`cipher_kmod` tag) with the new srcversion recorded in memory."*
**Justification:** unblocks every density requirement. Eng cost ≤1.5 days. Single change is small enough to be reviewed in one sitting. No upstream surprises.

**Prompt 2 — KV-dedup model-aware keying (G3) + CIPHER_REGISTER_MODEL ABI (G10).**
*"Add a new ioctl `CIPHER_REGISTER_MODEL(model_path, hf_config_hash) → model_uuid` at NR 27 (additive per the cipher-abi-rule). Wire the plugin to call it at engine init, propagate `model_uuid` into `struct cipher_rt_kv_page_tag`, and re-key the kmod's KV-dedup xxhash64 lookup to (`model_uuid`, `layer_id`, `head_dim`, `dtype`, `content_hash`). Land with a teacher-forced KL gate on a synthetic Mistral-7B + Llama-3.1-8B N=2 cross-tenant test verifying that the dedup engine refuses to share pages across non-matching `(model, layer, head)` triples. Update `cipher_rt_kv_alloc.h:112` and `cipher_vllm_kvdedup.py:138-150`."*
**Justification:** closes the heterogeneous-model **correctness defect**. Until this lands, B2-B different-model multi-tenant has a silent attention-corruption surface. High-leverage: G10 also unblocks G4 and G12.

**Prompt 3 — Marlin tenant-scoped weight kit (G4) + Koopman registry model-keying (G12).**
*"Refactor `cipher_rt_marlin_engine.cpp` weight cache from `weight_ptr` key to `(model_uuid, layer_id, K, N)`. Refactor `cipher_rt_phase4/src/may13/cipher_recipes.cpp` and `cipher_kernel_table.cpp` lookup keys to include `model_uuid` prefix. Verify Marlin's full-GPU lane still respects the cipher-marlin-primary-ctx-pin contract under multiple model UUIDs. Re-bench on the Week 6 harness at 5 distinct models, B=1 each, single-process to verify the registry survives multi-model + that JIT cache amortization is acceptable."*
**Justification:** closes the multi-model actuator-routing path. Without it, even with G1 + G3 closed, you have a 100-tenant single-model demo, not a heterogeneous-model demo.

After these three prompts, the audit's `BUG`-classified cells flip to `SUPPORTED` (G1, G3, G4) or `PARTIAL` (G2 — still under product target if you want ≥100 unique arenas instead of 100). G5 (VA pool), G6 (kmod AUDIT chain), G7 (per-tenant power), G11 (FAIRNESS hot-path) follow as a coherent subsequent batch — these align cleanly with v1.2.2 W7-W11 timeline, so the user can decide between bundling them as the "v1.2.2 §7 re-sequence option (b)" of §5.4 or running them as parallel tracks.

### 7.3 Risks to flag (substrate assumptions that could invalidate the recommendation)

| Risk | Verification approach |
|------|------------------------|
| **R1** Raising `CIPHER_CP54_MAX_ALLOCS` to 128 may expose O(N²) loop bodies in the migration-FSM evaluator (`cp54_eval_migration`) that pass at N=64 but stall at N=128 | Grep `cipher_cp54_sched.c` for nested loops over the alloc table; benchmark `cp54_eval_migration` at N=128 with synthetic allocations; target ≤500 µs per call |
| **R2** The kmod's PID hashtable (1024 buckets) may collide significantly enough at N=100 to violate the sub-µs lookup target in `cipher_tenant_snapshot.c:366` | Synthetic PID-collision stress test; if collision-chain depth >4 at N=100, increase hashtable size to 4096 in `cipher_internal.h:73-74` (one-line change) |
| **R3** KV-dedup model-keying may inflate per-page xxhash64 cost from O(1) lookup to multi-key trie walk, breaking the dedup hot-path budget | Microbenchmark the new lookup vs. old at 8 192 keys × 100 lookups/s; if >1 µs per lookup, switch to a hash-of-hashes scheme keyed on `model_uuid` first |
| **R4** Marlin tenant-scoped cache may push the cubin JIT cache past the CUDA driver's internal limit at 100 unique (model_uuid, K, N) tuples, causing JIT thrash on tenant cold-start | At 5 distinct models × Mistral/Qwen/Llama shape sets, count unique (K, N) tuples; if >32 unique cubins per process, switch to a shared cubin keyed on (K, N) with model_uuid only in the weight cache |
| **R5** The May 13 POC's "7 problems" may not all map cleanly to v1.2.2 substrate primitives — some may have been *replaced* rather than ported (e.g., per-tenant KV-cache trick → buffer-ownership hook + dedup) | Cross-walk the POC's 7 problems against `CIPHER_PLAN_EXECUTIVE_SUMMARY.md` §3 measurement table to identify which still need direct reconstruction vs. which the current substrate covers better |
| **R6** The 100-different-models product framing may itself be wrong for v1 — the actual neocloud product might be **N customers × 1-5 model SKUs**, in which case G3/G4/G10 are still required but G13 (tiered residence) is far less urgent | Cross-check against `NEOCLOUD_SUBSTRATE_AUDIT.md` B1/B2/B3 categories with the user; explicitly ask whether "100 agents × 100 different models" or "100 agents × ≤10 model SKUs" is the v1 product |

---

## 8. Appendix: file:line citations

Every claim in §2 and §3 traces to one of these.

### 8.1 kmod (`/home/ubuntu/cipher_kmod/`, HEAD `2fc70c3`)
- `cipher_cp54_sched.c:123` — `CIPHER_CP54_MAX_ALLOCS = 64` (the binding cap)
- `cipher_cp54_sched.c:118-139` — CP 5.4 SM-arbitration ledger state
- `cipher_cp54_sched.c:158` — ALLOCATE under `cipher_cp54_lock` mutex
- `cipher_cp54_sched.c:74-80` — per-group atomic encoding (free/OWNED/RESERVED)
- `cipher_cp54_sched.c:365-414` — Track 3 DSM FSM
- `cipher_cp54_sched.c:383-398` — `cp54_commit_migration`
- `cipher_cp54_sched.c:402-414` — `cp54_abort_migration`
- `cipher_cp54_sched.c:445+` — `cp54_eval_migration`
- `cipher_cp54_sched.c:93` — `CIPHER_CP54_MIG_TIMEOUT_NS = 30 s`
- `cipher_cp54_sched.c:702-804` — DSM ioctl handlers (nrs 16-20)
- `cipher_internal.h:73-74` — PID hashtable 1024 buckets
- `cipher_internal.h:32-35` — 24 NV_ESC slots + 2 aggregate
- `cipher_internal.h:172-234` — `struct cipher_pid_stats` definition
- `cipher_internal.h:230` — `fairness_quota_remaining_pct` stub
- `cipher_internal.h:207-233` — readers-tolerate-torn-reads doc
- `cipher_probe.c:41-76` — `cipher_pid_get_or_create`
- `cipher_probe.c:184-221` — `do_exit` kprobe + RCU reaper
- `cipher_dev.c:37-40` — `cipher_dev_open` (no-op)
- `cipher_dev.c:58-96` — `CIPHER_REGISTER_TENANT` handler
- `cipher_dev.c:70-71` — anti-spoof check
- `cipher_dev.c:85-93` — hashtable upsert under lock
- `cipher_dev.c:244-297` — ioctl dispatch
- `cipher_state_updater.c:132-133` — FAIRNESS stub default 100%
- `cipher_weight_arena.c:53` — `CIPHER_WA_REAP_SECS = 5`
- `cipher_weight_arena.c:68` — `cipher_wa_lock` mutex
- `cipher_weight_arena.c:121` — `wa_release` fput
- `cipher_weight_arena.c:128-152` — periodic reaper
- `cipher_weight_arena.c:157-214` — `ARENA_REGISTER` handler
- `cipher_weight_arena.c:220-277` — arena import path
- `cipher_kvdedup.c:1-35` — `/dev/cipher_kvdedup` device
- `cipher_kvdedup.c:60` — `kvd_lock` mutex
- `cipher_kvdedup.c:143-200+` — KVDEDUP_PUT handler
- `cipher_kvdedup.h:82` — `CIPHER_KVDEDUP_MAX_PER_TENANT = 8192`
- `cipher_ioctl.h:481` — `CIPHER_WA_MAX_ARENAS = 16` (referenced from cipher_weight_arena.c)
- `cipher_ioctl.h:169` — snapshot tail field for fairness_quota_remaining_pct
- `cipher_tenant_snapshot.c:366` — sub-µs RCU snapshot path

### 8.2 libcipher_rt (`/home/ubuntu/cipher_rt_phase4/`, HEAD `ec0e005`)
- `cipher_inject.c:32, 67` — `pthread_once` init body
- `cipher_inject.c:37` — `cipher_v2_tenant_register()` call
- `cipher_inject.c:44-47` — Marlin registration at init
- `cipher_inject.c:50-56` — GOT-patch registration (cublasGemmEx + SDPA mangled)
- `cipher_inject.c:64-76` — `InitializeInjection` / `InitializeInjection2`
- `cipher_rt_volt.c:54-63` — batch-to-MHz hardcoded
- `cipher_rt_volt.c:65-81` — single nvmlDevice_t per process
- `cipher_rt_volt.c:83-102` — NVML `dlopen` once
- `cipher_rt_volt.c:203-204` — `CIPHER_VOLT_ENABLED` env
- `cipher_rt_marlin_actuator.c:51` — `STABILITY_THRESHOLD = 4`
- `cipher_rt_marlin_actuator.c:53-57` — process-global atomic counters
- `cipher_rt_marlin_actuator.c:131-146` — gate constraints
- `cipher_rt_marlin_actuator.c:150-169` — weight observation + quantize path
- `cipher_rt_marlin_actuator.c:157-158` — quantize_repack call (keyed by `weight_ptr`)
- `cipher_rt_marlin_actuator.c:197` — priority-10 registration
- `cipher_rt_marlin_actuator.c:203-214` — `CIPHER_MARLIN` env gate
- `cipher_rt_marlin_actuator.c:237-242` — global counters
- `cipher_rt_matmul_dispatch.c:15-29` — 16-slot registry
- `cipher_rt_matmul_dispatch.c:52-82` — insertion sort
- `cipher_rt_matmul_dispatch.c:70-74` — priority-sorted insertion (lower runs first)
- `cipher_rt_matmul_dispatch.c:93-99` — append-only zero-lock hot path
- `cipher_rt_matmul_dispatch.c:118-123` — global counters
- `cipher_rt_matmul_dispatch.h:79` — priority int field
- `cipher_rt_attn_dispatch.cpp:50-55` — three mangled SDPA call symbols
- `cipher_rt_attn_dispatch.cpp:57-68` — 16-slot registry
- `cipher_rt_attn_dispatch.cpp:218-226` — sort
- `cipher_rt_attn_dispatch.cpp:258-268` — per-backend counters
- `cipher_rt_attn_dispatch.cpp:445-455` — SDPA registration
- `cipher_rt_attn_dispatch.h:110` — priority int field
- `cipher_rt_audit.h:1-20` — AUDIT op surface
- `cipher_rt_audit.h:14, 59` — ~50 ns HMAC inline
- `cipher_rt_audit.h:43-52` — entry struct (no tenant_id)
- `cipher_rt_audit.h:46` — `timestamp_delta` field
- `cipher_rt_audit.c:132-137` — AUDIT registers priority 0 on both
- `cipher_rt_tenant.cpp:65` — `__thread g_tls_cache`
- `cipher_rt_tenant.cpp:78` — `my_tid()` = `gettid()`
- `cipher_rt_tenant.cpp:83-94, 109-116` — lazy `/dev/cipher` open
- `cipher_rt_tenant.cpp:100-103, 139-159` — `/proc/cipher/stats` snapshot path
- `cipher_rt_tenant.cpp:221-232` — TLS refresh
- `cipher_rt_tenant.cpp:225-226` — ioctl by tid
- `cipher_rt_tenant.cpp:235` — `cipher_rt_tenant_cached()`
- `cipher_rt_tenant.cpp:237-239` — TLS snapshot return
- `cipher_rt_tenant.h:9, 14` — ~10 ns TLS read; ~5 µs ioctl miss
- `cipher_kv_bridge.cpp:43, 51-64` — `struct cipher_rt_kv_page_tag` with `tenant_id`
- `cipher_rt_kv_alloc.c:54` — single per-process slab mutex
- `cipher_rt_kv_alloc.h:112` — xxhash64 over 2 MiB body
- `cipher_rt_kv_alloc.h:21, 64` — T4.6.3 / T4.6.4 reserved
- `cipher_rt_kv_alloc.h:149-150` — `cipher_rt_kv_dedup_alias` surface
- `cipher_rt_got_patch.c:52-53` — `g_targets[16]`
- `cipher_rt_got_patch.c:58-76` — `cipher_rt_got_register`
- `cipher_rt_got_patch.c:114-122` — RELRO mprotect
- `cipher_rt_got_patch.c:186-204` — GOT slot rewrite
- `cipher_rt_got_patch.c:212` — `cipher_rt_got_patch_apply`
- `cipher_rt_got_patch.c:217` — `dl_iterate_phdr`
- `cipher_rt_got_patch.c:222` — `cipher_rt_got_patch_init`
- `cipher_rt_phase4/src/may13/cipher_dispatch.cpp:25-45` — Koopman weak ref + EDMD comment
- `cipher_rt_phase4/src/may13/cipher_dispatch.cpp:38` — weak `cipher_edmd_live_collect`

### 8.3 vLLM plugin (`/home/ubuntu/cipher_vllm_plugin/`)
- `cipher_vllm_kv.py:46` — `CIPHER_TENANT_NUM` env (defaults 0)
- `cipher_vllm_kv.py:50-62` — `cipher_kv_bridge.init(va_gib * GiB)`
- `cipher_vllm_kv.py:58` — `va_gib = 80` default
- `cipher_vllm_kv.py:87` — `cipher_kv_bridge.vmm_zeros` with tenant_id
- `cipher_vllm_kvdedup.py:62` — `CIPHER_TENANT_NUM` env
- `cipher_vllm_kvdedup.py:138-150` — `cipher_rt_kv_dedup_alias` per 2 MiB page
- `cipher_kv_offload.py:72-102` — `bind_kv_caches` block-dim scan
- `cipher_kv_offload.py:127` — `torch.index_select` gather
- `cipher_kv_offload.py:160` — shape-incompatibility check missing

### 8.4 v1.2.2 plan + supporting docs (`/home/ubuntu/cipher-fusion-evidence/`)
- `CIPHER_REENGINEERING_PLAN.md:1197` — §7 INTEGRATION SEQUENCE
- `CIPHER_REENGINEERING_PLAN.md:1209` — v1.2.2 amendment moving W5 to KV-dedup
- `CIPHER_REENGINEERING_PLAN.md:1215-1226` — week-by-week table
- `CIPHER_REENGINEERING_PLAN.md:1515` — CP 5.5 100-tenant heterogeneous
- `CIPHER_REENGINEERING_PLAN.md:1521` — workload mix 5+80+15
- `CIPHER_REENGINEERING_PLAN.md:1527-1530` — verification gates
- `CIPHER_REENGINEERING_PLAN.md:1532` — `/proc/cipher/op_status` requirement
- `CIPHER_REENGINEERING_PLAN.md:1548` — §8 R-C2 weight-arena slot mitigation
- `CIPHER_REENGINEERING_PLAN.md:1549` — §8 R-C3 Marlin partitioning limit
- `CIPHER_REENGINEERING_PLAN.md:1593` — §8 R-W7.2 N=100 contention risk
- `CIPHER_REENGINEERING_PLAN.md:1726-1748` — V1.2.2 AUDIT TRAIL (ADJUDICATIONS A1, A2, A3)
- `CIPHER_REENGINEERING_PLAN.md:1735` — Koopman tier into v1 ADJUDICATION A1
- `CIPHER_WORKLOAD_ARCHITECTURE.md:38` — Class D definition
- `CIPHER_WORKLOAD_ARCHITECTURE.md:62` — "bursty, heterogeneous control flow"
- `CIPHER_WORKLOAD_ARCHITECTURE.md:127` — 100 agents × 4K system prompt case study
- `NEOCLOUD_SUBSTRATE_AUDIT.md:12-14` — B1/B2-A/B2-B/B3 readiness
- `NEOCLOUD_SUBSTRATE_AUDIT.md:321` — VOLT is GPU-global
- `NEOCLOUD_SUBSTRATE_AUDIT.md:482-500` — B2-A details
- `NEOCLOUD_SUBSTRATE_AUDIT.md:501-520` — B2-B different-model lifts
- `NEOCLOUD_SUBSTRATE_AUDIT.md:503` — B2-B Mistral+TinyLlama scenario
- `NEOCLOUD_SUBSTRATE_AUDIT.md:508-509` — ~0% cross-model dedup hits
- `NEOCLOUD_SUBSTRATE_AUDIT.md:517-518` — B2-B lifts fired
- `NEOCLOUD_SUBSTRATE_AUDIT.md:547-563` — B3 deferred to CP 5.5
- `WEEK_6_BENCH_HARNESS_REWRITE.md:177` — GPU-only power limitation

### 8.5 May 13 POC (`/home/ubuntu/cipher-may13-evidence/`)
- `CIPHER_MULTITENANT_POC.md:38` — `LD_PRELOAD=libcipher_hook.so`
- `CIPHER_MULTITENANT_POC.md:72` — op31-prod dispatch surface
- `CIPHER_MULTITENANT_POC.md:72-77` — env-var contract (`CIPHER_FP8_COMPUTE`, `CIPHER_SUBSTITUTE_V2`, `CIPHER_FUSION_KERNELS`, `CIPHER_NCCL_V4`)
- `multi_tenant_poc.py:31-100` — POC entry point
- `multi_tenant_poc.py:60-68` — child-process spawn loop
- `src/cipher_edmd_live.cpp` — Koopman EDMD live runtime (no caller in current substrate)

---

**End of audit.** No source-tree changes were made.
