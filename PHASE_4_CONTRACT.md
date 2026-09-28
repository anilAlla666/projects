# Phase 4 — Kernel-Internal Tenant Snapshot Contract (T4.0.2)

**Date:** 2026-05-13
**Target module:** cipher_kmod 0.4.0 (Phase 3 baseline = 0.3.1, srcversion B1AF5E2A)
**Status:** SPEC — header additions are DRAFT in this document. Live `cipher_internal.h` is NOT modified by T4.0.2. Application happens in T4.1.2 as part of bumping cipher_kmod to 0.4.0.

## Design principles

1. **ABI-additive only.** Phase 3 ioctls 1/5/6/7 unchanged. New ioctl nr 8 (CIPHER_GET_TENANT_SNAPSHOT). Reserved nrs 2/3/4 still `-ENOSYS`.
2. **Three-identity bridge.** PHASE_4_OP_AUDIT.md Section "cipher_fairness tenant logic" found three tenant conventions in active use: Phase 3's `char tenant_id[64]` string, FAIRNESS's 64-bit `session_fp`, FAIRNESS_SHM's 32-bit `tenant_id`. The snapshot holds all three so each subsystem reads what it needs.
3. **RCU read-side, lock-free writes.** WRITE_ONCE per independent field. Snapshot reads are single-bucket walks keyed on `current->pid`.
4. **Cache-line aligned.** Target 384 bytes = 6 × 64-byte cache lines. Reserved padding for forward compat.
5. **Latency budget.** `cipher_get_current_tenant_snapshot()` must complete in < 200 ns on H100 host CPU (measured via kprobes benchmark in T4.1.9).
6. **EXPORT_SYMBOL_GPL** on the three lookup functions — enables future kernel-space `cipher_rt_km` module (Phase 6 substrate).

## Struct definition (draft for cipher_internal.h)

```c
/* Phase 4 — per-tenant snapshot contract.
 *
 * Read via cipher_get_current_tenant_snapshot() / by_id() / enumerate().
 * Populated by cipher_state_updater kthread (1 kHz) from:
 *   - cipher_pid_stats (Phase 3 telemetry: sm_util, mem_util, launches)
 *   - cipher_gpu_state (Phase 3 device-wide: clock, temp, power)
 *   - Phase 4 actuator write-back paths (sm_partition_mask, fusion_recipe_id)
 *
 * Reader contract: must hold rcu_read_lock() across access. Returned pointer
 * valid only until next rcu_read_unlock(); copy fields out for longer use.
 */
struct cipher_tenant_snapshot {
	/* --- Identity (all three for cross-subsystem compat) --- */
	char     tenant_id_str[CIPHER_TENANT_ID_LEN];  /* Phase 3 string id */
	__u64    tenant_session_fp;                     /* FAIRNESS: FNV-64(tenant_id_str) */
	__u32    tenant_handle_u32;                     /* FAIRNESS_SHM: murmur32(tenant_id) */
	__u32    pid;                                   /* leader-thread LWP */
	__u32    tgid;
	__u32    _pad_identity;                         /* align to 8 */

	/* --- Live telemetry from Phase 3 ioctls (mirrored from cipher_pid_stats) --- */
	__u32    sm_util_pct;                           /* PROCESS_UTIL */
	__u32    mem_util_pct;                          /* PROCESS_UTIL */
	__u64    launches_total;                        /* LAUNCH_STATS */
	__u64    grid_ops_total;                        /* LAUNCH_STATS */
	__u64    ioctls_total;                          /* rollup of nvidia_unlocked_ioctl probes */

	/* --- Derived: SM-partition cluster (P4.2) --- */
	__u32    sm_partition_mask;                     /* bitmask over 33 partition slots */
	__u32    sm_partition_count;                    /* popcount of mask */

	/* --- Derived: DVFS/thermal cluster (P4.3) --- */
	__u32    thermal_headroom_pct;                  /* (90 - temp_c) / 90 * 100 */
	__u32    power_headroom_w;                      /* 700 - power_mw / 1000 */
	__u32    sustained_clock_mhz;                   /* EMA of gpu_state.sm_clock */
	__u32    voltage_envelope_mv;                   /* derived from clock+power */

	/* --- Derived: L2 cluster (P4.4) --- */
	__u32    l2_residency_kb;                       /* sum of tenant hot-region sizes */
	__u32    hot_region_count;                      /* distinct hot pointers */
	__u64    predicted_hot_regions[8];              /* top-8 pointer hashes from PREDICT */

	/* --- Derived: KV cluster (P4.6) --- */
	__u32    kv_cache_size_mb;                      /* KV_REDIRECT registered regions */
	__u32    kv_compression_ratio_pct;              /* KV_COMPRESS dictionary */

	/* --- Derived: Weight cluster (P4.5) --- */
	__u32    weight_dedup_savings_mb;               /* WEIGHT_SHARE cross-tenant savings */
	__u32    weight_content_hash_count;             /* per-tenant slot count */

	/* --- Derived: Fusion cluster (P4.7) --- */
	__u32    fusion_recipe_id;                      /* last RECIPES match */
	__u32    fusion_eligible_flag;                  /* FUSION/FLOW_PATTERNS detection */

	/* --- Derived: Agentic / SLO cluster (P4.7) --- */
	__u32    session_band;                          /* SENSE: 0=HUMAN 1=AGENT 2=BATCH */
	__u32    slo_priority;                          /* SHIELD latency budget class */
	__u32    fairness_quota_remaining_pct;          /* FAIRNESS table */
	__u32    _pad_agentic;

	/* --- Derived: Attention cluster (P4.6 secondary) --- */
	__u32    graph_capture_state;                   /* GRAPH state machine */
	__u32    koopman_substitution_eligibility;      /* ATTN_KOOPMAN per-tenant flag */

	/* --- Freshness --- */
	__u64    snapshot_jiffies;                      /* when state_updater last touched */
	__u32    reserved[16];                          /* forward compat */
};

/* Sanity bound: 384 B target (6 cache lines on x86-64). */
static_assert(sizeof(struct cipher_tenant_snapshot) <= 384,
              "cipher_tenant_snapshot exceeds 384B target");
```

**Size computation (best estimate before build):**

| Bucket | Bytes |
|---|---|
| tenant_id_str[64] | 64 |
| tenant_session_fp + handle_u32 + pid + tgid + _pad_identity | 24 |
| Live telemetry (sm_util u32, mem_util u32, launches u64, grid_ops u64, ioctls u64) | 32 |
| SM-partition (mask u32 + count u32) | 8 |
| DVFS/thermal (4 × u32) | 16 |
| L2 cluster (residency u32, count u32, predicted u64×8) | 72 |
| KV cluster (2 × u32) | 8 |
| Weight cluster (2 × u32) | 8 |
| Fusion cluster (2 × u32) | 8 |
| Agentic / SLO (4 × u32) | 16 |
| Attention (2 × u32) | 8 |
| Freshness (jiffies u64 + reserved u32×16) | 72 |
| **Total** | **336 B** |

336 B fits in 6 × 64 = 384 B comfortably; the static_assert at 384 leaves 48 B headroom for one future cluster.

## Hashtable extension to cipher_pid_stats

The snapshot is **NOT** stored as a separate hashtable. It's a **view** computed on demand from existing `cipher_pid_stats` entries + new derived fields.

Phase 4.1 extends `cipher_pid_stats` (already extended in Phase 3) with the **derived** fields that don't have direct ioctl writers:

```c
struct cipher_pid_stats {
	/* ... existing fields through last_telemetry_jiffies ... */

	/* Phase 4 derived (written by cipher_state_updater kthread, not ioctls) */
	__u64    tenant_session_fp;            /* FNV-64(tenant_id), set on REGISTER_TENANT */
	__u32    tenant_handle_u32;            /* murmur32(tenant_id), set on REGISTER_TENANT */
	__u32    sm_partition_mask;            /* written by P4.2 allocator */
	__u32    sm_partition_count;
	__u32    thermal_headroom_pct;
	__u32    power_headroom_w;
	__u32    sustained_clock_mhz;
	__u32    voltage_envelope_mv;
	__u32    l2_residency_kb;
	__u32    hot_region_count;
	__u64    predicted_hot_regions[8];
	__u32    kv_cache_size_mb;
	__u32    kv_compression_ratio_pct;
	__u32    weight_dedup_savings_mb;
	__u32    weight_content_hash_count;
	__u32    fusion_recipe_id;
	__u32    fusion_eligible_flag;
	__u32    session_band;
	__u32    slo_priority;
	__u32    fairness_quota_remaining_pct;
	__u32    graph_capture_state;
	__u32    koopman_substitution_eligibility;
	__u64    snapshot_jiffies;
};
```

`cipher_get_current_tenant_snapshot()` returns a const pointer to a per-CPU thread-local **assembled** struct populated by copying the relevant `cipher_pid_stats` fields. This avoids a full struct copy on the hot path; the assembly is in-register-cache for sub-200ns target.

## Public API functions

```c
/* Hot path — single hashtable lookup keyed on current->pid.
 * Caller MUST be inside rcu_read_lock(). Returned pointer is per-CPU
 * thread-local; copy fields out before rcu_read_unlock().
 * Returns NULL if current task has no cipher_pid_stats entry.
 * Performance: < 200 ns on H100 host CPU. */
const struct cipher_tenant_snapshot *
cipher_get_current_tenant_snapshot(void);

/* Cross-tenant lookup. Linear scan, O(N) where N ≤ 150.
 * Caller MUST be inside rcu_read_lock().
 * Returns NULL if tenant_id not found. */
const struct cipher_tenant_snapshot *
cipher_get_tenant_snapshot_by_id(const char *tenant_id);

/* Snapshot enumeration for Prometheus / arbitration code.
 * Copies up to `max` snapshots into `out`. Returns count copied.
 * No RCU requirement (uses internal locking).
 * Safe to call from process context, not interrupt context. */
int cipher_enumerate_tenants(struct cipher_tenant_snapshot *out, int max);

/* Phase 4 actuator write-back paths — kernel-side allocator writes
 * partition decisions, kthread reads gpu_state to compute derived. */
int cipher_set_sm_partition_mask(pid_t target_pid, u32 mask);
int cipher_get_sm_partition_mask(pid_t target_pid, u32 *out_mask);
```

All three lookup functions are `EXPORT_SYMBOL_GPL` so out-of-tree modules (future `cipher_rt_km`) can resolve them.

## New ioctl: CIPHER_GET_TENANT_SNAPSHOT (nr 8)

```c
/* Phase 4 — per-tenant snapshot query from userspace.
 * Cross-process: target_pid or target_tenant_id (string).
 * No CAP_SYS_ADMIN required (read-only). */
struct cipher_tenant_snapshot_query {
	__u32 target_pid;                                /* 0 = use target_tenant_id */
	char  target_tenant_id[CIPHER_TENANT_ID_LEN];   /* "" = use target_pid */
	struct cipher_tenant_snapshot snapshot;          /* filled by kernel */
};

#define CIPHER_GET_TENANT_SNAPSHOT \
	_IOWR(CIPHER_IOCTL_MAGIC, 8, struct cipher_tenant_snapshot_query)
```

Direction is `_IOWR` because userspace writes the query fields and kernel writes the snapshot reply. Reserved nrs 2/3/4 still return `-ENOSYS`. Nrs 1/5/6/7 frozen. Nr 9 reserved for P4.2's CIPHER_REQUEST_SM_PARTITION.

## Identity bridge: REGISTER_TENANT extension

T4.7.1 spec says SENSE produces `session_fp = FNV-64(tenant_id)` and `tenant_handle_u32 = murmur32(tenant_id)`. To preserve Phase 3 ABI exactly, the registration ioctl `CIPHER_REGISTER_TENANT` (nr 1) **payload is unchanged**. Instead, the kernel computes the two derived ids inside the handler when the tenant_id string is stored on the pid entry:

```c
/* In cipher_dev_register_tenant() after strscpy of tenant_id: */
e->tenant_session_fp  = fnv64(e->tenant_id, strnlen(e->tenant_id, sizeof(e->tenant_id)));
e->tenant_handle_u32  = murmur3_32(e->tenant_id, strnlen(...), CIPHER_MURMUR_SEED);
```

Phase 3 callers see no behavior change. Phase 4 readers get all three ids on the snapshot.

## Snapshot read latency analysis

Target < 200 ns. Decomposition:

| Step | Estimated ns |
|---|---|
| `current->pid` access (register) | 1 |
| Hash + table index (`pid % HASH_SIZE`) | 2 |
| Hashtable bucket walk (avg load factor < 1) | 20-40 |
| RCU dereference (per CPU pointer) | 5 |
| Field-copy assembly into per-CPU thread-local | 80-120 |
| Cache miss penalty (worst case) | 50 |
| **Total worst case** | **~200 ns** |

If measured > 200 ns: cache the assembled snapshot per-pid-entry rather than re-assembling on each call. Trade-off: more memory (336 B × 150 tenants = 50 KB), faster reads. T4.1.3 makes the call after measurement.

## RCU contract for writers

Writers (cipher_state_updater kthread, P4.2 partition allocator, REGISTER_TENANT handler) update individual fields on `cipher_pid_stats` using `WRITE_ONCE`. No struct-wide locking. Readers see eventual consistency across the cluster of fields.

This is acceptable because:
- Each field is read independently by its consuming actuator (PARTITION_ROUTER reads sm_partition_mask only; VOLT reads voltage_envelope_mv only).
- Cross-field invariants (e.g., partition_count == popcount(partition_mask)) are weakly enforced — readers tolerate brief inconsistency.
- The snapshot_jiffies field is the freshness signal; readers can detect stale snapshots by comparing to `get_jiffies_64()`.

Memory ordering: `WRITE_ONCE` is compiler-fence only. On x86-64, single-word aligned writes are torn-write-free without barriers. On other archs (none in this project), `smp_store_release` would be needed.

## P4.0 dependents

| Subsequent task | What it consumes from T4.0.2 |
|---|---|
| T4.0.3 cipher_state_updater | Derived field list + WRITE_ONCE contract |
| T4.0.4 cipher_rt_tenant.h | Public API signatures + ioctl nr 8 |
| T4.0.5 measurement harness | Schema for parsing /proc/cipher/stats Phase 4 columns |
| T4.0.7 PHASE_4_ARCHITECTURE.md | Full struct + lookup latency budget |
| T4.1.2 cipher_internal.h | DRAFT applied to live tree |
| T4.1.3 cipher_get_current_tenant_snapshot() | Implementation |
| T4.1.7 ioctl nr 8 handler | Implementation |

## What this spec does NOT do (deferred to T4.1 or later)

- Apply changes to live `cipher_internal.h` (T4.1.2)
- Implement the lookup functions (T4.1.3-T4.1.5)
- Implement cipher_state_updater kthread (T4.0.3 writes the .c file; T4.1.6 wires it in)
- Bump module version (T4.1.1)
- Test ioctl nr 8 (T4.1.9)

## Sign-off (preconditions, end of T4.0.2)

This document creates no compiled artifact. Phase 3 substrate guaranteed unchanged. To be verified after T4.0.7:
- Fallback `cipher_kmod.ko.v0.2.0` md5 unchanged
- Fallback `libcipher_v2.so.v0.2.0` md5 unchanged
- cipher_kmod 0.3.1 still loaded, taint stable, /proc/cipher entries readable

T4.0.3 begins with cipher_state_updater.c skeleton.
