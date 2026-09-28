/* SPDX-License-Identifier: GPL-2.0
 *
 * cipher_ioctl.h -- public ABI between cipher_kmod and userspace clients.
 *
 * IMPORTANT: this is a stable ABI. Once shipped, do not change struct
 * layouts or ioctl nrs. Add new ioctls via fresh nrs only. Existing
 * fields are never reordered, renamed, or repurposed.
 *
 * Phase 2: CIPHER_REGISTER_TENANT (nr 1) is the only live ioctl.
 * Phase 3 (Task 3): nrs 5/6/7 added for GPU-state spine telemetry.
 * Phase 6: nrs 2/3/4 are reserved -- the kmod returns -ENOSYS for them.
 */
#ifndef CIPHER_IOCTL_H
#define CIPHER_IOCTL_H

#include <linux/types.h>
#include <linux/ioctl.h>

#define CIPHER_IOCTL_MAGIC    'C'

#define CIPHER_TENANT_ID_LEN  64

/*
 * CIPHER_REGISTER_TENANT (nr 1, _IOW)
 *
 * Caller declares its tenant identity. cipher_kmod stamps tenant_id
 * onto the per-PID hashtable entry for the calling task (the task's
 * `pid` is its LWP; `tgid` is its thread-group id).
 *
 * Security: the kernel verifies that payload.pid == current->pid and
 * payload.tgid == current->tgid before accepting. Spoofing another
 * task's identity returns -EPERM.
 */
struct cipher_register_tenant {
	__u32 pid;
	__u32 tgid;
	char  tenant_id[CIPHER_TENANT_ID_LEN];
};

#define CIPHER_REGISTER_TENANT \
	_IOW(CIPHER_IOCTL_MAGIC, 1, struct cipher_register_tenant)

/*
 * Reserved for Phase 6. ABI layout below is provisional; do NOT depend
 * on it. cipher_kmod currently returns -ENOSYS for these nrs.
 */
struct cipher_snapshot {
	__u32 size_used;
	char  buf[4096];
};

#define CIPHER_SNAPSHOT      _IOR(CIPHER_IOCTL_MAGIC, 2, struct cipher_snapshot)
#define CIPHER_RESET         _IOW(CIPHER_IOCTL_MAGIC, 3, __u32)
#define CIPHER_GET_VERSION   _IOR(CIPHER_IOCTL_MAGIC, 4, __u32)

/*
 * Phase 3 Task 3 -- GPU-state spine telemetry submission.
 *
 * The cipher-gpustate userspace daemon polls NVML/GPM/CUPTI and pushes
 * three streams of data into the kernel:
 *
 *   nr 5  SUBMIT_GPU_STATE      device-wide GPU state (single global, latest-wins)
 *   nr 6  SUBMIT_PROCESS_UTIL   per-process GPM utilisation (keyed by target PID)
 *   nr 7  SUBMIT_LAUNCH_STATS   CUPTI launch counters from the workload itself
 *
 * Layout rule: every struct ends with a __u32 reserved[N] tail so future
 * extensions land inside the same fixed sizeof -- no nr churn.
 */

struct cipher_gpu_state {
	__u64 timestamp_ns;       /* CLOCK_MONOTONIC at sample */
	__u32 power_mw;           /* milliwatts */
	__u32 temp_c;             /* degrees C */
	__u32 sm_clock_mhz;
	__u32 mem_clock_mhz;
	__u32 sm_util_pct;        /* GPU overall, 0-100 */
	__u32 mem_util_pct;
	__u32 fb_used_mb;
	__u32 reserved[8];
};

struct cipher_process_util {
	__u32 pid;                /* target PID, kernel task->pid */
	__u32 tgid;
	__u64 timestamp_ns;
	__u32 sm_util_pct;        /* per-process from NVML GPM */
	__u32 mem_util_pct;
	__u32 fb_used_mb;
	__u32 enc_util_pct;
	__u32 dec_util_pct;
	__u32 reserved[8];
};

struct cipher_launch_stats {
	__u32 pid;                /* caller's pid -- anti-spoof verified */
	__u32 tgid;               /* caller's tgid -- anti-spoof verified */
	__u64 timestamp_ns;
	__u64 launches_total;     /* CUPTI counter snapshot */
	__u64 grid_ops_total;     /* sum of grid_dim products */
	__u32 reserved[4];
};

#define CIPHER_SUBMIT_GPU_STATE \
	_IOW(CIPHER_IOCTL_MAGIC, 5, struct cipher_gpu_state)
#define CIPHER_SUBMIT_PROCESS_UTIL \
	_IOW(CIPHER_IOCTL_MAGIC, 6, struct cipher_process_util)
#define CIPHER_SUBMIT_LAUNCH_STATS \
	_IOW(CIPHER_IOCTL_MAGIC, 7, struct cipher_launch_stats)

/*
 * Phase 4.1 Task T4.1.7 — per-tenant snapshot query (nr 8).
 *
 * Cross-process read-only query: userspace fills target_pid OR
 * target_tenant_id (non-empty string takes precedence). Kernel fills the
 * snapshot from cipher_pid_table via the contract in PHASE_4_CONTRACT.md.
 *
 * Returns 0 on success with snapshot populated; -ENOENT if not found.
 * No CAP_SYS_ADMIN: this is read-only.
 *
 * Layout note: this struct is the userspace view of the kernel-internal
 * struct cipher_tenant_snapshot. Field names + offsets match exactly.
 */
struct cipher_tenant_snapshot_user {
	/* Identity */
	char  tenant_id_str[CIPHER_TENANT_ID_LEN];
	__u64 tenant_session_fp;
	__u32 tenant_handle_u32;
	__u32 pid;
	__u32 tgid;
	__u32 _pad_identity;

	/* Live telemetry */
	__u32 sm_util_pct;
	__u32 mem_util_pct;
	__u64 launches_total;
	__u64 grid_ops_total;
	__u64 ioctls_total;

	/* SM partition cluster */
	__u32 sm_partition_mask;
	__u32 sm_partition_count;

	/* DVFS / thermal */
	__u32 thermal_headroom_pct;
	__u32 power_headroom_w;
	__u32 sustained_clock_mhz;
	__u32 voltage_envelope_mv;

	/* L2 */
	__u32 l2_residency_kb;
	__u32 hot_region_count;
	__u64 predicted_hot_regions[8];

	/* KV */
	__u32 kv_cache_size_mb;
	__u32 kv_compression_ratio_pct;

	/* Weight */
	__u32 weight_dedup_savings_mb;
	__u32 weight_content_hash_count;

	/* Fusion */
	__u32 fusion_recipe_id;
	__u32 fusion_eligible_flag;

	/* Agentic / SLO */
	__u32 session_band;
	__u32 slo_priority;
	__u32 fairness_quota_remaining_pct;
	__u32 _pad_agentic;

	/* Attention */
	__u32 graph_capture_state;
	__u32 koopman_substitution_eligibility;

	/* Freshness */
	__u64 snapshot_jiffies;

	/* Cb.2 reserved-tail bump (Wave 5 §5.5 Week 1, Option A adjudication).
	 * Two new __u32 fields land at the HEAD of the reserved tail. Userspace
	 * mirror of the kmod-internal struct cipher_pid_stats (cipher_internal.h)
	 * — names + offsets match exactly. */
	__u32 recommended_sm_count;
	__u32 tenant_billing_class;
	/* W7-9 Step 1 G10 (kmod 0.5.5): per-tenant model identity (16 B from
	 * reserved tail; reserved[14] -> reserved[10]). */
	__u8  model_uuid[16];
	__u32 reserved[10];
	/* W7-9 Step 3 G6 (kmod 0.5.5 -> 0.6.0): per-tenant HMAC-SHA256 chain
	 * accumulator (Option A locked by user 2026-05-23). hmac_state[32] +
	 * hmac_chain_head[32] = 64 B; struct grows 336 -> 400 B. ABI bump
	 * signals the layout change; consumers re-validate via sizeof. */
	__u8  hmac_state[32];
	__u8  hmac_chain_head[32];
};

struct cipher_tenant_snapshot_query {
	__u32 target_pid;                                /* 0 = use target_tenant_id */
	char  target_tenant_id[CIPHER_TENANT_ID_LEN];   /* "" = use target_pid */
	struct cipher_tenant_snapshot_user snapshot;
};

#define CIPHER_GET_TENANT_SNAPSHOT \
	_IOWR(CIPHER_IOCTL_MAGIC, 8, struct cipher_tenant_snapshot_query)

/*
 * Phase 4.2 T4.2.1 — SM partition request (nr 9).
 *
 * Caller hints number of partitions desired (1..8). Kernel returns the
 * granted mask + count.
 *
 * Default (flags=0, T4.2.1 behavior): grow-only-idempotent. Re-calling
 * with a larger hint may expand if free slots exist; a smaller hint
 * returns the current larger mask unchanged (the kmod does NOT release
 * slots on its own).
 *
 * With CIPHER_PARTITION_FLAG_FIT_HINT (B10, kmod 0.4.5+): "fit-to-N"
 * semantic. Caller asserts "make my granted set exactly hint_partitions
 * slots if possible". If currently holding more than hint, the kmod
 * releases the excess. If holding fewer, grows (same as default) up to
 * hint or pool-availability. If holding exactly hint, no-op. Allows
 * cooperative fair-share across tenants:
 *   fair = max(1, min(8, 32 / N_peers))
 *   poll-thread issues nr 9 with hint=fair, flags=FIT_HINT
 *
 * With FIT_HINT, hint_partitions=0 is the explicit "release-all" form
 * (no clamp to 1). Without the flag, hint=0 still clamps to 1
 * (preserves T4.2.1 behavior for old callers).
 *
 * Any flag bit other than FIT_HINT is reserved-for-future; the kmod
 * returns -EINVAL if any unknown bit is set.
 *
 * No CAP_SYS_ADMIN. Tenants request slots for themselves; the underlying
 * lookup uses current->pid.
 */
struct cipher_partition_request {
	__u32 hint_partitions;       /* 1..8 (default); 0..8 with FIT_HINT */
	__u32 partition_mask_out;    /* filled by kernel */
	__u32 partition_count_out;   /* popcount of mask_out */
	__u32 flags;                 /* B10 (kmod 0.4.5+); was 'reserved' in 0.4.4 */
};

/* B10 / kmod 0.4.5: see comment above CIPHER_REQUEST_SM_PARTITION. */
#define CIPHER_PARTITION_FLAG_FIT_HINT    (1U << 0)
#define CIPHER_PARTITION_FLAGS_ALL        (CIPHER_PARTITION_FLAG_FIT_HINT)

#define CIPHER_REQUEST_SM_PARTITION \
	_IOWR(CIPHER_IOCTL_MAGIC, 9, struct cipher_partition_request)

/*
 * Phase 4.3 T4.3.2 — SM clock actuation via kmod-mediated usermode helper
 * (nr 10).
 *
 * Closes the privilege gap from T4.3.1: the libcipher_rt user-process VOLT
 * actuator cannot call nvmlDeviceSetGpuLockedClocks directly because NVML
 * returns NVML_ERROR_NOT_SUPPORTED from a non-root injection context.
 * This ioctl gives the user-process actuator a privileged path: kmod
 * invokes `nvidia-smi -lgc <mhz>` via call_usermodehelper().
 *
 * Caller passes a single __u32:
 *   mhz == 0           → reset (nvidia-smi -rgc, unlock)
 *   mhz in [210,1980]  → lock (nvidia-smi -lgc <mhz>)
 *   anything else      → -EINVAL
 *
 * Trust model: /dev/cipher is 0666 (matches existing CIPHER ABI pattern —
 * REQUEST_SM_PARTITION etc also accept non-root). Operator-deployed
 * environments can restrict via udev rule + cipher group if a stricter
 * policy is required. The kmod's call_usermodehelper inherits root
 * regardless of caller — that is the design intent (T4.3.2 closes the
 * non-root user-process → privileged actuation gap from T4.3.1).
 *
 * Bounds are enforced regardless of caller — a malicious tenant cannot
 * exit the safe [210, 1980] MHz range.
 *
 * Latency: nvidia-smi spawn ~100-300 ms. Init-time use only, not per-launch.
 *
 * Returns 0 on success, -EINVAL on bad mhz, -EIO on usermode-helper
 * failure, -ERESTARTSYS on signal during contention.
 */
#define CIPHER_SET_CLOCK_MHZ \
	_IOW(CIPHER_IOCTL_MAGIC, 10, __u32)

/*
 * CP 3.3 — continuous hardware FLOP telemetry (nrs 11/12).
 *
 * The cipher_flopd root daemon does the hardware-counter read via CUPTI
 * PM Sampling (the GH100 SM perfmon PRI register map is not in the open
 * kernel tree, so a kmod-direct read is not possible). The daemon submits
 * a device-wide achieved-FLOP sample ~10 Hz via SUBMIT_FLOP_SAMPLE; the
 * kmod owns the continuous FLOP-series ring and derives per-tenant FLOPs
 * + MFU from the per-tenant launches_total it already keeps.
 *
 * Build STEP-1 finding (2026-05-15): PM Sampling on this GH100 fits exactly
 * one metric per single-pass config, so the daemon samples the tensor pipe
 * (sm__pipe_tensor_cycles_active) — the dominant FLOP path for LLM
 * inference and the quantity MFU measures. The CUDA-core FP pipe is not
 * co-samplable; device_fp_flops_per_s is reserved (0) accordingly.
 *
 * nr 11  SUBMIT_FLOP_SAMPLE  daemon -> kmod, CAP_SYS_ADMIN, ~10 Hz
 * nr 12  QUERY_FLOPS         consumer -> kmod, unprivileged read
 */

struct cipher_flop_sample {
	__u64 timestamp_ns;               /* daemon CLOCK_MONOTONIC at sample */
	__u64 interval_ns;                /* window this sample integrates */
	__u64 device_tensor_flops_per_s;  /* tensor-pipe achieved FLOP/s (primary) */
	__u64 device_fp_flops_per_s;      /* CUDA-core FP pipe; 0 = not sampled */
	__u32 tensor_pipe_milli_pct;      /* tensor pipe util %, x1000 (0..100000) */
	__u32 sm_clock_mhz;               /* NVML, same instant — cross-check */
	__u32 sm_util_pct;                /* NVML, same instant — cross-check */
	__u32 power_mw;                   /* NVML, same instant */
	__u32 source_seq;                 /* daemon monotonic seq; gap detect */
	__u32 reserved[7];
};

struct cipher_flop_tenant {
	__u32 pid;
	__u32 tgid;
	char  tenant_id[CIPHER_TENANT_ID_LEN];
	__u64 launches_delta;             /* launches in the last FLOP interval */
	__u64 attributed_flops_per_s;     /* device FLOP/s x launch-share */
	__u32 launch_share_ppm;           /* parts-per-million of total launches */
	__u32 mfu_milli_pct;              /* per-tenant MFU %, x1000 */
	__u32 reserved[4];
};

/* status field of cipher_flop_query */
#define CIPHER_FLOP_STATUS_LIVE            0  /* fresh daemon sample */
#define CIPHER_FLOP_STATUS_STALE           1  /* daemon sample is old */
#define CIPHER_FLOP_STATUS_NO_SOURCE       2  /* no daemon sample ever seen */

#define CIPHER_FLOP_QUERY_MAX_TENANTS     64

struct cipher_flop_query {
	/* [out] device-wide header */
	__u64 latest_timestamp_ns;
	__u64 device_tensor_flops_per_s;
	__u32 device_mfu_milli_pct;       /* device MFU %, x1000 (vs 989 TFLOP/s) */
	__u32 sm_clock_mhz;
	__u32 sm_util_pct;
	__u32 ring_samples;               /* populated samples in the kmod ring */
	__u32 sample_age_ms;              /* age of latest sample */
	__u32 status;                     /* CIPHER_FLOP_STATUS_* */
	/* [in] caller capacity / [out] filled count */
	__u32 max_tenants;
	__u32 n_tenants;
	struct cipher_flop_tenant tenants[CIPHER_FLOP_QUERY_MAX_TENANTS];
	__u32 reserved[8];
};

#define CIPHER_SUBMIT_FLOP_SAMPLE \
	_IOW(CIPHER_IOCTL_MAGIC, 11, struct cipher_flop_sample)
#define CIPHER_QUERY_FLOPS \
	_IOWR(CIPHER_IOCTL_MAGIC, 12, struct cipher_flop_query)

/*
 * CP 5.4 — per-tenant arbitration: 8-SM-group allocation ledger
 * (nrs 13/14/15). The kmod-resident cross-process SM allocator. CUDA green
 * contexts on H100 partition in multiples of 8 SMs; the device is allocated
 * as 15 × 8-SM groups (120 of 132 SMs — H100 hardware split measured in
 * CP 5.4 Step 1.3a, see PHASE_1_3A_PROBE.md). Replaces the legacy nr-9
 * 4-SM-slot allocator, which is deactivated (returns -ENOSYS) — the two
 * address the same physical SMs and must not both grant.
 *
 * qos_class:
 *   PARTITION — own exclusive partition of ceil(sm_count/8) 8-SM groups.
 *   SHARED    — batch-pool member; owns no groups, uses the pool.
 *   POOL      — the batch-pool owner (singleton); granted the residual.
 */
#define CIPHER_CP54_QOS_PARTITION  0
#define CIPHER_CP54_QOS_SHARED     1
#define CIPHER_CP54_QOS_POOL       2

struct cipher_cp54_allocate {
	__u32 qos_class;       /* in : CIPHER_CP54_QOS_* */
	__u32 sm_count;        /* in : PARTITION SMs wanted (rounded up to 8);
	                        *      ignored for SHARED / POOL */
	__u32 grp_mask_out;    /* out: bitmask of granted 8-SM groups, bit g ⇒
	                        *      group g (15 groups live; see Step 1.3a) */
	__u32 grp_count_out;   /* out: popcount(grp_mask_out) */
	__u32 reserved[4];
};

struct cipher_cp54_query {
	__u32 n_partitions;    /* out: registered PARTITION tenants */
	__u32 pool_grp_count;  /* out: 8-SM groups held by the batch pool */
	__u32 free_grp_count;  /* out: unallocated 8-SM groups */
	__u32 my_grp_mask;     /* out: caller's own group mask */
	__u32 my_qos_class;    /* out: caller's qos_class, 0xFFFFFFFF if none */
	__u32 reserved[3];
};

/* ALLOCATE — register/allocate; POOL is a singleton (a second distinct PID
 *            registering POOL returns -EEXIST). Idempotent per PID.
 * FREE     — release the caller's allocation (no payload).
 * QUERY    — read-only ledger snapshot. */
#define CIPHER_CP54_ALLOCATE \
	_IOWR(CIPHER_IOCTL_MAGIC, 13, struct cipher_cp54_allocate)
#define CIPHER_CP54_FREE \
	_IO(CIPHER_IOCTL_MAGIC, 14)
#define CIPHER_CP54_QUERY \
	_IOR(CIPHER_IOCTL_MAGIC, 15, struct cipher_cp54_query)

/*
 * Track 3 (Dynamic SM Migration) — SC2 kmod migration state machine
 * (nrs 16-20). Additive ABI extension — see TRACK_3_ABI.md. NRs 16-20 are
 * PERMANENTLY assigned to these five ioctls; future maintainers must not
 * collapse the numbering or repurpose them ([[cipher-abi-rule]]).
 *
 * Live migration of a PARTITION tenant's 8-SM-group placement so the kmod
 * can compact the ledger and keep the batch POOL on a maximal contiguous
 * low prefix under arbitrary partition churn. Migration is COUNT-PRESERVING
 * (popcount(new_mask) == popcount(old_mask)) and OPT-IN (a tenant is `pinned`
 * unless it calls SUBSCRIBE_MIGRATE(1)). With zero migratable tenants the
 * ledger behaves byte-identically to nrs 13/14/15 alone.
 *
 * Migration state machine (per-tenant):
 *   IDLE --[kmod PROPOSE]--> PROPOSED --[tenant START]--> MIGRATING
 *                                |                            |
 *                       [timeout, no START]          [tenant ACK(ok=1)]--> COMMIT
 *                                v                            |
 *                              ABORT <--[tenant ACK(ok=0) NACK]+
 *   COMMIT / ABORT return the tenant to IDLE. The kmod never unilaterally
 *   aborts a MIGRATING migration; a tenant MAY abort its own MIGRATING
 *   migration by ACK(ok=0) (it has authoritative knowledge it did not swap
 *   its green context — verify-before-swap, SC3). This is why ACK_MIGRATE
 *   carries a __u32 status rather than being a payload-less _IO.
 *
 *   nr 16  SUBSCRIBE_MIGRATE  __u32: 1 = migratable, 0 = pinned (default)
 *   nr 17  POLL_MIGRATE       tenant reads migrate_state + target_mask
 *   nr 18  START_MIGRATE      PROPOSED -> MIGRATING (commit of intent)
 *   nr 19  ACK_MIGRATE        __u32: 1 = COMMIT, 0 = tenant-NACK -> ABORT
 *   nr 20  COMPACT_MIGRATE    operator/test: force a compaction-eval pass
 */

/* migrate_state — cipher_cp54_migrate_poll.migrate_state */
#define CIPHER_CP54_MIG_IDLE        0
#define CIPHER_CP54_MIG_PROPOSED    1
#define CIPHER_CP54_MIG_MIGRATING   2

/* last_outcome — cipher_cp54_migrate_poll.last_outcome (item-5 addition):
 * the outcome of the tenant's most recent migration, for debugging / error
 * reporting. Set on every transition back to IDLE; survives until the next
 * migration. Removes the bug class where tenant code mis-infers the outcome
 * from a cur_mask comparison. */
#define CIPHER_CP54_MIGOUT_NONE             0  /* no migration has completed */
#define CIPHER_CP54_MIGOUT_COMMITTED        1  /* migration committed */
#define CIPHER_CP54_MIGOUT_ABORTED_TIMEOUT  2  /* PROPOSED expired, no START */
#define CIPHER_CP54_MIGOUT_ABORTED_KMOD_REFUSED 3 /* kmod refused (reserved) */
#define CIPHER_CP54_MIGOUT_ABORTED_TENANT_NACK  4 /* tenant ACK(ok=0) */

struct cipher_cp54_migrate_poll {
	__u32 migrate_state;   /* out: CIPHER_CP54_MIG_* */
	__u32 target_mask;     /* out: proposed new_mask; 0 unless PROPOSED/MIGRATING */
	__u32 cur_mask;        /* out: caller's current owned 8-SM-group mask */
	__u32 last_outcome;    /* out: CIPHER_CP54_MIGOUT_* */
	__u32 reserved[4];
};

#define CIPHER_CP54_SUBSCRIBE_MIGRATE \
	_IOW(CIPHER_IOCTL_MAGIC, 16, __u32)
#define CIPHER_CP54_POLL_MIGRATE \
	_IOR(CIPHER_IOCTL_MAGIC, 17, struct cipher_cp54_migrate_poll)
#define CIPHER_CP54_START_MIGRATE \
	_IO(CIPHER_IOCTL_MAGIC, 18)
#define CIPHER_CP54_ACK_MIGRATE \
	_IOW(CIPHER_IOCTL_MAGIC, 19, __u32)
#define CIPHER_CP54_COMPACT_MIGRATE \
	_IO(CIPHER_IOCTL_MAGIC, 20)

/*
 * Track 2 SC5 — weight-arena fd custodian (nrs 21-24). Additive ABI
 * extension — see TRACK_2_ABI.md. NRs 21-24 are PERMANENTLY assigned to
 * these four ioctls; future maintainers must not collapse the numbering or
 * repurpose them ([[cipher-abi-rule]]).
 *
 * Cross-tenant weight sharing: a producer REGISTERs its exported VMM POSIX
 * fd with the kmod; consumers IMPORT a dup'd fd by arena id — whether or not
 * the producer is still alive. The kmod is the fd custodian: it holds the
 * fd, never calls the CUDA driver (the driver already cross-process-
 * refcounts the physical memory — SC5 design memo §1). Opaque metadata
 * blobs pass by a userspace pointer so the ioctl structs stay small.
 */
#define CIPHER_WA_BLOB_MAX     65536    /* opaque metadata blob cap */
#define CIPHER_WA_MAX_ARENAS   100 /* W6 G2: 16 -> 100 per WEEK_6_ARCHITECTURE_GAP_AUDIT.md; struct cipher_arena_query.arenas[] payload grows 392 B -> 2408 B (userspace ABI break signalled by 0.4.8 -> 0.5.0 MODULE_VERSION bump); downstream tooling (phase_c/sc5_arena_ioctl.py) must update its mirror constant. */

struct cipher_arena_register {
	__s32 fd;            /* in : exported VMM POSIX fd */
	__u32 arena_id_out;  /* out: assigned arena id */
	__u64 base;          /* in : arena base devptr */
	__u64 size;          /* in : arena byte size */
	__u64 blob_ptr;      /* in : userspace ptr to the metadata blob */
	__u32 blob_len;      /* in : blob length, <= CIPHER_WA_BLOB_MAX */
	__u32 reserved[5];
};

struct cipher_arena_import {
	__u32 arena_id;      /* in : arena to import */
	__s32 fd_out;        /* out: a fd in the caller, dup'd from the kmod's */
	__u64 base;          /* out: arena base devptr */
	__u64 size;          /* out: arena byte size */
	__u64 blob_ptr;      /* in : userspace OUT buffer for the metadata blob */
	__u32 blob_cap;      /* in : blob_ptr buffer capacity */
	__u32 blob_len;      /* out: actual blob length */
	__u32 reserved[4];
};

struct cipher_arena_query {
	__u32 n_arenas;      /* out */
	__u32 reserved;
	struct {
		__u32 arena_id;
		__u32 producer_pid;
		__u32 n_consumers;
		__u32 _pad;
		__u64 size;
	} arenas[CIPHER_WA_MAX_ARENAS];
};

#define CIPHER_ARENA_REGISTER \
	_IOWR(CIPHER_IOCTL_MAGIC, 21, struct cipher_arena_register)
#define CIPHER_ARENA_IMPORT \
	_IOWR(CIPHER_IOCTL_MAGIC, 22, struct cipher_arena_import)
#define CIPHER_ARENA_LEAVE \
	_IOW(CIPHER_IOCTL_MAGIC, 23, __u32)
#define CIPHER_ARENA_QUERY \
	_IOR(CIPHER_IOCTL_MAGIC, 24, struct cipher_arena_query)

/* Week 2 Step 6 — userspace pushes the cipher_rt_classify_observer
 * snapshot to kmod for /proc/cipher/classify_stats consumption.
 * SET semantics: kmod counters become last-known observer values.
 * Payload struct mirrors cipher_rt_classify_observer_stats exactly
 * (216 B; 27 × __u64). Piggybacked on the existing 256-launch flush
 * in cipher_cupti.c. */
struct cipher_classify_stats_push {
	__u64 total;
	__u64 handled;
	__u64 passthrough;
	__u64 per_op_class[16];
	__u64 reserved[8];
};

#define CIPHER_PUSH_CLASSIFY_STATS \
	_IOW(CIPHER_IOCTL_MAGIC, 25, struct cipher_classify_stats_push)

/* Week 3 Step 4 Option II-a — DSM PROPOSE bridge. Userspace's
 * transition-detection wrapper above SENSE pushes proposals to kmod;
 * kmod buffers them in a bounded ring (Pattern (a) observability,
 * per WEEK_3_STEP_4_OPT2_PREFLIGHT.md §6.2) and exposes them via
 * /proc/cipher/dsm_proposals. v1: no auto-action; operator-readable
 * only. 56 bytes; 8-byte aligned. */
struct cipher_dsm_propose_push {
	__u32 tenant_id;
	__u32 source_slot;     /* current session class (CipherSessionType) */
	__u32 target_slot;     /* recommended new class */
	__u32 reason;          /* CIPHER_RT_SENSE_* reason code */
	__u32 confidence;      /* 0..100 */
	__u32 reserved_u32;    /* alignment */
	__u64 reserved[4];     /* Cb.2 future expansion */
};

#define CIPHER_DSM_PROPOSE \
	_IOW(CIPHER_IOCTL_MAGIC, 26, struct cipher_dsm_propose_push)

/* W7-9 Step 1 G10: CIPHER_REGISTER_MODEL — model identity propagation
 * through the kmod boundary. Tenants register a (model_path, hf_config_hash,
 * model_arch) tuple; the kmod assigns a stable 128-bit model_uuid keyed on
 * the hf_config_hash. Idempotent: same hash always returns the same uuid.
 * Unblocks W10-12 G3 (KV-dedup model-keying), W10-12 G4 (Marlin tenant-scoped
 * weight kit), W13-14 G12 (Koopman registry model-keying).
 *
 * NR 27 is additive per [[cipher-abi-rule]] (no NR repurposing). MODULE_VERSION
 * bumps 0.5.0 -> 0.5.5 (additive ioctl + additive struct field within
 * cipher_pid_stats reserved tail; no layout break).
 *
 * Payload is ~4 KiB (PATH_MAX-driven); kmod handler heap-allocates per the
 * W6 G1+G2 cipher_arena_query kzalloc precedent (1024 B kernel-stack budget).
 */
enum cipher_model_arch {
	CIPHER_MODEL_ARCH_MISTRAL  = 1,
	CIPHER_MODEL_ARCH_QWEN     = 2,
	CIPHER_MODEL_ARCH_LLAMA    = 3,
	CIPHER_MODEL_ARCH_GPT_NEOX = 4,
	CIPHER_MODEL_ARCH_OTHER    = 255,
};

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

struct cipher_register_model {
	char     model_path[PATH_MAX];   /* in : canonical model dir path */
	__u8     hf_config_hash[32];     /* in : sha256 over hf_config canonical JSON */
	__u32    model_arch;             /* in : enum cipher_model_arch */
	__u8     model_uuid[16];         /* out: 128-bit UUID; sentinel-zero means MODEL_UNKNOWN */
	__u32    flags;                  /* in/out: reserved, set to 0 */
	__u8     reserved[12];           /* additive headroom */
};

/* All-zero uuid sentinel meaning "model identity unknown — downstream
 * model-keyed actuators (Marlin / KV-dedup / Koopman) treat as pass-through". */
#define CIPHER_MODEL_UUID_UNKNOWN_INIT { 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 }

#define CIPHER_REGISTER_MODEL \
	_IOWR(CIPHER_IOCTL_MAGIC, 27, struct cipher_register_model)

/* W7-9 Step 3 G6: kmod-resident AUDIT chain record (NR 28).
 * Caller hands the (tenant_id, commit_seq, actuator_id, optional payload)
 * tuple; kmod computes HMAC-SHA256(per-tenant key, prev_chain_head ||
 * payload), appends to the per-tenant ring buffer, and updates
 * cipher_pid_stats.hmac_chain_head. Composes with COMMIT primitive
 * Step 1 (cipher_rt_phase4/cipher_rt_commit.c) — Step 1's audit-chain-
 * advance slot writes via this ioctl.
 *
 * Userspace consumers also mmap /dev/cipher to read the ring buffer for
 * offline HMAC chain verification (PROT_READ; .mmap fops handler returns
 * remap_vmalloc_range over the ~32 MiB ring storage). */
struct cipher_audit_record_req {
	__u32 tenant_id;
	__u32 actuator_id;
	__u64 commit_seq;
	__u8  payload[128];     /* optional payload to HMAC-chain */
	__u32 payload_len;
	__u32 flags;            /* reserved */
};

#define CIPHER_AUDIT_RECORD \
	_IOW(CIPHER_IOCTL_MAGIC, 28, struct cipher_audit_record_req)

/* Helper enum (actuator_id values) — additive; downstream consumers may
 * pass their own. */
enum cipher_audit_actuator_id {
	CIPHER_AUDIT_ACTUATOR_COMMIT    = 1,
	CIPHER_AUDIT_ACTUATOR_MARLIN    = 2,
	CIPHER_AUDIT_ACTUATOR_KV_DEDUP  = 3,
	CIPHER_AUDIT_ACTUATOR_VOLT      = 4,
	CIPHER_AUDIT_ACTUATOR_KOOPMAN   = 5,
};

/* W7-9 Step 5: multi-tenant resolver — CIPHER_REGISTER_STREAMS (NR 29).
 *
 * Tenants enumerate their owned named CUDA streams after REGISTER_TENANT
 * (NR 1) and pass the opaque stream handles. cipher_kmod stamps the
 * (current->tgid, stream_handle) -> tenant_id mapping into a vmalloc'd
 * view-slot table that cipher_rt_phase4 mmaps for lock-free hot-path
 * lookup.
 *
 * Policy (v1 named-streams): stream_handles[i] == 0 (the default-stream
 * sentinel) is rejected with -EINVAL. v1 requires tenant processes to use
 * named cudaStream_t handles. vLLM does this natively. v2 graceful
 * degradation (default-stream → shared fairness bucket) is deferred.
 *
 * tgid keying: stream handles are process-local CUDA pointers. Two
 * different processes can hold identical cudaStream_t values referring to
 * distinct streams. The kmod records current->tgid at register time and
 * keys lookups on (tgid, handle), not handle alone. */
#define CIPHER_STREAM_MAX_PER_TENANT 16

struct cipher_register_streams {
	__u32 tenant_id;
	__u32 num_streams;          /* 1..CIPHER_STREAM_MAX_PER_TENANT */
	__u64 stream_handles[CIPHER_STREAM_MAX_PER_TENANT];
};

#define CIPHER_REGISTER_STREAMS \
	_IOW(CIPHER_IOCTL_MAGIC, 29, struct cipher_register_streams)

/* W.6 sub-C: GPU co-residence registry — CIPHER_COHORT_REGISTER (NR 30) +
 * CIPHER_COHORT_QUERY (NR 31).
 *
 * A per-tgid table {tgid, model_fingerprint, last_seen_ns} that lets a
 * tenant discover which OTHER tenant processes share this GPU right now,
 * each with its substrate model fingerprint (cipher_workload_model_
 * fingerprint, W.6 sub-B). W.4 POOL consumes (co-resident pids + per-pid
 * fingerprint) to form legal cross-tenant GEMM-coalescing groups: same
 * fingerprint => same model => coalescable.
 *
 * Liveness: an entry is live if seen within CIPHER_COHORT_STALE_NS (30 s).
 * QUERY (NR 31) is the heartbeat — it does heartbeat-or-insert of the
 * caller's own entry (keyed on current->tgid, using caller_fingerprint)
 * and prunes peers stale > 30 s. A process that stops calling QUERY
 * (exited or fully idle) ages out. REGISTER (NR 30) is the explicit
 * first-stable-fingerprint signal; correctness does NOT depend on it
 * because QUERY re-inserts a caller that was pruned during idle.
 *
 * Anti-spoof: both ioctls key on current->tgid; the userspace tgid field
 * is informational and ignored by the kmod.
 *
 * Scope: the registry is kmod-global == host-global. v1 assumes one GPU
 * per host (this pod: single H100). Multi-GPU co-residence keying on
 * (gpu_uuid, tgid) is v1.x debt — see W6_SUBC_CLOSE_REPORT §E.
 */
#define CIPHER_COHORT_MAX        128            /* G1+G2 tenant cap */
#define CIPHER_COHORT_STALE_NS   (30ULL * 1000000000ULL)  /* 30 s liveness */

struct cipher_cohort_register {
	__u32 tgid;                  /* informational; kmod uses current->tgid */
	__u32 _pad;
	__u64 model_fingerprint;     /* cipher_workload_model_fingerprint() */
};

struct cipher_cohort_entry_abi {
	__u32 tgid;
	__u32 _pad;
	__u64 model_fingerprint;
};

struct cipher_cohort_query {
	__u64 caller_fingerprint;    /* IN: heartbeat-or-insert self with this fp; 0 = query-only */
	__u32 max_entries;           /* IN: capacity of entries[] (clamped to CIPHER_COHORT_MAX) */
	__u32 n_resident;            /* OUT: distinct LIVE tgids incl. self (may exceed max_entries => truncated) */
	struct cipher_cohort_entry_abi entries[CIPHER_COHORT_MAX];  /* OUT */
};

#define CIPHER_COHORT_REGISTER \
	_IOW(CIPHER_IOCTL_MAGIC, 30, struct cipher_cohort_register)
#define CIPHER_COHORT_QUERY \
	_IOWR(CIPHER_IOCTL_MAGIC, 31, struct cipher_cohort_query)

/* D.8 FAIRNESS + SHIELD — CIPHER_FAIRNESS_REGISTER (NR 32).
 *
 * Arms (or re-arms) the calling tenant's slot in the cross-tenant work-ledger
 * (cipher_fairness_ledger.c) and sets its SHIELD band. The kmod uses
 * current->tgid (the `tgid` field is informational). Returns the assigned
 * slot index in `out_slot`; the tenant then self-accounts its GEMM volume
 * into that slot via the RW mmap window at CIPHER_FAIR_VIEW_MMAP_PGOFF and
 * reads the whole region to compute its burst-fairness / SHIELD throttle
 * decision in userspace (libcipher_rt). Enforcement is timing-only and
 * entirely libcipher_rt-side; this ioctl only provides slot arbitration.
 *
 * band: 0 = throughput/default (eligible to be throttled under burst);
 *       >=1 = latency-sensitive (SHIELD-protected, never self-throttles).
 *
 * ADDITIVE: fresh NR 32; no existing NR changes. The reserved NRs (2/3/4)
 * still return -ENOSYS. */
struct cipher_fairness_register {
	__u32 tgid;       /* IN: informational; kmod uses current->tgid */
	__u32 band;       /* IN: SHIELD band (0 = throughput, >=1 = latency-sensitive) */
	__u32 out_slot;   /* OUT: assigned ledger slot index */
	__u32 flags;      /* IN: reserved, must be 0 */
};

#define CIPHER_FAIRNESS_REGISTER \
	_IOWR(CIPHER_IOCTL_MAGIC, 32, struct cipher_fairness_register)

#endif /* CIPHER_IOCTL_H */
