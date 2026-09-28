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
	__u32 reserved[16];
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

#endif /* CIPHER_IOCTL_H */
