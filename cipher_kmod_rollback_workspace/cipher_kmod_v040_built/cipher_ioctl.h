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

#endif /* CIPHER_IOCTL_H */
