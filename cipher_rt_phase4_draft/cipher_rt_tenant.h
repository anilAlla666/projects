/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_tenant.h — Phase 4 actuator-side API for tenant-aware fusion.
 *
 * Three access modes:
 *   Mode 1 (HOT, ioctl)       — sub-5µs per call; per-launch decisions
 *   Mode 2 (COLD, /proc poll) — 100ms refresh; arbitration / Stage 2 threads
 *   Mode 3 (CACHED, TLS)      — sub-µs; per-launch hot path, refresh on
 *                                each Stage 0 ring write
 *
 * Header is consumed by every cipher_rt actuator that needs per-tenant
 * state. The cached snapshot lives in a thread-local struct refreshed
 * by Stage 0 ring-write hook; readers in SUBSTITUTE/ARBITRATE/etc.
 * touch it without a syscall.
 *
 * STAGED: not yet integrated into cipher_rt build. Lands in T4.1+
 * when cipher_kmod 0.4.0 ships and ioctl nr 8 is live.
 */
#ifndef CIPHER_RT_TENANT_H
#define CIPHER_RT_TENANT_H

#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>

/* Match kernel-side cipher_internal.h definitions. Keep in sync with
 * cipher_kmod/cipher_ioctl.h once T4.1.2 publishes the user-facing copy. */

#ifndef CIPHER_TENANT_ID_LEN
#  define CIPHER_TENANT_ID_LEN  64
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* User-facing mirror of struct cipher_tenant_snapshot.
 * Identical layout to kernel side; receive via ioctl nr 8. */
struct cipher_tenant_snapshot {
	char     tenant_id_str[CIPHER_TENANT_ID_LEN];
	uint64_t tenant_session_fp;
	uint32_t tenant_handle_u32;
	uint32_t pid;
	uint32_t tgid;
	uint32_t _pad_identity;

	uint32_t sm_util_pct;
	uint32_t mem_util_pct;
	uint64_t launches_total;
	uint64_t grid_ops_total;
	uint64_t ioctls_total;

	uint32_t sm_partition_mask;
	uint32_t sm_partition_count;

	uint32_t thermal_headroom_pct;
	uint32_t power_headroom_w;
	uint32_t sustained_clock_mhz;
	uint32_t voltage_envelope_mv;

	uint32_t l2_residency_kb;
	uint32_t hot_region_count;
	uint64_t predicted_hot_regions[8];

	uint32_t kv_cache_size_mb;
	uint32_t kv_compression_ratio_pct;

	uint32_t weight_dedup_savings_mb;
	uint32_t weight_content_hash_count;

	uint32_t fusion_recipe_id;
	uint32_t fusion_eligible_flag;

	uint32_t session_band;
	uint32_t slo_priority;
	uint32_t fairness_quota_remaining_pct;
	uint32_t _pad_agentic;

	uint32_t graph_capture_state;
	uint32_t koopman_substitution_eligibility;

	uint64_t snapshot_jiffies;
	uint32_t reserved[16];
};

/* Query envelope for ioctl nr 8 (CIPHER_GET_TENANT_SNAPSHOT). */
struct cipher_tenant_snapshot_query {
	uint32_t target_pid;                          /* 0 = use target_tenant_id */
	char     target_tenant_id[CIPHER_TENANT_ID_LEN];
	struct cipher_tenant_snapshot snapshot;       /* filled by kernel */
};

/* ----- Mode 1: hot ioctl path ----- */

/* Open /dev/cipher and cache the fd for reuse. Returns 0 on success.
 * Caller responsibility: call once at cipher_rt init. */
int cipher_rt_tenant_open(void);

/* Per-call ioctl. ~5 µs. Use sparingly — prefer Mode 3 cache. */
int cipher_rt_tenant_query_by_pid(pid_t pid,
                                  struct cipher_tenant_snapshot *out);
int cipher_rt_tenant_query_by_id(const char *tenant_id,
                                 struct cipher_tenant_snapshot *out);

/* ----- Mode 2: cold /proc/cipher/stats poll ----- */

/* Enumerate all tenants by parsing /proc/cipher/stats. Used by ARBITRATE
 * and Stage 2 threads at 100 ms cadence. Returns count copied.
 * `out` must have room for `max` entries. */
int cipher_rt_tenant_enumerate(struct cipher_tenant_snapshot *out, int max);

/* ----- Mode 3: thread-local cached snapshot ----- */

/* Refreshes the current thread's cached snapshot. Called by the Stage 0
 * ring-write hook after every successful ring entry. Cheap; reads the
 * already-open ioctl fd, ~5 µs once per launch — but the SUBSTITUTE /
 * ARBITRATE / per-launch routing actuators read the cache in ~10 ns. */
int cipher_rt_tenant_refresh_cached(void);

/* Hot-path read. Returns pointer to the thread-local cached snapshot.
 * Pointer valid until next refresh on the same thread. Caller is
 * responsible for handling NULL (no tenant registered for this LWP yet). */
const struct cipher_tenant_snapshot *cipher_rt_tenant_cached(void);

/* Cache freshness in nanoseconds since last refresh. Use to decide
 * whether to skip a stale-cache read. */
uint64_t cipher_rt_tenant_cached_age_ns(void);

/* ----- Lifecycle ----- */

void cipher_rt_tenant_close(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CIPHER_RT_TENANT_H */
