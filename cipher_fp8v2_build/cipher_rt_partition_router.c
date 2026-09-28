/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_partition_router.c -- Phase 4 T4.2.2 PARTITION_ROUTER actuator.
 *
 * See cipher_rt_partition_router.h for the design.
 *
 * Implementation notes:
 *  - Per-stream cache is a fixed-size open-addressed hashtable. 256
 *    entries is plenty for steady-state CUDA workloads (PyTorch typically
 *    uses ≤16 streams; vLLM ≤32). Capacity exceedance is logged once.
 *  - Inserts protected by a process-wide spinlock to serialize concurrent
 *    "first observation" attempts from multi-threaded launches. Lookups
 *    are lock-free (atomic_load on the slot's stream-handle field).
 *  - Tenant snapshot fetched via cipher_rt_tenant_query_by_pid() — that
 *    function lazily opens a per-thread /dev/cipher fd per B1 fix.
 *  - Stream attribute writes use cuStreamSetAttribute(); the priority is
 *    derived from launches_total decile-band (0..8) and the sync-domain
 *    map from tenant_handle_u32 % MEM_SYNC_DOMAIN_COUNT.
 */
#include <pthread.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>

#include <cuda.h>

#include "cipher_v2_internal.h"
#include "cipher_rt_partition_router.h"
#include "cipher_rt_tenant.h"
/* CP 5.4 Step 1.3: cipher_rt_arbitrate.h retired (ARB / ioctl nr 9). */

#define CIPHER_RT_PR_CACHE_SLOTS    256
#define CIPHER_RT_PR_PRIORITY_HIGH  (-5)   /* H100 supports -5..0 */
#define CIPHER_RT_PR_PRIORITY_LOW   ( 0)

/* Slot is "free" iff stream_handle == 0 AND configured == 0. We use the
 * combination because CUDA's "legacy NULL stream" has handle 0 which we
 * also want to track exactly once. The configured flag disambiguates. */
struct cipher_rt_pr_slot {
	_Atomic uintptr_t stream_handle;
	_Atomic int       configured;
};

static struct cipher_rt_pr_slot g_slots[CIPHER_RT_PR_CACHE_SLOTS];
static pthread_mutex_t          g_pr_lock = PTHREAD_MUTEX_INITIALIZER;
static atomic_ulong             g_observed_count   = 0;
static atomic_ulong             g_configured_count = 0;
static atomic_int               g_pr_initialized   = 0;
static atomic_int               g_pr_capacity_warned = 0;

/* Cached tenant identity for derive_priority / derive_sync_domain.
 * Updated lazily from cipher_rt_tenant snapshot per process. */
static atomic_uint              g_cached_tenant_handle = 0;
static atomic_ulong             g_cached_launches_total = 0;
static atomic_ulong             g_tenant_last_refresh_ns = 0;
#define TENANT_REFRESH_INTERVAL_NS  (500ULL * 1000000ULL)  /* 500 ms */

static uint64_t monotonic_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void cipher_rt_pr_refresh_tenant(void);

/* T4.2.4b: getter for the cached tenant_handle. Consumed by
 * cipher_rt_green_ctx.c to pick a per-process green-ctx group.
 * Returns 0 if tenant snapshot has not been refreshed yet (this is a
 * defensible fallback — caller will simply pick group 0 deterministically). */
unsigned int cipher_rt_tenant_handle_u32_for_self(void)
{
	unsigned int h = atomic_load_explicit(&g_cached_tenant_handle,
	                                      memory_order_relaxed);
	if (h == 0) {
		cipher_rt_pr_refresh_tenant();
		h = atomic_load_explicit(&g_cached_tenant_handle,
		                         memory_order_relaxed);
	}
	return h;
}

static void cipher_rt_pr_refresh_tenant(void)
{
	struct cipher_tenant_snapshot snap;
	int rc;

	rc = cipher_rt_tenant_query_by_pid((pid_t)syscall(SYS_gettid), &snap);
	if (rc != 0) {
		cipher_dbg("PR: tenant query failed rc=%d", rc);
		return;
	}
	atomic_store_explicit(&g_cached_tenant_handle, snap.tenant_handle_u32,
	                      memory_order_relaxed);
	atomic_store_explicit(&g_cached_launches_total, snap.launches_total,
	                      memory_order_relaxed);
	atomic_store_explicit(&g_tenant_last_refresh_ns, monotonic_ns(),
	                      memory_order_relaxed);
}

static int derive_priority_from_launches(uint64_t launches)
{
	/* Coarse heuristic: tenants with > 1M cumulative launches are
	 * "hot" and get priority -5 (highest on H100). Below that, default
	 * priority 0. Refined by Task 4 ARBITRATE (quartile rebalance). */
	if (launches > 1000000ULL) return CIPHER_RT_PR_PRIORITY_HIGH;
	return CIPHER_RT_PR_PRIORITY_LOW;
}

static void derive_sync_domain_map(uint32_t tenant_handle, CUlaunchMemSyncDomainMap *out)
{
	/* Currently a no-op (default_=0, remote=0). The previous attempt to
	 * spread tenants across the two H100 SXM5 sync domains via
	 * (tenant_handle % 2) caused a measurable throughput regression on
	 * single-tenant compute-bound workloads (WL14 -5.7% tok/s in T4.2.2
	 * dry run) because setting default_!=0 inserts system-scope fences on
	 * any memory op crossing into the new domain — even within the same
	 * tenant, when intermediate kernels use a different domain.
	 *
	 * Proper sync-domain partitioning requires knowing the workload's
	 * cross-tenant memory access pattern (which we do not have yet).
	 * Deferred until cipher_rt has per-tenant access-pattern telemetry
	 * (PHASE_4_BACKLOG.md candidate for B7 if not opened already). */
	(void)tenant_handle;
	out->default_ = 0;
	out->remote   = 0;
}

int cipher_rt_pr_init(void)
{
	int expected = 0;
	cipher_log("PR: cipher_rt_pr_init entered");
	if (!atomic_compare_exchange_strong(&g_pr_initialized, &expected, 1)) {
		cipher_log("PR: already initialized, returning");
		return 0;
	}
	for (int i = 0; i < CIPHER_RT_PR_CACHE_SLOTS; i++) {
		atomic_store(&g_slots[i].stream_handle, 0);
		atomic_store(&g_slots[i].configured, 0);
	}
	cipher_rt_pr_refresh_tenant();
	cipher_log("PR: partition router initialized (cache=%d slots, "
	           "tenant_handle=0x%08x)",
	           CIPHER_RT_PR_CACHE_SLOTS,
	           atomic_load(&g_cached_tenant_handle));
	return 0;
}

/* Fast lookup: scan the open-addressed table for matching stream handle.
 * Returns slot index on hit, -1 on miss. Lock-free (atomic loads only). */
static int cipher_rt_pr_lookup(uintptr_t needle)
{
	uintptr_t h = (needle * 2654435761ULL) & (CIPHER_RT_PR_CACHE_SLOTS - 1);
	for (int probe = 0; probe < CIPHER_RT_PR_CACHE_SLOTS; probe++) {
		int idx = (int)((h + probe) & (CIPHER_RT_PR_CACHE_SLOTS - 1));
		uintptr_t s = atomic_load_explicit(&g_slots[idx].stream_handle,
		                                   memory_order_acquire);
		if (s == needle &&
		    atomic_load_explicit(&g_slots[idx].configured,
		                         memory_order_relaxed))
			return idx;
		if (s == 0)
			return -1;
	}
	return -1;
}

/* Slow path: insert + configure the stream. Lock held. */
static int cipher_rt_pr_configure_locked(uintptr_t needle)
{
	CUresult cr;
	CUstreamAttrValue val;
	int priority;
	CUlaunchMemSyncDomainMap dmap;
	uintptr_t h, s;
	int idx, probe;
	uint64_t now;

	/* Re-check under lock — another thread may have just configured this. */
	idx = cipher_rt_pr_lookup(needle);
	if (idx >= 0)
		return 0;

	/* Find an empty slot via linear probe. */
	h = (needle * 2654435761ULL) & (CIPHER_RT_PR_CACHE_SLOTS - 1);
	idx = -1;
	for (probe = 0; probe < CIPHER_RT_PR_CACHE_SLOTS; probe++) {
		int i = (int)((h + probe) & (CIPHER_RT_PR_CACHE_SLOTS - 1));
		s = atomic_load_explicit(&g_slots[i].stream_handle,
		                         memory_order_relaxed);
		if (s == 0) { idx = i; break; }
	}
	if (idx < 0) {
		if (!atomic_load(&g_pr_capacity_warned)) {
			atomic_store(&g_pr_capacity_warned, 1);
			cipher_log("PR: stream cache full (%d slots) -- subsequent "
			           "streams will not be configured. Workload may have "
			           "more streams than expected; raise "
			           "CIPHER_RT_PR_CACHE_SLOTS if needed.",
			           CIPHER_RT_PR_CACHE_SLOTS);
		}
		return -1;
	}

	/* Refresh tenant snapshot if stale. */
	now = monotonic_ns();
	if (now - atomic_load(&g_tenant_last_refresh_ns) > TENANT_REFRESH_INTERVAL_NS)
		cipher_rt_pr_refresh_tenant();

	/* CP 5.4 Step 1.3: the T4.2.3 ARBITRATE call (ioctl nr 9) is retired.
	 * nr 9 returns -ENOSYS on the CP 5.4 kmod; SM allocation now goes through
	 * the CP 5.4 group ledger (CIPHER_CP54_ALLOCATE, issued at injection-init
	 * by cipher_rt_green_ctx_cp54_init). The partition router keeps only its
	 * stream PRIORITY + MEM_SYNC_DOMAIN_MAP duties below. */

	/* Set PRIORITY. */
	priority = derive_priority_from_launches(
	               atomic_load_explicit(&g_cached_launches_total,
	                                    memory_order_relaxed));
	memset(&val, 0, sizeof(val));
	val.priority = priority;
	cr = cuStreamSetAttribute((CUstream)needle,
	                          CU_STREAM_ATTRIBUTE_PRIORITY, &val);
	if (cr != CUDA_SUCCESS) {
		cipher_dbg("PR: cuStreamSetAttribute(PRIORITY=%d) -> %d "
		           "for stream=%p", priority, cr, (void *)needle);
		/* Continue: a stream that doesn't accept priority (e.g.,
		 * legacy NULL stream) is still recorded so we don't keep
		 * retrying. */
	}

	/* Set MEM_SYNC_DOMAIN_MAP. */
	derive_sync_domain_map(
	    atomic_load_explicit(&g_cached_tenant_handle, memory_order_relaxed),
	    &dmap);
	memset(&val, 0, sizeof(val));
	val.memSyncDomainMap = dmap;
	cr = cuStreamSetAttribute((CUstream)needle,
	                          CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, &val);
	if (cr != CUDA_SUCCESS) {
		cipher_dbg("PR: cuStreamSetAttribute(MEM_SYNC_DOMAIN_MAP "
		           "default=%u remote=%u) -> %d for stream=%p",
		           dmap.default_, dmap.remote, cr, (void *)needle);
	}

	/* Record the configuration. Order matters: stream_handle first (so
	 * a concurrent lookup sees the slot occupied), configured last (so
	 * a concurrent lookup either sees configured==1 or treats the entry
	 * as in-progress and falls through). */
	atomic_store_explicit(&g_slots[idx].stream_handle, needle,
	                      memory_order_release);
	atomic_store_explicit(&g_slots[idx].configured, 1,
	                      memory_order_release);
	atomic_fetch_add_explicit(&g_configured_count, 1, memory_order_relaxed);
	cipher_dbg("PR: configured stream=%p priority=%d sync_map={%u,%u}",
	           (void *)needle, priority, dmap.default_, dmap.remote);
	return 0;
}

int cipher_rt_pr_observe_stream(void *stream_handle)
{
	uintptr_t needle = (uintptr_t)stream_handle;
	int idx;

	atomic_fetch_add_explicit(&g_observed_count, 1, memory_order_relaxed);

	idx = cipher_rt_pr_lookup(needle);
	if (idx >= 0)
		return 0;   /* hot path: already configured */

	pthread_mutex_lock(&g_pr_lock);
	cipher_rt_pr_configure_locked(needle);
	pthread_mutex_unlock(&g_pr_lock);
	return 0;
}

unsigned long cipher_rt_pr_observed_count(void)
{
	return atomic_load_explicit(&g_observed_count, memory_order_relaxed);
}

unsigned long cipher_rt_pr_configured_count(void)
{
	return atomic_load_explicit(&g_configured_count, memory_order_relaxed);
}
