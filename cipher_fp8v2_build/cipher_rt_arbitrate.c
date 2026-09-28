/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * ============================================================================
 * RETIRED — CP 5.4 Step 1.3 (2026-05-18).
 *
 * This translation unit is no longer built into libcipher_rt (removed from
 * the Makefile OBJS) and no longer called from anywhere. ARBITRATE drove SM
 * allocation through ioctl nr 9 (CIPHER_REQUEST_SM_PARTITION), which the
 * CP 5.4 kmod deactivates (returns -ENOSYS) — its 30 s poll thread would have
 * failed every wake. SM allocation is now the CP 5.4 group ledger:
 * cipher_rt_green_ctx_cp54_init() issues CIPHER_CP54_ALLOCATE (nr 13) at
 * injection-init. The CP 5.4 scheduler subsumes ARB's fair-share logic.
 * The file is kept in-tree, unbuilt, for historical reference only.
 * ============================================================================
 *
 * cipher_rt_arbitrate.c -- T4.2.3 ARBITRATE implementation + B7 poll-thread fix.
 *
 * Calls CIPHER_REQUEST_SM_PARTITION (ioctl nr 9) via a per-process fd
 * (NOT shared with cipher_rt_tenant.cpp's per-thread fd — this is a
 * cold-path + slow-cadence call, locked, so a shared fd is acceptable
 * here. It is NOT in the hot path).
 *
 * Hint derivation policy: quartile of this tenant's launches_total
 * among the tenants currently visible via /proc/cipher/stats enumerate.
 *
 * B7 fix (2026-05-14): periodic poll thread (30 s cadence) re-evaluates
 * the quartile rank and re-issues CIPHER_REQUEST_SM_PARTITION when the
 * derived hint changes. This closes Gap 2 from the T4.2.2 advisor review
 * — the original cold-path call fires at first-stream-observation when
 * launches_total is 0, producing degenerate ranks.
 *
 * Idempotency map (measured 2026-05-14 by cipher_test_b7_idempotency):
 *   hint grows  -> mask updates (grows)
 *   hint same   -> mask identical
 *   hint shrinks -> mask STAYS at the larger previous value (NO release)
 *
 * Consequence: B7 poll thread is **asymmetric grow-only**. On rank
 * improvement (smaller my_rank → larger hint), re-issue. On rank
 * degradation (larger my_rank → smaller hint), SKIP — the kmod won't
 * shrink anyway, so the ioctl would be wasted, and we'd misrepresent
 * the cached mask. Slot-pool can leak under tenant rank-down
 * transitions (a tenant that briefly hit top-quartile keeps the mask
 * forever). Documented limitation; deferred kmod-side fix tracked.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <pthread.h>

/* cipher_rt_tenant.h has a `struct cipher_tenant_snapshot_query` that
 * conflicts with the kernel's homonym in cipher_ioctl.h. Avoid pulling
 * cipher_ioctl.h into the same TU; instead, manually declare the bits
 * we need from it (CIPHER_REQUEST_SM_PARTITION and the request struct). */
#include <sys/ioctl.h>
#include <linux/ioctl.h>

#include "cipher_v2_internal.h"
#include "cipher_rt_arbitrate.h"
#include "cipher_rt_tenant.h"

/* Mirror of struct cipher_partition_request (cipher_ioctl.h nr 9).
 * B10 (kmod 0.4.5+): the last field is now `flags` (was `reserved`).
 * Layout unchanged; flags=0 means T4.2.1-era behavior. */
struct cipher_arb_request_local {
	unsigned int hint_partitions;
	unsigned int partition_mask_out;
	unsigned int partition_count_out;
	unsigned int flags;
};
#define CIPHER_ARB_IOCTL_MAGIC  'C'
#define CIPHER_ARB_REQUEST_SM_PARTITION \
	_IOWR(CIPHER_ARB_IOCTL_MAGIC, 9, struct cipher_arb_request_local)
#define CIPHER_ARB_FLAG_FIT_HINT  (1U << 0)

#define CIPHER_ARB_MAX_PEERS  64
#define CIPHER_ARB_POLL_CADENCE_SEC 30   /* B7: advisor-recommended 30s */

static int                 g_arb_fd = -1;
static pthread_mutex_t     g_arb_lock = PTHREAD_MUTEX_INITIALIZER;
static atomic_uint         g_last_mask  = 0;
static atomic_uint         g_last_count = 0;
static atomic_uint         g_last_hint  = 0;  /* B7: last hint we requested */
static atomic_int          g_arb_done   = 0;  /* per-process one-shot */
static atomic_int          g_arb_shutdown = 0;  /* B7: poll thread exit flag */
static pthread_t           g_arb_poll_tid;
static atomic_int          g_arb_poll_started = 0;
static atomic_ulong        g_arb_poll_iters   = 0;  /* diagnostic */
static atomic_ulong        g_arb_poll_grows   = 0;  /* count of mask grows */
static atomic_ulong        g_arb_poll_shrinks = 0;  /* count of mask shrinks (B10) */

/* Total SM partition slots in the kmod allocator (B6 cap = 32 on H100 SXM5).
 * Used by the B10 fair-share calculation. Should ideally be queried from
 * /proc/cipher but we hard-code matching the kmod constant for now. */
#define CIPHER_TOTAL_PARTITION_SLOTS  32

static int compare_launches_desc(const void *a, const void *b)
{
	const struct cipher_tenant_snapshot *ta = a;
	const struct cipher_tenant_snapshot *tb = b;
	if (tb->launches_total > ta->launches_total) return 1;
	if (tb->launches_total < ta->launches_total) return -1;
	return 0;
}

static unsigned derive_hint_from_quartile(unsigned my_rank, unsigned total)
{
	/* No effective contention -> top hint. With 32 slots and hint cap 8,
	 * the kmod allocator can serve up to 4 fully-greedy tenants without
	 * any contention; below that threshold quartile slicing produces
	 * pathologically small hints (solo tenant: total=1 → bottom). */
	if (total <= 4) return 8;
	if (my_rank < total / 4)        return 8;   /* top quartile */
	if (my_rank < total / 2)        return 6;   /* second quartile */
	if (my_rank < (3 * total) / 4)  return 4;   /* third quartile */
	return 2;                                    /* bottom quartile */
}

/* B7+B10 — fair-share-fit re-evaluation, used by the poll thread.
 * Defensive (NULL-safe). Returns 0 on success, -1 on defensive bail.
 *
 * Fair-share policy (advisor-binding, kmod 0.4.5 FIT_HINT enabled):
 *   - Enumerate N visible peers via /proc/cipher.
 *   - fair = max(1, min(8, TOTAL_SLOTS / N))  (e.g. N=8 → fair=4)
 *   - If currently held count > fair  → reissue with FIT_HINT, hint=fair (shrink)
 *   - If currently held count < fair  → reissue with FIT_HINT, hint=fair (grow)
 *   - If currently held count == fair → no-op
 *
 * Quartile policy from B7 is dropped at this layer — fair-share is the
 * base mechanism. Quartile-based unfair allocation can be layered on top
 * later if needed (e.g. top-quartile gets fair+1, bottom gets fair-1).
 */
static int cipher_rt_arb_reissue_fair_share(const char *reason)
{
	struct cipher_arb_request_local req;
	struct cipher_tenant_snapshot peers[CIPHER_ARB_MAX_PEERS];
	int npeers;
	unsigned fair, prev_hint, prev_count;
	int rc;
	int grow_or_shrink = 0;  /* +1 grow, -1 shrink, 0 no-op */

	if (g_arb_fd < 0) return -1;
	npeers = cipher_rt_tenant_enumerate(peers, CIPHER_ARB_MAX_PEERS);
	if (npeers < 1) npeers = 1;   /* solo */

	/* fair = clamp(TOTAL/N, 1, 8). 32/1=32→clamp 8; 32/8=4; 32/16=2. */
	fair = (unsigned)CIPHER_TOTAL_PARTITION_SLOTS / (unsigned)npeers;
	if (fair < 1) fair = 1;
	if (fair > 8) fair = 8;

	prev_hint  = atomic_load(&g_last_hint);
	prev_count = atomic_load(&g_last_count);

	if (prev_count == fair) {
		cipher_dbg("ARB-poll: fair=%u (npeers=%d) == held=%u, no-op (%s)",
		           fair, npeers, prev_count, reason);
		atomic_store(&g_last_hint, fair);   /* keep last_hint consistent */
		return 0;
	}
	if (prev_count > fair) grow_or_shrink = -1;
	else                   grow_or_shrink = +1;

	pthread_mutex_lock(&g_arb_lock);
	memset(&req, 0, sizeof(req));
	req.hint_partitions = fair;
	req.flags = CIPHER_ARB_FLAG_FIT_HINT;
	rc = ioctl(g_arb_fd, CIPHER_ARB_REQUEST_SM_PARTITION, &req);
	if (rc != 0) {
		cipher_log("ARB-poll: FIT_HINT reissue rc=%d errno=%d", rc, errno);
		pthread_mutex_unlock(&g_arb_lock);
		return -1;
	}
	atomic_store(&g_last_mask,  req.partition_mask_out);
	atomic_store(&g_last_count, req.partition_count_out);
	atomic_store(&g_last_hint,  fair);
	if (grow_or_shrink > 0) atomic_fetch_add(&g_arb_poll_grows, 1);
	else                    atomic_fetch_add(&g_arb_poll_shrinks, 1);
	pthread_mutex_unlock(&g_arb_lock);

	cipher_log("ARB-poll: %s npeers=%d fair=%u (was held=%u), "
	           "granted mask=0x%08x count=%u (%s)",
	           grow_or_shrink > 0 ? "GROW" : "SHRINK",
	           npeers, fair, prev_count,
	           req.partition_mask_out, req.partition_count_out, reason);
	return 0;
}

/* B7 — poll thread body. Wakes every CIPHER_ARB_POLL_CADENCE_SEC seconds
 * and re-evaluates the quartile rank. Defensive: any unexpected condition
 * logs and continues; thread MUST NOT crash the host process. */
static void *cipher_rt_arb_poll_thread(void *arg)
{
	(void)arg;
	cipher_log("ARB-poll: thread started, cadence=%ds",
	           CIPHER_ARB_POLL_CADENCE_SEC);

	while (!atomic_load(&g_arb_shutdown)) {
		/* Use a coarse sleep with short check granularity so shutdown
		 * is responsive without burning CPU. */
		for (int i = 0; i < CIPHER_ARB_POLL_CADENCE_SEC; i++) {
			if (atomic_load(&g_arb_shutdown)) goto out;
			sleep(1);
		}
		atomic_fetch_add(&g_arb_poll_iters, 1);
		(void)cipher_rt_arb_reissue_fair_share("poll");
	}
out:
	cipher_log("ARB-poll: thread exiting after %lu iterations, %lu grows",
	           atomic_load(&g_arb_poll_iters),
	           atomic_load(&g_arb_poll_grows));
	return NULL;
}

int cipher_rt_arb_init(void)
{
	pthread_attr_t attr;
	int rc;

	g_arb_fd = open("/dev/cipher", O_RDWR);
	if (g_arb_fd < 0) {
		cipher_log("ARB: open(/dev/cipher) failed: %s", strerror(errno));
		return -1;
	}
	cipher_log("ARB: arbitrate initialized (fd=%d)", g_arb_fd);

	/* B7: spawn the poll thread (detached). Failure to spawn is logged
	 * but not fatal — ARBITRATE then degrades to the original one-shot
	 * cold-path-only behavior. */
	int started_expected = 0;
	if (!atomic_compare_exchange_strong(&g_arb_poll_started,
	                                    &started_expected, 1)) {
		cipher_log("ARB-poll: already started, skipping spawn");
		return 0;
	}
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	rc = pthread_create(&g_arb_poll_tid, &attr,
	                    cipher_rt_arb_poll_thread, NULL);
	pthread_attr_destroy(&attr);
	if (rc != 0) {
		cipher_log("ARB-poll: pthread_create failed rc=%d; degrading to one-shot ARBITRATE",
		           rc);
		atomic_store(&g_arb_poll_started, 0);
	}
	return 0;
}

void cipher_rt_arb_close(void)
{
	/* Set shutdown flag — poll thread wakes within 1s and exits. */
	atomic_store(&g_arb_shutdown, 1);
	if (g_arb_fd >= 0) {
		/* Keep fd open for any final in-flight ioctl; close at exit. */
	}
}

unsigned int cipher_rt_arb_request_partition(void)
{
	struct cipher_arb_request_local req;
	struct cipher_tenant_snapshot peers[CIPHER_ARB_MAX_PEERS];
	int npeers = 0, my_rank = 0, found_self = 0;
	pid_t my_pid;
	unsigned hint = 4;
	int rc;
	int expected = 0;

	/* One-shot per process: subsequent calls return cached mask. */
	if (!atomic_compare_exchange_strong(&g_arb_done, &expected, 1))
		return atomic_load(&g_last_mask);

	if (g_arb_fd < 0) {
		cipher_log("ARB: fd not open, skipping request");
		return 0;
	}

	pthread_mutex_lock(&g_arb_lock);

	/* Enumerate active tenants and find our rank. */
	my_pid = getpid();
	npeers = cipher_rt_tenant_enumerate(peers, CIPHER_ARB_MAX_PEERS);
	if (npeers > 0) {
		qsort(peers, npeers, sizeof(peers[0]), compare_launches_desc);
		for (int i = 0; i < npeers; i++) {
			if ((pid_t)peers[i].pid == my_pid ||
			    (pid_t)peers[i].tgid == my_pid) {
				my_rank = i;
				found_self = 1;
				break;
			}
		}
		if (!found_self)
			my_rank = npeers;   /* treat as bottom */
		hint = derive_hint_from_quartile((unsigned)my_rank,
		                                 (unsigned)npeers);
		cipher_dbg("ARB: rank %d/%d, hint=%u", my_rank, npeers, hint);
	} else {
		hint = 4;   /* solo or pre-enumerate: middle hint */
		cipher_dbg("ARB: enumerate found no peers, hint=4 default");
	}

	memset(&req, 0, sizeof(req));
	req.hint_partitions = hint;
	rc = ioctl(g_arb_fd, CIPHER_ARB_REQUEST_SM_PARTITION, &req);
	if (rc != 0) {
		cipher_log("ARB: REQUEST_SM_PARTITION rc=%d errno=%d %s",
		           rc, errno, strerror(errno));
		pthread_mutex_unlock(&g_arb_lock);
		return 0;
	}

	atomic_store(&g_last_mask,  req.partition_mask_out);
	atomic_store(&g_last_count, req.partition_count_out);
	atomic_store(&g_last_hint,  hint);   /* B7: record initial hint */
	cipher_log("ARB: granted mask=0x%08x count=%u (hint=%u)",
	           req.partition_mask_out, req.partition_count_out, hint);

	pthread_mutex_unlock(&g_arb_lock);
	return req.partition_mask_out;
}

unsigned int cipher_rt_arb_last_mask(void)
{
	return atomic_load(&g_last_mask);
}

unsigned int cipher_rt_arb_last_count(void)
{
	return atomic_load(&g_last_count);
}
