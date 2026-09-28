/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_green_ctx.c -- per-process Green Context with real SM partition.
 *
 * CP 5.4 Step 1.3 (this build): the green context is driven by the kmod
 * CP 5.4 group ledger, replacing the prior `pid ^ tenant_handle` hash-pick.
 *
 *   - At injection-init, cipher_rt_green_ctx_cp54_init() reads CIPHER_QOS_CLASS
 *     / CIPHER_SM_COUNT, issues CIPHER_CP54_ALLOCATE (ioctl nr 13) on
 *     /dev/cipher, and caches the returned grp_mask. ALLOCATE runs on the
 *     long-lived injection-init thread (CP 5.4 Step 1.3 Q4) so the ledger
 *     entry's lifetime tracks the process, not a transient stream thread.
 *   - At first cuStreamCreate, cipher_rt_green_ctx_ensure() consumes the
 *     cached grp_mask: cuDevSmResourceSplitByCount(MIN_SM=8) → the 8-SM group
 *     resources; the set bits of grp_mask select this tenant's groups;
 *     cuDevResourceGenerateDesc over their union → a VARIABLE-size green ctx.
 *
 * qos_class routing (Step 1.3 scope — single-tenant PARTITION + SHARED):
 *   PARTITION — grp_mask is non-zero; green ctx is built over those groups.
 *   SHARED    — grp_mask is zero; this tenant owns no groups and creates no
 *               green context (it runs on the primary context). It will bind
 *               to the batch pool's green context once CP 5.4 Step 1.3b'
 *               (POOL / executor wiring) lands.
 *   POOL      — handled like PARTITION here (green ctx over the granted mask);
 *               full POOL/executor integration is Step 1.3b'.
 *
 * Group count: cuDevSmResourceSplitByCount(MIN_SM=8) on this H100 80GB SXM5
 * returns 15 8-SM groups + 12-SM remainder (NOT 16 + 4) — measured by the
 * CP 5.4 Step 1.3a probe (step1_3/PHASE_1_3A_PROBE.md); the kmod ledger is
 * sized to match (CIPHER_CP54_NUM_GROUPS = 15). This file never hard-codes
 * the count — it uses whatever the split returns.
 *
 * Fallback: if CP 5.4 ALLOCATE is unavailable (no /dev/cipher, ioctl failure),
 * cipher_rt_green_ctx_ensure() degrades to the prior hash-pick of a single
 * 8-SM group — graceful, no crash.
 *
 * Refresh policy: green ctx is created once at first cuStreamCreate and never
 * refreshed (PyTorch reuses warmup streams; destroying it mid-run would strand
 * them).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdatomic.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>

#include <cuda.h>

#include "cipher_v2_internal.h"
#include "cipher_rt_green_ctx.h"
#include "cipher_ioctl.h"          /* CP 5.4 ABI: CIPHER_CP54_ALLOCATE etc. */

/* H100 (compute 9.0+) green-ctx SM partition minimum is 8 SMs, multiples of 8.
 * CIPHER_RT_GREEN_MAX_GROUPS is an array-capacity upper bound only — the live
 * group count is whatever cuDevSmResourceSplitByCount returns (15 on this
 * H100; see the file header). */
#define CIPHER_RT_GREEN_MAX_GROUPS  16
#define CIPHER_RT_GREEN_MIN_SM      8

static CUgreenCtx       g_green_ctx       = NULL;
static CUcontext        g_green_cuctx     = NULL;
static atomic_int       g_green_init_attempted = 0;
static atomic_int       g_green_init_succeeded = 0;
static atomic_ulong     g_streams_observed = 0;
static pthread_mutex_t  g_green_lock      = PTHREAD_MUTEX_INITIALIZER;
static unsigned         g_green_group_id  = 0;    /* lowest group id bound */
static unsigned         g_green_sm_count  = 0;    /* verified SM count of green ctx */
static unsigned         g_green_cur_mask  = 0;    /* Track 3 SC3: mask the live
                                                   * green ctx was built from */

/* CP 5.4 cached allocation (filled by cipher_rt_green_ctx_cp54_init). */
static atomic_int g_cp54_init_done = 0;
static int        g_cp54_ok        = 0;   /* 1 = ALLOCATE succeeded */
static unsigned   g_cp54_qos       = CIPHER_CP54_QOS_SHARED;
static unsigned   g_cp54_grp_mask  = 0;   /* granted 8-SM group bitmask */
static unsigned   g_cp54_grp_count = 0;   /* popcount(grp_mask) */

/* Forward-declared accessor (defined in cipher_rt_partition_router.c). Used
 * ONLY by the hash-pick fallback path below. */
extern unsigned int cipher_rt_tenant_handle_u32_for_self(void);

int cipher_rt_green_ctx_is_initialized(void)
{
	return atomic_load(&g_green_init_succeeded);
}

unsigned long cipher_rt_green_ctx_streams_observed(void)
{
	return atomic_load(&g_streams_observed);
}

unsigned int cipher_rt_green_ctx_sm_count(void)
{
	if (!atomic_load(&g_green_init_succeeded))
		return 0;
	return g_green_sm_count;
}

unsigned int cipher_rt_green_ctx_group_id(void)
{
	if (!atomic_load(&g_green_init_succeeded))
		return 0;
	return g_green_group_id;
}

void *cipher_rt_green_ctx_handle_voidp(void)
{
	return (void *)g_green_ctx;
}

/* ---- CP 5.4 ALLOCATE (Step 1.3 Q4: run at injection-init) ---------------- */

static unsigned cipher_rt_qos_from_env(void)
{
	const char *q = getenv("CIPHER_QOS_CLASS");

	if (q && q[0]) {
		if (!strcasecmp(q, "partition")) return CIPHER_CP54_QOS_PARTITION;
		if (!strcasecmp(q, "pool"))      return CIPHER_CP54_QOS_POOL;
		if (!strcasecmp(q, "shared"))    return CIPHER_CP54_QOS_SHARED;
		cipher_log("GREEN/CP54: unknown CIPHER_QOS_CLASS='%s' — defaulting "
		           "to shared", q);
	}
	return CIPHER_CP54_QOS_SHARED;
}

/* Issue CIPHER_CP54_ALLOCATE and cache the granted grp_mask. Idempotent
 * (one-shot). Safe to call from any thread; intended to be called once at
 * injection-init. On any failure, leaves g_cp54_ok=0 → ensure() falls back
 * to the hash-pick path. Returns 0 if the allocation was cached, -1 otherwise. */
int cipher_rt_green_ctx_cp54_init(void)
{
	struct cipher_cp54_allocate a;
	unsigned qos, sm_count = 0;
	int fd, rc;

	if (atomic_exchange(&g_cp54_init_done, 1))
		return g_cp54_ok ? 0 : -1;          /* one-shot */

	qos = cipher_rt_qos_from_env();
	if (qos == CIPHER_CP54_QOS_PARTITION) {
		const char *s = getenv("CIPHER_SM_COUNT");

		if (s && s[0])
			sm_count = (unsigned)strtoul(s, NULL, 10);
		if (sm_count == 0) {
			sm_count = CIPHER_RT_GREEN_MIN_SM;
			cipher_log("GREEN/CP54: qos=partition with no/zero "
			           "CIPHER_SM_COUNT — defaulting to %u SMs", sm_count);
		}
	}

	fd = open("/dev/cipher", O_RDWR);
	if (fd < 0) {
		cipher_log("GREEN/CP54: open(/dev/cipher) failed (%s) — green ctx "
		           "will fall back to hash-pick", strerror(errno));
		return -1;
	}

	memset(&a, 0, sizeof(a));
	a.qos_class = qos;
	a.sm_count  = sm_count;
	rc = ioctl(fd, CIPHER_CP54_ALLOCATE, &a);

	/* Track 3 SC3 — opt-in plumbing (SC1 item 8: env var, no libcipher_v2
	 * change). A PARTITION tenant that sets CIPHER_MIGRATABLE declares
	 * itself migratable to the kmod ledger; unset -> pinned (default,
	 * byte-identical to today). Done on the same fd, before close. */
	if (rc == 0 && qos == CIPHER_CP54_QOS_PARTITION) {
		const char *m = getenv("CIPHER_MIGRATABLE");

		if (m && (m[0] == '1' || !strcasecmp(m, "true") ||
		          !strcasecmp(m, "yes"))) {
			unsigned int one = 1;
			int src = ioctl(fd, CIPHER_CP54_SUBSCRIBE_MIGRATE, &one);

			cipher_log("GREEN/CP54: CIPHER_MIGRATABLE set — "
			           "SUBSCRIBE_MIGRATE(1) rc=%d", src);
		}
	}
	close(fd);
	if (rc != 0) {
		cipher_log("GREEN/CP54: ALLOCATE(qos=%u sm_count=%u) failed "
		           "rc=%d errno=%d (%s) — fall back to hash-pick",
		           qos, sm_count, rc, errno, strerror(errno));
		return -1;
	}

	g_cp54_qos       = qos;
	g_cp54_grp_mask  = a.grp_mask_out;
	g_cp54_grp_count = a.grp_count_out;
	g_cp54_ok        = 1;
	cipher_log("GREEN/CP54: ALLOCATE ok — qos=%u sm_count=%u -> "
	           "grp_mask=0x%04x grp_count=%u",
	           qos, sm_count, a.grp_mask_out, a.grp_count_out);
	return 0;
}

/* T4.2.4d enforcement: persistently set the green CUcontext as the calling
 * thread's current. Skips cuCtxSetCurrent if already green. */
int cipher_rt_green_ctx_make_current(void)
{
	CUcontext cur = NULL;
	CUresult cr;
	if (!atomic_load(&g_green_init_succeeded))
		return -1;
	cr = cuCtxGetCurrent(&cur);
	if (cr == CUDA_SUCCESS && cur == g_green_cuctx)
		return 0;  /* already green */
	cr = cuCtxSetCurrent(g_green_cuctx);
	if (cr != CUDA_SUCCESS) {
		cipher_dbg("GREEN: cuCtxSetCurrent rc=%d", cr);
		return -1;
	}
	return 0;
}

int cipher_rt_green_ctx_ensure(void)
{
	CUdevice dev = 0;
	CUdevResource full_sm_res;
	CUdevResource groups[CIPHER_RT_GREEN_MAX_GROUPS];
	CUdevResource selected[CIPHER_RT_GREEN_MAX_GROUPS];
	CUdevResource remaining;
	CUdevResourceDesc desc;
	CUdevResource verify_res;
	CUresult cr;
	unsigned int nb_groups = CIPHER_RT_GREEN_MAX_GROUPS;
	unsigned int nsel = 0;
	unsigned int g;
	int use_fallback = 0;

	if (atomic_load(&g_green_init_succeeded))
		return 0;

	/* CP 5.4 allocation is normally done at injection-init; ensure it has
	 * run (idempotent) in case of an unusual init ordering. */
	if (!atomic_load(&g_cp54_init_done))
		(void)cipher_rt_green_ctx_cp54_init();

	pthread_mutex_lock(&g_green_lock);
	if (atomic_load(&g_green_init_succeeded)) {
		pthread_mutex_unlock(&g_green_lock);
		return 0;
	}
	if (atomic_exchange(&g_green_init_attempted, 1)) {
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}

	/* CP 5.4 SHARED tenant: owns no groups → no own green context. It runs
	 * on the primary context until it binds to the batch pool's green ctx
	 * (CP 5.4 Step 1.3b'). This is an intentional no-op, not a failure. */
	if (g_cp54_ok && g_cp54_grp_mask == 0) {
		cipher_log("GREEN: CP 5.4 qos=shared — tenant owns no SM groups; "
		           "no green context created (runs on primary context; "
		           "pool binding lands in Step 1.3b')");
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}

	/* Step 1: full SM resource set. */
	cr = cuDeviceGetDevResource(dev, &full_sm_res, CU_DEV_RESOURCE_TYPE_SM);
	if (cr != CUDA_SUCCESS) {
		cipher_log("GREEN: cuDeviceGetDevResource rc=%d", cr);
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}

	/* Step 2: split into 8-SM groups. nb_groups is set by the API to the
	 * actual count (15 on this H100). */
	cr = cuDevSmResourceSplitByCount(groups, &nb_groups, &full_sm_res,
	                                  &remaining, 0, CIPHER_RT_GREEN_MIN_SM);
	if (cr != CUDA_SUCCESS || nb_groups == 0) {
		cipher_log("GREEN: cuDevSmResourceSplitByCount rc=%d nb=%u — abort",
		           cr, nb_groups);
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}
	cipher_log("GREEN: device split into %u × %u-SM groups (%u SMs remain)",
	           nb_groups, CIPHER_RT_GREEN_MIN_SM, remaining.sm.smCount);

	/* Step 3: choose this tenant's groups. */
	if (g_cp54_ok && g_cp54_grp_mask != 0) {
		/* CP 5.4 path: select the groups named by the kmod's grp_mask. */
		if (g_cp54_grp_mask >> nb_groups) {
			/* a granted bit has no backing split group — kmod/hardware
			 * group-count mismatch. Defensive: do not build a bogus
			 * partition; fall back. */
			cipher_log("GREEN: CP 5.4 grp_mask=0x%04x has bit(s) >= "
			           "split count %u — falling back to hash-pick",
			           g_cp54_grp_mask, nb_groups);
			use_fallback = 1;
		} else {
			for (g = 0; g < nb_groups; g++)
				if (g_cp54_grp_mask & (1U << g))
					selected[nsel++] = groups[g];
			g_green_group_id = __builtin_ctz(g_cp54_grp_mask);
			cipher_log("GREEN: CP 5.4 grp_mask=0x%04x → %u group(s), "
			           "lowest id %u", g_cp54_grp_mask, nsel,
			           g_green_group_id);
		}
	} else {
		use_fallback = 1;
	}

	if (use_fallback) {
		/* Hash-pick fallback: one 8-SM group, selected by mixing
		 * tenant_handle with getpid() (xorshift-style — distributes
		 * uniformly even for collided tenant_handles). */
		unsigned int th = cipher_rt_tenant_handle_u32_for_self();
		unsigned int h = (unsigned int)getpid() ^ th;
		h ^= h >> 16; h *= 0x85ebca6bU;
		h ^= h >> 13; h *= 0xc2b2ae35U; h ^= h >> 16;
		g_green_group_id = h % nb_groups;
		selected[0] = groups[g_green_group_id];
		nsel = 1;
		cipher_log("GREEN: hash-pick fallback — tenant_handle=0x%08x "
		           "pid=%d → group %u of %u",
		           th, getpid(), g_green_group_id, nb_groups);
	}

	/* Step 4: descriptor over the selected groups (their union). */
	cr = cuDevResourceGenerateDesc(&desc, selected, nsel);
	if (cr != CUDA_SUCCESS) {
		cipher_log("GREEN: cuDevResourceGenerateDesc(nsel=%u) rc=%d",
		           nsel, cr);
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}

	/* Step 5: create the green context. */
	cr = cuGreenCtxCreate(&g_green_ctx, desc, dev,
	                      CU_GREEN_CTX_DEFAULT_STREAM);
	if (cr != CUDA_SUCCESS) {
		cipher_log("GREEN: cuGreenCtxCreate rc=%d", cr);
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}

	/* Step 6: convert to CUcontext for cuCtxPushCurrent. */
	cr = cuCtxFromGreenCtx(&g_green_cuctx, g_green_ctx);
	if (cr != CUDA_SUCCESS) {
		cipher_log("GREEN: cuCtxFromGreenCtx rc=%d", cr);
		(void)cuGreenCtxDestroy(g_green_ctx);
		g_green_ctx = NULL;
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}

	/* Step 7: VERIFY the green ctx has the partition we asked for. */
	cr = cuGreenCtxGetDevResource(g_green_ctx, &verify_res,
	                              CU_DEV_RESOURCE_TYPE_SM);
	if (cr != CUDA_SUCCESS) {
		cipher_log("GREEN: cuGreenCtxGetDevResource rc=%d (verify failed)", cr);
		(void)cuGreenCtxDestroy(g_green_ctx);
		g_green_ctx = NULL;
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}
	g_green_sm_count = verify_res.sm.smCount;
	if (g_green_sm_count == full_sm_res.sm.smCount) {
		cipher_log("GREEN: VERIFY FAILED — green ctx has %u SMs (full set); "
		           "partition did not take", g_green_sm_count);
		(void)cuGreenCtxDestroy(g_green_ctx);
		g_green_ctx = NULL;
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}
	if (g_green_sm_count != nsel * CIPHER_RT_GREEN_MIN_SM)
		cipher_log("GREEN: VERIFY WARNING — expected %u SMs (%u groups), "
		           "got %u (continuing; partition restricted but not exact)",
		           nsel * CIPHER_RT_GREEN_MIN_SM, nsel, g_green_sm_count);

	/* Track 3 SC3: record the mask the green ctx was built from — the
	 * count-preserving baseline cipher_rt_green_ctx_migrate() checks against. */
	g_green_cur_mask = use_fallback ? (1U << g_green_group_id)
	                                : g_cp54_grp_mask;

	atomic_store(&g_green_init_succeeded, 1);
	cipher_log("GREEN: green context ready — %u group(s), %u SMs "
	           "(CUcontext=%p, %s)",
	           nsel, g_green_sm_count, (void *)g_green_cuctx,
	           use_fallback ? "hash-pick fallback" : "CP 5.4 kmod-driven");
	pthread_mutex_unlock(&g_green_lock);
	return 0;
}

unsigned cipher_rt_green_ctx_cur_mask(void)
{
	if (!atomic_load(&g_green_init_succeeded))
		return 0;
	return g_green_cur_mask;
}

/* ---- Track 3 SC3: Dynamic SM Migration ---------------------------------
 *
 * cipher_rt_green_ctx_migrate(new_mask) rebuilds the per-process green
 * context on a different 8-SM-group set of the SAME size and atomically
 * retargets the live g_green_* state the CUPTI launch callback reads.
 *
 * Sequence: drain -> build (into locals) -> L1 structural verify -> swap ->
 * release. Any PRE-swap failure returns <0 with the old context intact and
 * current (the B-floor). Post-swap, a release failure is logged-and-leaked —
 * a committed migration is never unwound ([[cipher-incident-2-bar0-exit]]).
 *
 * The build steps duplicate cipher_rt_green_ctx_ensure() Steps 1-7 rather
 * than refactor them: ensure() is the all-pinned regression path (W1/W2/W3)
 * and is left byte-identical so the smoke test is guaranteed unaffected.
 */
int cipher_rt_green_ctx_migrate(unsigned new_mask)
{
	CUdevice dev = 0;
	CUdevResource full_sm_res;
	CUdevResource groups[CIPHER_RT_GREEN_MAX_GROUPS];
	CUdevResource selected[CIPHER_RT_GREEN_MAX_GROUPS];
	CUdevResource remaining, verify_res;
	CUdevResourceDesc desc;
	CUgreenCtx new_green = NULL, old_green;
	CUcontext  new_cuctx = NULL;
	CUresult cr;
	unsigned int nb_groups = CIPHER_RT_GREEN_MAX_GROUPS;
	unsigned int nsel = 0, g, new_sm_count = 0;
	const char *fault = getenv("CIPHER_SC3_FAULT");   /* SC3 test hook */
	int verify_ok;

	if (!atomic_load(&g_green_init_succeeded)) {
		cipher_log("GREEN/MIGRATE: no green context to migrate");
		return -1;
	}
	if (new_mask == 0) {
		cipher_log("GREEN/MIGRATE: new_mask=0 rejected");
		return -1;
	}
	/* Count-preserving precondition (SC1 fact 1; kmod enforces it at
	 * PROPOSE — checked both ends). */
	if (__builtin_popcount(new_mask) != __builtin_popcount(g_green_cur_mask)) {
		cipher_log("GREEN/MIGRATE: count mismatch — new_mask=0x%04x (%d grp) "
		           "vs current 0x%04x (%d grp); migration is count-preserving",
		           new_mask, __builtin_popcount(new_mask),
		           g_green_cur_mask, __builtin_popcount(g_green_cur_mask));
		return -1;
	}

	pthread_mutex_lock(&g_green_lock);

	/* Step A — drain: no kernel of the old green context is in flight. */
	(void)cuCtxSetCurrent(g_green_cuctx);
	cr = cuCtxSynchronize();
	if (cr != CUDA_SUCCESS) {
		cipher_log("GREEN/MIGRATE: drain cuCtxSynchronize rc=%d — abort, "
		           "old context intact", cr);
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}

	/* Step B — build the new green context from new_mask, into LOCALS. */
	cr = cuDeviceGetDevResource(dev, &full_sm_res, CU_DEV_RESOURCE_TYPE_SM);
	if (cr != CUDA_SUCCESS) {
		cipher_log("GREEN/MIGRATE: cuDeviceGetDevResource rc=%d — abort", cr);
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}
	cr = cuDevSmResourceSplitByCount(groups, &nb_groups, &full_sm_res,
	                                  &remaining, 0, CIPHER_RT_GREEN_MIN_SM);
	if (cr != CUDA_SUCCESS || nb_groups == 0) {
		cipher_log("GREEN/MIGRATE: cuDevSmResourceSplitByCount rc=%d nb=%u "
		           "— abort", cr, nb_groups);
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}
	if (new_mask >> nb_groups) {
		cipher_log("GREEN/MIGRATE: new_mask=0x%04x has bit(s) >= split count "
		           "%u — abort", new_mask, nb_groups);
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}
	for (g = 0; g < nb_groups; g++)
		if (new_mask & (1U << g))
			selected[nsel++] = groups[g];
	cr = cuDevResourceGenerateDesc(&desc, selected, nsel);
	if (cr != CUDA_SUCCESS) {
		cipher_log("GREEN/MIGRATE: cuDevResourceGenerateDesc(nsel=%u) rc=%d "
		           "— abort", nsel, cr);
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}
	cr = cuGreenCtxCreate(&new_green, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM);
	if (cr != CUDA_SUCCESS) {
		cipher_log("GREEN/MIGRATE: cuGreenCtxCreate rc=%d — abort", cr);
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}
	cr = cuCtxFromGreenCtx(&new_cuctx, new_green);
	if (cr != CUDA_SUCCESS) {
		cipher_log("GREEN/MIGRATE: cuCtxFromGreenCtx rc=%d — abort", cr);
		(void)cuGreenCtxDestroy(new_green);
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}

	/* Step C — L1 structural verify, BEFORE the swap. Count-preserving:
	 * the new ctx must have exactly the old ctx's SM count, and not the
	 * full device set. (L2 — the per-round %smid probe — runs tenant-side.) */
	cr = cuGreenCtxGetDevResource(new_green, &verify_res,
	                              CU_DEV_RESOURCE_TYPE_SM);
	new_sm_count = (cr == CUDA_SUCCESS) ? verify_res.sm.smCount : 0;
	verify_ok = (cr == CUDA_SUCCESS) &&
	            (new_sm_count == g_green_sm_count) &&
	            (new_sm_count != full_sm_res.sm.smCount);
	if (fault && !strcmp(fault, "verify")) {
		cipher_log("GREEN/MIGRATE: CIPHER_SC3_FAULT=verify — forcing L1 "
		           "verify failure (SC3 test hook)");
		verify_ok = 0;
	}
	if (!verify_ok) {
		cipher_log("GREEN/MIGRATE: L1 verify FAILED (rc=%d sm=%u expect=%u) "
		           "— abort, old context intact (NACK)", cr, new_sm_count,
		           g_green_sm_count);
		(void)cuGreenCtxDestroy(new_green);
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}

	/* Step D — swap: atomically retarget the live g_green_* state. The
	 * tenant is at a safe point (no launch in flight), so the CUPTI launch
	 * callback (which reads g_green_cuctx via make_current) does not run
	 * concurrently with this swap. */
	old_green = g_green_ctx;
	g_green_ctx      = new_green;
	g_green_cuctx    = new_cuctx;
	g_green_sm_count = new_sm_count;
	g_green_group_id = (unsigned)__builtin_ctz(new_mask);
	g_green_cur_mask = new_mask;
	(void)cuCtxSetCurrent(new_cuctx);

	/* Step E — release the old green context. POST-COMMIT: a destroy
	 * failure is logged and the handle leaked; a committed migration is
	 * never unwound. */
	cr = cuGreenCtxDestroy(old_green);
	if (fault && !strcmp(fault, "destroy")) {
		cipher_log("GREEN/MIGRATE: CIPHER_SC3_FAULT=destroy — simulating "
		           "old-context destroy failure (SC3 test hook)");
		cr = CUDA_ERROR_UNKNOWN;
	}
	if (cr != CUDA_SUCCESS)
		cipher_log("GREEN/MIGRATE: cuGreenCtxDestroy(old) rc=%d — migration "
		           "is COMMITTED; leaking the old handle (bounded)", cr);

	cipher_log("GREEN/MIGRATE: committed — now on mask 0x%04x "
	           "(%u SMs, lowest group %u)",
	           new_mask, new_sm_count, g_green_group_id);
	pthread_mutex_unlock(&g_green_lock);
	return 0;
}

int cipher_rt_green_ctx_push(void)
{
	CUresult cr;
	unsigned long n;
	if (!atomic_load(&g_green_init_succeeded)) {
		return -1;
	}
	n = atomic_fetch_add(&g_streams_observed, 1);
	cr = cuCtxPushCurrent(g_green_cuctx);
	if (cr != CUDA_SUCCESS) {
		cipher_dbg("GREEN: cuCtxPushCurrent rc=%d", cr);
		return -1;
	}
	if (n < 3)
		cipher_log("GREEN: push fired (observation #%lu, group %u)",
		           n + 1, g_green_group_id);
	return 0;
}

void cipher_rt_green_ctx_pop(void)
{
	CUcontext popped;
	if (!atomic_load(&g_green_init_succeeded)) return;
	(void)cuCtxPopCurrent(&popped);
}
