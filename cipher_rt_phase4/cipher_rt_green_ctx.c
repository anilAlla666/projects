/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_green_ctx.c -- T4.2.4b Green Context with real SM partition.
 *
 * v2 (T4.2.4b, this build): per-process green context restricted to ONE
 * 8-SM group, selected by `tenant_handle % NUM_GROUPS`. On H100
 * (compute 9.0+), Green Context SM partitions must be multiples of 8
 * (per CUDA spec), so 132 SMs split into NUM_GROUPS=16 groups of 8 SMs
 * each (128 SMs total, 4 SMs go into "remaining" — accepted 3% loss
 * for v1 simplicity).
 *
 * Important honest scope (advisor-binding pre-design):
 *
 *   The ARBITRATE mask (cipher_kmod's sm_partition_mask) carries COUNT
 *   and IDENTITY (slot indices owned by this tenant), NOT PHYSICAL
 *   SM PLACEMENT. The kmod slot array does not have a slot→SM mapping
 *   today; slot indices are purely kernel-internal data. Therefore
 *   T4.2.4b cannot drive SM placement from the mask. Instead:
 *     - mask popcount → unused in v1 (every tenant gets 1 group of 8 SMs)
 *     - tenant_handle (from REGISTER_TENANT, deterministic) → selects
 *       which of the NUM_GROUPS groups this tenant binds to
 *   T4.2.4c will add kmod-side slot→SM-index mapping so mask drives
 *   placement.
 *
 * Refresh policy: green ctx is created once at first cuStreamCreate and
 * never refreshed. PyTorch and other CUDA apps create streams at warmup
 * and reuse them; destroying the green ctx mid-run would leave the
 * existing streams bound to a stale ctx. B7 poll-thread mask updates
 * are observed but do not drive green-ctx recreation. v2's cold-path
 * assignment is "good enough" because B10 fair-share converges ranks
 * within 30s of run start.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <pthread.h>

#include <cuda.h>

#include "cipher_v2_internal.h"
#include "cipher_rt_green_ctx.h"

/* T4.2.4b: H100 (compute 9.0+) green-ctx SM partition minimum is 8 SMs,
 * multiples of 8. 132 / 8 = 16 groups (with 4 SMs left over). */
#define CIPHER_RT_GREEN_NUM_GROUPS  16
#define CIPHER_RT_GREEN_MIN_SM      8

static CUgreenCtx       g_green_ctx       = NULL;
static CUcontext        g_green_cuctx     = NULL;
static atomic_int       g_green_init_attempted = 0;
static atomic_int       g_green_init_succeeded = 0;
static atomic_ulong     g_streams_observed = 0;
static pthread_mutex_t  g_green_lock      = PTHREAD_MUTEX_INITIALIZER;
static unsigned         g_green_group_id  = 0;    /* which group this proc bound */
static unsigned         g_green_sm_count  = 0;    /* verified SM count of green ctx */

/* Forward-declared accessor (defined in cipher_rt_tenant.h but we avoid
 * pulling it in here due to header conflicts). The tenant_handle_u32 is
 * a stable identity hash from REGISTER_TENANT (FNV-32 of tenant_id) — we
 * use it to deterministically select which green-ctx group this process
 * binds to. */
extern unsigned int cipher_rt_tenant_handle_u32_for_self(void);

int cipher_rt_green_ctx_is_initialized(void)
{
	return atomic_load(&g_green_init_succeeded);
}

unsigned long cipher_rt_green_ctx_streams_observed(void)
{
	return atomic_load(&g_streams_observed);
}

/* CP 5.3 STEP 2: verified SM count of this process's green context, 0 if
 * none. g_green_sm_count is set once under g_green_lock in _ensure() and
 * never mutated after g_green_init_succeeded flips — the acquire load of
 * that flag orders the read. */
unsigned int cipher_rt_green_ctx_sm_count(void)
{
	if (!atomic_load(&g_green_init_succeeded))
		return 0;
	return g_green_sm_count;
}

/* CP 5.3 STEP 2: the partition group this process bound to. Same ordering
 * argument as _sm_count(); meaningful only when initialized. */
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

/* T4.2.4d enforcement: persistently set the green CUcontext as the
 * calling thread's current. Skips cuCtxSetCurrent if already green
 * (TLS check via cuCtxGetCurrent). */
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
	CUdevResource groups[CIPHER_RT_GREEN_NUM_GROUPS];
	CUdevResource remaining;
	CUdevResourceDesc desc;
	CUdevResource verify_res;
	CUresult cr;
	unsigned int nb_groups = CIPHER_RT_GREEN_NUM_GROUPS;
	unsigned int my_group_id;
	unsigned int tenant_handle;

	if (atomic_load(&g_green_init_succeeded))
		return 0;

	pthread_mutex_lock(&g_green_lock);
	if (atomic_load(&g_green_init_succeeded)) {
		pthread_mutex_unlock(&g_green_lock);
		return 0;
	}
	if (atomic_exchange(&g_green_init_attempted, 1)) {
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}

	/* Step 1: get device 0's full SM resource set. */
	cr = cuDeviceGetDevResource(dev, &full_sm_res, CU_DEV_RESOURCE_TYPE_SM);
	if (cr != CUDA_SUCCESS) {
		cipher_log("GREEN: cuDeviceGetDevResource rc=%d", cr);
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}
	cipher_log("GREEN: device 0 has %u SMs total; splitting into %u groups of %u SMs each",
	           full_sm_res.sm.smCount, nb_groups, CIPHER_RT_GREEN_MIN_SM);

	/* Step 2: split into NUM_GROUPS equal-size groups of MIN_SM SMs each.
	 * The remaining SMs (132 - 16*8 = 4) go into 'remaining'. */
	cr = cuDevSmResourceSplitByCount(groups, &nb_groups, &full_sm_res,
	                                  &remaining, 0,
	                                  CIPHER_RT_GREEN_MIN_SM);
	if (cr != CUDA_SUCCESS) {
		cipher_log("GREEN: cuDevSmResourceSplitByCount rc=%d", cr);
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}
	cipher_log("GREEN: split returned nb_groups=%u (requested %u); "
	           "remaining %u SMs unallocated",
	           nb_groups, CIPHER_RT_GREEN_NUM_GROUPS,
	           remaining.sm.smCount);

	if (nb_groups == 0) {
		cipher_log("GREEN: split returned 0 groups — abort");
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}

	/* Step 3: pick our group. Original v1 used `tenant_handle % nb_groups`
	 * but FNV-64 of close-prefix strings like wl05_tN collapses all 8 to
	 * the same residue (100% collision observed in T4.2.4b first run).
	 * v2 fix: combine tenant_handle with getpid() via xorshift-style mix,
	 * which distributes uniformly even for collided tenant_handles. */
	tenant_handle = cipher_rt_tenant_handle_u32_for_self();
	{
		unsigned int h = (unsigned int)getpid() ^ tenant_handle;
		h ^= h >> 16;
		h *= 0x85ebca6bU;
		h ^= h >> 13;
		h *= 0xc2b2ae35U;
		h ^= h >> 16;
		my_group_id = h % nb_groups;
	}
	g_green_group_id = my_group_id;
	cipher_log("GREEN: tenant_handle=0x%08x pid=%d → selecting group %u "
	           "of %u (each group has %u SMs)",
	           tenant_handle, getpid(), my_group_id, nb_groups,
	           groups[my_group_id].sm.smCount);

	/* Step 4: generate descriptor from our SELECTED group only. */
	cr = cuDevResourceGenerateDesc(&desc, &groups[my_group_id], 1);
	if (cr != CUDA_SUCCESS) {
		cipher_log("GREEN: cuDevResourceGenerateDesc rc=%d", cr);
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}

	/* Step 5: create the green context bound to our group. */
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

	/* Step 7: VERIFY the green ctx actually has the partition we asked
	 * for (advisor-binding "did the partition take" check). If it
	 * returns 132 SMs, the partition silently failed. */
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
		           "partition did not take",
		           g_green_sm_count);
		(void)cuGreenCtxDestroy(g_green_ctx);
		g_green_ctx = NULL;
		pthread_mutex_unlock(&g_green_lock);
		return -1;
	}
	if (g_green_sm_count != CIPHER_RT_GREEN_MIN_SM) {
		cipher_log("GREEN: VERIFY WARNING — expected %u SMs, got %u "
		           "(continuing; partition is restricted but not exact)",
		           CIPHER_RT_GREEN_MIN_SM, g_green_sm_count);
	}

	atomic_store(&g_green_init_succeeded, 1);
	cipher_log("GREEN: green context bound to group %u with %u SMs "
	           "(CUcontext=%p)",
	           my_group_id, g_green_sm_count, (void *)g_green_cuctx);
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
	/* Log first 3 push fires so we can verify the hook is being called
	 * during user stream creation. Subsequent pushes silent. */
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
