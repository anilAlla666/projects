/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_matmul_dispatch.c -- Phase 4.5 matmul-routing substrate impl.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <pthread.h>

#include "cipher_v2_internal.h"
#include "cipher_rt_matmul_dispatch.h"

#define CIPHER_RT_MATMUL_MAX_ACTUATORS 16

static struct {
	atomic_int                       inited;
	pthread_mutex_t                  reg_lock;
	int                              n_actuators;
	struct cipher_rt_matmul_actuator actuators[CIPHER_RT_MATMUL_MAX_ACTUATORS];

	/* Telemetry */
	atomic_ulong total;
	atomic_ulong handled;
	atomic_ulong passthrough;
} g_disp = {
	.reg_lock = PTHREAD_MUTEX_INITIALIZER,
};

static void cipher_rt_matmul_atexit_diag(void)
{
	unsigned long total      = atomic_load(&g_disp.total);
	unsigned long handled    = atomic_load(&g_disp.handled);
	unsigned long passthrough= atomic_load(&g_disp.passthrough);
	cipher_log("MATMUL: exit totals — calls=%lu handled=%lu passthrough=%lu (actuators=%d)",
	           total, handled, passthrough, g_disp.n_actuators);
}

int cipher_rt_matmul_dispatch_init(void)
{
	if (atomic_exchange(&g_disp.inited, 1)) return 0;
	atexit(cipher_rt_matmul_atexit_diag);
	cipher_log("MATMUL: substrate initialized "
	           "(max %d actuators; first registration awaited)",
	           CIPHER_RT_MATMUL_MAX_ACTUATORS);
	return 0;
}

/* Insertion-sort registration by priority (lower runs first). Stable for
 * equal priorities (preserves registration order). */
int cipher_rt_matmul_register_actuator(
	const struct cipher_rt_matmul_actuator *actuator)
{
	int i;

	if (!actuator || !actuator->maybe_handle || !actuator->name)
		return -1;

	pthread_mutex_lock(&g_disp.reg_lock);

	if (g_disp.n_actuators >= CIPHER_RT_MATMUL_MAX_ACTUATORS) {
		pthread_mutex_unlock(&g_disp.reg_lock);
		cipher_log("MATMUL: actuator registry full (max %d); '%s' refused",
		           CIPHER_RT_MATMUL_MAX_ACTUATORS, actuator->name);
		return -1;
	}

	/* Find insertion slot by priority. */
	for (i = g_disp.n_actuators; i > 0; i--) {
		if (g_disp.actuators[i - 1].priority <= actuator->priority) break;
		g_disp.actuators[i] = g_disp.actuators[i - 1];
	}
	g_disp.actuators[i] = *actuator;
	g_disp.n_actuators++;

	cipher_log("MATMUL: actuator '%s' registered at priority %d (slot %d/%d)",
	           actuator->name, actuator->priority,
	           i, g_disp.n_actuators);

	pthread_mutex_unlock(&g_disp.reg_lock);
	return 0;
}

int cipher_rt_matmul_dispatch(
	const struct cipher_rt_matmul_call *call,
	cipher_rt_cublasGemmEx_passthrough_t passthrough_fn)
{
	int i, n, result, status = 0;

	atomic_fetch_add(&g_disp.total, 1);

	/* Snapshot count (registry append-only after init). */
	n = g_disp.n_actuators;
	for (i = 0; i < n; i++) {
		result = g_disp.actuators[i].maybe_handle(call, &status);
		if (result == CIPHER_RT_MATMUL_HANDLED) {
			atomic_fetch_add(&g_disp.handled, 1);
			return status;
		}
		if (result == CIPHER_RT_MATMUL_ERROR) {
			cipher_dbg("MATMUL: actuator '%s' ERROR; falling through",
			           g_disp.actuators[i].name);
			break;
		}
		/* PASSTHROUGH → try next actuator */
	}

	atomic_fetch_add(&g_disp.passthrough, 1);
	return passthrough_fn(call->handle, call->transa, call->transb,
	                      call->m, call->n, call->k,
	                      call->alpha, call->A, call->Atype, call->lda,
	                      call->B, call->Btype, call->ldb,
	                      call->beta, call->C, call->Ctype, call->ldc,
	                      call->computeType, call->algo);
}

unsigned long cipher_rt_matmul_calls_total(void)
{ return atomic_load(&g_disp.total); }
unsigned long cipher_rt_matmul_calls_handled(void)
{ return atomic_load(&g_disp.handled); }
unsigned long cipher_rt_matmul_calls_passthrough(void)
{ return atomic_load(&g_disp.passthrough); }
