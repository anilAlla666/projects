/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_classify_substrate.cpp -- Week 2 Step 3 classify-routing substrate.
 *
 * Mirrors cipher_rt_matmul_dispatch.c shape: static g_disp struct with
 * atomic counters, pthread mutex around registration, priority-sorted
 * actuator array, atexit telemetry dump. The default may13 classifier
 * auto-registers at static-init time and wraps cipher::classify_launch
 * (pure geometry-based; safe at .so load time).
 *
 * cipher_rt_classify_route() is NOT called from any hot path in Step 3.
 * Step 6 wires it through the F1 cuLaunchKernel intercept.
 */
/* _GNU_SOURCE is provided transitively via included headers. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <atomic>
#include <pthread.h>

#include "cipher_v2_internal.h"
#include "cipher_rt_classify_substrate.h"
#include "may13/cipher_classify.hpp"

namespace {

struct ClassifySubstrateState {
	std::atomic<int>                       inited{0};
	pthread_mutex_t                        reg_lock = PTHREAD_MUTEX_INITIALIZER;
	int                                    n_actuators = 0;
	struct cipher_rt_classify_actuator     actuators[CIPHER_RT_CLASSIFY_MAX_ACTUATORS];

	/* Telemetry */
	std::atomic<unsigned long> total{0};
	std::atomic<unsigned long> handled{0};
	std::atomic<unsigned long> passthrough{0};
};

ClassifySubstrateState g_classify;

void cipher_rt_classify_atexit_diag(void)
{
	unsigned long total      = g_classify.total.load();
	unsigned long handled    = g_classify.handled.load();
	unsigned long passthrough= g_classify.passthrough.load();
	cipher_log("CLASSIFY: exit totals — calls=%lu handled=%lu passthrough=%lu (actuators=%d)",
	           total, handled, passthrough, g_classify.n_actuators);
}

/* may13 default classifier (priority-0). Wraps cipher::classify_launch
 * from include/may13/cipher_classify.hpp. Pure function; no CUDA/NVML
 * side effects; safe to register at .so load time. */
int may13_default_classifier(const struct cipher_rt_classify_call *call,
                             struct cipher_rt_classify_out        *out)
{
	if (!call || !out) return CIPHER_RT_CLASSIFY_ERROR;

	cipher::ClassifyResult r = cipher::classify_launch(
		call->fn,
		call->grid_x, call->grid_y, call->grid_z,
		call->block_x, call->block_y, call->block_z,
		call->shared_bytes);

	out->op_class   = static_cast<uint8_t>(r.op);
	out->confidence = r.confidence;
	out->cache_hit  = r.cache_hit ? 1 : 0;
	return CIPHER_RT_CLASSIFY_HANDLED;
}

/* Auto-register the may13 default classifier at static-init time.
 * cipher::classify_launch is pure-function safe; we are only storing
 * a function pointer here (we do NOT call it at this point), and the
 * substrate's mutex is statically initialized so registration is safe
 * even before any explicit init() call. */
struct AutoRegister {
	AutoRegister() {
		struct cipher_rt_classify_actuator a = {};
		a.name     = "may13_default";
		a.priority = 0;
		a.classify = may13_default_classifier;
		(void)cipher_rt_classify_register_actuator(&a);
	}
};

AutoRegister g_auto_register;

} /* anonymous namespace */

extern "C" {

int cipher_rt_classify_dispatch_init(void)
{
	int expected = 0;
	if (!g_classify.inited.compare_exchange_strong(expected, 1))
		return 0;
	atexit(cipher_rt_classify_atexit_diag);
	cipher_log("CLASSIFY: substrate initialized "
	           "(max %d actuators; %d already registered)",
	           CIPHER_RT_CLASSIFY_MAX_ACTUATORS, g_classify.n_actuators);
	return 0;
}

/* Insertion-sort registration by priority (lower runs first). Stable
 * for equal priorities (preserves registration order). */
int cipher_rt_classify_register_actuator(
	const struct cipher_rt_classify_actuator *actuator)
{
	int i;

	if (!actuator || !actuator->classify || !actuator->name)
		return -1;

	pthread_mutex_lock(&g_classify.reg_lock);

	if (g_classify.n_actuators >= CIPHER_RT_CLASSIFY_MAX_ACTUATORS) {
		pthread_mutex_unlock(&g_classify.reg_lock);
		cipher_log("CLASSIFY: actuator registry full (max %d); '%s' refused",
		           CIPHER_RT_CLASSIFY_MAX_ACTUATORS, actuator->name);
		return -1;
	}

	for (i = g_classify.n_actuators; i > 0; i--) {
		if (g_classify.actuators[i - 1].priority <= actuator->priority) break;
		g_classify.actuators[i] = g_classify.actuators[i - 1];
	}
	g_classify.actuators[i] = *actuator;
	g_classify.n_actuators++;

	cipher_log("CLASSIFY: actuator '%s' registered at priority %d (slot %d/%d)",
	           actuator->name, actuator->priority,
	           i, g_classify.n_actuators);

	pthread_mutex_unlock(&g_classify.reg_lock);
	return 0;
}

int cipher_rt_classify_route(
	const struct cipher_rt_classify_call *call,
	struct cipher_rt_classify_out        *out)
{
	int i, n, result;

	if (!call || !out) return CIPHER_RT_CLASSIFY_ERROR;

	g_classify.total.fetch_add(1);

	/* Initialize out to UNCLASSIFIED so callers see a defined state
	 * even if every actuator returns PASSTHROUGH/ERROR. */
	out->op_class   = CIPHER_RT_CLASSIFY_UNCLASSIFIED;
	out->confidence = 0;
	out->cache_hit  = 0;

	/* Snapshot count (registry append-only after init). */
	n = g_classify.n_actuators;
	for (i = 0; i < n; i++) {
		result = g_classify.actuators[i].classify(call, out);
		if (result == CIPHER_RT_CLASSIFY_HANDLED) {
			g_classify.handled.fetch_add(1);
			return CIPHER_RT_CLASSIFY_HANDLED;
		}
		if (result == CIPHER_RT_CLASSIFY_ERROR) {
			cipher_dbg("CLASSIFY: actuator '%s' ERROR; falling through",
			           g_classify.actuators[i].name);
			break;
		}
		/* PASSTHROUGH → try next */
	}

	g_classify.passthrough.fetch_add(1);
	return CIPHER_RT_CLASSIFY_PASSTHROUGH;
}

unsigned long cipher_rt_classify_calls_total(void)
{ return g_classify.total.load(); }
unsigned long cipher_rt_classify_calls_handled(void)
{ return g_classify.handled.load(); }
unsigned long cipher_rt_classify_calls_passthrough(void)
{ return g_classify.passthrough.load(); }

} /* extern "C" */
