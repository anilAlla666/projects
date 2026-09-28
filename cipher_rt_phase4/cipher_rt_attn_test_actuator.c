/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_attn_test_actuator.c - Smoke-test actuator for T4.6.1.
 *
 * Registers at libcipher_rt init (gated by CIPHER_ATTN_TEST=on). Logs
 * the first few attention calls + per-backend counts. Always returns
 * PASSTHROUGH so output is byte-identical to baseline.
 *
 * The point of this actuator is to verify three-indicator routing:
 *   (a) substrate telemetry shows N attention calls observed
 *   (b) this actuator's logs show those calls with correct shapes
 *   (c) workload output is byte-identical to baseline
 */

#include "cipher_rt_attn_dispatch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

static atomic_ulong g_seen_flash;
static atomic_ulong g_seen_eff;
static atomic_ulong g_seen_cudnn;
static atomic_ulong g_logged;

static const char* backend_name(enum cipher_rt_attn_backend b) {
	switch (b) {
	case CIPHER_RT_ATTN_BACKEND_FLASH:     return "FLASH";
	case CIPHER_RT_ATTN_BACKEND_EFFICIENT: return "EFF";
	case CIPHER_RT_ATTN_BACKEND_CUDNN:     return "CUDNN";
	default:                               return "?";
	}
}

static const char* dtype_name(int dt) {
	switch (dt) {
	case 5:  return "Half";   /* c10::kHalf */
	case 6:  return "Float";
	case 7:  return "Double";
	case 15: return "BFloat16";
	default: { static __thread char buf[16]; snprintf(buf, sizeof buf, "dt%d", dt); return buf; }
	}
}

static int test_maybe_handle(const struct cipher_rt_attn_call *call)
{
	switch (call->backend) {
	case CIPHER_RT_ATTN_BACKEND_FLASH:     atomic_fetch_add(&g_seen_flash, 1); break;
	case CIPHER_RT_ATTN_BACKEND_EFFICIENT: atomic_fetch_add(&g_seen_eff,   1); break;
	case CIPHER_RT_ATTN_BACKEND_CUDNN:     atomic_fetch_add(&g_seen_cudnn, 1); break;
	}
	unsigned long n = atomic_fetch_add(&g_logged, 1);
	if (n < 4) {
		fprintf(stderr,
			"[cipher-attn-test] %s call dropout=%g causal=%d "
			"Q=[%ld,%ld,%ld,%ld] K=[%ld,%ld,%ld,%ld] V=[%ld,%ld,%ld,%ld] "
			"dtype=%s\n",
			backend_name(call->backend),
			call->dropout_p, call->is_causal,
			(long)call->q.sizes[0], (long)call->q.sizes[1],
			(long)call->q.sizes[2], (long)call->q.sizes[3],
			(long)call->k.sizes[0], (long)call->k.sizes[1],
			(long)call->k.sizes[2], (long)call->k.sizes[3],
			(long)call->v.sizes[0], (long)call->v.sizes[1],
			(long)call->v.sizes[2], (long)call->v.sizes[3],
			dtype_name(call->q.dtype));
	}
	return CIPHER_RT_ATTN_PASSTHROUGH;
}

static struct cipher_rt_attn_actuator g_test_actuator = {
	.name = "test",
	.priority = 0,
	.maybe_handle = test_maybe_handle,
};

__attribute__((destructor))
static void cipher_attn_test_summary(void)
{
	if (!getenv("CIPHER_ATTN_TEST")) return;
	fprintf(stderr,
		"[cipher-attn-test] SUMMARY by backend: flash=%lu eff=%lu cudnn=%lu\n",
		atomic_load(&g_seen_flash),
		atomic_load(&g_seen_eff),
		atomic_load(&g_seen_cudnn));
}

/* Called by cipher_inject's libcipher_rt init body. */
int cipher_rt_attn_test_actuator_init(void)
{
	const char* on = getenv("CIPHER_ATTN_TEST");
	if (!on || !(on[0] == '1' || on[0] == 'y' || on[0] == 'Y' ||
	             on[0] == 't' || on[0] == 'T' || on[0] == 'o')) {
		fprintf(stderr, "[cipher-attn-test] disabled (CIPHER_ATTN_TEST unset)\n");
		return 0;
	}
	return cipher_rt_attn_register_actuator(&g_test_actuator);
}
