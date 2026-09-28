/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_fp8_actuator.c -- D.9 FP8 actuator on the Phase 4.5 matmul substrate.
 *
 * SECOND actuator alongside Marlin (priority 20). Gate is the COMPLEMENT of
 * Marlin's M<=64: FP8 engages the LARGE-M GEMMs (call->n > 64 = large batch /
 * prefill / training-forward) that Marlin declines. ALL linears engaged
 * (q,k,v,o,gate,up,down,lm_head) — NO layer dropped (Anil 2026-05-30: no
 * scope-down). Per-tensor scalar fused FP8 via the engine. Marlin + cuBLAS shim
 * are UNTOUCHED (Mem #13 additive). Default-OFF (CIPHER_FP8). Worst case =
 * cuBLAS passthrough (Stage 0 sacred).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

#include "cipher_v2_internal.h"
#include "cipher_rt_matmul_dispatch.h"
#include "cipher_rt_fp8.h"

#define CUDA_R_16F          2
#define CUDA_R_16BF         14
#define FP8_MAX_M_GATE      64    /* complement of Marlin: FP8 engages call->n > 64 */
#define FP8_STABILITY       2     /* observe N times before one-time weight prequant */
#define FP8_MIN_DIM         64
#define FP8_SHAPE_LOG_MAX   16

static atomic_int   g_enabled       = 0;
static atomic_int   g_verbose       = 0;
static atomic_ulong g_calls_total   = 0;   /* eligible by env+dtype+gate */
static atomic_ulong g_calls_handled = 0;   /* dispatched to FP8 */
static atomic_ulong g_calls_skipped = 0;
static atomic_uint  g_max_n         = 0;   /* Guard 3: largest engaged batch (cublas n) */

/* Guard 3: distinct engaged (m,k) shapes logged once (direct shape evidence). */
static atomic_int   g_shape_log_n   = 0;
static struct { int m, k; } g_shape_log[FP8_SHAPE_LOG_MAX];

static void fp8_log_shape(int m, int k, int n, int dtype)
{
	int i, cnt = atomic_load(&g_shape_log_n);
	for (i = 0; i < cnt && i < FP8_SHAPE_LOG_MAX; i++)
		if (g_shape_log[i].m == m && g_shape_log[i].k == k) return;  /* seen */
	int slot = atomic_fetch_add(&g_shape_log_n, 1);
	if (slot < FP8_SHAPE_LOG_MAX) {
		g_shape_log[slot].m = m; g_shape_log[slot].k = k;
		cipher_log("FP8: ENGAGED shape #%d out(m)=%d in(k)=%d batch(n)=%d dtype=%s",
		           slot, m, k, n, (dtype == CUDA_R_16BF) ? "bf16" : "fp16");
	}
}

static int maybe_handle_fp8(const struct cipher_rt_matmul_call *call, int *out_status)
{
	int m, n, k, dtype, hits;

	if (!atomic_load(&g_enabled))
		return CIPHER_RT_MATMUL_PASSTHROUGH;

	/* dtype: weight/act/out must match and be fp16 or bf16 (the app's dtype). */
	dtype = call->Atype;
	if (call->Btype != dtype || call->Ctype != dtype ||
	    (dtype != CUDA_R_16F && dtype != CUDA_R_16BF)) {
		atomic_fetch_add(&g_calls_skipped, 1);
		return CIPHER_RT_MATMUL_PASSTHROUGH;
	}

	/* cuBLAS->actuator axis map (same as Marlin): batch = call->n, out = call->m,
	 * in = call->k. FP8 complement gate: engage LARGE batch (n > 64); leave
	 * decode (n <= 64) to Marlin / cuBLAS. */
	n = call->n;   /* batch (Marlin's M) */
	m = call->m;   /* out_features */
	k = call->k;   /* in_features  */
	if (n <= FP8_MAX_M_GATE) {
		atomic_fetch_add(&g_calls_skipped, 1);
		return CIPHER_RT_MATMUL_PASSTHROUGH;
	}
	/* FP8 (e4m3 TN) shape constraints: K multiple of 16, dims not tiny. */
	if ((k & 15) != 0 || m < FP8_MIN_DIM || k < FP8_MIN_DIM) {
		atomic_fetch_add(&g_calls_skipped, 1);
		return CIPHER_RT_MATMUL_PASSTHROUGH;
	}

	atomic_fetch_add(&g_calls_total, 1);

	/* one-time per-tensor weight prequant on the Nth stable observation. */
	hits = cipher_rt_fp8_engine_observe_weight(call->A, k, m);
	if (!cipher_rt_fp8_engine_is_ready(call->A)) {
		if (hits < FP8_STABILITY) {
			atomic_fetch_add(&g_calls_skipped, 1);
			return CIPHER_RT_MATMUL_PASSTHROUGH;
		}
		if (cipher_rt_fp8_engine_quantize_weight(call->A, dtype, k, m, call->stream) != 0) {
			if (atomic_load(&g_verbose))
				cipher_log("FP8: weight prequant failed W=%p (k=%d m=%d)", call->A, k, m);
			atomic_fetch_add(&g_calls_skipped, 1);
			return CIPHER_RT_MATMUL_PASSTHROUGH;
		}
		if (atomic_load(&g_verbose))
			cipher_log("FP8: prequantized W=%p (in=%d out=%d) — %lu weights",
			           call->A, k, m, cipher_rt_fp8_engine_weights_count());
	}

	/* the substitution: inline per-tensor act quant + scalar FP8 GEMM -> fp16/bf16. */
	if (cipher_rt_fp8_engine_matmul(call->A, call->B, call->C, dtype,
	                                m, n, k, call->stream) != 0) {
		if (atomic_load(&g_verbose))
			cipher_log("FP8: matmul declined W=%p (m=%d n=%d k=%d) — passthrough", call->A, m, n, k);
		atomic_fetch_add(&g_calls_skipped, 1);
		return CIPHER_RT_MATMUL_PASSTHROUGH;
	}

	/* Guard 3 telemetry: max_n + direct per-shape engaged log. */
	{
		unsigned int un = (unsigned int)n, prev = atomic_load(&g_max_n);
		while (un > prev && !atomic_compare_exchange_weak(&g_max_n, &prev, un)) {}
		fp8_log_shape(m, k, n, dtype);
	}
	atomic_fetch_add(&g_calls_handled, 1);
	*out_status = 0;  /* CUBLAS_STATUS_SUCCESS */
	return CIPHER_RT_MATMUL_HANDLED;
}

static const struct cipher_rt_matmul_actuator g_fp8_actuator = {
	.name         = "FP8_E4M3",
	.priority     = 20,   /* after Marlin (10): Marlin gets M<=64, FP8 gets the large-M complement */
	.maybe_handle = maybe_handle_fp8,
};

int cipher_rt_fp8_init(void)
{
	const char *env  = getenv("CIPHER_FP8");
	const char *verb = getenv("CIPHER_FP8_VERBOSE");
	int on;

	on = env && (strcmp(env, "on") == 0 || strcmp(env, "1") == 0 || strcmp(env, "ON") == 0);
	atomic_store(&g_enabled, on);
	atomic_store(&g_verbose, verb && verb[0] && verb[0] != '0');

	if (!on) {
		cipher_log("FP8: actuator DISABLED (CIPHER_FP8 not set) — OFF == byte-identical");
		return 0;
	}
	if (cipher_rt_fp8_engine_init() != 0) {
		cipher_log("FP8: engine init FAILED — actuator DISABLED");
		atomic_store(&g_enabled, 0);
		return -1;
	}
	if (cipher_rt_matmul_register_actuator(&g_fp8_actuator) != 0) {
		cipher_log("FP8: substrate registration FAILED — actuator DISABLED");
		atomic_store(&g_enabled, 0);
		return -1;
	}
	cipher_log("FP8: actuator ENABLED (per-tensor scalar E4M3, gate: batch>%d, "
	           "ALL layers, fp16/bf16; stability=%d)", FP8_MAX_M_GATE, FP8_STABILITY);
	return 0;
}

int           cipher_rt_fp8_is_active(void)        { return atomic_load(&g_enabled); }
unsigned long cipher_rt_fp8_calls_total(void)      { return atomic_load(&g_calls_total); }
unsigned long cipher_rt_fp8_calls_handled(void)    { return atomic_load(&g_calls_handled); }
unsigned long cipher_rt_fp8_calls_skipped(void)    { return atomic_load(&g_calls_skipped); }
unsigned long cipher_rt_fp8_weights_quantized(void){ return cipher_rt_fp8_engine_weights_count(); }
unsigned int  cipher_rt_fp8_max_n(void)            { return atomic_load(&g_max_n); }
