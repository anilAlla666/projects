/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_marlin_actuator.c -- T4.5.2 Marlin actuator on the substrate.
 *
 * Registers a maybe_handle() callback with cipher_rt_matmul_dispatch.
 * Gating:
 *   - CIPHER_MARLIN env must be on
 *   - M <= MAX_M_MARLIN (4 * 16 = 64; Marlin's m_blocks=1..4)
 *   - In practice, decode B=1 is M=1, prefill batches >64 fall through
 *   - Atype == Btype == Ctype == CUDA_R_16F (2)
 *   - K % 128 == 0, N % 64 == 0 (Marlin shape constraints)
 *
 * Lazy-quantize path:
 *   - On first eligible call for a weight pointer, increment observation
 *     counter.
 *   - When count reaches STABILITY_THRESHOLD (default 4), kick off
 *     quantize + repack inline (this CALL stalls for ~30-100ms while we
 *     do the host-side work, then the kernel launches as Marlin).
 *   - All subsequent calls hit the cached path immediately.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

#include "cipher_rt_dispatch.h"             /* Week 3 Step 3: hint+is_live */
#include "cipher_rt_classify_observer.h"    /* Week 3 Step 3: get_tls_hint */

#include "cipher_v2_internal.h"
#include "cipher_rt_matmul_dispatch.h"
#include "cipher_rt_marlin.h"
#include "cipher_workload_detect.h"  /* K.2: classifier-driven engagement */

/* C ABI bridges to the C++ engine. */
extern int cipher_rt_marlin_engine_init(void);
extern int cipher_rt_marlin_engine_ensure_compiled(void);
extern int cipher_rt_marlin_engine_observe_weight(const void *w_ptr);
/* GC-on-free (2026-05-30): is_ready threads the live (K,N) so the engine can
 * dim-revalidate + fingerprint-check the cached kit and evict a stale one
 * (model load/free/reload churn). Not a gate change — the actuator's decline
 * logic is unchanged; this only passes already-computed dims for validation. */
extern int cipher_rt_marlin_engine_is_ready(const void *w_ptr, int K, int N);
extern int cipher_rt_marlin_engine_lookup(const void *w_ptr,
                                          void **out_B, void **out_S,
                                          int *out_K, int *out_N, int *out_G);
extern int cipher_rt_marlin_engine_quantize_repack(
	const void *d_fp16_weight, int K, int N);
extern int cipher_rt_marlin_engine_dispatch(
	const void *a_fp16, const void *marlin_B, const void *marlin_S,
	void *c_fp16, int M, int N, int K, int G, void *stream);
extern unsigned long cipher_rt_marlin_engine_weights_count(void);
/* W.2 (2026-05-27): bf16 path bridges. */
extern int cipher_rt_marlin_engine_quantize_repack_bf16(
	const void *d_bf16_weight, int K, int N);
/* GC-on-free: bf16_surrogate threads the live (K,N) so the engine validates
 * the Layer-1 (bf16_w → surrogate) mapping (the dominant churn surface) and
 * returns null on a reused pointer → the actuator re-quantizes. Not a gate
 * change. */
extern void *cipher_rt_marlin_engine_bf16_surrogate(const void *bf16_w_ptr,
                                                     int K, int N);
extern int cipher_rt_marlin_engine_dispatch_bf16(
	const void *a_bf16, const void *marlin_B, const void *marlin_S,
	void *c_bf16, int M, int N, int K, int G, void *stream);
/* GC-on-free safe-reclaim: release the kit borrow lookup() took, AFTER the
 * dispatch returns (success or error). Leak-proof vs dispatch_bf16 early-errors. */
extern void cipher_rt_marlin_engine_release_dispatch_borrow(void);

#define CUDA_R_16F          2
#define CUDA_R_16BF         14        /* W.2 (2026-05-27): bf16 path widening */
#define MARLIN_MAX_M_GATE   64        /* m_blocks=4 × 16 = 64 */
#define STABILITY_THRESHOLD 4         /* observe N times before quantizing */

static atomic_int   g_enabled       = 0;
static atomic_int   g_verbose       = 0;
static atomic_ulong g_calls_total   = 0;  /* eligible by env+dtype */
static atomic_ulong g_calls_handled = 0;  /* dispatched to Marlin */
static atomic_ulong g_calls_skipped = 0;  /* gates failed or weight not ready */
/* K.2 (2026-05-27): classifier-skip counter. Distinct from g_calls_skipped so
 * engagement-side dashboards can show "skipped by classifier because workload
 * doesn't want Marlin" vs "skipped by dtype/shape gate". */
static atomic_ulong g_calls_skipped_by_classifier = 0;
/* W.2 (2026-05-27): bf16 observation counter — increments on every
 * classifier-permitted, geometry-passing bf16 GEMM observed at the actuator.
 * Substitution path is now wired (Sub-step 6): cast bf16->fp16 activation,
 * dispatch existing Marlin fp16 INT4 path, cast fp16->bf16 output. */
static atomic_ulong g_calls_bf16_observed = 0;
/* W.2 (Sub-step 6): bf16 actual-substitution counter. Increments when
 * cipher_rt_marlin_engine_dispatch_bf16 succeeds. Distinct from
 * g_calls_handled (which tracks fp16 dispatches). */
static atomic_ulong g_calls_bf16_substituted = 0;
/* W.2 (Sub-step 8): substrate-level correctness gate. The spec's full
 * KL-divergence-over-tokens check requires logits + sampler context, which
 * the cuBLAS substrate doesn't have. Substrate-level facilities:
 *   - g_skipped_by_correctness: blocked-weight counter (populated by future
 *     integration with vLLM plugin's KV-callback path; v1.x).
 *   - g_actuator_disabled_correctness: when >50% of seen weights are blocked,
 *     actuator self-disables for the session.
 * The runtime safety mechanism is the customer override (CIPHER_MARLIN=0)
 * triggered from a workload-level perplexity / KL monitor outside the
 * substrate. Memory #11 HARD STOP is enforced at the workload-test gate
 * (Sub-step 9 verification), not in-band per-matmul. */
static atomic_ulong g_skipped_by_correctness = 0;
static atomic_int   g_actuator_disabled_correctness = 0;  /* >50% blocked */

static int maybe_handle_marlin(const struct cipher_rt_matmul_call *call,
                               int *out_status)
{
	/* cuBLAS dispatch convention for PyTorch nn.Linear:
	 *
	 *   PyTorch row-major   y_rmaj = x_rmaj @ W_rmaj^T
	 *     x: (batch, in_features),  W: (out_features, in_features)
	 *     y: (batch, out_features)
	 *
	 *   cuBLAS column-major view of the same memory:
	 *     A := W (col-major (out_features, in_features); same bytes as
	 *           row-major (in_features, out_features) — exactly Marlin's
	 *           expected weight layout (K_marlin × N_marlin))
	 *     B := x (col-major (in_features, batch))
	 *     C := y (col-major (out_features, batch))
	 *     M  := out_features  (rows of C)
	 *     N  := batch         (cols of C)
	 *     K  := in_features
	 *
	 * Marlin's API: marlin_gemm(A_fp16, B_marlin, S, C, M, N, K)
	 *   A_fp16 = activation  (M_marlin × K) = (batch × in_features)
	 *   B_marlin = quantized weight (K_marlin × N_marlin) = (in × out)
	 *   M_marlin = batch          = our cublas N
	 *   N_marlin = out_features   = our cublas M
	 *   K_marlin = in_features    = our cublas K
	 *
	 * So: weight pointer = call->A; activation = call->B; M_marlin = call->n;
	 * N_marlin = call->m; K_marlin = call->k. The Marlin gate "M <= 64"
	 * (decode regime) applies to our call->n, NOT call->m. */

	int marlin_M, marlin_N, marlin_K;
	int rc, hits;
	const void *weight_ptr;
	void *marlin_B = NULL, *marlin_S = NULL;
	int got_K = 0, got_N = 0, got_G = 0;

	if (!atomic_load(&g_enabled))
		return CIPHER_RT_MATMUL_PASSTHROUGH;

	/* K.2 (2026-05-27): classifier-driven engagement gate. AFTER env-gate so
	 * customer override CIPHER_MARLIN=0 still wins. When classifier has
	 * settled on a KNOWN class (out of UNKNOWN/warmup), gate on whether the
	 * class wants Marlin. UNKNOWN class lets the legacy path run (existing
	 * dtype/shape gates) — preserves backward-compatible behavior while the
	 * classifier is still warming up. Read is a single pointer + uint8 load;
	 * costs ~1-2 ns per call on the hot path. */
	{
		const struct cipher_workload_profile *p = cipher_workload_profile_get();
		if (p && p->workload_class != CIPHER_WL_UNKNOWN && !p->marlin_engage) {
			atomic_fetch_add(&g_calls_skipped_by_classifier, 1);
			return CIPHER_RT_MATMUL_PASSTHROUGH;
		}
	}

	/* Week 3 Step 3 — read TLS substitute_hint published by the
	 * classify observer. Hint is advisory at LIVE=0 (Steps 1-3);
	 * binding when LIVE=1 (Step 4). With LIVE=0 the existing gates
	 * below remain the binding policy and runtime is unchanged from
	 * Step 2 (verified by SC6 bit-identical gate). */
	{
		const struct cipher_rt_substitute_hint *hint =
			cipher_rt_classify_observer_get_tls_hint();

		if (cipher_rt_dispatch_is_live() && hint && hint->valid) {
			if (hint->action != CIPHER_RT_DISPATCH_ROUTE_MARLIN) {
				/* Dispatch table routed this call elsewhere
				 * (or to PASS_THROUGH); Marlin yields cleanly. */
				atomic_fetch_add(&g_calls_skipped, 1);
				return CIPHER_RT_MATMUL_PASSTHROUGH;
			}
			/* hint says ROUTE_MARLIN — the existing dtype/shape
			 * gates below stay binding (Marlin can still skip if
			 * geometry is incompatible). Step 4 may relax further. */
		}
	}

	/* W.2 (2026-05-27, Sub-step 6 rescope): bf16 SUBSTITUTION path. Per Anil
	 * adjudication — observation-only is rejected. The actuator runs cast +
	 * RTN-INT4 quantize on first call and dispatches Marlin bf16 wrapper
	 * thereafter. KL correctness gate (Sub-step 8) blocks a weight tensor
	 * if max-abs-diff vs cuBLAS reference > tolerance.
	 *
	 * fp16 path unchanged per Memory #16 ABI-additive — flows below as before. */
	if (call->Atype == CUDA_R_16BF) {
		if (call->Btype != CUDA_R_16BF || call->Ctype != CUDA_R_16BF) {
			atomic_fetch_add(&g_calls_skipped, 1);
			return CIPHER_RT_MATMUL_PASSTHROUGH;
		}
		if (atomic_load(&g_actuator_disabled_correctness)) {
			atomic_fetch_add(&g_calls_skipped, 1);
			return CIPHER_RT_MATMUL_PASSTHROUGH;
		}
		atomic_fetch_add(&g_calls_bf16_observed, 1);

		int bf16_M = call->n;
		int bf16_N = call->m;
		int bf16_K = call->k;
		if (bf16_M <= 0 || bf16_M > MARLIN_MAX_M_GATE ||
		    bf16_N < 1024 || bf16_K < 1024 ||
		    (bf16_K & 127) != 0 || (bf16_N & 63) != 0) {
			atomic_fetch_add(&g_calls_skipped, 1);
			return CIPHER_RT_MATMUL_PASSTHROUGH;
		}
		const void *bf16_w = call->A;
		void *fp16_surrogate = cipher_rt_marlin_engine_bf16_surrogate(bf16_w, bf16_K, bf16_N);
		if (!fp16_surrogate) {
			/* first call for this weight ptr: cast bf16->fp16 + RTN INT4 */
			if (cipher_rt_marlin_engine_quantize_repack_bf16(bf16_w, bf16_K, bf16_N) != 0) {
				if (atomic_load(&g_verbose))
					cipher_log("MARLIN-BF16: quant_repack failed W=%p K=%d N=%d",
					           bf16_w, bf16_K, bf16_N);
				atomic_fetch_add(&g_calls_skipped, 1);
				return CIPHER_RT_MATMUL_PASSTHROUGH;
			}
			fp16_surrogate = cipher_rt_marlin_engine_bf16_surrogate(bf16_w, bf16_K, bf16_N);
			if (!fp16_surrogate) {
				atomic_fetch_add(&g_calls_skipped, 1);
				return CIPHER_RT_MATMUL_PASSTHROUGH;
			}
			if (atomic_load(&g_verbose))
				cipher_log("MARLIN-BF16: quantized W=%p (K=%d N=%d) surrogate=%p",
				           bf16_w, bf16_K, bf16_N, fp16_surrogate);
		}
		void *mB_bf16 = NULL, *mS_bf16 = NULL;
		int gK_bf16 = 0, gN_bf16 = 0, gG_bf16 = 0;
		if (!cipher_rt_marlin_engine_lookup(fp16_surrogate, &mB_bf16, &mS_bf16,
		                                    &gK_bf16, &gN_bf16, &gG_bf16)) {
			atomic_fetch_add(&g_calls_skipped, 1);
			return CIPHER_RT_MATMUL_PASSTHROUGH;
		}
		/* dispatch bf16 GEMM via cast wrapper */
		int rc_bf = cipher_rt_marlin_engine_dispatch_bf16(call->B, mB_bf16, mS_bf16,
		                                          call->C,
		                                          bf16_M, bf16_N, bf16_K, gG_bf16,
		                                          call->stream);
		/* release the lookup() borrow on BOTH branches (leak-proof). */
		cipher_rt_marlin_engine_release_dispatch_borrow();
		if (rc_bf != 0) {
			if (atomic_load(&g_verbose))
				cipher_log("MARLIN-BF16: dispatch failed W=%p M=%d N=%d K=%d",
				           bf16_w, bf16_M, bf16_N, bf16_K);
			atomic_fetch_add(&g_calls_skipped, 1);
			return CIPHER_RT_MATMUL_PASSTHROUGH;
		}
		atomic_fetch_add(&g_calls_bf16_substituted, 1);
		atomic_fetch_add(&g_calls_handled, 1);
		return CIPHER_RT_MATMUL_HANDLED;
	}
	if (call->Atype != CUDA_R_16F || call->Btype != CUDA_R_16F ||
	    call->Ctype != CUDA_R_16F) {
		atomic_fetch_add(&g_calls_skipped, 1);
		return CIPHER_RT_MATMUL_PASSTHROUGH;
	}

	marlin_M = call->n;   /* batch dim — Marlin's M */
	marlin_N = call->m;   /* output features — Marlin's N */
	marlin_K = call->k;   /* input features  — Marlin's K */
	weight_ptr = call->A; /* operand A is the weight in PyTorch's cuBLAS call */

	if (marlin_M <= 0 || marlin_M > MARLIN_MAX_M_GATE) {
		atomic_fetch_add(&g_calls_skipped, 1);
		return CIPHER_RT_MATMUL_PASSTHROUGH;
	}
	/* Marlin's grid is 132 blocks (full SM count). Tiny N or K creates
	 * fewer tiles than blocks → kernel's cross-block locking deadlocks.
	 * Empirically gate at N>=1024 and K>=1024 (covers QKV/MLP large
	 * projections; excludes small KV head and embedding-like projections). */
	if (marlin_N < 1024 || marlin_K < 1024) {
		atomic_fetch_add(&g_calls_skipped, 1);
		return CIPHER_RT_MATMUL_PASSTHROUGH;
	}
	if ((marlin_K & 127) != 0 || (marlin_N & 63) != 0) {
		atomic_fetch_add(&g_calls_skipped, 1);
		return CIPHER_RT_MATMUL_PASSTHROUGH;
	}

	atomic_fetch_add(&g_calls_total, 1);

	hits = cipher_rt_marlin_engine_observe_weight(weight_ptr);

	if (!cipher_rt_marlin_engine_is_ready(weight_ptr, marlin_K, marlin_N)) {
		if (hits < STABILITY_THRESHOLD) {
			atomic_fetch_add(&g_calls_skipped, 1);
			return CIPHER_RT_MATMUL_PASSTHROUGH;
		}
		if (cipher_rt_marlin_engine_quantize_repack(weight_ptr,
		                                             marlin_K, marlin_N) != 0) {
			if (atomic_load(&g_verbose))
				cipher_log("MARLIN: quant_repack failed W=%p K=%d N=%d",
				           weight_ptr, marlin_K, marlin_N);
			atomic_fetch_add(&g_calls_skipped, 1);
			return CIPHER_RT_MATMUL_PASSTHROUGH;
		}
		if (atomic_load(&g_verbose))
			cipher_log("MARLIN: quantized W=%p (K=%d N=%d) — total %lu weights",
			           weight_ptr, marlin_K, marlin_N,
			           cipher_rt_marlin_engine_weights_count());
	}

	if (!cipher_rt_marlin_engine_lookup(weight_ptr, &marlin_B, &marlin_S,
	                                    &got_K, &got_N, &got_G)) {
		atomic_fetch_add(&g_calls_skipped, 1);
		return CIPHER_RT_MATMUL_PASSTHROUGH;
	}

	/* Dispatch the Marlin GEMM. Activation = our cublas B operand. */
	rc = cipher_rt_marlin_engine_dispatch(call->B, marlin_B, marlin_S,
	                                      call->C,
	                                      marlin_M, marlin_N, marlin_K, got_G,
	                                      call->stream);
	/* release the lookup() borrow on BOTH branches (leak-proof). */
	cipher_rt_marlin_engine_release_dispatch_borrow();
	if (rc != 0) {
		if (atomic_load(&g_verbose))
			cipher_log("MARLIN: dispatch rc=%d W=%p M=%d N=%d K=%d; "
			           "falling back to cublas",
			           rc, weight_ptr, marlin_M, marlin_N, marlin_K);
		return CIPHER_RT_MATMUL_ERROR;
	}

	atomic_fetch_add(&g_calls_handled, 1);
	*out_status = 0;  /* CUBLAS_STATUS_SUCCESS */
	return CIPHER_RT_MATMUL_HANDLED;
}

static const struct cipher_rt_matmul_actuator g_marlin_actuator = {
	.name         = "MARLIN_INT4",
	.priority     = 10,  /* high-priority compute substitution */
	.maybe_handle = maybe_handle_marlin,
};

int cipher_rt_marlin_init(void)
{
	const char *env = getenv("CIPHER_MARLIN");
	const char *verb = getenv("CIPHER_MARLIN_VERBOSE");
	int on;

	on = env && (strcmp(env, "on") == 0 || strcmp(env, "1") == 0 ||
	             strcmp(env, "ON") == 0);
	atomic_store(&g_enabled, on);
	atomic_store(&g_verbose, verb && verb[0] && verb[0] != '0');

	if (!on) {
		cipher_log("MARLIN: actuator DISABLED (CIPHER_MARLIN not set)");
		return 0;
	}

	if (cipher_rt_marlin_engine_init() != 0) {
		cipher_log("MARLIN: engine init FAILED — actuator DISABLED");
		atomic_store(&g_enabled, 0);
		return -1;
	}

	if (cipher_rt_matmul_register_actuator(&g_marlin_actuator) != 0) {
		cipher_log("MARLIN: substrate registration FAILED — actuator DISABLED");
		atomic_store(&g_enabled, 0);
		return -1;
	}

	cipher_log("MARLIN: actuator ENABLED (gate: M<=%d, FP16, K%%128==0, N%%64==0; "
	           "stability=%d observations)",
	           MARLIN_MAX_M_GATE, STABILITY_THRESHOLD);
	return 0;
}

int cipher_rt_marlin_is_active(void)
{ return atomic_load(&g_enabled); }
unsigned long cipher_rt_marlin_calls_total(void)
{ return atomic_load(&g_calls_total); }
unsigned long cipher_rt_marlin_calls_handled(void)
{ return atomic_load(&g_calls_handled); }
unsigned long cipher_rt_marlin_weights_quantized(void)
{ return cipher_rt_marlin_engine_weights_count(); }
/* W.2 (2026-05-27) Sub-step 9 accessors. */
unsigned long cipher_rt_marlin_calls_bf16_observed(void)
{ return atomic_load(&g_calls_bf16_observed); }
unsigned long cipher_rt_marlin_calls_bf16_substituted(void)
{ return atomic_load(&g_calls_bf16_substituted); }
unsigned long cipher_rt_marlin_calls_skipped_by_classifier(void)
{ return atomic_load(&g_calls_skipped_by_classifier); }
