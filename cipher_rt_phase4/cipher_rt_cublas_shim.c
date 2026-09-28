/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_cublas_shim.c -- T4.5.1 cuBLAS interception via versioned-symbol
 * export. LD_PRELOAD'd alongside CUDA_INJECTION64_PATH so PyTorch's
 * link-time references to cublasGemmEx@libcublas.so.13 resolve to our
 * shim, which routes through the matmul-dispatch substrate.
 *
 * Build dependency: cublas_version.map (linker --version-script) tags
 * our exported symbols with @libcublas.so.13.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <dlfcn.h>
#include <string.h>
#include <stdbool.h>

#include "cipher_v2_internal.h"
#include "cipher_rt_matmul_dispatch.h"
#include "cipher_rt_got_patch.h"
#include "cipher_rt_commit.h"
#include "cipher_stream_resolver.h"
#include "cipher_rt_pool.h"          /* W.4a coalesce-eligibility (decision-only) */
#include "cipher_rt_ring_write.h"
#include "cipher_rt_tc_probe.h"
#include "cipher_workload_detect.h"   /* K.1 workload classifier observe_gemm */
#include "cipher_rt_fairness.h"       /* D.8 FAIRNESS+SHIELD timing-only throttle */

/* W14 Step 2 E gamma fix (2026-05-23) — natural calibration wire-up at the
 * cublas_shim level. Root cause: the may13-era apply_recipe path in
 * cipher_dispatch.cpp (which previously called edmd_live_post_relaunch_hook)
 * is structurally dead in rt_phase4. CUPTI cb at cipher_cupti.c:174-179
 * explicitly routes through cipher_rt_classify_route, NOT cipher_dispatch.
 * Direct-call cipher_edmd_live_collect from the shim feeds real GEMM
 * snapshots into the EDMD live calibration pipeline (the W14 Step 2 B.0
 * ported cipher_edmd_live.cpp).
 *
 * Idempotent: cipher_edmd_live_collect's internal s->registered/failed
 * guards make repeated calls for the same shape a no-op
 * (cipher_edmd_live.cpp:436), so calls after Koopman has registered the
 * shape are skipped — Koopman-substituted calls (where ptr_C holds the
 * approximated output) cannot corrupt calibration.
 *
 * Weak-linked for graceful degradation: if a future build drops the
 * cipher_edmd_live port, the symbol resolves to NULL and the if-guard
 * makes this a zero-overhead no-op. */
extern bool cipher_edmd_live_collect(
    int M_py, int K_dim, int N_dim,
    int weight_dtype, const void *weight_gpu,
    int activation_dtype, const void *activation_gpu,
    int output_dtype, const void *output_gpu) __attribute__((weak));

/* cuBLAS types we use locally (avoid pulling cublas.h). */
typedef int   cublasStatus_t;
typedef void *cublasHandle_t;
typedef int (*cublasGetStream_v2_fn)(cublasHandle_t, void **);

typedef cublasStatus_t (*cublasGemmEx_fn)(
	cublasHandle_t, int, int, int, int, int,
	const void *, const void *, int, int,
	const void *, int, int,
	const void *, void *, int, int, int, int);

/* cuBLASLt cublasLtMatmul signature (16 args; CUDA 12 + CUDA 13 stable per
 * cublasLt.h audit 2026-05-26). F-B.3.6 remediation: modern torch defaults to
 * cublasLtMatmul for fp32/tf32 matmul hot path; CIPHER must intercept this
 * surface to engage Goals 1-4 on default customer workloads. v1 intercept is
 * passthrough+telemetry only; v1.x extends to actuator routing via matmul
 * dispatch (requires extracting M/N/K from opaque cublasLtMatrixLayout_t
 * descriptors). */
typedef void *cublasLtHandle_t;
typedef void *cublasLtMatmulDesc_t;
typedef void *cublasLtMatrixLayout_t;
typedef void *cublasLtMatmulAlgo_t;  /* const pointer at call site */
typedef int (*cublasLtMatmul_fn)(
	cublasLtHandle_t lightHandle,
	cublasLtMatmulDesc_t computeDesc,
	const void *alpha,
	const void *A, cublasLtMatrixLayout_t Adesc,
	const void *B, cublasLtMatrixLayout_t Bdesc,
	const void *beta,
	const void *C, cublasLtMatrixLayout_t Cdesc,
	void *D, cublasLtMatrixLayout_t Ddesc,
	const cublasLtMatmulAlgo_t *algo,
	void *workspace,
	unsigned long workspaceSizeInBytes,
	void *stream);

static cublasGemmEx_fn      g_real_gemmEx      = NULL;
static cublasGetStream_v2_fn g_cublasGetStream = NULL;
static cublasLtMatmul_fn     g_real_ltMatmul   = NULL;
static atomic_int            g_shim_init_done  = 0;
static atomic_ulong          g_shim_calls      = 0;
static atomic_ulong          g_lt_shim_calls   = 0;

/* Resolve the real cublas symbol via versioned dlvsym to bypass our own
 * shim. RTLD_NOLOAD first (lib already in process from PyTorch). */
static void *resolve_real_in(const char *soname, const char *sym)
{
	void *h = dlopen(soname, RTLD_NOW | RTLD_NOLOAD);
	if (!h) h = dlopen(soname, RTLD_NOW);
	if (!h) return NULL;
	void *p = dlvsym(h, sym, soname);
	if (!p) p = dlsym(h, sym);
	return p;
}

static int shim_lazy_init(void)
{
	if (atomic_load(&g_shim_init_done)) return g_real_gemmEx ? 0 : -1;

	g_real_gemmEx = (cublasGemmEx_fn)resolve_real_in("libcublas.so.13", "cublasGemmEx");
	if (!g_real_gemmEx)
		g_real_gemmEx = (cublasGemmEx_fn)resolve_real_in("libcublas.so.12", "cublasGemmEx");
	g_cublasGetStream = (cublasGetStream_v2_fn)resolve_real_in("libcublas.so.13", "cublasGetStream_v2");
	if (!g_cublasGetStream)
		g_cublasGetStream = (cublasGetStream_v2_fn)resolve_real_in("libcublas.so.12", "cublasGetStream_v2");
	/* F-B.3.6: cublasLtMatmul resolver. Lives in libcublasLt.so.<ver>. */
	g_real_ltMatmul = (cublasLtMatmul_fn)resolve_real_in("libcublasLt.so.13", "cublasLtMatmul");
	if (!g_real_ltMatmul)
		g_real_ltMatmul = (cublasLtMatmul_fn)resolve_real_in("libcublasLt.so.12", "cublasLtMatmul");

	if (!g_real_gemmEx) {
		cipher_log("CUBLAS-SHIM: failed to resolve real cublasGemmEx via either "
		           "libcublas.so.13 or .12");
		atomic_store(&g_shim_init_done, 1);
		return -1;
	}
	cipher_log("CUBLAS-SHIM: real cublasGemmEx=%p getStream=%p ltMatmul=%p resolved",
	           (void *)g_real_gemmEx, (void *)g_cublasGetStream,
	           (void *)g_real_ltMatmul);
	atomic_store(&g_shim_init_done, 1);
	return 0;
}

cublasStatus_t cipher_rt_cublasGemmEx_impl(
	cublasHandle_t handle, int transa, int transb,
	int m, int n, int k,
	const void *alpha, const void *A, int Atype, int lda,
	const void *B, int Btype, int ldb,
	const void *beta, void *C, int Ctype, int ldc,
	int computeType, int algo)
{
	void *stream = NULL;
	struct cipher_rt_matmul_call call;
	unsigned long n_calls;

	if (!g_real_gemmEx) {
		if (shim_lazy_init() < 0)
			return 15;  /* CUBLAS_STATUS_NOT_SUPPORTED */
	}

	/* Resolve the cuBLAS stream bound to this handle. Not fatal if it
	 * fails (NULL stream → default; passthrough actuators won't use it). */
	if (g_cublasGetStream)
		(void)g_cublasGetStream(handle, &stream);

	n_calls = atomic_fetch_add(&g_shim_calls, 1) + 1;
	if (n_calls <= 3 || (n_calls & 0xfff) == 0)
		cipher_dbg("CUBLAS-SHIM #%lu M=%d N=%d K=%d Atype=%d Btype=%d Ctype=%d",
		           n_calls, m, n, k, Atype, Btype, Ctype);

	/* K.1 workload classifier GEMM observe. Cheap relative to cuBLAS
	 * microsecond dispatch; relaxed atomic increments only. */
	cipher_workload_observe_gemm(m, n, k, Atype, Btype, Ctype, stream);

	/* W.4a POOL coalesce-eligibility hook — DECISION-ONLY (always passthrough;
	 * transport is the sequenced W.4b CP 5.6 re-port). Records the (K,N,dtype)
	 * the current same-fingerprint group would coalesce at; reads a published
	 * atomic snapshot, no ioctl. Coalesce key = shared-weight B shape (k,n). */
	(void)cipher_rt_pool_observe_gemm(k, n, Atype);

	/* D.8 FAIRNESS + SHIELD — burst-fairness quota + noisy-neighbor p99
	 * protection. Records this tenant's GEMM in the kmod cross-tenant ledger
	 * (NR 32) and, if armed and the band-modulated burst/SHIELD rule fires,
	 * performs a bounded CPU sleep BEFORE the GEMM is submitted below.
	 * Timing-only: kernels/args/order are untouched → per-tenant output is
	 * bit-identical (KL=0, Mem #11). No-op unless CIPHER_FAIRNESS/CIPHER_SHIELD
	 * is set (default-OFF == pre-D.8 behavior). */
	cipher_rt_fairness_record_and_maybe_throttle();

	memset(&call, 0, sizeof(call));
	call.handle     = handle;
	call.transa     = transa;
	call.transb     = transb;
	call.m          = m;
	call.n          = n;
	call.k          = k;
	call.alpha      = alpha;
	call.A          = A;
	call.Atype      = Atype;
	call.lda        = lda;
	call.B          = B;
	call.Btype      = Btype;
	call.ldb        = ldb;
	call.beta       = beta;
	call.C          = C;
	call.Ctype      = Ctype;
	call.ldc        = ldc;
	call.computeType= computeType;
	call.algo       = algo;
	call.stream     = stream;

	/* W10-12 Step 2 — TC saturation probe (producer-only; emits via
	 * RING_WRITE slot CIPHER_RT_RING_EVENT_TC_PROBE = 6). Classifier
	 * consumer wiring is W13-14 work. Pure-function probe is < 20 ns
	 * uncontended; off the hot-path budget. */
	{
		enum cipher_rt_tc_precision prec =
		    cipher_rt_tc_precision_from_cudatype(Atype);
		enum cipher_rt_tc_class tc_class = cipher_rt_tc_probe(
		    (uint32_t)m, (uint32_t)n, (uint32_t)k, prec,
		    CIPHER_RT_TC_LAYOUT_ROW_MAJOR,
		    CIPHER_RT_TC_LAYOUT_ROW_MAJOR,
		    CIPHER_RT_TC_LAYOUT_ROW_MAJOR);
		struct tc_probe_event {
			uint32_t classification;
			uint32_t M, N, K;
			uint32_t precision;
			uint8_t  layout_a, layout_b, layout_c, _pad;
		} ev = {
			(uint32_t)tc_class, (uint32_t)m, (uint32_t)n, (uint32_t)k,
			(uint32_t)prec, 0u, 0u, 0u, 0u,
		};
		cipher_rt_ring_write(0u, CIPHER_RT_RING_EVENT_TC_PROBE,
		                     (uint32_t)tc_class,
		                     /* commit_seq */ 0u, &ev, sizeof(ev));
	}

	cublasStatus_t rc = cipher_rt_matmul_dispatch(&call, g_real_gemmEx);

	/* W14 Step 2 E gamma fix — feed real GEMM snapshots to EDMD live
	 * calibration. Convention mirrors cipher_dispatch.cpp:229: cuBLAS
	 * (m, n, k) → EDMD (M_py=n, K_dim=k, N_dim=m); weight=A, activation=B,
	 * output=C. Idempotent in the collect impl: shape-already-registered
	 * calls are no-op early-exits, so Koopman-substituted calls do not
	 * corrupt calibration. */
	if (cipher_edmd_live_collect) {
		(void)cipher_edmd_live_collect(
			call.n, call.k, call.m,
			call.Atype, call.A,
			call.Btype, call.B,
			call.Ctype, call.C);
	}

	/* W7-9 Step 4 — COMMIT hot-path wire post matmul_dispatch return.
	 * W7-9 Step 5 — replace hardcoded tenant_id=0 with stream-keyed
	 * resolver lookup. Untracked streams (e.g. default stream, or pre-
	 * REGISTER_STREAMS launches) fall back to tenant_id=0 (the single-
	 * tenant compatibility path). */
	uint32_t tid = cipher_v2_current_tenant_id_from_stream((uintptr_t)stream);
	if (tid == CIPHER_STREAM_TENANT_NONE_USER) tid = 0u;
	(void)cipher_rt_commit_observe_and_publish(tid);

	return rc;
}

/* F-B.3.6 cublasLtMatmul intercept (passthrough+telemetry; v1 scope).
 *
 * Modern torch routes fp32/tf32 matmul through cublasLtMatmul (the cuBLAS-Lt
 * API), not cublasGemmEx. Without this intercept, default customer workloads
 * bypass CIPHER's substrate entirely. v1 intercepts and counts; v1.x extends
 * to actuator routing by decoding M/N/K from the opaque cublasLtMatrixLayout_t
 * descriptors (requires cublasLtMatrixLayoutGetAttribute + workspace; defer).
 *
 * COMMIT-publish wire fires at end of call (mirrors cipher_rt_cublasGemmEx_impl
 * tail) so the audit chain and tenant snapshot stay coherent across both
 * cuBLAS entry points.
 */
int cipher_rt_cublasLtMatmul_impl(
	cublasLtHandle_t lightHandle,
	cublasLtMatmulDesc_t computeDesc,
	const void *alpha,
	const void *A, cublasLtMatrixLayout_t Adesc,
	const void *B, cublasLtMatrixLayout_t Bdesc,
	const void *beta,
	const void *C, cublasLtMatrixLayout_t Cdesc,
	void *D, cublasLtMatrixLayout_t Ddesc,
	const cublasLtMatmulAlgo_t *algo,
	void *workspace,
	unsigned long workspaceSizeInBytes,
	void *stream)
{
	unsigned long n_calls;

	if (!g_real_ltMatmul) {
		if (shim_lazy_init() < 0 || !g_real_ltMatmul)
			return 15;  /* CUBLAS_STATUS_NOT_SUPPORTED */
	}

	n_calls = atomic_fetch_add(&g_lt_shim_calls, 1) + 1;
	if (n_calls <= 3 || (n_calls & 0xfff) == 0)
		cipher_dbg("CUBLAS-LT-SHIM #%lu lightHandle=%p stream=%p",
		           n_calls, lightHandle, stream);

	cublasStatus_t rc = g_real_ltMatmul(
		lightHandle, computeDesc, alpha,
		A, Adesc, B, Bdesc, beta,
		C, Cdesc, D, Ddesc,
		algo, workspace, workspaceSizeInBytes, stream);

	/* COMMIT-publish wire mirrors cublasGemmEx tail. Lt stream is the
	 * direct arg (no GetStream_v2 query). Untracked stream falls back to
	 * tenant_id=0 same as cublasGemmEx path. */
	uint32_t tid = cipher_v2_current_tenant_id_from_stream((uintptr_t)stream);
	if (tid == CIPHER_STREAM_TENANT_NONE_USER) tid = 0u;
	(void)cipher_rt_commit_observe_and_publish(tid);

	return rc;
}

/* CP 2.5 — GOT-patch registration. The shim no longer exports a
 * cublasGemmEx symbol (the T4.5 `.symver`/version-script and the
 * un-versioned alias are gone): LD_PRELOAD link-order interposition is
 * replaced by GOT patching driven from InitializeInjection2. The patcher
 * writes &cipher_rt_cublasGemmEx_impl into every loaded module's GOT slot
 * for `cublasGemmEx`. Real-fn resolution stays in shim_lazy_init
 * (dlopen libcublas + dlvsym) — unaffected by, and robust against, the
 * lazy-PLT value of the patched slot. Called from cipher_inject.c before
 * cipher_rt_got_patch_init().
 *
 * F-B.3.6: additionally register cublasLtMatmul GOT target so modern
 * torch's default fp32/tf32 hot path routes through CIPHER. */
void cipher_rt_cublas_shim_register_got(void)
{
	if (cipher_rt_got_register("cublasGemmEx",
	                           (void *)cipher_rt_cublasGemmEx_impl,
	                           NULL) != 0)
		cipher_log("CUBLAS-SHIM: GOT registration failed (cublasGemmEx)");
	if (cipher_rt_got_register("cublasLtMatmul",
	                           (void *)cipher_rt_cublasLtMatmul_impl,
	                           NULL) != 0)
		cipher_log("CUBLAS-SHIM: GOT registration failed (cublasLtMatmul)");
}

unsigned long cipher_rt_cublas_shim_calls(void)
{
	return atomic_load(&g_shim_calls);
}

/* F-B.3.6: cublasLt intercept counter accessor. Parallel to
 * cipher_rt_cublas_shim_calls; tests / cipher-platform verify read both. */
unsigned long cipher_rt_cublaslt_shim_calls(void)
{
	return atomic_load(&g_lt_shim_calls);
}
