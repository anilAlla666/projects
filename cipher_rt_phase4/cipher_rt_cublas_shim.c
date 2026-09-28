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

#include "cipher_v2_internal.h"
#include "cipher_rt_matmul_dispatch.h"
#include "cipher_rt_got_patch.h"

/* cuBLAS types we use locally (avoid pulling cublas.h). */
typedef int   cublasStatus_t;
typedef void *cublasHandle_t;
typedef int (*cublasGetStream_v2_fn)(cublasHandle_t, void **);

typedef cublasStatus_t (*cublasGemmEx_fn)(
	cublasHandle_t, int, int, int, int, int,
	const void *, const void *, int, int,
	const void *, int, int,
	const void *, void *, int, int, int, int);

static cublasGemmEx_fn      g_real_gemmEx      = NULL;
static cublasGetStream_v2_fn g_cublasGetStream = NULL;
static atomic_int            g_shim_init_done  = 0;
static atomic_ulong          g_shim_calls      = 0;

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

	if (!g_real_gemmEx) {
		cipher_log("CUBLAS-SHIM: failed to resolve real cublasGemmEx via either "
		           "libcublas.so.13 or .12");
		atomic_store(&g_shim_init_done, 1);
		return -1;
	}
	cipher_log("CUBLAS-SHIM: real cublasGemmEx=%p getStream=%p resolved",
	           (void *)g_real_gemmEx, (void *)g_cublasGetStream);
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

	return cipher_rt_matmul_dispatch(&call, g_real_gemmEx);
}

/* CP 2.5 — GOT-patch registration. The shim no longer exports a
 * cublasGemmEx symbol (the T4.5 `.symver`/version-script and the
 * un-versioned alias are gone): LD_PRELOAD link-order interposition is
 * replaced by GOT patching driven from InitializeInjection2. The patcher
 * writes &cipher_rt_cublasGemmEx_impl into every loaded module's GOT slot
 * for `cublasGemmEx`. Real-fn resolution stays in shim_lazy_init
 * (dlopen libcublas + dlvsym) — unaffected by, and robust against, the
 * lazy-PLT value of the patched slot. Called from cipher_inject.c before
 * cipher_rt_got_patch_init(). */
void cipher_rt_cublas_shim_register_got(void)
{
	if (cipher_rt_got_register("cublasGemmEx",
	                           (void *)cipher_rt_cublasGemmEx_impl,
	                           NULL) != 0)
		cipher_log("CUBLAS-SHIM: GOT registration failed");
}

unsigned long cipher_rt_cublas_shim_calls(void)
{
	return atomic_load(&g_shim_calls);
}
