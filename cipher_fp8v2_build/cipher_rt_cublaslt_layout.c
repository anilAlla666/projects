/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_cublaslt_layout.c -- D.10 LT-ROUTE: decode opaque cublasLt
 * descriptors into the gemmEx-shaped struct cipher_rt_matmul_call so the
 * typed-variant full shims can route through cipher_rt_matmul_dispatch.
 *
 * cublasLtMatmul computes  D = alpha * op(A) @ op(B) + beta * C,  with
 * op = transa/transb (from computeDesc) and A/B/C/D shapes from layout
 * descriptors. Mapping to the gemmEx (m,n,k) convention (C is m x n,
 * col-major):
 *     m = Cdesc.ROWS ,  n = Cdesc.COLS
 *     k = (transa==OP_T) ? Adesc.ROWS : Adesc.COLS
 *     Atype/lda from Adesc ; Btype/ldb from Bdesc ; Ctype/ldc from Ddesc
 *     transa/transb/computeType from computeDesc
 * Verified against CUBLASLT_LOG_LEVEL=4 ground truth (HSH: transa=OP_T,
 * Adesc=[4096,4096], Cdesc=[4096,56] -> m=4096 n=56 k=4096).
 *
 * Mem #11: a misread here corrupts silently and is worse than blind, so the
 * decoder is conservative -- ANY GetAttribute failure or inconsistent dim
 * makes decode return -1 (the shim then passes through to the real variant,
 * i.e. vanilla, never routing on a bad decode). Public cublasLt introspection
 * API only (Mem #24); no torch/cuBLAS source patch.
 */
#define _GNU_SOURCE
#include "cipher_rt_matmul_dispatch.h"
#include "cipher_v2_internal.h"
#include <dlfcn.h>
#include <stdint.h>
#include <string.h>

/* enum values from cublasLt.h (CUDA 13; stable across 12) */
#define LT_LAYOUT_TYPE   0
#define LT_LAYOUT_ROWS   2
#define LT_LAYOUT_COLS   3
#define LT_LAYOUT_LD     4
#define LT_DESC_COMPUTE_TYPE 0
#define LT_DESC_TRANSA   3
#define LT_DESC_TRANSB   4
#define LT_OP_N          0
#define LT_OP_T          1

typedef int (*lt_desc_get_fn)(void *desc, int attr, void *buf,
                              size_t sizeInBytes, size_t *sizeWritten);
typedef int (*lt_layout_get_fn)(void *layout, int attr, void *buf,
                                size_t sizeInBytes, size_t *sizeWritten);

static lt_desc_get_fn   g_desc_get;
static lt_layout_get_fn g_layout_get;
static int              g_resolved;   /* 0 unknown, 1 ok, -1 failed */

static int layout_get_ld(void *layout, int64_t *out);  /* fwd decl */

static void *lt_resolve(const char *sym)
{
	void *h = dlopen("libcublasLt.so.13", RTLD_NOW | RTLD_NOLOAD);
	if (!h) h = dlopen("libcublasLt.so.13", RTLD_NOW);
	if (!h) h = dlopen("libcublasLt.so.12", RTLD_NOW | RTLD_NOLOAD);
	if (!h) h = dlopen("libcublasLt.so.12", RTLD_NOW);
	if (!h) return NULL;
	return dlsym(h, sym);
}

int cipher_rt_cublaslt_layout_init(void)
{
	if (g_resolved) return g_resolved == 1 ? 0 : -1;
	g_desc_get   = (lt_desc_get_fn)lt_resolve("cublasLtMatmulDescGetAttribute");
	g_layout_get = (lt_layout_get_fn)lt_resolve("cublasLtMatrixLayoutGetAttribute");
	g_resolved = (g_desc_get && g_layout_get) ? 1 : -1;
	cipher_log("LT-LAYOUT: descGet=%p layoutGet=%p (%s)",
	           (void *)g_desc_get, (void *)g_layout_get,
	           g_resolved == 1 ? "resolved" : "FAILED");
	return g_resolved == 1 ? 0 : -1;
}

static int layout_u64(void *layout, int attr, uint64_t *out)
{
	size_t w = 0;
	if (!g_layout_get) return -1;
	if (g_layout_get(layout, attr, out, sizeof(*out), &w) != 0 || w != sizeof(*out))
		return -1;
	return 0;
}

static int layout_i32(void *layout, int attr, int32_t *out)
{
	size_t w = 0;
	if (!g_layout_get) return -1;
	if (g_layout_get(layout, attr, out, sizeof(*out), &w) != 0 || w != sizeof(*out))
		return -1;
	return 0;
}

static int desc_i32(void *desc, int attr, int32_t *out)
{
	size_t w = 0;
	if (!g_desc_get) return -1;
	if (g_desc_get(desc, attr, out, sizeof(*out), &w) != 0 || w != sizeof(*out))
		return -1;
	return 0;
}

/* Decode (computeDesc, A,Adesc, B,Bdesc, C/D,Cdesc/Ddesc) -> *call (gemmEx shape).
 * Returns 0 on a fully-consistent decode; -1 otherwise (shim must pass through).
 * The output buffer D (Ddesc) is the substitution target (== C for in-place). */
int cipher_rt_cublaslt_decode(void *computeDesc,
                              const void *A, void *Adesc,
                              const void *B, void *Bdesc,
                              const void *C, void *Cdesc,
                              void *D, void *Ddesc,
                              const void *alpha, const void *beta,
                              void *stream,
                              struct cipher_rt_matmul_call *call)
{
	int32_t transa = LT_OP_N, transb = LT_OP_N, computeType = 0;
	int32_t Atype = 0, Btype = 0, Dtype = 0;
	uint64_t a_rows = 0, a_cols = 0, c_rows = 0, c_cols = 0;
	int64_t lda = 0, ldb = 0, ldd = 0;
	void *out_desc = Ddesc ? Ddesc : Cdesc;
	void *out_buf  = D ? D : (void *)C;

	if (g_resolved != 1) return -1;
	if (!computeDesc || !Adesc || !Bdesc || !out_desc) return -1;

	if (desc_i32(computeDesc, LT_DESC_TRANSA, &transa)) transa = LT_OP_N;
	if (desc_i32(computeDesc, LT_DESC_TRANSB, &transb)) transb = LT_OP_N;
	(void)desc_i32(computeDesc, LT_DESC_COMPUTE_TYPE, &computeType);

	if (layout_u64(Adesc, LT_LAYOUT_ROWS, &a_rows) ||
	    layout_u64(Adesc, LT_LAYOUT_COLS, &a_cols) ||
	    layout_u64(out_desc, LT_LAYOUT_ROWS, &c_rows) ||
	    layout_u64(out_desc, LT_LAYOUT_COLS, &c_cols))
		return -1;

	{
		int64_t tmp;
		if (layout_get_ld(Adesc, &tmp))    return -1;
		lda = tmp;
		if (layout_get_ld(Bdesc, &tmp))    return -1;
		ldb = tmp;
		if (layout_get_ld(out_desc, &tmp)) return -1;
		ldd = tmp;
	}
	if (layout_i32(Adesc, LT_LAYOUT_TYPE, &Atype) ||
	    layout_i32(Bdesc, LT_LAYOUT_TYPE, &Btype) ||
	    layout_i32(out_desc, LT_LAYOUT_TYPE, &Dtype))
		return -1;

	memset(call, 0, sizeof(*call));
	call->m = (int)c_rows;
	call->n = (int)c_cols;
	call->k = (transa == LT_OP_T) ? (int)a_rows : (int)a_cols;
	call->transa = transa;
	call->transb = transb;
	call->A = A; call->Atype = Atype; call->lda = (int)lda;
	call->B = B; call->Btype = Btype; call->ldb = (int)ldb;
	call->C = out_buf; call->Ctype = Dtype; call->ldc = (int)ldd;
	call->alpha = alpha; call->beta = beta;
	call->computeType = computeType;
	call->algo = -1;
	call->stream = (cipher_rt_cuda_stream_t)stream;

	/* consistency guards (Mem #11): positive dims; the non-contracted A dim
	 * must equal m (op(A) is m x k). */
	if (call->m <= 0 || call->n <= 0 || call->k <= 0) return -1;
	{
		uint64_t a_m = (transa == LT_OP_T) ? a_cols : a_rows;
		if ((int)a_m != call->m) return -1;
	}
	return 0;
}

static int layout_get_ld(void *layout, int64_t *out)
{
	size_t w = 0;
	if (!g_layout_get) return -1;
	if (g_layout_get(layout, LT_LAYOUT_LD, out, sizeof(*out), &w) != 0 ||
	    w != sizeof(*out))
		return -1;
	return 0;
}
