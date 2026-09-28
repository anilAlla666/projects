/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_matmul_dispatch.h -- Phase 4.5 matmul-routing substrate.
 *
 * This is the substrate. Marlin is the first actuator on it. Future
 * actuators (FP8, fusion, speculative decode, per-tenant dispatch) plug
 * in via the same registry without touching the cuBLAS shim.
 *
 * Design:
 *
 *   cublasGemmEx shim (cipher_rt_cublas_shim.c)
 *     │
 *     ▼
 *   cipher_rt_matmul_dispatch(call) — for each registered actuator:
 *     │  actuator->maybe_handle(call):
 *     │    return HANDLED (substituted), PASSTHROUGH (skip me),
 *     │    or ERROR (real cublas path).
 *     │
 *     ├── if any actuator returns HANDLED → done
 *     └── else → real cublasGemmEx
 *
 * Ordering: actuators called in registration order. Priority field
 * reserved for future use (multi-actuator coordination). Each actuator
 * defines its own cache key for the weight pointer (B operand) —
 * Marlin keys on B-ptr, FP8 keys on (A-ptr, B-ptr, dtype), spec-decode
 * keys differently.
 */
#ifndef CIPHER_RT_MATMUL_DISPATCH_H
#define CIPHER_RT_MATMUL_DISPATCH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle types passed through; the shim doesn't include cublas.h. */
typedef void *cipher_rt_cublas_handle_t;
typedef void *cipher_rt_cuda_stream_t;

/* Call descriptor. Substrate fills this from cublasGemmEx params; actuators
 * read it. Layout is stable: append-only for new fields. */
struct cipher_rt_matmul_call {
	cipher_rt_cublas_handle_t handle;
	int           transa;
	int           transb;
	int           m;
	int           n;
	int           k;
	const void   *alpha;
	const void   *A;
	int           Atype;     /* cudaDataType: 2=FP16, 4=FP32, ... */
	int           lda;
	const void   *B;
	int           Btype;
	int           ldb;
	const void   *beta;
	void         *C;
	int           Ctype;
	int           ldc;
	int           computeType;
	int           algo;
	cipher_rt_cuda_stream_t stream;  /* resolved via cublasGetStream_v2 */

	/* Reserved for future extension (e.g. cublasLtMatmul descriptors). */
	uint64_t reserved[4];
};

enum cipher_rt_matmul_result {
	CIPHER_RT_MATMUL_HANDLED     = 0,  /* actuator dispatched; substrate returns success */
	CIPHER_RT_MATMUL_PASSTHROUGH = 1,  /* actuator skipped; try next, then real cublas */
	CIPHER_RT_MATMUL_ERROR       = 2,  /* actuator attempted and failed; substrate falls back */
};

/* Actuator registration. */
struct cipher_rt_matmul_actuator {
	const char *name;
	int         priority;  /* lower runs first; 0..255; reserved for future use */
	int (*maybe_handle)(const struct cipher_rt_matmul_call *call,
	                    int *out_cublas_status);
	/* maybe_handle: returns CIPHER_RT_MATMUL_HANDLED on substitution
	 * (and fills out_cublas_status with the cuBLAS-style return code,
	 * typically 0/CUBLAS_STATUS_SUCCESS). Returns _PASSTHROUGH to skip.
	 * The actuator is responsible for any internal locking and for
	 * dispatching its replacement kernel on call->stream. */
};

/* Register an actuator. Called from actuator-init paths at libcipher_rt
 * load time (e.g. from cipher_rt_marlin_init via cipher_inject.c).
 * Returns 0 on success, -1 if registry full. */
int cipher_rt_matmul_register_actuator(
	const struct cipher_rt_matmul_actuator *actuator);

/* Substrate entry. Called by the cublasGemmEx shim with the call
 * descriptor + a function pointer to the real cublasGemmEx for
 * passthrough. Returns the cuBLAS-style status.
 *
 * Substrate behavior:
 *   - foreach registered actuator (in priority order):
 *       result = actuator->maybe_handle(call, &status)
 *       if result == HANDLED:  return status
 *       if result == ERROR:    fall through to real cublas (logged)
 *       if result == PASSTHROUGH: try next actuator
 *   - if no actuator handled: call passthrough_fn with same args
 */
typedef int (*cipher_rt_cublasGemmEx_passthrough_t)(
	cipher_rt_cublas_handle_t, int, int, int, int, int,
	const void *, const void *, int, int,
	const void *, int, int,
	const void *, void *, int, int, int, int);

int cipher_rt_matmul_dispatch(
	const struct cipher_rt_matmul_call *call,
	cipher_rt_cublasGemmEx_passthrough_t passthrough_fn);

/* D.10: actuator chain only (no passthrough) — for non-gemmEx callers (cublasLt
 * full shims). Returns CIPHER_RT_MATMUL_HANDLED (with *out_status) or _PASSTHROUGH. */
int cipher_rt_matmul_try_actuators(const struct cipher_rt_matmul_call *call,
                                   int *out_status);

/* Substrate init — called from libcipher_rt init body. Idempotent. */
int cipher_rt_matmul_dispatch_init(void);

/* Diagnostic accessors. */
unsigned long cipher_rt_matmul_calls_total(void);
unsigned long cipher_rt_matmul_calls_handled(void);
unsigned long cipher_rt_matmul_calls_passthrough(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_MATMUL_DISPATCH_H */
