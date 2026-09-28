/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_intercept.h — V1 Phase B B.6''.9.8.1 may13 driver-API intercept port.
 *
 * Public API for the cipher_intercept_cudart.cpp + cipher_graph_inspect.cpp
 * ports from cipher-may13-evidence (md5s 75c555/05b9ac at port time).
 *
 * Adapted from may13's __attribute__((constructor(101))) entry to an explicit
 * init function called from cipher_inject.c:cipher_v2_init_body() AFTER
 * cipher_rt_got_patch_init(). pthread_once in the caller gates idempotency;
 * compare_exchange in the impl is a second-layer safety net.
 *
 * Provides driver-API + cudart launch interception:
 *   - cuGetProcAddress / cuGetProcAddress_v2 (the cuGetProcAddress hook)
 *   - cuLaunchKernel + _ptsz + cuLaunchKernelEx + _ptsz
 *   - cudaLaunchKernel + _ptsz + __cudaLaunchKernel + _ptsz
 *   - cudaLaunchKernelExC + _ptsz
 *   - cuModuleLoadData + LoadDataEx + GetFunction
 *   - cuLibraryLoadData + GetKernel + cuKernelGetFunction
 *   - __cudaRegisterFatBinary + End + Function
 *   - cudaMemcpy + Async (env-gated probe)
 *
 * NOT provided (intentionally dropped to avoid conflict with cipher_rt_phase4
 * canonical hooks): cublasGemmEx, cublasLtMatmul, ncclAllReduce.
 */
#ifndef CIPHER_RT_INTERCEPT_H
#define CIPHER_RT_INTERCEPT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Idempotent. Called by cipher_inject.c:cipher_v2_init_body(). */
void cipher_rt_intercept_init(void);

/* Stats accessors — read-only. */
uint64_t cipher_intercept_count(void);

/* B.6''.9.8.4: graph-replay observability counter. Incremented per
 * cuGraphLaunch / cuGraphLaunch_ptsz / cuGraphLaunchPipelined call. */
uint64_t cipher_rt_graph_launch_calls(void);

/* B.6''.9.8.5: graph-capture observability counters.
 *
 * cipher_rt_graph_add_kernel_calls: incremented per cuGraphAddKernelNode
 *   / _v2 / cuGraphExecKernelNodeSetParams / _v2 / cudaGraphAddKernelNode.
 *
 * cipher_rt_graph_node_substitutions: incremented when
 *   cipher_rt_graph_substitute_decide returns a non-NULL substitute that
 *   differs from the original CUfunction. Substitution at graph capture
 *   time persists into all subsequent replays — no per-launch hook cost. */
uint64_t cipher_rt_graph_add_kernel_calls(void);
uint64_t cipher_rt_graph_node_substitutions(void);

/* GOT-patch repatch entry — for re-running dl_iterate_phdr after a new
 * CUDA-bearing DSO loads. Safe to call multiple times. */
void cipher_repatch(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_INTERCEPT_H */
