/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_green_ctx.h -- Phase 4 T4.2.4a Green Context scaffold.
 *
 * Lazily creates one CUDA Green Context per process at first observation
 * of cuStreamCreate (or cudaStreamCreate). The green context covers ALL
 * device SMs for now (no real partition restriction). Streams created
 * while the green context is current become bound to it; this is the
 * plumbing layer for T4.2.4b which will restrict to actual partition
 * subsets keyed by ARBITRATE's mask.
 *
 * Hook strategy: CUPTI API_ENTER callback for cuStreamCreate /
 * cuStreamCreateWithPriority / cudaStreamCreate pushes our green-ctx
 * CUcontext as current. API_EXIT pops. The cuStreamCreate proceeds with
 * the green context as current, so the resulting stream is bound to it.
 */
#ifndef CIPHER_RT_GREEN_CTX_H
#define CIPHER_RT_GREEN_CTX_H

#ifdef __cplusplus
extern "C" {
#endif

/* Lazily create the per-process green context. Idempotent. Returns 0
 * on success, -1 on failure (logged once). Safe to call from any
 * thread; internally locked. */
int cipher_rt_green_ctx_ensure(void);

/* Push the green-ctx CUcontext as current on the calling thread.
 * Returns 0 on success, -1 if green ctx is not initialized. */
int cipher_rt_green_ctx_push(void);

/* Pop one context from the calling thread. */
void cipher_rt_green_ctx_pop(void);

/* T4.2.4d: persistent enforcement. Ensures the green CUcontext is the
 * calling thread's current context, replacing whatever was there.
 * Cheap fast path (returns immediately if already green). Idempotent.
 * Returns 0 if green is current after the call, -1 if green ctx not
 * initialized (still no-op). */
int cipher_rt_green_ctx_make_current(void);

/* Diagnostic accessors. */
unsigned long cipher_rt_green_ctx_streams_observed(void);
int cipher_rt_green_ctx_is_initialized(void);
/* T4.2.4d: handle for the diagnostic in cipher_cupti.c. Returned as
 * void * to avoid a cuda.h dependency in this header. The caller in
 * cipher_cupti.c casts to CUgreenCtx. */
void *cipher_rt_green_ctx_handle_voidp(void);

/* CP 5.3 STEP 2: the VERIFIED SM count of this process's green context
 * (the value cuGreenCtxGetDevResource returned at create time), or 0 if
 * no green context is initialized. The Marlin engine reads this to set
 * grid = partition SM count instead of the device SM count. */
unsigned int cipher_rt_green_ctx_sm_count(void);

/* CP 5.3 STEP 2: which of the NUM_GROUPS partition groups this process
 * bound to. Used by the STEP 2 multi-partition gate to confirm two test
 * processes did not hash-collide into the same partition. Meaningful only
 * when cipher_rt_green_ctx_is_initialized() is true. */
unsigned int cipher_rt_green_ctx_group_id(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_GREEN_CTX_H */
