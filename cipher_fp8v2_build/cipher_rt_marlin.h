/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_marlin.h -- T4.5.2 public API for the Marlin actuator.
 *
 * Marlin is the first actuator landing on the T4.5.1 matmul-routing
 * substrate. It plugs in via cipher_rt_matmul_register_actuator(), gates
 * eligible cublasGemmEx calls (M ≤ 8, FP16 fp16 dtypes, K/N shape match),
 * lazy-quantizes weights to INT4 + Marlin XOR-swizzled layout on first
 * observation, then dispatches the Marlin INT4 GEMM kernel for subsequent
 * calls on the same weight pointer.
 *
 * Env vars:
 *   CIPHER_MARLIN=on|off   (default off)
 *   CIPHER_MARLIN_VERBOSE=1 (default 0; logs per-launch hits)
 */
#ifndef CIPHER_RT_MARLIN_H
#define CIPHER_RT_MARLIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Init Marlin: resolve NVRTC, kick the cubin compile (lazy on first
 * eligible call), register with the substrate. Idempotent. Returns 0
 * on success, -1 if NVRTC unavailable. */
int cipher_rt_marlin_init(void);

/* GC-on-free (2026-05-30): cache lifecycle API for model load/free/reload
 * churn (Goal-1 100 bursty agents). The residence/owner layer calls
 * evict_weight when a model's weights are freed (completing the bind_model
 * lifecycle); reclaim_retired drains the retired device buffers at a quiescent
 * point (it device-synchronizes first, so it is safe to call only when no
 * Marlin GEMM must remain in flight — e.g. after torch.cuda.synchronize at a
 * per-cycle barrier). Both are no-ops when nothing has been evicted, so they
 * never perturb the steady (no-churn) path. */
void cipher_rt_marlin_engine_evict_weight(const void *app_w_ptr);
int  cipher_rt_marlin_engine_reclaim_retired(void);
unsigned long cipher_rt_marlin_engine_gc_fp_failures(void);
/* Safe-reclaim borrow bookkeeping: release is called by the actuator after each
 * dispatch; the accessor is a self-check (must read 0 when no GEMM is mid
 * lookup→launch — assert after a soak). */
void cipher_rt_marlin_engine_release_dispatch_borrow(void);
int  cipher_rt_marlin_engine_dispatch_borrow(void);

/* Diagnostic accessors. */
int  cipher_rt_marlin_is_active(void);
unsigned long cipher_rt_marlin_calls_total(void);    /* eligible calls seen */
unsigned long cipher_rt_marlin_calls_handled(void);  /* dispatched to marlin kernel */
unsigned long cipher_rt_marlin_weights_quantized(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_MARLIN_H */
