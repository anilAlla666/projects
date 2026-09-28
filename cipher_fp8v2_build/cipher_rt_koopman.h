/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_koopman.h -- W14 Step 2 B.2 Koopman engine public API.
 *
 * The Koopman engine is the second actuator on the T4.5.1 matmul-routing
 * substrate (cipher_rt_matmul_dispatch). It plugs in via
 * cipher_rt_matmul_register_actuator(), gates eligible cublasGemmEx calls
 * to FP16-only shapes that have been calibrated into the .cu shape registry
 * by cipher_edmd_live_collect's background randomized-SVD pipeline, then
 * dispatches the cipher_koopman_fp16_launch_shape() narrow-domain O(1)
 * surrogate kernel from cipher_block_sub_kernel.cu.
 *
 * Three guards collapse into the .cu shape registry membership check:
 *   1. registry-hit (model-keyed)    — only calibrated shapes register
 *   2. shape in seeded narrow domain — calibration ran on real GEMM
 *                                       snapshots inside the narrow class
 *   3. model_uuid match              — calibration is per-shape per-model
 *
 * Env vars:
 *   CIPHER_KOOPMAN=1        (default 0; off for v1 safety)
 *   CIPHER_KOOPMAN_VERBOSE=1 (default 0; logs per-launch hits)
 *
 * Wire-up pattern parallel to cipher_rt_marlin_init() / cipher_rt_marlin.h.
 */
#ifndef CIPHER_RT_KOOPMAN_H
#define CIPHER_RT_KOOPMAN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Init Koopman engine: register with the substrate. Idempotent.
 * Returns 0 on success (or when env-gated off), -1 on substrate-register
 * failure. Called from cipher_inject.c at libcipher_rt load time. */
int cipher_rt_koopman_init(void);

/* Diagnostic accessors. */
int  cipher_rt_koopman_is_active(void);
unsigned long cipher_rt_koopman_calls_total(void);    /* eligible by env+dtype */
unsigned long cipher_rt_koopman_calls_handled(void);  /* dispatched to Koopman .cu kernel */
unsigned long cipher_rt_koopman_calls_skipped(void);  /* shape not in registry */
unsigned long cipher_rt_koopman_remember_emits(void); /* W14 S3.B0 RING_WRITE slot 3 emissions */

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_KOOPMAN_H */
