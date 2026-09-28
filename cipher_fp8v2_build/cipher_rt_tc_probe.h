/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_tc_probe — W10-12 Step 2: tensor-core saturation probe.
 *
 * Pure-function classifier that predicts whether a GEMM operation will
 * route through H100 tensor cores or fall through to CUDA cores. Three
 * inputs decide it:
 *
 *   Shape:     M, N, K must be divisible by the precision-specific
 *              alignment (16 for FP16/BF16, 32 for INT8, 16 for FP8).
 *              Mis-aligned shapes fall through to CUDA cores.
 *   Precision: FP16, BF16, INT8, FP8 are TC-eligible on H100 (sm_90).
 *              FP32 is NOT (no FP32 matmul TC pathway on H100).
 *   Layout:    A and B must use a layout pair the TC pipeline supports.
 *              cuBLAS handles most combos transparently, so the layout
 *              check is mostly a sanity guard against degenerate cases.
 *
 * The probe is producer-only in Step 2 — classifier consumer wiring
 * (CLASSIFY drain of TC_PROBE events) is reserved for W13-14 once the
 * Koopman tier lands. Step 2 ships the probe + event emission; downstream
 * use is future work.
 */
#ifndef CIPHER_RT_TC_PROBE_H
#define CIPHER_RT_TC_PROBE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum cipher_rt_tc_class {
	CIPHER_RT_TC_SATURABLE    = 0,  /* will use tensor cores */
	CIPHER_RT_TC_NEAR_MISS    = 1,  /* close to saturable; tune-able */
	CIPHER_RT_TC_INCOMPATIBLE = 2,  /* will fall through to CUDA cores */
};

enum cipher_rt_tc_precision {
	CIPHER_RT_TC_PREC_FP32 = 0,
	CIPHER_RT_TC_PREC_FP16 = 1,
	CIPHER_RT_TC_PREC_BF16 = 2,
	CIPHER_RT_TC_PREC_INT8 = 3,
	CIPHER_RT_TC_PREC_FP8  = 4,
};

enum cipher_rt_tc_layout {
	CIPHER_RT_TC_LAYOUT_ROW_MAJOR = 0,
	CIPHER_RT_TC_LAYOUT_COL_MAJOR = 1,
};

/* Pure-function classifier. No side effects, no atomics, no syscalls.
 * Returns one of cipher_rt_tc_class. Cost target: < 20 ns; called on
 * every cublasGemmEx intercept before dispatch. */
enum cipher_rt_tc_class
cipher_rt_tc_probe(uint32_t M, uint32_t N, uint32_t K,
                   enum cipher_rt_tc_precision prec,
                   enum cipher_rt_tc_layout layout_a,
                   enum cipher_rt_tc_layout layout_b,
                   enum cipher_rt_tc_layout layout_c);

/* Convert cublasGemmEx Atype/Btype/Ctype tags (cuDataType) to the
 * cipher_rt_tc_precision enum. Returns CIPHER_RT_TC_PREC_FP32 (the
 * NOT-TC-eligible bucket) for any unrecognized cuDataType. */
enum cipher_rt_tc_precision
cipher_rt_tc_precision_from_cudatype(int cu_type);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_TC_PROBE_H */
