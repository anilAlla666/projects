/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_tc_probe — pure-function GEMM saturability classifier.
 *
 * Decision table on H100 sm_90:
 *
 *   precision     | alignment | TC-eligible | notes
 *   FP32          | n/a       | no          | CUDA cores only
 *   FP16          | 16        | yes         | wmma m16n16k16 fragments
 *   BF16          | 16        | yes         | same family as FP16
 *   INT8          | 32        | yes         | IMMA, k must align 32
 *   FP8           | 16        | yes         | E4M3/E5M2; cuBLAS routes
 *
 * NEAR_MISS heuristic: shape within 1 alignment unit of saturable (i.e.
 * one of M, N, K is `aligned + 1..alignment-1` while the other two are
 * aligned). Pad-by-1 would route this back to saturable; useful for the
 * downstream classifier (W13-14) to recommend shape adjustment.
 */
#include "cipher_rt_tc_probe.h"

/* cuDataType values from cuda/library_types.h. Mirrored here so the
 * probe stays header-light. */
#define CUDA_R_16F   2   /* FP16 */
#define CUDA_R_32F   0   /* FP32 */
#define CUDA_R_16BF 14   /* BF16 */
#define CUDA_R_8I    3   /* INT8 */
#define CUDA_R_8F_E4M3 28  /* FP8 E4M3 */
#define CUDA_R_8F_E5M2 29  /* FP8 E5M2 */

enum cipher_rt_tc_precision
cipher_rt_tc_precision_from_cudatype(int cu_type)
{
	switch (cu_type) {
	case CUDA_R_16F:    return CIPHER_RT_TC_PREC_FP16;
	case CUDA_R_16BF:   return CIPHER_RT_TC_PREC_BF16;
	case CUDA_R_8I:     return CIPHER_RT_TC_PREC_INT8;
	case CUDA_R_8F_E4M3:
	case CUDA_R_8F_E5M2:
		return CIPHER_RT_TC_PREC_FP8;
	case CUDA_R_32F:
	default:            return CIPHER_RT_TC_PREC_FP32;
	}
}

static inline uint32_t prec_alignment(enum cipher_rt_tc_precision prec)
{
	switch (prec) {
	case CIPHER_RT_TC_PREC_FP16: return 16;
	case CIPHER_RT_TC_PREC_BF16: return 16;
	case CIPHER_RT_TC_PREC_INT8: return 32;
	case CIPHER_RT_TC_PREC_FP8:  return 16;
	default:                     return 0;     /* FP32: no TC alignment */
	}
}

enum cipher_rt_tc_class
cipher_rt_tc_probe(uint32_t M, uint32_t N, uint32_t K,
                   enum cipher_rt_tc_precision prec,
                   enum cipher_rt_tc_layout layout_a,
                   enum cipher_rt_tc_layout layout_b,
                   enum cipher_rt_tc_layout layout_c)
{
	/* Precision check: FP32 is never TC-eligible on H100 matmul. */
	uint32_t align = prec_alignment(prec);
	if (align == 0) return CIPHER_RT_TC_INCOMPATIBLE;

	(void)layout_a; (void)layout_b; (void)layout_c;
	/* H100 cuBLAS handles all four (row/col × row/col) layout combos
	 * transparently for FP16/BF16/INT8/FP8. The layout fields are
	 * recorded here for v2 telemetry but not gated against in v1. */

	/* Shape alignment: all of M, N, K must be align-divisible to hit
	 * the wmma fragment shapes (m16n16k16 family for FP16/BF16,
	 * m16n16k32 for INT8). */
	uint32_t m_rem = M % align;
	uint32_t n_rem = N % align;
	uint32_t k_rem = K % align;

	if (m_rem == 0 && n_rem == 0 && k_rem == 0) {
		return CIPHER_RT_TC_SATURABLE;
	}

	/* NEAR_MISS: exactly one of M/N/K is mis-aligned but within one
	 * alignment unit (i.e. trivially pad-able). The downstream
	 * classifier can recommend a small pad to route back to TC. */
	int misaligned = (m_rem != 0) + (n_rem != 0) + (k_rem != 0);
	if (misaligned == 1) {
		return CIPHER_RT_TC_NEAR_MISS;
	}

	return CIPHER_RT_TC_INCOMPATIBLE;
}
