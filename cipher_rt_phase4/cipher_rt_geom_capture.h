/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_geom_capture.h -- TEST-ONLY launch-geometry capture (default-OFF).
 *
 * CIPHER_GEOM_CAPTURE=1 dumps one JSONL line per unique kernel fn-pointer to
 * /tmp/cipher_geom_capture.jsonl: the captured launch geometry + the
 * geometry-classifier verdict (cipher::classify_launch via classify_route's
 * out.op_class). Joined offline (on fn) with /tmp/cipher_kernel_table.json
 * (fn -> name -> name_class) to validate geometry-recognition vs the name
 * oracle. Default-OFF => byte-identical; observe-only (no substitution).
 */
#ifndef CIPHER_RT_GEOM_CAPTURE_H
#define CIPHER_RT_GEOM_CAPTURE_H
#ifdef __cplusplus
extern "C" {
#endif
void cipher_rt_geom_capture(const void *fn,
                            unsigned gx, unsigned gy, unsigned gz,
                            unsigned bx, unsigned by, unsigned bz,
                            unsigned shmem, int geom_class);

/* Geometry-keyed attention intercept (CIPHER_GEOM_ATTN) — framework-agnostic
 * attention detection counts. cipher_rt_geom_attn_intercepts() is validated
 * against the symbol-keyed 6-pattern's FA count (parity) on vLLM, and proves
 * agnosticism on a second framework (torch SDPA) where the 6-pattern is 0. */
unsigned long cipher_rt_geom_attn_intercepts(void);
unsigned long cipher_rt_geom_launches(void);
#ifdef __cplusplus
}
#endif
#endif /* CIPHER_RT_GEOM_CAPTURE_H */
