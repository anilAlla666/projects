/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_sm_packer.h -- Phase 4 T4.2.3 SM_PACKER actuator.
 *
 * SM_PACKER detects sequences of consecutive small kernel launches from
 * the same stream and counts them as a "packing opportunity". Actual
 * substitution requires a resident persistent kernel (T4.2.4 PERSIST_ENGINE).
 *
 * This T4.2.3 build instruments only — it counts:
 *   - total launches observed
 *   - launches considered "small" (gridDimX * gridDimY * gridDimZ *
 *     blockDimX * blockDimY * blockDimZ < CIPHER_RT_SMP_SMALL_THRESHOLD)
 *   - longest small-launch streak
 *
 * These counters are visible via the cipher_rt_smp_* accessors. The
 * /proc/cipher exporter does not currently surface them; that's a Phase
 * 4.4 / 4.7 wiring step.
 */
#ifndef CIPHER_RT_SM_PACKER_H
#define CIPHER_RT_SM_PACKER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define CIPHER_RT_SMP_SMALL_THRESHOLD   (16 * 256)   /* 16 blocks × 256 threads */

int cipher_rt_smp_init(void);

/* Hot-path observation: record one launch with these dims. May be
 * called from CUPTI callback. */
void cipher_rt_smp_observe(unsigned int gridX, unsigned int gridY,
                           unsigned int gridZ, unsigned int blockX,
                           unsigned int blockY, unsigned int blockZ,
                           void *stream_handle);

unsigned long cipher_rt_smp_total_launches(void);
unsigned long cipher_rt_smp_small_launches(void);
unsigned long cipher_rt_smp_longest_streak(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_SM_PACKER_H */
