/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_coresidence.h — W.6 sub-C substrate co-residence client.
 *
 * Thin libcipher_rt client over the kmod GPU co-residence registry
 * (cipher_kmod CIPHER_COHORT_REGISTER nr 30 / CIPHER_COHORT_QUERY nr 31).
 * The classifier calls cipher_rt_coresidence_update() each classify pass to
 *   (1) heartbeat-or-insert this process's substrate model fingerprint
 *       (cipher_workload_model_fingerprint, W.6 sub-B) into the kmod table,
 *   (2) learn how many distinct tenant processes share this GPU right now.
 * W.4 POOL consumes (co-resident pids + per-pid fingerprint) to form legal
 * cross-tenant GEMM-coalescing groups.
 *
 * Memory #30 substrate-line: kmod ioctl only; no plugin, no app coupling.
 */
#ifndef CIPHER_RT_CORESIDENCE_H
#define CIPHER_RT_CORESIDENCE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Mirror of struct cipher_cohort_entry_abi (cipher_kmod cipher_ioctl.h). */
struct cipher_cohort_peer {
	uint32_t tgid;
	uint32_t _pad;
	uint64_t model_fingerprint;
};

/* Called by the classifier on each classify pass. Registers/heartbeats the
 * caller's fingerprint into the kmod co-residence table (NR 31, throttled to
 * ~1 ioctl/sec) and returns the current number of distinct LIVE tenant
 * processes on this GPU, INCLUDING self (>= 1 once registered).
 *
 *   my_fingerprint : cipher_workload_model_fingerprint(); 0 => not yet warmed
 *                    up (no heartbeat is sent; the last cached count returns).
 *   peers_out      : optional; up to max_peers live peers (incl. self) written.
 *   n_total_out    : optional; receives the TRUE live count (may exceed
 *                    max_peers => peers_out truncated).
 *
 * Never fails up the launch path — on any error (kmod absent, ioctl error)
 * the last cached count is returned (default 0). */
uint32_t cipher_rt_coresidence_update(uint64_t my_fingerprint,
                                      struct cipher_cohort_peer *peers_out,
                                      uint32_t max_peers,
                                      uint32_t *n_total_out);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_CORESIDENCE_H */
