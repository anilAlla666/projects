/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_arbitrate.h -- Phase 4 T4.2.3 ARBITRATE actuator.
 *
 * On first observation of a stream (cold path), call
 * CIPHER_REQUEST_SM_PARTITION (kmod ioctl nr 9). The hint is derived from
 * the tenant's position among active tenants:
 *
 *   - Active-tenant enumerate via cipher_rt_tenant_enumerate (Mode 2,
 *     /proc/cipher/stats parse).
 *   - Compute this tenant's launches_total rank.
 *   - Top quartile  -> hint = 8 (max per tenant).
 *   - Mid quartiles -> hint = 4.
 *   - Bottom quartile or unranked -> hint = 2.
 *
 * The kernel-side allocator (cipher_partition_request) clamps hint to
 * [1, 8] and returns the granted mask. On contention (free pool
 * insufficient), the kernel returns the partial mask it could grant. We
 * record that for telemetry but do not retry — Phase 4.7 will add the
 * back-off policy.
 *
 * Closes Gap 1 surfaced by advisor in T4.2.2 review: the ioctl nr 9
 * exists and is now actually called by the userland actuator.
 */
#ifndef CIPHER_RT_ARBITRATE_H
#define CIPHER_RT_ARBITRATE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize ARBITRATE state. Idempotent. */
int cipher_rt_arb_init(void);

/* Cold path: request a partition for this process. Called by
 * cipher_rt_pr_configure_locked() on first stream observation. Idempotent
 * per process (records the granted mask in process-local state so
 * subsequent first-stream observations skip the ioctl).
 * Returns granted mask, 0 on failure / not-tenant. */
unsigned int cipher_rt_arb_request_partition(void);

/* Last granted mask + popcount (telemetry). */
unsigned int cipher_rt_arb_last_mask(void);
unsigned int cipher_rt_arb_last_count(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_ARBITRATE_H */
