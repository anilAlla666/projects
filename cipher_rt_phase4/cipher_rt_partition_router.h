/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_partition_router.h -- Phase 4 T4.2.2 PARTITION_ROUTER actuator.
 *
 * On first observation of a CUDA stream (intercepted via CUPTI launch
 * callback), set stream attributes derived from the tenant snapshot:
 *   - CU_LAUNCH_ATTRIBUTE_PRIORITY        (priority from launches_total
 *                                          quartile, depth win #1a)
 *   - CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP (per-tenant sync domain to
 *                                          prevent cross-tenant fence stalls,
 *                                          depth win #1b)
 *
 * Subsequent launches on the same stream are an O(1) hashtable check.
 *
 * SM partition mask itself (from cipher_kmod ioctl nr 9) is fetched lazily
 * via cipher_rt_tenant snapshot (Mode 3 TLS cache, refreshed by background
 * poll). The router records the mask in the per-stream entry for downstream
 * Green Context binding (T4.2.4); for T4.2.2 we only set the attribute pair.
 */
#ifndef CIPHER_RT_PARTITION_ROUTER_H
#define CIPHER_RT_PARTITION_ROUTER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize partition router state. Idempotent. Called from
 * cipher_v2_init_body() after tenant register / CUPTI subscribe. */
int cipher_rt_pr_init(void);

/* Hot-path entry, called from the CUPTI launch callback BEFORE the
 * existing launch-count logic. stream may be 0 (cudaStreamLegacy / NULL)
 * which is treated as a real value (cached separately from other handles).
 * Returns 0 on success, negative on internal error (never propagated to
 * the launch, but logged via cipher_log()). */
int cipher_rt_pr_observe_stream(void *stream_handle);

/* Diagnostic: number of unique streams observed since init. */
unsigned long cipher_rt_pr_observed_count(void);

/* Diagnostic: number of streams successfully configured with stream
 * attributes. */
unsigned long cipher_rt_pr_configured_count(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_PARTITION_ROUTER_H */
