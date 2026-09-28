/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_fairness — D.8 FAIRNESS + SHIELD, libcipher_rt side.
 *
 * Userspace half of the kmod cross-tenant work-ledger (cipher_fairness_ledger
 * in cipher_kmod, NR 32 + RW mmap window at pgoff 0x200000). Each tenant
 * self-accounts its GEMM volume into its own ledger slot and reads the whole
 * region to decide whether to self-throttle. Enforcement is TIMING-ONLY —
 * a bounded CPU sleep BEFORE the GEMM is submitted — so kernels, arguments,
 * and per-tenant submission order are untouched and per-tenant output is
 * bit-identical (KL=0 by construction, Mem #11).
 *
 * Two capabilities, one mechanism:
 *   R-D5 FAIRNESS — a throughput tenant (band 0) bursting past ~2x the active
 *     average self-throttles, so victims keep their quota share.
 *   R-I1 SHIELD   — a latency-sensitive tenant (band >=1) is never throttled;
 *     a band-0 tenant additionally yields when it out-paces an active
 *     latency-sensitive neighbor, bounding that neighbor's p99.
 *
 * Default-OFF: unarmed (no CIPHER_FAIRNESS / CIPHER_SHIELD env) => every entry
 * point is a no-op and behavior is byte-identical to the pre-D.8 substrate.
 */
#ifndef CIPHER_RT_FAIRNESS_H
#define CIPHER_RT_FAIRNESS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Lazy, idempotent. Reads env, and if armed: opens /dev/cipher, registers via
 * NR 32 (sets this tenant's band), RW-maps the ledger window. Safe to call
 * repeatedly from the hot path; does its work once. No-op (and stays
 * disarmed) if env is unset or the kmod lacks NR 32 / the region. */
void cipher_rt_fairness_init(void);

/* Hot-path entry: record one GEMM for this tenant, then — if armed and the
 * band-modulated burst/SHIELD rule fires — perform a bounded CPU sleep before
 * returning (the caller then submits the GEMM). No-op when disarmed. */
void cipher_rt_fairness_record_and_maybe_throttle(void);

/* Diagnostics (for the gate harness; 0 when disarmed). */
unsigned long long cipher_rt_fairness_self_calls(void);
unsigned long long cipher_rt_fairness_self_yields(void);
int                cipher_rt_fairness_armed(void);
int                cipher_rt_fairness_self_band(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_FAIRNESS_H */
