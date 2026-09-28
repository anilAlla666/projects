/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_pool.h — W.4a cross-tenant POOL eligibility + correctness substrate.
 *
 * The UNBREAKABLE safety layer for Goal 1 heterogeneous multiplexing. W.4a is
 * DECISION-ONLY: it partitions co-resident tenants (W.6sC cohort) into
 * same-model coalesce-groups (W.6sB fingerprint) and provides the per-tenant
 * correctness backstop. It performs NO operand transport and NO coalescing —
 * every GEMM still passes through the single-tenant path. The cross-process
 * transport (CP 5.6 socket-IPC + decode-step-barrier coordinator) is the
 * sequenced W.4b re-port, gated by this layer.
 *
 * Memory #11 invariant (THE catastrophic guard): two tenants may share a
 * coalesce-group ONLY IF their model fingerprints are bit-identical and both
 * nonzero (warmed). Distinct or unwarmed fingerprints NEVER group — this is
 * what prevents silent cross-model corruption once W.4b transport exists.
 *
 * Memory #24 substrate-line: cohort read via the W.6sC ioctl client; decision
 * at the cuBLAS dispatch boundary. No application coupling.
 */
#ifndef CIPHER_RT_POOL_H
#define CIPHER_RT_POOL_H

#include <stdint.h>
#include <stddef.h>
#include "cipher_rt_coresidence.h"   /* struct cipher_cohort_peer */

#ifdef __cplusplus
extern "C" {
#endif

#define CIPHER_POOL_MAX 128   /* matches CIPHER_COHORT_MAX */

/* A coalesce-group descriptor. group_id == fingerprint so every same-model
 * tenant independently computes the same id (needed for W.4b coordination).
 * peer_tgids[] holds the same-fingerprint co-resident tenants (incl. self).
 * The (K,N,dtype) the group would coalesce at is recorded by observe_gemm for
 * W.4b; W.4a leaves it as the caller's most-recent GEMM shape. */
struct cipher_rt_pool_group {
    uint64_t group_id;        /* == fingerprint */
    uint64_t fingerprint;
    uint32_t self_tgid;       /* getpid() = CONTAINER pid; do NOT compare against
                               * peer_tgids[] (kmod stores HOST tgids). Identity
                               * is by fingerprint. W.4b resolves self's host tgid. */
    uint32_t n_in_group;      /* same-fp tenants incl. self (present in snapshot); >=2 => eligible */
    uint32_t distinct_rejected; /* co-resident peers with a DIFFERENT fp (never grouped) */
    uint32_t eligible;        /* 1 iff n_in_group>=2 && fingerprint!=0 */
    uint32_t peer_tgids[CIPHER_POOL_MAX];
    /* shape this group would coalesce at (recorded for W.4b; 0 until a GEMM seen) */
    int32_t  K, N, dtype;
};

/* === Pure decision functions (no GPU, no ioctl — unit-testable) ========== */

/* Partition the cohort into the caller's coalesce-group. Adds every peer whose
 * fingerprint == my_fingerprint (both nonzero) to the group; counts the rest as
 * distinct_rejected. self is always counted in n_in_group when my_fingerprint!=0.
 * my_fingerprint==0 (unwarmed) => solo group, eligible=0. NEVER groups a
 * distinct or zero fingerprint (Memory #11 structural guard). */
void cipher_rt_pool_partition(uint32_t self_tgid, uint64_t my_fingerprint,
                              const struct cipher_cohort_peer *peers, uint32_t n_peers,
                              struct cipher_rt_pool_group *out);

/* Per-tenant correctness backstop. For each tenant t, compute the max
 * element-wise relative difference between out_t and ref_t over `len` elements;
 * tenant passes iff max_rel_diff <= tol. Writes 1/0 into per_tenant_pass[t].
 * Returns the number of FAILING tenants (0 => all coherent). This is the check
 * that catches any fingerprint-collision edge before W.4b scatters results. */
uint32_t cipher_rt_pool_correctness_check(uint32_t n_tenants,
                                          const float *const *outputs,
                                          const float *const *refs,
                                          size_t len, double tol,
                                          int *per_tenant_pass);

/* === Live wiring (classifier + dispatch) ================================ */

/* Called from the classifier (~1/sec, same cadence as the cohort query) when
 * the workload is A1/multi-tenant. Reads the W.6sC cohort, partitions, and
 * publishes the current group + updates counters. peers/n come from
 * cipher_rt_coresidence_update so no extra ioctl is issued. */
void cipher_rt_pool_update(uint64_t my_fingerprint,
                           const struct cipher_cohort_peer *peers, uint32_t n_peers);

/* Per-GEMM dispatch hook (DECISION-ONLY in W.4a — always returns 0 = passthrough).
 * Records (K,N,dtype) onto the current group for W.4b and bumps counters. Cheap:
 * no ioctl, reads a published atomic snapshot. Returns 1 only if a future W.4b
 * transport would coalesce (always 0 here). */
int cipher_rt_pool_observe_gemm(int K, int N, int dtype);

/* Snapshot the current published group (caller-owned copy). */
void cipher_rt_pool_group_get(struct cipher_rt_pool_group *out);

/* === Enforcement (decision logic now; consumed by W.4b transport) ======= */

/* Mark a (fingerprint, K, N, dtype) group BLOCKED after a correctness failure;
 * eligibility returns solo for it thereafter. Tracks per-tenant + aggregate
 * block ratios and disables POOL per-tenant (>25%) or session-wide (>50%). */
void cipher_rt_pool_mark_blocked(uint64_t fingerprint, int K, int N, int dtype);
int  cipher_rt_pool_is_blocked(uint64_t fingerprint, int K, int N, int dtype);

/* === Counter accessors (CLASSIFY log + tests) =========================== */
uint64_t cipher_rt_pool_eligible_groups(void);   /* times a >=2 same-fp group formed */
uint64_t cipher_rt_pool_solo(void);              /* times caller was solo */
uint64_t cipher_rt_pool_distinct_fp_rejected(void); /* false-negative guard firings */
uint64_t cipher_rt_pool_blocked_by_correctness(void);
uint32_t cipher_rt_pool_current_group_size(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_POOL_H */
