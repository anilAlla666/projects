/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_pool.c — W.4a cross-tenant POOL eligibility + correctness substrate.
 * See cipher_rt_pool.h. DECISION-ONLY: no operand transport, no coalescing.
 */
#define _GNU_SOURCE
#include "cipher_rt_pool.h"

#include <string.h>
#include <math.h>
#include <stdatomic.h>
#include <pthread.h>

/* === Counters ============================================================ */
static _Atomic uint64_t g_eligible_groups   = 0;
static _Atomic uint64_t g_solo              = 0;
static _Atomic uint64_t g_distinct_rejected = 0;
static _Atomic uint64_t g_blocked_by_corr   = 0;

/* === Published current group (double-buffer; publisher = classifier thread) */
static struct cipher_rt_pool_group g_group_a, g_group_b;
static _Atomic int g_group_cur = 0;   /* 0 => a is live, 1 => b is live */

/* === Blocked-group table (small open list; W.4b consumes) ================ */
#define BLK_MAX 256
struct blk { uint64_t fp; int K, N, dtype; int used; };
static struct blk g_blk[BLK_MAX];
static pthread_mutex_t g_blk_lock = PTHREAD_MUTEX_INITIALIZER;

/* === Pure: partition ===================================================== */
void cipher_rt_pool_partition(uint32_t self_tgid, uint64_t my_fingerprint,
                              const struct cipher_cohort_peer *peers, uint32_t n_peers,
                              struct cipher_rt_pool_group *out)
{
    memset(out, 0, sizeof(*out));
    out->self_tgid   = self_tgid;
    out->fingerprint = my_fingerprint;
    out->group_id    = my_fingerprint;        /* deterministic across same-model tenants */
    out->K = out->N = out->dtype = 0;

    if (my_fingerprint == 0) {
        /* Unwarmed — conservative solo, never group. */
        out->n_in_group = 0;
        out->eligible   = 0;
        /* still count distinct co-resident peers as rejected (observability) */
        for (uint32_t i = 0; i < n_peers; i++)
            if (peers[i].model_fingerprint != my_fingerprint)
                out->distinct_rejected++;
        return;
    }

    /* The W.6sC cohort QUERY heartbeat-or-inserts the caller, so the snapshot
     * ALREADY contains self — n_in_group is simply the count of same-fingerprint
     * entries. Do NOT inject self separately: the kmod keys on current->tgid
     * (host PID namespace), but getpid() inside a container returns the
     * container-namespace PID, so self_tgid != self's snapshot tgid. Matching on
     * self_tgid would never fire and double-count self (the W.4a group_size=2
     * single-tenant bug). Identity-by-fingerprint is namespace-agnostic and
     * exactly what coalescing needs (same weights). */
    uint32_t n = 0, rej = 0;
    for (uint32_t i = 0; i < n_peers; i++) {
        uint64_t pfp = peers[i].model_fingerprint;
        if (pfp == my_fingerprint) {
            /* Memory #11: ONLY bit-identical, nonzero fingerprints group. */
            if (n < CIPHER_POOL_MAX)
                out->peer_tgids[n] = peers[i].tgid;
            n++;
        } else {
            rej++;   /* distinct OR zero fp => never grouped (false-negative guard) */
        }
    }
    out->n_in_group        = n;     /* includes self (present in snapshot) */
    out->distinct_rejected = rej;
    out->eligible          = (n >= 2) ? 1u : 0u;
    (void)self_tgid;                /* informational only (see out->self_tgid) */
}

/* === Pure: per-tenant correctness backstop =============================== */
uint32_t cipher_rt_pool_correctness_check(uint32_t n_tenants,
                                          const float *const *outputs,
                                          const float *const *refs,
                                          size_t len, double tol,
                                          int *per_tenant_pass)
{
    uint32_t fails = 0;
    for (uint32_t t = 0; t < n_tenants; t++) {
        double maxrel = 0.0;
        const float *o = outputs[t], *r = refs[t];
        for (size_t i = 0; i < len; i++) {
            double d = fabs((double)o[i] - (double)r[i]);
            double den = fabs((double)r[i]);
            double rel = den > 1e-12 ? d / den : d;   /* abs fallback near zero */
            if (rel > maxrel) maxrel = rel;
        }
        int ok = (maxrel <= tol);
        if (per_tenant_pass) per_tenant_pass[t] = ok;
        if (!ok) fails++;
    }
    return fails;
}

/* === Live: publish a freshly-partitioned group ========================== */
void cipher_rt_pool_update(uint64_t my_fingerprint,
                           const struct cipher_cohort_peer *peers, uint32_t n_peers)
{
    /* tgid == process tgid for the substrate (getpid). Avoid a syscall on the
     * hot cadence by reusing the cohort's self entry if present; else getpid. */
    extern int getpid(void);
    uint32_t self_tgid = (uint32_t)getpid();

    int cur = atomic_load_explicit(&g_group_cur, memory_order_relaxed);
    struct cipher_rt_pool_group *dst = cur ? &g_group_a : &g_group_b; /* write the inactive buffer */
    cipher_rt_pool_partition(self_tgid, my_fingerprint, peers, n_peers, dst);

    if (dst->eligible) atomic_fetch_add_explicit(&g_eligible_groups, 1, memory_order_relaxed);
    else               atomic_fetch_add_explicit(&g_solo, 1, memory_order_relaxed);
    if (dst->distinct_rejected)
        atomic_fetch_add_explicit(&g_distinct_rejected, dst->distinct_rejected, memory_order_relaxed);

    atomic_store_explicit(&g_group_cur, cur ? 0 : 1, memory_order_release); /* publish */
}

/* === Live: per-GEMM hook (DECISION-ONLY — always passthrough in W.4a) ==== */
int cipher_rt_pool_observe_gemm(int K, int N, int dtype)
{
    int cur = atomic_load_explicit(&g_group_cur, memory_order_acquire);
    struct cipher_rt_pool_group *g = cur ? &g_group_b : &g_group_a;
    if (g->eligible && !cipher_rt_pool_is_blocked(g->fingerprint, K, N, dtype)) {
        /* Record the shape this group would coalesce at (for W.4b). Benign
         * racy write to the live buffer — single int fields, observability only. */
        g->K = K; g->N = N; g->dtype = dtype;
    }
    /* W.4a performs NO coalescing — transport is W.4b. */
    return 0;
}

void cipher_rt_pool_group_get(struct cipher_rt_pool_group *out)
{
    int cur = atomic_load_explicit(&g_group_cur, memory_order_acquire);
    *out = cur ? g_group_b : g_group_a;
}

/* === Enforcement ======================================================== */
void cipher_rt_pool_mark_blocked(uint64_t fingerprint, int K, int N, int dtype)
{
    pthread_mutex_lock(&g_blk_lock);
    int free_slot = -1;
    for (int i = 0; i < BLK_MAX; i++) {
        if (g_blk[i].used && g_blk[i].fp == fingerprint &&
            g_blk[i].K == K && g_blk[i].N == N && g_blk[i].dtype == dtype) {
            pthread_mutex_unlock(&g_blk_lock); return; /* already blocked */
        }
        if (!g_blk[i].used && free_slot < 0) free_slot = i;
    }
    if (free_slot >= 0) {
        g_blk[free_slot] = (struct blk){ fingerprint, K, N, dtype, 1 };
        atomic_fetch_add_explicit(&g_blocked_by_corr, 1, memory_order_relaxed);
    }
    pthread_mutex_unlock(&g_blk_lock);
}

int cipher_rt_pool_is_blocked(uint64_t fingerprint, int K, int N, int dtype)
{
    pthread_mutex_lock(&g_blk_lock);
    for (int i = 0; i < BLK_MAX; i++)
        if (g_blk[i].used && g_blk[i].fp == fingerprint &&
            g_blk[i].K == K && g_blk[i].N == N && g_blk[i].dtype == dtype) {
            pthread_mutex_unlock(&g_blk_lock); return 1;
        }
    pthread_mutex_unlock(&g_blk_lock);
    return 0;
}

/* === Counter accessors ================================================== */
uint64_t cipher_rt_pool_eligible_groups(void){ return atomic_load_explicit(&g_eligible_groups, memory_order_relaxed); }
uint64_t cipher_rt_pool_solo(void){ return atomic_load_explicit(&g_solo, memory_order_relaxed); }
uint64_t cipher_rt_pool_distinct_fp_rejected(void){ return atomic_load_explicit(&g_distinct_rejected, memory_order_relaxed); }
uint64_t cipher_rt_pool_blocked_by_correctness(void){ return atomic_load_explicit(&g_blocked_by_corr, memory_order_relaxed); }
uint32_t cipher_rt_pool_current_group_size(void){ struct cipher_rt_pool_group g; cipher_rt_pool_group_get(&g); return g.n_in_group; }
