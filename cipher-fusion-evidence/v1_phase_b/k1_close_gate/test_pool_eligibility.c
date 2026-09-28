/* W.4a — deterministic eligibility-partition + correctness-backstop test.
 *
 * Verifies the UNBREAKABLE Memory #11 guard WITHOUT a GPU: the pure partition
 * function never groups distinct/zero fingerprints, and the correctness backstop
 * flags forced divergence. Exercises M1 (homogeneous), M2 (heterogeneous —
 * THE catastrophic false-negative case), M3 (mixed), unwarmed, and the backstop.
 *
 * Build: gcc -O2 -I/home/ubuntu/cipher_rt_phase4 -o test_pool_eligibility \
 *          test_pool_eligibility.c /home/ubuntu/cipher_rt_phase4/cipher_rt_pool.c -lm -lpthread
 */
#include <stdio.h>
#include <string.h>
#include "cipher_rt_pool.h"

#define FP_A 0xd40e456727923bceULL   /* Llama-3 */
#define FP_B 0x9fde4562bbbb73ceULL   /* Mistral */
#define FP_C 0x458e457798608581ULL   /* TinyLlama-AWQ */

static int g_fail = 0;
#define CHECK(c,m,...) do{ if(c) printf("  PASS: " m "\n",##__VA_ARGS__); else {printf("  FAIL: " m "\n",##__VA_ARGS__); g_fail++;} }while(0)

/* Reconstruct: how many of the group's peer_tgids actually carry my_fp?
 * (cross-validates the Memory #11 invariant against the input). */
static int count_group_with_fp(const struct cipher_rt_pool_group *g,
                               const struct cipher_cohort_peer *peers, uint32_t n,
                               uint64_t fp, uint32_t self_tgid)
{
    int ok = 1;
    for (uint32_t i = 0; i < g->n_in_group; i++) {
        uint32_t t = g->peer_tgids[i];
        if (t == self_tgid) continue;          /* self always legitimately in group */
        uint64_t pf = 0; int found = 0;
        for (uint32_t j = 0; j < n; j++) if (peers[j].tgid == t) { pf = peers[j].model_fingerprint; found = 1; break; }
        if (!found || pf != fp) ok = 0;        /* a grouped tgid that is NOT same-fp => guard breach */
    }
    return ok;
}

int main(void)
{
    struct cipher_rt_pool_group g;

    /* NOTE: self_tgid is deliberately NOT one of the peer tgids in these cases,
     * mirroring the real PID-namespace mismatch (kmod records host tgid; the
     * substrate's getpid() is the container tgid). Identity is by fingerprint;
     * self is already present in the cohort snapshot, so n_in_group must NOT be
     * inflated by a phantom self (the W.4a group_size=2 single-tenant bug). */

    printf("[ST1] single-tenant: snapshot has only self (1 same-fp entry, self_tgid mismatched)\n");
    struct cipher_cohort_peer st1[1] = {{2495996,0,FP_A}};  /* host tgid */
    cipher_rt_pool_partition(171 /*container pid != host tgid*/, FP_A, st1, 1, &g);
    CHECK(g.n_in_group==1, "single-tenant n_in_group=1 (got %u) — NO phantom self", g.n_in_group);
    CHECK(g.eligible==0, "single-tenant solo (eligible=0)");
    CHECK(g.distinct_rejected==0, "distinct_rejected=0");

    printf("[M1] homogeneous: 4x same fp (FP_A), self_tgid mismatched\n");
    struct cipher_cohort_peer m1[4] = {{100,0,FP_A},{101,0,FP_A},{102,0,FP_A},{103,0,FP_A}};
    cipher_rt_pool_partition(999 /*not in list*/, FP_A, m1, 4, &g);
    CHECK(g.n_in_group==4, "n_in_group=4 (got %u) — exactly the snapshot count, no phantom", g.n_in_group);
    CHECK(g.eligible==1, "eligible=1");
    CHECK(g.distinct_rejected==0, "distinct_rejected=0 (got %u)", g.distinct_rejected);
    CHECK(g.group_id==FP_A, "group_id==fingerprint");
    CHECK(count_group_with_fp(&g,m1,4,FP_A,100), "[#11] every grouped tgid carries FP_A");

    printf("[M2] heterogeneous: FP_A(self) + FP_B  — THE catastrophic case\n");
    struct cipher_cohort_peer m2[2] = {{200,0,FP_A},{201,0,FP_B}};
    cipher_rt_pool_partition(200, FP_A, m2, 2, &g);
    CHECK(g.n_in_group==1, "self solo: n_in_group=1 (got %u)", g.n_in_group);
    CHECK(g.eligible==0, "eligible=0 (no cross-model group)");
    CHECK(g.distinct_rejected==1, "distinct_fp_rejected=1 (FP_B never grouped)");
    CHECK(count_group_with_fp(&g,m2,2,FP_A,200), "[#11] FP_B NOT in group — guard holds");
    /* and from B's perspective, symmetric */
    cipher_rt_pool_partition(201, FP_B, m2, 2, &g);
    CHECK(g.n_in_group==1 && g.distinct_rejected==1, "[#11] symmetric: B also solo, rejects A");

    printf("[M3] mixed: FP_A, FP_A, FP_B\n");
    struct cipher_cohort_peer m3[3] = {{300,0,FP_A},{301,0,FP_A},{302,0,FP_B}};
    cipher_rt_pool_partition(300, FP_A, m3, 3, &g);
    CHECK(g.n_in_group==2, "FP_A group of 2 (got %u)", g.n_in_group);
    CHECK(g.eligible==1, "eligible=1");
    CHECK(g.distinct_rejected==1, "FP_B rejected");
    CHECK(count_group_with_fp(&g,m3,3,FP_A,300), "[#11] only FP_A tenants grouped");

    printf("[WARMUP] unwarmed self fp=0 => conservative solo\n");
    struct cipher_cohort_peer w[2] = {{400,0,FP_A},{401,0,FP_A}};
    cipher_rt_pool_partition(400, 0 /*unwarmed*/, w, 2, &g);
    CHECK(g.eligible==0 && g.n_in_group==0, "unwarmed never groups (eligible=0)");

    printf("[WARMUP2] peer unwarmed (fp=0) is NOT grouped\n");
    struct cipher_cohort_peer w2[2] = {{500,0,FP_A},{501,0,0 /*peer not warm*/}};
    cipher_rt_pool_partition(500, FP_A, w2, 2, &g);
    CHECK(g.n_in_group==1 && g.distinct_rejected==1, "unwarmed peer rejected (fp=0 != FP_A)");

    printf("[BACKSTOP] correctness check: same matches, forced-distinct flagged\n");
    float a[4]={1,2,3,4}, b[4]={1,2,3,4}, c[4]={1,2,3,9 /*diverged*/};
    const float *outs_ok[2]={a,b}; const float *refs_ok[2]={a,b};
    int pass[2];
    uint32_t fails = cipher_rt_pool_correctness_check(2, outs_ok, refs_ok, 4, 0.01, pass);
    CHECK(fails==0 && pass[0] && pass[1], "same-fp coalesced outputs PASS (fails=0)");
    const float *outs_bad[2]={a,c}; const float *refs_bad[2]={a,b};
    fails = cipher_rt_pool_correctness_check(2, outs_bad, refs_bad, 4, 0.01, pass);
    CHECK(fails==1 && pass[0]==1 && pass[1]==0, "forced divergence FLAGGED (tenant1 fail) — backstop works");

    printf("\n=== test_pool_eligibility: %s (%d failures) ===\n", g_fail?"FAIL":"PASS", g_fail);
    return g_fail?1:0;
}
