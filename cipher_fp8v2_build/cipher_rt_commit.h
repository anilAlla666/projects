/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_commit.h — W7-9 Step 2: COMMIT atomic state-transition primitive.
 *
 * Per CIPHER_REENGINEERING_PLAN.md v1.2.3 §4.8 (lines 893-919), COMMIT is the
 * named atomic primitive that promotes dispatch return from implicit phase
 * to a publish-under-release-fence point. The 5-step deterministic order:
 *
 *   1. AUDIT chain HMAC entry          (cipher_rt_audit_record)
 *   2. FAIRNESS quota + arbitration    (cipher_fairness_observe + SHM aggregate)
 *   3. CARBON power accounting         (cipher_carbon_observe)
 *   4. RECEIPT billing surface         (cipher_receipt_observe)
 *   5. kmod-resident tenant state      (cipher_pid_stats via state_updater)
 *
 * Atomicity uses a per-tenant seqlock writer pattern (single writer per tenant,
 * many readers). NOT a global lock — at N=128 tenants a global lock would
 * serialise the entire dispatch hot path. Per-tenant slot is cache-line-padded
 * to avoid false sharing.
 *
 * Step 2 establishes the snapshot CONTRACT + primitive. Step 4 ports the 21
 * overlay-op _report() functions to read this snapshot under the acquire-fence
 * reader pattern below.
 */
#ifndef CIPHER_RT_COMMIT_H
#define CIPHER_RT_COMMIT_H

#include <stdint.h>
#include <string.h>

/* C/C++ ABI-clash workaround. C++ files (the may13 ports + the SDPA
 * dispatch shim) already include <atomic>; pulling <stdatomic.h> here
 * collides with libstdc++'s shared_ptr_atomic.h. We expose the seq slot
 * as a plain uint64_t in the struct layout and use GCC __atomic builtins
 * (which produce identical machine code to C11 atomic_* and C++
 * std::atomic) for all loads, stores, fences. cipher_rt_commit.c keeps
 * <stdatomic.h> internally for its own non-public state — the public
 * struct + inline helpers are atomic-header-free. */

#ifdef __cplusplus
extern "C" {
#endif

/* Per-tenant cap. Must be >= CIPHER_CP54_MAX_ALLOCS (currently 128 post-W6 G1). */
#define CIPHER_RT_MAX_TENANTS 128

/* Snapshot published by COMMIT. Overlay ops' _report() (Step 4) read under
 * the seqlock reader pattern. The 5 fields below mirror the 5-step order
 * from §4.8; additional fields may be appended by Step 4 as overlay ops
 * surface their counter needs. Layout is forward-compatible (no field
 * removal; new fields only appended). */
struct cipher_rt_snapshot {
    uint64_t              seq;                /* seqlock: odd = writer-in-progress, even = published; access via __atomic_*_n */
    /* §4.8 5-step deterministic order (Step 2 substrate) */
    uint64_t              audit_chain_head;   /* Step 1 of §4.8 — Step 4 mirrors kmod chain head low-8B */
    uint64_t              fairness_quota;     /* Step 2 of §4.8 — per-tenant work units */
    uint64_t              carbon_joules_x1e6; /* Step 3 of §4.8 — fixed-point µJ accumulator */
    uint64_t              receipt_seq;        /* Step 4 of §4.8 — per-session billing chain */
    uint64_t              kmod_tenant_state;  /* Step 5 of §4.8 — snapshot tag from state_updater */
    uint64_t              model_uuid_lo;      /* W7-9 Step 1 G10 — low 64 bits of model_uuid */
    uint64_t              model_uuid_hi;      /* W7-9 Step 1 G10 — high 64 bits of model_uuid */
    uint64_t              commits_total;      /* total COMMIT calls on this tenant since init */
    /* W7-9 Step 4 overlay-op counter mirrors (mechanical-port targets) */
    uint64_t              determinism_xor;    /* cipher_determinism_hash() */
    uint64_t              determinism_count;  /* cipher_determinism_count() */
    uint64_t              guard_leak_count;   /* cipher_guard leak total */
    uint64_t              guard_session_count;/* cipher_guard_session_count */
    uint64_t              loop_runaway_count; /* cipher_loop runaway total */
    uint64_t              loop_session_count; /* cipher_loop session total */
    uint64_t              pulse_ecc_inject;   /* cipher_pulse ECC inject delta */
    uint64_t              sense_session_count;/* cipher_sense_current_session count */
    uint64_t              pipeline_edge_count;/* cipher_pipeline Jaccard edge count */
    uint64_t              trace_written;      /* cipher_trace_written() */
    uint64_t              trace_dropped;      /* cipher_trace_dropped() */
    uint64_t              kernel_table_size;  /* cipher_kt observed-kernel count */
    uint64_t              topo_device_count;  /* cipher_topology device count */
    uint64_t              struct_override_count; /* struct_lookup override count */
    uint64_t              continuity_session_count; /* cipher_continuity session count */
    uint64_t              oracle_flops_substituted; /* oracle_billing FLOPs-sub */
    uint64_t              oracle_total_decisions;   /* oracle phase-detector decisions */
    uint64_t              token_boundary_count;     /* per-token AUDIT records emitted */
};

/* Initialize the per-tenant snapshot table. Idempotent. Called from
 * cipher_v2_init_body (cipher_inject.c) after tenant/CUPTI init. */
int cipher_rt_commit_init(void);

/* Begin a COMMIT cycle on `tenant_id`. Bumps the seqlock to odd (writer-
 * in-progress marker) and returns the new odd seq. Caller MUST follow with
 * cipher_rt_commit_end() to close the cycle. */
uint64_t cipher_rt_commit_begin(uint32_t tenant_id);

/* End a COMMIT cycle. Writes the 5 snapshot fields in §4.8 deterministic
 * order, then bumps the seqlock to even with release fence. After return,
 * acquire-loading readers see the new snapshot.
 *
 * The `seq` parameter is the value returned by cipher_rt_commit_begin().
 * Step 4 will refine field-source plumbing as overlay ops are ported; Step 2
 * accepts NULL field sources and writes zeros (the primitive correctness is
 * orthogonal to the field values). */
struct cipher_rt_commit_fields {
    uint64_t audit_chain_head;
    uint64_t fairness_quota;
    uint64_t carbon_joules_x1e6;
    uint64_t receipt_seq;
    uint64_t kmod_tenant_state;
    uint64_t model_uuid_lo;
    uint64_t model_uuid_hi;
    /* W7-9 Step 4 ext */
    uint64_t determinism_xor;
    uint64_t determinism_count;
    uint64_t guard_leak_count;
    uint64_t guard_session_count;
    uint64_t loop_runaway_count;
    uint64_t loop_session_count;
    uint64_t pulse_ecc_inject;
    uint64_t sense_session_count;
    uint64_t pipeline_edge_count;
    uint64_t trace_written;
    uint64_t trace_dropped;
    uint64_t kernel_table_size;
    uint64_t topo_device_count;
    uint64_t struct_override_count;
    uint64_t continuity_session_count;
    uint64_t oracle_flops_substituted;
    uint64_t oracle_total_decisions;
    uint64_t token_boundary_count;
};

void cipher_rt_commit_end(uint32_t tenant_id, uint64_t seq,
                          const struct cipher_rt_commit_fields *fields);

/* W7-9 Step 4 — hot-path observe-and-publish helper. Queries each observer's
 * accessor (cipher_carbon_*, cipher_fairness_*, etc.), packs into a fields
 * struct, calls begin + end. Single seqlock cycle; intended for invocation
 * from the post-dispatch path (cipher_rt_cublasGemmEx_impl post return) and
 * from the SDPA dispatch shim. Cost target: ≤ 200 ns p99 uncontended.
 *
 * Returns commits_total post-bump (for trace). Returns 0 if init not done. */
uint64_t cipher_rt_commit_observe_and_publish(uint32_t tenant_id);

/* W7-9 Step 4 — per-token AUDIT boundary hook. Emits one CIPHER_AUDIT_RECORD
 * ioctl per logical decode token. Called from cipher_vllm_kvdedup plugin's
 * post-step hook (Option β per scope-lock §6). Returns 0 on success, -errno
 * on ioctl failure. Idempotent w.r.t. /dev/cipher fd caching. */
int cipher_rt_commit_token_boundary(uint32_t tenant_id, uint64_t token_index);

/* Snapshot accessor for overlay-op _report() consumers (Step 4 ports them
 * to this). Returned pointer is stable for the process lifetime — readers
 * acquire-load the seq, read fields, acquire-load seq again, retry if
 * mismatch or odd. Returns NULL if tenant_id is out of range. */
const struct cipher_rt_snapshot *
cipher_get_current_tenant_snapshot(uint32_t tenant_id);

/* Helper for the seqlock reader pattern. Returns 0 on a coherent read,
 * non-zero if the snapshot was being written or the caller should retry.
 * Step 4 reader code can use the open-coded pattern; this helper exists
 * for the test_commit_atomicity microbench. */
static inline int
cipher_rt_snapshot_consistent(const struct cipher_rt_snapshot *snap,
                              uint64_t s1)
{
    uint64_t s2;
    if (s1 & 1ULL) return 1;                           /* writer in progress */
    s2 = __atomic_load_n(&snap->seq, __ATOMIC_ACQUIRE);
    return (s1 == s2) ? 0 : 1;                         /* changed → retry */
}

/* W7-9 Step 4 — coherent snapshot copy. Overlay-op _report() callers use
 * this to read a coherent picture of the COMMIT-published state under the
 * acquire-fence reader pattern. The output `out` is fully zeroed if the
 * tenant has no published snapshot yet (init not run, or out-of-range id).
 *
 * Slow-path API — used by _report() functions that emit JSON. Hot-path
 * fields should still read direct atomics where they are the source of
 * truth (the snapshot is a publish copy, not a primary).
 */
static inline void
cipher_rt_snapshot_acquire(uint32_t tenant_id,
                           struct cipher_rt_snapshot *out)
{
    const struct cipher_rt_snapshot *snap =
        cipher_get_current_tenant_snapshot(tenant_id);
    if (!snap) {
        memset(out, 0, sizeof(*out));
        return;
    }
    uint64_t s1 = 0, s2 = 0;
    int spins = 0;
    /* Explicit `for (;;) ... break` (NOT do-while) — the do-while form has
     * a subtle bug: the `continue` after the odd-s1 retry skips the s2
     * load, so the while-clause `s1 != s2` is checked against stale stack
     * garbage. If stack garbage happens to equal s1, the loop falsely
     * exits with an incoherent snapshot. for(;;) + break is bug-free. */
    for (;;) {
        s1 = __atomic_load_n(&snap->seq, __ATOMIC_ACQUIRE);
        if (s1 & 1ULL) {                              /* writer mid-commit */
            if (++spins > 1024) {                     /* fallback: stale read */
                memset(out, 0, sizeof(*out));
                return;
            }
            continue;
        }
        memcpy(out, snap, sizeof(*out));
        s2 = __atomic_load_n(&snap->seq, __ATOMIC_ACQUIRE);
        if (s1 == s2) break;
    }
    /* Coherent exit: s1 == s2, both even. Normalize out->seq to s1 with
     * __atomic_store_n so the compiler does not elide this store as
     * redundant with the memcpy of snap->seq (the live snap byte may have
     * been bumped between the s1 load and the memcpy read of that byte
     * under concurrent writers; out->seq from memcpy alone is not safe). */
    __atomic_store_n(&out->seq, s1, __ATOMIC_RELAXED);
}

/* Total COMMIT calls observed since cipher_rt_commit_init. For
 * test_commit_atomicity throughput verification. */
uint64_t cipher_rt_commit_total_count(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_COMMIT_H */
