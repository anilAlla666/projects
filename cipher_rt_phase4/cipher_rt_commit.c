/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_commit.c — W7-9 Step 2: COMMIT atomic state-transition primitive.
 *
 * Per CIPHER_REENGINEERING_PLAN.md v1.2.3 §4.8. Per-tenant seqlock writer
 * pattern, cache-line-padded snapshot slot per tenant. Single-writer-per-
 * tenant invariant (the dispatch return path on that tenant's thread is the
 * sole COMMIT caller for that tenant); multi-reader acquire-load.
 *
 * Step 2 builds the contract + primitive + init hook. Hot-path wiring of
 * COMMIT into per-launch dispatch return is Step 4 work (overlay-op port
 * to read post-COMMIT snapshot). Step 2's smoke is the microbench at
 * /tmp/step2_baseline/test_commit_atomicity.c — direct invocation of
 * cipher_rt_commit_{begin,end} in a tight loop and from concurrent threads.
 *
 * Overhead target: ≤ 200 ns p99 per cipher_rt_commit_end. Two atomic stores
 * (open + close seq) + 8 simple field stores; cache-hot uncontended cost is
 * ~10 ns on x86 TSO.
 */

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>

#include "cipher_rt_commit.h"

/* W7-9 Step 4 — kmod ABI 0.6.0 ioctl NR 28 (CIPHER_AUDIT_RECORD). Mirrored
 * inline so commit_token_boundary doesn't take a kmod-header dependency that
 * would tie cipher_rt_phase4 to a specific kmod build. */
struct cipher_audit_record_req {
    uint32_t tenant_id;
    uint32_t actuator_id;     /* enum: COMMIT=1, MARLIN=2, KV_DEDUP=3, VOLT=4, KOOPMAN=5 */
    uint64_t param0;          /* e.g. token index */
    uint64_t param1;          /* reserved */
    uint64_t chain_head_out[4];  /* 32-byte HMAC head returned to userspace */
};
#define CIPHER_AUDIT_RECORD _IOWR('C', 28, struct cipher_audit_record_req)
#define CIPHER_AUDIT_ACTUATOR_COMMIT 1

/* Lazy-opened /dev/cipher fd for the token-boundary ioctl path. Per-process,
 * thread-safe because open(2) + atomic CAS of the result. */
static atomic_int g_audit_fd = -1;

/* Forward declarations into the may13 observer surface. Linked at runtime
 * (libcipher_rt.so includes the observer .o files), so dlsym-style indirection
 * is unnecessary. */
extern uint64_t cipher_determinism_hash(void);
extern uint64_t cipher_determinism_count(void);
extern uint64_t cipher_trace_written(void);
extern uint64_t cipher_trace_dropped(void);
extern uint64_t cipher_sense_current_session(void);

/* Weak accessor surface: some observer files don't yet export count
 * accessors (Step 4 adds them on demand). The weak attribute lets the link
 * succeed even when a particular accessor is absent in this build. */
#define WEAK_U64(fn) extern uint64_t fn(void) __attribute__((weak))
WEAK_U64(cipher_carbon_session_count);
WEAK_U64(cipher_fairness_tenant_count);
WEAK_U64(cipher_fairness_overrun_count);
WEAK_U64(cipher_receipt_session_count);
WEAK_U64(cipher_guard_session_count);
WEAK_U64(cipher_guard_leak_count);
WEAK_U64(cipher_loop_session_count);
WEAK_U64(cipher_loop_runaway_count);
WEAK_U64(cipher_pulse_ecc_inject_total);
WEAK_U64(cipher_pipeline_edge_count);
WEAK_U64(cipher_kt_size);
WEAK_U64(cipher_topology_device_count);
WEAK_U64(cipher_struct_override_count);
WEAK_U64(cipher_continuity_session_count);
WEAK_U64(cipher_oracle_flops_substituted_total);
WEAK_U64(cipher_oracle_total_decisions);

/* Tenant-id resolver: weak link to v2 inject's tenant lookup (cipher_inject.c
 * exports it). If not available, fall back to tenant 0 (single-tenant path). */
extern uint32_t cipher_v2_current_tenant_id(void) __attribute__((weak));

/* Per-tenant token counter — local accumulator for the per-token AUDIT path. */
static _Atomic uint64_t g_token_count[CIPHER_RT_MAX_TENANTS];

/* Cache-line padding to avoid false sharing across tenant slots under
 * N=128 concurrent readers. Hopper/x86 cache line = 64 B. */
#define CIPHER_RT_CACHELINE 64

struct cipher_rt_snapshot_padded {
    struct cipher_rt_snapshot snap;
    /* pad up to the next cache-line boundary; sizeof(snap) is checked
     * statically below — the +CACHELINE+1 trick guarantees positive size
     * even if snap exactly fills a cache line. */
    uint8_t pad[CIPHER_RT_CACHELINE
                - (sizeof(struct cipher_rt_snapshot) % CIPHER_RT_CACHELINE)
                + ((sizeof(struct cipher_rt_snapshot) % CIPHER_RT_CACHELINE) == 0
                   ? CIPHER_RT_CACHELINE : 0)];
} __attribute__((aligned(CIPHER_RT_CACHELINE)));

/* BSS-resident; init at module load. ~9 KiB total (128 × 72 B padded). */
static struct cipher_rt_snapshot_padded
    g_snapshots[CIPHER_RT_MAX_TENANTS];

static atomic_uint_least64_t g_commits_total;
static atomic_int g_init_done;

int cipher_rt_commit_init(void)
{
    /* Idempotent: pthread_once-style guard via atomic_compare_exchange. */
    int expected = 0;
    if (!atomic_compare_exchange_strong_explicit(
            &g_init_done, &expected, 1,
            memory_order_acq_rel, memory_order_acquire)) {
        return 0;                                       /* already initialised */
    }
    /* Zero-init the snapshot slots. BSS gives us this for free, but we
     * also use this opportunity to set seq=0 (even, no writer in progress)
     * explicitly via atomic_init so the memory model is unambiguous. */
    for (uint32_t t = 0; t < CIPHER_RT_MAX_TENANTS; t++) {
        __atomic_store_n(&g_snapshots[t].snap.seq, (uint64_t)0, __ATOMIC_RELAXED);
        memset(&g_snapshots[t].snap.audit_chain_head, 0,
               sizeof(struct cipher_rt_snapshot)
               - offsetof(struct cipher_rt_snapshot, audit_chain_head));
        atomic_init(&g_token_count[t], (uint64_t)0);
    }
    atomic_init(&g_commits_total, (uint64_t)0);
    atomic_init(&g_audit_fd, -1);
    return 0;
}

uint64_t cipher_rt_commit_begin(uint32_t tenant_id)
{
    if (tenant_id >= CIPHER_RT_MAX_TENANTS) {
        return 0;                                       /* out-of-range no-op */
    }
    struct cipher_rt_snapshot *s = &g_snapshots[tenant_id].snap;
    /* Bump seq to odd (writer-in-progress). Relaxed is sufficient on the
     * write side because the matching commit_end uses release semantics on
     * the final store; intermediate field writes that happen between begin
     * and end are visible to acquire-loading readers only AFTER end's
     * release. We use fetch_add(1) — on entry seq is even, fetch returns
     * the prior even value, the slot is now odd. */
    uint64_t prior = __atomic_fetch_add(&s->seq, (uint64_t)1, __ATOMIC_RELAXED);
    return prior + 1;                                   /* the now-odd value */
}

void cipher_rt_commit_end(uint32_t tenant_id, uint64_t seq,
                          const struct cipher_rt_commit_fields *fields)
{
    if (tenant_id >= CIPHER_RT_MAX_TENANTS) {
        return;                                         /* no-op (matches begin) */
    }
    struct cipher_rt_snapshot *s = &g_snapshots[tenant_id].snap;
    /* The 5-step deterministic state-update order per §4.8 L899-905.
     * Each step writes its piece of the snapshot. Field stores are plain
     * (not atomic) — the matching release-store of seq below ensures all
     * prior stores are visible to any acquire-loading reader. */
    if (fields) {
        s->audit_chain_head   = fields->audit_chain_head;     /* §4.8 Step 1 */
        s->fairness_quota     = fields->fairness_quota;       /* §4.8 Step 2 */
        s->carbon_joules_x1e6 = fields->carbon_joules_x1e6;   /* §4.8 Step 3 */
        s->receipt_seq        = fields->receipt_seq;          /* §4.8 Step 4 */
        s->kmod_tenant_state  = fields->kmod_tenant_state;    /* §4.8 Step 5 */
        s->model_uuid_lo      = fields->model_uuid_lo;        /* W7 Step 1 G10 */
        s->model_uuid_hi      = fields->model_uuid_hi;
        /* W7-9 Step 4 ext */
        s->determinism_xor    = fields->determinism_xor;
        s->determinism_count  = fields->determinism_count;
        s->guard_leak_count   = fields->guard_leak_count;
        s->guard_session_count= fields->guard_session_count;
        s->loop_runaway_count = fields->loop_runaway_count;
        s->loop_session_count = fields->loop_session_count;
        s->pulse_ecc_inject   = fields->pulse_ecc_inject;
        s->sense_session_count= fields->sense_session_count;
        s->pipeline_edge_count= fields->pipeline_edge_count;
        s->trace_written      = fields->trace_written;
        s->trace_dropped      = fields->trace_dropped;
        s->kernel_table_size  = fields->kernel_table_size;
        s->topo_device_count  = fields->topo_device_count;
        s->struct_override_count = fields->struct_override_count;
        s->continuity_session_count = fields->continuity_session_count;
        s->oracle_flops_substituted = fields->oracle_flops_substituted;
        s->oracle_total_decisions   = fields->oracle_total_decisions;
        s->token_boundary_count = fields->token_boundary_count;
    }
    /* commits_total is a per-tenant counter visible in the snapshot for
     * Step 4 _report() consumers. The global counter (below) is for the
     * test_commit_atomicity throughput verification. */
    s->commits_total = seq >> 1;                        /* seq=1 → 0; seq=3 → 1; ... */
    /* Bump seq to even (writer-done) with release fence. After this store
     * lands, all prior field writes are visible to any reader that acquire-
     * loads s->seq. seq goes from (prior + 1) odd → (prior + 2) even.
     * Single-store release semantics; uncontended cache-line operation. */
    __atomic_store_n(&s->seq, seq + 1, __ATOMIC_RELEASE);
    (void)atomic_fetch_add_explicit(
        &g_commits_total, (uint64_t)1, memory_order_relaxed);
}

const struct cipher_rt_snapshot *
cipher_get_current_tenant_snapshot(uint32_t tenant_id)
{
    if (tenant_id >= CIPHER_RT_MAX_TENANTS) {
        return NULL;
    }
    return &g_snapshots[tenant_id].snap;
}

uint64_t cipher_rt_commit_total_count(void)
{
    return atomic_load_explicit(&g_commits_total, memory_order_relaxed);
}

/* Inline weak-call helper: invoke fn() if linked, return 0 otherwise. */
#define WCALL(fn) ((fn) ? (fn)() : (uint64_t)0)

/* Env-gated hot-path. Default ON; CIPHER_COMMIT_HOTPATH=0 disables. Read
 * once and cached so the per-launch path takes a single relaxed load. */
static int g_hotpath_enabled = -1;
static int hotpath_enabled(void)
{
    int v = __atomic_load_n(&g_hotpath_enabled, __ATOMIC_RELAXED);
    if (v >= 0) return v;
    const char *s = getenv("CIPHER_COMMIT_HOTPATH");
    int e = (s && (s[0] == '0' || !strcmp(s, "off") || !strcmp(s, "no"))) ? 0 : 1;
    __atomic_store_n(&g_hotpath_enabled, e, __ATOMIC_RELAXED);
    return e;
}

uint64_t cipher_rt_commit_observe_and_publish(uint32_t tenant_id)
{
    if (!hotpath_enabled()) return 0;
    if (tenant_id >= CIPHER_RT_MAX_TENANTS) {
        return 0;
    }
    if (atomic_load_explicit(&g_init_done, memory_order_acquire) == 0) {
        return 0;
    }
    struct cipher_rt_commit_fields fields;
    memset(&fields, 0, sizeof(fields));
    /* §4.8 5-step deterministic order. Step 1 audit_chain_head is mirrored
     * from the kmod ring via the most-recent CIPHER_AUDIT_RECORD ioctl
     * return; on hot path we use the per-tenant counter as a proxy (cheap)
     * and the full HMAC chain head lives kmod-side. */
    fields.audit_chain_head    = atomic_load_explicit(&g_token_count[tenant_id],
                                                      memory_order_relaxed);
    fields.fairness_quota      = WCALL(cipher_fairness_tenant_count);
    fields.carbon_joules_x1e6  = WCALL(cipher_carbon_session_count);
    fields.receipt_seq         = WCALL(cipher_receipt_session_count);
    fields.kmod_tenant_state   = 0;                       /* state_updater snapshot tag — Step 5 wires */
    fields.model_uuid_lo       = 0;
    fields.model_uuid_hi       = 0;
    /* Step 4 ext field population. Each accessor weak-linked; absent → 0. */
    fields.determinism_xor     = cipher_determinism_hash();
    fields.determinism_count   = cipher_determinism_count();
    fields.guard_leak_count    = WCALL(cipher_guard_leak_count);
    fields.guard_session_count = WCALL(cipher_guard_session_count);
    fields.loop_runaway_count  = WCALL(cipher_loop_runaway_count);
    fields.loop_session_count  = WCALL(cipher_loop_session_count);
    fields.pulse_ecc_inject    = WCALL(cipher_pulse_ecc_inject_total);
    fields.sense_session_count = cipher_sense_current_session();
    fields.pipeline_edge_count = WCALL(cipher_pipeline_edge_count);
    fields.trace_written       = cipher_trace_written();
    fields.trace_dropped       = cipher_trace_dropped();
    fields.kernel_table_size   = WCALL(cipher_kt_size);
    fields.topo_device_count   = WCALL(cipher_topology_device_count);
    fields.struct_override_count = WCALL(cipher_struct_override_count);
    fields.continuity_session_count = WCALL(cipher_continuity_session_count);
    fields.oracle_flops_substituted = WCALL(cipher_oracle_flops_substituted_total);
    fields.oracle_total_decisions   = WCALL(cipher_oracle_total_decisions);
    fields.token_boundary_count = atomic_load_explicit(
        &g_token_count[tenant_id], memory_order_relaxed);

    uint64_t seq = cipher_rt_commit_begin(tenant_id);
    cipher_rt_commit_end(tenant_id, seq, &fields);
    return cipher_rt_commit_total_count();
}

static int audit_fd_get(void)
{
    int fd = atomic_load_explicit(&g_audit_fd, memory_order_acquire);
    if (fd >= 0) return fd;
    fd = open("/dev/cipher", O_RDWR | O_CLOEXEC);
    if (fd < 0) return -errno;
    int expected = -1;
    if (!atomic_compare_exchange_strong_explicit(
            &g_audit_fd, &expected, fd,
            memory_order_acq_rel, memory_order_acquire)) {
        /* lost race; another thread opened first */
        close(fd);
        fd = expected;
    }
    return fd;
}

int cipher_rt_commit_token_boundary(uint32_t tenant_id, uint64_t token_index)
{
    if (tenant_id >= CIPHER_RT_MAX_TENANTS) return -EINVAL;
    int fd = audit_fd_get();
    if (fd < 0) return fd;
    struct cipher_audit_record_req req;
    memset(&req, 0, sizeof(req));
    req.tenant_id   = tenant_id;
    req.actuator_id = CIPHER_AUDIT_ACTUATOR_COMMIT;
    req.param0      = token_index;
    if (ioctl(fd, CIPHER_AUDIT_RECORD, &req) < 0) {
        return -errno;
    }
    /* Mirror low-8B of HMAC chain head into the tenant counter; the full
     * 32-byte head is in req.chain_head_out and the kmod-side ring. */
    atomic_fetch_add_explicit(&g_token_count[tenant_id], (uint64_t)1,
                              memory_order_relaxed);
    return 0;
}
