/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_remember_consumer.cpp -- W14 Step 3 S3.B1 REMEMBER consumer drain.
 *
 * Drains RING_WRITE slot 2 (CIPHER_RT_RING_CONSUMER_REMEMBER) into the CfC
 * LNN forward path. v1 scope is wire-up + correctness — the CfC weights
 * are read-only at runtime; the persistent hidden state h is updated by
 * cipher_lnn_forward(). Decision output is consumed but not actuated (the
 * Koopman actuator decisions are still driven by the .cu shape registry
 * via the matmul dispatch substrate).
 *
 * Loop structure: a single pthread iterates all 128 tenants per cycle,
 * draining up to BATCH_PER_TENANT entries per tenant; each entry triggers
 * one cipher_lnn_decide() call. A short usleep separates cycles when no
 * tenant had work in the last pass (avoids busy-spin).
 *
 * The consumer is the sole reader of slot 2 (per W7-9 Step 5 single-reader-
 * per-consumer-slot guarantee at cipher_rt_ring_write.c:9). Initial
 * read_seq[REMEMBER] is bumped to current write_seq at thread start so the
 * producer does not back-pressure on the consumer's cold-start cursor.
 *
 * Compiled as C++ because cipher_lnn.h transitively includes the C++
 * std::atomic in cipher_liquid_state.h:38; the public API is wrapped in
 * extern "C".
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>

extern "C" {
#include "cipher_rt_remember_consumer.h"
#include "cipher_rt_ring_write.h"
}
#include "include/may13/cipher_lnn.h"

#define BATCH_PER_TENANT  256      /* R-W14.3 mitigation; scope-lock line 147 */
#define IDLE_SLEEP_US     500      /* 0.5 ms between cycles when no work */

extern "C" {
extern CipherLnnDecision cipher_lnn_decide(
    CipherLnnState*             state,
    uint8_t                     op_class,
    uint32_t                    grid_x,
    uint32_t                    grid_y,
    uint32_t                    grid_z,
    uint32_t                    block_size,
    uint32_t                    shmem_bytes,
    const CipherLiquidStateMgr* liquid,
    float                       confidence_threshold);

extern void cipher_lnn_init(CipherLnnState* state);
}

static pthread_t                g_thread;
static std::atomic<int>         g_active{0};   /* set after pthread_create */
static std::atomic<int>         g_stop{0};     /* exit signal */
static std::atomic<uint64_t>    g_drained_total{0};
static std::atomic<uint64_t>    g_lnn_invocations{0};
static std::atomic<uint64_t>    g_active_cycles{0};
static std::atomic<uint64_t>    g_total_cycles{0};

/* CfC LNN state owned by the consumer. The CfC hidden state h here is the
 * "persistent" state mutated by drained REMEMBER events. */
static CipherLnnState           g_lnn_state;
static int                      g_lnn_initialized = 0;

/* Cold-start cursor sync: drain whatever's already in the ring at thread
 * start and discard. This advances read_seq[REMEMBER] to the current
 * write_seq for each tenant, so the producer does not see a back-pressured
 * REMEMBER slot.
 *
 * v1 housekeeping: ALSO advance read_seq[CLASSIFY] and read_seq[ORACLE]
 * cursors via drain-and-discard. v1 has no production drainer for those
 * slots (only producers exist: cipher_rt_classify_observer.c:102 and
 * cipher_rt_cublas_shim.c:172). Without this, the producer's min_read_seq()
 * stays at 0 and the overflow check at cipher_rt_ring_write.c:198 fires on
 * the 4097th emit per tenant — dropping all subsequent REMEMBER events that
 * the consumer was supposed to drain. Drained CLASSIFY/ORACLE entries are
 * discarded (process_entry filters non-REMEMBER events anyway). This is a
 * v1-only patch; v2 will introduce per-slot consumer registration so the
 * producer's overflow check considers only ACTIVE consumers. */
static void coldstart_sync_all_tenants(void)
{
    struct cipher_rt_ring_entry tmp[256];
    for (uint32_t t = 0; t < CIPHER_RT_RING_MAX_TENANTS; t++) {
        int drained;
        do {
            drained = cipher_rt_ring_drain(
                t, CIPHER_RT_RING_CONSUMER_REMEMBER, tmp, 256);
        } while (drained > 0);
        /* v1: advance the un-drained slots too. */
        do {
            drained = cipher_rt_ring_drain(
                t, CIPHER_RT_RING_CONSUMER_CLASSIFY, tmp, 256);
        } while (drained > 0);
        do {
            drained = cipher_rt_ring_drain(
                t, CIPHER_RT_RING_CONSUMER_ORACLE, tmp, 256);
        } while (drained > 0);
    }
}

/* Process one drained event. v1: synthesize CipherLnnInput from the
 * (M, K_dim, N_dim) shape and call cipher_lnn_decide with liquid=NULL so
 * the oracle gates are bypassed. The forward pass runs unconditionally
 * and updates state->h. */
static void process_entry(const struct cipher_rt_ring_entry *e)
{
    if (e->event_type != CIPHER_RT_RING_EVENT_REMEMBER) return;

    /* REMEMBER event payload layout matches cipher_rt_koopman_event in
     * cipher_rt_koopman_engine.cpp: { uint32_t M, K_dim, N_dim, op_class;
     * uint64_t ptr_B, ptr_C, reserved }. */
    uint32_t M     = ((const uint32_t*)e->payload)[0];
    uint32_t K_dim = ((const uint32_t*)e->payload)[1];
    uint32_t N_dim = ((const uint32_t*)e->payload)[2];

    /* op_class GEMM = 0. grid_x/y/z, block_size, shmem_bytes map from the
     * Koopman shape: M is batch, K is input feature dim, N is output dim.
     * block_size = 256 mirrors cuBLAS GemmEx typical block; shmem = 0
     * (FP16 GEMM tile fits in registers + L1 on H100 SM-90). liquid=NULL
     * skips the oracle gates (decision path runs forward unconditionally
     * for hidden-state update). threshold 0.5f is conservative; the
     * decision is not actuated in v1. */
    (void)cipher_lnn_decide(&g_lnn_state,
                            /* op_class */ 0u,
                            /* grid_x   */ M,
                            /* grid_y   */ K_dim,
                            /* grid_z   */ N_dim,
                            /* block    */ 256u,
                            /* shmem    */ 0u,
                            /* liquid   */ nullptr,
                            /* threshold*/ 0.5f);

    g_lnn_invocations.fetch_add(1, std::memory_order_relaxed);
}

static void *consumer_loop(void *arg)
{
    (void)arg;

    if (!g_lnn_initialized) {
        cipher_lnn_init(&g_lnn_state);
        g_lnn_initialized = 1;
    }
    coldstart_sync_all_tenants();

    struct cipher_rt_ring_entry batch[BATCH_PER_TENANT];

    while (!g_stop.load(std::memory_order_relaxed)) {
        uint64_t cycle_drained = 0;
        for (uint32_t t = 0; t < CIPHER_RT_RING_MAX_TENANTS; t++) {
            int n = cipher_rt_ring_drain(
                t, CIPHER_RT_RING_CONSUMER_REMEMBER,
                batch, BATCH_PER_TENANT);
            if (n > 0) {
                for (int i = 0; i < n; i++) process_entry(&batch[i]);
                cycle_drained += (uint64_t)n;
            }
            /* v1 housekeeping: advance CLASSIFY + ORACLE cursors via
             * drain-and-discard. Without this the producer's overflow
             * check drops emits after 4096 per tenant. See
             * coldstart_sync_all_tenants comment for the full story. */
            (void)cipher_rt_ring_drain(
                t, CIPHER_RT_RING_CONSUMER_CLASSIFY, batch, BATCH_PER_TENANT);
            (void)cipher_rt_ring_drain(
                t, CIPHER_RT_RING_CONSUMER_ORACLE, batch, BATCH_PER_TENANT);
        }
        g_total_cycles.fetch_add(1, std::memory_order_relaxed);
        if (cycle_drained > 0) {
            g_drained_total.fetch_add(cycle_drained,
                                      std::memory_order_relaxed);
            g_active_cycles.fetch_add(1, std::memory_order_relaxed);
        } else {
            usleep(IDLE_SLEEP_US);
        }
    }
    return nullptr;
}

extern "C" int cipher_rt_remember_consumer_init(void)
{
    const char *env = getenv("CIPHER_REMEMBER");
    int enabled = (env && env[0] == '1') ? 1 : 0;
    if (!enabled) {
        fprintf(stderr,
                "[CIPHER REMEMBER] env-gated off (CIPHER_REMEMBER!=1)\n");
        return 0;
    }

    int expected = 0;
    if (!g_active.compare_exchange_strong(expected, 1,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire)) {
        return 0;                                       /* already running */
    }

    /* Ensure ring substrate is initialized (idempotent). */
    cipher_rt_ring_write_init();

    if (pthread_create(&g_thread, nullptr, consumer_loop, nullptr) != 0) {
        g_active.store(0, std::memory_order_release);
        fprintf(stderr,
                "[CIPHER REMEMBER] pthread_create failed\n");
        return -1;
    }
    fprintf(stderr,
            "[CIPHER REMEMBER] consumer thread spawned "
            "(batch=%d tenants=%d slot=%u)\n",
            BATCH_PER_TENANT,
            (int)CIPHER_RT_RING_MAX_TENANTS,
            (unsigned)CIPHER_RT_RING_CONSUMER_REMEMBER);
    return 0;
}

extern "C" void cipher_rt_remember_consumer_exit(void)
{
    int was_active = g_active.exchange(0, std::memory_order_acq_rel);
    if (!was_active) return;
    g_stop.store(1, std::memory_order_release);
    pthread_join(g_thread, nullptr);
    g_stop.store(0, std::memory_order_release);
}

extern "C" int cipher_rt_remember_consumer_is_active(void)
{
    return g_active.load(std::memory_order_relaxed);
}

extern "C" uint64_t cipher_rt_remember_consumer_drained(void)
{
    return g_drained_total.load(std::memory_order_relaxed);
}

extern "C" uint64_t cipher_rt_remember_consumer_lnn_invocations(void)
{
    return g_lnn_invocations.load(std::memory_order_relaxed);
}

extern "C" uint64_t cipher_rt_remember_consumer_active_cycles(void)
{
    return g_active_cycles.load(std::memory_order_relaxed);
}

extern "C" uint64_t cipher_rt_remember_consumer_total_cycles(void)
{
    return g_total_cycles.load(std::memory_order_relaxed);
}
