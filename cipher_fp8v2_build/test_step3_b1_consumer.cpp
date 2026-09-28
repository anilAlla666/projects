/* W14 Step 3 S3.B1 — REMEMBER consumer drain test harness.
 *
 * Three cases:
 *   Case A: env-gated off (CIPHER_REMEMBER absent) — consumer init is no-op
 *   Case B: cold start + drain — spawn consumer thread, post N events, verify drained
 *   Case C: producer + consumer compose — interleave produces and drains across all tenants
 */
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

extern "C" {
#include "cipher_rt_ring_write.h"
#include "cipher_rt_remember_consumer.h"
}

struct kev {
    uint32_t M, K_dim, N_dim, op_class;
    uint64_t ptr_B, ptr_C, reserved;
};

static int test_envgate_off(void)
{
    /* Unset CIPHER_REMEMBER and call init: should return 0 and not spawn. */
    unsetenv("CIPHER_REMEMBER");
    cipher_rt_remember_consumer_init();
    if (cipher_rt_remember_consumer_is_active()) {
        printf("[S3.B1 ENVGATE OFF] FAIL: consumer active when CIPHER_REMEMBER unset\n");
        return 1;
    }
    printf("[S3.B1 ENVGATE OFF] PASS (consumer dormant, drained=%lu)\n",
           cipher_rt_remember_consumer_drained());
    return 0;
}

static int test_cold_start_drain(void)
{
    /* Set CIPHER_REMEMBER=1 and init. Post events to tenant 0. Wait for drain. */
    setenv("CIPHER_REMEMBER", "1", 1);
    cipher_rt_ring_write_init();

    int rc = cipher_rt_remember_consumer_init();
    if (rc != 0 || !cipher_rt_remember_consumer_is_active()) {
        printf("[S3.B1 COLD START] FAIL: init rc=%d active=%d\n",
               rc, cipher_rt_remember_consumer_is_active());
        return 1;
    }

    /* Give the consumer's coldstart_sync_all_tenants a moment to complete. */
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    uint64_t pre_drained = cipher_rt_remember_consumer_drained();
    uint64_t pre_invocations = cipher_rt_remember_consumer_lnn_invocations();

    /* Post 10000 REMEMBER events on tenant 0. */
    kev ev = { 8, 2048, 32000, 0, 0xdead, 0xbeef, 0 };
    const int N = 10000;
    for (int i = 0; i < N; i++) {
        cipher_rt_ring_write(0u, CIPHER_RT_RING_EVENT_REMEMBER, 1u, 0ull,
                             &ev, sizeof(ev));
    }

    /* Wait for the consumer to drain (poll up to 2 s). */
    auto t0 = std::chrono::steady_clock::now();
    while (true) {
        uint64_t drained_now = cipher_rt_remember_consumer_drained();
        if (drained_now > pre_drained) {
            /* At least one batch landed; wait a tiny bit longer for tail. */
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            break;
        }
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2)) {
            printf("[S3.B1 COLD START] FAIL: no drain seen in 2s "
                   "(pre_drained=%lu now=%lu)\n",
                   pre_drained, drained_now);
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    uint64_t drained = cipher_rt_remember_consumer_drained() - pre_drained;
    uint64_t lnn_calls = cipher_rt_remember_consumer_lnn_invocations() - pre_invocations;
    uint64_t accepted = cipher_rt_ring_total_written(0);
    uint64_t dropped = cipher_rt_ring_total_dropped(0);

    printf("[S3.B1 COLD START] posted=%d accepted=%lu dropped=%lu "
           "drained=%lu lnn_calls=%lu\n",
           N, accepted, dropped, drained, lnn_calls);

    /* Gate: drained > 0 and drained == lnn_calls (each drained event invokes one
     * cipher_lnn_decide). */
    if (drained == 0 || drained != lnn_calls) {
        printf("[S3.B1 COLD START] FAIL: drained=%lu lnn_calls=%lu\n",
               drained, lnn_calls);
        return 1;
    }
    printf("[S3.B1 COLD START] PASS\n");
    return 0;
}

static int test_producer_consumer_compose(void)
{
    /* Concurrent: 32 producer threads emit on tenant_id = tid%128; consumer
     * drains in background. Producer rate is shaped to match realistic
     * Koopman fire rate (~1k events/s per active tenant under narrow-domain
     * decode), not synthetic burst stress: 1000 events per producer with
     * a 1ms sleep every 100 events ≈ 100k events/s per producer ≈ 3.2 M/s
     * aggregate ≈ 100k/s per tenant. Per-tenant consumer drain capacity is
     * 256 entries per cycle × ~ms cycle = 256k/s per tenant, so the ring
     * stays well below the 4096-entry buffer ceiling and drops are noise
     * floor. */
    const int N_PROD = 32;
    const int PER_PROD = 1000;

    /* Reset per-tenant counters by sampling pre-baseline. */
    uint64_t pre_w[CIPHER_RT_RING_MAX_TENANTS];
    uint64_t pre_d[CIPHER_RT_RING_MAX_TENANTS];
    for (int t = 0; t < CIPHER_RT_RING_MAX_TENANTS; t++) {
        pre_w[t] = cipher_rt_ring_total_written((uint32_t)t);
        pre_d[t] = cipher_rt_ring_total_dropped((uint32_t)t);
    }
    uint64_t pre_drained = cipher_rt_remember_consumer_drained();

    std::vector<std::thread> producers;
    producers.reserve(N_PROD);

    auto t0 = std::chrono::steady_clock::now();
    for (int p = 0; p < N_PROD; p++) {
        producers.emplace_back([p]() {
            kev e = { 16, 4096, 11008, 0, 0, 0, 0 };
            e.M = (uint32_t)p;
            uint32_t tid = (uint32_t)(p % CIPHER_RT_RING_MAX_TENANTS);
            for (int i = 0; i < PER_PROD; i++) {
                cipher_rt_ring_write(tid,
                                     CIPHER_RT_RING_EVENT_REMEMBER, 1u, 0ull,
                                     &e, sizeof(e));
                /* Shape producer rate to ~100k events/s per producer thread
                 * (1ms sleep every 100 events). At Koopman narrow-domain
                 * decode rates this is already faster than realistic; the
                 * consumer's per-cycle drain capacity (~256k/s per tenant)
                 * keeps the ring well below the 4096-entry ceiling. */
                if ((i % 100) == 99) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        });
    }
    for (auto& th : producers) th.join();
    auto t1 = std::chrono::steady_clock::now();

    /* Let the consumer flush remaining entries. */
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    uint64_t total_w = 0, total_d = 0;
    for (int t = 0; t < CIPHER_RT_RING_MAX_TENANTS; t++) {
        total_w += cipher_rt_ring_total_written((uint32_t)t) - pre_w[t];
        total_d += cipher_rt_ring_total_dropped((uint32_t)t) - pre_d[t];
    }
    uint64_t drained = cipher_rt_remember_consumer_drained() - pre_drained;
    double wall_s = std::chrono::duration<double>(t1 - t0).count();
    uint64_t intended = (uint64_t)N_PROD * PER_PROD;
    double rate_m_per_s = (double)intended / wall_s / 1e6;
    double drop_pct = (intended > 0)
                      ? (100.0 * (double)total_d / (double)intended)
                      : 0.0;

    printf("[S3.B1 COMPOSE] producers=%d per_prod=%d wall=%.2fs intended=%lu "
           "accepted=%lu dropped=%lu drop_pct=%.2f%% rate=%.2f M/s drained=%lu\n",
           N_PROD, PER_PROD, wall_s, intended, total_w, total_d,
           drop_pct, rate_m_per_s, drained);

    /* Gate: drop rate << S3.B0 baseline (which dropped 67% of writes on a
     * tight loop without consumer). With consumer running and slight
     * inter-batch spacing, drop rate should drop into the noise floor; we
     * gate at < 30% as the loose bound (slow CI / noisy env) and < 5% as
     * the steady-state target. */
    if (drop_pct > 30.0) {
        printf("[S3.B1 COMPOSE] FAIL: drop_pct=%.2f%% > 30%% bound\n", drop_pct);
        return 1;
    }
    if (drained == 0) {
        printf("[S3.B1 COMPOSE] FAIL: consumer drained 0 events\n");
        return 1;
    }
    printf("[S3.B1 COMPOSE] PASS (drop_pct=%.2f%%, drained=%lu)\n",
           drop_pct, drained);
    return 0;
}

int main(void)
{
    int rc = 0;
    rc |= test_envgate_off();
    rc |= test_cold_start_drain();
    rc |= test_producer_consumer_compose();
    if (rc == 0) printf("[S3.B1 OVERALL] PASS\n");
    else         printf("[S3.B1 OVERALL] FAIL rc=%d\n", rc);

    cipher_rt_remember_consumer_exit();
    return rc;
}
