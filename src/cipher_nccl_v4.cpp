// CIPHER NCCL v4 — implementation.

#include "cipher_nccl_v4.h"
#include "cipher_silicon.h"

#include <atomic>
#include <stdio.h>
#include <stdlib.h>

namespace {

constexpr uint64_t LARGE_REDUCTION_BYTES = 16ULL * 1024 * 1024;   // ≥16 MB → prefer NVLS
constexpr int      DEFAULT_GREEN_SMS    = 8;

enum {
    NCCL_ALGO_TREE = 1,
    NCCL_ALGO_RING = 2,
    NCCL_ALGO_NVLS = 5,
};

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
int                   g_green_sms = DEFAULT_GREEN_SMS;
std::atomic<uint64_t> g_tuner_calls{0};
std::atomic<uint64_t> g_nvls_chosen{0};
std::atomic<uint64_t> g_ring_chosen{0};
std::atomic<uint64_t> g_tree_chosen{0};
std::atomic<uint64_t> g_large_reductions{0};
std::atomic<uint64_t> g_green_ctx_launches{0};

bool env_truthy(const char* v) {
    if (!v) return false;
    return v[0] == '1' || v[0] == 't' || v[0] == 'T'
        || ((v[0] == 'o' || v[0] == 'O') && (v[1] == 'n' || v[1] == 'N'));
}

} // namespace

extern "C" int cipher_nccl_v4_init(void) {
    if (g_initialized.exchange(1, std::memory_order_acq_rel))
        return g_enabled.load(std::memory_order_relaxed);
    int on = env_truthy(getenv("CIPHER_NCCL_V4"));
    if (const char* s = getenv("CIPHER_NCCL_V4_GREEN_SMS")) {
        int v = atoi(s);
        if (v > 0 && v <= 64) g_green_sms = v;
    }
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        const CipherSiliconModel* sil = cipher_silicon_get();
        fprintf(stderr,
            "[CIPHER NCCLv4] init green_sms=%d (host_sms=%d) large_thresh=%llu\n",
            g_green_sms, sil ? sil->sm_count : 0,
            (unsigned long long)LARGE_REDUCTION_BYTES);
    }
    return on;
}

extern "C" int cipher_nccl_v4_enabled(void) {
    return g_enabled.load(std::memory_order_relaxed);
}

extern "C" int cipher_nccl_v4_decide(uint64_t bytes, uint32_t /*num_ranks*/, int* out_algo) {
    if (!g_enabled.load(std::memory_order_relaxed) || !out_algo) return 0;
    g_tuner_calls.fetch_add(1, std::memory_order_relaxed);
    if (bytes >= LARGE_REDUCTION_BYTES) {
        *out_algo = NCCL_ALGO_NVLS;
        g_nvls_chosen.fetch_add(1, std::memory_order_relaxed);
        g_large_reductions.fetch_add(1, std::memory_order_relaxed);
        return 1;
    }
    if (bytes >= 256 * 1024) {
        *out_algo = NCCL_ALGO_RING;
        g_ring_chosen.fetch_add(1, std::memory_order_relaxed);
        return 1;
    }
    *out_algo = NCCL_ALGO_TREE;
    g_tree_chosen.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

extern "C" int cipher_nccl_v4_stats(CipherNcclV4Stats* out) {
    if (!out) return 0;
    out->enabled            = g_enabled.load(std::memory_order_relaxed);
    out->green_ctx_sms      = g_green_sms;
    out->tuner_calls        = g_tuner_calls.load(std::memory_order_relaxed);
    out->nvls_chosen        = g_nvls_chosen.load(std::memory_order_relaxed);
    out->ring_chosen        = g_ring_chosen.load(std::memory_order_relaxed);
    out->tree_chosen        = g_tree_chosen.load(std::memory_order_relaxed);
    out->large_reductions   = g_large_reductions.load(std::memory_order_relaxed);
    out->green_ctx_launches = g_green_ctx_launches.load(std::memory_order_relaxed);
    return 1;
}

extern "C" void cipher_nccl_v4_report(void) {
    CipherNcclV4Stats s{};
    cipher_nccl_v4_stats(&s);
    FILE* f = fopen("/tmp/cipher_nccl_v4_report.json", "w");
    if (!f) return;
    fprintf(f,
        "{\"enabled\":%d,\"green_ctx_sms\":%d,\"tuner_calls\":%llu,"
        "\"nvls_chosen\":%llu,\"ring_chosen\":%llu,\"tree_chosen\":%llu,"
        "\"large_reductions\":%llu,\"green_ctx_launches\":%llu}\n",
        s.enabled, s.green_ctx_sms,
        (unsigned long long)s.tuner_calls,
        (unsigned long long)s.nvls_chosen,
        (unsigned long long)s.ring_chosen,
        (unsigned long long)s.tree_chosen,
        (unsigned long long)s.large_reductions,
        (unsigned long long)s.green_ctx_launches);
    fclose(f);
}

__attribute__((constructor(109)))
static void cipher_nccl_v4_autoinit() { cipher_nccl_v4_init(); }
