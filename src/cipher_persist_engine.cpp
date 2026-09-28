// CIPHER Persistence Engine — implementation.

#include "cipher_persist_engine.h"
#include "cipher_silicon.h"

#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {

struct Region {
    void*  ptr;
    size_t bytes;
    double freq_score;
    double admit_fraction;   // 0.0..1.0; 0 = not admitted
    int    in_use;
};

struct Engine {
    std::mutex mu;                                  // guards table + budget
    Region     table[CIPHER_PERSIST_MAX_REGIONS]{};
    size_t     budget_bytes = 0;                    // copy of silicon->l2_persist_max

    std::atomic<int>      enabled{0};
    std::atomic<int>      initialized{0};
    std::atomic<int>      registered_count{0};
    std::atomic<int>      admitted_count{0};
    std::atomic<size_t>   admitted_bytes{0};

    std::atomic<uint64_t> register_calls{0};
    std::atomic<uint64_t> unregister_calls{0};
    std::atomic<uint64_t> window_lookups{0};
    std::atomic<uint64_t> window_hits{0};
    std::atomic<uint64_t> recompute_calls{0};
    std::atomic<uint64_t> apply_to_stream_calls{0};
    std::atomic<uint64_t> l2_resets{0};
};

Engine g_engine;

bool env_truthy(const char* v) {
    if (!v) return false;
    return (v[0] == '1') || (v[0] == 'o' && v[1] == 'n' && v[2] == '\0')
        || (v[0] == 'O' && v[1] == 'N' && v[2] == '\0')
        || (v[0] == 't' || v[0] == 'T');
}

// Slot lookup; returns index in table[], or -1.
int find_slot_locked(void* ptr) {
    for (int i = 0; i < CIPHER_PERSIST_MAX_REGIONS; ++i) {
        if (g_engine.table[i].in_use && g_engine.table[i].ptr == ptr) return i;
    }
    return -1;
}

int find_free_slot_locked() {
    for (int i = 0; i < CIPHER_PERSIST_MAX_REGIONS; ++i) {
        if (!g_engine.table[i].in_use) return i;
    }
    return -1;
}

// Fractional knapsack: maximize sum(freq_score * admit_fraction * bytes/bytes)
// subject to sum(admit_fraction * bytes) <= budget_bytes.
// Optimal greedy by density = freq_score / bytes.
void recompute_locked() {
    int idx[CIPHER_PERSIST_MAX_REGIONS];
    int n = 0;
    for (int i = 0; i < CIPHER_PERSIST_MAX_REGIONS; ++i) {
        if (g_engine.table[i].in_use && g_engine.table[i].bytes > 0) {
            idx[n++] = i;
            g_engine.table[i].admit_fraction = 0.0;   // reset
        }
    }
    auto density = [](const Region& r) {
        return (r.bytes > 0) ? (r.freq_score / static_cast<double>(r.bytes)) : 0.0;
    };
    std::sort(idx, idx + n, [&](int a, int b) {
        return density(g_engine.table[a]) > density(g_engine.table[b]);
    });

    size_t remaining = g_engine.budget_bytes;
    int    admitted_n = 0;
    size_t admitted_b = 0;
    for (int k = 0; k < n; ++k) {
        Region& r = g_engine.table[idx[k]];
        if (remaining == 0) break;
        if (r.bytes <= remaining) {
            r.admit_fraction = 1.0;
            remaining   -= r.bytes;
            admitted_b  += r.bytes;
            ++admitted_n;
        } else {
            // partial admit on the last region
            r.admit_fraction = static_cast<double>(remaining) / static_cast<double>(r.bytes);
            admitted_b  += remaining;
            remaining    = 0;
            ++admitted_n;
        }
    }
    g_engine.admitted_count.store(admitted_n, std::memory_order_relaxed);
    g_engine.admitted_bytes.store(admitted_b, std::memory_order_relaxed);
    g_engine.recompute_calls.fetch_add(1, std::memory_order_relaxed);
}

} // namespace

extern "C" int cipher_persist_engine_init(void) {
    if (g_engine.initialized.load(std::memory_order_acquire)) return 1;

    const char* env = getenv("CIPHER_PERSIST_ENGINE");
    if (!env_truthy(env)) {
        // Disabled — but mark initialized so subsequent calls are stable no-ops.
        g_engine.initialized.store(1, std::memory_order_release);
        return 0;
    }

    const CipherSiliconModel* sil = cipher_silicon_get();
    if (!sil) {
        // Silicon not ready; cannot derive budget. Stay disabled.
        return 0;
    }

    g_engine.budget_bytes = sil->l2_persist_max;
    g_engine.enabled.store(1, std::memory_order_release);
    g_engine.initialized.store(1, std::memory_order_release);

    if (getenv("CIPHER_PERSIST_ENGINE_VERBOSE")) {
        fprintf(stderr,
            "[CIPHER PERSIST] engine initialized: budget=%zu bytes (%.1f MB), "
            "max_regions=%d\n",
            g_engine.budget_bytes,
            g_engine.budget_bytes / (1024.0 * 1024.0),
            CIPHER_PERSIST_MAX_REGIONS);
    }
    return 1;
}

extern "C" int cipher_persist_engine_enabled(void) {
    return g_engine.enabled.load(std::memory_order_relaxed);
}

extern "C" int cipher_persist_engine_register(void* ptr, size_t bytes, double freq_score) {
    if (!g_engine.enabled.load(std::memory_order_relaxed)) return 0;
    if (!ptr || bytes == 0) return 0;
    g_engine.register_calls.fetch_add(1, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lk(g_engine.mu);
    int i = find_slot_locked(ptr);
    if (i < 0) {
        i = find_free_slot_locked();
        if (i < 0) return 0;                // table full
        g_engine.table[i].in_use = 1;
        g_engine.table[i].ptr    = ptr;
        g_engine.registered_count.fetch_add(1, std::memory_order_relaxed);
    }
    g_engine.table[i].bytes      = bytes;
    g_engine.table[i].freq_score = freq_score;
    recompute_locked();
    return 1;
}

extern "C" int cipher_persist_engine_unregister(void* ptr) {
    if (!g_engine.enabled.load(std::memory_order_relaxed)) return 0;
    if (!ptr) return 0;
    g_engine.unregister_calls.fetch_add(1, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lk(g_engine.mu);
    int i = find_slot_locked(ptr);
    if (i < 0) return 0;
    g_engine.table[i] = Region{};
    g_engine.registered_count.fetch_sub(1, std::memory_order_relaxed);
    recompute_locked();
    return 1;
}

extern "C" void cipher_persist_engine_recompute_budget(void) {
    if (!g_engine.enabled.load(std::memory_order_relaxed)) return;
    std::lock_guard<std::mutex> lk(g_engine.mu);
    recompute_locked();
}

extern "C" int cipher_persist_engine_get_window(void* ptr, cudaAccessPolicyWindow* out) {
    if (!out) return 0;
    if (!g_engine.enabled.load(std::memory_order_relaxed)) return 0;
    g_engine.window_lookups.fetch_add(1, std::memory_order_relaxed);

    // Lock-free scan: acceptable race with mutator; benign result is "miss".
    for (int i = 0; i < CIPHER_PERSIST_MAX_REGIONS; ++i) {
        const Region& r = g_engine.table[i];
        if (!r.in_use || r.ptr != ptr || r.admit_fraction <= 0.0) continue;
        memset(out, 0, sizeof(*out));
        out->base_ptr   = r.ptr;
        out->num_bytes  = r.bytes;
        out->hitRatio   = static_cast<float>(r.admit_fraction);
        out->hitProp    = cudaAccessPropertyPersisting;
        out->missProp   = cudaAccessPropertyStreaming;
        g_engine.window_hits.fetch_add(1, std::memory_order_relaxed);
        return 1;
    }
    return 0;
}

extern "C" int cipher_persist_engine_get_top_window(cudaAccessPolicyWindow* out) {
    if (!out || !g_engine.enabled.load(std::memory_order_relaxed)) return 0;
    g_engine.window_lookups.fetch_add(1, std::memory_order_relaxed);

    // Lock-free scan; pick highest score×admit. Race with mutator is benign.
    double best = 0.0;
    void*  best_ptr = nullptr;
    size_t best_bytes = 0;
    double best_admit = 0.0;
    for (int i = 0; i < CIPHER_PERSIST_MAX_REGIONS; ++i) {
        const Region& r = g_engine.table[i];
        if (!r.in_use || r.bytes == 0 || r.admit_fraction <= 0.0) continue;
        double rank = r.freq_score * r.admit_fraction;
        if (rank > best) {
            best = rank;
            best_ptr = r.ptr;
            best_bytes = r.bytes;
            best_admit = r.admit_fraction;
        }
    }
    if (!best_ptr) return 0;
    memset(out, 0, sizeof(*out));
    out->base_ptr  = best_ptr;
    out->num_bytes = best_bytes;
    out->hitRatio  = static_cast<float>(best_admit);
    out->hitProp   = cudaAccessPropertyPersisting;
    out->missProp  = cudaAccessPropertyStreaming;
    g_engine.window_hits.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

extern "C" int cipher_persist_engine_get_top_window_raw(void* out_32) {
    if (!out_32) return 0;
    cudaAccessPolicyWindow win{};
    if (!cipher_persist_engine_get_top_window(&win)) return 0;
    memcpy(out_32, &win, sizeof(win));
    return 1;
}

extern "C" int cipher_persist_engine_apply_to_stream(void* stream_handle) {
    if (!g_engine.enabled.load(std::memory_order_relaxed)) return 0;
    if (!stream_handle) return 0;

    // Pick the top admitted region (highest density × admit_fraction).
    cudaAccessPolicyWindow win{};
    bool have_win = false;
    {
        std::lock_guard<std::mutex> lk(g_engine.mu);
        double best = 0.0;
        const Region* pick = nullptr;
        for (int i = 0; i < CIPHER_PERSIST_MAX_REGIONS; ++i) {
            const Region& r = g_engine.table[i];
            if (!r.in_use || r.bytes == 0 || r.admit_fraction <= 0.0) continue;
            double rank = r.freq_score * r.admit_fraction;
            if (rank > best) { best = rank; pick = &r; }
        }
        if (pick) {
            win.base_ptr   = pick->ptr;
            win.num_bytes  = pick->bytes;
            win.hitRatio   = static_cast<float>(pick->admit_fraction);
            win.hitProp    = cudaAccessPropertyPersisting;
            win.missProp   = cudaAccessPropertyStreaming;
            have_win = true;
        }
    }
    if (!have_win) return 0;

    cudaStreamAttrValue val{};
    val.accessPolicyWindow = win;
    cudaError_t rc = cudaStreamSetAttribute(
        static_cast<cudaStream_t>(stream_handle),
        cudaStreamAttributeAccessPolicyWindow,
        &val);
    if (rc != cudaSuccess) return 0;

    g_engine.apply_to_stream_calls.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

extern "C" void cipher_persist_engine_reset_l2(void) {
    if (!g_engine.enabled.load(std::memory_order_relaxed)) return;
    if (cudaCtxResetPersistingL2Cache() == cudaSuccess) {
        g_engine.l2_resets.fetch_add(1, std::memory_order_relaxed);
    }
}

extern "C" int cipher_persist_engine_stats(CipherPersistEngineStats* out) {
    if (!out) return 0;
    out->registered_count    = g_engine.registered_count.load(std::memory_order_relaxed);
    out->admitted_count      = g_engine.admitted_count.load(std::memory_order_relaxed);
    out->admitted_bytes      = g_engine.admitted_bytes.load(std::memory_order_relaxed);
    out->budget_bytes        = g_engine.budget_bytes;
    out->register_calls      = g_engine.register_calls.load(std::memory_order_relaxed);
    out->unregister_calls    = g_engine.unregister_calls.load(std::memory_order_relaxed);
    out->window_lookups      = g_engine.window_lookups.load(std::memory_order_relaxed);
    out->window_hits         = g_engine.window_hits.load(std::memory_order_relaxed);
    out->recompute_calls     = g_engine.recompute_calls.load(std::memory_order_relaxed);
    out->apply_to_stream_calls = g_engine.apply_to_stream_calls.load(std::memory_order_relaxed);
    out->l2_resets           = g_engine.l2_resets.load(std::memory_order_relaxed);
    return 1;
}

extern "C" void cipher_persist_engine_report(void) {
    CipherPersistEngineStats s{};
    cipher_persist_engine_stats(&s);

    FILE* f = fopen("/tmp/cipher_persist_engine_report.json", "w");
    if (!f) return;
    fprintf(f,
        "{\n"
        "  \"enabled\": %d,\n"
        "  \"budget_bytes\": %zu,\n"
        "  \"registered_count\": %d,\n"
        "  \"admitted_count\": %d,\n"
        "  \"admitted_bytes\": %zu,\n"
        "  \"register_calls\": %llu,\n"
        "  \"unregister_calls\": %llu,\n"
        "  \"window_lookups\": %llu,\n"
        "  \"window_hits\": %llu,\n"
        "  \"recompute_calls\": %llu,\n"
        "  \"apply_to_stream_calls\": %llu,\n"
        "  \"l2_resets\": %llu,\n"
        "  \"regions\": [\n",
        cipher_persist_engine_enabled(),
        s.budget_bytes,
        s.registered_count, s.admitted_count, s.admitted_bytes,
        (unsigned long long)s.register_calls,
        (unsigned long long)s.unregister_calls,
        (unsigned long long)s.window_lookups,
        (unsigned long long)s.window_hits,
        (unsigned long long)s.recompute_calls,
        (unsigned long long)s.apply_to_stream_calls,
        (unsigned long long)s.l2_resets);

    {
        std::lock_guard<std::mutex> lk(g_engine.mu);
        bool first = true;
        for (int i = 0; i < CIPHER_PERSIST_MAX_REGIONS; ++i) {
            const Region& r = g_engine.table[i];
            if (!r.in_use) continue;
            fprintf(f, "%s    {\"ptr\":\"%p\",\"bytes\":%zu,\"freq_score\":%.6g,\"admit_fraction\":%.6g}",
                    first ? "" : ",\n", r.ptr, r.bytes, r.freq_score, r.admit_fraction);
            first = false;
        }
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
}

// Self-init at priority 103 — after silicon (102), so silicon model is ready.
__attribute__((constructor(103)))
static void cipher_persist_engine_autoinit() {
    cipher_persist_engine_init();
}
