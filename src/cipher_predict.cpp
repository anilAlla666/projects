// Op 17 PREDICT — v1 observer.
//
// Per-shape dispatch counter + short-gap reuse counter. At report time,
// emits hot shapes (count >= HOT_COUNT AND short_gap/count >= 0.5) as
// preload candidates. No CUDA, no actuation. Conforms to OP_CONTRACT.md
// I1–I6.
//
// Eviction policy on full table: replace the entry with the lowest count
// (ties broken by oldest last_ts_ns). Fixed-size array, no allocation.

#include "cipher_predict.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>

namespace {

constexpr unsigned MAX_SHAPES      = 4096;   // must be power of two
constexpr unsigned REPORT_TOP_N    = 32;
constexpr uint32_t HOT_COUNT       = 200;
constexpr uint64_t HOT_GAP_NS      = 10ULL * 1000 * 1000;  // 10 ms
constexpr float    HOT_RATIO       = 0.5f;

struct ShapeEntry {
    std::atomic<uint64_t> key;         // 0 = empty
    uint32_t count;
    uint32_t short_gap_count;
    uint64_t first_ts_ns;
    uint64_t last_ts_ns;
};

alignas(64) ShapeEntry g_table[MAX_SHAPES];

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
std::atomic<unsigned> g_shape_count{0};
std::atomic<unsigned> g_eviction_count{0};
std::atomic<unsigned> g_candidate_count{0};

uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

uint64_t shape_key(const CipherRingEntry* ev) {
    return ev->params_hash ? ev->params_hash : ev->func_ptr_hash;
}

// Open-address linear probe; returns index of matching or freshly-claimed slot,
// or -1 if table full (caller performs eviction).
int probe_or_insert(uint64_t key, uint64_t t) {
    uint64_t h = mix64(key);
    for (unsigned probe = 0; probe < MAX_SHAPES; ++probe) {
        unsigned i = (unsigned)((h + probe) & (MAX_SHAPES - 1));
        uint64_t cur = g_table[i].key.load(std::memory_order_acquire);
        if (cur == key) return (int)i;
        if (cur == 0) {
            uint64_t expected = 0;
            if (g_table[i].key.compare_exchange_strong(
                    expected, key, std::memory_order_acq_rel)) {
                g_table[i].count = 0;
                g_table[i].short_gap_count = 0;
                g_table[i].first_ts_ns = t;
                g_table[i].last_ts_ns = t;
                g_shape_count.fetch_add(1, std::memory_order_relaxed);
                return (int)i;
            }
            // Lost race; recheck this slot.
            if (g_table[i].key.load(std::memory_order_acquire) == key) return (int)i;
        }
    }
    return -1;
}

// Evict entry with min count (ties: oldest last_ts_ns). Reinitialise for key.
int evict_and_install(uint64_t key, uint64_t t) {
    unsigned victim = 0;
    uint32_t best_count = UINT32_MAX;
    uint64_t best_last  = UINT64_MAX;
    for (unsigned i = 0; i < MAX_SHAPES; ++i) {
        uint32_t c = g_table[i].count;
        uint64_t l = g_table[i].last_ts_ns;
        if (c < best_count || (c == best_count && l < best_last)) {
            best_count = c;
            best_last  = l;
            victim = i;
        }
    }
    g_table[victim].key.store(key, std::memory_order_release);
    g_table[victim].count = 0;
    g_table[victim].short_gap_count = 0;
    g_table[victim].first_ts_ns = t;
    g_table[victim].last_ts_ns = t;
    g_eviction_count.fetch_add(1, std::memory_order_relaxed);
    return (int)victim;
}

} // namespace

extern "C" int cipher_predict_init(void) {
    int already = g_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_enabled.load(std::memory_order_relaxed);
    const char* env = std::getenv("CIPHER_PREDICT");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        std::fprintf(stderr, "[CIPHER Op17] PREDICT enabled — slots=%u "
                             "hot_count=%u hot_gap_ns=%llu ratio=%.2f\n",
                     MAX_SHAPES, HOT_COUNT,
                     (unsigned long long)HOT_GAP_NS, HOT_RATIO);
    }
    return on;
}

extern "C" void cipher_predict_observe(const CipherRingEntry* ev) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;
    if (!ev) return;

    uint64_t key = shape_key(ev);
    if (key == 0) return;

    uint64_t t = ev->timestamp_ns;
    if (t == 0) t = now_ns();

    int idx = probe_or_insert(key, t);
    if (idx < 0) idx = evict_and_install(key, t);

    ShapeEntry& e = g_table[idx];
    if (e.count > 0 && t > e.last_ts_ns && (t - e.last_ts_ns) < HOT_GAP_NS) {
        e.short_gap_count++;
    }
    e.count++;
    e.last_ts_ns = t;
}

extern "C" unsigned cipher_predict_shape_count(void) {
    return g_shape_count.load(std::memory_order_relaxed);
}

extern "C" unsigned cipher_predict_candidate_count(void) {
    return g_candidate_count.load(std::memory_order_relaxed);
}

namespace {

struct Cand {
    uint64_t key;
    uint32_t count;
    uint32_t short_gap_count;
    uint64_t first_ts_ns;
    uint64_t last_ts_ns;
};

} // namespace

extern "C" void cipher_predict_report(void) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;

    // First pass: collect candidates.
    Cand cands[REPORT_TOP_N];
    unsigned n = 0;
    unsigned total_candidates = 0;
    for (unsigned i = 0; i < MAX_SHAPES; ++i) {
        uint64_t k = g_table[i].key.load(std::memory_order_acquire);
        if (k == 0) continue;
        uint32_t c = g_table[i].count;
        uint32_t sg = g_table[i].short_gap_count;
        if (c < HOT_COUNT) continue;
        float ratio = (float)sg / (float)c;
        if (ratio < HOT_RATIO) continue;
        total_candidates++;
        Cand cur = { k, c, sg, g_table[i].first_ts_ns, g_table[i].last_ts_ns };
        if (n < REPORT_TOP_N) {
            cands[n++] = cur;
        } else {
            // Replace min-count slot if this one is hotter.
            unsigned min_i = 0;
            for (unsigned j = 1; j < n; ++j)
                if (cands[j].count < cands[min_i].count) min_i = j;
            if (cur.count > cands[min_i].count) cands[min_i] = cur;
        }
    }
    // Insertion sort by count desc.
    for (unsigned i = 1; i < n; ++i) {
        Cand key = cands[i]; int j = (int)i - 1;
        while (j >= 0 && cands[j].count < key.count) { cands[j+1] = cands[j]; --j; }
        cands[j+1] = key;
    }
    g_candidate_count.store(total_candidates, std::memory_order_relaxed);

    FILE* fp = std::fopen("/tmp/cipher_predict_report.json", "w");
    if (!fp) return;
    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"shape_count\": %u,\n",
                 g_shape_count.load(std::memory_order_relaxed));
    std::fprintf(fp, "  \"eviction_count\": %u,\n",
                 g_eviction_count.load(std::memory_order_relaxed));
    std::fprintf(fp, "  \"candidate_count\": %u,\n", total_candidates);
    std::fprintf(fp, "  \"hot_count_threshold\": %u,\n", HOT_COUNT);
    std::fprintf(fp, "  \"hot_gap_ns\": %llu,\n", (unsigned long long)HOT_GAP_NS);
    std::fprintf(fp, "  \"hot_ratio_threshold\": %.2f,\n", HOT_RATIO);
    std::fprintf(fp, "  \"candidates\": [\n");
    for (unsigned i = 0; i < n; ++i) {
        float r = (float)cands[i].short_gap_count / (float)cands[i].count;
        std::fprintf(fp,
            "    {\"key\": \"%016llx\", \"count\": %u, \"short_gap_count\": %u,"
            " \"short_gap_ratio\": %.4f,"
            " \"first_ts_ns\": %llu, \"last_ts_ns\": %llu}%s\n",
            (unsigned long long)cands[i].key,
            cands[i].count, cands[i].short_gap_count, r,
            (unsigned long long)cands[i].first_ts_ns,
            (unsigned long long)cands[i].last_ts_ns,
            (i + 1 < n) ? "," : "");
    }
    std::fprintf(fp, "  ]\n}\n");
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op17] PREDICT report -> "
                         "/tmp/cipher_predict_report.json (shapes=%u cands=%u evict=%u)\n",
                 g_shape_count.load(std::memory_order_relaxed),
                 total_candidates,
                 g_eviction_count.load(std::memory_order_relaxed));
}
