// Op 16 GUARD — v1 observer.
//
// Detects cross-session reuse of the same params_hash within
// RESIDENCY_NS. For each entry we record (last_session_fp, last_ts_ns). A
// later observe with a *different* current session and delta < RESIDENCY_NS
// is a leak candidate.
//
// Fixed-size array, open-address linear probe, no allocation on hot path,
// no CUDA. Contract I1–I6.

#include "cipher_guard.h"
#include "cipher_sense.h"
#include "cipher_op_counters.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>

namespace {

constexpr unsigned MAX_SHAPES   = 4096;                   // power of two
constexpr unsigned REPORT_TOP_N = 32;
constexpr uint64_t RESIDENCY_NS = 2ULL * 1000 * 1000 * 1000; // 2 s

struct Entry {
    std::atomic<uint64_t> key;          // params_hash; 0 = empty
    uint64_t session_fp;
    uint64_t last_ts_ns;
    uint32_t seen_count;
    uint32_t leak_count;
};

alignas(64) Entry g_table[MAX_SHAPES];

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
std::atomic<unsigned> g_shape_count{0};
std::atomic<unsigned> g_leak_events{0};
std::atomic<uint64_t> g_last_session_seen{0};
std::atomic<unsigned> g_session_count{0};

inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

int probe(uint64_t key) {
    uint64_t h = mix64(key);
    for (unsigned p = 0; p < MAX_SHAPES; ++p) {
        unsigned i = (unsigned)((h + p) & (MAX_SHAPES - 1));
        uint64_t cur = g_table[i].key.load(std::memory_order_acquire);
        if (cur == key) return (int)i;
        if (cur == 0) {
            uint64_t expected = 0;
            if (g_table[i].key.compare_exchange_strong(
                    expected, key, std::memory_order_acq_rel)) {
                g_table[i].session_fp = 0;
                g_table[i].last_ts_ns = 0;
                g_table[i].seen_count = 0;
                g_table[i].leak_count = 0;
                g_shape_count.fetch_add(1, std::memory_order_relaxed);
                return (int)i;
            }
            if (g_table[i].key.load(std::memory_order_acquire) == key) return (int)i;
        }
    }
    // Table full — fall back to fixed overflow slot 0 (rare in v1).
    return 0;
}

} // namespace

// ── OP 16 — cudaMalloc/Free byte tracking + cap enforcement ────────────────
namespace {

// 16K-slot open-address ptr→size table. Linear probe; entries cleared
// on free. Sized to comfortably hold PyTorch caching-allocator workloads
// where individual cudaMalloc count is dominated by ~32 large pool blocks.
constexpr unsigned MEM_SLOTS = 16384;
struct MemEntry {
    std::atomic<uintptr_t> ptr;   // 0 = empty
    uint64_t               size;
};
alignas(64) MemEntry g_mem[MEM_SLOTS];

std::atomic<int>      g_mem_enforce{0};
std::atomic<uint64_t> g_mem_cap{0};
std::atomic<uint64_t> g_mem_allocs{0};
std::atomic<uint64_t> g_mem_frees{0};
std::atomic<uint64_t> g_mem_in_flight{0};
std::atomic<uint64_t> g_mem_peak{0};
std::atomic<uint64_t> g_mem_blocked{0};
std::atomic<uint64_t> g_mem_unmatched{0};

inline unsigned ptr_slot(uintptr_t p) {
    uint64_t h = mix64((uint64_t)p);
    return (unsigned)(h & (MEM_SLOTS - 1));
}

// CAS-insert (ptr,size). Returns true on success; false if table full.
bool mem_insert(uintptr_t p, uint64_t size) {
    unsigned start = ptr_slot(p);
    for (unsigned probe = 0; probe < MEM_SLOTS; ++probe) {
        unsigned i = (start + probe) & (MEM_SLOTS - 1);
        uintptr_t cur = g_mem[i].ptr.load(std::memory_order_acquire);
        if (cur == 0) {
            uintptr_t expected = 0;
            if (g_mem[i].ptr.compare_exchange_strong(
                    expected, p, std::memory_order_acq_rel)) {
                g_mem[i].size = size;
                return true;
            }
        }
    }
    return false;
}

// Lookup-and-erase. Returns size on hit, 0 on miss.
uint64_t mem_pop(uintptr_t p) {
    unsigned start = ptr_slot(p);
    for (unsigned probe = 0; probe < MEM_SLOTS; ++probe) {
        unsigned i = (start + probe) & (MEM_SLOTS - 1);
        uintptr_t cur = g_mem[i].ptr.load(std::memory_order_acquire);
        if (cur == 0) return 0;
        if (cur == p) {
            uint64_t sz = g_mem[i].size;
            g_mem[i].ptr.store(0, std::memory_order_release);
            return sz;
        }
    }
    return 0;
}

void mem_init_from_env() {
    const char* env = std::getenv("CIPHER_GUARD_BYTES_CAP");
    if (!env || !*env) return;
    char* end = nullptr;
    uint64_t v = std::strtoull(env, &end, 10);
    if (v == 0) return;
    // Allow K/M/G suffix.
    if (end && *end) {
        switch (*end) {
            case 'K': case 'k': v <<= 10; break;
            case 'M': case 'm': v <<= 20; break;
            case 'G': case 'g': v <<= 30; break;
            default: break;
        }
    }
    g_mem_cap.store(v, std::memory_order_release);
    g_mem_enforce.store(1, std::memory_order_release);
    std::fprintf(stderr, "[CIPHER Op16] GUARD bytes-cap enforce — cap=%llu bytes (~%.2f GiB)\n",
                 (unsigned long long)v, (double)v / (1024.0 * 1024.0 * 1024.0));
}

} // namespace

extern "C" int cipher_comply_audit(const char* kind, const char* json_kv);

extern "C" int cipher_guard_alloc_admit(uint64_t size) {
    if (!g_enabled.load(std::memory_order_relaxed)) return 1;
    if (!g_mem_enforce.load(std::memory_order_relaxed)) return 1;
    uint64_t cur = g_mem_in_flight.load(std::memory_order_relaxed);
    uint64_t cap = g_mem_cap.load(std::memory_order_relaxed);
    if (cur + size > cap) {
        g_mem_blocked.fetch_add(1, std::memory_order_relaxed);
        char kv[160];
        std::snprintf(kv, sizeof(kv),
            "{\"bytes\":%llu,\"cap\":%llu,\"in_flight\":%llu}",
            (unsigned long long)size, (unsigned long long)cap,
            (unsigned long long)cur);
        cipher_comply_audit("alloc_blocked", kv);
        return 0;
    }
    return 1;
}

extern "C" void cipher_guard_alloc_record(void* ptr, uint64_t size) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;
    if (!ptr || size == 0) return;
    g_mem_allocs.fetch_add(1, std::memory_order_relaxed);
    uint64_t after = g_mem_in_flight.fetch_add(size, std::memory_order_relaxed) + size;
    uint64_t peak  = g_mem_peak.load(std::memory_order_relaxed);
    while (after > peak) {
        if (g_mem_peak.compare_exchange_weak(peak, after,
                std::memory_order_relaxed, std::memory_order_relaxed))
            break;
    }
    if (!mem_insert((uintptr_t)ptr, size)) {
        // Table full; we'll see a phantom-leak when its free arrives.
        g_mem_unmatched.fetch_add(1, std::memory_order_relaxed);
    }
}

extern "C" void cipher_guard_free_record(void* ptr) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;
    if (!ptr) return;
    uint64_t sz = mem_pop((uintptr_t)ptr);
    g_mem_frees.fetch_add(1, std::memory_order_relaxed);
    if (sz == 0) {
        g_mem_unmatched.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    g_mem_in_flight.fetch_sub(sz, std::memory_order_relaxed);
}

extern "C" int cipher_guard_memory_stats(CipherGuardMemoryStats* out) {
    if (!out) return 0;
    out->enabled         = g_enabled.load(std::memory_order_relaxed);
    out->enforce         = g_mem_enforce.load(std::memory_order_relaxed);
    out->bytes_cap       = g_mem_cap.load(std::memory_order_relaxed);
    out->allocs_total    = g_mem_allocs.load(std::memory_order_relaxed);
    out->frees_total     = g_mem_frees.load(std::memory_order_relaxed);
    out->bytes_in_flight = g_mem_in_flight.load(std::memory_order_relaxed);
    out->peak_bytes      = g_mem_peak.load(std::memory_order_relaxed);
    out->allocs_blocked  = g_mem_blocked.load(std::memory_order_relaxed);
    out->frees_unmatched = g_mem_unmatched.load(std::memory_order_relaxed);
    return 1;
}

extern "C" int cipher_guard_init(void) {
    int already = g_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_enabled.load(std::memory_order_relaxed);
    const char* env = std::getenv("CIPHER_GUARD");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        std::fprintf(stderr, "[CIPHER Op16] GUARD enabled — slots=%u "
                             "residency_ns=%llu\n",
                     MAX_SHAPES, (unsigned long long)RESIDENCY_NS);
        mem_init_from_env();
    }
    return on;
}

__attribute__((constructor(112)))
static void cipher_guard_autoinit() { cipher_guard_init(); }

extern "C" void cipher_guard_observe(const CipherRingEntry* ev) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;
    if (!ev) return;

    uint64_t key = ev->params_hash ? ev->params_hash : ev->func_ptr_hash;
    if (key == 0) return;

    uint64_t sess = cipher_sense_current_session();
    if (sess == 0) return;  // require SENSE to attribute a session.

    uint64_t t = ev->timestamp_ns ? ev->timestamp_ns : now_ns();

    // Track distinct session fps.
    uint64_t prev_seen = g_last_session_seen.load(std::memory_order_relaxed);
    if (prev_seen != sess) {
        g_last_session_seen.store(sess, std::memory_order_relaxed);
        g_session_count.fetch_add(1, std::memory_order_relaxed);
        cipher_op_inc(OP_GUARD);
    }

    int idx = probe(key);
    Entry& e = g_table[idx];
    e.seen_count++;
    uint64_t prev_sess = e.session_fp;
    uint64_t prev_ts   = e.last_ts_ns;
    if (prev_sess != 0 && prev_sess != sess
        && t > prev_ts && (t - prev_ts) < RESIDENCY_NS) {
        e.leak_count++;
        g_leak_events.fetch_add(1, std::memory_order_relaxed);
    }
    e.session_fp = sess;
    e.last_ts_ns = t;
}

extern "C" unsigned cipher_guard_leak_count(void) {
    return g_leak_events.load(std::memory_order_relaxed);
}

extern "C" unsigned cipher_guard_session_count(void) {
    return g_session_count.load(std::memory_order_relaxed);
}

namespace {

struct Row {
    uint64_t key;
    uint32_t leak;
    uint32_t seen;
    uint64_t last_ts;
};

} // namespace

extern "C" void cipher_guard_report(void) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;

    Row top[REPORT_TOP_N];
    unsigned n = 0;
    unsigned leaked_shapes = 0;

    for (unsigned i = 0; i < MAX_SHAPES; ++i) {
        uint64_t k = g_table[i].key.load(std::memory_order_acquire);
        if (k == 0) continue;
        uint32_t lk = g_table[i].leak_count;
        if (lk == 0) continue;
        leaked_shapes++;
        Row r = { k, lk, g_table[i].seen_count, g_table[i].last_ts_ns };
        if (n < REPORT_TOP_N) top[n++] = r;
        else {
            unsigned min_i = 0;
            for (unsigned j = 1; j < n; ++j)
                if (top[j].leak < top[min_i].leak) min_i = j;
            if (r.leak > top[min_i].leak) top[min_i] = r;
        }
    }
    for (unsigned i = 1; i < n; ++i) {
        Row key = top[i]; int j = (int)i - 1;
        while (j >= 0 && top[j].leak < key.leak) { top[j+1] = top[j]; --j; }
        top[j+1] = key;
    }

    FILE* fp = std::fopen("/tmp/cipher_guard_report.json", "w");
    if (!fp) return;
    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"shape_count\": %u,\n",
                 g_shape_count.load(std::memory_order_relaxed));
    std::fprintf(fp, "  \"session_count\": %u,\n",
                 g_session_count.load(std::memory_order_relaxed));
    std::fprintf(fp, "  \"leak_count\": %u,\n",
                 g_leak_events.load(std::memory_order_relaxed));
    std::fprintf(fp, "  \"leaked_shapes\": %u,\n", leaked_shapes);
    std::fprintf(fp, "  \"residency_ns\": %llu,\n",
                 (unsigned long long)RESIDENCY_NS);
    std::fprintf(fp, "  \"leaks\": [\n");
    for (unsigned i = 0; i < n; ++i) {
        std::fprintf(fp,
            "    {\"key\": \"%016llx\", \"leak_count\": %u, \"seen_count\": %u,"
            " \"last_ts_ns\": %llu}%s\n",
            (unsigned long long)top[i].key, top[i].leak, top[i].seen,
            (unsigned long long)top[i].last_ts,
            (i + 1 < n) ? "," : "");
    }
    std::fprintf(fp, "  ]\n}\n");
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op16] GUARD report -> "
                         "/tmp/cipher_guard_report.json (shapes=%u sess=%u leaks=%u)\n",
                 g_shape_count.load(std::memory_order_relaxed),
                 g_session_count.load(std::memory_order_relaxed),
                 g_leak_events.load(std::memory_order_relaxed));
}
