// OP 15 — SPECULATE_CHECK kernel output cache.
#include "cipher_speculate_check.h"
#include "cipher_op_counters.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace {
constexpr unsigned MAX_SLOTS = 4096;   // power of two

struct Slot {
    std::atomic<uint64_t> key;          // params_hash; 0 = empty
    std::atomic<uint64_t> output;       // last seen output_hash
    std::atomic<uint32_t> seen;
};

alignas(64) Slot g_table[MAX_SLOTS];

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
std::atomic<uint32_t> g_entries{0};
std::atomic<uint64_t> g_records{0};
std::atomic<uint64_t> g_lookups{0};
std::atomic<uint64_t> g_hits{0};
std::atomic<uint64_t> g_misses{0};
std::atomic<uint64_t> g_mismatches{0};

inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

int probe_or_insert(uint64_t key) {
    uint64_t h = mix64(key);
    for (unsigned p = 0; p < MAX_SLOTS; ++p) {
        unsigned i = (unsigned)((h + p) & (MAX_SLOTS - 1));
        uint64_t cur = g_table[i].key.load(std::memory_order_acquire);
        if (cur == key) return (int)i;
        if (cur == 0) {
            uint64_t expected = 0;
            if (g_table[i].key.compare_exchange_strong(
                    expected, key, std::memory_order_acq_rel)) {
                g_table[i].output.store(0, std::memory_order_release);
                g_table[i].seen.store(0, std::memory_order_release);
                g_entries.fetch_add(1, std::memory_order_relaxed);
                return (int)i;
            }
            if (g_table[i].key.load(std::memory_order_acquire) == key) return (int)i;
        }
    }
    return -1;  // table full — silent drop
}
} // namespace

extern "C" int cipher_speculate_check_init(void) {
    if (g_initialized.exchange(1, std::memory_order_acq_rel))
        return g_enabled.load(std::memory_order_relaxed);
    const char* env = std::getenv("CIPHER_SPECULATE_CHECK");
    int on = env && env[0] && (env[0]=='1' || env[0]=='t' || env[0]=='T'
                            || env[0]=='o' || env[0]=='O');
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        std::fprintf(stderr,
            "[CIPHER OP15] SPECULATE_CHECK enabled — slots=%u\n", MAX_SLOTS);
    }
    return on;
}

extern "C" void cipher_speculate_check_record(uint64_t params_hash,
                                               uint64_t output_hash) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;
    if (params_hash == 0) return;
    int idx = probe_or_insert(params_hash);
    if (idx < 0) return;
    uint64_t prev = g_table[idx].output.load(std::memory_order_acquire);
    g_records.fetch_add(1, std::memory_order_relaxed);
    if (prev != 0 && prev != output_hash) {
        g_mismatches.fetch_add(1, std::memory_order_relaxed);
    }
    g_table[idx].output.store(output_hash, std::memory_order_release);
    g_table[idx].seen.fetch_add(1, std::memory_order_relaxed);
    cipher_op_inc(OP_SPECULATE_WRITE);
}

extern "C" uint64_t cipher_speculate_check_lookup(uint64_t params_hash) {
    if (!g_enabled.load(std::memory_order_relaxed)) return 0;
    if (params_hash == 0) return 0;
    g_lookups.fetch_add(1, std::memory_order_relaxed);
    cipher_op_inc(OP_SPECULATE_CHECK);
    uint64_t h = mix64(params_hash);
    for (unsigned p = 0; p < MAX_SLOTS; ++p) {
        unsigned i = (unsigned)((h + p) & (MAX_SLOTS - 1));
        uint64_t cur = g_table[i].key.load(std::memory_order_acquire);
        if (cur == 0) {
            g_misses.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        if (cur == params_hash) {
            uint64_t out = g_table[i].output.load(std::memory_order_acquire);
            if (out != 0) {
                g_hits.fetch_add(1, std::memory_order_relaxed);
            } else {
                g_misses.fetch_add(1, std::memory_order_relaxed);
            }
            return out;
        }
    }
    g_misses.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

extern "C" int cipher_speculate_check_stats(CipherSpeculateCheckStats* out) {
    if (!out) return 0;
    out->enabled    = g_enabled.load(std::memory_order_relaxed);
    out->entries    = g_entries.load(std::memory_order_relaxed);
    out->records    = g_records.load(std::memory_order_relaxed);
    out->lookups    = g_lookups.load(std::memory_order_relaxed);
    out->hits       = g_hits.load(std::memory_order_relaxed);
    out->misses     = g_misses.load(std::memory_order_relaxed);
    out->mismatches = g_mismatches.load(std::memory_order_relaxed);
    out->capacity   = MAX_SLOTS;
    return 1;
}

__attribute__((constructor(116)))
static void cipher_speculate_check_autoinit() {
    cipher_speculate_check_init();
}
