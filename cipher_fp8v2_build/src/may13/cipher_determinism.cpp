// Op 21 DETERMINISM — v1 dispatch-sequence fingerprint.
//
// Mixes the ordered params_hash stream into a 64-bit accumulator. No CUDA,
// no allocation. Reports JSON with hash + count.

#include "may13/cipher_determinism.h"

#include "cipher_rt_commit.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace {

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
constexpr uint64_t FNV_PRIME  = 0x100000001b3ULL;

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
std::atomic<uint64_t> g_hash{FNV_OFFSET};
std::atomic<uint64_t> g_count{0};

inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

} // namespace

extern "C" int cipher_determinism_init(void) {
    int already = g_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_enabled.load(std::memory_order_relaxed);
    const char* env = std::getenv("CIPHER_DETERMINISM");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_enabled.store(on, std::memory_order_release);
    g_hash.store(FNV_OFFSET, std::memory_order_release);
    g_count.store(0, std::memory_order_release);
    if (on) {
        std::fprintf(stderr, "[CIPHER Op21] DETERMINISM enabled — "
                             "FNV-64 + splitmix dispatch-sequence fingerprint\n");
    }
    return on;
}

extern "C" void cipher_determinism_observe(const CipherRingEntry* ev) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;
    if (!ev) return;
    uint64_t ph = ev->params_hash;
    if (ph == 0) return;  // ASLR-stable only.
    uint64_t m = mix64(ph);
    uint64_t prev = g_hash.load(std::memory_order_relaxed);
    uint64_t next;
    do {
        next = (prev ^ m) * FNV_PRIME;
    } while (!g_hash.compare_exchange_weak(
        prev, next, std::memory_order_release, std::memory_order_relaxed));
    g_count.fetch_add(1, std::memory_order_relaxed);
}

extern "C" uint64_t cipher_determinism_hash(void) {
    return g_hash.load(std::memory_order_relaxed);
}

extern "C" uint64_t cipher_determinism_count(void) {
    return g_count.load(std::memory_order_relaxed);
}

extern "C" void cipher_determinism_report(void) {
    /* W7-9 Step 4 — coherent snapshot acquire (slow-path; emits the
     * COMMIT-published state alongside this report's existing aggregate). */
    struct cipher_rt_snapshot _snap;
    cipher_rt_snapshot_acquire(0u, &_snap);
    (void)_snap;

    if (!g_enabled.load(std::memory_order_relaxed)) return;
    uint64_t h = g_hash.load(std::memory_order_relaxed);
    uint64_t n = g_count.load(std::memory_order_relaxed);
    FILE* fp = std::fopen("/tmp/cipher_determinism_report.json", "w");
    if (!fp) return;
    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"dispatch_hash\": \"%016llx\",\n",
                 (unsigned long long)h);
    std::fprintf(fp, "  \"dispatch_count\": %llu\n", (unsigned long long)n);
    std::fprintf(fp, "}\n");
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op21] DETERMINISM report -> "
                         "/tmp/cipher_determinism_report.json (hash=%016llx count=%llu)\n",
                 (unsigned long long)h, (unsigned long long)n);
}
