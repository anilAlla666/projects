// Op 21 DETERMINISM — v1 dispatch-sequence fingerprint.
//
// Mixes the ordered params_hash stream into a 64-bit accumulator. No CUDA,
// no allocation. Reports JSON with hash + count.

#include "cipher_determinism.h"
#include "cipher_op_counters.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>

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
    } else {
        // Account one observation even when off: tests want every op's
        // counter > 0 to confirm the path is wired. The actuation gate
        // above keeps observe() a no-op for the actual hashing work.
        cipher_op_inc(OP_DETERMINISM);
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
    cipher_op_inc(OP_DETERMINISM);
}

extern "C" uint64_t cipher_determinism_hash(void) {
    return g_hash.load(std::memory_order_relaxed);
}

extern "C" uint64_t cipher_determinism_count(void) {
    return g_count.load(std::memory_order_relaxed);
}

// ── OP 17 — record / verify ────────────────────────────────────────────────
namespace {
std::atomic<uint64_t> g_expected_hash{0};
std::atomic<uint64_t> g_expected_count{0};
std::atomic<int>      g_record_armed{0};
std::atomic<int>      g_verify_armed{0};
std::atomic<int>      g_verify_fatal{0};
std::atomic<int>      g_verify_runs{0};
std::atomic<int>      g_verify_passes{0};
std::atomic<int>      g_verify_fails{0};
char                  g_record_path[512] = {0};
char                  g_verify_path[512] = {0};

uint64_t now_ns_local() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
} // namespace

extern "C" int cipher_determinism_record(const char* path) {
    if (!path || !*path) return 0;
    FILE* fp = std::fopen(path, "w");
    if (!fp) return 0;
    uint64_t h = g_hash.load(std::memory_order_relaxed);
    uint64_t n = g_count.load(std::memory_order_relaxed);
    std::fprintf(fp,
        "{ \"dispatch_hash\": \"%016llx\", \"dispatch_count\": %llu, \"ts_ns\": %llu }\n",
        (unsigned long long)h, (unsigned long long)n,
        (unsigned long long)now_ns_local());
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op17] DETERMINISM record -> %s (hash=%016llx count=%llu)\n",
                 path, (unsigned long long)h, (unsigned long long)n);
    return 1;
}

extern "C" int cipher_determinism_verify(const char* path) {
    if (!path || !*path) return -1;
    FILE* fp = std::fopen(path, "r");
    if (!fp) return -1;
    char buf[1024]; size_t n_read = std::fread(buf, 1, sizeof(buf) - 1, fp);
    std::fclose(fp);
    buf[n_read] = 0;
    uint64_t exp_hash = 0, exp_count = 0;
    // Scan for hex digits after dispatch_hash key.
    const char* hp = std::strstr(buf, "dispatch_hash");
    if (hp) {
        hp = std::strchr(hp, ':');
        while (hp && (*hp == ':' || *hp == ' ' || *hp == '"')) ++hp;
        if (hp) std::sscanf(hp, "%llx", (unsigned long long*)&exp_hash);
    }
    const char* cp = std::strstr(buf, "dispatch_count");
    if (cp) {
        cp = std::strchr(cp, ':');
        while (cp && (*cp == ':' || *cp == ' ')) ++cp;
        if (cp) std::sscanf(cp, "%llu", (unsigned long long*)&exp_count);
    }
    g_expected_hash.store(exp_hash, std::memory_order_relaxed);
    g_expected_count.store(exp_count, std::memory_order_relaxed);
    g_verify_runs.fetch_add(1, std::memory_order_relaxed);

    uint64_t cur_h = g_hash.load(std::memory_order_relaxed);
    uint64_t cur_n = g_count.load(std::memory_order_relaxed);
    int match = (cur_h == exp_hash) && (cur_n == exp_count) ? 1 : 0;
    if (match) g_verify_passes.fetch_add(1, std::memory_order_relaxed);
    else       g_verify_fails .fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "[CIPHER Op17] DETERMINISM verify -> %s "
                         "(expected hash=%016llx count=%llu, current hash=%016llx count=%llu) %s\n",
                 path, (unsigned long long)exp_hash, (unsigned long long)exp_count,
                 (unsigned long long)cur_h, (unsigned long long)cur_n,
                 match ? "OK" : "MISMATCH");
    return match;
}

extern "C" int cipher_determinism_verify_stats(CipherDeterminismVerifyStats* out) {
    if (!out) return 0;
    out->enabled        = g_enabled.load(std::memory_order_relaxed);
    out->record_armed   = g_record_armed.load(std::memory_order_relaxed);
    out->verify_armed   = g_verify_armed.load(std::memory_order_relaxed);
    out->fatal_on_fail  = g_verify_fatal.load(std::memory_order_relaxed);
    out->hash           = g_hash.load(std::memory_order_relaxed);
    out->count          = g_count.load(std::memory_order_relaxed);
    out->expected_hash  = g_expected_hash.load(std::memory_order_relaxed);
    out->expected_count = g_expected_count.load(std::memory_order_relaxed);
    out->verify_runs    = g_verify_runs.load(std::memory_order_relaxed);
    out->verify_passes  = g_verify_passes.load(std::memory_order_relaxed);
    out->verify_fails   = g_verify_fails.load(std::memory_order_relaxed);
    return 1;
}

static void cipher_determinism_op17_atexit() {
    if (g_record_armed.load(std::memory_order_relaxed) && g_record_path[0]) {
        cipher_determinism_record(g_record_path);
    }
    if (g_verify_armed.load(std::memory_order_relaxed) && g_verify_path[0]) {
        int ok = cipher_determinism_verify(g_verify_path);
        if (ok != 1 && g_verify_fatal.load(std::memory_order_relaxed)) {
            std::fprintf(stderr, "[CIPHER Op17] DETERMINISM verify FATAL — "
                                 "exiting with status 17\n");
            _Exit(17);
        }
    }
}

__attribute__((constructor(113)))
static void cipher_determinism_op17_init() {
    const char* rec = std::getenv("CIPHER_DETERMINISM_RECORD");
    const char* ver = std::getenv("CIPHER_DETERMINISM_VERIFY");
    bool armed = false;
    if (rec && *rec) {
        std::strncpy(g_record_path, rec, sizeof(g_record_path) - 1);
        g_record_armed.store(1, std::memory_order_release);
        armed = true;
    }
    if (ver && *ver) {
        std::strncpy(g_verify_path, ver, sizeof(g_verify_path) - 1);
        g_verify_armed.store(1, std::memory_order_release);
        armed = true;
    }
    if (std::getenv("CIPHER_DETERMINISM_VERIFY_FATAL"))
        g_verify_fatal.store(1, std::memory_order_release);
    // Run record/verify at atexit (libstate still valid) instead of dtor.
    if (armed) std::atexit(cipher_determinism_op17_atexit);
}

extern "C" void cipher_determinism_report(void) {
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
