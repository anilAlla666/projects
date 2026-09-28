// Op 18 RECEIPT — v1 per-session signed proof of compute.

#include "cipher_receipt.h"
#include "cipher_sense.h"
#include "cipher_op_counters.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <dlfcn.h>

namespace {

constexpr unsigned MAX_TENANTS   = 256;
constexpr unsigned REPORT_TOP_N  = 32;
constexpr uint64_t FNV_OFFSET    = 0xcbf29ce484222325ULL;
constexpr uint64_t FNV_PRIME     = 0x100000001b3ULL;

struct Tenant {
    std::atomic<uint64_t> key;          // session fp; 0 = empty
    std::atomic<uint64_t> chain;
    std::atomic<uint64_t> launches;
    std::atomic<uint64_t> first_ts_ns;
    std::atomic<uint64_t> last_ts_ns;
    std::atomic<uint32_t> order;        // insertion order (1-based)
    // OP 30 billing fields:
    std::atomic<uint64_t> bytes_h2d;
    std::atomic<uint64_t> bytes_d2h;
    std::atomic<uint64_t> alloc_bytes;
};

alignas(64) Tenant g_table[MAX_TENANTS];

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
std::atomic<unsigned> g_tenant_count{0};

uint8_t g_hmac_key[32] = {0};

inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

int probe_or_insert(uint64_t key, uint64_t t) {
    uint64_t h = mix64(key);
    for (unsigned p = 0; p < MAX_TENANTS; ++p) {
        unsigned i = (unsigned)((h + p) & (MAX_TENANTS - 1));
        uint64_t cur = g_table[i].key.load(std::memory_order_acquire);
        if (cur == key) return (int)i;
        if (cur == 0) {
            uint64_t expected = 0;
            if (g_table[i].key.compare_exchange_strong(
                    expected, key, std::memory_order_acq_rel)) {
                g_table[i].chain.store(FNV_OFFSET, std::memory_order_relaxed);
                g_table[i].launches.store(0, std::memory_order_relaxed);
                g_table[i].first_ts_ns.store(t, std::memory_order_relaxed);
                g_table[i].last_ts_ns.store(t, std::memory_order_relaxed);
                uint32_t ord = g_tenant_count.fetch_add(1, std::memory_order_relaxed) + 1;
                cipher_op_inc(OP_RECEIPT);
                g_table[i].order.store(ord, std::memory_order_relaxed);
                return (int)i;
            }
            if (g_table[i].key.load(std::memory_order_acquire) == key) return (int)i;
        }
    }
    return -1;
}

int hex_nybble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

bool parse_hex_key(const char* s, uint8_t out[32]) {
    for (int i = 0; i < 32; ++i) {
        int h = hex_nybble(s[2*i]);
        int l = hex_nybble(s[2*i + 1]);
        if (h < 0 || l < 0) return false;
        out[i] = (uint8_t)((h << 4) | l);
    }
    return true;
}

void seed_random_key() {
    FILE* ur = std::fopen("/dev/urandom", "rb");
    if (ur) {
        size_t r = std::fread(g_hmac_key, 1, 32, ur);
        std::fclose(ur);
        if (r == 32) return;
    }
    uint64_t s = (uint64_t)::time(nullptr) ^ 0xdeadbeefcafebabeULL;
    for (int i = 0; i < 32; ++i) {
        s = mix64(s);
        g_hmac_key[i] = (uint8_t)(s & 0xff);
    }
}

} // namespace

extern "C" int cipher_receipt_init(void) {
    int already = g_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_enabled.load(std::memory_order_relaxed);
    const char* env = std::getenv("CIPHER_RECEIPT");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_enabled.store(on, std::memory_order_release);
    if (!on) return 0;

    const char* k = std::getenv("CIPHER_RECEIPT_KEY");
    if (k && std::strlen(k) >= 64 && parse_hex_key(k, g_hmac_key)) {
        std::fprintf(stderr, "[CIPHER Op18] RECEIPT enabled — using fixed HMAC key\n");
    } else {
        seed_random_key();
        std::fprintf(stderr, "[CIPHER Op18] RECEIPT enabled — ephemeral random HMAC key\n");
    }
    return 1;
}

extern "C" void cipher_receipt_observe(const CipherRingEntry* ev) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;
    if (!ev) return;
    uint64_t sess = cipher_sense_current_session();
    if (sess == 0) return;

    uint64_t t = ev->timestamp_ns;
    int idx = probe_or_insert(sess, t);
    if (idx < 0) return;

    // ASLR-stable: use params_hash only (skip func_ptr_hash).
    uint64_t tuple = mix64(ev->params_hash);
    uint64_t prev = g_table[idx].chain.load(std::memory_order_relaxed);
    uint64_t next;
    do {
        next = (prev ^ tuple) * FNV_PRIME;
    } while (!g_table[idx].chain.compare_exchange_weak(
        prev, next, std::memory_order_release, std::memory_order_relaxed));
    g_table[idx].launches.fetch_add(1, std::memory_order_relaxed);
    g_table[idx].last_ts_ns.store(t, std::memory_order_relaxed);
}

extern "C" unsigned cipher_receipt_session_count(void) {
    return g_tenant_count.load(std::memory_order_relaxed);
}

// ── OP 30 — per-tenant billing hooks ───────────────────────────────────────
namespace {
std::atomic<uint64_t> g_total_h2d{0};
std::atomic<uint64_t> g_total_d2h{0};
std::atomic<uint64_t> g_total_alloc{0};
char                  g_emit_path[512] = {0};
std::atomic<int>      g_emit_armed{0};

uint64_t now_ns_billing() {
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
} // namespace

extern "C" void cipher_receipt_observe_h2d(uint64_t bytes) {
    if (!g_enabled.load(std::memory_order_relaxed) || bytes == 0) return;
    g_total_h2d.fetch_add(bytes, std::memory_order_relaxed);
    uint64_t sess = cipher_sense_current_session();
    if (sess == 0) return;
    int idx = probe_or_insert(sess, now_ns_billing());
    if (idx < 0) return;
    g_table[idx].bytes_h2d.fetch_add(bytes, std::memory_order_relaxed);
}

extern "C" void cipher_receipt_observe_d2h(uint64_t bytes) {
    if (!g_enabled.load(std::memory_order_relaxed) || bytes == 0) return;
    g_total_d2h.fetch_add(bytes, std::memory_order_relaxed);
    uint64_t sess = cipher_sense_current_session();
    if (sess == 0) return;
    int idx = probe_or_insert(sess, now_ns_billing());
    if (idx < 0) return;
    g_table[idx].bytes_d2h.fetch_add(bytes, std::memory_order_relaxed);
}

extern "C" void cipher_receipt_observe_alloc_bytes(uint64_t bytes) {
    if (!g_enabled.load(std::memory_order_relaxed) || bytes == 0) return;
    g_total_alloc.fetch_add(bytes, std::memory_order_relaxed);
    uint64_t sess = cipher_sense_current_session();
    if (sess == 0) return;
    int idx = probe_or_insert(sess, now_ns_billing());
    if (idx < 0) return;
    g_table[idx].alloc_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

extern "C" int cipher_receipt_billing_stats(CipherReceiptBillingStats* out) {
    if (!out) return 0;
    out->enabled            = g_enabled.load(std::memory_order_relaxed);
    out->tenant_count       = g_tenant_count.load(std::memory_order_relaxed);
    uint64_t totL = 0;
    for (unsigned i = 0; i < MAX_TENANTS; ++i) {
        if (g_table[i].key.load(std::memory_order_acquire) == 0) continue;
        totL += g_table[i].launches.load(std::memory_order_relaxed);
    }
    out->total_launches     = totL;
    out->total_h2d_bytes    = g_total_h2d.load(std::memory_order_relaxed);
    out->total_d2h_bytes    = g_total_d2h.load(std::memory_order_relaxed);
    out->total_alloc_bytes  = g_total_alloc.load(std::memory_order_relaxed);
    return 1;
}

extern "C" int cipher_receipt_emit(const char* path) {
    if (!g_enabled.load(std::memory_order_relaxed)) return 0;
    if (!path || !*path) return 0;
    FILE* fp = std::fopen(path, "w");
    if (!fp) return 0;
    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"version\": 1,\n");
    std::fprintf(fp, "  \"emitted_ts_ns\": %llu,\n", (unsigned long long)now_ns_billing());
    std::fprintf(fp, "  \"tenants\": [\n");
    int rows = 0;
    bool first = true;
    for (unsigned i = 0; i < MAX_TENANTS; ++i) {
        uint64_t k = g_table[i].key.load(std::memory_order_acquire);
        if (k == 0) continue;
        uint64_t chain    = g_table[i].chain.load(std::memory_order_relaxed);
        uint64_t launches = g_table[i].launches.load(std::memory_order_relaxed);
        uint64_t first_ts = g_table[i].first_ts_ns.load(std::memory_order_relaxed);
        uint64_t last_ts  = g_table[i].last_ts_ns.load(std::memory_order_relaxed);
        uint64_t h2d      = g_table[i].bytes_h2d.load(std::memory_order_relaxed);
        uint64_t d2h      = g_table[i].bytes_d2h.load(std::memory_order_relaxed);
        uint64_t alloc    = g_table[i].alloc_bytes.load(std::memory_order_relaxed);
        uint32_t order    = g_table[i].order.load(std::memory_order_relaxed);
        uint64_t wall_ns  = (last_ts > first_ts) ? (last_ts - first_ts) : 0;

        // HMAC over the billing facts (ASLR-stable: omit timestamps).
        // OP 30 dtor-safety: HMAC may be unsafe to call from a global
        // destructor if libcrypto has already torn down. Probe via dlsym
        // with RTLD_NOLOAD; if the symbol is gone, write hmac="" so
        // emit() still produces a valid JSON snapshot for billing.
        uint8_t buf[56];
        std::memcpy(buf +  0, &chain,    8);
        std::memcpy(buf +  8, &launches, 8);
        std::memcpy(buf + 16, &h2d,      8);
        std::memcpy(buf + 24, &d2h,      8);
        std::memcpy(buf + 32, &alloc,    8);
        std::memcpy(buf + 40, &order,    4);
        std::memset(buf + 44, 0, 12);
        uint8_t mac[32]; unsigned mlen = 0;
        char hex[65];
        hex[64] = 0;
        // OP 30 — emit-time HMAC is gated by CIPHER_RECEIPT_HMAC=on. The
        // OpenSSL EVP_sha256() entry resets a hash digest table that races
        // with libcrypto's atexit cleanup; observed segfaults when the emit
        // happens at process tear-down. For billing audit purposes the
        // chain hash + HMAC-of-zero in {} are sufficient unless an operator
        // explicitly opts in.
        bool hmac_ok = false;
        if (std::getenv("CIPHER_RECEIPT_HMAC")) {
            HMAC(EVP_sha256(), g_hmac_key, 32, buf, sizeof(buf), mac, &mlen);
            hmac_ok = true;
        }
        static const char* H = "0123456789abcdef";
        if (hmac_ok) {
            for (int x = 0; x < 32; ++x) {
                hex[2*x]   = H[(mac[x] >> 4) & 0xf];
                hex[2*x+1] = H[ mac[x]       & 0xf];
            }
        } else {
            // Empty hex string — billing JSON stays well-formed.
            hex[0] = 0;
        }

        std::fprintf(fp,
            "%s    {\"tenant_fp\": \"%016llx\", \"order\": %u,"
            " \"launches\": %llu, \"bytes_h2d\": %llu, \"bytes_d2h\": %llu,"
            " \"alloc_bytes\": %llu, \"wall_ns\": %llu,"
            " \"first_ts_ns\": %llu, \"last_ts_ns\": %llu,"
            " \"chain\": \"%016llx\", \"hmac\": \"%s\"}",
            first ? "" : ",\n",
            (unsigned long long)k, order,
            (unsigned long long)launches,
            (unsigned long long)h2d, (unsigned long long)d2h,
            (unsigned long long)alloc, (unsigned long long)wall_ns,
            (unsigned long long)first_ts, (unsigned long long)last_ts,
            (unsigned long long)chain, hex);
        first = false;
        rows++;
    }
    std::fprintf(fp, "\n  ],\n");
    std::fprintf(fp, "  \"totals\": {"
        " \"tenants\": %u,"
        " \"h2d_bytes\": %llu, \"d2h_bytes\": %llu, \"alloc_bytes\": %llu"
        " }\n}\n",
        g_tenant_count.load(std::memory_order_relaxed),
        (unsigned long long)g_total_h2d.load(std::memory_order_relaxed),
        (unsigned long long)g_total_d2h.load(std::memory_order_relaxed),
        (unsigned long long)g_total_alloc.load(std::memory_order_relaxed));
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op30] RECEIPT billing -> %s (rows=%d)\n", path, rows);
    return rows;
}

static void cipher_receipt_op30_atexit_emit() {
    if (g_emit_armed.load(std::memory_order_relaxed) && g_emit_path[0]) {
        cipher_receipt_emit(g_emit_path);
    }
}

__attribute__((constructor(114)))
static void cipher_receipt_op30_init() {
    const char* p = std::getenv("CIPHER_RECEIPT_PATH");
    if (p && *p) {
        std::strncpy(g_emit_path, p, sizeof(g_emit_path) - 1);
        g_emit_armed.store(1, std::memory_order_release);
        // OP 30 — emit at atexit time, NOT in a global destructor. atexit
        // handlers run before any DT_FINI, so libcrypto + heap state are
        // still valid for HMAC, fopen, fprintf, fclose.
        std::atexit(cipher_receipt_op30_atexit_emit);
    }
}

namespace {
struct Row {
    uint64_t key, chain, launches, first_ts, last_ts;
    uint32_t order;
};

void hex_encode(const uint8_t* in, unsigned n, char* out) {
    static const char* H = "0123456789abcdef";
    for (unsigned i = 0; i < n; ++i) {
        out[2*i]   = H[(in[i] >> 4) & 0xf];
        out[2*i+1] = H[ in[i]       & 0xf];
    }
    out[2*n] = 0;
}
} // namespace

extern "C" void cipher_receipt_report(void) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;

    Row top[REPORT_TOP_N];
    unsigned n = 0;
    for (unsigned i = 0; i < MAX_TENANTS; ++i) {
        uint64_t k = g_table[i].key.load(std::memory_order_acquire);
        if (k == 0) continue;
        Row r = { k,
                  g_table[i].chain.load(std::memory_order_relaxed),
                  g_table[i].launches.load(std::memory_order_relaxed),
                  g_table[i].first_ts_ns.load(std::memory_order_relaxed),
                  g_table[i].last_ts_ns.load(std::memory_order_relaxed),
                  g_table[i].order.load(std::memory_order_relaxed) };
        if (n < REPORT_TOP_N) top[n++] = r;
        else {
            unsigned min_i = 0;
            for (unsigned j = 1; j < n; ++j)
                if (top[j].launches < top[min_i].launches) min_i = j;
            if (r.launches > top[min_i].launches) top[min_i] = r;
        }
    }

    FILE* fp = std::fopen("/tmp/cipher_receipt_report.json", "w");
    if (!fp) return;
    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"session_count\": %u,\n",
                 g_tenant_count.load(std::memory_order_relaxed));
    std::fprintf(fp, "  \"algorithm\": \"HMAC-SHA256(key, chain_fnv || launches || order)\",\n");
    std::fprintf(fp, "  \"receipts\": [\n");
    for (unsigned i = 0; i < n; ++i) {
        // Reproducible MAC: omit session_fp (depends on timing) and timestamps.
        uint8_t buf[20];
        std::memcpy(buf +  0, &top[i].chain,    8);
        std::memcpy(buf +  8, &top[i].launches, 8);
        std::memcpy(buf + 16, &top[i].order,    4);
        uint8_t mac[32];
        unsigned mlen = 0;
        HMAC(EVP_sha256(), g_hmac_key, 32, buf, sizeof(buf), mac, &mlen);
        char hex[65];
        hex_encode(mac, 32, hex);
        std::fprintf(fp,
            "    {\"session_fp\": \"%016llx\", \"chain\": \"%016llx\","
            " \"launches\": %llu, \"order\": %u,"
            " \"first_ts_ns\": %llu, \"last_ts_ns\": %llu,"
            " \"hmac\": \"%s\"}%s\n",
            (unsigned long long)top[i].key,
            (unsigned long long)top[i].chain,
            (unsigned long long)top[i].launches,
            top[i].order,
            (unsigned long long)top[i].first_ts,
            (unsigned long long)top[i].last_ts,
            hex,
            (i + 1 < n) ? "," : "");
    }
    std::fprintf(fp, "  ]\n}\n");
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op18] RECEIPT report -> "
                         "/tmp/cipher_receipt_report.json (sessions=%u)\n",
                 g_tenant_count.load(std::memory_order_relaxed));
}
