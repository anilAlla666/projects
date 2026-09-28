// Op 18 RECEIPT — v1 per-session signed proof of compute.

#include "may13/cipher_receipt.h"
#include "may13/cipher_sense.h"

#include "cipher_rt_commit.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <openssl/hmac.h>
#include <openssl/evp.h>

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
    /* W7-9 Step 4 — coherent snapshot acquire (slow-path; emits the
     * COMMIT-published state alongside this report's existing aggregate). */
    struct cipher_rt_snapshot _snap;
    cipher_rt_snapshot_acquire(0u, &_snap);
    (void)_snap;

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
