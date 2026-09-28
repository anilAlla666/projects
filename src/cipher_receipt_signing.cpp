// CIPHER receipt signing — Tier B isolated DSO.
// Build: links libssl + libcrypto. Loaded standalone via LD_PRELOAD or
// dlopen — does NOT depend on libcipher_rt.so or libcipher_hook.so.

#include "cipher_receipt_signing.h"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <mutex>
#include <openssl/hmac.h>
#include <openssl/sha.h>
#include <pthread.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

constexpr int KEY_BYTES = 32;        // 256-bit HMAC key
constexpr int HMAC_HEX_LEN = 64;     // SHA-256 = 32 bytes = 64 hex

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_started{0};
std::mutex            g_mu;
int                   g_log_fd = -1;
std::string           g_log_path;
uint8_t               g_key[KEY_BYTES] = {0};
std::atomic<uint64_t> g_seq{0};
std::atomic<uint64_t> g_lines_written{0};
std::atomic<uint64_t> g_bytes_written{0};
std::atomic<uint64_t> g_key_load_failures{0};
std::atomic<uint64_t> g_write_failures{0};
char                  g_prev_chain_hex[HMAC_HEX_LEN + 1] = {0};

uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void hex_encode(const uint8_t* in, int n, char* out) {
    static const char H[] = "0123456789abcdef";
    for (int i = 0; i < n; i++) {
        out[2*i]   = H[(in[i] >> 4) & 0xF];
        out[2*i+1] = H[in[i] & 0xF];
    }
    out[2*n] = 0;
}

int hex_decode_one(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

bool load_key_from_file(const char* path) {
    FILE* fp = std::fopen(path, "r");
    if (!fp) return false;
    char hex[2 * KEY_BYTES + 8] = {0};
    size_t n = std::fread(hex, 1, sizeof(hex) - 1, fp);
    std::fclose(fp);
    // Strip trailing whitespace.
    while (n > 0 && (hex[n-1] == '\n' || hex[n-1] == '\r' || hex[n-1] == ' ')) {
        hex[--n] = 0;
    }
    if (n < 2 * KEY_BYTES) return false;
    for (int i = 0; i < KEY_BYTES; i++) {
        int hi = hex_decode_one(hex[2*i]);
        int lo = hex_decode_one(hex[2*i+1]);
        if (hi < 0 || lo < 0) return false;
        g_key[i] = (uint8_t)((hi << 4) | lo);
    }
    return true;
}

} // namespace

extern "C" int cipher_receipt_signing_init(const char* log_path,
                                            const char* key_path) {
    if (g_started.exchange(1, std::memory_order_acq_rel))
        return g_enabled.load(std::memory_order_relaxed);

    // Resolve key path: arg → env → /etc/cipher/hmac.key
    const char* k = key_path;
    if (!k || !*k) k = std::getenv("CIPHER_RECEIPT_KEY");
    if (!k || !*k) k = "/etc/cipher/hmac.key";
    if (!load_key_from_file(k)) {
        g_key_load_failures.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr,
            "[CIPHER RECEIPT-SIGN] key load failed from %s — disabled\n", k);
        return 0;
    }

    // Resolve log path: arg → env → /var/log/cipher/receipts.jsonl
    const char* lp = log_path;
    if (!lp || !*lp) lp = std::getenv("CIPHER_RECEIPT_LOG");
    if (!lp || !*lp) lp = "/var/log/cipher/receipts.jsonl";
    g_log_path = lp;

    g_log_fd = ::open(lp, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (g_log_fd < 0) {
        std::fprintf(stderr,
            "[CIPHER RECEIPT-SIGN] open(%s) failed: %s — disabled\n",
            lp, std::strerror(errno));
        return 0;
    }
    g_enabled.store(1, std::memory_order_release);
    std::fprintf(stderr,
        "[CIPHER RECEIPT-SIGN] enabled key=%s log=%s\n", k, lp);
    return 1;
}

extern "C" int cipher_receipt_signing_append(const char* tenant_id,
                                              const char* event,
                                              const char* data_json) {
    if (!g_enabled.load(std::memory_order_relaxed)) return 0;
    if (!tenant_id) tenant_id = "?";
    if (!event)     event     = "?";
    if (!data_json) data_json = "{}";

    uint64_t seq = g_seq.fetch_add(1, std::memory_order_relaxed);
    uint64_t ts  = now_ns();

    // Build the to-sign payload — prev_chain || tenant || event || data
    // as a flat byte buffer. HMAC over it.
    std::lock_guard<std::mutex> lk(g_mu);
    char payload[8192];
    int plen = std::snprintf(payload, sizeof(payload),
        "%s|%s|%s|%s",
        g_prev_chain_hex, tenant_id, event, data_json);
    if (plen < 0 || plen >= (int)sizeof(payload)) {
        g_write_failures.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    uint8_t mac[32]; unsigned mlen = 0;
    HMAC(EVP_sha256(), g_key, KEY_BYTES,
         (const uint8_t*)payload, (size_t)plen, mac, &mlen);
    if (mlen != 32) {
        g_write_failures.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    char hmac_hex[HMAC_HEX_LEN + 1];
    hex_encode(mac, 32, hmac_hex);

    char line[8192 + 256];
    int n = std::snprintf(line, sizeof(line),
        "{\"seq\":%llu,\"ts_ns\":%llu,\"pid\":%d,"
        "\"tenant\":\"%s\",\"event\":\"%s\","
        "\"data\":%s,\"prev_chain\":\"%s\",\"hmac\":\"%s\"}\n",
        (unsigned long long)seq, (unsigned long long)ts, (int)getpid(),
        tenant_id, event, data_json, g_prev_chain_hex, hmac_hex);
    if (n < 0 || n >= (int)sizeof(line)) {
        g_write_failures.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    ssize_t w = ::write(g_log_fd, line, (size_t)n);
    if (w != (ssize_t)n) {
        g_write_failures.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    g_lines_written.fetch_add(1, std::memory_order_relaxed);
    g_bytes_written.fetch_add((uint64_t)w, std::memory_order_relaxed);
    // Update prev_chain → this line's hmac for the next event.
    std::memcpy(g_prev_chain_hex, hmac_hex, HMAC_HEX_LEN + 1);
    return 1;
}

extern "C" void cipher_receipt_signing_flush(void) {
    if (g_log_fd >= 0) {
        ::fsync(g_log_fd);
    }
}

extern "C" int cipher_receipt_signing_verify_chain(const char* log_path,
                                                    const char* key_path) {
    // Load key independently — verify is offline, doesn't touch g_key.
    uint8_t key[KEY_BYTES] = {0};
    {
        const char* kp = key_path ? key_path : "/etc/cipher/hmac.key";
        FILE* kf = std::fopen(kp, "r");
        if (!kf) return -1;
        char hex[2*KEY_BYTES + 8] = {0};
        size_t n = std::fread(hex, 1, sizeof(hex) - 1, kf);
        std::fclose(kf);
        while (n > 0 && (hex[n-1] == '\n' || hex[n-1] == '\r')) hex[--n] = 0;
        if (n < 2*KEY_BYTES) return -1;
        for (int i = 0; i < KEY_BYTES; i++) {
            int hi = hex_decode_one(hex[2*i]);
            int lo = hex_decode_one(hex[2*i+1]);
            if (hi < 0 || lo < 0) return -1;
            key[i] = (uint8_t)((hi << 4) | lo);
        }
    }
    FILE* lf = std::fopen(log_path, "r");
    if (!lf) return -1;
    char line[16384];
    char prev[HMAC_HEX_LEN + 1] = {0};
    int line_no = 0;
    int verified = 0;
    while (std::fgets(line, sizeof(line), lf)) {
        line_no++;
        // Find prev_chain, tenant, event, data, hmac fields. Brittle parse;
        // we trust the writer's JSON shape.
        char *pc_q = std::strstr(line, "\"prev_chain\":\"");
        char *t_q  = std::strstr(line, "\"tenant\":\"");
        char *e_q  = std::strstr(line, "\"event\":\"");
        char *d_q  = std::strstr(line, "\"data\":");
        char *h_q  = std::strstr(line, "\"hmac\":\"");
        if (!pc_q || !t_q || !e_q || !d_q || !h_q) continue;
        // prev_chain may be empty (first line). Parse defensively: read up
        // to closing quote.
        char pc[HMAC_HEX_LEN + 1] = {0};
        char* pc_start = pc_q + 14;
        char* pc_end   = std::strchr(pc_start, '"');
        if (pc_end && (size_t)(pc_end - pc_start) <= HMAC_HEX_LEN) {
            std::memcpy(pc, pc_start, pc_end - pc_start);
            pc[pc_end - pc_start] = 0;
        }
        char hmac_hex[HMAC_HEX_LEN + 1]; std::memcpy(hmac_hex, h_q + 8, HMAC_HEX_LEN); hmac_hex[HMAC_HEX_LEN] = 0;
        // Extract tenant
        char tenant[128]; char *tend = std::strchr(t_q + 10, '"');
        if (!tend || (tend - (t_q + 10)) >= 127) continue;
        size_t tn = tend - (t_q + 10); std::memcpy(tenant, t_q + 10, tn); tenant[tn] = 0;
        // Extract event
        char event[128]; char *eend = std::strchr(e_q + 9, '"');
        if (!eend || (eend - (e_q + 9)) >= 127) continue;
        size_t en = eend - (e_q + 9); std::memcpy(event, e_q + 9, en); event[en] = 0;
        // Extract data — from {  to matching } (depth-balanced; trivial parser).
        char *ds = std::strchr(d_q, '{');
        if (!ds) continue;
        int depth = 0; char *de = ds;
        for (; *de; de++) {
            if (*de == '{') depth++;
            else if (*de == '}') { depth--; if (depth == 0) { de++; break; } }
        }
        if (depth != 0) continue;
        std::string data(ds, de - ds);

        // Verify HMAC against (prev || tenant || event || data) using the
        // chain prev we've been tracking. On line 1, prev is empty; pc
        // should also be empty.
        if (std::strcmp(prev, pc) != 0) {
            std::fclose(lf);
            return -2 - line_no;        // chain break
        }
        char payload[8192];
        int plen = std::snprintf(payload, sizeof(payload),
            "%s|%s|%s|%s", prev, tenant, event, data.c_str());
        if (plen < 0 || plen >= (int)sizeof(payload)) continue;
        uint8_t mac[32]; unsigned mlen = 0;
        HMAC(EVP_sha256(), key, KEY_BYTES,
             (const uint8_t*)payload, (size_t)plen, mac, &mlen);
        char ver[HMAC_HEX_LEN + 1];
        hex_encode(mac, 32, ver);
        if (std::memcmp(ver, hmac_hex, HMAC_HEX_LEN) != 0) {
            std::fclose(lf);
            return -2 - line_no;
        }
        std::memcpy(prev, hmac_hex, HMAC_HEX_LEN + 1);
        verified++;
    }
    std::fclose(lf);
    return verified;
}

extern "C" int cipher_receipt_signing_stats(struct CipherReceiptSigningStats* out) {
    if (!out) return 0;
    out->enabled            = g_enabled.load(std::memory_order_relaxed);
    out->lines_written      = g_lines_written.load(std::memory_order_relaxed);
    out->bytes_written      = g_bytes_written.load(std::memory_order_relaxed);
    out->key_load_failures  = g_key_load_failures.load(std::memory_order_relaxed);
    out->write_failures     = g_write_failures.load(std::memory_order_relaxed);
    return 1;
}

// Auto-init when CIPHER_RECEIPT_SIGN=on (or when CIPHER_PROFILE flips
// CIPHER_RECEIPT=on with a key file present).
__attribute__((constructor(118)))
static void cipher_receipt_signing_autoinit() {
    const char* e = std::getenv("CIPHER_RECEIPT_SIGN");
    bool sign_on = e && (e[0]=='1'||e[0]=='o'||e[0]=='O'||e[0]=='t'||e[0]=='T');
    if (!sign_on) {
        // Auto-enable only if RECEIPT was turned on AND a key file exists.
        if (!std::getenv("CIPHER_RECEIPT")) return;
        const char* kp = std::getenv("CIPHER_RECEIPT_KEY");
        if (!kp) kp = "/etc/cipher/hmac.key";
        struct stat st;
        if (::stat(kp, &st) != 0) return;
    }
    cipher_receipt_signing_init(nullptr, nullptr);
}
