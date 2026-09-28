// Op 24 FAIRNESS — v1 per-tenant work quota observer.
//
// Open-address table keyed by session_fp; accumulates grid*block volume.

#include "may13/cipher_fairness.h"
#include "may13/cipher_sense.h"

#include "cipher_rt_commit.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace {

constexpr unsigned MAX_TENANTS   = 256;
constexpr unsigned REPORT_TOP_N  = 32;
constexpr uint64_t DEFAULT_QUOTA = 5ULL * 1000 * 1000 * 1000;  // 5e9 work units

struct Tenant {
    std::atomic<uint64_t> key;       // session fp; 0 = empty
    std::atomic<uint64_t> work;
    std::atomic<uint64_t> launches;
    std::atomic<uint32_t> overrun_flag;
};

alignas(64) Tenant g_table[MAX_TENANTS];

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
std::atomic<uint64_t> g_quota{DEFAULT_QUOTA};
std::atomic<unsigned> g_tenant_count{0};
std::atomic<unsigned> g_overrun_count{0};

inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

int probe_or_insert(uint64_t key) {
    uint64_t h = mix64(key);
    for (unsigned p = 0; p < MAX_TENANTS; ++p) {
        unsigned i = (unsigned)((h + p) & (MAX_TENANTS - 1));
        uint64_t cur = g_table[i].key.load(std::memory_order_acquire);
        if (cur == key) return (int)i;
        if (cur == 0) {
            uint64_t expected = 0;
            if (g_table[i].key.compare_exchange_strong(
                    expected, key, std::memory_order_acq_rel)) {
                g_table[i].work.store(0, std::memory_order_relaxed);
                g_table[i].launches.store(0, std::memory_order_relaxed);
                g_table[i].overrun_flag.store(0, std::memory_order_relaxed);
                g_tenant_count.fetch_add(1, std::memory_order_relaxed);
                return (int)i;
            }
            if (g_table[i].key.load(std::memory_order_acquire) == key) return (int)i;
        }
    }
    return -1;
}

} // namespace

extern "C" int cipher_fairness_init(void) {
    int already = g_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_enabled.load(std::memory_order_relaxed);
    const char* env = std::getenv("CIPHER_FAIRNESS");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_enabled.store(on, std::memory_order_release);

    const char* q = std::getenv("CIPHER_FAIRNESS_QUOTA");
    if (q) {
        unsigned long long v = std::strtoull(q, nullptr, 10);
        if (v > 0) g_quota.store((uint64_t)v, std::memory_order_release);
    }

    if (on) {
        std::fprintf(stderr, "[CIPHER Op24] FAIRNESS enabled — quota=%llu work_units\n",
                     (unsigned long long)g_quota.load(std::memory_order_relaxed));
    }
    return on;
}

extern "C" void cipher_fairness_observe(const CipherRingEntry* ev) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;
    if (!ev) return;
    uint64_t sess = cipher_sense_current_session();
    if (sess == 0) return;

    uint64_t gx = ev->grid_x ? ev->grid_x : 1u;
    uint64_t gy = ev->grid_y ? ev->grid_y : 1u;
    uint64_t gz = ev->grid_z ? ev->grid_z : 1u;
    uint64_t bx = ev->block_x ? ev->block_x : 1u;
    uint64_t by = ev->block_y ? ev->block_y : 1u;
    uint64_t bz = ev->block_z ? ev->block_z : 1u;
    uint64_t units = gx * gy * gz * bx * by * bz;
    if (units == 0) units = 1;

    int idx = probe_or_insert(sess);
    if (idx < 0) return;

    Tenant& t = g_table[idx];
    uint64_t prev_work = t.work.fetch_add(units, std::memory_order_relaxed);
    t.launches.fetch_add(1, std::memory_order_relaxed);

    uint64_t q = g_quota.load(std::memory_order_relaxed);
    if (prev_work < q && (prev_work + units) >= q) {
        uint32_t prev_flag = t.overrun_flag.exchange(1, std::memory_order_relaxed);
        if (prev_flag == 0) {
            g_overrun_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

extern "C" unsigned cipher_fairness_tenant_count(void) {
    return g_tenant_count.load(std::memory_order_relaxed);
}

extern "C" unsigned cipher_fairness_overrun_count(void) {
    return g_overrun_count.load(std::memory_order_relaxed);
}

namespace {
struct Row { uint64_t key, work, launches; uint32_t overrun; };
} // namespace

extern "C" void cipher_fairness_report(void) {
    /* W7-9 Step 4 — coherent snapshot acquire (slow-path; emits the
     * COMMIT-published state alongside this report's existing aggregate). */
    struct cipher_rt_snapshot _snap;
    cipher_rt_snapshot_acquire(0u, &_snap);
    (void)_snap;

    if (!g_enabled.load(std::memory_order_relaxed)) return;
    uint64_t q = g_quota.load(std::memory_order_relaxed);

    Row top[REPORT_TOP_N];
    unsigned n = 0;
    for (unsigned i = 0; i < MAX_TENANTS; ++i) {
        uint64_t k = g_table[i].key.load(std::memory_order_acquire);
        if (k == 0) continue;
        Row r = { k,
                  g_table[i].work.load(std::memory_order_relaxed),
                  g_table[i].launches.load(std::memory_order_relaxed),
                  g_table[i].overrun_flag.load(std::memory_order_relaxed) };
        if (n < REPORT_TOP_N) top[n++] = r;
        else {
            unsigned min_i = 0;
            for (unsigned j = 1; j < n; ++j)
                if (top[j].work < top[min_i].work) min_i = j;
            if (r.work > top[min_i].work) top[min_i] = r;
        }
    }
    for (unsigned i = 1; i < n; ++i) {
        Row key = top[i]; int j = (int)i - 1;
        while (j >= 0 && top[j].work < key.work) { top[j+1] = top[j]; --j; }
        top[j+1] = key;
    }

    FILE* fp = std::fopen("/tmp/cipher_fairness_report.json", "w");
    if (!fp) return;
    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"quota\": %llu,\n", (unsigned long long)q);
    std::fprintf(fp, "  \"tenant_count\": %u,\n",
                 g_tenant_count.load(std::memory_order_relaxed));
    std::fprintf(fp, "  \"overrun_count\": %u,\n",
                 g_overrun_count.load(std::memory_order_relaxed));
    std::fprintf(fp, "  \"tenants\": [\n");
    for (unsigned i = 0; i < n; ++i) {
        std::fprintf(fp,
            "    {\"session_fp\": \"%016llx\", \"work\": %llu, \"launches\": %llu,"
            " \"overrun\": %u}%s\n",
            (unsigned long long)top[i].key,
            (unsigned long long)top[i].work,
            (unsigned long long)top[i].launches,
            top[i].overrun,
            (i + 1 < n) ? "," : "");
    }
    std::fprintf(fp, "  ]\n}\n");
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op24] FAIRNESS report -> "
                         "/tmp/cipher_fairness_report.json (tenants=%u over=%u quota=%llu)\n",
                 g_tenant_count.load(std::memory_order_relaxed),
                 g_overrun_count.load(std::memory_order_relaxed),
                 (unsigned long long)q);
}
