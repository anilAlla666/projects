// Op 23 CARBON — v1 per-tenant carbon estimate.

#include "cipher_carbon.h"
#include "cipher_sense.h"
#include "cipher_op_counters.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace {

constexpr unsigned MAX_TENANTS  = 256;
constexpr unsigned REPORT_TOP_N = 32;

struct Tenant {
    std::atomic<uint64_t> key;        // session fp; 0 = empty
    std::atomic<uint64_t> work;
    std::atomic<uint64_t> launches;
};

alignas(64) Tenant g_table[MAX_TENANTS];

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
std::atomic<unsigned> g_tenant_count{0};

double g_j_per_unit     = 1e-9;
double g_gco2_per_kwh   = 400.0;

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
                g_tenant_count.fetch_add(1, std::memory_order_relaxed);
                cipher_op_inc(OP_CARBON);
                return (int)i;
            }
            if (g_table[i].key.load(std::memory_order_acquire) == key) return (int)i;
        }
    }
    return -1;
}

} // namespace

extern "C" int cipher_carbon_init(void) {
    int already = g_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_enabled.load(std::memory_order_relaxed);
    const char* env = std::getenv("CIPHER_CARBON");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_enabled.store(on, std::memory_order_release);

    const char* j = std::getenv("CIPHER_CARBON_J_PER_UNIT");
    if (j) { double v = std::strtod(j, nullptr); if (v > 0) g_j_per_unit = v; }
    const char* g = std::getenv("CIPHER_CARBON_GCO2_PER_KWH");
    if (g) { double v = std::strtod(g, nullptr); if (v > 0) g_gco2_per_kwh = v; }

    if (on) {
        std::fprintf(stderr, "[CIPHER Op23] CARBON enabled — "
                             "J_per_unit=%.3e gCO2_per_kWh=%.2f\n",
                     g_j_per_unit, g_gco2_per_kwh);
    }
    return on;
}

extern "C" void cipher_carbon_observe(const CipherRingEntry* ev) {
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
    g_table[idx].work.fetch_add(units, std::memory_order_relaxed);
    g_table[idx].launches.fetch_add(1, std::memory_order_relaxed);
}

extern "C" unsigned cipher_carbon_session_count(void) {
    return g_tenant_count.load(std::memory_order_relaxed);
}

namespace {
struct Row { uint64_t key, work, launches; double gco2; };
} // namespace

extern "C" void cipher_carbon_report(void) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;

    Row top[REPORT_TOP_N];
    unsigned n = 0;
    double total_gco2 = 0.0;

    // gCO2 = J * (kWh / 3.6e6 J) * gCO2_per_kWh.
    const double j_to_gco2 = (g_gco2_per_kwh / 3.6e6);

    for (unsigned i = 0; i < MAX_TENANTS; ++i) {
        uint64_t k = g_table[i].key.load(std::memory_order_acquire);
        if (k == 0) continue;
        uint64_t w = g_table[i].work.load(std::memory_order_relaxed);
        double joules = (double)w * g_j_per_unit;
        double gco2 = joules * j_to_gco2;
        total_gco2 += gco2;
        Row r = { k, w, g_table[i].launches.load(std::memory_order_relaxed), gco2 };
        if (n < REPORT_TOP_N) top[n++] = r;
        else {
            unsigned min_i = 0;
            for (unsigned j = 1; j < n; ++j)
                if (top[j].gco2 < top[min_i].gco2) min_i = j;
            if (r.gco2 > top[min_i].gco2) top[min_i] = r;
        }
    }
    for (unsigned i = 1; i < n; ++i) {
        Row key = top[i]; int j = (int)i - 1;
        while (j >= 0 && top[j].gco2 < key.gco2) { top[j+1] = top[j]; --j; }
        top[j+1] = key;
    }

    FILE* fp = std::fopen("/tmp/cipher_carbon_report.json", "w");
    if (!fp) return;
    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"session_count\": %u,\n",
                 g_tenant_count.load(std::memory_order_relaxed));
    std::fprintf(fp, "  \"j_per_unit\": %.6e,\n", g_j_per_unit);
    std::fprintf(fp, "  \"gco2_per_kwh\": %.6f,\n", g_gco2_per_kwh);
    std::fprintf(fp, "  \"total_gco2\": %.6e,\n", total_gco2);
    std::fprintf(fp, "  \"sessions\": [\n");
    for (unsigned i = 0; i < n; ++i) {
        std::fprintf(fp,
            "    {\"session_fp\": \"%016llx\", \"work\": %llu, \"launches\": %llu,"
            " \"gco2\": %.6e}%s\n",
            (unsigned long long)top[i].key,
            (unsigned long long)top[i].work,
            (unsigned long long)top[i].launches,
            top[i].gco2,
            (i + 1 < n) ? "," : "");
    }
    std::fprintf(fp, "  ]\n}\n");
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op23] CARBON report -> "
                         "/tmp/cipher_carbon_report.json (sessions=%u total_gco2=%.6e)\n",
                 g_tenant_count.load(std::memory_order_relaxed), total_gco2);
}
