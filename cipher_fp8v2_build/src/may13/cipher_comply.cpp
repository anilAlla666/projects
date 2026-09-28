// Op 29 COMPLY — v1 compliance pack.

#include "may13/cipher_comply.h"
#include "may13/cipher_receipt.h"
#include "may13/cipher_carbon.h"
#include "may13/cipher_guard.h"
#include "may13/cipher_determinism.h"
#include "may13/cipher_fairness.h"
#include "may13/cipher_topology.h"

#include "cipher_rt_commit.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace {
std::atomic<int> g_enabled{0};
std::atomic<int> g_initialized{0};
std::atomic<int> g_last_ok{0};
} // namespace

extern "C" int cipher_comply_init(void) {
    int already = g_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_enabled.load(std::memory_order_relaxed);
    const char* env = std::getenv("CIPHER_COMPLY");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        std::fprintf(stderr, "[CIPHER Op29] COMPLY enabled — "
                             "bundles RECEIPT+CARBON+GUARD+DETERMINISM+FAIRNESS+TOPOLOGY\n");
    }
    return on;
}

extern "C" void cipher_comply_observe(const CipherRingEntry* ev) {
    (void)ev;  // aggregation-only; no hot-path work.
}

extern "C" int cipher_comply_ok(void) {
    return g_last_ok.load(std::memory_order_relaxed);
}

extern "C" void cipher_comply_report(void) {
    /* W7-9 Step 4 — coherent snapshot acquire (slow-path; emits the
     * COMMIT-published state alongside this report's existing aggregate). */
    struct cipher_rt_snapshot _snap;
    cipher_rt_snapshot_acquire(0u, &_snap);
    (void)_snap;

    if (!g_enabled.load(std::memory_order_relaxed)) return;

    unsigned receipt_sessions = cipher_receipt_session_count();
    unsigned carbon_sessions  = cipher_carbon_session_count();
    unsigned guard_sessions   = cipher_guard_session_count();
    unsigned guard_leaks      = cipher_guard_leak_count();
    uint64_t determ_hash      = cipher_determinism_hash();
    uint64_t determ_count     = cipher_determinism_count();
    unsigned fair_tenants     = cipher_fairness_tenant_count();
    unsigned fair_overruns    = cipher_fairness_overrun_count();
    unsigned topo_devices     = cipher_topology_device_count();

    int ok = (guard_leaks == 0)
          && (receipt_sessions >= 1)
          && (determ_count > 0)
          && (fair_overruns == 0);
    g_last_ok.store(ok, std::memory_order_release);

    FILE* fp = std::fopen("/tmp/cipher_comply_report.json", "w");
    if (!fp) return;
    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"compliance_ok\": %s,\n", ok ? "true" : "false");
    std::fprintf(fp, "  \"receipt\":     {\"session_count\": %u},\n", receipt_sessions);
    std::fprintf(fp, "  \"carbon\":      {\"session_count\": %u},\n", carbon_sessions);
    std::fprintf(fp, "  \"guard\":       {\"session_count\": %u, \"leak_count\": %u},\n",
                 guard_sessions, guard_leaks);
    std::fprintf(fp, "  \"determinism\": {\"dispatch_hash\": \"%016llx\", \"dispatch_count\": %llu},\n",
                 (unsigned long long)determ_hash, (unsigned long long)determ_count);
    std::fprintf(fp, "  \"fairness\":    {\"tenant_count\": %u, \"overrun_count\": %u},\n",
                 fair_tenants, fair_overruns);
    std::fprintf(fp, "  \"topology\":    {\"device_count\": %u}\n", topo_devices);
    std::fprintf(fp, "}\n");
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op29] COMPLY report -> "
                         "/tmp/cipher_comply_report.json (ok=%d)\n", ok);
}
