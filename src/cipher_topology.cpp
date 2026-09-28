// Op 25 TOPOLOGY — v1 NVLink/PCIe peer adjacency.
//
// Static inference: init-time CUDA queries + dense adjacency. Observer is
// a no-op.

#include "cipher_topology.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

constexpr unsigned MAX_DEV = 16;

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
std::atomic<unsigned> g_device_count{0};
uint8_t               g_adj[MAX_DEV * MAX_DEV] = {0};

} // namespace

extern "C" int cipher_topology_init(void) {
    int already = g_initialized.exchange(1, std::memory_order_acq_rel);
    if (already) return g_enabled.load(std::memory_order_relaxed);
    const char* env = std::getenv("CIPHER_TOPOLOGY");
    int on = (env != nullptr) && (std::strcmp(env, "on") == 0
                                || std::strcmp(env, "1") == 0
                                || std::strcmp(env, "ON") == 0);
    g_enabled.store(on, std::memory_order_release);
    if (!on) return 0;

    int dev_count = 0;
    cudaError_t e = cudaGetDeviceCount(&dev_count);
    if (e != cudaSuccess) dev_count = 0;
    if (dev_count < 0) dev_count = 0;
    if (dev_count > (int)MAX_DEV) dev_count = MAX_DEV;

    for (int i = 0; i < dev_count; ++i) {
        for (int j = 0; j < dev_count; ++j) {
            if (i == j) { g_adj[i * MAX_DEV + j] = 0; continue; }
            int can = 0;
            if (cudaDeviceCanAccessPeer(&can, i, j) != cudaSuccess) can = 0;
            g_adj[i * MAX_DEV + j] = (uint8_t)(can ? 1 : 0);
        }
    }
    g_device_count.store((unsigned)dev_count, std::memory_order_release);

    std::fprintf(stderr, "[CIPHER Op25] TOPOLOGY enabled — device_count=%d\n",
                 dev_count);
    return 1;
}

extern "C" void cipher_topology_observe(const CipherRingEntry* ev) {
    (void)ev;
    // v1: topology is static, no per-dispatch work.
}

extern "C" unsigned cipher_topology_device_count(void) {
    return g_device_count.load(std::memory_order_relaxed);
}

extern "C" void cipher_topology_report(void) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;
    unsigned n = g_device_count.load(std::memory_order_relaxed);

    FILE* fp = std::fopen("/tmp/cipher_topology_report.json", "w");
    if (!fp) return;
    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"device_count\": %u,\n", n);
    std::fprintf(fp, "  \"adjacency\": [\n");
    for (unsigned i = 0; i < n; ++i) {
        std::fprintf(fp, "    [");
        for (unsigned j = 0; j < n; ++j) {
            std::fprintf(fp, "%u%s", (unsigned)g_adj[i * MAX_DEV + j],
                         (j + 1 < n) ? ", " : "");
        }
        std::fprintf(fp, "]%s\n", (i + 1 < n) ? "," : "");
    }
    std::fprintf(fp, "  ],\n");

    unsigned edge_count = 0;
    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = 0; j < n; ++j)
            if (i != j && g_adj[i * MAX_DEV + j]) edge_count++;
    std::fprintf(fp, "  \"edge_count\": %u\n", edge_count);
    std::fprintf(fp, "}\n");
    std::fclose(fp);
    std::fprintf(stderr, "[CIPHER Op25] TOPOLOGY report -> "
                         "/tmp/cipher_topology_report.json (n=%u edges=%u)\n",
                 n, edge_count);
}
