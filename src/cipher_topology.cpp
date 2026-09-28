// Op 25 TOPOLOGY — v1 NVLink/PCIe peer adjacency.
//
// Static inference: init-time CUDA queries + dense adjacency. Observer is
// a no-op.

#include "cipher_topology.h"
#include "cipher_op_counters.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#include <dirent.h>
#include <dlfcn.h>
#include <sys/stat.h>

namespace {

constexpr unsigned MAX_DEV = 16;

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
std::atomic<unsigned> g_device_count{0};
uint8_t               g_adj[MAX_DEV * MAX_DEV] = {0};
uint8_t               g_nvlink[MAX_DEV * MAX_DEV] = {0};

// OP 33 NCCL tuner state ----------------------------------------------------
std::atomic<int>      g_tuner_initialized{0};
CipherNcclTunerStats  g_tuner = {};

bool env_truthy(const char* v) {
    if (!v) return false;
    return v[0] == '1' || v[0] == 't' || v[0] == 'T'
        || ((v[0] == 'o' || v[0] == 'O') && (v[1] == 'n' || v[1] == 'N'));
}

bool ib_present() {
    DIR* d = opendir("/sys/class/infiniband");
    if (!d) return false;
    bool any = false;
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        if (ent->d_name[0] == '.') continue;
        any = true;
        break;
    }
    closedir(d);
    return any;
}

// NVML probe via dlopen — we don't link against libnvidia-ml.so.
// Uses nvmlDeviceGetP2PStatus(NVML_P2P_CAPS_INDEX_NVLINK) which works
// uniformly whether the NVLink is direct or via NVSwitches (HGX/DGX),
// where nvmlDeviceGetNvLinkRemotePciInfo returns sentinel values.
// Returns # of (a,b) ordered pairs with active NVLink; fills g_nvlink[].
int probe_nvlinks(int dev_count) {
    if (dev_count <= 1) return 0;
    void* lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) lib = dlopen("libnvidia-ml.so", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) return 0;

    typedef int (*pf_init)(void);
    typedef int (*pf_shut)(void);
    typedef int (*pf_dev_by_idx)(unsigned, void**);
    typedef int (*pf_p2p_status)(void* /*d1*/, void* /*d2*/, int /*caps*/, int* /*status*/);

    pf_init        nvmlInit_v2 = (pf_init)        dlsym(lib, "nvmlInit_v2");
    pf_shut        nvmlShutdown= (pf_shut)        dlsym(lib, "nvmlShutdown");
    pf_dev_by_idx  nvmlGetByIdx= (pf_dev_by_idx)  dlsym(lib, "nvmlDeviceGetHandleByIndex_v2");
    pf_p2p_status  nvmlP2P     = (pf_p2p_status)  dlsym(lib, "nvmlDeviceGetP2PStatus");
    if (!nvmlInit_v2 || !nvmlGetByIdx || !nvmlP2P) { dlclose(lib); return 0; }
    if (nvmlInit_v2() != 0)                       { dlclose(lib); return 0; }

    void* handles[MAX_DEV] = {nullptr};
    for (int i = 0; i < dev_count; ++i) nvmlGetByIdx((unsigned)i, &handles[i]);

    constexpr int NVML_P2P_CAPS_INDEX_NVLINK = 2;
    constexpr int NVML_P2P_STATUS_OK         = 0;
    int pair_count = 0;
    int debug = env_truthy(getenv("CIPHER_NCCL_TUNER_DEBUG"));
    for (int a = 0; a < dev_count; ++a) {
        if (!handles[a]) continue;
        for (int b = 0; b < dev_count; ++b) {
            if (b == a || !handles[b]) continue;
            int status = -1;
            int rc = nvmlP2P(handles[a], handles[b], NVML_P2P_CAPS_INDEX_NVLINK, &status);
            if (debug) std::fprintf(stderr,
                "[CIPHER OP33] p2pStatus(NVLINK) a=%d b=%d rc=%d status=%d\n",
                a, b, rc, status);
            if (rc == 0 && status == NVML_P2P_STATUS_OK) {
                g_nvlink[a * MAX_DEV + b] = 1;
                pair_count++;
            }
        }
    }
    if (nvmlShutdown) nvmlShutdown();
    dlclose(lib);
    return pair_count;
}

// setenv() one var with overwrite=0 (don't clobber operator-set values).
// Reports whether the var ended up at our recommendation.
void tuner_setenv(const char* k, const char* v) {
    setenv(k, v, 0);
}

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
    cipher_op_inc(OP_TOPOLOGY);
    return 1;
}

extern "C" void cipher_topology_observe(const CipherRingEntry* ev) {
    (void)ev;
    // v1: topology is static, no per-dispatch work.
}

extern "C" unsigned cipher_topology_device_count(void) {
    return g_device_count.load(std::memory_order_relaxed);
}

// OP 33 — derive NCCL env hints from observed adjacency + NVLink + IB.
//
// Spec mapping:
//   NVLink GPU pairs:  NCCL_ALGO=Ring,  NCCL_PROTO=Simple,  NCCL_MIN_NCHANNELS=8
//   PCIe-only P2P:     NCCL_ALGO=Tree,  NCCL_PROTO=LL128,   NCCL_MIN_NCHANNELS=4
//   Single GPU:        skip (NCCL won't run; no envs set).
//   InfiniBand HCA:    NCCL_IB_DISABLE=0, NCCL_NET_GDR_LEVEL=5
//   No IB:             NCCL_IB_DISABLE=1
//
// All sets use setenv(..., 0) so an operator-supplied value wins.
extern "C" int cipher_topology_setenv_for_nccl(void) {
    if (g_tuner_initialized.exchange(1, std::memory_order_acq_rel))
        return g_tuner.ran;
    g_tuner.enabled = env_truthy(getenv("CIPHER_NCCL_TUNER"));
    if (!g_tuner.enabled) return 0;

    // Make sure topology is populated (CIPHER_TOPOLOGY may not be set).
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess) dev_count = 0;
    if (dev_count > (int)MAX_DEV) dev_count = MAX_DEV;
    g_tuner.gpu_count = dev_count;

    // Cheap P2P probe — separate from CIPHER_TOPOLOGY's matrix so we
    // don't depend on init order.
    int p2p_pairs = 0;
    for (int i = 0; i < dev_count; ++i)
        for (int j = 0; j < dev_count; ++j)
            if (i != j) {
                int can = 0;
                if (cudaDeviceCanAccessPeer(&can, i, j) == cudaSuccess && can) {
                    g_adj[i * MAX_DEV + j] = 1;
                    p2p_pairs++;
                }
            }

    g_tuner.nvlink_pairs    = probe_nvlinks(dev_count);
    g_tuner.nvlink_detected = (g_tuner.nvlink_pairs > 0) ? 1 : 0;
    g_tuner.pcie_only       = (g_tuner.nvlink_pairs == 0 && p2p_pairs > 0) ? 1 : 0;
    g_tuner.ib_detected     = ib_present() ? 1 : 0;
    g_tuner.ib_disable      = 2;     // unset
    g_tuner.net_gdr_level   = -1;

    if (dev_count >= 2) {
        if (g_tuner.nvlink_detected) {
            tuner_setenv("NCCL_ALGO",         "Ring");
            tuner_setenv("NCCL_PROTO",        "Simple");
            tuner_setenv("NCCL_MIN_NCHANNELS","8");
            std::strcpy(g_tuner.algo,  "Ring");
            std::strcpy(g_tuner.proto, "Simple");
            g_tuner.min_nchannels = 8;
        } else if (g_tuner.pcie_only) {
            tuner_setenv("NCCL_ALGO",         "Tree");
            tuner_setenv("NCCL_PROTO",        "LL128");
            tuner_setenv("NCCL_MIN_NCHANNELS","4");
            std::strcpy(g_tuner.algo,  "Tree");
            std::strcpy(g_tuner.proto, "LL128");
            g_tuner.min_nchannels = 4;
        }
    }

    if (g_tuner.ib_detected) {
        tuner_setenv("NCCL_IB_DISABLE",    "0");
        tuner_setenv("NCCL_NET_GDR_LEVEL", "5");
        g_tuner.ib_disable    = 0;
        g_tuner.net_gdr_level = 5;
    } else {
        tuner_setenv("NCCL_IB_DISABLE", "1");
        g_tuner.ib_disable = 1;
    }

    g_tuner.ran = 1;
    std::fprintf(stderr,
        "[CIPHER OP33] NCCL tuner — gpus=%d nvlink_pairs=%d pcie_only=%d ib=%d "
        "→ NCCL_ALGO=%s NCCL_PROTO=%s NCCL_MIN_NCHANNELS=%d "
        "NCCL_IB_DISABLE=%d NCCL_NET_GDR_LEVEL=%d\n",
        g_tuner.gpu_count, g_tuner.nvlink_pairs, g_tuner.pcie_only,
        g_tuner.ib_detected,
        g_tuner.algo[0]  ? g_tuner.algo  : "(unset)",
        g_tuner.proto[0] ? g_tuner.proto : "(unset)",
        g_tuner.min_nchannels,
        g_tuner.ib_disable, g_tuner.net_gdr_level);
    {
        extern int cipher_comply_audit(const char* kind, const char* json_kv);
        char kv[256];
        std::snprintf(kv, sizeof(kv),
            "{\"algo\":\"%s\",\"proto\":\"%s\",\"min_nch\":%d,"
            "\"nvlink_pairs\":%d,\"pcie_only\":%d,\"ib\":%d}",
            g_tuner.algo[0] ? g_tuner.algo : "(unset)",
            g_tuner.proto[0] ? g_tuner.proto : "(unset)",
            g_tuner.min_nchannels, g_tuner.nvlink_pairs,
            g_tuner.pcie_only, g_tuner.ib_detected);
        cipher_comply_audit("nccl_tuner", kv);
    }
    return 1;
}

extern "C" int cipher_topology_nccl_tuner_stats(CipherNcclTunerStats* out) {
    if (!out) return 0;
    *out = g_tuner;
    return 1;
}

__attribute__((constructor(116)))
static void cipher_topology_op33_autoinit() {
    // OP 33 fires whether or not CIPHER_TOPOLOGY=on; the only gate is
    // CIPHER_NCCL_TUNER=on. NCCL's first ncclCommInit* happens long after
    // C++ static init (during model.to('cuda') / ddp setup), so priority
    // 116 (after COMPLY's 115 audit-fd open) safely beats NCCL while
    // letting OP 31 capture the nccl_tuner event.
    cipher_topology_setenv_for_nccl();
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
