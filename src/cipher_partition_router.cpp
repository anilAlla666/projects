// CIPHER L2 Partition Router — implementation.

#include "cipher_partition_router.h"
#include "cipher_silicon.h"

#include <atomic>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

namespace { namespace partition_drv {
typedef int (*cuStreamCreate_fn)(void**, unsigned);
cuStreamCreate_fn g_streamCreate = nullptr;
void* g_libcuda = nullptr;
}}

namespace {

constexpr unsigned MAX_BINDINGS = 1024;
constexpr int      DEFAULT_PARTITION_COUNT = 2;   // H100 has 2 L2 partitions

struct Binding {
    int   in_use;
    void* ptr;
    int   partition;
    uint64_t hits;
};

Binding    g_bindings[MAX_BINDINGS]{};
std::mutex g_mu;

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_initialized{0};
int                   g_partition_count   = DEFAULT_PARTITION_COUNT;
int                   g_sms_per_partition = 0;

std::atomic<uint64_t> g_route_calls{0};
std::atomic<uint64_t> g_near_hits{0};
std::atomic<uint64_t> g_far_hits{0};
std::atomic<uint64_t> g_unbound_routes{0};
std::atomic<uint64_t> g_bound_routes{0};

bool env_truthy(const char* v) {
    if (!v) return false;
    return v[0] == '1' || v[0] == 't' || v[0] == 'T'
        || ((v[0] == 'o' || v[0] == 'O') && (v[1] == 'n' || v[1] == 'N'));
}

int find_locked(void* ptr) {
    for (unsigned i = 0; i < MAX_BINDINGS; ++i)
        if (g_bindings[i].in_use && g_bindings[i].ptr == ptr) return (int)i;
    return -1;
}
int find_free_locked() {
    for (unsigned i = 0; i < MAX_BINDINGS; ++i)
        if (!g_bindings[i].in_use) return (int)i;
    return -1;
}

uint64_t hash_ptr(void* p) {
    uint64_t x = (uint64_t)(uintptr_t)p;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    return x;
}

} // namespace

extern "C" int cipher_partition_router_init(void) {
    if (g_initialized.exchange(1, std::memory_order_acq_rel))
        return g_enabled.load(std::memory_order_relaxed);
    int on = env_truthy(getenv("CIPHER_PARTITION_ROUTER"));
    if (const char* s = getenv("CIPHER_PARTITION_COUNT")) {
        int v = atoi(s); if (v >= 1 && v <= 8) g_partition_count = v;
    }
    const CipherSiliconModel* sil = cipher_silicon_get();
    if (sil) g_sms_per_partition = sil->sm_count / g_partition_count;
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        fprintf(stderr,
            "[CIPHER PROUTER] init partitions=%d sms_per_partition=%d\n",
            g_partition_count, g_sms_per_partition);
    }
    return on;
}

extern "C" int cipher_partition_router_enabled(void) {
    return g_enabled.load(std::memory_order_relaxed);
}

extern "C" int cipher_partition_router_bind(void* ptr, int partition_hint) {
    if (!g_enabled.load(std::memory_order_relaxed) || !ptr) return -1;

    std::lock_guard<std::mutex> lk(g_mu);
    int idx = find_locked(ptr);
    int part;
    if (partition_hint >= 0 && partition_hint < g_partition_count) {
        part = partition_hint;
    } else {
        part = (int)(hash_ptr(ptr) % (uint64_t)g_partition_count);
    }
    if (idx < 0) {
        idx = find_free_locked();
        if (idx < 0) return -1;
        g_bindings[idx] = Binding{1, ptr, part, 0};
    } else {
        g_bindings[idx].partition = part;
    }
    return part;
}

extern "C" int cipher_partition_router_get(void* ptr) {
    if (!g_enabled.load(std::memory_order_relaxed) || !ptr) return -1;
    g_route_calls.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_mu);
    int idx = find_locked(ptr);
    if (idx < 0) {
        g_unbound_routes.fetch_add(1, std::memory_order_relaxed);
        // Hash-fall-back partition; not counted as a near-hit.
        return (int)(hash_ptr(ptr) % (uint64_t)g_partition_count);
    }
    g_bindings[idx].hits++;
    g_bound_routes.fetch_add(1, std::memory_order_relaxed);
    g_near_hits.fetch_add(1, std::memory_order_relaxed);
    return g_bindings[idx].partition;
}

// Stage 10 actuation: lazy CUstream-per-partition pool. Created on first
// stream_for() call. Each partition gets its own non-default CUstream.
namespace {
constexpr int MAX_PARTITIONS = 8;
void* g_partition_streams[MAX_PARTITIONS] = {0};
std::mutex g_streams_mu;
int        g_streams_inited = 0;

bool init_streams_locked() {
    if (g_streams_inited) return true;
    if (!partition_drv::g_libcuda) {
        partition_drv::g_libcuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!partition_drv::g_libcuda) return false;
    }
    if (!partition_drv::g_streamCreate) {
        partition_drv::g_streamCreate = (partition_drv::cuStreamCreate_fn)
            dlsym(partition_drv::g_libcuda, "cuStreamCreate");
        if (!partition_drv::g_streamCreate) return false;
    }
    int n = (g_partition_count <= MAX_PARTITIONS) ? g_partition_count : MAX_PARTITIONS;
    for (int i = 0; i < n; ++i) {
        if (g_partition_streams[i]) continue;
        void* s = nullptr;
        if (partition_drv::g_streamCreate(&s, 0) == 0 && s)
            g_partition_streams[i] = s;
    }
    g_streams_inited = 1;
    return true;
}
} // namespace

extern "C" void* cipher_partition_router_stream_for(void* ptr) {
    if (!g_enabled.load(std::memory_order_relaxed) || !ptr) return nullptr;
    int part = cipher_partition_router_get(ptr);
    if (part < 0 || part >= MAX_PARTITIONS) return nullptr;
    std::lock_guard<std::mutex> lk(g_streams_mu);
    if (!init_streams_locked()) return nullptr;
    return g_partition_streams[part];
}

extern "C" int cipher_partition_router_stats(CipherPartitionStats* out) {
    if (!out) return 0;
    out->enabled              = g_enabled.load(std::memory_order_relaxed);
    out->partition_count      = g_partition_count;
    out->sms_per_partition    = g_sms_per_partition;
    out->route_calls          = g_route_calls.load(std::memory_order_relaxed);
    out->near_partition_hits  = g_near_hits.load(std::memory_order_relaxed);
    out->far_partition_hits   = g_far_hits.load(std::memory_order_relaxed);
    out->unbound_routes       = g_unbound_routes.load(std::memory_order_relaxed);
    out->bound_routes         = g_bound_routes.load(std::memory_order_relaxed);
    return 1;
}

extern "C" void cipher_partition_router_report(void) {
    CipherPartitionStats s{};
    cipher_partition_router_stats(&s);
    FILE* f = fopen("/tmp/cipher_partition_router_report.json", "w");
    if (!f) return;
    fprintf(f,
        "{\"enabled\":%d,\"partition_count\":%d,\"sms_per_partition\":%d,"
        "\"route_calls\":%llu,\"near_partition_hits\":%llu,"
        "\"far_partition_hits\":%llu,\"unbound_routes\":%llu,"
        "\"bound_routes\":%llu}\n",
        s.enabled, s.partition_count, s.sms_per_partition,
        (unsigned long long)s.route_calls,
        (unsigned long long)s.near_partition_hits,
        (unsigned long long)s.far_partition_hits,
        (unsigned long long)s.unbound_routes,
        (unsigned long long)s.bound_routes);
    fclose(f);
}

__attribute__((constructor(110)))
static void cipher_partition_router_autoinit() { cipher_partition_router_init(); }
