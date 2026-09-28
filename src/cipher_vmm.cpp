// CIPHER VMM Pool — implementation. dlopen-based libcuda binding so we don't
// add a hard link-time dependency on -lcuda.

#include "cipher_vmm.h"
#include "cipher_silicon.h"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <atomic>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {

using CUdeviceptr = unsigned long long;
using CUmemGenericAllocationHandle = unsigned long long;

enum {
    CU_MEM_LOCATION_TYPE_DEVICE       = 1,
    CU_MEM_ALLOCATION_TYPE_PINNED     = 1,
    CU_MEM_ALLOCATION_GRANULARITY_MIN = 0,
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3,
};

struct CUmemLocation { int type; int id; };
struct CUmemAllocationProp {
    int type;
    int requestedHandleTypes;
    CUmemLocation location;
    void* win32HandleMetaData;
    unsigned long long allocFlags[4];
};
struct CUmemAccessDesc {
    CUmemLocation location;
    int           flags;
};

typedef int (*pf_cuInit)(unsigned);
typedef int (*pf_cuDeviceGet)(int*, int);
typedef int (*pf_cuMemGetAllocationGranularity)(size_t*, const CUmemAllocationProp*, int);
typedef int (*pf_cuMemAddressReserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
typedef int (*pf_cuMemAddressFree)(CUdeviceptr, size_t);
typedef int (*pf_cuMemCreate)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
typedef int (*pf_cuMemRelease)(CUmemGenericAllocationHandle);
typedef int (*pf_cuMemMap)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
typedef int (*pf_cuMemUnmap)(CUdeviceptr, size_t);
typedef int (*pf_cuMemSetAccess)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);

struct DriverApi {
    void* lib = nullptr;
    pf_cuInit                            cuInit = nullptr;
    pf_cuDeviceGet                       cuDeviceGet = nullptr;
    pf_cuMemGetAllocationGranularity     cuMemGetAllocationGranularity = nullptr;
    pf_cuMemAddressReserve               cuMemAddressReserve = nullptr;
    pf_cuMemAddressFree                  cuMemAddressFree = nullptr;
    pf_cuMemCreate                       cuMemCreate = nullptr;
    pf_cuMemRelease                      cuMemRelease = nullptr;
    pf_cuMemMap                          cuMemMap = nullptr;
    pf_cuMemUnmap                        cuMemUnmap = nullptr;
    pf_cuMemSetAccess                    cuMemSetAccess = nullptr;
    bool ok = false;
};

#define MAX_VMM_REGIONS 256

struct Region {
    CUdeviceptr                   ptr;        // 0 = empty
    size_t                        bytes;
    CUmemGenericAllocationHandle  handle;
    int                           in_use;
};

struct Engine {
    std::mutex                       mu;
    DriverApi                        api;
    Region                           regions[MAX_VMM_REGIONS]{};
    int                              gpu_id      = 0;
    size_t                           granularity = 0;
    CUmemAllocationProp              prop{};
    CUmemAccessDesc                  access{};
    std::atomic<int>                 enabled{0};
    std::atomic<int>                 initialized{0};
    std::atomic<size_t>              reserved_bytes{0};
    std::atomic<size_t>              committed_bytes{0};
    std::atomic<size_t>              live_bytes{0};
    std::atomic<size_t>              peak_live_bytes{0};
    std::atomic<uint64_t>            alloc_calls{0};
    std::atomic<uint64_t>            free_calls{0};
    std::atomic<uint64_t>            swap_calls{0};
    std::atomic<uint64_t>            map_calls{0};
    std::atomic<uint64_t>            unmap_calls{0};
};

Engine g_engine;

bool env_truthy(const char* v) {
    if (!v) return false;
    return (v[0] == '1') || (v[0] == 't') || (v[0] == 'T')
        || ((v[0] == 'o' || v[0] == 'O') && (v[1] == 'n' || v[1] == 'N'));
}

bool resolve_driver(DriverApi& a) {
    a.lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!a.lib) return false;
    a.cuInit                          = (pf_cuInit)dlsym(a.lib, "cuInit");
    a.cuDeviceGet                     = (pf_cuDeviceGet)dlsym(a.lib, "cuDeviceGet");
    a.cuMemGetAllocationGranularity   = (pf_cuMemGetAllocationGranularity)dlsym(a.lib, "cuMemGetAllocationGranularity");
    a.cuMemAddressReserve             = (pf_cuMemAddressReserve)dlsym(a.lib, "cuMemAddressReserve");
    a.cuMemAddressFree                = (pf_cuMemAddressFree)dlsym(a.lib, "cuMemAddressFree");
    a.cuMemCreate                     = (pf_cuMemCreate)dlsym(a.lib, "cuMemCreate");
    a.cuMemRelease                    = (pf_cuMemRelease)dlsym(a.lib, "cuMemRelease");
    a.cuMemMap                        = (pf_cuMemMap)dlsym(a.lib, "cuMemMap");
    a.cuMemUnmap                      = (pf_cuMemUnmap)dlsym(a.lib, "cuMemUnmap");
    a.cuMemSetAccess                  = (pf_cuMemSetAccess)dlsym(a.lib, "cuMemSetAccess");
    a.ok = (a.cuInit && a.cuMemGetAllocationGranularity && a.cuMemAddressReserve
            && a.cuMemAddressFree && a.cuMemCreate && a.cuMemRelease
            && a.cuMemMap && a.cuMemUnmap && a.cuMemSetAccess);
    return a.ok;
}

size_t round_up(size_t v, size_t g) {
    return (v + g - 1) / g * g;
}

int find_slot_locked(CUdeviceptr ptr) {
    for (int i = 0; i < MAX_VMM_REGIONS; ++i) {
        if (g_engine.regions[i].in_use && g_engine.regions[i].ptr == ptr) return i;
    }
    return -1;
}
int find_free_slot_locked() {
    for (int i = 0; i < MAX_VMM_REGIONS; ++i) {
        if (!g_engine.regions[i].in_use) return i;
    }
    return -1;
}

} // namespace

extern "C" int cipher_vmm_init(void) {
    if (g_engine.initialized.load(std::memory_order_acquire)) return g_engine.enabled.load();

    if (!env_truthy(getenv("CIPHER_VMM"))) {
        g_engine.initialized.store(1, std::memory_order_release);
        return 0;
    }

    if (!resolve_driver(g_engine.api)) {
        g_engine.initialized.store(1, std::memory_order_release);
        return 0;
    }

    // Pick device 0; later stages may parameterize.
    g_engine.api.cuInit(0);
    int dev = 0;
    g_engine.api.cuDeviceGet(&dev, 0);
    g_engine.gpu_id = dev;

    g_engine.prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    g_engine.prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
    g_engine.prop.location.id          = dev;
    g_engine.access.location           = g_engine.prop.location;
    g_engine.access.flags              = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    size_t gran = 0;
    if (g_engine.api.cuMemGetAllocationGranularity(&gran, &g_engine.prop,
                                                    CU_MEM_ALLOCATION_GRANULARITY_MIN) != 0) {
        return 0;
    }
    g_engine.granularity = (gran > 0) ? gran : 2 * 1024 * 1024;

    g_engine.enabled.store(1, std::memory_order_release);
    g_engine.initialized.store(1, std::memory_order_release);

    if (getenv("CIPHER_VMM_VERBOSE")) {
        fprintf(stderr,
            "[CIPHER VMM] initialized: gpu=%d granularity=%zu bytes (%.1f KB), "
            "max_regions=%d\n",
            g_engine.gpu_id, g_engine.granularity,
            g_engine.granularity / 1024.0, MAX_VMM_REGIONS);
    }
    return 1;
}

extern "C" int cipher_vmm_enabled(void) {
    return g_engine.enabled.load(std::memory_order_relaxed);
}

extern "C" unsigned long long cipher_vmm_alloc(size_t bytes) {
    if (!g_engine.enabled.load(std::memory_order_relaxed) || bytes == 0) return 0;
    g_engine.alloc_calls.fetch_add(1, std::memory_order_relaxed);

    size_t aligned = round_up(bytes, g_engine.granularity);

    std::lock_guard<std::mutex> lk(g_engine.mu);
    int slot = find_free_slot_locked();
    if (slot < 0) return 0;

    CUdeviceptr ptr = 0;
    if (g_engine.api.cuMemAddressReserve(&ptr, aligned, g_engine.granularity, 0, 0) != 0
        || ptr == 0) return 0;

    CUmemGenericAllocationHandle h = 0;
    if (g_engine.api.cuMemCreate(&h, aligned, &g_engine.prop, 0) != 0) {
        g_engine.api.cuMemAddressFree(ptr, aligned);
        return 0;
    }
    if (g_engine.api.cuMemMap(ptr, aligned, 0, h, 0) != 0) {
        g_engine.api.cuMemRelease(h);
        g_engine.api.cuMemAddressFree(ptr, aligned);
        return 0;
    }
    if (g_engine.api.cuMemSetAccess(ptr, aligned, &g_engine.access, 1) != 0) {
        g_engine.api.cuMemUnmap(ptr, aligned);
        g_engine.api.cuMemRelease(h);
        g_engine.api.cuMemAddressFree(ptr, aligned);
        return 0;
    }

    g_engine.regions[slot] = Region{ptr, aligned, h, 1};
    g_engine.reserved_bytes.fetch_add(aligned, std::memory_order_relaxed);
    g_engine.committed_bytes.fetch_add(aligned, std::memory_order_relaxed);
    size_t live = g_engine.live_bytes.fetch_add(aligned, std::memory_order_relaxed) + aligned;
    size_t prev_peak = g_engine.peak_live_bytes.load(std::memory_order_relaxed);
    while (live > prev_peak
           && !g_engine.peak_live_bytes.compare_exchange_weak(
                  prev_peak, live, std::memory_order_relaxed)) {}
    g_engine.map_calls.fetch_add(1, std::memory_order_relaxed);
    return ptr;
}

extern "C" int cipher_vmm_free(unsigned long long ptr) {
    if (!g_engine.enabled.load(std::memory_order_relaxed) || ptr == 0) return 0;
    g_engine.free_calls.fetch_add(1, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lk(g_engine.mu);
    int slot = find_slot_locked(ptr);
    if (slot < 0) return 0;

    Region r = g_engine.regions[slot];
    g_engine.api.cuMemUnmap(r.ptr, r.bytes);
    g_engine.api.cuMemRelease(r.handle);
    g_engine.api.cuMemAddressFree(r.ptr, r.bytes);
    g_engine.regions[slot] = Region{};

    g_engine.reserved_bytes.fetch_sub(r.bytes, std::memory_order_relaxed);
    g_engine.committed_bytes.fetch_sub(r.bytes, std::memory_order_relaxed);
    g_engine.live_bytes.fetch_sub(r.bytes, std::memory_order_relaxed);
    g_engine.unmap_calls.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

extern "C" int cipher_vmm_swap(unsigned long long dst_ptr, unsigned long long src_ptr) {
    if (!g_engine.enabled.load(std::memory_order_relaxed)) return 0;
    if (dst_ptr == 0 || src_ptr == 0 || dst_ptr == src_ptr) return 0;
    g_engine.swap_calls.fetch_add(1, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lk(g_engine.mu);
    int sd = find_slot_locked(dst_ptr);
    int ss = find_slot_locked(src_ptr);
    if (sd < 0 || ss < 0) return 0;
    if (g_engine.regions[sd].bytes != g_engine.regions[ss].bytes) return 0;

    Region D = g_engine.regions[sd];
    Region S = g_engine.regions[ss];

    // Unmap dst, map src's handle into dst's VA, leave src VA alone (caller
    // can free src after the swap if desired).
    if (g_engine.api.cuMemUnmap(D.ptr, D.bytes) != 0) return 0;
    g_engine.unmap_calls.fetch_add(1, std::memory_order_relaxed);
    if (g_engine.api.cuMemMap(D.ptr, D.bytes, 0, S.handle, 0) != 0) return 0;
    g_engine.map_calls.fetch_add(1, std::memory_order_relaxed);
    g_engine.api.cuMemSetAccess(D.ptr, D.bytes, &g_engine.access, 1);

    // Old D handle is orphaned; release it. Src now backs both D's VA and
    // its own VA; the slot at sd inherits the new handle.
    g_engine.api.cuMemRelease(D.handle);
    g_engine.regions[sd].handle = S.handle;
    return 1;
}

extern "C" int cipher_vmm_stats(CipherVmmStats* out) {
    if (!out) return 0;
    out->enabled          = g_engine.enabled.load(std::memory_order_relaxed);
    out->driver_available = g_engine.api.ok ? 1 : 0;
    out->gpu_id           = g_engine.gpu_id;
    out->granularity      = g_engine.granularity;
    out->reserved_bytes   = g_engine.reserved_bytes.load(std::memory_order_relaxed);
    out->committed_bytes  = g_engine.committed_bytes.load(std::memory_order_relaxed);
    out->live_bytes       = g_engine.live_bytes.load(std::memory_order_relaxed);
    out->peak_live_bytes  = g_engine.peak_live_bytes.load(std::memory_order_relaxed);
    out->alloc_calls      = g_engine.alloc_calls.load(std::memory_order_relaxed);
    out->free_calls       = g_engine.free_calls.load(std::memory_order_relaxed);
    out->swap_calls       = g_engine.swap_calls.load(std::memory_order_relaxed);
    out->map_calls        = g_engine.map_calls.load(std::memory_order_relaxed);
    out->unmap_calls      = g_engine.unmap_calls.load(std::memory_order_relaxed);
    return 1;
}

extern "C" void cipher_vmm_report(void) {
    CipherVmmStats s{};
    cipher_vmm_stats(&s);
    FILE* f = fopen("/tmp/cipher_vmm_report.json", "w");
    if (!f) return;
    fprintf(f,
        "{\"enabled\":%d,\"driver_available\":%d,\"gpu_id\":%d,"
        "\"granularity\":%zu,\"reserved_bytes\":%zu,\"committed_bytes\":%zu,"
        "\"live_bytes\":%zu,\"peak_live_bytes\":%zu,"
        "\"alloc_calls\":%llu,\"free_calls\":%llu,\"swap_calls\":%llu,"
        "\"map_calls\":%llu,\"unmap_calls\":%llu}\n",
        s.enabled, s.driver_available, s.gpu_id,
        s.granularity, s.reserved_bytes, s.committed_bytes,
        s.live_bytes, s.peak_live_bytes,
        (unsigned long long)s.alloc_calls,
        (unsigned long long)s.free_calls,
        (unsigned long long)s.swap_calls,
        (unsigned long long)s.map_calls,
        (unsigned long long)s.unmap_calls);
    fclose(f);
}

__attribute__((constructor(104)))
static void cipher_vmm_autoinit() {
    cipher_vmm_init();
}
