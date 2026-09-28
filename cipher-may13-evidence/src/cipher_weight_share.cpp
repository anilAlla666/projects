// CIPHER weight sharing via CUDA IPC + /dev/shm coordination.
//
// Protocol per slot:
//   /dev/shm/cipher_ws_<gpu>_slot<N>  — fixed-size 256 byte file:
//     uint32_t magic    = 0xC1F457E5
//     uint32_t version  = 1
//     uint64_t content_hash
//     uint64_t bytes
//     uint8_t  ipc_handle[64]
//     int32_t  refcount  (atomic via flock)
//     int32_t  ready     (1 once exported)
//
// Lookup direction:
//   Process A (publisher): cudaMalloc'd weight, content stable → cipher_weight_share_export()
//     - Computes a slot index from content_hash mod MAX_SLOTS
//     - Acquires flock on the shm file
//     - If ready=0: cuIpcGetMemHandle(); store handle, set ready=1, refcount=1
//     - If ready=1 and matches: refcount++
//     - Inserts (original_ptr → local shared_ptr=original_ptr) in g_local_map
//
//   Process B (subscriber): observes a different cudaMalloc'd weight
//     with the SAME content_hash and bytes
//     - Same slot lookup; if ready=1, cuIpcOpenMemHandle(); refcount++
//     - Inserts (original_ptr → shared_ptr) in g_local_map.  When the
//       cublasGemmEx hook later sees this pointer, it substitutes
//       shared_ptr.

#include "cipher_weight_share.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <mutex>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

constexpr uint32_t WS_MAGIC = 0xC1F457E5u;
constexpr int      MAX_SLOTS = 1024;
constexpr size_t   SLOT_BYTES = 256;
constexpr int      IPC_HANDLE_SIZE = 64;

struct WSHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t content_hash;     // hash of weight CONTENT (not base)
    uint64_t bytes;            // size of the WEIGHT region
    uint64_t base_bytes;       // total size of the underlying allocation
    uint64_t offset_in_base;   // weight_ptr − base_ptr in publisher's VA
    uint8_t  ipc_handle[IPC_HANDLE_SIZE]; // for the BASE allocation
    int32_t  refcount;
    int32_t  ready;
    int32_t  publisher_pid;
    uint8_t  pad[256 - 4 - 4 - 8 - 8 - 8 - 8 - IPC_HANDLE_SIZE - 4 - 4 - 4];
};
static_assert(sizeof(WSHeader) == SLOT_BYTES, "WSHeader must be exactly SLOT_BYTES");

// CUDA driver API resolution.
typedef int (*pf_cuIpcGetMemHandle)(void*, void*);   // CUipcMemHandle*, CUdeviceptr
typedef int (*pf_cuIpcOpenMemHandle)(void**, void*, unsigned int); // out CUdeviceptr*, handle, flags
typedef int (*pf_cuIpcCloseMemHandle)(void*);        // CUdeviceptr
typedef int (*pf_cuMemGetAddressRange)(void**, size_t*, void*); // out base, out size, ptr
typedef int (*pf_cuCtxGetDevice)(int*);
typedef int (*pf_cuMemAlloc)(void**, size_t);

constexpr unsigned int CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1;

struct CudaApi {
    void* libcuda = nullptr;
    pf_cuIpcGetMemHandle    getHandle   = nullptr;
    pf_cuIpcOpenMemHandle   openHandle  = nullptr;
    pf_cuIpcCloseMemHandle  closeHandle = nullptr;
    pf_cuMemGetAddressRange getRange    = nullptr;
    pf_cuCtxGetDevice       getDevice   = nullptr;
    bool ok = false;
};

CudaApi g_api;

bool resolve_api() {
    if (g_api.ok) return true;
    g_api.libcuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!g_api.libcuda) return false;
    g_api.getHandle = (pf_cuIpcGetMemHandle)
        dlsym(g_api.libcuda, "cuIpcGetMemHandle");
    g_api.openHandle = (pf_cuIpcOpenMemHandle)
        dlsym(g_api.libcuda, "cuIpcOpenMemHandle_v2");
    if (!g_api.openHandle)
        g_api.openHandle = (pf_cuIpcOpenMemHandle)
            dlsym(g_api.libcuda, "cuIpcOpenMemHandle");
    g_api.closeHandle = (pf_cuIpcCloseMemHandle)
        dlsym(g_api.libcuda, "cuIpcCloseMemHandle");
    g_api.getRange = (pf_cuMemGetAddressRange)
        dlsym(g_api.libcuda, "cuMemGetAddressRange_v2");
    if (!g_api.getRange)
        g_api.getRange = (pf_cuMemGetAddressRange)
            dlsym(g_api.libcuda, "cuMemGetAddressRange");
    g_api.getDevice = (pf_cuCtxGetDevice)
        dlsym(g_api.libcuda, "cuCtxGetDevice");
    g_api.ok = (g_api.getHandle && g_api.openHandle &&
                g_api.closeHandle && g_api.getRange && g_api.getDevice);
    return g_api.ok;
}

// Local process state.
std::atomic<int> g_enabled{0};
std::atomic<int> g_initialized{0};
std::atomic<int> g_active_imports{0};
std::atomic<int> g_active_exports{0};
std::atomic<uint64_t> g_substitutions{0};

struct LocalEntry {
    void*    original_ptr  = nullptr;
    void*    shared_ptr    = nullptr;
    size_t   bytes         = 0;
    uint64_t content_hash  = 0;
    int      slot_id       = -1;
    bool     is_publisher  = false;
};

constexpr int MAX_LOCAL = 2048;
LocalEntry g_local[MAX_LOCAL];
std::mutex g_local_mu;

int find_local_locked(void* p) {
    for (int i = 0; i < MAX_LOCAL; ++i) {
        if (g_local[i].original_ptr == p) return i;
    }
    return -1;
}

int alloc_local_locked() {
    for (int i = 0; i < MAX_LOCAL; ++i) {
        if (g_local[i].original_ptr == nullptr) return i;
    }
    return -1;
}

int gpu_id() {
    int dev = 0;
    if (g_api.getDevice) g_api.getDevice(&dev);
    return dev;
}

bool slot_path(int gpu, int slot, char* out, size_t cap) {
    int n = snprintf(out, cap, "/dev/shm/cipher_ws_g%d_s%d", gpu, slot);
    return n > 0 && (size_t)n < cap;
}

// Open or create the slot file, mmap to header.  Returns fd and header
// pointer.  Caller flocks.
int open_slot(int gpu, int slot, WSHeader** out) {
    char path[128];
    if (!slot_path(gpu, slot, path, sizeof(path))) return -1;
    int fd = open(path, O_RDWR | O_CREAT, 0666);
    if (fd < 0) return -1;
    if (ftruncate(fd, SLOT_BYTES) != 0) { close(fd); return -1; }
    void* m = mmap(nullptr, SLOT_BYTES, PROT_READ | PROT_WRITE,
                   MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { close(fd); return -1; }
    *out = (WSHeader*)m;
    return fd;
}

void close_slot(int fd, WSHeader* h) {
    if (h) munmap(h, SLOT_BYTES);
    if (fd >= 0) close(fd);
}

// Slot index from content hash (linear-probe on collisions).
int slot_index_for_hash(uint64_t content_hash) {
    return (int)(content_hash % MAX_SLOTS);
}

bool env_truthy(const char* v) {
    if (!v) return false;
    return v[0] == '1' || v[0] == 'o' || v[0] == 'O' ||
           v[0] == 't' || v[0] == 'T';
}

} // namespace

extern "C" int cipher_weight_share_init(void) {
    if (g_initialized.exchange(1, std::memory_order_acq_rel))
        return g_enabled.load(std::memory_order_relaxed);
    const char* env = getenv("CIPHER_WEIGHT_SHARE");
    int on = env_truthy(env);
    fprintf(stderr, "[CIPHER WS] init invoked: env='%s' on=%d\n",
        env ? env : "(null)", on);
    if (!on) {
        g_enabled.store(0, std::memory_order_release);
        return 0;
    }
    if (!resolve_api()) {
        fprintf(stderr,
            "[CIPHER WS] init: libcuda IPC symbols not resolved — disabled\n");
        g_enabled.store(0, std::memory_order_release);
        return 0;
    }
    g_enabled.store(1, std::memory_order_release);
    fprintf(stderr,
        "[CIPHER WS] init: enabled (max_slots=%d, slot_bytes=%zu)\n",
        MAX_SLOTS, SLOT_BYTES);
    return 1;
}

extern "C" int cipher_weight_share_enabled(void) {
    return g_enabled.load(std::memory_order_relaxed);
}

extern "C" int cipher_weight_share_active_count(void) {
    return g_active_imports.load(std::memory_order_relaxed) +
           g_active_exports.load(std::memory_order_relaxed);
}

namespace {
// FNV-1a 64-bit over 128 bytes from device memory.
constexpr size_t HASH_BYTES = 128;
uint64_t fnv1a64(const uint8_t* p, size_t n) {
    constexpr uint64_t OFF = 0xcbf29ce484222325ULL;
    constexpr uint64_t PRM = 0x100000001b3ULL;
    uint64_t h = OFF;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= PRM; }
    return h;
}

// Per-process observation table — separate from FP8's, so weight_share
// can run even when FP8 is off.
struct ObsEntry {
    void*    key;          // original device pointer
    size_t   bytes;
    uint64_t content_hash;
    int      hits;
    int      exported;     // 1 once cipher_weight_share_export succeeded
};
constexpr int MAX_OBS = 4096;
ObsEntry g_obs[MAX_OBS];
std::mutex g_obs_mu;
int find_obs_locked(void* p) {
    for (int i = 0; i < MAX_OBS; ++i)
        if (g_obs[i].key == p) return i;
    return -1;
}
int alloc_obs_locked() {
    for (int i = 0; i < MAX_OBS; ++i)
        if (g_obs[i].key == nullptr) return i;
    return -1;
}
} // namespace

extern "C" void cipher_weight_share_observe(void* dev_ptr, size_t bytes) {
    static std::atomic<uint64_t> s_obs{0};
    static std::atomic<uint64_t> s_obs_skipped{0};
    if (!g_enabled.load(std::memory_order_relaxed) || !dev_ptr ||
        bytes < HASH_BYTES) {
        if ((s_obs_skipped.fetch_add(1, std::memory_order_relaxed) & 0xFFFF) == 0)
            fprintf(stderr, "[CIPHER WS] observe skipped (en=%d ptr=%p bytes=%zu)\n",
                g_enabled.load(std::memory_order_relaxed), dev_ptr, bytes);
        return;
    }
    uint64_t n = s_obs.fetch_add(1, std::memory_order_relaxed);
    if (n < 5 || (n & 0x3FF) == 0)
        fprintf(stderr, "[CIPHER WS] observe #%lu ptr=%p bytes=%zu\n",
            (unsigned long)n, dev_ptr, bytes);

    // Fast path: already in obs table?
    int idx;
    {
        std::lock_guard<std::mutex> lk(g_obs_mu);
        idx = find_obs_locked(dev_ptr);
        if (idx >= 0 && g_obs[idx].exported) {
            g_obs[idx].hits++;
            return;
        }
    }

    // Sync hash compute (~1us D2H).
    uint8_t buf[HASH_BYTES];
    if (cudaMemcpy(buf, dev_ptr, HASH_BYTES, cudaMemcpyDeviceToHost) !=
            cudaSuccess)
        return;
    uint64_t hash = fnv1a64(buf, HASH_BYTES);
    if (hash == 0) hash = 1;  // keep 0 sentinel reserved

    int trigger_export = 0;
    int new_hits = 0;
    {
        std::lock_guard<std::mutex> lk(g_obs_mu);
        if (idx < 0) {
            idx = find_obs_locked(dev_ptr);
            if (idx < 0) idx = alloc_obs_locked();
            if (idx < 0) return;
            g_obs[idx] = ObsEntry{dev_ptr, bytes, hash, 1, 0};
            new_hits = 1;
        } else {
            ObsEntry& e = g_obs[idx];
            if (e.bytes != bytes) {
                // shape changed under us — reset
                e = ObsEntry{dev_ptr, bytes, hash, 1, 0};
            } else if (e.content_hash != hash) {
                // content drift — bnb-style scratch buffer, never share
                e.hits = 0;
                e.content_hash = 0; // mark transient
            } else if (e.content_hash != 0) {
                e.hits++;
                new_hits = e.hits;
                if (e.hits == 2 && !e.exported) {
                    trigger_export = 1;
                }
            }
        }
    }
    if (n < 5 || (n & 0x3FF) == 0) {
        fprintf(stderr,
            "[CIPHER WS] observe path: ptr=%p new_hits=%d trigger_export=%d\n",
            dev_ptr, new_hits, trigger_export);
    }
    if (trigger_export) {
        fprintf(stderr,
            "[CIPHER WS] TRIGGER export: ptr=%p bytes=%zu hash=%016lx\n",
            dev_ptr, bytes, (unsigned long)hash);
        int slot = cipher_weight_share_export(dev_ptr, bytes, hash);
        std::lock_guard<std::mutex> lk(g_obs_mu);
        int j = find_obs_locked(dev_ptr);
        if (j >= 0) g_obs[j].exported = (slot >= 0) ? 1 : -1;
        fprintf(stderr,
            "[CIPHER WS] export returned slot=%d -> exported=%d\n",
            slot, (j >= 0) ? g_obs[j].exported : -99);
    }
}

extern "C" int cipher_weight_share_export(void* dev_ptr, size_t bytes,
                                          uint64_t content_hash) {
    if (!g_enabled.load(std::memory_order_relaxed) || !dev_ptr ||
        bytes == 0 || content_hash == 0)
        return -1;
    if (!resolve_api()) return -1;

    // Already mapped locally?
    {
        std::lock_guard<std::mutex> lk(g_local_mu);
        int li = find_local_locked(dev_ptr);
        if (li >= 0) return g_local[li].slot_id;
    }

    int slot = slot_index_for_hash(content_hash);
    int gpu  = gpu_id();
    for (int probe = 0; probe < 8; ++probe) {
        int s = (slot + probe) % MAX_SLOTS;
        WSHeader* h = nullptr;
        int fd = open_slot(gpu, s, &h);
        if (fd < 0) continue;

        if (flock(fd, LOCK_EX) != 0) { close_slot(fd, h); continue; }

        // Existing publisher?
        if (h->magic == WS_MAGIC && h->ready &&
            h->content_hash == content_hash && h->bytes == bytes) {
            // SUBSCRIBE.  Open the IPC handle (which references the BASE
            // of the publisher's allocation), then add the saved offset.
            void* base_mapped = nullptr;
            // Try with the standard flag first; some driver versions also
            // accept flags=0.
            int rc = g_api.openHandle(&base_mapped, h->ipc_handle,
                                      CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
            if (rc != 0) {
                // Retry with flags=0 (some older paths).
                rc = g_api.openHandle(&base_mapped, h->ipc_handle, 0);
                if (rc != 0) {
                    fprintf(stderr,
                        "[CIPHER WS] cuIpcOpenMemHandle failed rc=%d "
                        "(both flag variants tried)\n", rc);
                }
            }
            if (rc == 0 && base_mapped) {
                void* shared = (void*)((char*)base_mapped + h->offset_in_base);
                h->refcount++;
                std::lock_guard<std::mutex> lk(g_local_mu);
                int li = alloc_local_locked();
                if (li >= 0) {
                    g_local[li] = LocalEntry{};
                    g_local[li].original_ptr = dev_ptr;
                    g_local[li].shared_ptr   = shared;
                    g_local[li].bytes        = bytes;
                    g_local[li].content_hash = content_hash;
                    g_local[li].slot_id      = s;
                    g_local[li].is_publisher = false;
                    g_active_imports.fetch_add(1, std::memory_order_relaxed);
                    flock(fd, LOCK_UN); close_slot(fd, h);
                    fprintf(stderr,
                        "[CIPHER WS] import slot=%d hash=%016lx bytes=%zu "
                        "orig=%p shared=%p (base=%p+%lu) refcount=%d\n",
                        s, (unsigned long)content_hash, bytes,
                        dev_ptr, shared, base_mapped,
                        (unsigned long)h->offset_in_base, h->refcount);
                    return s;
                }
            } else {
                fprintf(stderr,
                    "[CIPHER WS] cuIpcOpenMemHandle failed rc=%d\n", rc);
            }
            flock(fd, LOCK_UN); close_slot(fd, h);
            continue;
        }

        // Slot collision with different content?
        if (h->magic == WS_MAGIC && h->ready &&
            (h->content_hash != content_hash || h->bytes != bytes)) {
            flock(fd, LOCK_UN); close_slot(fd, h);
            continue; // try next probe
        }

        // Empty / stale slot — PUBLISH.  Get the BASE of the underlying
        // allocation; that's what cuIpcGetMemHandle requires.  Save the
        // offset so subscribers can reproduce the weight pointer.
        void*  base = nullptr;
        size_t base_bytes = 0;
        if (g_api.getRange(&base, &base_bytes, dev_ptr) != 0 || !base) {
            flock(fd, LOCK_UN); close_slot(fd, h);
            return -1;
        }
        uint64_t offset = (uint64_t)((char*)dev_ptr - (char*)base);
        if (offset + bytes > base_bytes) {
            flock(fd, LOCK_UN); close_slot(fd, h);
            return -1;
        }

        uint8_t handle[IPC_HANDLE_SIZE];
        memset(handle, 0, IPC_HANDLE_SIZE);
        int rc = g_api.getHandle(handle, base);  // export the BASE
        if (rc != 0) {
            flock(fd, LOCK_UN); close_slot(fd, h);
            // cuIpcGetMemHandle returned an error — typically because the
            // allocator (e.g. PyTorch's caching allocator) didn't allocate
            // this region with cudaMalloc semantics that support IPC.
            static int s_warn = 0;
            if (s_warn++ < 3) {
                fprintf(stderr,
                    "[CIPHER WS] cuIpcGetMemHandle(base=%p, base_bytes=%zu) "
                    "rc=%d — allocator likely doesn't support IPC export\n",
                    base, base_bytes, rc);
            }
            return -1;
        }
        h->magic           = WS_MAGIC;
        h->version         = 1;
        h->content_hash    = content_hash;
        h->bytes           = bytes;
        h->base_bytes      = base_bytes;
        h->offset_in_base  = offset;
        memcpy(h->ipc_handle, handle, IPC_HANDLE_SIZE);
        h->refcount        = 1;
        h->ready           = 1;
        h->publisher_pid   = (int)getpid();
        msync(h, SLOT_BYTES, MS_SYNC);

        std::lock_guard<std::mutex> lk(g_local_mu);
        int li = alloc_local_locked();
        if (li >= 0) {
            g_local[li] = LocalEntry{};
            g_local[li].original_ptr = dev_ptr;
            g_local[li].shared_ptr   = dev_ptr;   // publisher uses original
            g_local[li].bytes        = bytes;
            g_local[li].content_hash = content_hash;
            g_local[li].slot_id      = s;
            g_local[li].is_publisher = true;
            g_active_exports.fetch_add(1, std::memory_order_relaxed);
        }
        flock(fd, LOCK_UN); close_slot(fd, h);
        fprintf(stderr,
            "[CIPHER WS] export slot=%d hash=%016lx bytes=%zu ptr=%p "
            "base=%p base_bytes=%zu offset=%lu\n",
            s, (unsigned long)content_hash, bytes, dev_ptr, base, base_bytes,
            (unsigned long)offset);
        return s;
    }
    return -1;
}

extern "C" void* cipher_weight_share_lookup(void* original_ptr) {
    if (!g_enabled.load(std::memory_order_relaxed) || !original_ptr)
        return nullptr;
    std::lock_guard<std::mutex> lk(g_local_mu);
    int li = find_local_locked(original_ptr);
    if (li < 0) return nullptr;
    if (g_local[li].is_publisher) return nullptr; // no substitution needed
    g_substitutions.fetch_add(1, std::memory_order_relaxed);
    return g_local[li].shared_ptr;
}

extern "C" void cipher_weight_share_shutdown(void) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;
    std::lock_guard<std::mutex> lk(g_local_mu);
    int gpu = gpu_id();
    int n_imports = 0, n_exports = 0;
    for (int i = 0; i < MAX_LOCAL; ++i) {
        if (g_local[i].original_ptr == nullptr) continue;
        if (g_local[i].is_publisher) {
            n_exports++;
        } else if (g_local[i].shared_ptr) {
            if (g_api.closeHandle) g_api.closeHandle(g_local[i].shared_ptr);
            n_imports++;
        }
        // Decrement refcount under flock.
        WSHeader* h = nullptr;
        int fd = open_slot(gpu, g_local[i].slot_id, &h);
        if (fd >= 0) {
            if (flock(fd, LOCK_EX) == 0) {
                if (h->magic == WS_MAGIC) {
                    h->refcount--;
                    if (h->refcount <= 0) {
                        h->ready = 0;
                    }
                    msync(h, SLOT_BYTES, MS_SYNC);
                }
                flock(fd, LOCK_UN);
            }
            close_slot(fd, h);
        }
        g_local[i] = LocalEntry{};
    }
    fprintf(stderr,
        "[CIPHER WS] shutdown: closed %d imports, %d exports, "
        "subs_used=%llu\n",
        n_imports, n_exports,
        (unsigned long long)g_substitutions.load(std::memory_order_relaxed));
}

__attribute__((constructor(113)))
static void cipher_weight_share_autoinit() { cipher_weight_share_init(); }

__attribute__((destructor))
static void cipher_weight_share_autoshutdown() { cipher_weight_share_shutdown(); }
