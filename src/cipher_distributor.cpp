// CIPHER multi-tenant auto-distributor — implementation.
// See include/cipher_distributor.h for the design.

#include "cipher_distributor.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <pthread.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

constexpr uint32_t DISTRIBUTOR_MAGIC = 0xD15C0DE1u;
constexpr int      MAX_TENANTS       = 256;
constexpr int      MAX_GPUS          = 16;
constexpr int      LOAD_HISTORY_LEN  = 32;     // last 32 30-second buckets
constexpr uint64_t POLL_PERIOD_NS    = 50ULL * 1000 * 1000; // 50 ms
constexpr uint64_t WINDOW_NS_DEFAULT = 30ULL * 1000 * 1000 * 1000; // 30 s
constexpr float    OVERLOAD_FACTOR   = 1.5f;   // tenant > 1.5× p95 → overloaded

uint64_t window_ns() {
    static uint64_t s_w = 0;
    if (s_w) return s_w;
    const char* e = getenv("CIPHER_DIST_WINDOW_MS");
    s_w = e ? (uint64_t)atoi(e) * 1000 * 1000ULL : WINDOW_NS_DEFAULT;
    return s_w;
}

struct TenantSlot {
    std::atomic<uint32_t> tenant_id_hash;   // 0 = empty
    std::atomic<uint32_t> active;           // 1 = registered
    std::atomic<int>      gpu_id;
    std::atomic<uint64_t> kernel_count;     // monotonic
    std::atomic<uint64_t> last_seen_ns;
    std::atomic<uint64_t> last_window_count;     // value at start of current window
    std::atomic<uint64_t> last_window_start_ns;
    // Rolling 32-bucket history of "kernels per 30s" per tenant.
    std::atomic<uint64_t> history[LOAD_HISTORY_LEN];
    std::atomic<uint32_t> history_head;
    std::atomic<uint32_t> overload_count;        // consecutive overload windows
    std::atomic<uint32_t> should_migrate;        // 0/1 advisory
    std::atomic<uint32_t> pid;
};

struct GpuSlot {
    std::atomic<uint32_t> active_tenants;
    std::atomic<uint64_t> total_kernel_count;    // sum across tenants on this GPU
    std::atomic<uint64_t> last_update_ns;
};

struct DistributorShm {
    std::atomic<uint32_t> magic;
    std::atomic<uint32_t> initialized;
    std::atomic<uint64_t> intra_gpu_rebalances;
    std::atomic<uint64_t> cross_gpu_migrate_signals;
    std::atomic<uint64_t> spawn_router_picks;
    TenantSlot tenants[MAX_TENANTS];
    GpuSlot    gpus[MAX_GPUS];
};
constexpr size_t SHM_BYTES = sizeof(DistributorShm);

DistributorShm*       g_shm = nullptr;
std::atomic<int>      g_enabled{0};
std::atomic<int>      g_started{0};
int                   g_my_slot = -1;
uint32_t              g_my_tenant_hash = 0;
int                   g_my_gpu = 0;
pthread_t             g_poll_thread;
std::atomic<int>      g_poll_thread_running{0};

bool env_truthy(const char* v) {
    if (!v) return false;
    return v[0]=='1'||v[0]=='o'||v[0]=='O'||v[0]=='t'||v[0]=='T';
}
uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
uint32_t hash_tenant(const char* s) {
    if (!s) return 0;
    uint32_t h = 2166136261u;
    while (*s) { h ^= (uint8_t)*s++; h *= 16777619u; }
    return h ? h : 1;
}

// dlsym to cipher_op_inc — counter lives in this same DSO so the call
// is a normal function call, no dlsym needed.
extern "C" void cipher_op_inc(int op);
constexpr int OP_ARBITRATE = 17;     // matches CipherOpId

void* poll_thread_fn(void*) {
    while (g_poll_thread_running.load(std::memory_order_relaxed)) {
        struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = (long)POLL_PERIOD_NS;
        nanosleep(&ts, nullptr);
        if (!g_shm) continue;

        uint64_t now = now_ns();
        // Reap stale tenants (no activity in > 60 s).
        for (int i = 0; i < MAX_TENANTS; i++) {
            TenantSlot& t = g_shm->tenants[i];
            if (!t.active.load(std::memory_order_relaxed)) continue;
            uint64_t last = t.last_seen_ns.load(std::memory_order_relaxed);
            if (now > last && (now - last) > 60ULL * 1000 * 1000 * 1000) {
                t.active.store(0, std::memory_order_release);
                t.tenant_id_hash.store(0, std::memory_order_release);
            }
        }
        // For each tenant, advance the window if 30 s has elapsed since
        // window start, and check for overload.
        for (int i = 0; i < MAX_TENANTS; i++) {
            TenantSlot& t = g_shm->tenants[i];
            if (!t.active.load(std::memory_order_relaxed)) continue;
            uint64_t win_start = t.last_window_start_ns.load(std::memory_order_relaxed);
            if (win_start == 0) {
                t.last_window_start_ns.store(now, std::memory_order_relaxed);
                t.last_window_count.store(t.kernel_count.load(std::memory_order_relaxed),
                                          std::memory_order_relaxed);
                continue;
            }
            if (now - win_start < window_ns()) continue;
            // Window complete — record kernels in this 30 s.
            uint64_t cur = t.kernel_count.load(std::memory_order_relaxed);
            uint64_t prev = t.last_window_count.load(std::memory_order_relaxed);
            uint64_t delta = (cur > prev) ? cur - prev : 0;
            uint32_t head = t.history_head.fetch_add(1, std::memory_order_relaxed);
            t.history[head % LOAD_HISTORY_LEN].store(delta, std::memory_order_release);
            t.last_window_start_ns.store(now, std::memory_order_relaxed);
            t.last_window_count.store(cur, std::memory_order_relaxed);

            // Compute p95 of the history (sorted-rank approximation).
            uint64_t buckets[LOAD_HISTORY_LEN];
            int n = 0;
            for (int j = 0; j < LOAD_HISTORY_LEN; j++) {
                uint64_t v = t.history[j].load(std::memory_order_relaxed);
                if (v > 0) buckets[n++] = v;
            }
            if (n < 4) continue;     // not enough history
            // Simple insertion sort (n ≤ 32, trivial cost).
            for (int j = 1; j < n; j++) {
                uint64_t key = buckets[j]; int k = j - 1;
                while (k >= 0 && buckets[k] > key) { buckets[k+1] = buckets[k]; k--; }
                buckets[k+1] = key;
            }
            uint64_t p95 = buckets[(int)((n-1) * 0.95f)];
            bool overloaded = (delta > (uint64_t)((float)p95 * OVERLOAD_FACTOR));
            if (overloaded) {
                uint32_t cnt = t.overload_count.fetch_add(1, std::memory_order_relaxed) + 1;
                if (cnt >= 1 && t.should_migrate.load(std::memory_order_relaxed) == 0) {
                    t.should_migrate.store(1, std::memory_order_release);
                    g_shm->cross_gpu_migrate_signals.fetch_add(1, std::memory_order_relaxed);
                    g_shm->intra_gpu_rebalances.fetch_add(1, std::memory_order_relaxed);
                    cipher_op_inc(OP_ARBITRATE);
                    fprintf(stderr,
                        "[CIPHER DIST] OVERLOAD tenant_hash=0x%x gpu=%d "
                        "delta=%llu p95=%llu (%.2fx) → migrate signal armed\n",
                        t.tenant_id_hash.load(), t.gpu_id.load(),
                        (unsigned long long)delta, (unsigned long long)p95,
                        (float)delta / (float)p95);
                }
            } else {
                // De-arm if back to normal.
                t.overload_count.store(0, std::memory_order_relaxed);
                if (t.should_migrate.load(std::memory_order_relaxed)) {
                    t.should_migrate.store(0, std::memory_order_release);
                }
            }
        }
        // Update per-GPU totals.
        uint64_t gpu_totals[MAX_GPUS] = {0};
        uint32_t gpu_active[MAX_GPUS] = {0};
        for (int i = 0; i < MAX_TENANTS; i++) {
            TenantSlot& t = g_shm->tenants[i];
            if (!t.active.load(std::memory_order_relaxed)) continue;
            int gpu = t.gpu_id.load(std::memory_order_relaxed);
            if (gpu < 0 || gpu >= MAX_GPUS) continue;
            gpu_totals[gpu] += t.kernel_count.load(std::memory_order_relaxed);
            gpu_active[gpu] += 1;
        }
        for (int g = 0; g < MAX_GPUS; g++) {
            g_shm->gpus[g].active_tenants.store(gpu_active[g], std::memory_order_relaxed);
            g_shm->gpus[g].total_kernel_count.store(gpu_totals[g], std::memory_order_relaxed);
            g_shm->gpus[g].last_update_ns.store(now, std::memory_order_relaxed);
        }
    }
    return nullptr;
}

int register_self_slot() {
    if (!g_shm) return -1;
    if (g_my_slot >= 0) return g_my_slot;
    const char* tid_env = getenv("CIPHER_TENANT_ID");
    g_my_tenant_hash = tid_env ? hash_tenant(tid_env) : (uint32_t)getpid();
    const char* gpu_env = getenv("CUDA_VISIBLE_DEVICES");
    g_my_gpu = gpu_env ? atoi(gpu_env) : 0;
    if (g_my_gpu < 0 || g_my_gpu >= MAX_GPUS) g_my_gpu = 0;
    // Find an empty slot (CAS on tenant_id_hash 0 → ours).
    for (int i = 0; i < MAX_TENANTS; i++) {
        uint32_t expected = 0;
        if (g_shm->tenants[i].tenant_id_hash.compare_exchange_strong(
                expected, g_my_tenant_hash,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            g_shm->tenants[i].active.store(1, std::memory_order_release);
            g_shm->tenants[i].gpu_id.store(g_my_gpu, std::memory_order_release);
            g_shm->tenants[i].pid.store((uint32_t)getpid(), std::memory_order_release);
            g_shm->tenants[i].kernel_count.store(0, std::memory_order_release);
            g_shm->tenants[i].last_seen_ns.store(now_ns(), std::memory_order_release);
            g_shm->tenants[i].last_window_count.store(0, std::memory_order_release);
            g_shm->tenants[i].last_window_start_ns.store(0, std::memory_order_release);
            g_shm->tenants[i].history_head.store(0, std::memory_order_release);
            g_shm->tenants[i].overload_count.store(0, std::memory_order_release);
            g_shm->tenants[i].should_migrate.store(0, std::memory_order_release);
            for (int j = 0; j < LOAD_HISTORY_LEN; j++)
                g_shm->tenants[i].history[j].store(0, std::memory_order_release);
            g_my_slot = i;
            return i;
        }
        // Existing slot for our hash? (re-init after restart)
        if (g_shm->tenants[i].tenant_id_hash.load() == g_my_tenant_hash) {
            g_my_slot = i;
            return i;
        }
    }
    return -1;
}

} // namespace

extern "C" int cipher_distributor_init(void) {
    if (g_started.exchange(1, std::memory_order_acq_rel))
        return g_enabled.load(std::memory_order_relaxed);
    int on = env_truthy(getenv("CIPHER_DISTRIBUTOR"));
    if (!on) {
        // Default ON when CIPHER_FAIRNESS is on (distributor extends fairness).
        on = env_truthy(getenv("CIPHER_FAIRNESS"));
    }
    if (!on) {
        g_enabled.store(0, std::memory_order_release);
        return 0;
    }
    int fd = shm_open("/cipher_distributor", O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        fprintf(stderr, "[CIPHER DIST] shm_open failed: %m\n");
        g_enabled.store(0, std::memory_order_release);
        return 0;
    }
    if (ftruncate(fd, SHM_BYTES) != 0) {
        close(fd);
        g_enabled.store(0, std::memory_order_release);
        return 0;
    }
    void* p = mmap(nullptr, SHM_BYTES, PROT_READ | PROT_WRITE,
                   MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        g_enabled.store(0, std::memory_order_release);
        return 0;
    }
    g_shm = (DistributorShm*)p;
    // First-time init: stamp the magic + zero state. Guarded by CAS.
    uint32_t expected_magic = 0;
    if (g_shm->magic.compare_exchange_strong(expected_magic, DISTRIBUTOR_MAGIC,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
        memset(&g_shm->tenants[0], 0, sizeof(g_shm->tenants));
        memset(&g_shm->gpus[0], 0, sizeof(g_shm->gpus));
        g_shm->intra_gpu_rebalances.store(0);
        g_shm->cross_gpu_migrate_signals.store(0);
        g_shm->spawn_router_picks.store(0);
        g_shm->initialized.store(1, std::memory_order_release);
    }
    register_self_slot();
    // Start the polling thread (one per process).
    g_poll_thread_running.store(1, std::memory_order_release);
    pthread_create(&g_poll_thread, nullptr, poll_thread_fn, nullptr);
    pthread_detach(g_poll_thread);
    g_enabled.store(1, std::memory_order_release);
    fprintf(stderr,
        "[CIPHER DIST] init: tenant_hash=0x%x gpu=%d slot=%d (poll 50 ms)\n",
        g_my_tenant_hash, g_my_gpu, g_my_slot);
    return 1;
}

extern "C" void cipher_distributor_observe_kernel(uint32_t tenant_id) {
    (void)tenant_id;    // we use g_my_slot
    if (!g_enabled.load(std::memory_order_relaxed) || !g_shm) return;
    if (g_my_slot < 0) {
        if (register_self_slot() < 0) return;
    }
    g_shm->tenants[g_my_slot].kernel_count.fetch_add(1, std::memory_order_relaxed);
    g_shm->tenants[g_my_slot].last_seen_ns.store(now_ns(), std::memory_order_relaxed);
}

extern "C" int cipher_distributor_should_migrate(uint32_t tenant_id) {
    if (!g_enabled.load(std::memory_order_relaxed) || !g_shm) return 0;
    if (tenant_id == 0) {
        // Default: query for self.
        if (g_my_slot < 0) return 0;
        return g_shm->tenants[g_my_slot].should_migrate.load(std::memory_order_relaxed);
    }
    for (int i = 0; i < MAX_TENANTS; i++) {
        if (g_shm->tenants[i].tenant_id_hash.load(std::memory_order_relaxed) == tenant_id) {
            return g_shm->tenants[i].should_migrate.load(std::memory_order_relaxed);
        }
    }
    return 0;
}

extern "C" int cipher_distributor_pick_gpu_for_new_tenant(void) {
    if (!g_enabled.load(std::memory_order_relaxed) || !g_shm) return -1;
    int best_gpu = 0;
    uint64_t best_load = (uint64_t)-1;
    int n_gpus_seen = 0;
    for (int g = 0; g < MAX_GPUS; g++) {
        uint64_t up = g_shm->gpus[g].last_update_ns.load(std::memory_order_relaxed);
        if (up == 0) continue;
        n_gpus_seen++;
        uint64_t load = g_shm->gpus[g].total_kernel_count.load(std::memory_order_relaxed);
        if (load < best_load) { best_load = load; best_gpu = g; }
    }
    if (n_gpus_seen == 0) return -1;
    g_shm->spawn_router_picks.fetch_add(1, std::memory_order_relaxed);
    return best_gpu;
}

extern "C" int cipher_distributor_stats(struct CipherDistributorStats* out) {
    if (!out) return 0;
    out->enabled = g_enabled.load(std::memory_order_relaxed);
    if (!g_shm) return 1;
    int n_active = 0, n_overloaded = 0, n_gpus = 0;
    for (int i = 0; i < MAX_TENANTS; i++) {
        if (g_shm->tenants[i].active.load(std::memory_order_relaxed)) {
            n_active++;
            if (g_shm->tenants[i].should_migrate.load(std::memory_order_relaxed))
                n_overloaded++;
        }
    }
    for (int g = 0; g < MAX_GPUS; g++) {
        if (g_shm->gpus[g].last_update_ns.load(std::memory_order_relaxed) > 0)
            n_gpus++;
    }
    out->n_active_tenants    = n_active;
    out->n_gpus_seen         = n_gpus;
    out->n_overloaded_tenants = n_overloaded;
    out->intra_gpu_rebalances = g_shm->intra_gpu_rebalances.load(std::memory_order_relaxed);
    out->cross_gpu_migrate_signals = g_shm->cross_gpu_migrate_signals.load(std::memory_order_relaxed);
    out->spawn_router_picks  = g_shm->spawn_router_picks.load(std::memory_order_relaxed);
    return 1;
}

__attribute__((constructor(120)))
static void cipher_distributor_autoinit() {
    cipher_distributor_init();
}
