// CIPHER cross-tenant FAIRNESS via POSIX shared memory.

#include "may13/cipher_fairness_shm.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

constexpr uint32_t FAIRNESS_MAGIC = 0xC1F4C1F4u;
constexpr int      MAX_TENANTS    = 64;

struct TenantSlot {
    std::atomic<uint32_t> tenant_id;
    std::atomic<uint32_t> active;       // 1 = registered/alive
    std::atomic<uint64_t> gemm_calls;   // monotonically increasing
    std::atomic<uint64_t> last_active_ns;
    std::atomic<uint64_t> yield_count;
    std::atomic<uint64_t> pid;
};

struct CipherFairnessShm {
    std::atomic<uint32_t> magic;
    std::atomic<uint32_t> initialized;
    TenantSlot slots[MAX_TENANTS];
};

constexpr size_t SHM_BYTES = sizeof(CipherFairnessShm);

CipherFairnessShm* g_shm = nullptr;
std::atomic<int>   g_enabled{0};
std::atomic<int>   g_started{0};
int                g_my_slot = -1;
uint32_t           g_my_tenant_id = 0;

bool env_truthy(const char* v) {
    if (!v) return false;
    return v[0] == '1' || v[0] == 'o' || v[0] == 'O' ||
           v[0] == 't' || v[0] == 'T';
}

uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

} // namespace

extern "C" int cipher_fairness_shm_init(void) {
    if (g_started.exchange(1, std::memory_order_acq_rel))
        return g_enabled.load(std::memory_order_relaxed);

    int on = env_truthy(getenv("CIPHER_FAIRNESS"));
    if (!on) {
        g_enabled.store(0, std::memory_order_release);
        return 0;
    }

    int fd = shm_open("/cipher_fairness", O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        fprintf(stderr, "[CIPHER FAIR] shm_open failed: %m\n");
        g_enabled.store(0, std::memory_order_release);
        return 0;
    }
    if (ftruncate(fd, SHM_BYTES) != 0) {
        fprintf(stderr, "[CIPHER FAIR] ftruncate failed: %m\n");
        close(fd);
        g_enabled.store(0, std::memory_order_release);
        return 0;
    }
    void* m = mmap(nullptr, SHM_BYTES, PROT_READ | PROT_WRITE,
                   MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) {
        fprintf(stderr, "[CIPHER FAIR] mmap failed: %m\n");
        close(fd);
        g_enabled.store(0, std::memory_order_release);
        return 0;
    }
    close(fd);
    g_shm = (CipherFairnessShm*)m;
    // Initialize once via CAS.
    uint32_t expected = 0;
    if (g_shm->magic.compare_exchange_strong(expected, FAIRNESS_MAGIC,
                                              std::memory_order_acq_rel)) {
        for (int i = 0; i < MAX_TENANTS; ++i) {
            g_shm->slots[i].tenant_id.store(0, std::memory_order_relaxed);
            g_shm->slots[i].active.store(0, std::memory_order_relaxed);
            g_shm->slots[i].gemm_calls.store(0, std::memory_order_relaxed);
            g_shm->slots[i].last_active_ns.store(0, std::memory_order_relaxed);
            g_shm->slots[i].yield_count.store(0, std::memory_order_relaxed);
            g_shm->slots[i].pid.store(0, std::memory_order_relaxed);
        }
        g_shm->initialized.store(1, std::memory_order_release);
    }
    g_enabled.store(1, std::memory_order_release);
    fprintf(stderr,
        "[CIPHER FAIR] shm initialized (max_tenants=%d, region=%zu bytes)\n",
        MAX_TENANTS, SHM_BYTES);
    return 1;
}

extern "C" int cipher_fairness_shm_enabled(void) {
    return g_enabled.load(std::memory_order_relaxed);
}

extern "C" int cipher_fairness_shm_register(void) {
    if (!g_enabled.load(std::memory_order_relaxed) || !g_shm) return -1;
    if (g_my_slot >= 0) return g_my_slot;

    const char* env = getenv("CIPHER_TENANT_ID");
    g_my_tenant_id = env ? (uint32_t)atoi(env) : (uint32_t)getpid();
    uint64_t my_pid = (uint64_t)getpid();

    // Find existing slot for this tenant_id (re-registration).
    for (int i = 0; i < MAX_TENANTS; ++i) {
        if (g_shm->slots[i].tenant_id.load(std::memory_order_acquire) ==
                g_my_tenant_id &&
            g_shm->slots[i].active.load(std::memory_order_acquire)) {
            // Already registered — claim it.
            g_shm->slots[i].pid.store(my_pid, std::memory_order_release);
            g_shm->slots[i].last_active_ns.store(now_ns(), std::memory_order_release);
            g_my_slot = i;
            fprintf(stderr,
                "[CIPHER FAIR] re-registered tenant_id=%u in slot %d\n",
                g_my_tenant_id, i);
            return i;
        }
    }
    // Allocate a new slot via CAS on `active`.
    for (int i = 0; i < MAX_TENANTS; ++i) {
        uint32_t expected = 0;
        if (g_shm->slots[i].active.compare_exchange_strong(
                expected, 1, std::memory_order_acq_rel)) {
            g_shm->slots[i].tenant_id.store(g_my_tenant_id, std::memory_order_release);
            g_shm->slots[i].pid.store(my_pid, std::memory_order_release);
            g_shm->slots[i].gemm_calls.store(0, std::memory_order_release);
            g_shm->slots[i].yield_count.store(0, std::memory_order_release);
            g_shm->slots[i].last_active_ns.store(now_ns(), std::memory_order_release);
            g_my_slot = i;
            fprintf(stderr,
                "[CIPHER FAIR] registered tenant_id=%u in slot %d (pid=%lu)\n",
                g_my_tenant_id, i, my_pid);
            return i;
        }
    }
    fprintf(stderr, "[CIPHER FAIR] all slots full!\n");
    return -1;
}

extern "C" void cipher_fairness_shm_record_gemm(void) {
    if (!g_enabled.load(std::memory_order_relaxed) || !g_shm) return;
    if (g_my_slot < 0) {
        if (cipher_fairness_shm_register() < 0) return;
    }
    g_shm->slots[g_my_slot].gemm_calls.fetch_add(1, std::memory_order_relaxed);
    g_shm->slots[g_my_slot].last_active_ns.store(now_ns(),
        std::memory_order_release);
}

extern "C" int cipher_fairness_shm_should_yield(void) {
    if (!g_enabled.load(std::memory_order_relaxed) || !g_shm ||
        g_my_slot < 0)
        return 0;
    // Compute average gemm_calls across recently-active tenants.
    uint64_t total = 0;
    int n_active = 0;
    uint64_t now = now_ns();
    constexpr uint64_t IDLE_THRESH_NS = 2ULL * 1000 * 1000 * 1000; // 2 s
    for (int i = 0; i < MAX_TENANTS; ++i) {
        if (!g_shm->slots[i].active.load(std::memory_order_acquire)) continue;
        uint64_t la = g_shm->slots[i].last_active_ns.load(std::memory_order_acquire);
        if (la == 0 || now - la > IDLE_THRESH_NS) continue;
        total += g_shm->slots[i].gemm_calls.load(std::memory_order_relaxed);
        n_active++;
    }
    if (n_active <= 1) return 0;
    uint64_t avg = total / (uint64_t)n_active;
    uint64_t my  = g_shm->slots[g_my_slot].gemm_calls.load(std::memory_order_relaxed);
    if (my > 2 * avg && my > 1024) {
        g_shm->slots[g_my_slot].yield_count.fetch_add(1, std::memory_order_relaxed);
        return 1;
    }
    return 0;
}

extern "C" int cipher_fairness_shm_self_calls(uint64_t* out_calls,
                                              uint64_t* out_yields) {
    if (!g_enabled.load(std::memory_order_relaxed) || !g_shm ||
        g_my_slot < 0) return 0;
    if (out_calls)
        *out_calls = g_shm->slots[g_my_slot].gemm_calls.load(std::memory_order_relaxed);
    if (out_yields)
        *out_yields = g_shm->slots[g_my_slot].yield_count.load(std::memory_order_relaxed);
    return 1;
}

extern "C" void cipher_fairness_shm_shutdown(void) {
    if (!g_enabled.load(std::memory_order_relaxed) || !g_shm) return;
    if (g_my_slot >= 0) {
        uint64_t my_calls = g_shm->slots[g_my_slot].gemm_calls.load(std::memory_order_relaxed);
        uint64_t my_yields = g_shm->slots[g_my_slot].yield_count.load(std::memory_order_relaxed);
        fprintf(stderr,
            "[CIPHER FAIR] tenant_id=%u slot=%d calls=%lu yields=%lu — "
            "releasing slot\n",
            g_my_tenant_id, g_my_slot,
            (unsigned long)my_calls, (unsigned long)my_yields);
        g_shm->slots[g_my_slot].active.store(0, std::memory_order_release);
        g_shm->slots[g_my_slot].tenant_id.store(0, std::memory_order_release);
        g_my_slot = -1;
    }
    munmap(g_shm, SHM_BYTES);
    g_shm = nullptr;
}

__attribute__((constructor(114)))
static void cipher_fairness_shm_autoinit() { cipher_fairness_shm_init(); }

__attribute__((destructor))
static void cipher_fairness_shm_autoshutdown() { cipher_fairness_shm_shutdown(); }
