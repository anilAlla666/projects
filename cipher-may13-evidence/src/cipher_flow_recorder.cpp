// CIPHER pointer-flow recorder — Stage 1.

#include "cipher_flow_recorder.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>

namespace {

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_started{0};
std::atomic<uint64_t> g_seq{0};
std::atomic<uint64_t> g_total_launches{0};

CipherFlowEntry g_ring[CIPHER_FLOW_RING];
std::atomic<int> g_ring_head{0};
std::mutex       g_ring_mu;

// Heuristic: typical CUDA device VA addresses fall in this range on
// modern Linux + 64-bit drivers (after the OS UVM mapping shift).
constexpr uint64_t DEV_PTR_LOW  = 0x100000000ULL;
constexpr uint64_t DEV_PTR_HIGH = 0x800000000000ULL;

inline bool looks_like_dev_ptr(uint64_t v) {
    return v >= DEV_PTR_LOW && v < DEV_PTR_HIGH;
}

bool env_truthy(const char* v) {
    if (!v) return false;
    return v[0] == '1' || v[0] == 'o' || v[0] == 'O' ||
           v[0] == 't' || v[0] == 'T';
}

// cipher_kt_name_for is defined in cipher_kernel_table.cpp (same DSO).
extern "C" const char* cipher_kt_name_for(void* fn_handle);

const char* resolve_name(void* func) {
    return cipher_kt_name_for(func);
}

} // namespace

extern "C" int cipher_flow_recorder_init(void) {
    if (g_started.exchange(1, std::memory_order_acq_rel))
        return g_enabled.load(std::memory_order_relaxed);
    int on = env_truthy(getenv("CIPHER_FLOW_RECORD"));
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        for (int i = 0; i < CIPHER_FLOW_RING; ++i)
            g_ring[i] = CipherFlowEntry{};
        fprintf(stderr,
            "[CIPHER FLOW] recorder enabled (ring=%d)\n", CIPHER_FLOW_RING);
    }
    return on;
}

extern "C" int cipher_flow_recorder_enabled(void) {
    return g_enabled.load(std::memory_order_relaxed);
}

extern "C" void cipher_flow_recorder_observe(
    void* func, uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t smem, void** args, void* stream)
{
    static std::atomic<uint64_t> s_calls{0};
    uint64_t c = s_calls.fetch_add(1, std::memory_order_relaxed);
    if (c < 3) {
        fprintf(stderr,
            "[CIPHER FLOW] observe #%lu called (en=%d) func=%p\n",
            (unsigned long)c, g_enabled.load(std::memory_order_relaxed), func);
    }
    if (!g_enabled.load(std::memory_order_relaxed)) return;

    uint64_t seq = g_seq.fetch_add(1, std::memory_order_relaxed);
    g_total_launches.fetch_add(1, std::memory_order_relaxed);

    CipherFlowEntry e{};
    e.func_ptr = func;
    e.grid_x = gx; e.grid_y = gy; e.grid_z = gz;
    e.block_x = bx; e.block_y = by; e.block_z = bz;
    e.smem = smem;
    e.seq  = seq;
    e.stream = (uint64_t)stream;

    // Read args[] BEST-EFFORT.  PyTorch's cudaLaunchKernel doesn't null-
    // terminate the args array, so iterating off the end is undefined.
    // We bound the loop to a single page-aligned stack window: iterate
    // while args[i] looks like a sane stack pointer (canonical user
    // range AND on the same page as args[0]), then read 8 bytes.
    if (args && args[0]) {
        uintptr_t a0 = (uintptr_t)args[0];
        constexpr uintptr_t PAGE_MASK = ~((uintptr_t)4095);
        for (int i = 0; i < 8; ++i) {
            uintptr_t ai = (uintptr_t)args[i];
            if (ai == 0) break;
            // Sanity: canonical user-space range, same page as args[0].
            if (ai < 0x10000 || ai >= 0x800000000000ULL) break;
            if ((ai & PAGE_MASK) != (a0 & PAGE_MASK)) break;
            // 8-byte read is safe within a page (we already established
            // that ai fits within the stack page of args[0]).
            uint64_t v = 0;
            memcpy(&v, args[i], sizeof(uint64_t));
            e.args[i]   = v;
            e.is_ptr[i] = looks_like_dev_ptr(v) ? 1 : 0;
        }
    }

    // Trigger a dump every N launches if requested.
    static int dump_every = -1;
    if (dump_every < 0) {
        const char* s = getenv("CIPHER_FLOW_DUMP_EVERY");
        dump_every = s ? atoi(s) : 0;
    }
    if (dump_every > 0 && ((seq + 1) % (uint64_t)dump_every) == 0) {
        cipher_flow_recorder_dump();
    }

    int idx = g_ring_head.fetch_add(1, std::memory_order_acq_rel) %
              CIPHER_FLOW_RING;
    {
        std::lock_guard<std::mutex> lk(g_ring_mu);
        g_ring[idx] = e;
    }
}

namespace {
// Compact role hint based on arg position + value class.
const char* class_hint(uint64_t v, bool is_ptr) {
    if (!v) return "ZERO";
    if (is_ptr) return "DEV";
    if (v < (1ULL << 24)) return "SMALL";
    return "OTHER";
}
} // namespace

extern "C" void cipher_flow_recorder_dump(void) {
    fprintf(stderr,
        "\n[CIPHER FLOW] dump (total_launches=%llu)\n"
        "================================================================================\n",
        (unsigned long long)g_total_launches.load(std::memory_order_relaxed));

    // Walk oldest-first.
    int head = g_ring_head.load(std::memory_order_acquire) % CIPHER_FLOW_RING;
    std::lock_guard<std::mutex> lk(g_ring_mu);
    for (int k = 0; k < CIPHER_FLOW_RING; ++k) {
        int i = (head + k) % CIPHER_FLOW_RING;
        const CipherFlowEntry& e = g_ring[i];
        if (!e.func_ptr) continue;
        const char* nm = resolve_name(e.func_ptr);
        fprintf(stderr,
            "[FLOW seq=%llu] fn=%p grid=%ux%ux%u block=%ux%ux%u smem=%u "
            "stream=%p\n  name=%.130s\n",
            (unsigned long long)e.seq, e.func_ptr,
            e.grid_x, e.grid_y, e.grid_z,
            e.block_x, e.block_y, e.block_z, e.smem,
            (void*)e.stream, nm ? nm : "(unknown)");
        for (int a = 0; a < 8; ++a) {
            if (!e.args[a]) break;
            fprintf(stderr,
                "  args[%d] = 0x%016lx [%s]\n",
                a, (unsigned long)e.args[a],
                class_hint(e.args[a], e.is_ptr[a]));
        }
        // Cross-reference: which earlier entry's output ptrs equal any of
        // this entry's input-position ptrs?
        for (int a = 0; a < 8; ++a) {
            if (!e.is_ptr[a]) continue;
            uint64_t p = e.args[a];
            for (int kk = 1; kk <= k; ++kk) {
                int j = (head + k - kk) % CIPHER_FLOW_RING;
                const CipherFlowEntry& f = g_ring[j];
                if (!f.func_ptr) continue;
                for (int b = 0; b < 8; ++b) {
                    if (f.is_ptr[b] && f.args[b] == p) {
                        fprintf(stderr,
                            "  edge: args[%d]=%lx <- seq=%llu args[%d] "
                            "(%d launch(es) ago)\n",
                            a, (unsigned long)p,
                            (unsigned long long)f.seq, b, kk);
                        // First match per arg is enough.
                        kk = k + 1;
                        break;
                    }
                }
            }
        }
    }
    fprintf(stderr, "================================================================================\n\n");
}

__attribute__((constructor(115)))
static void cipher_flow_recorder_autoinit() { cipher_flow_recorder_init(); }

__attribute__((destructor))
static void cipher_flow_recorder_autodump() {
    if (g_enabled.load(std::memory_order_relaxed) &&
        env_truthy(getenv("CIPHER_FLOW_DUMP"))) {
        cipher_flow_recorder_dump();
    }
}
