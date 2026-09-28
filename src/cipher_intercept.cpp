#include <atomic>
// =============================================================================
// CIPHER — F1: Intercept Implementation
// cipher_intercept.cpp
//
// LD_PRELOAD workflow:
//   1. .so constructor calls cipher_intercept_init()
//   2. We dlsym(libcuda.so, "cuGetProcAddress") — get the real resolver
//   3. We export our own cuGetProcAddress symbol — intercepts all future
//      framework calls to resolve CUDA driver functions
//   4. When any framework asks for "cuLaunchKernel" we return our shim
//   5. Shim stamps timestamp, calls cipher_dispatch(), passthroughs if needed
//
// Thread safety: all hot-path state is either thread-local or atomic.
// No locks on the dispatch fast path.
// =============================================================================

#define _GNU_SOURCE
#include "cipher_intercept.h"
#include <dlfcn.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

typedef CUresult (*real_cuLaunchKernel_t)(
    CUfunction, uint32_t, uint32_t, uint32_t,
    uint32_t,   uint32_t, uint32_t,
    uint32_t, CUstream, void**, void**);

typedef CUresult (*real_cuGetProcAddress_t)(
    const char*, void**, int, uint64_t, CUdriverProcAddressQueryResult*);

// ---------------------------------------------------------------------------
// Module-level state
// ---------------------------------------------------------------------------

static real_cuLaunchKernel_t   g_real_launch     = NULL;
static real_cuGetProcAddress_t g_real_proc_addr   = NULL;
static std::atomic<bool>            g_initialized      = false;
static std::atomic<bool>            g_shutting_down    = false;

// Stats — all atomic, updated on every intercept, zero overhead read
static std::atomic<uint64_t> g_stat_total      = 0;
static std::atomic<uint64_t> g_stat_subst      = 0;
static std::atomic<uint64_t> g_stat_pass       = 0;
static std::atomic<uint64_t> g_stat_deferred   = 0;
static std::atomic<uint64_t> g_stat_ns_sum     = 0;
static std::atomic<uint64_t> g_stat_ns_max     = 0;

// ---------------------------------------------------------------------------
// Timing — CLOCK_MONOTONIC_RAW for actual hardware cycles, no NTP jumps
// ---------------------------------------------------------------------------

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// ---------------------------------------------------------------------------
// F1 core: our cuLaunchKernel shim
// This is the function every framework actually calls after we're loaded.
// Hot path budget: <100ns total (intercept + classify + passthrough decision)
// ---------------------------------------------------------------------------

static CUresult cipher_launch_kernel_shim(
    CUfunction fn,
    uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t shared_bytes,
    CUstream stream,
    void**   params,
    void**   extra)
{
    // Stamp entry immediately — before anything else
    uint64_t t0 = now_ns();

    // If shutting down or not fully initialized: pure passthrough, <10ns
    if (__builtin_expect(
            g_shutting_down.load(std::memory_order_relaxed) ||
            !atomic_load_explicit(&g_initialized,  std::memory_order_relaxed),
            0))
    {
        return g_real_launch(fn, gx, gy, gz, bx, by, bz, shared_bytes,
                             stream, params, extra);
    }

    // Build descriptor on stack — zero heap alloc
    CipherKernelDesc desc = {
        .fn           = fn,
        .grid_x       = gx,  .grid_y  = gy,  .grid_z  = gz,
        .block_x      = bx,  .block_y = by,  .block_z = bz,
        .shared_bytes = shared_bytes,
        .stream       = stream,
        .params       = params,
        .extra        = extra,
        .op_class     = 0xFF,   // unclassified
        .confidence   = 0,
        .intercept_ns = t0,
    };

    // Dispatch — cipher_dispatch() owns classification + routing decision
    CipherDispatchResult result = cipher_dispatch(&desc);

    // Update stats — relaxed stores, never on the critical path
    uint64_t elapsed = now_ns() - t0;
    g_stat_total.fetch_add(1, std::memory_order_relaxed);
    g_stat_ns_sum.fetch_add(elapsed, std::memory_order_relaxed);

    uint64_t prev_max = g_stat_ns_max.load(std::memory_order_relaxed);
    if (elapsed > prev_max)
        g_stat_ns_max.store(elapsed, std::memory_order_relaxed);

    switch (result) {
        case CIPHER_SUBSTITUTED:
            atomic_fetch_add_explicit(&g_stat_subst,    1, std::memory_order_relaxed);
            return CUDA_SUCCESS;   // Neural equivalent already dispatched

        case CIPHER_DEFERRED:
            g_stat_deferred.fetch_add(1, std::memory_order_relaxed);
            // Fall through to passthrough — Layer 1 will handle async
            __attribute__((fallthrough));

        case CIPHER_PASS_THROUGH:
        default:
            atomic_fetch_add_explicit(&g_stat_pass,     1, std::memory_order_relaxed);
            return g_real_launch(fn, gx, gy, gz, bx, by, bz, shared_bytes,
                                 stream, params, extra);
    }
}

// ---------------------------------------------------------------------------
// Our cuGetProcAddress override — exported symbol, picked up by LD_PRELOAD
//
// Every framework that dlopen()s libcuda.so and calls cuGetProcAddress to
// resolve driver functions will land here first. We intercept the request
// for "cuLaunchKernel" and return our shim. Everything else is forwarded
// to the real resolver unchanged.
// ---------------------------------------------------------------------------

CUresult cuGetProcAddress(
    const char* symbol,
    void**      pfn,
    int         cudaVersion,
    uint64_t    flags,
    CUdriverProcAddressQueryResult* symbolStatus)
{
    if (g_real_proc_addr == NULL) {
        // Bootstrap: find the real cuGetProcAddress before we've initialized
        g_real_proc_addr = (real_cuGetProcAddress_t)
            dlsym(RTLD_NEXT, "cuGetProcAddress");
        if (!g_real_proc_addr) {
            fprintf(stderr, "[CIPHER] FATAL: cannot find real cuGetProcAddress\n");
            return CUDA_ERROR_NOT_FOUND;
        }
    }

    CUresult r = g_real_proc_addr(symbol, pfn, cudaVersion, flags, symbolStatus);

    if (r == CUDA_SUCCESS && symbol != NULL) {
        // Intercept cuLaunchKernel — this is the only hook we need
        if (strcmp(symbol, "cuLaunchKernel") == 0) {
            // Store the real function pointer before overwriting
            if (g_real_launch == NULL)
                g_real_launch = (real_cuLaunchKernel_t)*pfn;
            // Redirect to our shim
            *pfn = (void*)cipher_launch_kernel_shim;
        }
        // Future: intercept cuLaunchKernelEx for cooperative groups
    }

    return r;
}

// ---------------------------------------------------------------------------
// cipher_intercept_init — called from .so constructor
// ---------------------------------------------------------------------------

CUresult cipher_intercept_init(void) {
    // Resolve real cuGetProcAddress via RTLD_NEXT (skips our own symbol)
    g_real_proc_addr = (real_cuGetProcAddress_t)
        dlsym(RTLD_NEXT, "cuGetProcAddress");
    if (!g_real_proc_addr) {
        fprintf(stderr, "[CIPHER F1] Cannot resolve cuGetProcAddress: %s\n",
                dlerror());
        return CUDA_ERROR_NOT_FOUND;
    }

    // Resolve cuLaunchKernel directly as well — covers frameworks that
    // dlsym() it directly rather than going through cuGetProcAddress
    g_real_launch = (real_cuLaunchKernel_t)
        dlsym(RTLD_NEXT, "cuLaunchKernel");
    if (!g_real_launch) {
        fprintf(stderr, "[CIPHER F1] Cannot resolve cuLaunchKernel: %s\n",
                dlerror());
        return CUDA_ERROR_NOT_FOUND;
    }

    g_initialized.store(true);
    fprintf(stderr, "[CIPHER F1] cuLaunchKernel hook installed. "
                    "real_launch=%p shim=%p\n",
            (void*)g_real_launch, (void*)cipher_launch_kernel_shim);
    return CUDA_SUCCESS;
}

void cipher_intercept_teardown(void) {
    g_shutting_down.store(true);
    // Drain in-flight calls — spin max 1ms
    for (int i = 0; i < 1000; i++) {
        struct timespec ts = { .tv_nsec = 1000 };
        nanosleep(&ts, NULL);
    }
}

// ---------------------------------------------------------------------------
// cipher_passthrough — direct call to real cuLaunchKernel
// Used by cipher_dispatch() when confidence is insufficient.
// ---------------------------------------------------------------------------

CUresult cipher_passthrough(const CipherKernelDesc* d) {
    return g_real_launch(d->fn, d->grid_x, d->grid_y, d->grid_z,
                         d->block_x, d->block_y, d->block_z,
                         d->shared_bytes, d->stream, d->params, d->extra);
}

// ---------------------------------------------------------------------------
// Stats readout (lock-free snapshot)
// ---------------------------------------------------------------------------

static CipherInterceptStats g_stats_snapshot;

const CipherInterceptStats* cipher_intercept_stats(void) {
    g_stats_snapshot.total_intercepts  = g_stat_total.load();
    g_stats_snapshot.substitutions     = g_stat_subst.load();
    g_stats_snapshot.passthroughs      = g_stat_pass.load();
    g_stats_snapshot.deferred          = g_stat_deferred.load();
    g_stats_snapshot.overhead_ns_sum   = g_stat_ns_sum.load();
    g_stats_snapshot.overhead_ns_max   = g_stat_ns_max.load();
    return &g_stats_snapshot;
}

// ---------------------------------------------------------------------------
// .so constructor / destructor — automatic install on LD_PRELOAD
// ---------------------------------------------------------------------------

__attribute__((constructor))
static void cipher_so_init(void) {
    CUresult r = cipher_intercept_init();
    if (r != CUDA_SUCCESS)
        fprintf(stderr, "[CIPHER F1] Init failed: %d\n", r);
}

__attribute__((destructor))
static void cipher_so_fini(void) {
    cipher_intercept_teardown();
    const CipherInterceptStats* s = cipher_intercept_stats();
    fprintf(stderr,
        "[CIPHER F1] Teardown. Intercepts: %lu | Substitutions: %lu "
        "(%.1f%%) | Avg overhead: %.1fns | Max: %luns\n",
        s->total_intercepts,
        s->substitutions,
        s->total_intercepts > 0
            ? (double)s->substitutions * 100.0 / s->total_intercepts
            : 0.0,
        s->total_intercepts > 0
            ? (double)s->overhead_ns_sum / s->total_intercepts
            : 0.0,
        s->overhead_ns_max);
}
