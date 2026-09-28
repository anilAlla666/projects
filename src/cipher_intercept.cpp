// CPU stub redirect
#ifdef CIPHER_CPU_STUB
#  include "cipher_stubs.h"
#endif
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
#include "cipher_10ops.h"
#include <dlfcn.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

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

typedef CUresult (*real_cuLaunchKernelEx_t)(
    void*, CUfunction, void**, void**);
static real_cuLaunchKernelEx_t g_real_launch_ex  = NULL;

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

// Thread-local: when set, cipher_launch_kernel_shim passthroughs via
// g_real_launch_ex with the saved config instead of g_real_launch
static __thread const void* tls_launch_ex_config = NULL;

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
            !g_initialized.load(std::memory_order_relaxed),
            0))
    {
        if (tls_launch_ex_config && g_real_launch_ex)
            return g_real_launch_ex((void*)tls_launch_ex_config, fn, params, extra);
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


    // ── SPECULATE check — Stage 0 addition, ~2ns ─────────────────────────
    // Look-aside buffer written by Stage 1 shadow thread.
    // On hit: pre-computed payload ready, skip CLASSIFY+SUBSTITUTE entirely.
    // On miss: standard path below, no regression.
    {
        int spec_hit = cipher_lookaside_check(&g_cipher_10ops.look_aside,
                                               (int)desc.op_class);
        g_cipher_10ops.speculate_total.fetch_add(1, std::memory_order_relaxed);
        if (spec_hit &&
            g_cipher_10ops.initialized.load(std::memory_order_acquire)) {
            g_cipher_10ops.speculate_hits.fetch_add(1, std::memory_order_relaxed);
            // Op 2 SPECULATE check: HIT — kernel skipped
            {
                static uint64_t spec_hit_log = 0;
                spec_hit_log++;
                if (spec_hit_log <= 5 || (spec_hit_log % 100) == 0) {
                    fprintf(stderr,
                        "[CIPHER Op2] SPECULATE HIT: class=%d skipped kernel fn=%p count=%llu\n",
                        (int)desc.op_class, desc.fn,
                        (unsigned long long)spec_hit_log);
                }
            }
            // Pre-computed result treated as SUBSTITUTED
            g_stat_subst.fetch_add(1, std::memory_order_relaxed);
            return CUDA_SUCCESS;
        }
    }

    // Dispatch — cipher_dispatch() owns classification + routing decision
    CipherDispatchResult result = cipher_dispatch(&desc);

    // Update stats — relaxed stores, never on the critical path
    uint64_t elapsed = now_ns() - t0;
    g_stat_total.fetch_add(1, std::memory_order_relaxed);
    g_stat_ns_sum.fetch_add(elapsed, std::memory_order_relaxed);

    uint64_t prev_max = g_stat_ns_max.load(std::memory_order_relaxed);
    if (elapsed > prev_max)
        g_stat_ns_max.store(elapsed, std::memory_order_relaxed);


    // ── Ring write — Stage 0 addition, ~10ns ─────────────────────────────
    // Single atomic release store — plain MOV on x86 TSO.
    // Stage 1 (REMEMBER+VALIDATE+AUDIT+SPECULATE) reads this asynchronously.
    // Never blocks. If ring full: lossy drop (Stage 1 fell too far behind).
    if (g_cipher_10ops.initialized.load(std::memory_order_acquire)) {
        CipherRingEntry _ring_ev = {};
        _ring_ev.sequence        = g_stat_total.load(std::memory_order_relaxed);
        _ring_ev.timestamp_ns    = t0;
        _ring_ev.timestamp_delta = elapsed;
        _ring_ev.kernel_class    = desc.op_class;
        _ring_ev.grid_x          = gx; _ring_ev.grid_y = gy; _ring_ev.grid_z = gz;
        _ring_ev.block_x         = bx; _ring_ev.block_y = by; _ring_ev.block_z = bz;
        _ring_ev.func_ptr_hash   = (uint64_t)(uintptr_t)fn;
        _ring_ev.confidence      = (float)desc.confidence / 100.0f;
        _ring_ev.decision        = (uint8_t)result;
        if (cipher_ring_write(&g_cipher_10ops.ring, &_ring_ev))
            g_cipher_10ops.ring_writes.fetch_add(1, std::memory_order_relaxed);
    }

    switch (result) {
        case CIPHER_SUBSTITUTED:
            g_stat_subst.fetch_add(1, std::memory_order_relaxed);
            return CUDA_SUCCESS;   // Neural equivalent already dispatched

        case CIPHER_DEFERRED:
            g_stat_deferred.fetch_add(1, std::memory_order_relaxed);
            // Fall through to passthrough — Layer 1 will handle async
            __attribute__((fallthrough));

        case CIPHER_PASS_THROUGH:
        default:
            g_stat_pass.fetch_add(1, std::memory_order_relaxed);
            if (tls_launch_ex_config && g_real_launch_ex)
                return g_real_launch_ex((void*)tls_launch_ex_config, fn, params, extra);
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
        if (strcmp(symbol, "cuLaunchKernel") == 0 ||
            strcmp(symbol, "cuLaunchKernel_ptsz") == 0) {
            // Store the real function pointer before overwriting
            if (g_real_launch == NULL)
                g_real_launch = (real_cuLaunchKernel_t)*pfn;
            // Redirect to our shim
            *pfn = (void*)cipher_launch_kernel_shim;
        }
        if (strcmp(symbol, "cuLaunchKernelEx") == 0 ||
            strcmp(symbol, "cuLaunchKernelEx_ptsz") == 0) {
            if (g_real_launch_ex == NULL)
                g_real_launch_ex = (real_cuLaunchKernelEx_t)*pfn;
            *pfn = (void*)cuLaunchKernelEx;
        }
        // Intercept cuGetProcAddress_v2 — redirect to ourselves
        if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
            *pfn = (void*)cuGetProcAddress;
        }
    }

    return r;
}

// ---------------------------------------------------------------------------
// cuLaunchKernelEx shim — CUDA 12+ path used by PyTorch
// Extracts grid/block from config struct, delegates to same dispatch logic.
// ---------------------------------------------------------------------------

struct CipherLaunchConfig {
    unsigned gridDimX, gridDimY, gridDimZ;
    unsigned blockDimX, blockDimY, blockDimZ;
    unsigned sharedMemBytes;
    void*    hStream;
    void*    attrs;
    unsigned numAttrs;
};

extern "C" __attribute__((visibility("default")))
CUresult cuLaunchKernelEx(
    const CUlaunchConfig* config,
    CUfunction fn,
    void**     params,
    void**     extra)
{
    if (!g_real_launch_ex) return CUDA_ERROR_NOT_FOUND;

    if (g_shutting_down.load(std::memory_order_relaxed) ||
        !g_initialized.load(std::memory_order_relaxed))
        return g_real_launch_ex((void*)config, fn, params, extra);

    auto* cfg = (const CipherLaunchConfig*)config;

    // Save config so passthrough uses g_real_launch_ex
    tls_launch_ex_config = config;

    CUresult r = cipher_launch_kernel_shim(
        fn,
        cfg->gridDimX, cfg->gridDimY, cfg->gridDimZ,
        cfg->blockDimX, cfg->blockDimY, cfg->blockDimZ,
        cfg->sharedMemBytes, (CUstream)cfg->hStream,
        params, extra);

    tls_launch_ex_config = NULL;
    return r;
}

// ---------------------------------------------------------------------------
// cipher_intercept_init — called from .so constructor
// ---------------------------------------------------------------------------

CUresult cipher_intercept_init(void) {
    // Resolve via explicit libcuda handle — no cuGetProcAddress needed
    void* lc = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!lc) lc = dlopen("libcuda.so.1", RTLD_NOW);

    // Resolve real cuGetProcAddress (optional — not required for interception)
    if (lc) {
        g_real_proc_addr = (real_cuGetProcAddress_t)
            dlsym(lc, "cuGetProcAddress");
    }

    // Resolve cuLaunchKernel via GOT-compatible dlsym on libcuda
    if (lc) {
        g_real_launch = (real_cuLaunchKernel_t)dlsym(lc, "cuLaunchKernel");
    }
    if (!g_real_launch) {
        // Fallback: try RTLD_DEFAULT (covers LD_PRELOAD resolution order)
        g_real_launch = (real_cuLaunchKernel_t)dlsym(RTLD_DEFAULT, "cuLaunchKernel");
    }
    if (!g_real_launch) {
        fprintf(stderr, "[CIPHER F1] Cannot resolve cuLaunchKernel — "
                         "CUDA driver not loaded yet, deferring.\n");
        return CUDA_SUCCESS;
    }

    // Resolve cuLaunchKernelEx (CUDA 12+ — used by PyTorch)
    if (lc) {
        g_real_launch_ex = (real_cuLaunchKernelEx_t)dlsym(lc, "cuLaunchKernelEx");
    }
    if (!g_real_launch_ex) {
        g_real_launch_ex = (real_cuLaunchKernelEx_t)dlsym(RTLD_DEFAULT, "cuLaunchKernelEx");
    }

    g_initialized.store(true);
    fprintf(stderr, "[CIPHER F1] cuLaunchKernel hook installed. "
                    "real_launch=%p shim=%p\n",
            (void*)g_real_launch, (void*)cipher_launch_kernel_shim);
    return CUDA_SUCCESS;
}

// ---------------------------------------------------------------------------
// cipher_f1_ring_write — called from dispatch_and_log in libcipher_hook.so
// Writes a ring entry for Stage 1 consumption. No kernel launch involved.
// ---------------------------------------------------------------------------

extern "C" __attribute__((visibility("default")))
void cipher_f1_ring_write(uint8_t op_class, float confidence, uint64_t seq) {
    if (!g_cipher_10ops.initialized.load(std::memory_order_acquire)) return;
    CipherRingEntry ev = {};
    ev.sequence     = seq;
    ev.kernel_class = op_class;
    ev.confidence   = confidence;
    ev.decision     = 0;
    if (cipher_ring_write(&g_cipher_10ops.ring, &ev))
        g_cipher_10ops.ring_writes.fetch_add(1, std::memory_order_relaxed);
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
    if (tls_launch_ex_config && g_real_launch_ex)
        return g_real_launch_ex((void*)tls_launch_ex_config, d->fn, d->params, d->extra);
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
    // Initialize 10-operation Stage 1 + Stage 2 threads
    cipher_10ops_init();
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
