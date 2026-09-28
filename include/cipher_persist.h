// =============================================================================
// CIPHER — Persistent Kernel Mode (Change 2)
// cipher_persist.h
//
// Detects stable repeating launch sequences and bypasses CIPHER's per-launch
// classify/dispatch/ring overhead on recognised repeats. Does NOT cache kernel
// arguments or replay real CUDA launches — every launch still goes through
// g_real_cuLaunch*. The speedup is pure CPU overhead elimination on the shim.
//
// Geometry only — keyed on (fn_ptr, grid, block, shmem). No model / layer /
// kernel-name references.
//
// All state is confined to libcipher_hook.so; this header is the only public
// surface.
// =============================================================================

#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compute a 64-bit fingerprint for one kernel launch. Pure geometry.
// Cheap (< 20 ns) — called on every cuLaunchKernelEx.
uint64_t cipher_persist_fingerprint(
    const void* fn,
    unsigned    gx, unsigned gy, unsigned gz,
    unsigned    bx, unsigned by, unsigned bz,
    unsigned    shared_bytes);

// Check if this launch should take the fast path (skip classify/dispatch/ring).
// Returns true if the caller should just invoke the real launch and return.
// Advances per-thread state if we are mid-sequence.
bool cipher_persist_try_fast_path(uint64_t fp);

// Feed a launch fingerprint into the repeat detector.  Called AFTER the
// normal-path launch succeeds.  May promote a new stable block as a side
// effect (rare).
void cipher_persist_observe(uint64_t fp);

// Counters for test and diagnostic use.
uint64_t cipher_persist_fast_path_count(void);
uint64_t cipher_persist_promotion_count(void);
uint64_t cipher_persist_observe_count(void);

// Human-readable report (stderr).
void cipher_persist_report(void);

// Whether Change 2 is enabled (env CIPHER_PERSIST; default ON).
bool cipher_persist_enabled(void);

#ifdef __cplusplus
}
#endif
