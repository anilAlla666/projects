// =============================================================================
// CIPHER — Persistent Kernel Mode (Change 2)
// src/cipher_persist.cpp
//
// Resides in libcipher_hook.so for zero-cross-DSO overhead on the shim hot
// path.
//
// Mechanism:
//   - fingerprint  := FNV-1a of (fn_ptr, grid{x,y,z}, block{x,y,z}, shmem)
//   - per-thread rolling history of the last 64 launch fingerprints
//   - tandem-repeat detector at L ∈ {4, 8, 16}: checks
//       hist[now-L..now) == hist[now-2L..now-L)
//     and counts how many consecutive launches satisfy the property
//   - when a per-thread counter reaches PROMOTION_THRESHOLD (=10), the
//     detected L-fingerprint signature is copied into a process-global
//     StableBlock table (small mutex taken only on promotion)
//   - on subsequent calls, cipher_persist_try_fast_path(fp) checks:
//       · if we are mid-replay in a stable block: verify fp matches and
//         advance; on mismatch, abort replay and fall through
//       · otherwise: scan the stable table for any block whose sig[0]==fp;
//         on match, enter replay mode for the next L-1 calls
//   - "fast path" means the shim skips classify, dispatch, ring write, and
//     the ~2-5 µs of CIPHER overhead that go with them.  It still calls the
//     real CUDA launch entry point.
//
// Drift rule ✓: fingerprints use only fn_ptr and launch geometry.
// =============================================================================

#include "cipher_persist.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

// ---- Tunables ---------------------------------------------------------------

constexpr int      HIST_LEN             = 64;
constexpr int      MAX_BLOCKS           = 32;
constexpr int      PROMOTION_THRESHOLD  = 10;    // spec: "repeating 10+ times"
constexpr int      NUM_L                = 3;
constexpr uint32_t SEQ_LENGTHS[NUM_L]   = { 4, 8, 16 };

// ---- Per-thread detector state ----------------------------------------------

struct PerThreadState {
    uint64_t hist[HIST_LEN];
    uint32_t hist_count;               // monotonic (uint32 overflow is fine)

    // Tandem-repeat consecutive-hit counters, one per L
    uint32_t stable_count[NUM_L];

    // Active replay cursor
    int      replay_block;             // -1 if not replaying
    uint32_t replay_step;              // next index within sig[]
    uint32_t replay_len;               // L of the active block
};

thread_local PerThreadState t_state = {
    /*hist*/ {},
    /*hist_count*/ 0,
    /*stable_count*/ {0, 0, 0},
    /*replay_block*/ -1,
    /*replay_step*/ 0,
    /*replay_len*/ 0,
};

// ---- Global stable-block table ----------------------------------------------

struct StableBlock {
    uint8_t               L;            // 4, 8, or 16
    uint64_t              sig[16];      // length L used; trailing unused
    std::atomic<uint64_t> hits;         // total fast-path entries using this
    bool                  active;
};

static StableBlock      g_blocks[MAX_BLOCKS];
static std::atomic<int> g_n_blocks{0};
static std::mutex       g_promote_mtx;

// ---- Global counters (lock-free) --------------------------------------------

static std::atomic<uint64_t> g_fast_path_count   {0};
static std::atomic<uint64_t> g_promotion_count   {0};
static std::atomic<uint64_t> g_observe_count     {0};

// ---- Enable flag (read once, cached) ----------------------------------------

static int g_enabled = -1;    // -1 uninit, 0 off, 1 on (default on)

static bool enabled_cached() {
    int v = g_enabled;
    if (v >= 0) return v == 1;
    const char* e = std::getenv("CIPHER_PERSIST");
    v = (e && e[0] == '0') ? 0 : 1;    // default ON
    g_enabled = v;
    return v == 1;
}

// ---- Fingerprint: FNV-1a 64-bit over 8 uint64_t --------------------------

inline uint64_t fnv1a_u64_mix(uint64_t h, uint64_t v) {
    // Process v byte-by-byte; constant-time, no branching
    const uint64_t prime = 1099511628211ULL;
    for (int i = 0; i < 8; i++) {
        h ^= (v >> (i * 8)) & 0xFFull;
        h *= prime;
    }
    return h;
}

inline uint64_t compute_fingerprint(
    const void* fn,
    unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shared_bytes)
{
    uint64_t h = 14695981039346656037ULL;       // FNV offset basis
    h = fnv1a_u64_mix(h, (uint64_t)(uintptr_t)fn);
    h = fnv1a_u64_mix(h, ((uint64_t)gx) | ((uint64_t)gy << 32));
    h = fnv1a_u64_mix(h, ((uint64_t)gz) | ((uint64_t)bx << 32));
    h = fnv1a_u64_mix(h, ((uint64_t)by) | ((uint64_t)bz << 32));
    h = fnv1a_u64_mix(h, (uint64_t)shared_bytes);
    return h;
}

// Tandem-repeat check at length L over the thread-local history.
// Returns true iff the last 2L fingerprints form the pattern X|X.
inline bool tandem_repeat_at(const PerThreadState& s, int L) {
    if (s.hist_count < (uint32_t)(2 * L)) return false;
    // hist is a circular buffer; positions from newest back:
    //   newest at index (hist_count - 1) % HIST_LEN
    // We need indices [count-2L, count-L) to equal [count-L, count).
    for (int i = 0; i < L; i++) {
        uint32_t recent_idx = (s.hist_count - L + i) % HIST_LEN;
        uint32_t older_idx  = (s.hist_count - 2 * L + i) % HIST_LEN;
        if (s.hist[recent_idx] != s.hist[older_idx]) return false;
    }
    return true;
}

// Copy the most-recent L fingerprints (in chronological order) out of the
// history buffer.
inline void copy_recent(const PerThreadState& s, int L, uint64_t* out) {
    for (int i = 0; i < L; i++) {
        uint32_t idx = (s.hist_count - L + i) % HIST_LEN;
        out[i] = s.hist[idx];
    }
}

// Promotion: copy signature into the global table.  Caller holds no lock.
void promote(int L, const uint64_t* sig) {
    if (L < 1 || L > 16) return;
    std::lock_guard<std::mutex> lock(g_promote_mtx);

    // Avoid duplicates — scan existing active blocks for matching signature.
    int n = g_n_blocks.load(std::memory_order_acquire);
    for (int i = 0; i < n; i++) {
        if (!g_blocks[i].active) continue;
        if (g_blocks[i].L != (uint8_t)L) continue;
        bool same = true;
        for (int k = 0; k < L; k++) {
            if (g_blocks[i].sig[k] != sig[k]) { same = false; break; }
        }
        if (same) return;   // already known, nothing to do
    }

    if (n >= MAX_BLOCKS) return;
    StableBlock& b = g_blocks[n];
    b.L = (uint8_t)L;
    memset(b.sig, 0, sizeof(b.sig));
    for (int k = 0; k < L; k++) b.sig[k] = sig[k];
    b.hits.store(0, std::memory_order_relaxed);
    b.active = true;
    g_n_blocks.store(n + 1, std::memory_order_release);
    g_promotion_count.fetch_add(1, std::memory_order_relaxed);

    fprintf(stderr,
            "[CIPHER PERSIST] PROMOTED block #%d L=%d first_fp=%016lx "
            "(n_blocks=%d)\n",
            n, L, (unsigned long)sig[0], n + 1);
}

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

extern "C" uint64_t cipher_persist_fingerprint(
    const void* fn,
    unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shared_bytes)
{
    return compute_fingerprint(fn, gx, gy, gz, bx, by, bz, shared_bytes);
}

extern "C" bool cipher_persist_try_fast_path(uint64_t fp) {
    if (!enabled_cached()) return false;

    PerThreadState& s = t_state;

    // Case 1 — already mid-replay: verify fp matches the expected slot.
    if (s.replay_block >= 0) {
        const StableBlock& b = g_blocks[s.replay_block];
        if (s.replay_step < s.replay_len && fp == b.sig[s.replay_step]) {
            s.replay_step++;
            g_fast_path_count.fetch_add(1, std::memory_order_relaxed);
            // End of this cycle — rearm for next repetition.
            if (s.replay_step >= s.replay_len) {
                s.replay_block = -1;
                s.replay_step  = 0;
                s.replay_len   = 0;
            }
            return true;
        }
        // Mismatch — sequence broken, fall through to normal path.
        s.replay_block = -1;
        s.replay_step  = 0;
        s.replay_len   = 0;
        return false;
    }

    // Case 2 — not in a replay.  Check whether fp starts any stable block.
    int n = g_n_blocks.load(std::memory_order_acquire);
    for (int i = 0; i < n; i++) {
        const StableBlock& b = g_blocks[i];
        if (!b.active) continue;
        if (b.sig[0] != fp) continue;
        // Enter replay mode
        s.replay_block = i;
        s.replay_step  = 1;                 // we just consumed sig[0]
        s.replay_len   = b.L;
        g_blocks[i].hits.fetch_add(1, std::memory_order_relaxed);
        g_fast_path_count.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    return false;
}

extern "C" void cipher_persist_observe(uint64_t fp) {
    if (!enabled_cached()) return;

    PerThreadState& s = t_state;

    // Append to circular history
    s.hist[s.hist_count % HIST_LEN] = fp;
    s.hist_count++;
    g_observe_count.fetch_add(1, std::memory_order_relaxed);

    // Check tandem repeats at each L and promote on threshold.
    for (int li = 0; li < NUM_L; li++) {
        int L = (int)SEQ_LENGTHS[li];
        if (tandem_repeat_at(s, L)) {
            s.stable_count[li]++;
            if (s.stable_count[li] == PROMOTION_THRESHOLD) {
                uint64_t sig[16] = {};
                copy_recent(s, L, sig);
                promote(L, sig);
            }
        } else {
            s.stable_count[li] = 0;
        }
    }
}

extern "C" uint64_t cipher_persist_fast_path_count(void) {
    return g_fast_path_count.load(std::memory_order_relaxed);
}

extern "C" uint64_t cipher_persist_promotion_count(void) {
    return g_promotion_count.load(std::memory_order_relaxed);
}

extern "C" uint64_t cipher_persist_observe_count(void) {
    return g_observe_count.load(std::memory_order_relaxed);
}

extern "C" bool cipher_persist_enabled(void) {
    return enabled_cached();
}

extern "C" void cipher_persist_report(void) {
    int n = g_n_blocks.load(std::memory_order_acquire);
    uint64_t fp_count = g_fast_path_count.load(std::memory_order_relaxed);
    uint64_t prom     = g_promotion_count.load(std::memory_order_relaxed);
    uint64_t obs      = g_observe_count.load(std::memory_order_relaxed);
    fprintf(stderr,
        "[CIPHER PERSIST REPORT] enabled=%d blocks=%d "
        "observe=%llu promotions=%llu fast_path=%llu\n",
        enabled_cached() ? 1 : 0, n,
        (unsigned long long)obs,
        (unsigned long long)prom,
        (unsigned long long)fp_count);
    for (int i = 0; i < n; i++) {
        const StableBlock& b = g_blocks[i];
        fprintf(stderr,
            "  block[%d] L=%u hits=%llu sig[0]=%016lx\n",
            i, (unsigned)b.L,
            (unsigned long long)b.hits.load(std::memory_order_relaxed),
            (unsigned long)b.sig[0]);
    }
}
