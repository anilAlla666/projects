// CIPHER pointer-flow substitution — Stage 3.
//
// Suppression window driven by recipes from Stage 2.  Per-stream state
// tracks whether we're inside a known sequence; on the final step we
// invoke cipher_fused_rmsnorm with the captured (X, W, Y) pointers and
// suppress all 6 launches.
//
// Important: extracting the actual X/W/Y device pointers from ATen's
// kernel args is non-trivial because PyTorch passes them inside
// TensorIterator/ReduceOp structs (the first 8 bytes of args[i] is a
// vtable or strides field, not a data pointer).  Stage 2 leaves those
// pointers nil.  This file therefore SHIPS WITH A SAFE STUB:
//
//   * When CIPHER_FLOW_SUBSTITUTE=on AND a recipe matches AND we have
//     extracted X/W/Y, we DO suppress + dispatch fused.
//   * Otherwise we fall back to PASSTHROUGH (return 0, real launch
//     happens) and merely COUNT the would-have-suppressed window.
//
// This preserves correctness on every supported model and exposes the
// "wired but waiting on pointer extraction" wins via stats.

#include "cipher_flow_substitute.h"
#include "cipher_flow_patterns.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_started{0};
std::atomic<uint64_t> g_window_open{0};
std::atomic<uint64_t> g_window_complete{0};
std::atomic<uint64_t> g_window_aborted{0};
std::atomic<uint64_t> g_substitute_real{0};
std::atomic<uint64_t> g_substitute_stub{0};

bool env_truthy(const char* v) {
    if (!v) return false;
    return v[0]=='1' || v[0]=='o' || v[0]=='O' || v[0]=='t' || v[0]=='T';
}
uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

constexpr uint64_t MAX_GAP_NS_NORMAL = 100ULL * 1000ULL;          // 100us
constexpr uint64_t MAX_GAP_NS_SCAN   = 50ULL * 1000ULL * 1000ULL; // 50ms during scan probe (fprintf is slow)
constexpr int      SCAN_PER_ARG_BYTES = 128;
constexpr int      SCAN_MAX_ARGS = 8;
constexpr int      SCAN_MAX_STEPS = 6;
constexpr int      SCAN_CANDS_PER_STEP = 32;

struct WindowCtx {
    int      recipe_id = -1;
    int      step      = 0;     // next expected K index (we anchor on K1)
    uint64_t stream    = 0;
    uint64_t last_ns   = 0;
    uint64_t x_ptr     = 0;
    uint64_t w_ptr     = 0;
    uint64_t y_ptr     = 0;
    int      rows      = 0;
    int      hidden    = 0;
    float    eps       = 1e-5f;

    // Scan probe: captured device-pointer candidates per step.
    // step_cands[s][i] holds candidate ptrs found in step s.
    int      n_cands[SCAN_MAX_STEPS] = {0};
    uint64_t cands[SCAN_MAX_STEPS][SCAN_CANDS_PER_STEP] = {{0}};
};
thread_local WindowCtx t_win;

std::atomic<int> g_scan_enabled{-1};   // -1 = lazy, 0/1 = resolved
std::atomic<uint64_t> g_scan_dumps{0};
constexpr int SCAN_DUMP_LIMIT = 3;       // first 3 windows only

// dlsym'd pointer to the rt's fused RMSNorm — only invoked when we have
// real X/W/Y.
typedef int (*fused_rmsnorm_fn)(void*, void*, void*, int, int, float, void*);
static fused_rmsnorm_fn g_fused = nullptr;
static int              g_fused_resolved = 0;
fused_rmsnorm_fn resolve_fused() {
    if (g_fused_resolved) return g_fused;
    g_fused = (fused_rmsnorm_fn)dlsym(RTLD_DEFAULT, "cipher_fused_rmsnorm");
    g_fused_resolved = 1;
    return g_fused;
}

void abort_window(const char* reason) {
    if (t_win.recipe_id >= 0) {
        g_window_aborted.fetch_add(1, std::memory_order_relaxed);
        static uint64_t s_log = 0;
        if (++s_log < 5) {
            fprintf(stderr,
                "[CIPHER FLOW SUBST] window abort step=%d reason=%s\n",
                t_win.step, reason);
        }
    }
    t_win = WindowCtx{};
}

bool scan_enabled() {
    int v = g_scan_enabled.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* s = getenv("CIPHER_FLOW_SCAN");
        v = (s && (s[0]=='1'||s[0]=='o'||s[0]=='O'||s[0]=='t'||s[0]=='T')) ? 1 : 0;
        g_scan_enabled.store(v, std::memory_order_relaxed);
    }
    return v == 1;
}

// Scan first 128 bytes of each args[i] for 8-byte-aligned values that
// look like device pointers (canonical CUDA VA range).  Page-bounded
// reads — bail if we'd cross a page boundary.
void scan_step_args(int step_idx, void** args) {
    if (!args || step_idx < 0 || step_idx >= SCAN_MAX_STEPS) return;
    auto& cands = t_win.cands[step_idx];
    int& n = t_win.n_cands[step_idx];
    n = 0;
    constexpr uintptr_t PAGE_MASK = ~((uintptr_t)4095);
    constexpr uintptr_t PAGE_SIZE = 4096;
    constexpr uint64_t  DEV_LO = 0x100000000ULL;
    constexpr uint64_t  DEV_HI = 0x800000000000ULL;
    // Use mincore to verify the page is resident before reading.
    for (int i = 0; i < SCAN_MAX_ARGS; ++i) {
        if (!args[i]) break;
        uintptr_t base = (uintptr_t)args[i];
        if (base < 0x10000) break;
        // Restrict reads to args that fall on a single page (no cross-page
        // reads).  Compute how many bytes lie before the next page boundary.
        uintptr_t page_end = (base & PAGE_MASK) + PAGE_SIZE;
        size_t safe_bytes = (page_end > base) ? (page_end - base) : 0;
        if (safe_bytes > SCAN_PER_ARG_BYTES) safe_bytes = SCAN_PER_ARG_BYTES;
        // mincore probe: is this page resident?
        unsigned char vec = 0;
        void* page_start = (void*)(base & PAGE_MASK);
        if (mincore(page_start, PAGE_SIZE, &vec) != 0 || (vec & 1) == 0) {
            // not resident or query failed — skip
            continue;
        }
        // Walk in 8-byte strides.
        for (size_t off = 0; off + 8 <= safe_bytes; off += 8) {
            uint64_t v = 0;
            memcpy(&v, (uint8_t*)args[i] + off, 8);
            if (v < DEV_LO || v >= DEV_HI) continue;
            // Filter: ignore stack-looking pointers (0x7f...).
            if ((v & 0xFF00000000000000ULL) >= 0x7F00000000000000ULL)
                continue;
            // Dedupe within this step.
            bool dup = false;
            for (int j = 0; j < n; ++j) if (cands[j] == v) { dup = true; break; }
            if (!dup && n < SCAN_CANDS_PER_STEP) cands[n++] = v;
        }
    }
}

// Read up to `n_bytes` from a host pointer if its page is resident.
// Returns bytes actually read (0 on failure).
size_t safe_read_host(const void* p, void* dst, size_t n_bytes) {
    if (!p || !dst) return 0;
    constexpr uintptr_t PAGE_MASK = ~((uintptr_t)4095);
    constexpr uintptr_t PAGE_SIZE = 4096;
    uintptr_t base = (uintptr_t)p;
    if (base < 0x10000) return 0;
    uintptr_t page_end = (base & PAGE_MASK) + PAGE_SIZE;
    size_t safe = (page_end > base) ? (page_end - base) : 0;
    if (safe > n_bytes) safe = n_bytes;
    unsigned char vec = 0;
    if (mincore((void*)(base & PAGE_MASK), PAGE_SIZE, &vec) != 0 ||
        (vec & 1) == 0) return 0;
    memcpy(dst, p, safe);
    return safe;
}

// True for "looks like a HOST pointer" (heap or lib): excludes CUDA VA.
bool looks_like_host_ptr(uint64_t v) {
    // Heap: 0x55xxxxx..0x66xxxxx range (Linux ASLR).
    // Stack/lib: 0x7fxxxxxxxxxx range.
    // Reject CUDA VA (0x1a4xxxxxx etc.) — that's < 0x100000000000.
    if (v < 0x500000000000ULL) return false;
    if (v >= 0x800000000000ULL) return false;   // canonical user max
    return true;
}
bool looks_like_dev_ptr_2(uint64_t v) {
    // CUDA VA on this H100 pod is 0x1a48_xxxx_xxxx range.  Tight filter
    // on 0x10_0000_0000_0000 .. 0x40_0000_0000_0000 (1 TB .. 64 TB) to
    // avoid stack-frame noise that just happens to fit canonical user.
    return v >= 0x100000000000ULL && v < 0x400000000000ULL;
}

// Deep raw dump of args[i] up to N bytes; for any 8-byte slot that looks
// like a HOST pointer, follow it one level of indirection and dump 64
// bytes from that target, scanning for device pointers.
void scan_raw_dump_deep(int step_idx, int arg_idx, void** args,
                       const char* tag, size_t n_bytes) {
    if (!args || !args[arg_idx]) return;
    fprintf(stderr,
        "[CIPHER FLOW DEEP] %s K%d args[%d] @ %p (%zu bytes, with 1-level deref):\n",
        tag, step_idx, arg_idx, args[arg_idx], n_bytes);
    uint8_t buf[256] = {0};
    size_t got = safe_read_host(args[arg_idx], buf, n_bytes);
    if (got == 0) {
        fprintf(stderr, "  (page not resident)\n");
        return;
    }
    // Hex dump of the entire window.
    fprintf(stderr, "  bytes:");
    for (size_t off = 0; off < got; ++off) {
        if ((off & 31) == 0) fprintf(stderr, "\n  +%3zu:", off);
        if ((off & 7) == 0) fprintf(stderr, " ");
        fprintf(stderr, "%02x", buf[off]);
    }
    fprintf(stderr, "\n");
    // Walk EVERY byte offset (1-byte stride) — args[i] may be unaligned,
    // so 8-byte stride from offset 0 could miss aligned values.  Only log
    // matches (DEV / HOST) to keep output bounded.
    int n_dev_top = 0, n_host_top = 0;
    for (size_t off = 0; off + 8 <= got; off += 1) {
        uint64_t v = 0; memcpy(&v, buf + off, 8);
        if (looks_like_dev_ptr_2(v)) {
            fprintf(stderr, "  +%3zu = 0x%016lx [DEV]\n",
                off, (unsigned long)v);
            n_dev_top++;
        } else if (looks_like_host_ptr(v)) {
            // Follow one level of indirection.
            uint8_t deep[64] = {0};
            size_t deep_got = safe_read_host((void*)v, deep, sizeof(deep));
            if (deep_got >= 8) {
                int n_dev_deep = 0;
                for (size_t doff = 0; doff + 8 <= deep_got; doff += 1) {
                    uint64_t dv = 0; memcpy(&dv, deep + doff, 8);
                    if (looks_like_dev_ptr_2(dv)) {
                        fprintf(stderr,
                            "  +%3zu = 0x%016lx [HOST] -> +%2zu = 0x%016lx [DEV]\n",
                            off, (unsigned long)v, doff, (unsigned long)dv);
                        n_dev_deep++;
                        n_dev_top++;
                        if (n_dev_deep >= 4) break;  // bound output
                    }
                }
                if (n_dev_deep > 0) n_host_top++;
            }
        }
    }
    fprintf(stderr,
        "  [scan summary] direct DEV ptrs: %d, HOST ptrs leading to DEV: %d\n",
        n_dev_top, n_host_top);
}

// Raw hex dump of args[0..3] for one step.  No filtering.  Used to see
// exactly what bytes ATen passes on the kernel param stack.  Especially
// useful for the BinaryFunctor<Mul> Scale step, where args[2] is
// `std::array<char*, 3>` = [Y, normed_x, W] — 24 bytes of pointers.
void scan_raw_dump_step(int step_idx, void** args, const char* tag) {
    if (!args) return;
    fprintf(stderr,
        "[CIPHER FLOW RAW] %s step=K%d  raw args dump:\n",
        tag, step_idx);
    constexpr uintptr_t PAGE_MASK = ~((uintptr_t)4095);
    constexpr uintptr_t PAGE_SIZE = 4096;
    for (int i = 0; i < 6; ++i) {
        if (!args[i]) {
            fprintf(stderr, "  args[%d] = NULL\n", i);
            break;
        }
        uintptr_t base = (uintptr_t)args[i];
        if (base < 0x10000) {
            fprintf(stderr, "  args[%d] = 0x%lx (low — skipped)\n",
                i, (unsigned long)base);
            continue;
        }
        unsigned char vec = 0;
        void* page_start = (void*)(base & PAGE_MASK);
        if (mincore(page_start, PAGE_SIZE, &vec) != 0 || (vec & 1) == 0) {
            fprintf(stderr, "  args[%d] = 0x%lx (page not resident)\n",
                i, (unsigned long)base);
            continue;
        }
        // How many bytes can we safely read on this page?
        uintptr_t page_end = (base & PAGE_MASK) + PAGE_SIZE;
        size_t safe = (page_end > base) ? (page_end - base) : 0;
        if (safe > 64) safe = 64;   // dump up to 64 bytes per arg
        uint8_t buf[64] = {0};
        memcpy(buf, args[i], safe);
        fprintf(stderr, "  args[%d] @ 0x%lx (%zu bytes):", i,
            (unsigned long)base, safe);
        for (size_t off = 0; off < safe; ++off) {
            if ((off & 7) == 0) fprintf(stderr, " ");
            fprintf(stderr, "%02x", buf[off]);
        }
        fprintf(stderr, "\n");
        // Interpret as up to 8 uint64_t values.
        for (size_t off = 0; off + 8 <= safe; off += 8) {
            uint64_t v = 0; memcpy(&v, buf + off, 8);
            fprintf(stderr, "    +%2zu = 0x%016lx\n",
                off, (unsigned long)v);
        }
    }
    fprintf(stderr, "\n");
}

// Cross-reference scan results across the 6 steps and log candidates
// for X / W / Y based on the brief's rules.  Logs first 3 windows only.
void scan_dump_and_classify(uint64_t window_seq) {
    int n_steps = 0;
    for (int s = 0; s < SCAN_MAX_STEPS; ++s) if (t_win.n_cands[s] > 0) n_steps++;
    if (n_steps < 4) return;  // not enough data

    uint64_t dump_n = g_scan_dumps.fetch_add(1, std::memory_order_relaxed);
    if (dump_n >= SCAN_DUMP_LIMIT) return;

    fprintf(stderr,
        "\n[CIPHER FLOW SCAN] window=%llu  candidates per step:\n",
        (unsigned long long)window_seq);
    for (int s = 0; s < SCAN_MAX_STEPS; ++s) {
        if (t_win.n_cands[s] == 0) continue;
        fprintf(stderr, "  K%d: %d cands [", s, t_win.n_cands[s]);
        for (int i = 0; i < t_win.n_cands[s]; ++i) {
            fprintf(stderr, " 0x%lx",
                (unsigned long)t_win.cands[s][i]);
        }
        fprintf(stderr, " ]\n");
    }

    // Cross-reference: count how many steps each unique pointer appears in.
    // We pool all unique candidates first, then re-scan to count occurrences.
    uint64_t pool[SCAN_MAX_STEPS * SCAN_CANDS_PER_STEP];
    int      pool_n = 0;
    int      pool_steps[SCAN_MAX_STEPS * SCAN_CANDS_PER_STEP] = {0};
    int      first_seen[SCAN_MAX_STEPS * SCAN_CANDS_PER_STEP] = {0};
    for (int s = 0; s < SCAN_MAX_STEPS; ++s) {
        for (int i = 0; i < t_win.n_cands[s]; ++i) {
            uint64_t p = t_win.cands[s][i];
            int found = -1;
            for (int j = 0; j < pool_n; ++j)
                if (pool[j] == p) { found = j; break; }
            if (found < 0 && pool_n < (int)(sizeof(pool)/sizeof(pool[0]))) {
                pool[pool_n] = p;
                pool_steps[pool_n] = (1 << s);
                first_seen[pool_n] = s;
                pool_n++;
            } else if (found >= 0) {
                pool_steps[found] |= (1 << s);
            }
        }
    }
    fprintf(stderr,
        "  cross-ref (pool_n=%d):\n", pool_n);
    for (int j = 0; j < pool_n; ++j) {
        int popc = __builtin_popcount(pool_steps[j]);
        const char* role = "candidate";
        if (popc >= 2 && (pool_steps[j] & (1<<0))) role = "X (multi-step, K0)";
        else if (popc >= 2 && first_seen[j] <= 1) role = "X-or-intermediate";
        else if (popc == 1 && first_seen[j] == SCAN_MAX_STEPS - 1)
            role = "Y-or-W (K5 only)";
        else if (popc == 2 && (pool_steps[j] & ((1<<1)|(1<<2)))) role = "intermediate";
        fprintf(stderr,
            "    0x%016lx  step_mask=0x%02x first=K%d popc=%d  -> %s\n",
            (unsigned long)pool[j],
            pool_steps[j], first_seen[j], popc, role);
    }
    fprintf(stderr, "\n");
}

} // namespace

extern "C" int cipher_flow_substitute_init(void) {
    if (g_started.exchange(1, std::memory_order_acq_rel))
        return g_enabled.load(std::memory_order_relaxed);
    int on = env_truthy(getenv("CIPHER_FLOW_SUBSTITUTE"));
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        fprintf(stderr,
            "[CIPHER FLOW SUBST] enabled (suppression-aware, dispatches "
            "cipher_fused_rmsnorm when X/W/Y extracted)\n");
    }
    return on;
}

extern "C" int cipher_flow_substitute_enabled(void) {
    return g_enabled.load(std::memory_order_relaxed);
}

extern "C" int cipher_flow_substitute_consider(
    void* func, uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t smem, void** args, void* stream)
{
    (void)gy; (void)gz; (void)bx; (void)by; (void)bz; (void)smem; (void)args;
    if (!g_enabled.load(std::memory_order_relaxed)) return 0;

    uint64_t now = now_ns();
    uint64_t stream_id = (uint64_t)stream;

    // Stream / timeout safety on partial windows.  When the scan probe
    // is on, fprintf-heavy dumps inflate inter-launch latency, so we use
    // a much wider window to keep the probe useful.
    if (t_win.recipe_id >= 0) {
        uint64_t max_gap = scan_enabled() ? MAX_GAP_NS_SCAN
                                          : MAX_GAP_NS_NORMAL;
        if (t_win.stream != stream_id) { abort_window("stream-change"); }
        else if (now - t_win.last_ns > max_gap) { abort_window("timeout"); }
    }

    // ── Window not open: try to start one ──────────────────────────────
    if (t_win.recipe_id < 0) {
        int rid = cipher_flow_patterns_lookup_first(func);
        if (rid < 0) return 0;       // no recipe match — pass through
        const CipherFlowRecipe* r = cipher_flow_patterns_get_recipe(rid);
        if (!r || !r->valid) return 0;
        t_win.recipe_id = rid;
        // We just matched K1 (MeanOps); the NEXT call should match K2.
        t_win.step      = 2;
        t_win.stream    = stream_id;
        t_win.last_ns   = now;
        t_win.rows      = (int)gx;    // grid.x ≈ rows for the reduce kernel
        t_win.eps       = r->eps;
        g_window_open.fetch_add(1, std::memory_order_relaxed);
        // Probe: scan K1 args.
        if (scan_enabled()) scan_step_args(1, args);
        // For the SAFE STUB path: do NOT suppress.  We let the real
        // sequence run while we count would-be-suppressions.  (Real
        // suppression requires extracting X/W/Y from struct args.)
        return 0;
    }

    // ── Window open: verify the next func_ptr matches the recipe ───────
    const CipherFlowRecipe* r =
        cipher_flow_patterns_get_recipe(t_win.recipe_id);
    if (!r || !r->valid) { abort_window("recipe-gone"); return 0; }
    if (t_win.step >= 7) { abort_window("over-step"); return 0; }
    if (r->fns[t_win.step] != func) {
        abort_window("step-mismatch"); return 0;
    }
    // Probe: scan this step's args before advancing.
    if (scan_enabled()) {
        scan_step_args(t_win.step, args);
        // Raw-byte dump for first 5 windows.
        uint64_t wn = g_window_complete.load(std::memory_order_relaxed);
        if (wn < 5 && t_win.step >= 2 && t_win.step <= 6) {
            const char* tag = (t_win.step == 6) ? "K6-Scale*W" :
                              (t_win.step == 5) ? "K5-MulRsq " :
                              (t_win.step == 4) ? "K4-Rsqrt  " :
                              (t_win.step == 3) ? "K3-AddEps " :
                                                   "K2-Anchor ";
            scan_raw_dump_step(t_win.step, args, tag);
        }
        // DEEP dump for K5 args[1] (the lambda capture) for first 3
        // windows.  Reads 256 bytes and follows one level of host-pointer
        // indirection looking for device pointers.
        if (wn < 3 && t_win.step == 5) {
            scan_raw_dump_deep(5, 1, args, "K5-LAMBDA  ", 256);
        }
    }
    t_win.last_ns = now;
    t_win.step++;

    if (t_win.step >= 7) {
        // Sequence complete (now 6 follow-up steps after MeanReduce
        // anchor: K2..K6).
        if (scan_enabled()) {
            scan_dump_and_classify(g_window_complete.load(std::memory_order_relaxed));
        }
        g_window_complete.fetch_add(1, std::memory_order_relaxed);
        bool can_real = (t_win.x_ptr != 0 && t_win.w_ptr != 0 &&
                          t_win.y_ptr != 0 && t_win.hidden > 0 &&
                          t_win.rows > 0 && resolve_fused() != nullptr);
        if (can_real) {
            int rc = g_fused((void*)t_win.x_ptr, (void*)t_win.w_ptr,
                             (void*)t_win.y_ptr,
                             t_win.rows, t_win.hidden, t_win.eps,
                             stream);
            if (rc == 1) {
                g_substitute_real.fetch_add(1, std::memory_order_relaxed);
                t_win = WindowCtx{};
                return 1;   // suppress this last launch — fused already ran
            }
        }
        g_substitute_stub.fetch_add(1, std::memory_order_relaxed);
        static uint64_t s_log = 0;
        if (++s_log < 3) {
            fprintf(stderr,
                "[CIPHER FLOW SUBST] STUB count #%llu  recipe=%d  "
                "(X/W/Y not yet extracted from struct args — passthrough)\n",
                (unsigned long long)s_log, t_win.recipe_id);
        }
        t_win = WindowCtx{};
        return 0;
    }
    // Mid-window: we've verified this kernel matches the recipe step.
    // SAFE STUB: do NOT suppress.  Real suppression would `return 1` here
    // for steps 1..5 once X/W/Y extraction is wired.
    return 0;
}

extern "C" void cipher_flow_substitute_report(void) {
    fprintf(stderr,
        "[CIPHER FLOW SUBST] report: windows_opened=%llu "
        "completed=%llu aborted=%llu  real_substituted=%llu  "
        "stub_count=%llu\n",
        (unsigned long long)g_window_open.load(),
        (unsigned long long)g_window_complete.load(),
        (unsigned long long)g_window_aborted.load(),
        (unsigned long long)g_substitute_real.load(),
        (unsigned long long)g_substitute_stub.load());
}

__attribute__((constructor(117)))
static void cipher_flow_substitute_autoinit() {
    cipher_flow_substitute_init();
}

__attribute__((destructor))
static void cipher_flow_substitute_autoreport() {
    if (g_enabled.load(std::memory_order_relaxed))
        cipher_flow_substitute_report();
}
