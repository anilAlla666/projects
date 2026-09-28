// CIPHER pointer-flow pattern matcher — Stage 2.
//
// Anchor-driven state machine.  Trigger is `reduce_kernel<...MeanOps...>`,
// which is the variance reduction at the heart of RMSNorm.  From there we
// walk forward 4 launches (eps-add, rsqrt, multiply-by-rsqrt, scale-by-
// weight); each step verifies the data-flow edge before transitioning.
// On full match, the 6 func_ptrs become a Recipe; Stage 3 suppresses the
// sequence and calls cipher_fused_rmsnorm directly.

#include "cipher_flow_patterns.h"
#include "cipher_flow_recorder.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>

extern "C" const char* cipher_kt_name_for(void* fn_handle);

namespace {

constexpr int      MAX_RECIPES = 8;
constexpr uint64_t MAX_GAP_NS  = 100ULL * 1000ULL;   // 100us

CipherFlowRecipe g_recipes[MAX_RECIPES];
std::atomic<int> g_n_recipes{0};
std::mutex       g_recipes_mu;

std::atomic<int>      g_enabled{0};
std::atomic<int>      g_started{0};
std::atomic<uint64_t> g_anchor_hits{0};
std::atomic<uint64_t> g_match_complete{0};
std::atomic<uint64_t> g_match_aborted{0};

bool env_truthy(const char* v) {
    if (!v) return false;
    return v[0]=='1' || v[0]=='o' || v[0]=='O' || v[0]=='t' || v[0]=='T';
}
uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// State machine — single-stream simplification.  Real PyTorch inference
// is dominated by one stream; multi-stream would need per-stream state.
// Extended to 6 follow-up steps: PyTorch's RMSNorm in eager mode emits
// two distinct multiplications after rsqrt — first the in-place
// `x * rsqrt` (binary, 2-ptr array), then the final `* weight` Scale
// (BinaryFunctor<Mul>, 3-ptr array).
enum MatchState {
    IDLE = 0,
    SAW_MEAN,
    SAW_ADDEPS,
    SAW_RSQRT,
    SAW_MUL,
    SAW_NORMED,   // x*rsqrt observed; next is *weight (the Scale step)
};

struct MatchCtx {
    MatchState state = IDLE;
    void*    fns[7]   = {0};   // K0..K6 — fns[0] always nil (Pow not anchored)
    uint64_t var_in_ptr   = 0;   // ptr that K1 (Pow) wrote / K2 (Mean) reads
    uint64_t mean_out_ptr = 0;
    uint64_t eps_out_ptr  = 0;
    uint64_t rsqrt_out_ptr= 0;
    uint64_t normed_ptr   = 0;
    uint64_t weight_ptr   = 0;
    uint64_t out_ptr      = 0;
    uint64_t x_ptr        = 0;   // INPUT to RMSNorm — back-found from K0
    int      rows_grid    = 0;
    uint32_t numel        = 0;
    float    eps          = 0.0f;
    uint64_t last_ns      = 0;
    uint64_t stream       = 0;
    void*    fn_pre_anchor = nullptr;  // launch right BEFORE MeanOps
};

thread_local MatchCtx t_ctx;

// Pull an output ptr heuristic from a flow entry: the LAST device-pointer
// arg, since most ATen kernels write the result via the last pointer arg
// (or via a TensorIterator wrapped struct — for vectorized_elementwise the
// "data" array (Lm3) holds {out, in0, in1} in that order, but we conservatively
// take "the last DEV-ptr arg" as out).
uint64_t pick_out_ptr(uint64_t a[8], const uint8_t is_ptr[8]) {
    for (int i = 7; i >= 0; --i)
        if (is_ptr[i]) return a[i];
    return 0;
}
uint64_t pick_first_ptr(uint64_t a[8], const uint8_t is_ptr[8]) {
    for (int i = 0; i < 8; ++i)
        if (is_ptr[i]) return a[i];
    return 0;
}

bool name_contains_meanops(const char* nm) {
    if (!nm) return false;
    return strstr(nm, "MeanOps") != nullptr;
}

void reset_ctx(const char* reason) {
    if (t_ctx.state != IDLE) {
        g_match_aborted.fetch_add(1, std::memory_order_relaxed);
        if (g_match_aborted.load() < 5) {
            fprintf(stderr,
                "[CIPHER FLOW MATCH] abort state=%d reason=%s\n",
                (int)t_ctx.state, reason);
        }
    }
    t_ctx = MatchCtx{};
}

// Try to install a recipe; returns -1 if dup, else slot id.
int install_recipe(void* fns[7], int rows_grid, uint32_t numel, float eps) {
    std::lock_guard<std::mutex> lk(g_recipes_mu);
    int n = g_n_recipes.load(std::memory_order_relaxed);
    for (int i = 0; i < n; ++i) {
        bool same = true;
        // Compare slots 1..6 (slot 0 is K0=Pow which we don't anchor).
        for (int k = 1; k < 7; ++k) {
            if (g_recipes[i].fns[k] != fns[k]) { same = false; break; }
        }
        if (same) {
            g_recipes[i].n_observed++;
            return -1;  // already known
        }
    }
    if (n >= MAX_RECIPES) return -1;
    CipherFlowRecipe r{};
    r.valid = 1;
    for (int k = 0; k < 7; ++k) r.fns[k] = fns[k];
    r.eps = eps;
    r.rows_grid_axis = 0;
    r.x_arg_idx_k0 = -1;       // not yet pinned
    r.out_arg_idx_k5 = -1;
    r.weight_arg_idx_k5 = -1;
    r.hidden_arg_idx = -1;
    r.n_observed = 1;
    g_recipes[n] = r;
    g_n_recipes.store(n + 1, std::memory_order_release);
    return n;
}

} // namespace

extern "C" int cipher_flow_patterns_init(void) {
    if (g_started.exchange(1, std::memory_order_acq_rel))
        return g_enabled.load(std::memory_order_relaxed);
    int on = env_truthy(getenv("CIPHER_FLOW_MATCH"));
    g_enabled.store(on, std::memory_order_release);
    if (on) {
        fprintf(stderr,
            "[CIPHER FLOW MATCH] enabled (anchor=MeanOps, max_recipes=%d)\n",
            MAX_RECIPES);
    }
    return on;
}
extern "C" int cipher_flow_patterns_enabled(void) {
    return g_enabled.load(std::memory_order_relaxed);
}

extern "C" void cipher_flow_patterns_check(
    void* func, uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t smem, void** args, void* stream)
{
    (void)bx; (void)by; (void)bz; (void)smem; (void)gy; (void)gz;
    if (!g_enabled.load(std::memory_order_relaxed)) return;

    // Read raw arg values for the current kernel — same page-bounded probe
    // the recorder uses (cheap, ~50ns).
    uint64_t a[8] = {0};
    uint8_t  is_ptr[8] = {0};
    if (args && args[0]) {
        uintptr_t a0 = (uintptr_t)args[0];
        constexpr uintptr_t PAGE_MASK = ~((uintptr_t)4095);
        for (int i = 0; i < 8; ++i) {
            uintptr_t ai = (uintptr_t)args[i];
            if (ai == 0) break;
            if (ai < 0x10000 || ai >= 0x800000000000ULL) break;
            if ((ai & PAGE_MASK) != (a0 & PAGE_MASK)) break;
            uint64_t v = 0;
            memcpy(&v, args[i], sizeof(uint64_t));
            a[i] = v;
            is_ptr[i] = (v >= 0x100000000ULL && v < 0x800000000000ULL) ? 1 : 0;
        }
    }

    uint64_t now = now_ns();
    uint64_t stream_id = (uint64_t)stream;
    const char* nm = cipher_kt_name_for(func);

    // Stream change or timeout aborts a partial match.
    if (t_ctx.state != IDLE) {
        if (t_ctx.stream != stream_id) {
            reset_ctx("stream-change");
        } else if (now - t_ctx.last_ns > MAX_GAP_NS) {
            reset_ctx("timeout");
        }
    }

    // ── Anchor: reduce_kernel<...MeanOps...> ──────────────────────────
    if (t_ctx.state == IDLE) {
        if (name_contains_meanops(nm)) {
            // Save the kernel that ran RIGHT BEFORE this one as our
            // back-pointer for the Pow stage.  We don't strictly need
            // it for the recipe key (the func_ptr alone is enough),
            // but capturing it lets the recipe later identify K0.
            t_ctx = MatchCtx{};
            t_ctx.state = SAW_MEAN;
            t_ctx.fns[1] = func;                    // K1 = MeanOps
            t_ctx.mean_out_ptr = pick_out_ptr(a, is_ptr);
            t_ctx.var_in_ptr   = pick_first_ptr(a, is_ptr);
            t_ctx.last_ns = now;
            t_ctx.stream = stream_id;
            t_ctx.fn_pre_anchor = nullptr;          // filled later via ring lookback
            g_anchor_hits.fetch_add(1, std::memory_order_relaxed);
            // For initial pinning we leave t_ctx.fns[0] = nullptr.  Stage 2
            // can backfill it from the recorder's ring on a later iteration;
            // for now mark NULL so Stage 3 only suppresses from K1 onward.
        }
        return;
    }

    // ── Subsequent steps walk the sequence forward ────────────────────
    // Name-based matching.  Pointer-flow edges aren't usable here because
    // ATen kernels embed data pointers INSIDE struct args (TensorIterator,
    // ReduceOp) — not as direct kernel parameters.  The first 8 bytes of
    // args[i] is typically a vtable/strides field, not a data pointer.
    // Sequence shape is still stable, so we match on kernel-name fragments.
    switch (t_ctx.state) {
        case SAW_MEAN: {
            // Expected: in-place add of eps.  ATen name fragment:
            // "CUDAFunctorOnSelf_add" (the "OnSelf" variant).
            if (!nm || (strstr(nm, "CUDAFunctorOnSelf_add") == nullptr &&
                        strstr(nm, "FunctorOnSelf_add") == nullptr)) {
                reset_ctx("addeps-no-name"); return;
            }
            t_ctx.fns[2] = func;
            // Sniff eps from any small-float arg slot.
            for (int i = 0; i < 8; ++i) {
                if (!is_ptr[i] && a[i] != 0) {
                    float f;
                    uint32_t lo = (uint32_t)(a[i] & 0xFFFFFFFF);
                    memcpy(&f, &lo, 4);
                    if (f > 1e-9f && f < 1e-2f) { t_ctx.eps = f; break; }
                }
            }
            t_ctx.state = SAW_ADDEPS;
            t_ctx.last_ns = now;
            return;
        }
        case SAW_ADDEPS: {
            // Expected: a vectorized/unrolled elementwise (rsqrt lambda).
            if (!nm ||
                (strstr(nm, "vectorized_elementwise_kernel") == nullptr &&
                 strstr(nm, "unrolled_elementwise_kernel")   == nullptr &&
                 strstr(nm, "elementwise_kernel")            == nullptr)) {
                reset_ctx("rsqrt-no-elemwise"); return;
            }
            t_ctx.fns[3] = func;
            t_ctx.state = SAW_RSQRT;
            t_ctx.last_ns = now;
            return;
        }
        case SAW_RSQRT: {
            // Expected: x * rsqrt — elementwise kernel.
            if (!nm ||
                (strstr(nm, "vectorized_elementwise_kernel") == nullptr &&
                 strstr(nm, "unrolled_elementwise_kernel")   == nullptr &&
                 strstr(nm, "elementwise_kernel")            == nullptr)) {
                reset_ctx("mul-no-elemwise"); return;
            }
            t_ctx.fns[4] = func;
            t_ctx.rows_grid = (int)gx;
            t_ctx.state = SAW_MUL;
            t_ctx.last_ns = now;
            return;
        }
        case SAW_MUL: {
            // K5 (x * rsqrt) — expected: any elementwise (binary in-place).
            if (!nm ||
                (strstr(nm, "vectorized_elementwise_kernel") == nullptr &&
                 strstr(nm, "elementwise_kernel")            == nullptr &&
                 strstr(nm, "unrolled_elementwise_kernel")   == nullptr)) {
                reset_ctx("xrsqrt-no-elemwise"); return;
            }
            t_ctx.fns[5] = func;
            t_ctx.state = SAW_NORMED;
            t_ctx.last_ns = now;
            return;
        }
        case SAW_NORMED: {
            // K6 (* weight, the Scale step) — broadly accept any
            // elementwise / binary kernel.  Whether it's actually
            // BinaryFunctor<Mul> with 3-ptr array is what we're trying
            // to discover — let it through and inspect the args dump.
            if (!nm ||
                (strstr(nm, "BinaryFunctor")                 == nullptr &&
                 strstr(nm, "vectorized_elementwise_kernel") == nullptr &&
                 strstr(nm, "elementwise_kernel")            == nullptr &&
                 strstr(nm, "unrolled_elementwise_kernel")   == nullptr)) {
                static int s_log = 0;
                if (s_log++ < 5) {
                    fprintf(stderr,
                        "[CIPHER FLOW MATCH] step6 saw unexpected kernel: "
                        "%.140s\n", nm ? nm : "(null)");
                }
                reset_ctx("scale-no-binary"); return;
            }
            t_ctx.fns[6] = func;

            // ── MATCHED ────────────────────────────────────────────────
            // Backfill K0 (Pow): we don't strictly know it yet.  Use the
            // var_in_ptr — that's the buffer K0 wrote and K1 (MeanOps)
            // read.  Caller can resolve via the flow recorder ring; for
            // now leave fns[0] = NULL and Stage 3 substitutes from K1.
            int rec = install_recipe(t_ctx.fns, t_ctx.rows_grid,
                                       0, t_ctx.eps);
            uint64_t mc = g_match_complete.fetch_add(1, std::memory_order_relaxed);
            if (mc < 5 || (mc & 0xFF) == 0) {
                fprintf(stderr,
                    "[CIPHER FLOW MATCH] RMSNorm matched #%llu  "
                    "rows=%d eps=%.2e  "
                    "fns=[K1=%p K2=%p K3=%p K4=%p K5=%p K6=%p]  recipe=%d\n",
                    (unsigned long long)mc, t_ctx.rows_grid, t_ctx.eps,
                    t_ctx.fns[1], t_ctx.fns[2], t_ctx.fns[3],
                    t_ctx.fns[4], t_ctx.fns[5], t_ctx.fns[6], rec);
            }
            t_ctx = MatchCtx{};
            return;
        }
        default:
            reset_ctx("unknown-state");
            return;
    }
}

extern "C" int cipher_flow_patterns_lookup_first(void* func) {
    int n = g_n_recipes.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i) {
        if (g_recipes[i].valid && g_recipes[i].fns[1] == func) {
            // We anchor on K1 (MeanOps) since K0 (Pow) wasn't pinned.
            return i;
        }
    }
    return -1;
}

extern "C" const CipherFlowRecipe* cipher_flow_patterns_get_recipe(int id) {
    int n = g_n_recipes.load(std::memory_order_acquire);
    if (id < 0 || id >= n) return nullptr;
    return &g_recipes[id];
}

extern "C" int cipher_flow_patterns_recipe_count(void) {
    return g_n_recipes.load(std::memory_order_relaxed);
}

__attribute__((constructor(116)))
static void cipher_flow_patterns_autoinit() { cipher_flow_patterns_init(); }

__attribute__((destructor))
static void cipher_flow_patterns_autodump() {
    if (!g_enabled.load(std::memory_order_relaxed)) return;
    fprintf(stderr,
        "[CIPHER FLOW MATCH] teardown: anchor_hits=%llu matches=%llu "
        "aborts=%llu recipes=%d\n",
        (unsigned long long)g_anchor_hits.load(),
        (unsigned long long)g_match_complete.load(),
        (unsigned long long)g_match_aborted.load(),
        g_n_recipes.load());
}
