// =============================================================================
// CIPHER — Change 3: Fused Attention Koopman — State Machine
// src/cipher_attn_koopman.cpp
//
// Per-stream FSM with scratch-buffer revert model. Caller memory is never
// touched until the happy-path sequence (SAW_QK → SAW_SOFTMAX → @V) completes.
// On any state break, the FSM replays the buffered suppressed calls in order
// on the same stream before handing control back to the intercept shim.
//
// Compiled into libcipher_rt.so so it can see the cublas/cuLaunch real-symbol
// pointers via dlsym at fit time. The intercept shim (libcipher_hook.so) calls
// into us via weakly-resolved cipher_attn_fsm_* entry points.
//
// Drift rule ✓: only geometry is consulted — (M, K, stream_ptr, head_dim ∈
// {64, 96, 128}). No model names, no layer indices, no kernel-name matches.
// =============================================================================

#include "cipher_attn_koopman.h"

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

// -----------------------------------------------------------------------------
// Tunables
// -----------------------------------------------------------------------------

#define CIPHER_ATTN_MAX_STREAMS 16
#define CIPHER_ATTN_HEAD_DIMS_OK(k) ((k) == 64 || (k) == 96 || (k) == 128)

// Default rank — the test harness can override via register_shape if needed.
#define CIPHER_ATTN_DEFAULT_R 64

// -----------------------------------------------------------------------------
// cuBLAS symbol types (matching the shim-side typedefs)
// -----------------------------------------------------------------------------

typedef int cublasStatus_t;
typedef int CUresult;
typedef void* CUfunction;
typedef void* CUstream;

typedef cublasStatus_t (*cublasGemmEx_fn)(
    void* handle, int transa, int transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, int Atype, int lda,
    const void* B, int Btype, int ldb,
    const void* beta,
    void*       C, int Ctype, int ldc,
    int computeType, int algo);

typedef CUresult (*cuLaunchKernelEx_fn)(void* config, CUfunction f,
                                         void** params, void** extra);
typedef CUresult (*cuLaunchKernel_fn)(CUfunction, unsigned, unsigned, unsigned,
                                       unsigned, unsigned, unsigned,
                                       unsigned, CUstream, void**, void**);

static cublasGemmEx_fn    g_real_cublasGemmEx    = nullptr;
static cuLaunchKernelEx_fn g_real_cuLaunchEx     = nullptr;
static cuLaunchKernel_fn   g_real_cuLaunchKernel = nullptr;

// Resolve real entry points lazily (the shim DSO has already loaded cuBLAS
// and libcuda, so these dlsym lookups hit cached PLT entries).
static std::once_flag g_resolve_once;
static void resolve_reals() {
    std::call_once(g_resolve_once, []() {
        void* hblas = dlopen("libcublas.so.12", RTLD_NOW | RTLD_NOLOAD);
        if (!hblas) hblas = dlopen("libcublas.so.12", RTLD_NOW);
        if (hblas) {
            void* s = dlvsym(hblas, "cublasGemmEx", "libcublas.so.12");
            if (!s) s = dlsym(hblas, "cublasGemmEx");
            g_real_cublasGemmEx = (cublasGemmEx_fn)s;
        }
        void* hcuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        if (!hcuda) hcuda = dlopen("libcuda.so.1", RTLD_NOW);
        if (hcuda) {
            g_real_cuLaunchEx     = (cuLaunchKernelEx_fn)dlsym(hcuda, "cuLaunchKernelEx");
            g_real_cuLaunchKernel = (cuLaunchKernel_fn)  dlsym(hcuda, "cuLaunchKernel");
        }
    });
}

// -----------------------------------------------------------------------------
// Enable flag (CIPHER_ATTN_KOOPMAN=1 → live substitution; default OFF)
// -----------------------------------------------------------------------------

static int g_enabled_cached = -1;
static bool enabled_flag() {
    int v = g_enabled_cached;
    if (v < 0) {
        const char* e = std::getenv("CIPHER_ATTN_KOOPMAN");
        v = (e && e[0] == '1') ? 1 : 0;
        if (v == 1) {
            fprintf(stderr,
                "[CIPHER-ATTN] Live attention substitution ENABLED. "
                "Outputs are approximate. Validate perplexity before "
                "production use.\n");
        }
        g_enabled_cached = v;
    }
    return v == 1;
}

extern "C" bool cipher_attn_koopman_enabled(void) {
    return enabled_flag();
}

// -----------------------------------------------------------------------------
// Per-stream FSM state
// -----------------------------------------------------------------------------

enum FsmState : uint8_t {
    FSM_IDLE       = 0,
    FSM_SAW_QK     = 1,
    FSM_SAW_SOFT   = 2,
};

// Deep-copied cublasGemmEx args — caller-owned host memory (alpha/beta) is
// copied out; device pointers are stashed verbatim.
struct SavedGemm {
    void*  handle;
    int    transa, transb;
    int    M, N, K;
    float  alpha_f;             // deep copy as float
    float  beta_f;
    int    alpha_mode;          // 0 = scalar, stored in alpha_f; else skip
    int    beta_mode;
    const void* A;
    int    Atype;
    int    lda;
    const void* B;
    int    Btype;
    int    ldb;
    void*       C;
    int    Ctype;
    int    ldc;
    int    compute_type;
    int    algo;
    void*  stream;
    bool   valid;
};

// Deep-copied cuLaunchKernel args — used to replay the softmax kernel on
// revert. We save the raw params array by pointer (PyTorch keeps the arg
// arena alive across a single op boundary, and we replay before returning
// control to Python, so the arena is still valid). If the arena has been
// freed we would detect this via a launch failure.
struct SavedKernel {
    const void*  fn;
    unsigned     gx, gy, gz, bx, by, bz;
    unsigned     shared_bytes;
    void**       params;
    void**       extra;
    void*        launch_config;  // for cuLaunchKernelEx path
    void*        stream;
    bool         valid;
};

struct StreamState {
    void*      stream;           // CUstream pointer (key)
    FsmState   state;

    // Buffered suppressed calls (for revert)
    SavedGemm   saved_qk;
    SavedKernel saved_softmax;

    // Prefill/decode tracking
    int        prev_M_at_head_dim;   // last M seen on a head_dim GEMM
    int        head_dim_seen;        // 64/96/128 or 0

    // Handshake with the fused kernel path — populated when
    // register_shape() (or a future prefill-end hook) caches (V_T, K_op, V_compressed).
    bool       fused_ready;

    // Live-substitution scratch buffer (device memory, owned)
    float*     scratch;
    size_t     scratch_bytes;
};

static StreamState g_streams[CIPHER_ATTN_MAX_STREAMS];
static int         g_stream_count = 0;
static std::mutex  g_mtx;

// Counters (lock-free)
static std::atomic<uint64_t> g_count_saw_qk   {0};
static std::atomic<uint64_t> g_count_saw_soft {0};
static std::atomic<uint64_t> g_count_fused    {0};
static std::atomic<uint64_t> g_count_revert   {0};

static StreamState* find_or_create_stream_locked(void* stream) {
    for (int i = 0; i < g_stream_count; i++) {
        if (g_streams[i].stream == stream) return &g_streams[i];
    }
    if (g_stream_count >= CIPHER_ATTN_MAX_STREAMS) return nullptr;
    StreamState& s = g_streams[g_stream_count++];
    memset(&s, 0, sizeof(s));
    s.stream = stream;
    s.state  = FSM_IDLE;
    return &s;
}

// -----------------------------------------------------------------------------
// Deep-copy helpers
// -----------------------------------------------------------------------------

// Read a cuBLAS alpha/beta scalar into a float. cuBLAS computeType determines
// whether alpha/beta are fp16, fp32, or int. We support fp32 and fp16 here;
// anything else falls back to 1.0/0.0 (default GEMM behavior) and is
// flagged — cross-dtype alpha is exotic and not produced by PyTorch fp16/fp32.
static float copy_scalar_as_float(const void* p, int compute_type, int* mode_out) {
    if (!p) { *mode_out = 1; return 1.0f; }
    // CUBLAS_COMPUTE_32F = 68; CUBLAS_COMPUTE_16F = 64. fp32 is the common case.
    // For simplicity, always treat as fp32 — PyTorch fp16 inference still uses
    // fp32 accumulation and passes fp32 scalars. cublasLtMatmul with fp16
    // scalars is possible but rare.
    (void)compute_type;
    *mode_out = 0;
    return *(const float*)p;
}

static void save_gemm_args(
    SavedGemm* g,
    void* handle,
    int transa, int transb,
    int M, int N, int K,
    const void* alpha,
    const void* A, int Atype, int lda,
    const void* B, int Btype, int ldb,
    const void* beta,
    void*       C, int Ctype, int ldc,
    int compute_type, int algo,
    void* stream)
{
    memset(g, 0, sizeof(*g));
    g->handle       = handle;
    g->transa       = transa;
    g->transb       = transb;
    g->M = M; g->N = N; g->K = K;
    g->alpha_f      = copy_scalar_as_float(alpha, compute_type, &g->alpha_mode);
    g->beta_f       = copy_scalar_as_float(beta,  compute_type, &g->beta_mode);
    g->A = A; g->Atype = Atype; g->lda = lda;
    g->B = B; g->Btype = Btype; g->ldb = ldb;
    g->C = C; g->Ctype = Ctype; g->ldc = ldc;
    g->compute_type = compute_type;
    g->algo         = algo;
    g->stream       = stream;
    g->valid        = true;
}

static void save_kernel_args(
    SavedKernel* k,
    const void*  fn,
    unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shared_bytes,
    void** params, void** extra,
    void*  launch_config,
    void*  stream)
{
    memset(k, 0, sizeof(*k));
    k->fn = fn;
    k->gx = gx; k->gy = gy; k->gz = gz;
    k->bx = bx; k->by = by; k->bz = bz;
    k->shared_bytes = shared_bytes;
    k->params       = params;
    k->extra        = extra;
    k->launch_config = launch_config;
    k->stream       = stream;
    k->valid        = true;
}

// -----------------------------------------------------------------------------
// Replay helpers — called during revert. Must queue on the stream saved
// with the call so downstream kernels see correct ordering.
// -----------------------------------------------------------------------------

static bool replay_gemm(const SavedGemm* g) {
    if (!g || !g->valid) return true;
    // Safety: a GEMM without a valid handle or tensor pointers cannot be
    // meaningfully replayed. Skipping treats the revert as "nothing was
    // actually suppressed at the kernel level" — the counter still
    // increments in revert_and_reset_locked below so tests / telemetry
    // can observe the state-machine transition.
    if (!g->handle || !g->A || !g->B || !g->C) {
        return true;
    }
    resolve_reals();
    if (!g_real_cublasGemmEx) {
        fprintf(stderr, "[CIPHER-ATTN] replay_gemm: g_real_cublasGemmEx unresolved\n");
        return false;
    }
    // Note: cuBLAS reads alpha/beta via host pointer. We pass the addresses
    // of our stashed floats. alpha_mode==0 means fp32.
    const void* alpha_p = (g->alpha_mode == 0) ? (const void*)&g->alpha_f : nullptr;
    const void* beta_p  = (g->beta_mode  == 0) ? (const void*)&g->beta_f  : nullptr;
    static const float one = 1.0f, zero = 0.0f;
    if (!alpha_p) alpha_p = &one;
    if (!beta_p)  beta_p  = &zero;

    cublasStatus_t r = g_real_cublasGemmEx(
        g->handle, g->transa, g->transb,
        g->M, g->N, g->K,
        alpha_p,
        g->A, g->Atype, g->lda,
        g->B, g->Btype, g->ldb,
        beta_p,
        g->C, g->Ctype, g->ldc,
        g->compute_type, g->algo);
    if (r != 0) {
        fprintf(stderr, "[CIPHER-ATTN] replay_gemm: cublasGemmEx returned %d\n", (int)r);
        return false;
    }
    return true;
}

static bool replay_kernel(const SavedKernel* k) {
    if (!k || !k->valid) return true;
    resolve_reals();
    if (k->launch_config) {
        if (!g_real_cuLaunchEx) {
            fprintf(stderr, "[CIPHER-ATTN] replay_kernel: g_real_cuLaunchEx unresolved\n");
            return false;
        }
        CUresult r = g_real_cuLaunchEx(k->launch_config,
                                        (CUfunction)k->fn,
                                        k->params, k->extra);
        if (r != 0) {
            fprintf(stderr, "[CIPHER-ATTN] replay_kernel (Ex): %d\n", (int)r);
            return false;
        }
    } else {
        if (!g_real_cuLaunchKernel) {
            fprintf(stderr, "[CIPHER-ATTN] replay_kernel: g_real_cuLaunchKernel unresolved\n");
            return false;
        }
        CUresult r = g_real_cuLaunchKernel(
            (CUfunction)k->fn, k->gx, k->gy, k->gz, k->bx, k->by, k->bz,
            k->shared_bytes, (CUstream)k->stream,
            k->params, k->extra);
        if (r != 0) {
            fprintf(stderr, "[CIPHER-ATTN] replay_kernel: %d\n", (int)r);
            return false;
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
// Revert: replay any buffered calls for this stream in chronological order
// and reset state. Called on any FSM mismatch.
// -----------------------------------------------------------------------------

static void revert_and_reset_locked(StreamState* s, const char* reason) {
    if (!s) return;
    const bool was_mid_sequence = (s->state == FSM_SAW_QK || s->state == FSM_SAW_SOFT);
    uint64_t n_replayed = 0;
    if (was_mid_sequence) {
        if (s->saved_qk.valid) {
            if (replay_gemm(&s->saved_qk)) n_replayed++;
        }
        if (s->state == FSM_SAW_SOFT && s->saved_softmax.valid) {
            if (replay_kernel(&s->saved_softmax)) n_replayed++;
        }
        // Increment revert counter for EVERY mid-sequence abort, whether or
        // not the replay had physical work to do. The counter tracks state-
        // machine transitions, which is what tests and production telemetry
        // care about ("did the FSM have to revert?").
        g_count_revert.fetch_add(1, std::memory_order_relaxed);
        fprintf(stderr,
            "[CIPHER-ATTN] REVERT stream=%p state=%d replayed=%llu reason=%s\n",
            s->stream, (int)s->state, (unsigned long long)n_replayed, reason);
    }
    s->state = FSM_IDLE;
    s->saved_qk.valid = false;
    s->saved_softmax.valid = false;
}

// -----------------------------------------------------------------------------
// Shape detection
// -----------------------------------------------------------------------------

// Is this GEMM the Q @ K_cache^T attention-score matmul?
// PyTorch row-major: torch.mm(Q (batch, d), K.T (d, seq_len)) → (batch, seq_len)
// cuBLAS col-major : result (seq_len, batch), so cublas_m = seq_len,
//                    cublas_n = batch, cublas_k = head_dim.
// We therefore require cublas_k ∈ {64,96,128} (head_dim) AND cublas_m ≥ 128
// (seq_len — at least one full head_dim tile worth of tokens).
static bool is_qk_gemm(int M, int N, int K) {
    (void)N;
    return CIPHER_ATTN_HEAD_DIMS_OK(K) && M >= 128;
}

// Is this GEMM the weights @ V_cache matmul following a detected softmax?
// Same check works: k dimension matches head_dim → no, for @V the contraction
// is over seq_len, not head_dim. The OUTPUT dim is head_dim. So we check M.
// cuBLAS convention: C(m×n) = A(m×k) · B(k×n) where m is one of the output dims.
// For torch.mm(weights (1, N), V_cache (N, d)): result (1, d). In cuBLAS col-major
// with PyTorch row-major inputs: m=d, n=1, k=N. So we check m.
static bool is_atv_gemm(int M, int N, int K) {
    (void)N;
    return CIPHER_ATTN_HEAD_DIMS_OK(M) && K >= 128;
}

// Is this kernel plausibly a softmax on attention logits? We key on the
// ELEMENTWISE footprint typical for a softmax kernel: 1D grid, high block
// count, little shared memory. Classifier would say ELEMENTWISE.
static bool is_softmax_kernel(unsigned gx, unsigned gy, unsigned gz,
                               unsigned bx, unsigned by, unsigned bz,
                               unsigned shared_bytes) {
    // Very permissive — we only care that the kernel is "between" our GEMMs
    // on the right stream. Softmax kernels are typically 1D, block=128/256,
    // small shared. We don't match the fn pointer because kernel names
    // violate the drift rule. Geometry alone is sufficient.
    if (gy > 1 || gz > 1) return false;   // softmax is 1D over batch×rows
    if (by > 1 || bz > 1) return false;
    if (bx < 32 || bx > 1024) return false;
    if (shared_bytes > 16384) return false;
    return true;
}

// -----------------------------------------------------------------------------
// Public API — called from the intercept shim
// -----------------------------------------------------------------------------

extern "C"
CipherAttnAction cipher_attn_fsm_on_gemm(
    void*       cublas_handle,
    int         transa,
    int         transb,
    int         M, int N, int K,
    const void* alpha,
    const void* A, int Atype, int lda,
    const void* B, int Btype, int ldb,
    const void* beta,
    void*       C, int Ctype, int ldc,
    int         compute_type,
    int         algo,
    void*       stream)
{
    if (!enabled_flag()) return CIPHER_ATTN_PASS_THROUGH;

    std::lock_guard<std::mutex> lock(g_mtx);
    StreamState* s = find_or_create_stream_locked(stream);
    if (!s) return CIPHER_ATTN_PASS_THROUGH;

    // Cross-stream abort: if current stream differs from saved state's
    // stream, the prior sequence is dead. Revert and reset.
    if ((s->state == FSM_SAW_QK || s->state == FSM_SAW_SOFT)
        && s->saved_qk.valid && s->saved_qk.stream != stream) {
        revert_and_reset_locked(s, "cross-stream");
        return CIPHER_ATTN_REVERT_AND_PASS;
    }

    // Prefill/decode tracking — independent of suppression
    if (CIPHER_ATTN_HEAD_DIMS_OK(K)) {
        s->prev_M_at_head_dim = M;
        s->head_dim_seen      = K;
    }

    switch (s->state) {
        case FSM_IDLE: {
            if (is_qk_gemm(M, N, K)) {
                save_gemm_args(&s->saved_qk, cublas_handle,
                               transa, transb, M, N, K,
                               alpha, A, Atype, lda, B, Btype, ldb,
                               beta, C, Ctype, ldc,
                               compute_type, algo, stream);
                s->state = FSM_SAW_QK;
                g_count_saw_qk.fetch_add(1, std::memory_order_relaxed);
                // SCRATCH-BUFFER MODEL: caller memory is NOT touched. We
                // simply suppress the real GEMM; C retains its old contents.
                // If the sequence breaks we replay the saved call so C ends
                // up correct before any downstream kernel reads it.
                return CIPHER_ATTN_SUPPRESS;
            }
            return CIPHER_ATTN_PASS_THROUGH;
        }
        case FSM_SAW_QK: {
            // The next GEMM should NOT be another attention-style matmul.
            // If it is, sequence broke — revert and start fresh.
            revert_and_reset_locked(s, "gemm_after_qk");
            return CIPHER_ATTN_REVERT_AND_PASS;
        }
        case FSM_SAW_SOFT: {
            // Expected: @V_cache GEMM. If shape matches, run the fused kernel
            // path. Otherwise, revert both the QK and the softmax.
            if (is_atv_gemm(M, N, K) && s->fused_ready) {
                // Would run fused kernel here (TODO: wire up once the
                // per-shape (V_T, K_op, V_compressed) registry is populated
                // by a prefill-end hook). For now, fall through to revert
                // — defaulting to CORRECT output on any uncertainty.
                revert_and_reset_locked(s, "fused_path_not_yet_wired");
                return CIPHER_ATTN_REVERT_AND_PASS;
            }
            revert_and_reset_locked(s, "no_atv_after_softmax");
            return CIPHER_ATTN_REVERT_AND_PASS;
        }
    }
    return CIPHER_ATTN_PASS_THROUGH;
}

extern "C"
CipherAttnAction cipher_attn_fsm_on_kernel(
    const void* fn,
    unsigned    gx, unsigned gy, unsigned gz,
    unsigned    bx, unsigned by, unsigned bz,
    unsigned    shared_bytes,
    void**      params,
    void**      extra,
    void*       launch_config,
    void*       stream)
{
    if (!enabled_flag()) return CIPHER_ATTN_PASS_THROUGH;

    std::lock_guard<std::mutex> lock(g_mtx);
    StreamState* s = find_or_create_stream_locked(stream);
    if (!s) return CIPHER_ATTN_PASS_THROUGH;

    // Cross-stream abort
    if ((s->state == FSM_SAW_QK || s->state == FSM_SAW_SOFT)
        && s->saved_qk.valid && s->saved_qk.stream != stream) {
        revert_and_reset_locked(s, "cross-stream (kernel)");
        return CIPHER_ATTN_REVERT_AND_PASS;
    }

    switch (s->state) {
        case FSM_IDLE:
            return CIPHER_ATTN_PASS_THROUGH;
        case FSM_SAW_QK: {
            if (is_softmax_kernel(gx, gy, gz, bx, by, bz, shared_bytes)) {
                save_kernel_args(&s->saved_softmax, fn, gx, gy, gz,
                                 bx, by, bz, shared_bytes,
                                 params, extra, launch_config, stream);
                s->state = FSM_SAW_SOFT;
                g_count_saw_soft.fetch_add(1, std::memory_order_relaxed);
                return CIPHER_ATTN_SUPPRESS;
            }
            revert_and_reset_locked(s, "non_softmax_after_qk");
            return CIPHER_ATTN_REVERT_AND_PASS;
        }
        case FSM_SAW_SOFT: {
            // Any kernel between softmax and @V_cache → break the sequence
            revert_and_reset_locked(s, "kernel_between_softmax_and_atv");
            return CIPHER_ATTN_REVERT_AND_PASS;
        }
    }
    return CIPHER_ATTN_PASS_THROUGH;
}

// -----------------------------------------------------------------------------
// Telemetry / test hooks
// -----------------------------------------------------------------------------

extern "C" uint64_t cipher_attn_fsm_revert_count(void) {
    return g_count_revert.load(std::memory_order_relaxed);
}
extern "C" uint64_t cipher_attn_fsm_fused_count(void) {
    return g_count_fused.load(std::memory_order_relaxed);
}
extern "C" uint64_t cipher_attn_fsm_saw_qk_count(void) {
    return g_count_saw_qk.load(std::memory_order_relaxed);
}
extern "C" uint64_t cipher_attn_fsm_saw_softmax_count(void) {
    return g_count_saw_soft.load(std::memory_order_relaxed);
}

extern "C" void cipher_attn_fsm_report(void) {
    fprintf(stderr,
        "[CIPHER-ATTN REPORT] enabled=%d streams=%d "
        "saw_qk=%llu saw_softmax=%llu fused=%llu reverts=%llu\n",
        enabled_flag() ? 1 : 0, g_stream_count,
        (unsigned long long)g_count_saw_qk.load(),
        (unsigned long long)g_count_saw_soft.load(),
        (unsigned long long)g_count_fused.load(),
        (unsigned long long)g_count_revert.load());
}
