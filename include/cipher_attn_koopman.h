// =============================================================================
// CIPHER — Change 3: Fused Attention Koopman
// include/cipher_attn_koopman.h
//
// Replaces the three-step attention sequence
//     Q @ K_cache^T  →  softmax  →  @ V_cache
// with a single fused rank-r kernel of cost O(M · r · d), independent of
// the sequence length N. Cost of naïve attention at decode time is O(N · d).
//
// Architecture:
//   1. Per-stream state machine (IDLE → SAW_QK → SAW_SOFTMAX → fused)
//   2. Scratch-buffer revert model: caller memory never touched until the
//      happy path completes. Revert = replay saved calls in order.
//   3. Default OFF — enable via CIPHER_ATTN_KOOPMAN=1. Live substitution is
//      a lossy approximation; correctness depends on the effective rank of
//      the workload's K_cache, which must be validated per model.
//
// Geometry only. No kernel-name matching, no layer indices, no model
// knowledge. Detection keys on (head_dim ∈ {64,96,128}, M transitions,
// stream pointer identity). Head dimensions are universal across modern
// transformer architectures.
// =============================================================================

#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Observer return values — returned by the FSM entry points. The intercept
// shim acts on these and never makes substitution decisions itself.
// ---------------------------------------------------------------------------

typedef enum {
    CIPHER_ATTN_PASS_THROUGH  = 0,   // shim runs the call normally (default)
    CIPHER_ATTN_SUPPRESS      = 1,   // shim returns success without running the real call
    CIPHER_ATTN_REVERT_AND_PASS = 2, // FSM already replayed buffered suppressed
                                     // calls; shim now runs the current call
                                     // normally (do not suppress).
} CipherAttnAction;

// ---------------------------------------------------------------------------
// FSM entry points — called from the intercept shim on every relevant call.
// Each returns a CipherAttnAction that the shim acts on.
// ---------------------------------------------------------------------------

// Called from the cublasGemmEx / cublasLtMatmul shims BEFORE running the real
// GEMM. The FSM inspects the shape and the current per-stream state and
// decides what the shim should do. If the return is CIPHER_ATTN_REVERT_AND_PASS,
// the FSM has already replayed any buffered suppressed calls in order on the
// same stream — the shim should now proceed to run the current call normally.
//
// The FSM deep-copies alpha/beta host scalars and snapshots all other args so
// it can replay the call later without depending on caller-owned memory.
CipherAttnAction cipher_attn_fsm_on_gemm(
    void*       cublas_handle,   // opaque cublas handle; process-global, safe to reuse
    int         transa,
    int         transb,
    int         M, int N, int K,
    const void* alpha,           // host pointer to fp16/fp32 scalar
    const void* A, int Atype, int lda,
    const void* B, int Btype, int ldb,
    const void* beta,            // host pointer to fp16/fp32 scalar
    void*       C, int Ctype, int ldc,
    int         compute_type,
    int         algo,
    void*       stream);

// Called from the cuLaunchKernelEx / cuLaunchKernel / cudaLaunchKernel shims
// BEFORE running the real kernel. Used to detect the softmax kernel between
// the Q@K.T GEMM and the @V GEMM. Returns an action the shim acts on.
CipherAttnAction cipher_attn_fsm_on_kernel(
    const void* fn,
    unsigned    gx, unsigned gy, unsigned gz,
    unsigned    bx, unsigned by, unsigned bz,
    unsigned    shared_bytes,
    void**      params,
    void**      extra,
    void*       launch_config,   // non-null for cuLaunchKernelEx path (so we can replay)
    void*       stream);

// ---------------------------------------------------------------------------
// Test / telemetry hooks
// ---------------------------------------------------------------------------

bool      cipher_attn_koopman_enabled(void);
uint64_t  cipher_attn_fsm_revert_count(void);     // # times FSM has replayed suppressed calls
uint64_t  cipher_attn_fsm_fused_count(void);      // # times fused kernel fired end-to-end
uint64_t  cipher_attn_fsm_saw_qk_count(void);
uint64_t  cipher_attn_fsm_saw_softmax_count(void);
void      cipher_attn_fsm_report(void);

// ---------------------------------------------------------------------------
// Direct fused-kernel entry point (for the gate test to exercise the kernel
// without going through the intercept shim).
//
// All pointers are GPU fp32 device memory. Shapes:
//   Q            : (M, d)            row-major, fp32
//   V_T          : (r, d)            row-major, fp32  — Koopman query basis (V_x^T from SVD)
//   K_op         : (r, r)            row-major, fp32  — spectral scaling (often diag)
//   V_compressed : (r, d)            row-major, fp32  — U_x^T · V_cache, d-wide
//   out          : (M, d)            row-major, fp32  — attention output
//
// The kernel computes:
//   alpha   = Q · V_T^T           (M, r)
//   alpha2  = alpha · K_op^T      (M, r)   note: K_op^T matches kernel convention
//   weights = softmax_rowwise(alpha2)  over r dims
//   out     = weights · V_compressed   (M, d)
//
// Returns 0 on success.
int cipher_attn_fused_launch_fp32(
    const float* Q,
    const float* V_T,
    const float* K_op,
    const float* V_compressed,
    float*       out,
    int          M,
    int          r,
    int          d,
    void*        stream);

// Precompute V_compressed = U_x^T · V_cache on the GPU. Used by the FSM at
// prefill-end and also by the gate test to build test inputs.
//
//   U_x          : (N, r)  row-major fp32  — left singular vectors of Q-snapshots
//   V_cache      : (N, d)  row-major fp32
//   V_compressed : (r, d)  row-major fp32  — output
//
// Returns 0 on success.
int cipher_attn_precompute_vcompressed(
    const float* U_x,
    const float* V_cache,
    float*       V_compressed,
    int          N, int r, int d,
    void*        stream);

#ifdef __cplusplus
}
#endif
