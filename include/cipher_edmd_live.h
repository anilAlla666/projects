// =============================================================================
// CIPHER — EDMD Live Calibration (Change 1)
// cipher_edmd_live.h
//
// Auto-derives a rank-r surrogate for each (K_dim, N_dim) GEMM shape by
// snapshotting real passthrough executions and running randomized SVD on the
// weight operand. On success, calls cipher_koopman_fp16_register_shape so
// future calls are routed through the existing shape registry / cache path.
//
// Geometry only — keyed on (K_dim, N_dim). No model/layer knowledge.
// =============================================================================

#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Collect a snapshot from one intercepted GEMM passthrough.
//
// All pointers are GPU device pointers using the PyTorch-side convention:
//   weight_gpu:     (K_dim, N_dim)  row-major  — cuBLAS 'A' operand
//   activation_gpu: (M_py,  K_dim)  row-major  — cuBLAS 'B' operand
//   output_gpu:     (M_py,  N_dim)  row-major  — cuBLAS 'C' operand
//
// dtypes are the cudaDataType values used by cublasGemmEx
//   (0 = CUDA_R_32F, 2 = CUDA_R_16F). Other dtypes are ignored.
//
// Effect: copies up to min(M_py, 8) rows of activation/output into a per-shape
// host buffer. When accumulated rows reach the target threshold, runs the
// randomized SVD fit on weight_gpu and registers the shape.
//
// Returns true if this call triggered a successful shape registration.
bool cipher_edmd_live_collect(
    int         M_py,
    int         K_dim,
    int         N_dim,
    int         weight_dtype,
    const void* weight_gpu,
    int         activation_dtype,
    const void* activation_gpu,
    int         output_dtype,
    const void* output_gpu);

// Query: has this shape already been registered by live calibration?
bool cipher_edmd_live_is_registered(int K_dim, int N_dim);

// Diagnostic report (stderr) — per-shape counts, residual ratio, max_diff.
void cipher_edmd_live_report(void);

// Test-only: force the fit + register for a shape using the current weight
// pointer, regardless of row count. Returns true on successful register.
// Used by tests/test_edmd_live_calibration.py to avoid tying the gate to a
// specific snapshot count.
bool cipher_edmd_live_force_fit(int K_dim, int N_dim);

// Read the last-fit diagnostics for a shape. Returns true if a fit has been
// performed. All output pointers may be NULL.
bool cipher_edmd_live_get_stats(
    int    K_dim,
    int    N_dim,
    float* residual_ratio_out,
    float* energy_captured_out,
    float* max_diff_out,
    int*   rows_collected_out);

#ifdef __cplusplus
}
#endif
