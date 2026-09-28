// =============================================================================
// CIPHER — L3.5: EDMD Pipeline — Novel Operation Koopman Derivation
// cipher_edmd.h
//
// For any operation outside the 7-family taxonomy, CIPHER passively collects
// 20-100 input-output snapshot pairs, then solves for the Koopman operator K
// via EDMD (Extended Dynamic Mode Decomposition).
//
// ALGORITHM:
//   1. Collect m snapshot pairs: {(x_1,y_1), ..., (x_m,y_m)}
//      where x_k = input, y_k = f(x_k) = output of the target operation.
//   2. Lift to observable space via dictionary Ψ:
//      ΨX = [ψ(x_1), ..., ψ(x_m)]  ∈ ℝ^{K×m}
//      ΨY = [ψ(y_1), ..., ψ(y_m)]  ∈ ℝ^{K×m}
//   3. Solve K = argmin ||ΨY - K·ΨX||²_F  via least squares:
//      K = ΨY · ΨX†  (ΨX† = pseudoinverse via QR decomposition)
//   4. Registered surrogate: given new input x*, predict f(x*) ≈ Ψ†(K·ψ(x*))
//
// DICTIONARY (Chebyshev observables on [-1,1]):
//   ψ_k(x) = T_k(x)  for k = 0,...,K-1
//   where T_k is the k-th Chebyshev polynomial of the first kind.
//   Chebyshev basis is optimal for smooth functions — exponential convergence.
//   For softmax (smooth, bounded): K=20 gives <1% error (EXP.A target).
//
// CONVERGENCE GUARANTEE (Philipp et al. 2024):
//   For Lipschitz-continuous f with Koopman-invariant subspace of dimension K,
//   EDMD with m = O(K²·log(1/ε)) snapshots achieves ε-error.
//   Softmax: K≤20 eigenfunctions, m≥50 snapshots → ε < 0.01.
//
// SUCCESS CRITERION: <1% error with <20 eigenfunctions on softmax attention.
// DEPENDENCIES: L3.1 (op class identification), F4 (liquid state registration).
// =============================================================================

#pragma once

#include "cipher_structural_lookup.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Dimensions and limits
// ---------------------------------------------------------------------------

#define CIPHER_EDMD_MAX_DICT_SIZE    32    // Max Chebyshev basis functions (K)
#define CIPHER_EDMD_MAX_SNAPSHOTS   128    // Max snapshot pairs stored (m)
#define CIPHER_EDMD_MAX_INPUT_DIM    64    // Max input/output vector dimension
#define CIPHER_EDMD_MIN_SNAPSHOTS    20    // Min pairs before solving
#define CIPHER_EDMD_TARGET_ERROR   0.01f   // 1% error target (EXP.A)

// Status of an EDMD derivation
typedef enum {
    CIPHER_EDMD_COLLECTING   = 0,  // Still gathering snapshots
    CIPHER_EDMD_READY        = 1,  // Enough snapshots, can solve
    CIPHER_EDMD_SOLVED       = 2,  // K derived, surrogate registered
    CIPHER_EDMD_FAILED       = 3,  // Solve failed (rank deficiency etc)
} CipherEdmdStatus;

// ---------------------------------------------------------------------------
// Snapshot buffer — stores (x, y) pairs
// ---------------------------------------------------------------------------

typedef struct {
    float    x[CIPHER_EDMD_MAX_INPUT_DIM];   // Input snapshot
    float    y[CIPHER_EDMD_MAX_INPUT_DIM];   // Output snapshot (f(x))
} CipherEdmdSnapshot;

typedef struct {
    CipherEdmdSnapshot snapshots[CIPHER_EDMD_MAX_SNAPSHOTS];
    uint32_t           count;           // Number of snapshots collected
    uint32_t           input_dim;       // Dimension of x (and y)
    bool               initialized;
} CipherEdmdBuffer;

// ---------------------------------------------------------------------------
// Koopman operator — result of EDMD solve
// ---------------------------------------------------------------------------

typedef struct {
    float    K[CIPHER_EDMD_MAX_DICT_SIZE][CIPHER_EDMD_MAX_DICT_SIZE]; // Koopman matrix
    uint32_t dict_size;       // K: number of observables used
    uint32_t input_dim;       // n: dimension of x/y
    float    fit_error;       // Residual ||ΨY - K·ΨX||_F / ||ΨY||_F
    uint32_t snapshots_used;  // m: snapshot count at solve time
    bool     valid;           // True if solve succeeded and error < target
} CipherKoopmanOp;

// ---------------------------------------------------------------------------
// EDMD Pipeline state — one per novel operation being identified
// ---------------------------------------------------------------------------

typedef struct {
    char              op_name[64];     // Name of the operation being identified
    CipherEdmdBuffer  buffer;          // Snapshot collection
    CipherKoopmanOp   koopman;         // Solved Koopman operator
    CipherEdmdStatus  status;
    uint32_t          dict_size;       // Number of Chebyshev observables to use

    // Statistics
    uint64_t  total_predictions;
    float     avg_prediction_error;   // Running average prediction error
    float     min_error_seen;
    uint64_t  registered_at_snapshot; // Which snapshot triggered registration
} CipherEdmdPipeline;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Initialize EDMD pipeline for a novel operation.
// dict_size: number of Chebyshev observables (default: 20 for EXP.A target).
void cipher_edmd_init(CipherEdmdPipeline* pipe,
                      const char*         op_name,
                      uint32_t            input_dim,
                      uint32_t            dict_size);

// Add a snapshot pair (x=input, y=output of the operation).
// Call this on every passthrough (when oracle decides not to substitute).
// Returns true if pipeline is now ready to solve.
bool cipher_edmd_collect(CipherEdmdPipeline* pipe,
                         const float*        x,
                         const float*        y);

// Solve for Koopman operator K via least squares.
// Automatically called when min_snapshots reached.
// Returns fit_error. If < CIPHER_EDMD_TARGET_ERROR: sets status = SOLVED.
float cipher_edmd_solve(CipherEdmdPipeline* pipe);

// Predict output for new input using derived Koopman operator.
// Returns true if prediction succeeded (status == SOLVED).
// Output written to y_pred (must be preallocated to input_dim floats).
bool cipher_edmd_predict(CipherEdmdPipeline* pipe,
                         const float*        x,
                         float*              y_pred);

// Compute prediction error vs ground truth.
// Normalised RMSE: ||y_pred - y_true|| / ||y_true||
float cipher_edmd_error(const float* y_pred,
                        const float* y_true,
                        uint32_t     dim);

void cipher_edmd_report(const CipherEdmdPipeline* pipe);

// ---------------------------------------------------------------------------
// Dictionary functions (Chebyshev observables)
// Exported for testing and Layer 1 use.
// ---------------------------------------------------------------------------

// Evaluate Chebyshev polynomial T_k(x) for scalar x ∈ [-1,1].
float cipher_cheb_eval(int k, float x);

// Lift a vector x (dim-dimensional) into observable space (dict_size-dimensional).
// ψ(x) = [T_0(x_0), T_1(x_0), ..., T_{K-1}(x_0),
//          T_0(x_1), T_1(x_1), ..., T_{K-1}(x_1), ...]
// Normalized to [-1,1] using provided x_min, x_max.
void cipher_edmd_lift(const float* x, uint32_t dim,
                      float* psi, uint32_t dict_size,
                      float x_min, float x_max);

#ifdef __cplusplus
}
#endif
