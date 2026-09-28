// =============================================================================
// CIPHER — L1.1: Runtime Koopman Derivation
// cipher_koopman_runtime.h
//
// Automatic pipeline: observe novel op → EDMD → register surrogate → O(1).
//
// WHEN AN UNKNOWN OP ARRIVES:
//   Phase 1 (snapshots < MIN_SNAPSHOTS):
//     - Force passthrough (execute real kernel)
//     - Collect (input_features, output_features) snapshot pair
//     - No substitution yet
//
//   Phase 2 (snapshots == MIN_SNAPSHOTS):
//     - Run EDMD solve: K = ΨY · ΨX† (Chebyshev observables)
//     - If fit_error < threshold: register surrogate in L1.3 registry
//     - Switch to substitution mode
//
//   Phase 3 (surrogate registered):
//     - All future calls go through O(1) Koopman surrogate
//     - EDMD continues collecting to improve the fit (online refinement)
//
// OP IDENTITY:
//   We identify an op by its (op_class, grid_x, grid_y, block_size, shmem)
//   tuple — the same geometry always means the same computation.
//   Hash: FNV-1a over the tuple → 64-bit op_hash.
//   Max tracked ops: 256 (covers a full transformer model's unique GEMM shapes)
//
// INTEGRATION:
//   Called from cipher_dispatch.cpp after the structural lookup returns OK
//   but before the LNN decides. If the op has a derived surrogate, the
//   dispatch uses it. If it's still collecting, dispatch forces passthrough.
//
// DEPENDENCIES: L3.5 (EDMD pipeline), L1.3 (registry), F4 (liquid state)
// =============================================================================

#pragma once

#include "cipher_edmd.h"
#include "cipher_classify.hpp"
#include "cipher_liquid_state.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#define CIPHER_KR_MAX_OPS        256    // Max unique op shapes tracked
#define CIPHER_KR_FEATURE_DIM     16    // Input feature dimension for EDMD
#define CIPHER_KR_DICT_SIZE       14    // Chebyshev observables (must be < CIPHER_KR_FEATURE_DIM)
#define CIPHER_KR_MIN_SNAPSHOTS   20    // Snapshots before first solve
#define CIPHER_KR_REFINE_EVERY    10    // Re-solve every N snapshots after first
#define CIPHER_KR_FIT_THRESHOLD   0.30f // Accept surrogate if fit_error < this

// Op derivation state
typedef enum {
    CIPHER_KR_COLLECTING  = 0,  // Still gathering snapshots
    CIPHER_KR_DERIVED     = 1,  // Surrogate derived and registered
    CIPHER_KR_FAILED      = 2,  // EDMD failed (high error) — keep collecting
} CipherKRState;

// ---------------------------------------------------------------------------
// Per-op derivation record
// ---------------------------------------------------------------------------

typedef struct {
    uint64_t          op_hash;          // FNV-1a hash of (cls, gx, gy, bz, shmem)
    char              op_name[48];      // Human-readable description
    uint8_t           op_class;         // From L3.1 classifier
    uint32_t          grid_x;
    uint32_t          grid_y;
    uint32_t          block_size;
    uint32_t          shmem_bytes;

    CipherEdmdPipeline edmd;            // EDMD snapshot buffer + Koopman op
    CipherKRState     state;

    // Statistics
    uint64_t  total_calls;             // Total times this op was seen
    uint64_t  passthroughs;            // Calls during collection phase
    uint64_t  substitutions;           // Calls using derived surrogate
    float     last_fit_error;

    // Registry slot (set when surrogate is registered)
    int       registry_slot;           // -1 = not registered
} CipherKRRecord;

// ---------------------------------------------------------------------------
// Runtime state — one global instance
// ---------------------------------------------------------------------------

typedef struct {
    CipherKRRecord records[CIPHER_KR_MAX_OPS];
    uint32_t       n_records;           // Active entries

    // Aggregate stats
    uint64_t  total_novel_ops;          // Unique op shapes seen
    uint64_t  total_derived;            // Ops with successful surrogate
    uint64_t  total_failed;             // Ops where EDMD failed
    uint64_t  total_collection_calls;   // Calls spent collecting snapshots
    uint64_t  total_surrogate_calls;    // Calls using derived surrogates

    bool      initialized;
} CipherKoopmanRuntime;

// ---------------------------------------------------------------------------
// Decision returned to cipher_dispatch
// ---------------------------------------------------------------------------

typedef struct {
    bool     should_substitute;     // Use derived surrogate?
    bool     is_collecting;         // Still in collection phase?
    int      record_idx;            // Index into records[] (-1 if unknown)
    float    fit_error;             // Current Koopman fit quality
    uint32_t snapshot_count;        // Snapshots collected so far
} CipherKRDecision;

// ---------------------------------------------------------------------------
// Feature extraction
//
// Convert kernel launch parameters into a normalized feature vector
// for EDMD snapshot collection. This is what the Koopman operator learns
// to map: input_features → output_features.
//
// For GEMM-class ops:
//   input features ≈ (M, N, K normalized, tile config, occupancy)
//   output features = the actual computation result representation
//
// In the CPU stub, we use the launch geometry as a proxy for the computation.
// On real GPU: we'd sample actual tensor values pre/post kernel.
// ---------------------------------------------------------------------------

// Extract normalized input features from kernel launch parameters
void cipher_kr_extract_features(
    uint8_t  op_class,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_size, uint32_t shmem_bytes,
    const CipherLiquidStateMgr* liquid,
    float*   features_out,    // CIPHER_KR_FEATURE_DIM floats
    uint32_t n_features);

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Initialize the runtime. Call once at CIPHER startup.
void cipher_kr_init(CipherKoopmanRuntime* kr);

// Main entry point — called from cipher_dispatch for every novel op.
//
// Returns a decision:
//   should_substitute = false, is_collecting = true:
//     → Execute real kernel, then call cipher_kr_record_output()
//   should_substitute = true:
//     → Use derived surrogate (skip real kernel)
//   should_substitute = false, is_collecting = false:
//     → Op failed EDMD, treat as passthrough forever
CipherKRDecision cipher_kr_decide(
    CipherKoopmanRuntime*       kr,
    uint8_t                     op_class,
    uint32_t                    grid_x,
    uint32_t                    grid_y,
    uint32_t                    grid_z,
    uint32_t                    block_size,
    uint32_t                    shmem_bytes,
    const CipherLiquidStateMgr* liquid);

// Record the output of a passthrough execution.
// Call this AFTER the real kernel runs, with the output features.
// This adds a snapshot to the EDMD buffer.
void cipher_kr_record_output(
    CipherKoopmanRuntime* kr,
    int                   record_idx,
    const float*          input_features,
    const float*          output_features,
    uint32_t              n_features);

// Predict the output for a given input using the derived Koopman surrogate.
// Returns true if prediction succeeded (surrogate is ready).
bool cipher_kr_predict(
    CipherKoopmanRuntime* kr,
    int                   record_idx,
    const float*          input_features,
    float*                output_features,
    uint32_t              n_features);

// Report summary statistics.
void cipher_kr_report(const CipherKoopmanRuntime* kr);

// Compute op hash from launch parameters.
uint64_t cipher_kr_hash(uint8_t op_class,
                         uint32_t grid_x, uint32_t grid_y,
                         uint32_t block_size, uint32_t shmem_bytes);

// Find or create a record for an op. Returns index into records[].
// Returns -1 if table is full.
int cipher_kr_find_or_create(CipherKoopmanRuntime* kr,
                              uint64_t op_hash,
                              uint8_t  op_class,
                              uint32_t grid_x, uint32_t grid_y,
                              uint32_t block_size, uint32_t shmem_bytes);

#ifdef __cplusplus
}
#endif
