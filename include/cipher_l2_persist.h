// =============================================================================
// CIPHER — F3: L2 Persistent Weight Loading
// cipher_l2_persist.h
//
// Forces CIPHER's LNN weights (<1MB each × 3 LNNs = <3MB total) to live
// permanently in the H100's 50MB L2 cache using cudaAccessPropertyPersisting.
//
// HOW IT WORKS:
// CUDA's Stream Attribute API lets us tag memory regions with access
// properties. CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW with
// cudaAccessPropertyPersisting pins data in L2 between kernel launches —
// it will only be evicted if the region is explicitly unpinned or another
// larger persistent region displaces it.
//
// MATH:
//   H100 L2 = 50MB
//   3 × LNN weight tensors ≤ 1MB each = ≤3MB
//   CIPHER L2 footprint = 3MB / 50MB = 6%
//   Remaining for user activations/KV-cache = 94%
//
// PROVEN: Experiment 1 this week confirmed LNN weights self-cache in L2
// with zero explicit management overhead. This formalizes that as policy.
//
// SUCCESS CRITERION: Weights hot in L2 across kernel launches.
// DEPENDENCY: F2 (Green Context streams must exist before pinning).
// =============================================================================

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum L2 persist budget CIPHER will claim (bytes)
// Keep well under 10% of 50MB H100 L2 to avoid displacing user activations
#define CIPHER_L2_PERSIST_MAX_BYTES  (4 * 1024 * 1024)   // 4MB ceiling

// One weight tensor registration entry
typedef struct {
    void*   device_ptr;          // cudaMalloc'd weight tensor
    size_t  size_bytes;          // Tensor size
    int     layer_id;            // 0=L3, 1=L2, 2=L1
    bool    pinned;              // Currently persisted in L2
    char    name[32];            // Debug label
} CipherWeightTensor;

#define CIPHER_MAX_WEIGHT_TENSORS  16   // 3 LNNs × ~5 layers each

typedef struct {
    CipherWeightTensor  tensors[CIPHER_MAX_WEIGHT_TENSORS];
    int                 count;
    size_t              total_bytes;         // Sum of all pinned sizes
    size_t              l2_capacity_bytes;   // Hardware L2 capacity
    cudaStream_t        persist_stream;      // Stream used for pin operations
    bool                initialized;
} CipherL2PersistState;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Initialize — queries L2 capacity, validates we're within budget.
cudaError_t cipher_l2_persist_init(CipherL2PersistState* state,
                                   cudaStream_t          stream);

// Register a weight tensor for L2 persistence.
// Call once per tensor after cudaMalloc + weight copy.
cudaError_t cipher_l2_persist_register(CipherL2PersistState* state,
                                       void*   device_ptr,
                                       size_t  size_bytes,
                                       int     layer_id,
                                       const char* name);

// Apply all registered persistence windows to the stream.
// Call after all tensors are registered. Safe to call multiple times.
cudaError_t cipher_l2_persist_apply(CipherL2PersistState* state);

// Remove L2 persistence for a specific tensor (e.g. for LNN weight update).
cudaError_t cipher_l2_persist_unpin(CipherL2PersistState* state,
                                    void* device_ptr);

// Remove all persistence windows — call before process exit or weight reload.
void cipher_l2_persist_reset(CipherL2PersistState* state);

// Report L2 footprint to stderr.
void cipher_l2_persist_report(const CipherL2PersistState* state);

#ifdef __cplusplus
}
#endif
