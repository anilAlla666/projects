// ==========================================================================
// HyperFlux SDK - Weight Loader with Integrity Verification
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// INTERNAL - Not part of public API.
// ==========================================================================
#pragma once

#include "../include/hyperflux/hyperflux_config.h"
#include <cstdint>
#include <cstddef>

namespace hyperflux {
namespace internal {

// Compiled-in model descriptor
struct LayerDesc {
    int in_dim;
    int out_dim;
    int in_dim_padded;          // Padded to SIMD alignment
    const int8_t* weights;      // Pointer into compiled data
    const float* scales;        // Per-output-channel scales
    const float* biases;        // Per-output biases
    float input_scale;          // Input quantization scale
    float input_zero_point;     // Input zero point
};

struct ModelDesc {
    int num_layers;
    LayerDesc layers[HF_MAX_LAYERS];
    int total_in_dim;           // First layer input dim
    int total_out_dim;          // Last layer output dim
    const char* name;
    uint32_t checksum;          // Weight integrity check
};

// Activation types
enum class Activation { NONE, RELU, SIGMOID, TANH };

/// Decode obfuscated weight block in-place.
/// Returns true if checksum passes.
bool DecodeWeights(int8_t* data, size_t len, uint32_t expected_checksum);

/// Compute checksum over weight data.
uint32_t ComputeChecksum(const int8_t* data, size_t len);

}} // namespace hyperflux::internal
