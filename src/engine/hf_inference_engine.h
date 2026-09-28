// ==========================================================================
// HyperFlux SDK - Core Inference Engine
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// INTERNAL - Not part of public API.
// ==========================================================================
#pragma once

#include "../include/hyperflux/hyperflux_config.h"
#include "../include/hyperflux/hyperflux_types.h"
#include "hf_weight_loader.h"
#include "hf_memory_pool.h"
#include "hf_simd_dispatch.h"

namespace hyperflux {
namespace internal {

/// Core inference: runs a full forward pass through a compiled model.
/// Input/output are FP32. Internal computation is INT8.
///
/// @param model    Model descriptor with layer info and weights
/// @param input    FP32 input array [model.total_in_dim]
/// @param output   FP32 output array [model.total_out_dim]
/// @param scratch  Pre-allocated scratch buffer
/// @param hidden_act  Activation for hidden layers (typically RELU)
/// @param output_act  Activation for output layer
/// @return Status::OK on success
Status RunInference(
    const ModelDesc& model,
    const float* input,
    float* output,
    ScratchBuffer& scratch,
    Activation hidden_act = Activation::RELU,
    Activation output_act = Activation::NONE
);

/// Batch inference: runs multiple inputs through the same model.
Status RunInferenceBatch(
    const ModelDesc& model,
    const float* inputs,        // [batch_size × model.total_in_dim]
    float* outputs,             // [batch_size × model.total_out_dim]
    uint32_t batch_size,
    ScratchBuffer& scratch,
    Activation hidden_act = Activation::RELU,
    Activation output_act = Activation::NONE
);

/// Quantize FP32 input to INT8 using scale and zero_point.
void QuantizeInput(const float* input, int8_t* output, int dim,
                   float scale, float zero_point);

/// Apply activation function in-place.
void ApplyActivation(float* data, int dim, Activation act);

}} // namespace hyperflux::internal
