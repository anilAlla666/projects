// ==========================================================================
// HyperFlux SDK - Core Inference Engine Implementation
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================

#include "hf_inference_engine.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace hyperflux {
namespace internal {

void QuantizeInput(const float* input, int8_t* output, int dim,
                   float scale, float zero_point) {
    for (int i = 0; i < dim; i++) {
        float val = input[i] / scale + zero_point;
        val = std::max(-128.0f, std::min(127.0f, std::round(val)));
        output[i] = static_cast<int8_t>(val);
    }
    // Zero-pad to alignment
    int padded = (dim + 15) & ~15;
    for (int i = dim; i < padded; i++) {
        output[i] = 0;
    }
}

void ApplyActivation(float* data, int dim, Activation act) {
    switch (act) {
        case Activation::RELU:
            for (int i = 0; i < dim; i++) {
                data[i] = std::max(0.0f, data[i]);
            }
            break;
        case Activation::SIGMOID:
            for (int i = 0; i < dim; i++) {
                data[i] = 1.0f / (1.0f + std::exp(-data[i]));
            }
            break;
        case Activation::TANH:
            for (int i = 0; i < dim; i++) {
                data[i] = std::tanh(data[i]);
            }
            break;
        case Activation::NONE:
            break;
    }
}

Status RunInference(
    const ModelDesc& model,
    const float* input,
    float* output,
    ScratchBuffer& scratch,
    Activation hidden_act,
    Activation output_act)
{
    HF_ASSERT(model.num_layers > 0 && model.num_layers <= HF_MAX_LAYERS);
    HF_ASSERT(g_matvec_int8 != nullptr);

    // Calculate scratch needed: 2 buffers for ping-pong
    // INT8 buffer (max dim padded) + FP32 buffer (max dim)
    int max_dim = 0;
    for (int l = 0; l < model.num_layers; l++) {
        max_dim = std::max(max_dim, model.layers[l].in_dim_padded);
        max_dim = std::max(max_dim, model.layers[l].out_dim);
    }

    size_t int8_buf_size = ((max_dim + 63) & ~63) * sizeof(int8_t);
    size_t fp32_buf_size = max_dim * sizeof(float);
    size_t total_scratch = int8_buf_size + fp32_buf_size;

    if (!scratch.Reserve(total_scratch)) {
        return Status::ERROR_OUT_OF_MEMORY;
    }

    int8_t* quant_buf = scratch.Data();
    float* float_buf = reinterpret_cast<float*>(scratch.Data() + int8_buf_size);

    // Layer 0: quantize FP32 input → INT8
    QuantizeInput(input, quant_buf,
                  model.layers[0].in_dim,
                  model.layers[0].input_scale,
                  model.layers[0].input_zero_point);

    // Forward pass through all layers
    for (int l = 0; l < model.num_layers; l++) {
        const LayerDesc& layer = model.layers[l];
        bool is_last = (l == model.num_layers - 1);

        // INT8 matmul → FP32 output
        float* layer_out = is_last ? output : float_buf;

        g_matvec_int8(
            layer.weights,
            quant_buf,
            layer_out,
            layer.scales,
            layer.biases,
            layer.input_scale,
            layer.in_dim_padded,
            layer.out_dim
        );

        // Apply activation
        Activation act = is_last ? output_act : hidden_act;
        if (act != Activation::NONE) {
            ApplyActivation(layer_out, layer.out_dim, act);
        }

        // Re-quantize for next layer (unless last)
        if (!is_last) {
            const LayerDesc& next = model.layers[l + 1];
            QuantizeInput(layer_out, quant_buf,
                         layer.out_dim,
                         next.input_scale,
                         next.input_zero_point);
        }
    }

    return Status::OK;
}

Status RunInferenceBatch(
    const ModelDesc& model,
    const float* inputs,
    float* outputs,
    uint32_t batch_size,
    ScratchBuffer& scratch,
    Activation hidden_act,
    Activation output_act)
{
    if (batch_size == 0 || batch_size > HF_MAX_BATCH) {
        return Status::ERROR_INVALID_BATCH_SIZE;
    }

    // Simple loop for now. TODO: tile for cache optimization.
    for (uint32_t b = 0; b < batch_size; b++) {
        Status s = RunInference(
            model,
            inputs + b * model.total_in_dim,
            outputs + b * model.total_out_dim,
            scratch,
            hidden_act,
            output_act
        );
        if (s != Status::OK) return s;
    }
    return Status::OK;
}

}} // namespace hyperflux::internal
