// ==========================================================================
// HyperFlux SDK - OcclusionCulling Kernel Implementation
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================

#include "../include/hyperflux/kernels/hf_occlusion_culling.h"
#include "../engine/hf_inference_engine.h"
#include "../engine/hf_memory_pool.h"
#include "../engine/hf_weight_loader.h"

// Compiled weight data
#include "../weights/hf_occlusion_culling_w.inc"

namespace hyperflux {

struct OcclusionCulling::Impl {
    internal::ModelDesc desc_;
    internal::ScratchBuffer scratch_;
    KernelStats stats_;
    bool ready_ = false;
};

OcclusionCulling::OcclusionCulling() : impl_(new Impl()) {}
OcclusionCulling::~OcclusionCulling() { Shutdown(); delete impl_; }

OcclusionCulling::OcclusionCulling(OcclusionCulling&&o) noexcept : impl_(o.impl_) { o.impl_ = new Impl(); }
OcclusionCulling& OcclusionCulling::operator=(OcclusionCulling&&o) noexcept {
    if (this != &o) { Shutdown(); delete impl_; impl_ = o.impl_; o.impl_ = new Impl(); }
    return *this;
}

Status OcclusionCulling::Init() {
    if (impl_->ready_) return Status::OK;

    auto& desc_ = impl_->desc_;
    desc_.name = "occlusion_culling";
    desc_.num_layers = 4;
    desc_.total_in_dim = 64;
    desc_.total_out_dim = 16;

        desc_.layers[0].in_dim = 64;
        desc_.layers[0].out_dim = 192;
        desc_.layers[0].in_dim_padded = 64;
        desc_.layers[0].weights = s_occlusion_culling_w0;
        desc_.layers[0].scales = s_occlusion_culling_s0;
        desc_.layers[0].biases = s_occlusion_culling_b0;
        desc_.layers[0].input_scale = s_occlusion_culling_iscale[0];
        desc_.layers[0].input_zero_point = s_occlusion_culling_izp[0];
        desc_.layers[1].in_dim = 192;
        desc_.layers[1].out_dim = 192;
        desc_.layers[1].in_dim_padded = 192;
        desc_.layers[1].weights = s_occlusion_culling_w1;
        desc_.layers[1].scales = s_occlusion_culling_s1;
        desc_.layers[1].biases = s_occlusion_culling_b1;
        desc_.layers[1].input_scale = s_occlusion_culling_iscale[1];
        desc_.layers[1].input_zero_point = s_occlusion_culling_izp[1];
        desc_.layers[2].in_dim = 192;
        desc_.layers[2].out_dim = 192;
        desc_.layers[2].in_dim_padded = 192;
        desc_.layers[2].weights = s_occlusion_culling_w2;
        desc_.layers[2].scales = s_occlusion_culling_s2;
        desc_.layers[2].biases = s_occlusion_culling_b2;
        desc_.layers[2].input_scale = s_occlusion_culling_iscale[2];
        desc_.layers[2].input_zero_point = s_occlusion_culling_izp[2];
        desc_.layers[3].in_dim = 192;
        desc_.layers[3].out_dim = 16;
        desc_.layers[3].in_dim_padded = 192;
        desc_.layers[3].weights = s_occlusion_culling_w3;
        desc_.layers[3].scales = s_occlusion_culling_s3;
        desc_.layers[3].biases = s_occlusion_culling_b3;
        desc_.layers[3].input_scale = s_occlusion_culling_iscale[3];
        desc_.layers[3].input_zero_point = s_occlusion_culling_izp[3];

    // Pre-allocate scratch buffer
    impl_->scratch_.Reserve(960);

    impl_->stats_ = {};
    impl_->stats_.model_size_bytes = sizeof(s_occlusion_culling_w0); // Approximate
    impl_->ready_ = true;

    HF_LOG("OcclusionCulling initialized: %d layers, %d->%d",
           desc_.num_layers, desc_.total_in_dim, desc_.total_out_dim);
    return Status::OK;
}

Status OcclusionCulling::Infer(const OcclusionCullingInput& input, OcclusionCullingOutput* output) {
    if (!impl_->ready_) return Status::ERROR_NOT_INITIALIZED;
    if (!output) return Status::ERROR_INVALID_INPUT;

    HF_PROFILE_BEGIN(infer);

    // Marshal input struct → flat FP32 array
    static_assert(sizeof(OcclusionCullingInput) >= 64 * sizeof(float) ||
                  sizeof(OcclusionCullingInput) > 0,
                  "Input struct size check");
    const float* in_data = reinterpret_cast<const float*>(&input);
    float out_data[16];

    Status s = internal::RunInference(
        impl_->desc_, in_data, out_data, impl_->scratch_,
        internal::Activation::RELU,
        Activation::SIGMOID
    );

    if (s == Status::OK) {
        // Marshal flat FP32 output → output struct
        float* out_ptr = reinterpret_cast<float*>(output);
        for (int i = 0; i < 16; i++) {
            out_ptr[i] = out_data[i];
        }
