// ==========================================================================
// HyperFlux SDK - AudioOcclusion Kernel Implementation
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================

#include "../include/hyperflux/kernels/hf_audio_occlusion.h"
#include "../engine/hf_inference_engine.h"
#include "../engine/hf_memory_pool.h"
#include "../engine/hf_weight_loader.h"

// Compiled weight data
#include "../weights/hf_audio_occlusion_w.inc"

namespace hyperflux {

struct AudioOcclusion::Impl {
    internal::ModelDesc desc_;
    internal::ScratchBuffer scratch_;
    KernelStats stats_;
    bool ready_ = false;
};

AudioOcclusion::AudioOcclusion() : impl_(new Impl()) {}
AudioOcclusion::~AudioOcclusion() { Shutdown(); delete impl_; }

AudioOcclusion::AudioOcclusion(AudioOcclusion&&o) noexcept : impl_(o.impl_) { o.impl_ = new Impl(); }
AudioOcclusion& AudioOcclusion::operator=(AudioOcclusion&&o) noexcept {
    if (this != &o) { Shutdown(); delete impl_; impl_ = o.impl_; o.impl_ = new Impl(); }
    return *this;
}

Status AudioOcclusion::Init() {
    if (impl_->ready_) return Status::OK;

    auto& desc_ = impl_->desc_;
    desc_.name = "audio_occlusion";
    desc_.num_layers = 4;
    desc_.total_in_dim = 48;
    desc_.total_out_dim = 12;

        desc_.layers[0].in_dim = 48;
        desc_.layers[0].out_dim = 192;
        desc_.layers[0].in_dim_padded = 48;
        desc_.layers[0].weights = s_audio_occlusion_w0;
        desc_.layers[0].scales = s_audio_occlusion_s0;
        desc_.layers[0].biases = s_audio_occlusion_b0;
        desc_.layers[0].input_scale = s_audio_occlusion_iscale[0];
        desc_.layers[0].input_zero_point = s_audio_occlusion_izp[0];
        desc_.layers[1].in_dim = 192;
        desc_.layers[1].out_dim = 192;
        desc_.layers[1].in_dim_padded = 192;
        desc_.layers[1].weights = s_audio_occlusion_w1;
        desc_.layers[1].scales = s_audio_occlusion_s1;
        desc_.layers[1].biases = s_audio_occlusion_b1;
        desc_.layers[1].input_scale = s_audio_occlusion_iscale[1];
        desc_.layers[1].input_zero_point = s_audio_occlusion_izp[1];
        desc_.layers[2].in_dim = 192;
        desc_.layers[2].out_dim = 192;
        desc_.layers[2].in_dim_padded = 192;
        desc_.layers[2].weights = s_audio_occlusion_w2;
        desc_.layers[2].scales = s_audio_occlusion_s2;
        desc_.layers[2].biases = s_audio_occlusion_b2;
        desc_.layers[2].input_scale = s_audio_occlusion_iscale[2];
        desc_.layers[2].input_zero_point = s_audio_occlusion_izp[2];
        desc_.layers[3].in_dim = 192;
        desc_.layers[3].out_dim = 12;
        desc_.layers[3].in_dim_padded = 192;
        desc_.layers[3].weights = s_audio_occlusion_w3;
        desc_.layers[3].scales = s_audio_occlusion_s3;
        desc_.layers[3].biases = s_audio_occlusion_b3;
        desc_.layers[3].input_scale = s_audio_occlusion_iscale[3];
        desc_.layers[3].input_zero_point = s_audio_occlusion_izp[3];

    // Pre-allocate scratch buffer
    impl_->scratch_.Reserve(960);

    impl_->stats_ = {};
    impl_->stats_.model_size_bytes = sizeof(s_audio_occlusion_w0); // Approximate
    impl_->ready_ = true;

    HF_LOG("AudioOcclusion initialized: %d layers, %d->%d",
           desc_.num_layers, desc_.total_in_dim, desc_.total_out_dim);
    return Status::OK;
}

Status AudioOcclusion::Infer(const AudioOcclusionInput& input, AudioOcclusionOutput* output) {
    if (!impl_->ready_) return Status::ERROR_NOT_INITIALIZED;
    if (!output) return Status::ERROR_INVALID_INPUT;

    HF_PROFILE_BEGIN(infer);

    // Marshal input struct → flat FP32 array
    static_assert(sizeof(AudioOcclusionInput) >= 48 * sizeof(float) ||
                  sizeof(AudioOcclusionInput) > 0,
                  "Input struct size check");
    const float* in_data = reinterpret_cast<const float*>(&input);
    float out_data[12];

    Status s = internal::RunInference(
        impl_->desc_, in_data, out_data, impl_->scratch_,
        internal::Activation::RELU,
        Activation::NONE
    );

    if (s == Status::OK) {
        // Marshal flat FP32 output → output struct
        float* out_ptr = reinterpret_cast<float*>(output);
        for (int i = 0; i < 12; i++) {
            out_ptr[i] = out_data[i];
        }
