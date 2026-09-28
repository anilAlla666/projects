// ==========================================================================
// HyperFlux SDK - AudioOcclusion Kernel
// Sound propagation and occlusion
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================
#pragma once

#include "../hyperflux_types.h"

namespace hyperflux {

class AudioOcclusion {
public:
    AudioOcclusion();
    ~AudioOcclusion();

    // Non-copyable, movable
    AudioOcclusion(const AudioOcclusion&) = delete;
    AudioOcclusion& operator=(const AudioOcclusion&) = delete;
    AudioOcclusion(AudioOcclusion&&) noexcept;
    AudioOcclusion& operator=(AudioOcclusion&&) noexcept;

    /// Initialize kernel. Call once before inference.
    Status Init();

    /// Run single inference.
    Status Infer(const AudioOcclusionInput& input, AudioOcclusionOutput* output);

    /// Run batched inference. batch_size must be <= HF_MAX_BATCH.
    Status InferBatch(const AudioOcclusionInput* inputs, AudioOcclusionOutput* outputs,
                      uint32_t batch_size);

    /// Release resources.
    void Shutdown();

    /// Get runtime statistics (available in profile mode).
    KernelStats GetStats() const;

    /// Check if kernel is initialized and ready.
    bool IsReady() const;

private:
    struct Impl;
    Impl* impl_;
};

} // namespace hyperflux
