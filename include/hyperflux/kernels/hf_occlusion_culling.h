// ==========================================================================
// HyperFlux SDK - OcclusionCulling Kernel
// GPU-free occlusion culling determination
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================
#pragma once

#include "../hyperflux_types.h"

namespace hyperflux {

class OcclusionCulling {
public:
    OcclusionCulling();
    ~OcclusionCulling();

    // Non-copyable, movable
    OcclusionCulling(const OcclusionCulling&) = delete;
    OcclusionCulling& operator=(const OcclusionCulling&) = delete;
    OcclusionCulling(OcclusionCulling&&) noexcept;
    OcclusionCulling& operator=(OcclusionCulling&&) noexcept;

    /// Initialize kernel. Call once before inference.
    Status Init();

    /// Run single inference.
    Status Infer(const OcclusionCullingInput& input, OcclusionCullingOutput* output);

    /// Run batched inference. batch_size must be <= HF_MAX_BATCH.
    Status InferBatch(const OcclusionCullingInput* inputs, OcclusionCullingOutput* outputs,
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
