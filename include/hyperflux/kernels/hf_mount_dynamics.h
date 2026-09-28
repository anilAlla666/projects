// ==========================================================================
// HyperFlux SDK - MountDynamics Kernel
// Vehicle/mount physics state prediction
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================
#pragma once

#include "../hyperflux_types.h"

namespace hyperflux {

class MountDynamics {
public:
    MountDynamics();
    ~MountDynamics();

    // Non-copyable, movable
    MountDynamics(const MountDynamics&) = delete;
    MountDynamics& operator=(const MountDynamics&) = delete;
    MountDynamics(MountDynamics&&) noexcept;
    MountDynamics& operator=(MountDynamics&&) noexcept;

    /// Initialize kernel. Call once before inference.
    Status Init();

    /// Run single inference.
    Status Infer(const MountDynamicsInput& input, MountDynamicsOutput* output);

    /// Run batched inference. batch_size must be <= HF_MAX_BATCH.
    Status InferBatch(const MountDynamicsInput* inputs, MountDynamicsOutput* outputs,
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
