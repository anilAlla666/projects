// ==========================================================================
// HyperFlux SDK - Destruction Kernel
// Destructible environment damage propagation
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================
#pragma once

#include "../hyperflux_types.h"

namespace hyperflux {

class Destruction {
public:
    Destruction();
    ~Destruction();

    // Non-copyable, movable
    Destruction(const Destruction&) = delete;
    Destruction& operator=(const Destruction&) = delete;
    Destruction(Destruction&&) noexcept;
    Destruction& operator=(Destruction&&) noexcept;

    /// Initialize kernel. Call once before inference.
    Status Init();

    /// Run single inference.
    Status Infer(const DestructionInput& input, DestructionOutput* output);

    /// Run batched inference. batch_size must be <= HF_MAX_BATCH.
    Status InferBatch(const DestructionInput* inputs, DestructionOutput* outputs,
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
