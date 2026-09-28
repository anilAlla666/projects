// ==========================================================================
// HyperFlux SDK - Visibility Kernel
// Line-of-sight visibility determination
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================
#pragma once

#include "../hyperflux_types.h"

namespace hyperflux {

class Visibility {
public:
    Visibility();
    ~Visibility();

    // Non-copyable, movable
    Visibility(const Visibility&) = delete;
    Visibility& operator=(const Visibility&) = delete;
    Visibility(Visibility&&) noexcept;
    Visibility& operator=(Visibility&&) noexcept;

    /// Initialize kernel. Call once before inference.
    Status Init();

    /// Run single inference.
    Status Infer(const VisibilityInput& input, VisibilityOutput* output);

    /// Run batched inference. batch_size must be <= HF_MAX_BATCH.
    Status InferBatch(const VisibilityInput* inputs, VisibilityOutput* outputs,
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
