// ==========================================================================
// HyperFlux SDK - LODSelection Kernel
// Level-of-detail selection for mesh rendering
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================
#pragma once

#include "../hyperflux_types.h"

namespace hyperflux {

class LODSelection {
public:
    LODSelection();
    ~LODSelection();

    // Non-copyable, movable
    LODSelection(const LODSelection&) = delete;
    LODSelection& operator=(const LODSelection&) = delete;
    LODSelection(LODSelection&&) noexcept;
    LODSelection& operator=(LODSelection&&) noexcept;

    /// Initialize kernel. Call once before inference.
    Status Init();

    /// Run single inference.
    Status Infer(const LODSelectionInput& input, LODSelectionOutput* output);

    /// Run batched inference. batch_size must be <= HF_MAX_BATCH.
    Status InferBatch(const LODSelectionInput* inputs, LODSelectionOutput* outputs,
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
