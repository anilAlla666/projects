// ==========================================================================
// HyperFlux SDK - SpawnSelection Kernel
// Optimal spawn point scoring and ranking
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================
#pragma once

#include "../hyperflux_types.h"

namespace hyperflux {

class SpawnSelection {
public:
    SpawnSelection();
    ~SpawnSelection();

    // Non-copyable, movable
    SpawnSelection(const SpawnSelection&) = delete;
    SpawnSelection& operator=(const SpawnSelection&) = delete;
    SpawnSelection(SpawnSelection&&) noexcept;
    SpawnSelection& operator=(SpawnSelection&&) noexcept;

    /// Initialize kernel. Call once before inference.
    Status Init();

    /// Run single inference.
    Status Infer(const SpawnSelectionInput& input, SpawnSelectionOutput* output);

    /// Run batched inference. batch_size must be <= HF_MAX_BATCH.
    Status InferBatch(const SpawnSelectionInput* inputs, SpawnSelectionOutput* outputs,
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
