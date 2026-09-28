// ==========================================================================
// HyperFlux SDK - LumenGI Kernel
// Global illumination probe computation
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================
#pragma once

#include "../hyperflux_types.h"

namespace hyperflux {

class LumenGI {
public:
    LumenGI();
    ~LumenGI();

    // Non-copyable, movable
    LumenGI(const LumenGI&) = delete;
    LumenGI& operator=(const LumenGI&) = delete;
    LumenGI(LumenGI&&) noexcept;
    LumenGI& operator=(LumenGI&&) noexcept;

    /// Initialize kernel. Call once before inference.
    Status Init();

    /// Run single inference.
    Status Infer(const LumenGIInput& input, LumenGIOutput* output);

    /// Run batched inference. batch_size must be <= HF_MAX_BATCH.
    Status InferBatch(const LumenGIInput* inputs, LumenGIOutput* outputs,
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
