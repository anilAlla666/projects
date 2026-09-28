// ==========================================================================
// HyperFlux SDK - Hitbox Kernel
// Collision detection against skeletal meshes
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================
#pragma once

#include "../hyperflux_types.h"

namespace hyperflux {

class Hitbox {
public:
    Hitbox();
    ~Hitbox();

    // Non-copyable, movable
    Hitbox(const Hitbox&) = delete;
    Hitbox& operator=(const Hitbox&) = delete;
    Hitbox(Hitbox&&) noexcept;
    Hitbox& operator=(Hitbox&&) noexcept;

    /// Initialize kernel. Call once before inference.
    Status Init();

    /// Run single inference.
    Status Infer(const HitboxInput& input, HitboxOutput* output);

    /// Run batched inference. batch_size must be <= HF_MAX_BATCH.
    Status InferBatch(const HitboxInput* inputs, HitboxOutput* outputs,
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
