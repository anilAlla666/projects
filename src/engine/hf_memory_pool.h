// ==========================================================================
// HyperFlux SDK - Aligned Memory Pool
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================
#pragma once

#include "../include/hyperflux/hyperflux_config.h"
#include <cstddef>
#include <cstdint>

namespace hyperflux {
namespace internal {

/// Allocate aligned memory (HF_ALIGNMENT byte boundary)
void* AlignedAlloc(size_t size);

/// Free aligned memory
void AlignedFree(void* ptr);

/// Allocate and zero-initialize aligned memory
void* AlignedCalloc(size_t count, size_t elem_size);

/// Pre-allocated scratch buffer for inference (per-thread)
class ScratchBuffer {
public:
    ScratchBuffer();
    ~ScratchBuffer();

    /// Ensure buffer has at least 'bytes' capacity
    bool Reserve(size_t bytes);

    /// Get pointer to scratch memory
    int8_t* Data() { return data_; }
    float* DataFloat() { return reinterpret_cast<float*>(data_); }

    size_t Capacity() const { return capacity_; }

private:
    int8_t* data_ = nullptr;
    size_t capacity_ = 0;
};

}} // namespace hyperflux::internal
