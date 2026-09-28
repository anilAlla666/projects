// ==========================================================================
// HyperFlux SDK - Aligned Memory Pool Implementation
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================

#include "hf_memory_pool.h"
#include <cstdlib>
#include <cstring>

namespace hyperflux {
namespace internal {

void* AlignedAlloc(size_t size) {
    // Round up to alignment
    size = (size + HF_ALIGNMENT - 1) & ~(size_t)(HF_ALIGNMENT - 1);

#if HF_PLATFORM_WINDOWS
    return _aligned_malloc(size, HF_ALIGNMENT);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, HF_ALIGNMENT, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void AlignedFree(void* ptr) {
    if (!ptr) return;
#if HF_PLATFORM_WINDOWS
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void* AlignedCalloc(size_t count, size_t elem_size) {
    size_t total = count * elem_size;
    void* ptr = AlignedAlloc(total);
    if (ptr) {
        memset(ptr, 0, total);
    }
    return ptr;
}

ScratchBuffer::ScratchBuffer() : data_(nullptr), capacity_(0) {}

ScratchBuffer::~ScratchBuffer() {
    if (data_) AlignedFree(data_);
}

bool ScratchBuffer::Reserve(size_t bytes) {
    if (bytes <= capacity_) return true;
    if (data_) AlignedFree(data_);
    data_ = static_cast<int8_t*>(AlignedAlloc(bytes));
    if (!data_) {
        capacity_ = 0;
        return false;
    }
    capacity_ = bytes;
    return true;
}

}} // namespace hyperflux::internal
