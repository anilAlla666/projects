// ==========================================================================
// HyperFlux SDK - Weight Loader Implementation
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================

#include "hf_weight_loader.h"
#include <cstring>

namespace hyperflux {
namespace internal {

// Obfuscation key (rotated XOR)
static constexpr uint8_t OBF_KEY[] = {
    0x4E, 0x44, 0x48, 0x46, 0x2D, 0x31, 0x2E, 0x30,  // "NDHF-1.0"
    0xA7, 0x3B, 0xC5, 0x19, 0xE2, 0x6F, 0x8D, 0x54,
    0x91, 0x0C, 0xF3, 0x7A, 0xB8, 0x25, 0xD6, 0x43,
    0x6E, 0x1D, 0xAF, 0x82, 0x59, 0xC0, 0x37, 0xE4
};
static constexpr size_t OBF_KEY_LEN = sizeof(OBF_KEY);

bool DecodeWeights(int8_t* data, size_t len, uint32_t expected_checksum) {
    // XOR decode
    uint8_t* udata = reinterpret_cast<uint8_t*>(data);
    for (size_t i = 0; i < len; i++) {
        udata[i] ^= OBF_KEY[i % OBF_KEY_LEN];
    }

    // Verify checksum
    uint32_t actual = ComputeChecksum(data, len);
    if (actual != expected_checksum) {
        // Re-encode on failure (don't leave decoded data in memory)
        for (size_t i = 0; i < len; i++) {
            udata[i] ^= OBF_KEY[i % OBF_KEY_LEN];
        }
        return false;
    }
    return true;
}

uint32_t ComputeChecksum(const int8_t* data, size_t len) {
    // FNV-1a hash
    uint32_t hash = 0x811C9DC5u;
    const uint8_t* udata = reinterpret_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; i++) {
        hash ^= udata[i];
        hash *= 0x01000193u;
    }
    return hash;
}

}} // namespace hyperflux::internal
