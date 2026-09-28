// =============================================================================
// CIPHER Classification Engine v0.1
// Neural Dynamics — Anil Kumar Alla
//
// Zero-training operation fingerprinter at cuLaunchKernel intercept.
// Classifies GPU kernel launches into 7 operation classes using grid/block/
// sharedMem geometry alone. No ML inference. No heap alloc. No training data.
// Ships day one.
//
// Target: <100ns per classify() call (cache miss), <5ns (cache hit)
// Thread safety: lock-free read path, atomic cache writes
// =============================================================================

#pragma once

#include <cstdint>
#include <cmath>
#include <atomic>
#include <array>
#include <algorithm>

namespace cipher {

// ---------------------------------------------------------------------------
// 7 Operation Classes
// ---------------------------------------------------------------------------
enum class OpClass : uint8_t {
    GEMM             = 0,  // Matrix multiply variants — tiled 2D grid, square blocks
    ATTENTION        = 1,  // Scaled dot-product attention — O(N²) seq structure
    CONVOLUTION      = 2,  // Spatial conv — 3D grid, 2D spatial block
    ELEMENTWISE      = 3,  // Pointwise ops — flat grid, 1D block, zero shared mem
    REDUCTION        = 4,  // Softmax/LayerNorm/Sum — small grid, large 1D block, shared ∝ block
    MEMCPY_TRANSPOSE = 5,  // Data movement — 2D tile, shared == tile² × dtype
    ITERATIVE_CUSTOM = 6,  // Catch-all — irregular geometry, custom loops
};

static constexpr const char* kOpClassName[] = {
    "GEMM", "ATTENTION", "CONVOLUTION",
    "ELEMENTWISE", "REDUCTION", "MEMCPY_TRANSPOSE", "ITERATIVE_CUSTOM"
};

// ---------------------------------------------------------------------------
// Kernel geometry — exactly what cuLaunchKernel exposes
// ---------------------------------------------------------------------------
struct KernelGeom {
    uint32_t gx, gy, gz;          // gridDim
    uint32_t bx, by, bz;          // blockDim
    uint32_t shared_bytes;
    const void* fn;                // kernel function pointer (cache key)
};

// ---------------------------------------------------------------------------
// Classification result
// ---------------------------------------------------------------------------
struct ClassifyResult {
    OpClass  op;
    uint8_t  confidence;           // 0–100, heuristic strength
    bool     cache_hit;
};

// ---------------------------------------------------------------------------
// Geometry helpers — all integer, branchless where possible
// ---------------------------------------------------------------------------
namespace detail {

inline uint32_t threads_per_block(const KernelGeom& g) noexcept {
    return g.bx * g.by * g.bz;
}

inline uint64_t total_threads(const KernelGeom& g) noexcept {
    return (uint64_t)g.gx * g.gy * g.gz * threads_per_block(g);
}

inline bool is_power_of_two(uint32_t v) noexcept {
    return v && !(v & (v - 1));
}

inline uint32_t isqrt(uint32_t n) noexcept {
    uint32_t r = (uint32_t)std::sqrt((float)n);
    while (r * r > n) --r;
    while ((r+1)*(r+1) <= n) ++r;
    return r;
}

inline bool is_square(uint32_t n) noexcept {
    uint32_t r = isqrt(n);
    return r * r == n;
}

// Shared mem consistent with a flat tile of floats: shared == side² × 4
inline bool shared_matches_tile(uint32_t shared, uint32_t tpb) noexcept {
    if (shared == 0 || shared > 49152) return false;
    uint32_t side = isqrt(tpb);
    uint32_t expected4 = side * side * 4;
    uint32_t expected8 = side * side * 8;  // fp16 × 2 banks
    return shared == expected4 || shared == expected8 ||
           shared == expected4 * 2;        // double buffer
}

// Shared mem consistent with a 1D reduction buffer
inline bool shared_matches_reduction(uint32_t shared, uint32_t bx) noexcept {
    if (shared == 0) return false;
    return shared == bx * 4 || shared == bx * 8 ||
           shared == bx * 4 * 2;           // warp reduction + partial sums
}

} // namespace detail

// ---------------------------------------------------------------------------
// Fingerprint engine — pure heuristics, no state, no alloc
// ---------------------------------------------------------------------------
inline OpClass fingerprint(const KernelGeom& g) noexcept {
    using namespace detail;

    const uint32_t tpb   = threads_per_block(g);
    const bool grid_1d   = (g.gy == 1 && g.gz == 1);
    const bool grid_2d   = (g.gz == 1 && g.gy > 1);
    const bool grid_3d   = (g.gz > 1);
    const bool block_1d  = (g.by == 1 && g.bz == 1);
    const bool block_2d  = (g.bz == 1 && g.by > 1);
    const bool tpb_valid = is_power_of_two(tpb) && tpb >= 32 && tpb <= 1024;

    // ------------------------------------------------------------------
    // ELEMENTWISE — flat 1D grid, 1D block, near-zero shared mem
    // cuBLAS axpy, fused activations, broadcast add, ReLU, GELU
    // ------------------------------------------------------------------
    if (grid_1d && block_1d && g.shared_bytes <= 128 && tpb >= 64) {
        return OpClass::ELEMENTWISE;
    }

    // ------------------------------------------------------------------
    // REDUCTION — grid collapsed (gx small), fat 1D block, shared ∝ bx
    // LayerNorm, softmax, mean, variance, global sum
    // ------------------------------------------------------------------
    if (block_1d && g.bx >= 128 && g.gy <= 16 && g.gz == 1 &&
        shared_matches_reduction(g.shared_bytes, g.bx)) {
        return OpClass::REDUCTION;
    }

    // ------------------------------------------------------------------
    // MEMCPY / TRANSPOSE — 2D grid, square 2D block, shared == tile × dtype
    // cuBLAS SetMatrix, cudaMemcpy2D internal, NCHW↔NHWC
    // ------------------------------------------------------------------
    if (grid_2d && block_2d && g.bx == g.by && g.bz == 1 &&
        is_power_of_two(g.bx) && g.bx >= 16 && g.bx <= 32 &&
        shared_matches_tile(g.shared_bytes, tpb) &&
        g.gx <= 4096 && g.gy <= 4096) {
        return OpClass::MEMCPY_TRANSPOSE;
    }

    // ------------------------------------------------------------------
    // CONVOLUTION — 3D grid (batch × spatial × channel-tiles),
    // 2D spatial block, moderate shared mem (input + filter tile).
    // Must resolve before GEMM: conv shared mem < 8KB (filter+input tile);
    // batched GEMM needs ≥8KB for MMA tiles to be compute-bound.
    // ------------------------------------------------------------------
    if (grid_3d && block_2d && g.gz >= 2 &&
        g.shared_bytes >= 512 && g.shared_bytes < 8192 &&
        g.bx <= 32 && g.by <= 32 && tpb_valid) {
        return OpClass::CONVOLUTION;
    }

    // ------------------------------------------------------------------
    // ATTENTION — large 2D grid (batch×heads × seq), 1D or 2D block,
    // large shared mem (flash-attention KV tiles)
    // Heuristic: gx >> gy (seq dimension dominates), shared >= 8KB
    // ------------------------------------------------------------------
    if (grid_2d && g.shared_bytes >= 8192 &&
        g.gx >= 64 && g.gx >= g.gy * 4 &&
        tpb >= 64 && tpb <= 256) {
        return OpClass::ATTENTION;
    }
    // Flash-attention v2: block 1D, grid batch*heads × num_tile_rows
    if (grid_2d && block_1d && g.bx == 128 &&
        g.shared_bytes >= 16384 && g.gy >= 4 && g.gx >= 8) {
        return OpClass::ATTENTION;
    }

    // ------------------------------------------------------------------
    // GEMM — 2D grid, 2D block (tile-based MMA), large shared mem.
    // Batched GEMM: gz > 1 allowed. Shared ≥ 8KB distinguishes from conv.
    // ------------------------------------------------------------------
    const bool gemm_block = block_2d && tpb_valid &&
                            (tpb == 256 || tpb == 512 || tpb == 1024) &&
                            g.bx >= 16 && g.by >= 8;
    const bool gemm_shared = g.shared_bytes >= 8192 && g.shared_bytes <= 49152;
    if ((grid_2d || (grid_3d && g.gz <= 32)) && gemm_block && gemm_shared) {
        return OpClass::GEMM;
    }
    // Low-occupancy GEMM: small N — shared can be 4KB if tpb == 256
    if (grid_2d && gemm_block && g.shared_bytes >= 4096 && tpb == 256) {
        return OpClass::GEMM;
    }

    // ------------------------------------------------------------------
    // ITERATIVE_CUSTOM — everything else
    // Physics solvers, custom CUDA, sparse ops, RNG kernels
    // ------------------------------------------------------------------
    return OpClass::ITERATIVE_CUSTOM;
}

// ---------------------------------------------------------------------------
// Cache — function-pointer keyed, open-addressing, lock-free reads
// 512 slots → ~16KB, fits in L1. Collision = re-fingerprint (rare).
// ---------------------------------------------------------------------------
namespace detail {

struct CacheSlot {
    std::atomic<uintptr_t> key{0};
    std::atomic<uint8_t>   op{0xFF};
};

inline CacheSlot& cache_slot(uintptr_t key) noexcept {
    static std::array<CacheSlot, 512> tbl;
    return tbl[key & 511u];
}

} // namespace detail

// ---------------------------------------------------------------------------
// classify() — public API, <100ns, thread-safe
// ---------------------------------------------------------------------------
inline ClassifyResult classify(const KernelGeom& g) noexcept {
    const uintptr_t key = reinterpret_cast<uintptr_t>(g.fn);
    auto& slot = detail::cache_slot(key);

    // Fast path — relaxed load, validate key
    if (slot.key.load(std::memory_order_relaxed) == key) {
        uint8_t cached = slot.op.load(std::memory_order_relaxed);
        if (cached != 0xFF) {
            return { static_cast<OpClass>(cached), 95, true };
        }
    }

    // Slow path — fingerprint
    const OpClass op  = fingerprint(g);
    const uint8_t opu = static_cast<uint8_t>(op);

    // Write-back (best-effort, no locking — last writer wins, harmless)
    slot.key.store(key, std::memory_order_relaxed);
    slot.op.store(opu, std::memory_order_relaxed);

    // Confidence heuristic: ITERATIVE_CUSTOM is lower certainty
    const uint8_t conf = (op == OpClass::ITERATIVE_CUSTOM) ? 40u : 85u;
    return { op, conf, false };
}

// ---------------------------------------------------------------------------
// Convenience — classify directly from cuLaunchKernel args
// void cuLaunchKernel(fn, gx,gy,gz, bx,by,bz, sharedMem, stream, params, extra)
// ---------------------------------------------------------------------------
inline ClassifyResult classify_launch(
    const void* fn,
    uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t shared_bytes) noexcept
{
    return classify({ gx, gy, gz, bx, by, bz, shared_bytes, fn });
}

} // namespace cipher
