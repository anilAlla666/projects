// =============================================================================
// CIPHER — L1.4: Hardware Descriptor Normalization
// cipher_hw_desc.h
//
// Normalizes raw hardware metrics into a dimensionless descriptor vector.
// Enables LNN weight transfer across GPU generations without retraining.
//
// PROBLEM:
//   The LNN was trained (or analytically initialized) on H100 features:
//     sm_idle_fraction ∈ [0,1] — already dimensionless, no problem
//     hbm_bw_utilized ∈ [0,1]  — already dimensionless, no problem
//     BUT: the LNN also receives hardware *capacity* features for decisions
//     like "how many SMs are available for packing?" or "how tight is L2?"
//   These are hardware-specific. 132 SMs on H100 ≠ 108 SMs on A100.
//   A model trained on H100 raw counts breaks on A100.
//
// SOLUTION:
//   Normalize all capacity features relative to a reference hardware profile
//   (H100 SXM5). This gives dimensionless ratios:
//     sm_ratio         = active_SMs / reference_SMs         (H100: 132)
//     l2_ratio         = L2_size_MB / reference_L2_MB       (H100: 50MB)
//     bw_ratio         = peak_BW_GBps / reference_BW_GBps   (H100: 3350)
//     clock_ratio      = SM_clock_MHz / reference_clock_MHz (H100: 1980)
//     compute_ratio    = theoretical_TFLOPS / reference_TFLOPS (H100: 989)
//     nvlink_ratio     = NVLink_BW / reference_NVLink_BW    (H100: 900 GB/s)
//
//   On H100: all ratios = 1.0
//   On A100: sm_ratio=108/132=0.818, l2_ratio=40/50=0.8, bw_ratio=2000/3350=0.597
//   On H200: sm_ratio=132/132=1.0,   l2_ratio=96/50=1.92, bw_ratio=4800/3350=1.433
//
// CONVERGENCE GUARANTEE (build plan):
//   <10% accuracy degradation on unseen hardware generation.
//   Tested: same LNN weights, H100 vs A100 vs H200 inputs → recipes
//   differ by <10% in predicted efficiency.
//
// USAGE:
//   1. At startup, call cipher_hw_desc_detect() to populate CipherHwDesc
//   2. Call cipher_hw_desc_normalize() to get CipherHwDescNorm
//   3. Inject norm vector into LNN input (replaces raw hw features)
//
// DEPENDENCIES: F5 (telemetry for runtime hw queries), L1.3 (registry).
// =============================================================================

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Reference hardware profile — H100 SXM5 (baseline = 1.0)
// ---------------------------------------------------------------------------

#define CIPHER_REF_SM_COUNT         132
#define CIPHER_REF_L2_SIZE_MB        50.0f
#define CIPHER_REF_HBM_BW_GBPS    3350.0f
#define CIPHER_REF_CLOCK_MHZ       1980.0f
#define CIPHER_REF_TFLOPS_BF16      989.0f   // BF16 tensor core peak
#define CIPHER_REF_NVLINK_BW_GBPS   900.0f   // NVLink 4 bidirectional
#define CIPHER_REF_L2_BANKS          128      // L2 cache banks
#define CIPHER_REF_SHARED_MEM_KB      64      // Max shared mem per SM

// Known GPU architecture codes
typedef enum {
    CIPHER_ARCH_UNKNOWN  = 0,
    CIPHER_ARCH_AMPERE   = 80,   // A100 (sm_80)
    CIPHER_ARCH_ADA      = 89,   // RTX 4090 (sm_89)
    CIPHER_ARCH_HOPPER   = 90,   // H100 (sm_90)
    CIPHER_ARCH_BLACKWELL= 100,  // B100/B200 (sm_100)
    CIPHER_ARCH_CDNA3    = 942,  // AMD MI300X
} CipherArchCode;

// ---------------------------------------------------------------------------
// Raw hardware descriptor — populated by detect()
// ---------------------------------------------------------------------------

typedef struct {
    // Identity
    char           device_name[64];
    CipherArchCode arch;
    int            device_index;

    // Compute
    uint32_t  sm_count;           // Total SMs on device
    uint32_t  cuda_cores_per_sm;  // CUDA cores per SM (128 on H100)
    uint32_t  clock_mhz;          // SM clock in MHz

    // Memory
    float     l2_size_mb;         // L2 cache size in MB
    float     hbm_bw_gbps;        // Peak HBM bandwidth GB/s
    uint64_t  total_vram_mb;      // Total VRAM in MB
    uint32_t  shared_mem_per_sm_kb; // Max shared memory per SM in KB

    // Interconnect
    float     nvlink_bw_gbps;     // NVLink bandwidth GB/s (0 if none)
    uint32_t  nvlink_version;     // NVLink version (3, 4, 5...)
    uint32_t  num_gpus;           // GPUs in the node (for NCCL topology)

    // Derived
    float     peak_tflops_bf16;   // Peak BF16 tensor TFLOPS
    float     ridge_point;        // Arithmetic intensity where compute=BW bound
                                   // = peak_TFLOPS / peak_BW (FLOP/Byte)

    bool      detected;           // True if successfully populated
} CipherHwDesc;

// ---------------------------------------------------------------------------
// Normalized descriptor — all values dimensionless, H100=1.0
// ---------------------------------------------------------------------------

#define CIPHER_HW_NORM_DIM  16   // Dimensionality of normalized vector

typedef struct {
    // Ratios (all relative to H100 SXM5 reference)
    float sm_ratio;           // [0] sm_count / 132
    float l2_ratio;           // [1] l2_size_mb / 50
    float bw_ratio;           // [2] hbm_bw_gbps / 3350
    float clock_ratio;        // [3] clock_mhz / 1980
    float compute_ratio;      // [4] peak_tflops_bf16 / 989
    float nvlink_ratio;       // [5] nvlink_bw_gbps / 900 (0 if no NVLink)
    float vram_ratio;         // [6] total_vram_mb / 80000 (H100 80GB)
    float shmem_ratio;        // [7] shared_mem_per_sm_kb / 64

    // Architecture one-hot (4 dims)
    float is_ampere;          // [8]  1.0 if A100
    float is_hopper;          // [9]  1.0 if H100
    float is_blackwell;       // [10] 1.0 if B100/B200
    float is_amd;             // [11] 1.0 if AMD MI300X

    // Topology features
    float multi_gpu;          // [12] min(num_gpus/8, 1.0) — 8-GPU node = 1.0
    float has_nvlink;         // [13] 1.0 if NVLink present
    float memory_pressure;    // [14] l2_ratio / bw_ratio — higher = more L2 needed
    float compute_intensity;  // [15] compute_ratio / bw_ratio — higher = more compute-bound

    // Raw vector form for LNN injection
    float v[CIPHER_HW_NORM_DIM];  // Same 16 values in order above
} CipherHwDescNorm;

// ---------------------------------------------------------------------------
// Known hardware profiles — for CPU-stub and offline use
// ---------------------------------------------------------------------------

typedef struct {
    const char*    name;
    CipherArchCode arch;
    uint32_t       sm_count;
    float          l2_mb;
    float          bw_gbps;
    uint32_t       clock_mhz;
    float          tflops_bf16;
    float          nvlink_gbps;
    uint64_t       vram_mb;
    uint32_t       shmem_per_sm_kb;
} CipherHwProfile;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Detect hardware from CUDA device query.
// In stub mode: falls back to H100 profile.
void cipher_hw_desc_detect(CipherHwDesc* desc, int device_index);

// Look up hardware profile by name (for offline/stub use).
// Returns false if name not found.
bool cipher_hw_desc_from_name(CipherHwDesc* desc, const char* name);

// Normalize a raw descriptor to dimensionless ratios.
// Always succeeds — uses H100 defaults for any undetected fields.
CipherHwDescNorm cipher_hw_desc_normalize(const CipherHwDesc* desc);

// Normalize directly from a known profile name.
// Convenience wrapper for cipher_hw_desc_from_name + normalize.
CipherHwDescNorm cipher_hw_desc_normalize_by_name(const char* name);

// Compute transfer accuracy estimate between two hardware profiles.
// Returns expected accuracy retention [0,1] when using weights from
// src_norm on hardware described by dst_norm.
// Build plan target: >0.90 (i.e. <10% degradation).
float cipher_hw_desc_transfer_accuracy(const CipherHwDescNorm* src,
                                        const CipherHwDescNorm* dst);

// Print a summary of the descriptor and its normalized form.
void cipher_hw_desc_report(const CipherHwDesc* desc,
                            const CipherHwDescNorm* norm);

#ifdef __cplusplus
}
#endif
