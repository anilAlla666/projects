// CIPHER — F4: Shared Liquid State
// cipher_liquid_state.h
//
// The coordination backbone of all three LNNs. A fixed-dimension state vector
// in device memory (accessible from all Green Contexts via peer access)
// that carries context across every kernel launch.
//
// FIELDS (7):
//   substitution_counter     — N≤4 rule enforcement (L3.6)
//   ema_gradient_norm        — divergence detection (L3.7)
//   hardware_trajectory      — SM/L2/HBM/power history (L2 orchestration)
//   training_phase           — warmup vs convergence detector (L3.9)
//   error_accumulation       — running Barron bound estimate (accuracy oracle)
//   nccl_congestion_history  — AllReduce timing for overlap (L2.6)
//   workload_rhythm          — kernel sequence pattern (L2.2 fusion)
//
// DESIGN:
//   - Fixed-size struct, no dynamic allocation
//   - Lives in device global memory (cudaMallocManaged for host debug access)
//   - Written via L2 atomics from any of the 3 Green Contexts (<200ns)
//   - Read by all three LNNs on every forward pass
//   - Indexed by (layer_id, per-layer-slot) — no collisions
//
// SUCCESS CRITERION: State updates visible to all 3 LNNs in <200ns.
// DEPENDENCY: F2 (Green Contexts), F3 (L2 persistence for state struct).
// =============================================================================

#pragma once
#ifdef CIPHER_CPU_STUB
#  include "cipher_stubs.h"
#else
#  include <cuda.h>
#  include <cuda_runtime.h>
#endif

#include <stdint.h>
#include <stdbool.h>
#include <atomic>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Liquid state dimensions
// ---------------------------------------------------------------------------

#define CIPHER_MAX_LAYERS          128   // Max transformer layers tracked
#define CIPHER_HW_TRAJ_DIM         32    // Hardware trajectory vector length
#define CIPHER_NCCL_HIST_LEN       16    // AllReduce timing history slots
#define CIPHER_WORKLOAD_HIST_LEN   64    // Kernel sequence ring buffer size
#define CIPHER_OP_CLASS_COUNT      7     // From cipher_classify.hpp

// ---------------------------------------------------------------------------
// Per-layer substitution state — one entry per transformer layer
// Enforces the N≤4 rule independently per layer
// ---------------------------------------------------------------------------

typedef struct {
    uint8_t  sub_counter;       // 0-4: consecutive substitutions
    uint8_t  force_passthrough; // 1 = forced to full precision this step
    uint8_t  perm_passthrough;  // 1 = EMA monitor permanently demoted
    uint8_t  _pad;
    float    last_grad_norm;    // Last observed gradient norm (EMA smoothed)
} CipherLayerState;

// ---------------------------------------------------------------------------
// Hardware trajectory — 32-dim vector fed to Layer 2 LNN
// Updated async by F5 telemetry pipeline every 2ms
// ---------------------------------------------------------------------------

typedef struct {
    // SM utilization (normalized 0-1)
    float    sm_occupancy;          // Fraction of SMs active
    float    sm_idle_fraction;      // Fraction idle (scheduling waste)

    // Memory bandwidth
    float    l2_hit_rate;           // L2 cache hit fraction
    float    hbm_bw_utilized;       // Fraction of peak HBM BW
    float    hbm_bw_gbps;           // Absolute GB/s

    // NVLink / NCCL
    float    nvlink_tx_gbps;
    float    nvlink_rx_gbps;
    float    nvlink_utilization;    // Fraction of peak NVLink BW

    // Power / thermal
    float    gpu_temp_c;
    float    power_watts;
    float    power_fraction;        // Fraction of TDP

    // Padding to CIPHER_HW_TRAJ_DIM floats
    float    _reserved[21];
} CipherHwTrajectory;

// Verify at compile time
static_assert(sizeof(CipherHwTrajectory) == CIPHER_HW_TRAJ_DIM * sizeof(float),
               "CipherHwTrajectory must be exactly CIPHER_HW_TRAJ_DIM floats");

// ---------------------------------------------------------------------------
// NCCL congestion history — circular buffer of AllReduce durations
// Used by Layer 2 to predict AllReduce completion time for overlap scheduling
// ---------------------------------------------------------------------------

typedef struct {
    uint64_t duration_ns[CIPHER_NCCL_HIST_LEN];   // Ring buffer
    uint64_t msg_size_bytes[CIPHER_NCCL_HIST_LEN]; // Message sizes
    uint8_t  write_head;     // Next write position
    uint8_t  count;          // Valid entries (0 to CIPHER_NCCL_HIST_LEN)
    uint16_t _pad;
    float    ema_duration_ns; // Exponential moving average
} CipherNcclHistory;

// ---------------------------------------------------------------------------
// Workload rhythm — ring buffer of recent op_class sequence
// Layer 2 LNN uses this to detect fusion opportunities (L2.2)
// ---------------------------------------------------------------------------

typedef struct {
    uint8_t  op_class[CIPHER_WORKLOAD_HIST_LEN];   // Ring buffer
    uint8_t  write_head;
    uint8_t  fill;      // Entries filled so far (caps at HIST_LEN)
    uint16_t _pad;
    // Frequency counters for fast lookup
    uint16_t op_freq[CIPHER_OP_CLASS_COUNT];
} CipherWorkloadRhythm;

// ---------------------------------------------------------------------------
// Error accumulation — Barron bound running estimate
// Tracks cumulative approximation error per layer for safety oracle
// ---------------------------------------------------------------------------

typedef struct {
    float   barron_bound[CIPHER_MAX_LAYERS];  // Per-layer error bound
    float   total_accumulated;                 // Sum across active layers
    float   max_per_layer;                     // Worst-case single layer
    uint32_t violation_count;                  // Times N>4 rule fired
} CipherErrorAccum;

// ---------------------------------------------------------------------------
// THE LIQUID STATE STRUCT
// Fixed layout, no pointers (safe for device memory)
// Total size: ~8KB — fits in shared memory on a single SM
// ---------------------------------------------------------------------------

typedef struct {
    // --- Field 1: substitution_counter (per layer) ---
    CipherLayerState  layer[CIPHER_MAX_LAYERS];   // 512 bytes

    // --- Field 2: ema_gradient_norm (global + per layer) ---
    float    global_grad_ema;         // κ=0.999 EMA of ||∇L||
    float    grad_ema_baseline;       // Steady-state baseline
    float    grad_sigma;              // Running std dev for 2σ threshold
    uint32_t grad_step_count;         // Steps since last reset

    // --- Field 3: hardware_trajectory ---
    CipherHwTrajectory  hw;           // 128 bytes

    // --- Field 4: training_phase ---
    uint8_t  phase;                   // 0=warmup, 1=convergence, 2=finetune
    uint8_t  phase_step;              // Steps in current phase (mod 255)
    uint16_t warmup_steps_remaining;  // Phase detector countdown
    uint32_t total_steps;             // Total training steps seen

    // --- Field 5: error_accumulation ---
    CipherErrorAccum  error;          // ~516 bytes

    // --- Field 6: nccl_congestion_history ---
    CipherNcclHistory  nccl;          // ~152 bytes

    // --- Field 7: workload_rhythm ---
    CipherWorkloadRhythm  rhythm;     // ~80 bytes

    // --- Metadata ---
    uint64_t last_update_ns;          // Timestamp of last state write
    uint32_t update_count;            // Total state updates
    uint32_t _magic;                  // 0xC1PHERS for validation

} CipherLiquidState;

// Magic value for state validation
#define CIPHER_LIQUID_STATE_MAGIC  0xC1F4E350

// ---------------------------------------------------------------------------
// State manager — owns device allocation and host mirror
// ---------------------------------------------------------------------------

typedef struct {
    CipherLiquidState*  device;      // cudaMallocManaged — device + host
    size_t              size_bytes;
    bool                initialized;
} CipherLiquidStateMgr;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Allocate and zero-initialize the liquid state on device.
cudaError_t cipher_liquid_state_init(CipherLiquidStateMgr* mgr);

// Destroy device allocation.
void cipher_liquid_state_destroy(CipherLiquidStateMgr* mgr);

// Record a substitution event for layer `layer_idx`.
// Updates counter, checks N≤4 rule, sets force_passthrough if needed.
// Returns true if substitution is PERMITTED, false if N≤4 rule fires.
bool cipher_liquid_record_substitution(CipherLiquidStateMgr* mgr,
                                       int layer_idx);

// Record a passthrough (resets the substitution counter for that layer).
void cipher_liquid_record_passthrough(CipherLiquidStateMgr* mgr,
                                      int layer_idx);

// Update gradient EMA — call from training loop hook every 100 steps.
void cipher_liquid_update_grad_ema(CipherLiquidStateMgr* mgr,
                                   float batch_grad_norm);

// Update hardware trajectory from telemetry pipeline (F5).
void cipher_liquid_update_hw(CipherLiquidStateMgr* mgr,
                             const CipherHwTrajectory* hw);

// Record an NCCL AllReduce completion.
void cipher_liquid_record_nccl(CipherLiquidStateMgr* mgr,
                               uint64_t duration_ns,
                               uint64_t msg_size_bytes);

// Record a kernel op_class in the workload rhythm ring buffer.
void cipher_liquid_record_op(CipherLiquidStateMgr* mgr, uint8_t op_class);

// Advance training phase (warmup → convergence → finetune).
void cipher_liquid_advance_phase(CipherLiquidStateMgr* mgr);

// Report state summary to stderr.
void cipher_liquid_state_report(const CipherLiquidStateMgr* mgr);

#ifdef __cplusplus
}
#endif
