// =============================================================================
// CIPHER — Master Header
// cipher.h
//
// Single include for all CIPHER components. Initialization order:
//   1. cipher_init()         — calls all F1-F5 inits in dependency order
//   2. cipher_dispatch()     — called from F1 shim (already wired)
//   3. cipher_teardown()     — ordered shutdown
//
// Component dependency graph:
//   F1 (Hook) → [everything]
//   F2 (Green Ctx) → F3, F4
//   F3 (L2 Persist) → F4 (weights must be loaded before state)
//   F4 (Liquid State) → F5 (telemetry writes into state)
//   F5 (Telemetry) — reads hardware, writes to F4
//
// Usage:
//   LD_PRELOAD=libcipher.so python train.py
//   CIPHER=1 python train.py  (with wrapper script)
// =============================================================================

#pragma once

#include "may13/cipher_intercept.h"
#include "may13/cipher_green_ctx.h"
#include "may13/cipher_l2_persist.h"
#include "may13/cipher_liquid_state.h"
#include "may13/cipher_telemetry.h"
// Layer 3
#include "may13/cipher_classify.hpp"
#include "may13/cipher_structural_lookup.h"
#include "may13/cipher_oracle.h"
#include "may13/cipher_recipes.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CIPHER_VERSION_MAJOR  0
#define CIPHER_VERSION_MINOR  1
#define CIPHER_VERSION_PATCH  0
#define CIPHER_VERSION_STR    "0.1.0-phase0"

// ---------------------------------------------------------------------------
// Global CIPHER runtime state — one instance per process
// ---------------------------------------------------------------------------

typedef struct {
    CipherGreenCtxState     green_ctx;
    CipherL2PersistState    l2_persist;
    CipherLiquidStateMgr    liquid;
    CipherTelemetryState    telemetry;
    bool                    initialized;
    int                     device_ordinal;
} CipherRuntime;

// Global singleton — accessible from F1 dispatch path
extern CipherRuntime g_cipher;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Full initialization: F2 → F3 → F4 → F5 (F1 self-initializes via .so ctor)
// device_ordinal: which GPU to run CIPHER on (0 for single-GPU)
int cipher_init(int device_ordinal);

// Layer 3 initialization (called internally from cipher_init)
int cipher_layer3_init(void);

// Ordered teardown
void cipher_teardown(void);

// Called by F1 shim — routes kernel to appropriate handler
CipherDispatchResult cipher_dispatch(CipherKernelDesc* desc);

// Training loop hook — call every 100 steps with per-layer gradient norms
void cipher_training_step_hook(const float* grad_norms,
                               uint32_t     num_layers,
                               uint32_t     global_step);

// Accessor for Layer 3 subsystem state (for testing)
CipherOracleState* cipher_get_oracle(void);
CipherRegistry*    cipher_get_registry(void);

// Print full system report
void cipher_report(void);
void cipher_layer3_report(void);

#ifdef __cplusplus
}
#endif
