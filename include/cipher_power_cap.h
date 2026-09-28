// CIPHER Power Cap — Lever 1 of the energy plan.
//
// Drives nvmlDeviceSetPowerManagementLimit at decode start to move the GPU's
// operating point onto the (workload, batch)-specific tok/W maximum, with
// crash-safe restoration to the original cap on process exit / SIGTERM /
// SIGINT. Lookup table is populated from the Lever 1 sweep (see
// lever1_sweep_results.json + the reference table baked into the .cpp).
//
// Permission model: tries NVML in-process first; if NVML returns
// NO_PERMISSION, falls back to `sudo -n nvidia-smi -pl <W>` via subprocess.
// This pod is configured with passwordless sudo for nvidia-smi.
//
// Default OFF. Env: CIPHER_POWER_CAP=on.

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CipherPowerCapStats {
    int      enabled;
    int      original_cap_w;
    int      current_cap_w;
    int      last_applied_batch;
    int      last_applied_cap_w;
    uint64_t apply_calls;
    uint64_t restore_calls;
    uint64_t nvml_success;
    uint64_t subprocess_success;
    uint64_t failures;
} CipherPowerCapStats;

// Captures the original cap and registers atexit/SIGTERM/SIGINT restore.
// Idempotent. Returns 1 on success, 0 if cap could not be queried.
int  cipher_power_cap_init(void);

// Returns 1 if env CIPHER_POWER_CAP is on.
int  cipher_power_cap_enabled(void);

// Look up the optimal cap for a batch size (W). Returns -1 if no entry.
int  cipher_power_cap_get_optimal_w(int batch_size);

// Apply the optimal cap for the given batch. Returns 1 on success, 0 on
// failure or if disabled / no entry. Caller should pair with
// cipher_power_cap_restore() at decode end.
int  cipher_power_cap_apply_for_batch(int batch_size);

// Apply an explicit cap in watts. Returns 1 on success.
int  cipher_power_cap_apply_w(int watts);

// Restore the cap recorded at init.
int  cipher_power_cap_restore(void);

int  cipher_power_cap_stats(CipherPowerCapStats* out);
void cipher_power_cap_report(void);

#ifdef __cplusplus
}
#endif
