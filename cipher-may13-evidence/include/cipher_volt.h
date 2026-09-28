// Op 30 VOLT — Arithmetic-intensity-aware SM frequency steering.
//
// v1 actuation requires `nvmlDeviceSetGpuLockedClocks` (or the equivalent
// `nvmlDeviceSetGpcClkVfOffset`) write permission. On containerized cloud
// GPU pods this typically returns NVML_ERROR_NOT_SUPPORTED (rc=4); the op
// gracefully degrades to classifier-only mode in that case. atexit + signal
// handlers (SIGTERM/SIGINT/SIGSEGV/SIGABRT/SIGBUS) are still registered
// defensively whenever any frequency offset is applied — frequency offsets
// PERSIST ACROSS PROCESS EXIT in the NVML driver, so a crash without
// restoration would leave the GPU stuck in a reduced-frequency state.
//
// All NVML function pointers used by the signal handler are dlsym'd and
// cached at init time. NO dlsym calls inside any signal handler.
//
// Env vars (separate code paths):
//   CIPHER_VOLT=calibrate → run probe + per-shape calibration, write
//                            /tmp/cipher_volt_calibration.json, exit calibration.
//                            On unsupported pods writes a degraded JSON with
//                            actuation_supported=false and exits cleanly.
//   CIPHER_VOLT=on        → load calibration JSON; refuses if missing.
//                            Active steering when actuation_supported=true,
//                            classifier-only DEGRADED mode otherwise.
//   anything else / unset → off (single-cycle exit).
#pragma once
#include <stdint.h>
#include "cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CIPHER_VOLT_OFF       = 0,
    CIPHER_VOLT_CALIBRATE = 1,
    CIPHER_VOLT_ACTIVE    = 2,
    CIPHER_VOLT_DEGRADED  = 3,   // requested ACTIVE but actuation unsupported
} CipherVoltMode;

typedef enum {
    CIPHER_SHAPE_UNKNOWN           = 0,
    CIPHER_SHAPE_MEMORY_BOUND_SAFE = 1,   // AI < 10 → safe to reduce frequency
    CIPHER_SHAPE_FREQ_SENSITIVE    = 2,   // AI > 100 → never reduce
    CIPHER_SHAPE_AMBIGUOUS         = 3,   // 10 ≤ AI ≤ 100 → conservative: don't reduce
} CipherShapeClass;

int               cipher_volt_init(void);
void              cipher_volt_observe(const CipherRingEntry* ev);   // Stage 1
void              cipher_volt_actuate(void);                         // Stage 2
CipherVoltMode    cipher_volt_mode(void);
int               cipher_volt_actuation_supported(void);

CipherShapeClass  cipher_volt_classify(uint32_t M, uint32_t K, uint32_t N);
double            cipher_volt_arithmetic_intensity(uint32_t M, uint32_t K, uint32_t N);

unsigned          cipher_volt_reduce_event_count(void);
unsigned          cipher_volt_restore_event_count(void);
const char*       cipher_volt_status_string(void);

void              cipher_volt_force_actuate(void);                    // tests bypass cadence
void              cipher_volt_inject_supported(int forced_value);     // tests only

void              cipher_volt_report(void);   // /tmp/cipher_volt_report.json

#ifdef __cplusplus
}
#endif
