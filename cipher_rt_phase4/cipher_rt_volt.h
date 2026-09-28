/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_volt.h -- T4.3.1 VOLT NVML clock-lock actuator.
 *
 * v1 scope: env-driven static clock lock via nvmlDeviceSetGpuLockedClocks
 * applied at libcipher_rt init, restored at exit (atexit + signal handlers).
 * No per-launch classifier in v1; that's T4.3.2 once cuBLAS interception
 * or kernel-name-based shape inference is in place.
 *
 * Env vars:
 *   CIPHER_VOLT=off      (default) — no actuation
 *   CIPHER_VOLT=on       — apply CIPHER_VOLT_BATCH or CIPHER_VOLT_MHZ
 *   CIPHER_VOLT_BATCH=N  — pick clock from calibration table: 1→1000, 8→1600, 32/64→1980
 *   CIPHER_VOLT_MHZ=N    — explicit clock target (overrides BATCH if both set)
 *
 * Restoration: NVML clock lock PERSISTS across process exit. atexit +
 * SIGTERM/SIGINT/SIGSEGV/SIGABRT/SIGBUS handlers call cuCtx-free
 * nvmlDeviceResetGpuLockedClocks via cached function pointer (dlsym'd at
 * init, never inside the signal handler).
 */
#ifndef CIPHER_RT_VOLT_H
#define CIPHER_RT_VOLT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	CIPHER_RT_VOLT_OFF      = 0,
	CIPHER_RT_VOLT_ACTIVE   = 1,
	CIPHER_RT_VOLT_DEGRADED = 2,  /* probed but actuation unsupported */
} CipherRtVoltMode;

/* One-shot init from cipher_inject.c. Reads env vars, resolves NVML,
 * probes actuation, applies lock if configured. Idempotent.
 * Returns the resulting mode. */
int cipher_rt_volt_init(void);

/* Diagnostic accessors. */
int          cipher_rt_volt_mode(void);             /* CipherRtVoltMode */
unsigned int cipher_rt_volt_locked_mhz(void);       /* 0 if not locked */
const char  *cipher_rt_volt_status_string(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_VOLT_H */
