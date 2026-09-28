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
	CIPHER_RT_VOLT_ARMED    = 3,  /* W.1: NVML resolved + handlers installed;
	                                 lock NOT applied — awaiting classifier. */
} CipherRtVoltMode;

/* One-shot init from cipher_inject.c. Reads env vars, resolves NVML,
 * probes actuation, applies lock if configured. Idempotent.
 * Returns the resulting mode. */
int cipher_rt_volt_init(void);

/* W.1 (2026-05-27): classifier-driven engagement entry point. The K.1
 * classifier's background timer thread calls this periodically. If mode
 * is ARMED and the published CipherProfile says volt_engage=1, applies
 * the clock lock (transitions to ACTIVE) and bumps the engagement counter.
 * If profile says volt_engage=0, bumps the skipped-by-classifier counter
 * and stays ARMED. No-op when mode is OFF / ACTIVE / DEGRADED.
 * Idempotent + lock-free + thread-safe.  */
void cipher_rt_volt_classifier_poll(void);

/* W.1 engagement counters (read-only accessors).  */
unsigned long cipher_rt_volt_classifier_engagements(void);
unsigned long cipher_rt_volt_classifier_skipped(void);

/* Diagnostic accessors. */
int          cipher_rt_volt_mode(void);             /* CipherRtVoltMode */
unsigned int cipher_rt_volt_locked_mhz(void);       /* 0 if not locked */
const char  *cipher_rt_volt_status_string(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_VOLT_H */
