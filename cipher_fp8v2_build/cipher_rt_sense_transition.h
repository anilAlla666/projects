/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_sense_transition.h -- Week 3 Step 4 Option II-a invented sub-step.
 *
 * Transition-detection wrapper above SENSE. SENSE classifies sessions
 * into HUMAN/AGENT/BATCH/UNKNOWN (cipher_sense_get_type). This wrapper
 * watches for class transitions with hysteresis + emits DSM PROPOSE
 * candidates to a bounded internal queue. The queue drains via ioctl
 * push to kmod on the existing 256-launch flush cadence.
 *
 * DESIGN-MISMATCH note: SENSE itself does NOT produce migration
 * proposals — Wave 5 W3 named a behavior that this wrapper invents.
 * v1 thresholds (N=8 stability, M=4 transition-debounce, K=100ms
 * agent-idle) are best-guesses; measurement-driven tuning is future
 * work.
 *
 * Hot-path safety: per-launch observe is lock-free (atomic counter
 * updates only). Queue push is a single atomic CAS on the ring head.
 * No malloc, no syscall, no blocking.
 *
 * Env: CIPHER_SENSE=1 enables the entire path (cipher_sense_init
 * gates SENSE itself; this wrapper inherits the same env). When OFF,
 * all wrapper calls are single-load atomic checks (~3 cycles).
 */
#ifndef CIPHER_RT_SENSE_TRANSITION_H
#define CIPHER_RT_SENSE_TRANSITION_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Reason codes for proposals (logged in payload + /proc emit). */
enum cipher_rt_sense_propose_reason {
	CIPHER_RT_SENSE_TRANSITION_DETECTED = 1, /* class shifted past debounce */
	CIPHER_RT_SENSE_AGENT_IDLE          = 2, /* agent quiet for K ms */
};

/* Per-launch observe entry from CUPTI bridge. Called only when
 * CIPHER_SENSE is enabled (caller's responsibility to gate). Updates
 * the per-thread session-class history and may push a proposal to
 * the internal queue. Lock-free. */
void cipher_rt_sense_transition_observe(
	uint32_t  tenant_id,    /* gettid() proxy in v1 */
	uint32_t  op_class,     /* OpClass enum value from classify */
	uint8_t   confidence,   /* 0..100 */
	uint64_t  fingerprint,  /* SENSE session fingerprint */
	uint64_t  timestamp_ns);

/* Drain the internal queue into kmod via CIPHER_DSM_PROPOSE ioctl.
 * Called on the 256-launch flush cadence (same as classify_stats
 * push). Returns number of proposals pushed, or -1 on error. */
int cipher_rt_sense_transition_flush(void);

/* Diagnostic accessors. */
unsigned long cipher_rt_sense_transition_proposals_queued(void);
unsigned long cipher_rt_sense_transition_proposals_pushed(void);

/* Week 4 Step 6 — Sub-4 measurement infrastructure.
 *
 * Total tool-idle events fired since process start (drives K=100ms threshold
 * tuning in Week 5+; today's value is exposed but not consumed). */
unsigned long cipher_rt_sense_transition_tool_idle_count(void);

/* Total stable-class transitions across all tenants. */
unsigned long cipher_rt_sense_transition_transitions_total(void);

/* Write the measurement snapshot (per-tenant log + per-class transition
 * matrix + aggregate counters) as JSON to `path` (default:
 * /tmp/cipher_sense_transitions.json when path is NULL). Operator-
 * triggered, not on the hot path. Returns 0 on success, -1 on fopen
 * failure. */
int cipher_rt_sense_transition_report(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_SENSE_TRANSITION_H */
