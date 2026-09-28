/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_remember_consumer.h -- W14 Step 3 S3.B1 REMEMBER consumer.
 *
 * Drains RING_WRITE slot 3 (CIPHER_RT_RING_EVENT_REMEMBER) into the CfC
 * LNN forward path at cipher_lnn.cpp:401-460. Each drained event triggers
 * one cipher_lnn_decide() call, which runs cipher_lnn_forward() to update
 * the persistent hidden state h.
 *
 * v1 scope: wire-up + correctness only. The CfC weight updates and the
 * actuator-side use of the decision are v2 work (scope-lock §3.2). v1
 * answers the question "does the producer-consumer pipeline run end-to-end
 * without breaking the substrate?" and exposes accessors for measurement.
 *
 * Env gating: CIPHER_REMEMBER=1 to enable the consumer thread; default 0
 * keeps the consumer dormant (drain-and-discard no-op when called). Pattern
 * mirrors CIPHER_KOOPMAN=1 at cipher_rt_koopman_engine.cpp:148.
 *
 * Batched drain (256 entries per CfC forward — R-W14.3 mitigation per
 * scope-lock line 147) amortizes the per-call CfC cost against the higher
 * producer rate.
 */
#ifndef CIPHER_RT_REMEMBER_CONSUMER_H
#define CIPHER_RT_REMEMBER_CONSUMER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Init the consumer. Env-gated CIPHER_REMEMBER=1. Spawns one pthread that
 * iterates all CIPHER_RT_RING_MAX_TENANTS tenants, draining slot 2
 * (CIPHER_RT_RING_CONSUMER_REMEMBER), and runs cipher_lnn_decide() per
 * entry. Idempotent: re-call is a no-op (CAS-protected). Returns 0 on
 * success or when env-gated off, -1 on pthread spawn failure. */
int  cipher_rt_remember_consumer_init(void);

/* Stop the consumer thread. Idempotent. Sets the stop flag and joins. */
void cipher_rt_remember_consumer_exit(void);

/* True if the consumer thread is running. */
int  cipher_rt_remember_consumer_is_active(void);

/* Telemetry: total ring entries drained (across all tenants and cycles). */
uint64_t cipher_rt_remember_consumer_drained(void);

/* Telemetry: total CfC forward passes invoked (= entries drained). */
uint64_t cipher_rt_remember_consumer_lnn_invocations(void);

/* Telemetry: cycles where at least one tenant had work. */
uint64_t cipher_rt_remember_consumer_active_cycles(void);

/* Telemetry: total cycles run (active + idle). */
uint64_t cipher_rt_remember_consumer_total_cycles(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_REMEMBER_CONSUMER_H */
