/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_audit.h - Op 9 AUDIT, ported onto the Phase 4 substrate.
 *
 * AUDIT keeps a tamper-evident HMAC-SHA256 hash chain over every
 * operation the substrate dispatches. Each event links into the chain:
 * altering, reordering, or dropping any past event changes the chain
 * head, so the head is a compact proof of the dispatch history.
 *
 * Ported from op31-prod cipher_10ops_impl.cpp (audit_chain_update +
 * AUDIT_KEY). op31-prod ran AUDIT in a Stage-1 shadow thread; on the
 * Phase 4 substrate HMAC-SHA256 is ~50 ns/call (OpenSSL auto-selects
 * SHA-NI), within the R6 Stage-0 budget, so AUDIT records inline under
 * a short mutex — no shadow-thread / ring-consumer machinery needed.
 *
 * AUDIT is a passive observer: it registers priority-0 actuators on the
 * matmul + attention substrate registries that record the call and
 * always return PASSTHROUGH. cipher_rt_audit_record() is also exported
 * so future actuators (Op 3 SUBSTITUTE) can chain a substitution event.
 */
#ifndef CIPHER_RT_AUDIT_H
#define CIPHER_RT_AUDIT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum cipher_rt_audit_op_class {
	CIPHER_RT_AUDIT_OP_MATMUL     = 0,
	CIPHER_RT_AUDIT_OP_ATTENTION  = 1,
	CIPHER_RT_AUDIT_OP_SUBSTITUTE = 2,  /* reserved for Op 3 */
};

enum cipher_rt_audit_decision {
	CIPHER_RT_AUDIT_OBSERVED    = 0,  /* dispatch call seen at substrate entry */
	CIPHER_RT_AUDIT_SUBSTITUTED = 1,  /* an actuator substituted (Op 3, future) */
};

/* One audit event. Stable 40-byte layout (verifier depends on it). */
struct cipher_rt_audit_entry {
	uint64_t sequence;
	uint64_t timestamp_ns;
	uint64_t timestamp_delta;
	uint64_t call_hash;     /* FNV-1a of the call descriptor (NOT output
	                         * bytes — hot-path cheap, no GPU read) */
	uint32_t op_class;
	uint8_t  decision;
	uint8_t  _pad[3];
};

/* Init: env-gated by CIPHER_AUDIT. Registers priority-0 observer
 * actuators on the matmul + attention substrate registries. Idempotent.
 * Returns 0 on success. */
int cipher_rt_audit_init(void);

/* Append an event to the tamper-evident chain (mutex-guarded, ~50 ns).
 * Exported so other actuators can chain their own events. */
void cipher_rt_audit_record(uint32_t op_class, uint64_t call_hash,
                            uint8_t decision);

/* Copy the current 32-byte HMAC chain head. */
void cipher_rt_audit_chain_head(uint8_t out[32]);

/* Total events recorded since init. */
uint64_t cipher_rt_audit_count(void);

/* Copy the most recent min(count, ring_capacity, max) entries in
 * sequence order into out[]. Returns the number copied. */
size_t cipher_rt_audit_dump(struct cipher_rt_audit_entry *out, size_t max);

/* 1 if AUDIT is armed (CIPHER_AUDIT set and init ran). */
int cipher_rt_audit_enabled(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_AUDIT_H */
