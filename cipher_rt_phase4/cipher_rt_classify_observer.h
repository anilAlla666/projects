/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_classify_observer.h -- Week 2 Step 4 stub.
 *
 * Observer-side consumer interface for cipher_rt_classify_route(). Step 4
 * ships the data structure + 3 API functions; Step 6 wires the
 * observe() call into cipher_rt_classify_substrate.cpp's route loop and
 * the Step 5 /proc/cipher/classify_stats node's userspace-to-kmod bridge.
 *
 * Hot-path safety: observe() uses C11 atomics only, no locks, no
 * allocation. Snapshot returns a stable pointer to the static stats
 * struct (atomic-readable by /proc emitter and any future consumer).
 */
#ifndef CIPHER_RT_CLASSIFY_OBSERVER_H
#define CIPHER_RT_CLASSIFY_OBSERVER_H

#include <stdint.h>
#include "cipher_rt_classify_substrate.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Per-class counters snapshot. Consumed by /proc/cipher/classify_stats
 * (Step 5) and any future telemetry surface. */
struct cipher_rt_classify_observer_stats {
	uint64_t total_classifications;
	uint64_t handled;
	uint64_t passthrough;
	uint64_t per_op_class_counts[16];  /* indexed by OpClass enum */
	uint64_t reserved[8];              /* Cb.2 future expansion */
};

/* Init hook — called at libcipher_rt init time (Step 6 wires this from
 * cipher_rt_classify_dispatch_init or equivalent). Step 4 stub: returns
 * 0; counters are zero-init via static storage. Idempotent. */
int cipher_rt_classify_observer_init(void);

/* Observe hook — called by cipher_rt_classify_route(call, out) after
 * the route loop returns. Step 4 ships the atomic-counter logic; Step
 * 6 wires the call site. Hot-path safe (no locks, no allocation). */
void cipher_rt_classify_observer_observe(
	const struct cipher_rt_classify_call *call,
	const struct cipher_rt_classify_out  *out,
	int                                   result);

/* Snapshot getter. Returns pointer to the static stats struct;
 * readers should treat the uint64_t fields as atomic loads (the
 * struct's lifetime is the .so lifetime). Step 5 /proc/cipher/
 * classify_stats reads through this entry point. */
const struct cipher_rt_classify_observer_stats *
cipher_rt_classify_observer_snapshot(void);

/* Week 2 Step 6 — push observer snapshot to kmod via ioctl nr=25
 * (CIPHER_PUSH_CLASSIFY_STATS). SET semantics. Piggybacked on the
 * existing 256-launch flush in cipher_cupti.c. Returns 0 on success,
 * -1 on failure (logged, non-fatal — does NOT propagate up the
 * launch path). */
int cipher_rt_classify_push_to_kmod(void);

/* Week 3 Step 2 — per-thread substitute hint published by observer
 * when oracle PERMIT + op_class==GEMM + dispatch_lookup() returns a
 * non-PASS_THROUGH action. Marlin actuator reads this in Step 3;
 * the hint is acted upon when CIPHER_DISPATCH_LIVE=1 (Step 4).
 *
 * Wave 5 W3-1 SYNTHESIS-HYPOTHESIS resolution: Wave 5 named the
 * publishing function `maybe_handle` (the actuator pattern); Step 4
 * of Week 2 shipped `observe()` instead. Wave 5's prescription is
 * folded into the existing `observe()` per scope-lock §2 Step 2. */
struct cipher_rt_substitute_hint {
	uint8_t  action;        /* enum cipher_rt_dispatch_action */
	uint16_t actuator_hint; /* opaque hint slot from dispatch table */
	uint8_t  valid;         /* 1 = hint applies this call; 0 = no substitute */
	uint32_t reserved;
};

/* Per-thread getter. Reads __thread storage; no locks. Step 3
 * Marlin actuator's maybe_handle_marlin will call this on entry. */
const struct cipher_rt_substitute_hint *
cipher_rt_classify_observer_get_tls_hint(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_CLASSIFY_OBSERVER_H */
