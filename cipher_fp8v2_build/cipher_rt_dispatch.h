/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_dispatch.h -- Week 3 Step 1: GEMM dispatch routing table.
 *
 * Scaffolds the §4.5 dispatch table that maps ClassifyResult ->
 * dispatch decision (which actuator, which hint). GEMM-only in v1
 * per Wave 5 §5.5 W3 scope-correction (L721-726); attn lane defers
 * to v1.5 when the first attn substitute actuator (Op-3 SUBSTITUTE
 * for FAVOR+ / FlashSwiftKey) lands. The LP-2 refactor at Week 2
 * Step 1 made the attn defer safe.
 *
 * Env: CIPHER_DISPATCH_LIVE controls whether dispatch_lookup()'s
 * output is consumed by actuators downstream:
 *   - 0 (default through Steps 1-3): substrate exists; no actuator
 *     reads the decision; runtime behavior unchanged from week-2-complete.
 *   - 1 (Step 4 flip): classifier-driven routing becomes live;
 *     Marlin / cuBLAS shim consult the decision.
 *
 * Lookup is pure (no global state besides the static table); is_live
 * caches the env-read after first call. Both are hot-path safe.
 *
 * Step 1 deliverable: substrate scaffolding only. Lookup is callable
 * but NOT called from any hot path. Step 2 extends the classify
 * observer to publish TLS substitute_hint; Step 3 makes Marlin
 * consume the hint; Step 4 flips CIPHER_DISPATCH_LIVE=1.
 */
#ifndef CIPHER_RT_DISPATCH_H
#define CIPHER_RT_DISPATCH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Action tag — which actuator should handle this op. Append-only. */
enum cipher_rt_dispatch_action {
	CIPHER_RT_DISPATCH_PASS_THROUGH = 0, /* no actuator; vanilla path */
	CIPHER_RT_DISPATCH_ROUTE_MARLIN = 1, /* Marlin INT4 GEMM (Ca.9) */
	CIPHER_RT_DISPATCH_ROUTE_CUBLAS = 2, /* cuBLAS shim (CP 2.5) */
	CIPHER_RT_DISPATCH_ROUTE_ATTN   = 3, /* reserved; v1.5 attn substitute */
};

/* Decision — populated by lookup(). 8 bytes; reserved field for
 * Cb.2-style future extension. */
struct cipher_rt_dispatch_decision {
	uint8_t  op_class;       /* mirrors cipher::OpClass (0..6, 0xFF=UNCLASSIFIED) */
	uint8_t  action;         /* enum cipher_rt_dispatch_action */
	uint16_t actuator_hint;  /* opaque hint slot; Step 3 consumer */
	uint32_t reserved;
};

/* Lookup — table read from op_class. Returns 0 on success and
 * populates *out; returns -1 if op_class out of range or out is NULL.
 * Pure function (no I/O, no locks, no allocation; hot-path safe). */
int cipher_rt_dispatch_lookup(
	uint8_t                              op_class,
	struct cipher_rt_dispatch_decision  *out);

/* Live check — returns 1 if CIPHER_DISPATCH_LIVE env is set to "1",
 * else 0. Default 0. Env is parsed once at first call and cached
 * in an atomic; subsequent calls are lock-free atomic loads.
 * Step 4 is where this becomes the load-bearing toggle. */
int cipher_rt_dispatch_is_live(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_DISPATCH_H */
