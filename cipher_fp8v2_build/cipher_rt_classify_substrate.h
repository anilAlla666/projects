/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_classify_substrate.h -- Week 2 Step 3 classify-routing substrate.
 *
 * Third substrate in the cipher_rt_phase4 actuator-registry family
 * (matmul, attn, classify). Priority-ordered registry of operation
 * classifiers; future actuators (CUPTI-driven classifier overrides,
 * application-specific hints, learned-model classifiers) plug in via
 * the same registry without touching the hot-path interception code.
 *
 * Design:
 *
 *   F1 hook (cuLaunchKernel intercept) — Step 6 wiring target
 *     |
 *     v
 *   cipher_rt_classify_route(call, out) — for each registered actuator:
 *     |  actuator->classify(call, out):
 *     |    return HANDLED (out populated; substrate returns),
 *     |    PASSTHROUGH (skip me, try next),
 *     |    or ERROR (logged; fall through).
 *     |
 *     +-- if any actuator returns HANDLED -> done
 *     +-- if none HANDLED -> out->op_class = UNCLASSIFIED (0xFF), return PASSTHROUGH
 *
 * Default actuator (priority-0, auto-registered at static-init time):
 *   wraps may13's cipher::classify_launch() from
 *   include/may13/cipher_classify.hpp — a pure-function geometry-based
 *   classifier with no GPU/CUDA/NVML side effects. Safe at .so load time.
 *
 * Step 3 deliverable: substrate scaffolding only. Registry exists,
 * default classifier registered, but cipher_rt_classify_route() is
 * NOT yet called from any hot path. Step 6 wires it into the
 * cuLaunchKernel intercept (and downstream into cuBLAS/SDPA shims via
 * shared classification cache).
 */
#ifndef CIPHER_RT_CLASSIFY_SUBSTRATE_H
#define CIPHER_RT_CLASSIFY_SUBSTRATE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Call descriptor passed to classifiers. Append-only; populated by F1
 * hook before route() is called. Matches the geometry surface
 * cuLaunchKernel exposes (and may13's cipher::KernelGeom). */
struct cipher_rt_classify_call {
	const void *fn;            /* CUDA function pointer (cache key) */
	uint32_t    grid_x;
	uint32_t    grid_y;
	uint32_t    grid_z;
	uint32_t    block_x;
	uint32_t    block_y;
	uint32_t    block_z;
	uint32_t    shared_bytes;
	uint32_t    reserved_pad;  /* alignment + Cb.2 reserved-tail anchor */
	uint64_t    reserved[6];   /* future fields (stream, tenant_id, ...) */
};

/* Classifier output. op_class matches cipher::OpClass enum
 * (0=GEMM..6=ITERATIVE_CUSTOM, 0xFF=UNCLASSIFIED).
 * sizeof: 24 bytes total; first 4 bytes carry the active payload. */
struct cipher_rt_classify_out {
	uint8_t  op_class;         /* OpClass enum value */
	uint8_t  confidence;       /* 0..100 */
	uint8_t  cache_hit;        /* 0/1 from classifier's internal cache */
	uint8_t  reserved_pad[5];  /* alignment */
	uint64_t reserved[2];      /* Cb.2 future expansion */
};

/* Sentinel for "no classifier produced a verdict". */
#define CIPHER_RT_CLASSIFY_UNCLASSIFIED 0xFF

enum cipher_rt_classify_result {
	CIPHER_RT_CLASSIFY_HANDLED     = 0, /* out populated; substrate returns */
	CIPHER_RT_CLASSIFY_PASSTHROUGH = 1, /* skip me; try next */
	CIPHER_RT_CLASSIFY_ERROR       = 2, /* attempted and failed; fall through */
};

/* Actuator registration. */
struct cipher_rt_classify_actuator {
	const char *name;
	int         priority;  /* lower runs first; 0..255 */
	int       (*classify)(const struct cipher_rt_classify_call *call,
	                      struct cipher_rt_classify_out        *out);
};

#define CIPHER_RT_CLASSIFY_MAX_ACTUATORS 16

/* Register an actuator. Called from actuator-init paths at libcipher_rt
 * load time. Returns 0 on success, -1 if registry full or invalid. */
int cipher_rt_classify_register_actuator(
	const struct cipher_rt_classify_actuator *actuator);

/* Substrate init -- idempotent. Called from libcipher_rt init body
 * (cipher_inject.c) AND from the static-init constructor of the may13
 * default actuator (whichever fires first wins via atomic flag). */
int cipher_rt_classify_dispatch_init(void);

/* Substrate entry. Iterates registered actuators by priority order
 * (lowest first, stable for equal priorities). On HANDLED, populates
 * *out and returns CIPHER_RT_CLASSIFY_HANDLED. On no-match, sets
 * out->op_class = CIPHER_RT_CLASSIFY_UNCLASSIFIED and returns
 * CIPHER_RT_CLASSIFY_PASSTHROUGH. */
int cipher_rt_classify_route(
	const struct cipher_rt_classify_call *call,
	struct cipher_rt_classify_out        *out);

/* Diagnostic accessors. */
unsigned long cipher_rt_classify_calls_total(void);
unsigned long cipher_rt_classify_calls_handled(void);
unsigned long cipher_rt_classify_calls_passthrough(void);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_CLASSIFY_SUBSTRATE_H */
