/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_attn_dispatch.h - Phase 4.6.1 attention-routing substrate.
 *
 * Twin of T4.5.1 cuBLAS matmul substrate, for the PyTorch SDPA path.
 * Future actuators (KV dedup L1, L3 cuIpc cross-process, paged-attention
 * adapter, ...) plug in via the same registry without touching the shim.
 *
 * Design:
 *
 *   PyTorch nn.functional.scaled_dot_product_attention
 *     -> backend selection -> aten::_ops::_scaled_dot_product_*_attention::call
 *           (flash / efficient / cuDNN — one of three on this stack)
 *     |
 *     v   LD_PRELOAD interposition (plain mangled symbols, no .symver:
 *         ATen ops are not version-tagged on libtorch_cpu.so)
 *     |
 *   cipher_rt_attn_dispatch(call, passthrough_fn):
 *     foreach actuator:
 *       result = actuator->maybe_handle(call):
 *         HANDLED    -> substrate returns; actuator filled return slots
 *         REDIRECTED -> Q/K/V pointer substitution (T4.6.3+, mechanism
 *                       deferred until L1 actuator drives the choice)
 *         PASSTHROUGH -> try next actuator
 *         ERROR      -> log; fall through to real backend
 *     no actuator handled -> passthrough_fn(orig args)
 *
 * Verified-against torch: 2.11.0+cu130 (libtorch_cpu.so SDPA dispatcher
 * symbols are plain C++ mangled, un-versioned). Symbol drift safety:
 * substrate resolves originals via dlsym RTLD_NEXT at first call; on
 * dlsym failure the host application is hard-aborted (better than
 * silent corruption). If the substrate registers no shims (e.g. running
 * against a torch version that doesn't export these symbols), it stays
 * inactive and host application continues normally.
 */
#ifndef CIPHER_RT_ATTN_DISPATCH_H
#define CIPHER_RT_ATTN_DISPATCH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Tag for which SDPA backend this call entered through. Actuators may
 * key behavior on backend (e.g. L1 dedup logic is identical across
 * backends but quantized-flash variant carries extra descale tensors). */
enum cipher_rt_attn_backend {
	CIPHER_RT_ATTN_BACKEND_FLASH     = 1,
	CIPHER_RT_ATTN_BACKEND_EFFICIENT = 2,
	CIPHER_RT_ATTN_BACKEND_CUDNN     = 3,
};

/* Q/K/V tensor view, populated by the substrate from the at::Tensor refs.
 * Opaque pointers (the actuator never needs to dereference at::Tensor;
 * if it does it links against libtorch headers separately).
 *
 * Shape conventions reflect PyTorch SDPA: tensors are (B, H_kv_or_q, S, D)
 * for Q on K/V respectively. For GQA, K/V H is fewer than Q H.
 */
struct cipher_rt_attn_tensor {
	const void *data_ptr;   /* device pointer (CUDA) or host (math fallback) */
	int64_t     sizes[4];   /* (B, H, S, D) — zero-padded if rank < 4 */
	int64_t     strides[4]; /* element strides */
	int         rank;       /* actual rank, usually 4 */
	int         dtype;      /* c10::ScalarType raw int (5=Half, 6=Float, ...) */
	int         device_type;/* c10::DeviceType raw int (1=CPU, 2=CUDA) */
	int         device_index;
};

/* Call descriptor passed to actuators. Append-only for future fields. */
struct cipher_rt_attn_call {
	enum cipher_rt_attn_backend backend;
	struct cipher_rt_attn_tensor q;
	struct cipher_rt_attn_tensor k;
	struct cipher_rt_attn_tensor v;
	double      dropout_p;
	int         is_causal;
	int         return_debug_mask; /* flash + cudnn only */
	int         compute_log_sumexp;/* efficient + cudnn only */
	double      scale;            /* NaN if unset */
	int         scale_is_set;
	/* Optional attention bias (efficient/cudnn variants). NULL if not provided. */
	const void *attn_bias_data;
	int64_t     attn_bias_sizes[4];
	int         attn_bias_rank;
	uint64_t    reserved[6];
};

enum cipher_rt_attn_result {
	CIPHER_RT_ATTN_HANDLED     = 0, /* actuator filled return slots */
	CIPHER_RT_ATTN_PASSTHROUGH = 1, /* skip me, try next or real backend */
	CIPHER_RT_ATTN_REDIRECTED  = 2, /* T4.6.3+: substitute Q/K/V */
	CIPHER_RT_ATTN_ERROR       = 3, /* attempted but failed; fall through */
};

/* Actuator registration. */
struct cipher_rt_attn_actuator {
	const char *name;
	int         priority;  /* lower runs first; 0..255 */
	int       (*maybe_handle)(const struct cipher_rt_attn_call *call);
	/* T4.6.3+: when REDIRECTED is returned, substituted_qkv[] receives
	 * three replacement device pointers. NULL means "no substitution
	 * for this operand". Unused in T4.6.1. */
};

#define CIPHER_RT_ATTN_MAX_ACTUATORS 16

/* Register an actuator. Called from actuator init at libcipher_rt
 * load time. Returns 0 on success, -1 if registry full. */
int cipher_rt_attn_register_actuator(
	const struct cipher_rt_attn_actuator *actuator);

/* Substrate init - called from libcipher_rt init body. Idempotent.
 * Resolves the three SDPA ::call originals via dlsym RTLD_NEXT and
 * stores them for passthrough. Returns 0 on success. Returns
 * non-zero if any original is unresolvable — caller logs and falls
 * back to inactive substrate. */
int cipher_rt_attn_dispatch_init(void);

/* Diagnostic accessors. */
unsigned long cipher_rt_attn_calls_total(void);
unsigned long cipher_rt_attn_calls_handled(void);
unsigned long cipher_rt_attn_calls_passthrough(void);
unsigned long cipher_rt_attn_calls_redirected(void);
unsigned long cipher_rt_attn_calls_by_backend(enum cipher_rt_attn_backend b);

/* Verified-at PyTorch version. If torch version differs at runtime,
 * substrate logs a warning but does not refuse — symbol resolution
 * at dlsym time is the binding check. */
#define CIPHER_RT_ATTN_TORCH_VERIFIED "2.11.0+cu130"

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_ATTN_DISPATCH_H */
