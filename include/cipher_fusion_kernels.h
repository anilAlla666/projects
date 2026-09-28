// CIPHER fused-kernel substitutes — Track A (per-pattern transformer fusion).
//
// Each fused kernel replaces a sequence of PyTorch element-wise / reduction
// kernels that decompose a single transformer op (RMSNorm, SiLU·Mul,
// residual add). Compiled via Stage-6 NVRTC at first use; the resulting
// CUfunction handles are cached for the engine lifetime.
//
// Default OFF. Env: CIPHER_FUSION=on. Idempotent init.

#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int  cipher_fusion_kernels_init(void);
int  cipher_fusion_kernels_enabled(void);

// Pattern 1 — Fused RMSNorm.
//   out[r, i] = (x[r, i] * rsqrt(mean(x[r,:]^2) + eps)) * weight[i]
//   x, weight, out are fp16 device pointers; rows = batch*seq_len; hidden_dim
//   must be a multiple of 32. eps is float (e.g. 1e-6).
int cipher_fused_rmsnorm(
    void* x_fp16, void* weight_fp16, void* out_fp16,
    int rows, int hidden_dim, float eps, void* stream);

// Pattern 2 — Fused SiLU(gate) * up. All fp16, element-wise on numel.
int cipher_fused_silu_mul(
    void* gate_fp16, void* up_fp16, void* out_fp16,
    int numel, void* stream);

// Pattern 3 — Fused element-wise residual add: out[i] = x[i] + residual[i].
// fp16 in/out.
int cipher_fused_residual_add(
    void* x_fp16, void* residual_fp16, void* out_fp16,
    int numel, void* stream);

// Counters (testing / observability).
typedef struct {
    int      enabled;
    unsigned long long rmsnorm_calls;
    unsigned long long silu_mul_calls;
    unsigned long long residual_add_calls;
} CipherFusionStats;

int  cipher_fusion_kernels_stats(CipherFusionStats* out);

#ifdef __cplusplus
}
#endif
