#pragma once
#include "cipher_liquid_state.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CIPHER_FUSE_NONE           = 0,
    CIPHER_FUSE_GEMM_BIAS_GELU = 1,
    CIPHER_FUSE_GEMM_BIAS_SILU = 2,
    CIPHER_FUSE_LAYERNORM_GEMM = 3,
    CIPHER_FUSE_GEMM_GEMM_QKV  = 4,
    CIPHER_FUSE_GEMM_RESIDUAL  = 5,
} CipherFusionType;

typedef struct {
    CipherFusionType type;
    bool             should_fuse;
    uint8_t          pattern_length;
    float            predicted_speedup;
} CipherFusionDecision;

typedef struct {
    uint64_t total_decisions;
    uint64_t fusions_issued;
    uint64_t fusions_by_type[6];
    float    avg_speedup;
    bool     initialized;
} CipherFusionState;

void                 cipher_fusion_init(CipherFusionState* state);
CipherFusionDecision cipher_fusion_decide(CipherFusionState* state,
                                          const CipherLiquidStateMgr* liquid);
const char*          cipher_fusion_type_name(CipherFusionType type);
void                 cipher_fusion_report(const CipherFusionState* state);

#ifdef __cplusplus
}
#endif
