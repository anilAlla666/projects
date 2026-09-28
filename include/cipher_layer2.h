// =============================================================================
// CIPHER — L2.7: Layer 2 Full LNN Integration
// cipher_layer2.h
// =============================================================================

#pragma once

#include "cipher_sm_packer.h"
#include "cipher_fusion.h"
#include "cipher_mem_layout.h"
#include "cipher_nccl_bpf.h"
#include "cipher_nccl_neural.h"
#include "cipher_liquid_state.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t              sm_signal;
    CipherFusionDecision fusion;
    CipherLayoutDecision layout;
    CipherNcclPolicy     nccl_policy;
    bool                 nccl_active;
    float                predicted_ar_ns;
    bool                 overlap_window;
    uint64_t             decision_ns;
} CipherLayer2Decision;

typedef struct {
    CipherSmPackerState    sm_packer;
    CipherFusionState      fusion;
    CipherMemLayoutState   mem_layout;
    CipherNcclBpfState     nccl_bpf;
    CipherNcclNeuralState  nccl_neural;
    CipherOverlapState     overlap;
    bool                   initialized;
    uint64_t               total_decisions;
    float                  avg_decision_ns;
} CipherLayer2State;

int cipher_layer2_init(CipherLayer2State* state, bool amd_mode);

CipherLayer2Decision cipher_layer2_decide(
    CipherLayer2State*         state,
    const CipherLiquidStateMgr* liquid,
    uint8_t                    op_class,
    uint32_t                   dim_m,
    uint32_t                   dim_n,
    bool                       is_allreduce,
    size_t                     ar_msg_bytes,
    uint32_t                   num_ranks);

void cipher_layer2_destroy(CipherLayer2State* state);
void cipher_layer2_report(const CipherLayer2State* state);

#ifdef __cplusplus
}
#endif
