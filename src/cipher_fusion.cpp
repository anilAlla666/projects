#ifdef CIPHER_CPU_STUB
#  include "cipher_stubs.h"
#endif
#include "cipher_fusion.h"
#include "cipher_classify.hpp"
#include <stdio.h>
#include <string.h>

#define OP_GEMM  ((uint8_t)cipher::OpClass::GEMM)
#define OP_EW    ((uint8_t)cipher::OpClass::ELEMENTWISE)
#define OP_RED   ((uint8_t)cipher::OpClass::REDUCTION)

static uint8_t get_op(const CipherWorkloadRhythm* r, int offset) {
    if (r->fill == 0) return 0xFF;
    int idx = ((int)r->write_head - 1 - offset + CIPHER_WORKLOAD_HIST_LEN)
              % CIPHER_WORKLOAD_HIST_LEN;
    return r->op_class[idx];
}
static bool pat(const CipherWorkloadRhythm* r, const uint8_t* p, int n) {
    if ((int)r->fill < n) return false;
    for (int i = 0; i < n; i++) if (get_op(r,i) != p[i]) return false;
    return true;
}

typedef struct { uint8_t p[4]; int n; CipherFusionType t; float spd; } FP;
static const FP g_pat[] = {
    {{OP_EW,OP_EW,OP_GEMM,0}, 3, CIPHER_FUSE_GEMM_BIAS_GELU, 1.30f},
    {{OP_EW,OP_GEMM,0,0},     2, CIPHER_FUSE_GEMM_RESIDUAL,  1.12f},
    {{OP_GEMM,OP_RED,0,0},    2, CIPHER_FUSE_LAYERNORM_GEMM, 1.15f},
    {{OP_GEMM,OP_GEMM,0,0},   2, CIPHER_FUSE_GEMM_GEMM_QKV,  1.40f},
    {{0,0,0,0}, 0, CIPHER_FUSE_NONE, 0.0f},
};

void cipher_fusion_init(CipherFusionState* state) {
    memset(state, 0, sizeof(*state));
    state->initialized = true;
    fprintf(stderr, "[CIPHER L2.2] Fusion Engine initialized. %zu patterns.\n",
            sizeof(g_pat)/sizeof(g_pat[0])-1);
}

CipherFusionDecision cipher_fusion_decide(CipherFusionState* state,
                                           const CipherLiquidStateMgr* liquid) {
    CipherFusionDecision d = {CIPHER_FUSE_NONE, false, 0, 1.0f};
    if (!state->initialized) return d;
    state->total_decisions++;
    if (!liquid || !liquid->initialized || !liquid->device) return d;
    const CipherWorkloadRhythm* r = &liquid->device->rhythm;
    for (int i = 0; g_pat[i].n > 0; i++) {
        if (pat(r, g_pat[i].p, g_pat[i].n)) {
            d.type = g_pat[i].t; d.should_fuse = true;
            d.pattern_length = g_pat[i].n; d.predicted_speedup = g_pat[i].spd;
            state->fusions_issued++;
            if ((int)d.type < 6) state->fusions_by_type[(int)d.type]++;
            float n = (float)state->fusions_issued;
            state->avg_speedup = state->avg_speedup*(n-1)/n + d.predicted_speedup/n;
            return d;
        }
    }
    return d;
}

const char* cipher_fusion_type_name(CipherFusionType t) {
    switch(t) {
        case CIPHER_FUSE_GEMM_BIAS_GELU: return "GEMM+BIAS+GELU";
        case CIPHER_FUSE_GEMM_BIAS_SILU: return "GEMM+BIAS+SILU";
        case CIPHER_FUSE_LAYERNORM_GEMM: return "LAYERNORM+GEMM";
        case CIPHER_FUSE_GEMM_GEMM_QKV:  return "GEMM+GEMM(QKV)";
        case CIPHER_FUSE_GEMM_RESIDUAL:  return "GEMM+RESIDUAL";
        default: return "NONE";
    }
}

void cipher_fusion_report(const CipherFusionState* state) {
    fprintf(stderr, "[CIPHER L2.2] Fusion: %lu decisions, %lu fused (%.1f%%), avg %.2fx\n",
        state->total_decisions, state->fusions_issued,
        state->total_decisions > 0
            ? (double)state->fusions_issued*100.0/state->total_decisions : 0.0,
        state->avg_speedup);
}
