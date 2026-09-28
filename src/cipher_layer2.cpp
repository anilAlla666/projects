#ifdef CIPHER_CPU_STUB
#  include "cipher_stubs.h"
#endif
#include "cipher_layer2.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns_l27(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int cipher_layer2_init(CipherLayer2State* state, bool amd_mode) {
    memset(state, 0, sizeof(*state));
    cipher_sm_packer_init(&state->sm_packer);
    cipher_fusion_init(&state->fusion);
    cipher_mem_layout_init(&state->mem_layout, amd_mode);
    cipher_nccl_bpf_init(&state->nccl_bpf);
    cipher_nccl_neural_init(&state->nccl_neural);
    cipher_overlap_init(&state->overlap);
    state->initialized = true;
    fprintf(stderr, "[CIPHER L2] Layer 2 Koopman Orchestrator — INITIALIZED\n");
    return 0;
}

CipherLayer2Decision cipher_layer2_decide(
    CipherLayer2State* state, const CipherLiquidStateMgr* liquid,
    uint8_t op_class, uint32_t dim_m, uint32_t dim_n,
    bool is_allreduce, size_t ar_msg_bytes, uint32_t num_ranks)
{
    uint64_t t0 = now_ns_l27();
    CipherLayer2Decision dec = {};
    if (!state->initialized) return dec;

    dec.sm_signal = cipher_sm_packer_decide(&state->sm_packer, liquid);
    cipher_sm_packer_record(&state->sm_packer, dec.sm_signal);
    dec.fusion = cipher_fusion_decide(&state->fusion, liquid);
    dec.layout = cipher_mem_layout_decide(&state->mem_layout, liquid,
                                          op_class, dim_m, dim_n);

    dec.nccl_active = is_allreduce && (ar_msg_bytes > 0);
    if (dec.nccl_active) {
        float nvlink = 0.3f, ema = 50000000.0f;
        if (liquid && liquid->initialized && liquid->device) {
            nvlink = liquid->device->hw.nvlink_utilization;
            ema    = liquid->device->nccl.ema_duration_ns;
        }
        CipherNcclMsgCtx ctx = {ar_msg_bytes, num_ranks ? num_ranks : 8,
                                nvlink, ema, true};
        CipherNcclPolicy sp = cipher_nccl_bpf_decide(&state->nccl_bpf, &ctx, liquid);
        CipherNcclPolicy np = cipher_nccl_neural_decide(&state->nccl_neural, &ctx, liquid);
        dec.nccl_policy = (np.confidence > sp.confidence) ? np : sp;

        dec.predicted_ar_ns = cipher_overlap_schedule(
            &state->overlap, (uint64_t)(uintptr_t)&dec, ar_msg_bytes, liquid);
        dec.overlap_window = (dec.predicted_ar_ns > 1000000.0f);
    }

    dec.decision_ns = now_ns_l27() - t0;
    state->total_decisions++;
    float n = (float)state->total_decisions;
    state->avg_decision_ns = state->avg_decision_ns*(n-1)/n
                             + (float)dec.decision_ns/n;
    return dec;
}

void cipher_layer2_destroy(CipherLayer2State* state) {
    if (!state->initialized) return;
    cipher_nccl_bpf_destroy(&state->nccl_bpf);
    state->initialized = false;
    fprintf(stderr, "[CIPHER L2] Layer 2 destroyed.\n");
}

void cipher_layer2_report(const CipherLayer2State* state) {
    fprintf(stderr, "\n[CIPHER L2] Layer 2 Report: %lu decisions, avg %.0f ns\n",
            state->total_decisions, state->avg_decision_ns);
    cipher_sm_packer_report(&state->sm_packer);
    cipher_fusion_report(&state->fusion);
    cipher_mem_layout_report(&state->mem_layout);
    cipher_nccl_bpf_report(&state->nccl_bpf);
    cipher_nccl_neural_report(&state->nccl_neural);
    cipher_overlap_report(&state->overlap);
}
