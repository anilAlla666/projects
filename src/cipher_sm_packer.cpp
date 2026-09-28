#ifdef CIPHER_CPU_STUB
#  include "cipher_stubs.h"
#endif
#include "cipher_sm_packer.h"
#include <stdio.h>
#include <string.h>

void cipher_sm_packer_init(CipherSmPackerState* state) {
    memset(state, 0, sizeof(*state));
    state->initialized = true;
    fprintf(stderr, "[CIPHER L2.1] SM Work Packer initialized. Idle threshold: %.0f%%\n",
            CIPHER_SM_IDLE_THRESHOLD * 100.0f);
}

uint8_t cipher_sm_packer_decide(CipherSmPackerState* state,
                                const CipherLiquidStateMgr* liquid) {
    if (!state->initialized) return CIPHER_PACK_SIGNAL_NONE;
    float idle = 0.0f;
    if (liquid && liquid->initialized && liquid->device)
        idle = liquid->device->hw.sm_idle_fraction;
    const float kappa = 0.9f;
    state->idle_ema = kappa * state->idle_ema + (1.0f - kappa) * idle;
    state->total_decisions++;
    if (state->idle_ema > CIPHER_SM_IDLE_THRESHOLD) {
        state->pack_signals++;
        return CIPHER_PACK_SIGNAL_PACK;
    }
    state->hold_signals++;
    return CIPHER_PACK_SIGNAL_HOLD;
}

void cipher_sm_packer_record(CipherSmPackerState* state, uint8_t signal) {
    (void)signal;
    if (state->total_decisions > 0)
        state->pack_rate = (float)state->pack_signals / state->total_decisions;
}

void cipher_sm_packer_report(const CipherSmPackerState* state) {
    fprintf(stderr,
        "[CIPHER L2.1] SM Packer: %lu decisions, PACK=%.1f%%, idle_EMA=%.1f%%\n",
        state->total_decisions,
        state->total_decisions > 0
            ? (double)state->pack_signals * 100.0 / state->total_decisions : 0.0,
        state->idle_ema * 100.0f);
}
