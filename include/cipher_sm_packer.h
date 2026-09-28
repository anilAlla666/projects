#pragma once
#include "cipher_liquid_state.h"
#include <stdint.h>
#include <stdbool.h>

#define CIPHER_SM_IDLE_THRESHOLD  0.05f
#define CIPHER_PACK_SIGNAL_NONE   0x00
#define CIPHER_PACK_SIGNAL_PACK   0x01
#define CIPHER_PACK_SIGNAL_HOLD   0x02

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float    idle_ema;
    float    pack_rate;
    uint64_t total_decisions;
    uint64_t pack_signals;
    uint64_t hold_signals;
    bool     initialized;
} CipherSmPackerState;

void    cipher_sm_packer_init(CipherSmPackerState* state);
uint8_t cipher_sm_packer_decide(CipherSmPackerState* state,
                                const CipherLiquidStateMgr* liquid);
void    cipher_sm_packer_record(CipherSmPackerState* state, uint8_t signal);
void    cipher_sm_packer_report(const CipherSmPackerState* state);

#ifdef __cplusplus
}
#endif
