#pragma once
#include "cipher_liquid_state.h"
#include "cipher_classify.hpp"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CIPHER_LAYOUT_UNCHANGED   = 0,
    CIPHER_LAYOUT_ROW_MAJOR   = 1,
    CIPHER_LAYOUT_COL_MAJOR   = 2,
    CIPHER_LAYOUT_TILED_32    = 3,
    CIPHER_LAYOUT_TILED_128   = 4,
    CIPHER_LAYOUT_XCD_ALIGNED = 5,
} CipherLayoutHint;

typedef struct {
    CipherLayoutHint  hint;
    bool              change_recommended;
    float             estimated_bw_gain;
    const char*       reason;
} CipherLayoutDecision;

typedef struct {
    uint64_t total_decisions;
    uint64_t changes_recommended;
    float    avg_estimated_gain;
    bool     initialized;
    bool     amd_mode;
} CipherMemLayoutState;

void                  cipher_mem_layout_init(CipherMemLayoutState* state, bool amd_mode);
CipherLayoutDecision  cipher_mem_layout_decide(CipherMemLayoutState* state,
                                               const CipherLiquidStateMgr* liquid,
                                               uint8_t op_class,
                                               uint32_t dim_m, uint32_t dim_n);
const char*           cipher_layout_hint_name(CipherLayoutHint hint);
void                  cipher_mem_layout_report(const CipherMemLayoutState* state);

#ifdef __cplusplus
}
#endif
