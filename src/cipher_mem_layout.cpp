#ifdef CIPHER_CPU_STUB
#  include "cipher_stubs.h"
#endif
#include "cipher_mem_layout.h"
#include <stdio.h>
#include <string.h>

#define BW_HIGH 0.80f
#define BW_LOW  0.40f

void cipher_mem_layout_init(CipherMemLayoutState* state, bool amd_mode) {
    memset(state, 0, sizeof(*state));
    state->amd_mode = amd_mode;
    state->initialized = true;
    fprintf(stderr, "[CIPHER L2.3] Memory Layout Optimizer initialized. Mode: %s\n",
            amd_mode ? "AMD MI300X" : "NVIDIA");
}

CipherLayoutDecision cipher_mem_layout_decide(CipherMemLayoutState* state,
                                               const CipherLiquidStateMgr* liquid,
                                               uint8_t op_class,
                                               uint32_t dim_m, uint32_t dim_n) {
    CipherLayoutDecision d = {CIPHER_LAYOUT_UNCHANGED, false, 0.0f, "ok"};
    if (!state->initialized) return d;
    state->total_decisions++;

    float bw = 0.5f;
    if (liquid && liquid->initialized && liquid->device)
        bw = liquid->device->hw.hbm_bw_utilized;

    if (state->amd_mode && ((dim_m % 8 != 0) || (dim_n % 8 != 0))) {
        d.hint = CIPHER_LAYOUT_XCD_ALIGNED;
        d.change_recommended = true;
        d.estimated_bw_gain = 0.15f;
        d.reason = "AMD XCD: enforce workgroup multiple-of-8";
        goto done;
    }

    if (bw < BW_LOW) { d.reason = "BW already low"; goto done; }

    using namespace cipher;
    switch ((OpClass)op_class) {
        case OpClass::GEMM:
            if (dim_m == dim_n) {
                d.hint = CIPHER_LAYOUT_ROW_MAJOR;
                d.change_recommended = (bw > BW_HIGH);
                d.estimated_bw_gain = 0.08f;
                d.reason = "Square GEMM: row-major";
            } else if (dim_m < 64 && dim_n > 1024) {
                d.hint = CIPHER_LAYOUT_COL_MAJOR;
                d.change_recommended = true;
                d.estimated_bw_gain = 0.12f;
                d.reason = "Decode GEMM: col-major";
            }
            break;
        case OpClass::ATTENTION:
            if (dim_n == 128 || dim_n == 64) {
                d.hint = CIPHER_LAYOUT_TILED_128;
                d.change_recommended = (bw > BW_HIGH);
                d.estimated_bw_gain = 0.10f;
                d.reason = "Attention: tiled-128";
            }
            break;
        case OpClass::MEMCPY_TRANSPOSE:
            d.hint = CIPHER_LAYOUT_TILED_32;
            d.change_recommended = true;
            d.estimated_bw_gain = 0.15f;
            d.reason = "Transpose: tiled-32";
            break;
        case OpClass::ELEMENTWISE:
        case OpClass::REDUCTION:
            d.hint = CIPHER_LAYOUT_ROW_MAJOR;
            d.change_recommended = (bw > BW_HIGH);
            d.estimated_bw_gain = 0.05f;
            d.reason = "EW/Reduce: row-major";
            break;
        default: break;
    }

done:
    if (d.change_recommended) {
        state->changes_recommended++;
        float n = (float)state->changes_recommended;
        state->avg_estimated_gain = state->avg_estimated_gain*(n-1)/n
                                    + d.estimated_bw_gain/n;
    }
    return d;
}

const char* cipher_layout_hint_name(CipherLayoutHint h) {
    switch(h) {
        case CIPHER_LAYOUT_ROW_MAJOR:   return "ROW_MAJOR";
        case CIPHER_LAYOUT_COL_MAJOR:   return "COL_MAJOR";
        case CIPHER_LAYOUT_TILED_32:    return "TILED_32";
        case CIPHER_LAYOUT_TILED_128:   return "TILED_128";
        case CIPHER_LAYOUT_XCD_ALIGNED: return "XCD_ALIGNED";
        default: return "UNCHANGED";
    }
}

void cipher_mem_layout_report(const CipherMemLayoutState* state) {
    fprintf(stderr, "[CIPHER L2.3] Layout: %lu decisions, %lu changes (%.1f%%), avg gain %.1f%%\n",
        state->total_decisions, state->changes_recommended,
        state->total_decisions > 0
            ? (double)state->changes_recommended*100.0/state->total_decisions : 0.0,
        state->avg_estimated_gain * 100.0f);
}
