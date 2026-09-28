// CIPHER pointer-flow pattern matcher — Stage 2 of Path C driver-level
// kernel fusion.  Triggered per cudaLaunchKernel after the recorder.
#ifndef CIPHER_FLOW_PATTERNS_H
#define CIPHER_FLOW_PATTERNS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize.  Reads CIPHER_FLOW_MATCH=on to gate.  Returns 1 if active.
int  cipher_flow_patterns_init(void);
int  cipher_flow_patterns_enabled(void);

// Called from cudaLaunchKernel shim AFTER cipher_flow_recorder_observe().
// Inspects the most recent flow entry against the active state machine.
// On MATCHED, logs the capture and stores a recipe for Stage 3.
void cipher_flow_patterns_check(
    void* func, uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t smem, void** args, void* stream);

// Stage 3 lookup: is this func_ptr the FIRST kernel of a known recipe?
// Returns recipe id ≥ 0 if match, -1 otherwise.
int  cipher_flow_patterns_lookup_first(void* func);

// Recipe accessor — Stage 3 reads this to drive suppression.
struct CipherFlowRecipe {
    int    valid;
    void*  fns[7];          // func_ptrs in observed order: fns[1]=MeanReduce ... fns[6]=Scale
    // Arg-position hints captured at learn time.
    int    x_arg_idx_k0;    // index of X pointer in kernel 0 args
    int    out_arg_idx_k5;  // index of Y pointer in kernel 5 args
    int    weight_arg_idx_k5; // index of W pointer in kernel 5 args
    int    rows_grid_axis;  // 0 = grid.x, 1 = grid.y, ...
    int    hidden_arg_idx;  // arg slot holding numel
    float  eps;             // sniffed from add-eps kernel
    int    n_observed;      // how many times this recipe has been matched
};
const struct CipherFlowRecipe* cipher_flow_patterns_get_recipe(int id);
int    cipher_flow_patterns_recipe_count(void);

#ifdef __cplusplus
}
#endif

#endif
