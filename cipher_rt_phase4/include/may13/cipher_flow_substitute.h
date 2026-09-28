// CIPHER pointer-flow substitution — Stage 3.
//
// On every cudaLaunchKernel, checks if the func_ptr matches the FIRST kernel
// of a known FlowRecipe.  If yes, opens a suppression window of N-1 launches
// expecting the rest of the sequence.  Final step calls
// cipher_fused_rmsnorm(X, W, Y, rows, hidden_dim, eps, stream).
#ifndef CIPHER_FLOW_SUBSTITUTE_H
#define CIPHER_FLOW_SUBSTITUTE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int  cipher_flow_substitute_init(void);
int  cipher_flow_substitute_enabled(void);

// Returns 1 if this launch should be SUPPRESSED (skip the real cuda
// launch, return cudaSuccess); 0 to launch normally.  When transitioning
// from SUPPRESSING to RESUMED on the final step, may issue the fused
// substitute call internally before returning 1.
int  cipher_flow_substitute_consider(
    void* func, uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t smem, void** args, void* stream);

void cipher_flow_substitute_report(void);

#ifdef __cplusplus
}
#endif

#endif
