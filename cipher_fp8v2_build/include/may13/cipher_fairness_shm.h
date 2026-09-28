// CIPHER cross-tenant FAIRNESS via POSIX shared memory.  Pure C atomics +
// shm_open/mmap.  No torch, no rt-side abstractions.
#ifndef CIPHER_FAIRNESS_SHM_H
#define CIPHER_FAIRNESS_SHM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize SHM region.  Reads CIPHER_FAIRNESS=on to gate.
// Returns 1 if active, 0 if disabled or shm setup failed.
int  cipher_fairness_shm_init(void);
int  cipher_fairness_shm_enabled(void);

// Register the calling process.  Reads CIPHER_TENANT_ID env (default
// pid).  Idempotent.  Returns the slot index assigned.
int  cipher_fairness_shm_register(void);

// Per-GEMM hooks.  Cheap atomic ops.
void cipher_fairness_shm_record_gemm(void);

// Should this tenant yield?  Returns 1 if its gemm rate is more than
// 2× the average across active tenants.
int  cipher_fairness_shm_should_yield(void);

// Stats accessor — pulls the local tenant's slot snapshot.
int  cipher_fairness_shm_self_calls(uint64_t* out_calls,
                                    uint64_t* out_yields);

void cipher_fairness_shm_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
