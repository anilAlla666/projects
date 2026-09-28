// CIPHER NCCL Tuner v4 + Green Context — Stage 9
//
// v4 ABI augmentation that prefers NVLS for large reductions and reserves a
// Green Context with N SMs for NCCL co-scheduling on H100.
// Default OFF. Env: CIPHER_NCCL_V4=on.

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CipherNcclV4Stats {
    int      enabled;
    int      green_ctx_sms;
    uint64_t tuner_calls;
    uint64_t nvls_chosen;
    uint64_t ring_chosen;
    uint64_t tree_chosen;
    uint64_t large_reductions;
    uint64_t green_ctx_launches;
} CipherNcclV4Stats;

int  cipher_nccl_v4_init(void);
int  cipher_nccl_v4_enabled(void);

// Called from libcipher_nccl_tuner.so via dlsym, gives v4 a chance to bias
// the algorithm choice. Returns 1 if it overrode the decision.
int  cipher_nccl_v4_decide(uint64_t bytes, uint32_t num_ranks, int* out_algo);

int  cipher_nccl_v4_stats(CipherNcclV4Stats* out);
void cipher_nccl_v4_report(void);

#ifdef __cplusplus
}
#endif
