// Op 21 DETERMINISM — reproducible dispatch-sequence fingerprint.
//
// Stage 1 observer — folds the ordered `params_hash` stream into a running
// 64-bit FNV-mix hash. For identical deterministic workloads (same seed,
// same shapes, same call order) the reported hash matches across reruns.
//
// Default OFF: env var `CIPHER_DETERMINISM=on`. Contract I1–I6.
#pragma once
#include <stdint.h>
#include "may13/cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_determinism_init(void);
void     cipher_determinism_observe(const CipherRingEntry* ev);
uint64_t cipher_determinism_hash(void);
uint64_t cipher_determinism_count(void);
void     cipher_determinism_report(void);

#ifdef __cplusplus
}
#endif
