// Op 24 FAIRNESS — kernel-level tenant work quota.
//
// Accumulates per-session "work units" (grid*block volume) as a FLOP proxy;
// flags tenants that exceed CIPHER_FAIRNESS_QUOTA. Session attribution
// requires SENSE. Hint-only v1 — no enforcement.
//
// Default OFF: env var `CIPHER_FAIRNESS=on`. Contract I1–I6.
#pragma once
#include <stdint.h>
#include "may13/cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_fairness_init(void);
void     cipher_fairness_observe(const CipherRingEntry* ev);
unsigned cipher_fairness_tenant_count(void);
unsigned cipher_fairness_overrun_count(void);
void     cipher_fairness_report(void);

#ifdef __cplusplus
}
#endif
