// Op 23 CARBON — per-session carbon certificate.
//
// Accumulates per-session kernel work-units, multiplies by a configurable
// joules-per-unit constant and grid intensity (gCO2/kWh) to emit an
// estimated gCO2 figure per tenant.
//
// Default OFF: env var `CIPHER_CARBON=on`.
//   CIPHER_CARBON_J_PER_UNIT   (default 1e-9 J per work unit)
//   CIPHER_CARBON_GCO2_PER_KWH (default 400 gCO2/kWh)
// Contract I1–I6. Requires SENSE.
#pragma once
#include <stdint.h>
#include "cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_carbon_init(void);
void     cipher_carbon_observe(const CipherRingEntry* ev);
unsigned cipher_carbon_session_count(void);
void     cipher_carbon_report(void);

#ifdef __cplusplus
}
#endif
