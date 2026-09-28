// Op 29 COMPLY — regulatory compliance artifact bundler.
//
// Aggregates live state from RECEIPT, CARBON, GUARD, DETERMINISM,
// FAIRNESS, TOPOLOGY via their C APIs and emits a single JSON
// "compliance pack" at report time. Observer is a no-op; all work runs
// in report.
//
// Default OFF: env var `CIPHER_COMPLY=on`. Contract I1–I6.
#pragma once
#include <stdint.h>
#include "may13/cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int  cipher_comply_init(void);
void cipher_comply_observe(const CipherRingEntry* ev);
int  cipher_comply_ok(void);
void cipher_comply_report(void);

#ifdef __cplusplus
}
#endif
