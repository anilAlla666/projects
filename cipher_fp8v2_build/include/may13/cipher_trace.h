// Op 28 TRACE — bounded kernel-level execution trace exporter.
//
// Stage 1 observer appends a compact record to a fixed-size ring. Report
// flushes to JSONL (one record per line) for offline OTLP ingest. No CUDA,
// no allocation. Drop-on-full (we report overflow count).
//
// Default OFF: env var `CIPHER_TRACE=on`. Contract I1–I6.
#pragma once
#include <stdint.h>
#include "may13/cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_trace_init(void);
void     cipher_trace_observe(const CipherRingEntry* ev);
uint64_t cipher_trace_written(void);
uint64_t cipher_trace_dropped(void);
void     cipher_trace_report(void);

#ifdef __cplusplus
}
#endif
