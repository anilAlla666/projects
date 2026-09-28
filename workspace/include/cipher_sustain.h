// Op 15 SUSTAIN — KV-cache pressure detector (Stage 1 inline).
//
// Default OFF: env var `CIPHER_SUSTAIN=on` enables (also requires
// CIPHER_SENSE=on, since SUSTAIN reads SENSE's session classification for
// the AGENT-relaxed-threshold rule).
//
// Per active session, runs a rolling linear regression of decode-step
// latency vs step number over the last 20 decode events. When
//   slope > CIPHER_SUSTAIN_SLOPE_THRESHOLD_NS_PER_STEP   (default 50,000 ns/step)
// the session's `sustain_compress_flag` is set (atomic).
//
// AGENT_AUTONOMOUS sessions with decode_count > 500 use 3× the threshold
// (long reasoning chains legitimately accumulate KV state faster than
// conversations).
//
// v1: the sustain_compress_flag is NOT yet read by Stage 0 — wiring into
// Op 3 SUBSTITUTE (`src/cipher_block_sub_kernel.cu`) is a separate Stage-0
// plan requiring explicit approval, same family as SHIELD's deferred flags.
#pragma once
#include <stdint.h>
#include "cipher_10ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int      cipher_sustain_init(void);
void     cipher_sustain_observe(const CipherRingEntry* ev);
unsigned cipher_sustain_pressure_event_count(void);
unsigned cipher_sustain_compress_flag_count(void);
double   cipher_sustain_last_slope_ns_per_step(uint64_t session_id);
void     cipher_sustain_report(void);

#ifdef __cplusplus
}
#endif
