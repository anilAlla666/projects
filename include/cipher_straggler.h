// Op (Phase 3) — Straggler Detection (local v1).
//
// HONEST SPEC DEVIATION:
//   The Track A Phase 3 spec assumed `ncclTunerPlugin_v2` could observe
//   per-rank completion times via the tuner callback. That ABI does NOT
//   exist (see cipher_nccl_tuner_abi.h:86–105 — only init/getCollInfo/
//   destroy callbacks, no post-collective callback, no per-rank timing).
//   ncclAllReduce on the calling rank is synchronous from the caller's
//   perspective: the duration we measure is dominated by the slowest rank
//   but cannot be attributed to a specific rank from one rank's intercept.
//
// v1 implementation: LOCAL slowdown detector + per-rank labeled telemetry.
//   - Reads $RANK / $LOCAL_RANK / $WORLD_SIZE at init (torchrun / mpirun /
//     slurm conventions). Falls back to rank=-1 (local-only) if absent.
//   - Per AllReduce: appends one JSON line to
//       /tmp/cipher_straggler_rank_<RANK>.jsonl
//     (operator runs an offline aggregator across all ranks' files to do
//     true cross-rank attribution; tools/straggler_aggregate.py does this.)
//   - Detects local slowdown: EMA of (dur_ns / bytes) per size bucket;
//     sustained 5-call deviation > 1.5x EMA flips the local-slowdown flag.
//   - When in ACTIVE mode and slowdown sustains, returns a tail-tolerant
//     algorithm hint (RING) via cipher_straggler_algo_hint(bytes).
//
// Cross-rank rank attribution is DEFERRED to a future "Straggler
// Coordinator" op (or operator-level aggregator script).
//
// Env vars:
//   CIPHER_STRAGGLER=observe → log telemetry + detect local slowdown
//   CIPHER_STRAGGLER=active  → above + return algo hints to ORCHESTRATE
//   anything else / unset    → off (single-cycle exit)
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CIPHER_STRAGGLER_OFF      = 0,
    CIPHER_STRAGGLER_OBSERVE  = 1,
    CIPHER_STRAGGLER_ACTIVE   = 2,
} CipherStragglerMode;

int       cipher_straggler_init(void);
void      cipher_straggler_observe(uint64_t bytes, uint64_t dur_ns, int chosen_algo);
int       cipher_straggler_local_slowdown_active(void);
unsigned  cipher_straggler_event_count(void);
int       cipher_straggler_my_rank(void);                  // -1 = unknown
int       cipher_straggler_world_size(void);               // 0 = unknown
const char* cipher_straggler_status_string(void);
int       cipher_straggler_algo_hint(uint64_t bytes);      // -1 = no hint, else NcclAlgo

// Test-only injection
void      cipher_straggler_inject_dur(uint64_t bytes, uint64_t dur_ns_inflated);
void      cipher_straggler_clear_injection(void);
void      cipher_straggler_report(void);

#ifdef __cplusplus
}
#endif
