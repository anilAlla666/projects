// =============================================================================
// CIPHER — central per-op activity counters
// One atomic counter per logical op. Ops increment via cipher_op_inc(OP_FOO);
// Stress-test orchestrator dumps the table per process at exit so the report
// can honestly say which ops fired in which tenant.
//
// The counter array lives in libcipher_rt.so. Code in libcipher_hook.so calls
// cipher_op_inc() through a weak symbol — when rt is loaded (always for
// CIPHER stress tests via ctypes.CDLL), the weak ref binds to the strong
// impl. When rt isn't loaded, the call is a no-op.
// =============================================================================
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum CipherOpId {
    OP_CLASSIFY = 0,
    OP_PREDICT,
    OP_RING_WRITE,
    OP_FLOW_RECORD,
    OP_SPECULATE_CHECK,
    OP_FLOW_MATCH,
    OP_GUARD,
    OP_DETERMINISM,
    OP_SUBSTITUTE_FP8,
    OP_SUBSTITUTE_KOOPMAN,
    OP_SUBSTITUTE_MARLIN,
    OP_FLOW_SUBSTITUTE,
    OP_FUSE,
    OP_NCCL_TUNER,
    OP_PERSIST,
    OP_THERMOSTAT,
    OP_FAIRNESS,
    OP_ARBITRATE,
    OP_PIPELINE,
    OP_CONTINUITY,
    OP_RECEIPT,
    OP_CARBON,
    OP_TRACE,
    OP_COMPLY,
    OP_AUDIT,
    OP_LOOP,
    OP_GRAPH_ENGINE,
    OP_REMEMBER,
    OP_ADAPT,
    OP_SPECULATE_WRITE,
    OP_VALIDATE,
    OP_TOPOLOGY,
    OP_WORKLOAD_OBSERVE,
    OP_COUNT
};

void        cipher_op_inc(int op);
uint64_t    cipher_op_counter_get(int op);
const char* cipher_op_name(int op);
void        cipher_op_counters_dump_json(const char* path);

#ifdef __cplusplus
}
#endif
