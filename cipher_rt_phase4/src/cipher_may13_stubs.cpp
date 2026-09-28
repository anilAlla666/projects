// H1 stub per A4 closure adjudication (WEEK_2_STEP_2_A4_ORPHAN_CONSUMER.md).
//
// cipher_intercept_stats() returns counters for F1 hook activity
// (LD_PRELOAD link-order interposition implemented in
// cipher-may13-evidence/src/cipher_intercept.cpp). cipher_rt_phase4
// uses GOT patching (cipher_rt_got_patch.c) per CP 2.5 D2(iii), not
// F1 hooks, so F1 activity is zero by design.
//
// This stub returns truthful zero counters. The sole consumer is
// cipher_runtime.cpp::cipher_report() — a PURE-LOG diagnostic dumper
// with defensive zero-guards on the two division sites; zero counters
// produce coherent output ("[F1] Intercept Stats: 0 ...").
//
// If a future revision installs F1 hooks (none planned in v1.2.2),
// this stub must be replaced with the real implementation.

#include "may13/cipher_intercept.h"

namespace {
const CipherInterceptStats g_zero_intercept_stats = {0};
}

extern "C" const CipherInterceptStats* cipher_intercept_stats(void) {
    return &g_zero_intercept_stats;
}
