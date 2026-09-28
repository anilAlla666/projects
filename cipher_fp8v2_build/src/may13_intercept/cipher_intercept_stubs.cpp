/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_intercept_stubs.cpp -- V1 Phase B B.6''.9.8.2 inert stubs.
 *
 * may13 cipher_intercept_cudart.cpp references 6 helpers from may13's flow
 * substrate (cipher_persist.cpp, cipher_flow_*.cpp) that are NOT ported
 * into cipher_rt_phase4 in this substep (B.6''.9.8 scope is the
 * cuGetProcAddress foundation; flow substrate is a separate v1.x work).
 *
 * These stubs make the helpers null-effect: fingerprint returns 0,
 * try_fast_path returns false (always passthrough), observe is a no-op.
 * The may13 hot path code respects these results and falls through to
 * normal cuLaunchKernel behavior — substrate's own counters fire normally,
 * persist-engine optimization simply does not engage.
 *
 * v1.x port of cipher_persist.cpp + cipher_flow_*.cpp will replace these
 * with the real implementations when CIPHER_PERSIST=on / CIPHER_FLOW=on
 * actuators are scoped in.
 */

#include <stdint.h>
#include <stddef.h>

extern "C" {

uint64_t cipher_persist_fingerprint(const void* /*f*/,
                                    unsigned /*gx*/, unsigned /*gy*/, unsigned /*gz*/,
                                    unsigned /*bx*/, unsigned /*by*/, unsigned /*bz*/,
                                    unsigned /*shared*/)
{
    return 0;
}

bool cipher_persist_try_fast_path(uint64_t /*fp*/) { return false; }

void cipher_persist_observe(uint64_t /*fp*/) {}

void cipher_flow_recorder_observe(void* /*func*/,
                                  uint32_t /*gx*/, uint32_t /*gy*/, uint32_t /*gz*/,
                                  uint32_t /*bx*/, uint32_t /*by*/, uint32_t /*bz*/,
                                  uint32_t /*smem*/, void** /*args*/, void* /*stream*/) {}

void cipher_flow_patterns_check(void* /*func*/,
                                uint32_t /*gx*/, uint32_t /*gy*/, uint32_t /*gz*/,
                                uint32_t /*bx*/, uint32_t /*by*/, uint32_t /*bz*/,
                                uint32_t /*smem*/, void** /*args*/, void* /*stream*/) {}

int cipher_flow_substitute_consider(void* /*func*/,
                                    uint32_t /*gx*/, uint32_t /*gy*/, uint32_t /*gz*/,
                                    uint32_t /*bx*/, uint32_t /*by*/, uint32_t /*bz*/,
                                    uint32_t /*smem*/, void** /*args*/, void* /*stream*/) {
    return 0;
}

} /* extern "C" */
