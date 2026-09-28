/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_oracle_bridge.h -- Week 4 Step 1 C-callable surface
 * over cipher_oracle (C++ TU). Hides the namespace/enum-class
 * machinery in cipher_classify.hpp so C consumers (observer.c)
 * can call oracle_decide() without going through C++.
 */
#ifndef CIPHER_RT_ORACLE_BRIDGE_H
#define CIPHER_RT_ORACLE_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Lazy init via atomic CAS. Idempotent; safe to call from any
 * thread. cipher_oracle_init(state, NULL, NULL) is pure-function
 * safe per inspection of oracle.cpp:80-119 + structural_lookup.cpp
 * :169-188 (no CUDA/NVML/kmod; only memset + config defaults +
 * cache pre-warm + stderr report). */
void cipher_rt_oracle_bridge_init_lazy(void);

/* Returns 1 = PERMIT, 0 = DENY. Builds CipherOracleQuery from the
 * two CUPTI-known fields; passes through cipher_oracle_decide()'s
 * 5-gate logic (warmup, confidence, structural, EMA, N<=4 rule).
 * Hot path: <50ns per oracle.h:174. */
int  cipher_rt_oracle_bridge_decide(uint8_t op_class,
				    uint8_t confidence);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_ORACLE_BRIDGE_H */
