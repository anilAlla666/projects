/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_oracle_bridge.cpp -- Week 4 Step 1 thin C-callable
 * bridge over cipher_oracle.cpp.
 *
 * cipher_oracle.h transitively pulls cipher_classify.hpp which uses
 * namespace + enum class + <atomic>; cipher_rt_classify_observer.c
 * is a C TU and cannot include it directly. This bridge owns the
 * BSS-resident CipherOracleState + lazy CAS init flag and exposes
 * a C surface (cipher_rt_oracle_bridge.h).
 *
 * cipher_oracle_init(state, NULL, NULL) is pure-function safe per
 * oracle.cpp:80-119 inspection (memset + config defaults + sigma
 * init + cipher_struct_lookup_init (pure: cache clear + pre-warm
 * + fprintf) + stderr report). No CUDA / NVML / kmod calls; safe
 * to run under cipher_rt_phase4's deferred-init model.
 */
#include "cipher_rt_oracle_bridge.h"
#include "may13/cipher_oracle.h"

#include <atomic>

namespace {

CipherOracleState g_rt_oracle{};
std::atomic<int>  g_inited{0};

}  /* anonymous namespace */

extern "C" {

void cipher_rt_oracle_bridge_init_lazy(void)
{
	if (g_inited.load(std::memory_order_acquire) != 0)
		return;

	int expected = 0;
	if (g_inited.compare_exchange_strong(expected, 1)) {
		/* Winner runs init; losers skip. compare_exchange_strong
		 * is acq_rel by default; subsequent loads observe the
		 * full write to g_rt_oracle. */
		cipher_oracle_init(&g_rt_oracle, nullptr, nullptr);
	}
}

int cipher_rt_oracle_bridge_decide(uint8_t op_class, uint8_t confidence)
{
	if (g_inited.load(std::memory_order_acquire) == 0)
		return 0;  /* not yet initialized: deny */

	CipherOracleQuery q{};
	q.op_class     = op_class;
	q.confidence   = confidence;
	q.kernel_name  = nullptr;
	q.is_backward  = false;
	q.is_optimizer = false;
	/* layer_idx / total_layers default-zero is fine; Marlin's own
	 * gate at L100-114 enforces the shape/min-confidence filters. */

	CipherOracleResult r = cipher_oracle_decide(&g_rt_oracle, &q);
	return (r.decision == CIPHER_ORACLE_PERMIT) ? 1 : 0;
}

}  /* extern "C" */
