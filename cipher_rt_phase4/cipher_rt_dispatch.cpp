/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_dispatch.cpp -- Week 3 Step 1: GEMM dispatch routing scaffolding.
 *
 * Static 16-entry table indexed by OpClass. v1 routes GEMM -> Marlin;
 * everything else -> PASS_THROUGH. CIPHER_DISPATCH_LIVE env-gated so
 * the substrate is observe-only through Steps 1-3.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <atomic>

#include "cipher_v2_internal.h"
#include "cipher_rt_dispatch.h"
#include "cipher_rt_classify_substrate.h"   /* OpClass enum semantics */

namespace {

/* Static dispatch table — 16 slots indexed by op_class.
 *
 * OpClass enum (cipher_classify.hpp):
 *   0=GEMM, 1=ATTENTION, 2=CONVOLUTION, 3=ELEMENTWISE, 4=REDUCTION,
 *   5=MEMCPY_TRANSPOSE, 6=ITERATIVE_CUSTOM, 7..14=reserved,
 *   15=UNCLASSIFIED (observer bucket convention).
 *
 * v1 routes GEMM->Marlin only. Wave 5 W3 narrows the substrate to
 * GEMM-only; attn defers to v1.5. */
const cipher_rt_dispatch_decision g_dispatch_table[16] = {
	{ 0,  CIPHER_RT_DISPATCH_ROUTE_MARLIN, 0, 0 },   /* GEMM */
	{ 1,  CIPHER_RT_DISPATCH_PASS_THROUGH, 0, 0 },   /* ATTENTION (v1.5) */
	{ 2,  CIPHER_RT_DISPATCH_PASS_THROUGH, 0, 0 },   /* CONVOLUTION */
	{ 3,  CIPHER_RT_DISPATCH_PASS_THROUGH, 0, 0 },   /* ELEMENTWISE */
	{ 4,  CIPHER_RT_DISPATCH_PASS_THROUGH, 0, 0 },   /* REDUCTION */
	{ 5,  CIPHER_RT_DISPATCH_PASS_THROUGH, 0, 0 },   /* MEMCPY_TRANSPOSE */
	{ 6,  CIPHER_RT_DISPATCH_PASS_THROUGH, 0, 0 },   /* ITERATIVE_CUSTOM */
	{ 7,  CIPHER_RT_DISPATCH_PASS_THROUGH, 0, 0 },   /* reserved */
	{ 8,  CIPHER_RT_DISPATCH_PASS_THROUGH, 0, 0 },
	{ 9,  CIPHER_RT_DISPATCH_PASS_THROUGH, 0, 0 },
	{ 10, CIPHER_RT_DISPATCH_PASS_THROUGH, 0, 0 },
	{ 11, CIPHER_RT_DISPATCH_PASS_THROUGH, 0, 0 },
	{ 12, CIPHER_RT_DISPATCH_PASS_THROUGH, 0, 0 },
	{ 13, CIPHER_RT_DISPATCH_PASS_THROUGH, 0, 0 },
	{ 14, CIPHER_RT_DISPATCH_PASS_THROUGH, 0, 0 },
	{ 15, CIPHER_RT_DISPATCH_PASS_THROUGH, 0, 0 },   /* UNCLASSIFIED */
};

/* CIPHER_DISPATCH_LIVE cache. -1 = not yet read; 0 = unset/0; 1 = "1". */
std::atomic<int> g_live_cached{-1};

int read_live_env_once(void)
{
	int expected = -1;
	int parsed   = 1;  /* Week 3 Step 4 II-a: default flipped to 1 */
	const char *env = getenv("CIPHER_DISPATCH_LIVE");
	if (env && env[0] == '0' && env[1] == '\0')
		parsed = 0;  /* explicit "0" disables (rollback path) */
	/* Race-tolerant: any concurrent first-readers compute the same
	 * parsed value, so a CAS race is benign. */
	g_live_cached.compare_exchange_strong(expected, parsed);
	return g_live_cached.load();
}

/* Static-init banner at .so load. C++ ctor priority not required;
 * the banner just announces the substrate's existence. The env read
 * happens here to make the LIVE state visible in the load-time log. */
struct DispatchLoadBanner {
	DispatchLoadBanner() {
		int live = read_live_env_once();
		cipher_log("DISPATCH: substrate registered, LIVE=%d "
		           "(1 = classifier-driven routing engaged by default; "
		           "set CIPHER_DISPATCH_LIVE=0 to disable)", live);
	}
};

DispatchLoadBanner g_load_banner;

} /* anonymous namespace */

extern "C" {

int cipher_rt_dispatch_lookup(
	uint8_t                             op_class,
	struct cipher_rt_dispatch_decision *out)
{
	if (!out) return -1;
	if (op_class >= 16) {
		/* Out-of-range op_class still gets a defined PASS_THROUGH
		 * decision (cleanest defensive default) but we return -1
		 * to let the caller log/diagnose. */
		out->op_class      = op_class;
		out->action        = CIPHER_RT_DISPATCH_PASS_THROUGH;
		out->actuator_hint = 0;
		out->reserved      = 0;
		return -1;
	}
	*out = g_dispatch_table[op_class];
	return 0;
}

int cipher_rt_dispatch_is_live(void)
{
	int v = g_live_cached.load();
	if (v < 0) return read_live_env_once();
	return v;
}

} /* extern "C" */
