/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_inject.c -- CUDA injection entrypoints.
 *
 * The CUDA driver, on cuInit(), checks CUDA_INJECTION64_PATH. If set,
 * it dlopens that .so and dlsyms one of:
 *   InitializeInjection(void *pfnGetExportTable)
 *   InitializeInjection2(void)
 * Either signature is accepted by different driver builds; we expose
 * both and route them to the same one-shot init. Returning non-zero
 * tells the driver "OK, proceed with cuInit". Returning zero blocks
 * CUDA from initializing; we never want that for tenant registration
 * failure -- the workload should run whether or not we can stamp the
 * tenant.
 */
#include <pthread.h>

#include "cipher_v2_internal.h"
#include "cipher_rt_volt.h"
#include "cipher_rt_matmul_dispatch.h"
#include "cipher_rt_marlin.h"
#include "cipher_rt_attn_dispatch.h"
#include "cipher_rt_audit.h"
#include "cipher_rt_got_patch.h"

extern int cipher_rt_attn_test_actuator_init(void);
/* CP 2.5 — GOT-patch registration entrypoints (defined in the substrates). */
extern void cipher_rt_cublas_shim_register_got(void);
extern void cipher_rt_attn_register_got(void);

static pthread_once_t cipher_v2_init_once = PTHREAD_ONCE_INIT;

static void cipher_v2_init_body(void)
{
	cipher_dbg("init body running");
	(void)cipher_v2_tenant_register();   /* Phase 2 path (ioctl nr 1) */
	(void)cipher_rt_arb_init();          /* Phase 4 T4.2.3 ARBITRATE (opens fd) */
	(void)cipher_rt_smp_init();          /* Phase 4 T4.2.3 SM_PACKER (counters) */
	(void)cipher_rt_pr_init();           /* Phase 4 T4.2.2 partition router */
	(void)cipher_v2_cupti_init();        /* Phase 3 Task 5 (ioctl nr 7) */
	(void)cipher_rt_volt_init();         /* Phase 4 T4.3.1 VOLT DVFS */
	(void)cipher_rt_matmul_dispatch_init(); /* Phase 4.5 matmul substrate */
	(void)cipher_rt_marlin_init();       /* Phase 4.5.2 Marlin actuator (env-gated) */
	(void)cipher_rt_attn_dispatch_init();/* Phase 4.6.1 attention substrate */
	(void)cipher_rt_attn_test_actuator_init(); /* Phase 4.6.1 smoke actuator (env-gated) */
	(void)cipher_rt_audit_init();        /* Fusion Op 9 AUDIT (env-gated) */

	/* CP 2.5 — replace LD_PRELOAD interposition with GOT patching.
	 * Register the cuBLAS-GEMM and ATen-SDPA trampolines, then patch every
	 * loaded module's GOT so those calls route through the substrates
	 * without libcipher_rt being LD_PRELOAD'd. Runs last: it depends on the
	 * substrates above being initialized. */
	cipher_rt_cublas_shim_register_got();
	cipher_rt_attn_register_got();
	(void)cipher_rt_got_patch_init();
	/* All return values ignored: any partial failure logs to stderr
	 * and degrades that subsystem. We still return 1 to the driver
	 * so CUDA proceeds with cuInit regardless. */
}

/* x86-64 driver builds that take a function-table pointer. */
int InitializeInjection(void *pfnGetExportTable)
{
	(void)pfnGetExportTable;
	pthread_once(&cipher_v2_init_once, cipher_v2_init_body);
	return 1;
}

/* Newer convention: no-arg form. Some driver builds prefer this name. */
int InitializeInjection2(void)
{
	pthread_once(&cipher_v2_init_once, cipher_v2_init_body);
	return 1;
}
