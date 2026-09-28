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

static pthread_once_t cipher_v2_init_once = PTHREAD_ONCE_INIT;

static void cipher_v2_init_body(void)
{
	cipher_dbg("init body running");
	(void)cipher_v2_tenant_register();
	/* return value ignored: even if tenant registration fails, we
	 * return 1 to the driver below so CUDA proceeds. */
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
