/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_cuinit_hook.c -- V1 Phase A.1 Branch B cuInit LD_PRELOAD wrapper.
 *
 * Per V1_PHASE_A_SCOPE_LOCK.md §4.A.1 Branch B sub-element + Anil 2026-05-26
 * proceed signal. Interposes the CUDA driver entrypoint `cuInit` via
 * LD_PRELOAD symbol resolution. On first invocation in a process (parent or
 * fork-spawned worker), force-dlopens libcublas/libcudnn, refreshes the
 * substrate GOT patches against the now-loaded module set, then chains to
 * libcuda's real `cuInit` via `dlsym(RTLD_NEXT)`.
 *
 * Why this is the Branch B path (vs Branch A constructor):
 *   - V1 Phase A.1 R-A.1 surfaced 2026-05-26: vLLM V1 forks the EngineCore
 *     worker by default (VLLM_WORKER_MULTIPROC_METHOD=fork). fork() does
 *     NOT re-run constructors; the worker inherits parent's already-mapped
 *     libcipher_rt.so but no constructor re-fire happens in the child.
 *   - At constructor time in the parent, libcublas is NOT yet loaded (only
 *     ~24 base modules). The GOT-patch pass at constructor time finds no
 *     cublas slots to patch (the slot belongs to whoever calls cublasGemmEx,
 *     which is torch loaded later).
 *   - cuInit wrapper fires at the moment the application (parent or worker)
 *     actually initializes CUDA. By that point, torch + libcublas + libcudnn
 *     are loaded. GOT-patch refresh now finds the right caller-side slots
 *     and patches them. cipher_rt_cublas_shim_calls increments on subsequent
 *     cuBLAS calls.
 *
 * Compose with constructor (belt-and-suspenders per Anil 2026-05-26 Item 2 (a)):
 *   - Constructor at LD_PRELOAD load: force-dlopens libs (deterministic
 *     startup ordering), calls InitializeInjection2 to bring up the
 *     substrate. Under CUDA_INJECTION64_PATH path or single-process
 *     non-fork workloads where libcublas IS loaded at constructor time,
 *     constructor's GOT-patch covers the surface.
 *   - cuInit-wrapper at first cuInit in any process: force-dlopens libs
 *     again (idempotent at dynamic linker), refreshes GOT patches against
 *     the current loaded module set. Picks up modules that were not loaded
 *     at constructor time (the load-bearing case for fork-based workers).
 *
 * Fork+pthread_once correctness:
 *   POSIX inherits pthread_once "done" state via fork. If the parent's
 *   cuInit-wrapper fires before fork, its once-token is DONE; the worker
 *   inherits DONE and would short-circuit. We register pthread_atfork to
 *   reset the once-token in the child, ensuring the worker re-runs setup
 *   on its first cuInit. The atfork handler is registered at LD_PRELOAD
 *   load via a small constructor in this file (before any fork can occur).
 *
 * ABI: new T-symbol `cuInit` interposes the CUDA driver entry. ABI-additive
 * per cipher-abi-rule. The symbol name is mandated by libcuda's contract;
 * not CIPHER-namespaced, but that is the standard LD_PRELOAD interposition
 * pattern (the very feature that makes LD_PRELOAD work).
 *
 * Deployment ledger impact (V1_GOAL5_DEPLOYMENT_LEDGER.md): substrate-internal
 * mechanism; no customer-side step added. Goal 5 contract preserved.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "cipher_v2_internal.h"

/* Forward-declare CUresult. Matches /usr/include/cuda.h:5035 signature:
 *   CUresult CUDAAPI cuInit(unsigned int Flags);
 * CUresult is an int enum, CUDAAPI is empty on Linux. We avoid including
 * cuda.h here to keep the TU header-light. */
typedef int CUresult;
typedef CUresult (*cipher_real_cuInit_fn)(unsigned int);

extern int InitializeInjection2(void);
extern int cipher_rt_got_patch_init(void);

static pthread_once_t g_cuinit_setup_once = PTHREAD_ONCE_INIT;
static cipher_real_cuInit_fn g_real_cuInit;
static _Atomic int g_atfork_registered;

/* atfork child handler: reset the cuInit-wrapper once-token so the next
 * cuInit call in the forked child re-runs setup (GOT-patch refresh in
 * the child's address space). Registered once at LD_PRELOAD load via the
 * small constructor below.
 *
 * Note: g_real_cuInit pointer is ALSO reset because dlsym(RTLD_NEXT) may
 * resolve to a different address in the child's address space (rare on
 * Linux, but safer to re-resolve). */
static void cipher_rt_cuinit_atfork_child(void)
{
	pthread_once_t fresh = PTHREAD_ONCE_INIT;
	g_cuinit_setup_once = fresh;
	g_real_cuInit = NULL;
}

/* LD_PRELOAD-load-time constructor (separate from cipher_inject.c's
 * cipher_rt_auto_init_worker constructor). Single purpose: register the
 * atfork child handler before any fork can occur. Must fire before any
 * potential pthread_create + fork sequence; constructor order across TUs
 * is unspecified but this is independent of other constructors so order
 * is not load-bearing.
 *
 * The CIPHER_RT_DISABLE_CUINIT_HOOK env gate (checked in cuInit body, not
 * here) is the debug escape hatch. It does NOT prevent atfork registration
 * since registration is harmless even when the hook is disabled. */
static void __attribute__((constructor))
cipher_rt_cuinit_register_atfork(void)
{
	int expected = 0;
	if (atomic_compare_exchange_strong(&g_atfork_registered, &expected, 1)) {
		(void)pthread_atfork(NULL, NULL, cipher_rt_cuinit_atfork_child);
	}
}

static void cipher_rt_cuinit_setup(void)
{
	static const char *libcublas_sonames[] = {
		"libcublas.so.12", "libcublas.so.11", "libcublas.so", NULL
	};
	static const char *libcudnn_sonames[] = {
		"libcudnn.so.9", "libcudnn.so.8", "libcudnn.so", NULL
	};
	int i;

	/* Mirror Branch A constructor's force-dlopen pattern. Idempotent at
	 * the dynamic linker level: dlopen of an already-mapped SONAME bumps
	 * the ref count without re-mapping. Under fork inheritance both libs
	 * are typically already mapped; force-dlopen here is a no-op. Under
	 * spawn or in non-vLLM contexts the dlopen ensures the libs are
	 * visible before InitializeInjection2 patches their GOT slots. */
	for (i = 0; libcublas_sonames[i]; i++) {
		if (dlopen(libcublas_sonames[i], RTLD_LAZY | RTLD_GLOBAL))
			break;
	}
	for (i = 0; libcudnn_sonames[i]; i++) {
		if (dlopen(libcudnn_sonames[i], RTLD_LAZY | RTLD_GLOBAL))
			break;
	}

	/* Trigger substrate init. InitializeInjection2's pthread_once handles
	 * the repeat-call case: in parent's first cuInit, this brings up the
	 * full substrate; under fork in worker's first cuInit, pthread_once
	 * is inherited as DONE and this short-circuits (the substrate already
	 * came up in parent before fork). */
	(void)InitializeInjection2();

	/* The load-bearing call under fork: GOT-patch refresh. NOT gated by
	 * pthread_once (cipher_rt_got_patch_init uses an internal mutex but
	 * is otherwise re-entrant per call). In the worker's address space
	 * this walks dl_iterate_phdr to find the cublas/cudnn/torch modules
	 * that vLLM loaded AFTER the parent's constructor fired, and patches
	 * their caller-side GOT slots so subsequent cublasGemmEx calls route
	 * through cipher_rt_cublas_shim. */
	(void)cipher_rt_got_patch_init();

	/* Resolve real cuInit via dlsym(RTLD_NEXT). RTLD_NEXT walks past
	 * libcipher_rt.so in the load order to find the next definition,
	 * which is libcuda's real cuInit. */
	g_real_cuInit = (cipher_real_cuInit_fn)dlsym(RTLD_NEXT, "cuInit");
	if (!g_real_cuInit) {
		cipher_log("cuinit-hook: dlsym(RTLD_NEXT, cuInit) failed pid=%d"
		           " (cuInit will return CUDA_ERROR_NOT_INITIALIZED)",
		           (int)getpid());
		return;
	}

	cipher_log("cuinit-hook: setup complete pid=%d"
	           " (force-dlopen + InitializeInjection2 + GOT-patch refresh"
	           " + real_cuInit resolved)",
	           (int)getpid());
}

/* The interposed cuInit. LD_PRELOAD'd libcipher_rt.so exports this symbol
 * so the dynamic linker resolves the application's cuInit calls to OUR
 * cuInit first. We run setup-once (pthread_once-gated, atfork-reset for
 * fork-spawned workers), then chain to libcuda's real cuInit. */
CUresult cuInit(unsigned int Flags)
{
	/* Escape hatch for debugging (e.g., bisecting whether the wrapper
	 * causes a CUDA-init regression in some workload). Unset by default;
	 * Goal 5 contract not violated by env-gate existence (debug-only). */
	if (getenv("CIPHER_RT_DISABLE_CUINIT_HOOK")) {
		cipher_real_cuInit_fn real =
			(cipher_real_cuInit_fn)dlsym(RTLD_NEXT, "cuInit");
		return real ? real(Flags) : 3 /* CUDA_ERROR_NOT_INITIALIZED */;
	}

	pthread_once(&g_cuinit_setup_once, cipher_rt_cuinit_setup);

	if (!g_real_cuInit)
		return 3; /* CUDA_ERROR_NOT_INITIALIZED */
	return g_real_cuInit(Flags);
}
