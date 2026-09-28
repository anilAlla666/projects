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
#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

#include "cipher_v2_internal.h"
#include "cipher_rt_volt.h"
#include "cipher_rt_matmul_dispatch.h"
#include "cipher_rt_marlin.h"
#include "cipher_rt_fp8.h"      /* D.9 — FP8 actuator (env-gated CIPHER_FP8=on) */
#include "cipher_rt_koopman.h"  /* W14 Step 2 B.2 — Koopman engine (env-gated CIPHER_KOOPMAN=1) */
#include "cipher_rt_attn_dispatch.h"
#include "cipher_rt_audit.h"
#include "cipher_rt_got_patch.h"
#include "cipher_rt_green_ctx.h"
#include "cipher_rt_commit.h"   /* W7-9 Step 2: COMMIT atomic state-transition primitive */
#include "cipher_stream_resolver.h" /* W7-9 Step 5: multi-tenant CUDA-stream resolver */
#include "cipher_rt_ring_write.h" /* W10-12 Step 1: RING_WRITE producer substrate */
#include "cipher_rt_remember_consumer.h" /* W14 Step 3 S3.B1: REMEMBER consumer (env-gated CIPHER_REMEMBER=1) */
#include "cipher_rt_counter_dump.h" /* V1 Phase A.1: substrate counter dump (env-gated CIPHER_RT_COUNTER_DUMP_PATH) */
#include "cipher_rt_dlsym_hook.h"   /* F-B.3.6.1 dlsym/dlvsym interception (cuBLAS-Lt variant coverage) */
#include "cipher_rt_intercept.h"    /* V1 Phase B B.6''.9.8.1 may13 driver-API intercept port */
#include "cipher_workload_detect.h" /* K.1: workload classifier (architectural keystone) */

extern int cipher_rt_attn_test_actuator_init(void);
extern int cipher_rt_cublaslt_variants_init(void);  /* F-B.3.6.1 Path B' trampoline init */
extern int cipher_rt_machete_intercept_init(void);  /* W.2 (2026-05-27) Sub-step 7 */
extern int cipher_rt_attn_6pattern_init(void);       /* W.5 (2026-05-28) 6-pattern attn */
/* CP 2.5 — GOT-patch registration entrypoints (defined in the substrates). */
extern void cipher_rt_cublas_shim_register_got(void);
extern void cipher_rt_attn_register_got(void);

static pthread_once_t cipher_v2_init_once = PTHREAD_ONCE_INIT;

static void cipher_v2_init_body(void)
{
	cipher_dbg("init body running");
	(void)cipher_v2_tenant_register();   /* Phase 2 path (ioctl nr 1) */
	(void)cipher_rt_green_ctx_cp54_init();/* CP 5.4 Step 1.3 — CIPHER_CP54_ALLOCATE
	                                       * on this (long-lived init) thread; Q4 */
	(void)cipher_rt_smp_init();          /* Phase 4 T4.2.3 SM_PACKER (counters) */
	(void)cipher_rt_pr_init();           /* Phase 4 T4.2.2 partition router */
	(void)cipher_v2_cupti_init();        /* Phase 3 Task 5 (ioctl nr 7) */
	(void)cipher_workload_detect_init(); /* K.1 keystone: classifier substrate; observe_* wired downstream */
	(void)cipher_rt_volt_init();         /* Phase 4 T4.3.1 VOLT DVFS */
	(void)cipher_rt_matmul_dispatch_init(); /* Phase 4.5 matmul substrate */
	(void)cipher_rt_marlin_init();       /* Phase 4.5.2 Marlin actuator (env-gated) */
	(void)cipher_rt_fp8_init();          /* D.9 FP8 actuator (env-gated CIPHER_FP8; priority 20, after Marlin) */
	(void)cipher_rt_machete_intercept_init(); /* W.2 (2026-05-27) Sub-step 7 Machete GOT-patch */
	(void)cipher_rt_attn_6pattern_init();     /* W.5 (2026-05-28) 6-pattern attention intercept */
	(void)cipher_rt_koopman_init();      /* W14 Step 2 B.2 Koopman actuator (env-gated CIPHER_KOOPMAN=1) */
	(void)cipher_rt_attn_dispatch_init();/* Phase 4.6.1 attention substrate */
	(void)cipher_rt_attn_test_actuator_init(); /* Phase 4.6.1 smoke actuator (env-gated) */
	(void)cipher_rt_audit_init();        /* Fusion Op 9 AUDIT (env-gated) */
	(void)cipher_rt_commit_init();       /* W7-9 Step 2 §4.8 COMMIT atomic primitive */
	(void)cipher_stream_resolver_init(); /* W7-9 Step 5 multi-tenant resolver */
	(void)cipher_rt_ring_write_init();   /* W10-12 Step 1 §4.9 RING_WRITE producer */
	(void)cipher_rt_remember_consumer_init(); /* W14 Step 3 S3.B1 REMEMBER consumer (env-gated CIPHER_REMEMBER=1) */
	(void)cipher_rt_counter_dump_init(); /* V1 Phase A.1 substrate counter dump (env-gated CIPHER_RT_COUNTER_DUMP_PATH) */

	/* CP 2.5 — replace LD_PRELOAD interposition with GOT patching.
	 * Register the cuBLAS-GEMM and ATen-SDPA trampolines, then patch every
	 * loaded module's GOT so those calls route through the substrates
	 * without libcipher_rt being LD_PRELOAD'd. Runs last: it depends on the
	 * substrates above being initialized. */
	cipher_rt_cublas_shim_register_got();
	cipher_rt_attn_register_got();

	/* F-B.3.6.1 dlsym hook + cuBLAS-Lt variant trampolines. Register
	 * variant trampolines into the dlsym registry FIRST, then init the
	 * hook (which GOT-patches dlsym + dlvsym). After this, any torch
	 * dlsym(handle, "cublasLtSSSMatmul") returns CIPHER's trampoline.
	 * Order matters: variant trampolines populate the registry; the
	 * hook reads from it. Hook init must follow registry population. */
	(void)cipher_rt_cublaslt_variants_init();
	(void)cipher_rt_dlsym_hook_init();

	(void)cipher_rt_got_patch_init();

	/* V1 Phase B B.6''.9.8.1: may13 driver-API intercept port. Must follow
	 * cipher_rt_got_patch_init() so existing cuBLAS GOT slots are already
	 * cipher_rt_phase4-canonical; may13 patcher only touches its OWN
	 * symbol set (cuGetProcAddress + all cuLaunchKernel/cudaLaunchKernel
	 * variants + module/library load + __cudaRegister*). cuBLAS/Lt/NCCL
	 * dropped from may13 g_patches[]; non-overlapping with phase4. */
	cipher_rt_intercept_init();
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

/* V1 Phase A.1 substrate-side worker-init for LD_PRELOAD-only deployment.
 *
 * Goal 5 contract per Anil 2026-05-26 lock requires that the customer set
 * only LD_PRELOAD=libcipher_rt.so (or CUDA_INJECTION64_PATH equivalent) with
 * no plugin install. Before Phase A, the cipher_vllm_plugin general_plugins
 * entry point was the trigger that loaded libcublas and called
 * InitializeInjection2 in the vLLM V1 EngineCore worker subprocess. This
 * constructor moves that trigger into libcipher_rt.so itself.
 *
 * Mechanism: __attribute__((constructor)) runs at libcipher_rt.so dlopen
 * time. LD_PRELOAD causes every newly-forked process (including vLLM's
 * EngineCore worker subprocess) to dlopen this .so as part of process init,
 * so the constructor fires automatically per process.
 *
 * Force-dlopen sequence mirrors the plugin pattern at
 * cipher_vllm_plugin/cipher_vllm_kv.py:675-686. RTLD_LAZY + RTLD_GLOBAL.
 * Idempotency is provided by pthread_once inside cipher_v2_init_body and by
 * dlopen's own ref-count (re-load of an already-mapped SONAME is a no-op
 * with respect to the address space).
 *
 * Anil 2026-05-26 R-A.1 caveat: if constructor ordering is mechanically
 * fragile (libcublas race with torch.cuda.init, or torch import lock
 * contention), DO NOT push through silently. The Branch B fallback is a
 * cuInit-wrapper that interposes via LD_PRELOAD symbol resolution; surfacing
 * before switching is required.
 *
 * Compose with existing init paths:
 *   - CUDA_INJECTION64_PATH set:  CUDA driver auto-invokes InitializeInjection2
 *     at cuInit. The constructor also calls InitializeInjection2 at LD_PRELOAD
 *     load time. pthread_once inside cipher_v2_init_body short-circuits the
 *     second call. No double-init.
 *   - cipher_vllm_plugin still installed (dev / debug runs): plugin's
 *     _install_cipher_rt_got_patches calls InitializeInjection2 too. Same
 *     pthread_once short-circuit. No conflict. Plugin becomes OPTIONAL at
 *     Phase A close, fully deprecated at Phase B close.
 *   - Pure LD_PRELOAD only (Phase A target case): this constructor is the
 *     only InitializeInjection2 trigger. Worker GOT patches install at
 *     process init.
 */
static void __attribute__((constructor)) cipher_rt_auto_init_worker(void)
{
	static const char *libcublas_sonames[] = {
		"libcublas.so.12", "libcublas.so.11", "libcublas.so", NULL
	};
	static const char *libcudnn_sonames[] = {
		"libcudnn.so.9", "libcudnn.so.8", "libcudnn.so", NULL
	};
	int i;

	/* CIPHER_RT_DISABLE_AUTO_INIT escape hatch for debugging. Set this env
	 * to opt out of constructor init (e.g., if A.1 smoke surfaces ordering
	 * issues in a specific build). Goal 5 contract not violated by env-gate
	 * existence; gate is debug-only, default behavior is constructor fires. */
	if (getenv("CIPHER_RT_DISABLE_AUTO_INIT")) {
		cipher_log("auto-init: SKIPPED via CIPHER_RT_DISABLE_AUTO_INIT");
		return;
	}

	/* Force-load libcublas. Plugin pattern at cipher_vllm_kv.py:675-680. */
	for (i = 0; libcublas_sonames[i]; i++) {
		if (dlopen(libcublas_sonames[i], RTLD_LAZY | RTLD_GLOBAL))
			break;
	}

	/* Force-load libcudnn. Plugin pattern at cipher_vllm_kv.py:681-686. */
	for (i = 0; libcudnn_sonames[i]; i++) {
		if (dlopen(libcudnn_sonames[i], RTLD_LAZY | RTLD_GLOBAL))
			break;
	}

	/* Trigger full substrate init via the standard InitializeInjection2
	 * entry point. pthread_once inside cipher_v2_init_body makes this
	 * idempotent: subsequent calls (from CUDA driver under
	 * CUDA_INJECTION64_PATH, or from a still-installed plugin) short-circuit. */
	(void)InitializeInjection2();

	cipher_log("auto-init: V1 Phase A constructor complete pid=%d", (int)getpid());
}
