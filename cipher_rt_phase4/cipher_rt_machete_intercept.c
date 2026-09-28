/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_machete_intercept.c -- W.2 (2026-05-27) Sub-step 7
 *
 * GOT-patch intercept for vLLM's Machete GEMM entry:
 *   machete::mm_dispatch(machete::MMArgs)
 *   mangled: _ZN7machete11mm_dispatchENS_6MMArgsE
 *
 * The trampoline is a naked function preserving the SysV x86-64 ABI:
 * MMArgs is passed by value; depending on its size it lands in arg
 * registers (rdi/rsi/rdx/rcx) or via hidden pointer + stack. The
 * trampoline does not touch any arg-passing register — it bumps a
 * counter via r11 (call-clobbered, ABI-safe) and tail-jmps to the
 * resolved original.
 *
 * The original is resolved at init via dlsym(RTLD_NEXT) on first
 * library load. Storing the resolved address in g_machete_original_fn
 * lets the trampoline jmp through it without further indirection.
 *
 * If the symbol resolve fails (vLLM build without Machete, or symbol
 * renamed), the trampoline is not installed; the substrate's substitute
 * counter stays 0; substrate-level detection (g_machete_present) still
 * surfaces engagement evidence.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <dlfcn.h>
#include "cipher_v2_internal.h"
#include "cipher_rt_got_patch.h"

/* Resolved original symbol address (populated at init). */
void *g_machete_original_fn = NULL;

/* Engagement counters. Memory #28 marvel — visible to ctypes probes via the
 * accessor below. */
atomic_int   g_machete_present = 0;
atomic_ulong g_machete_intercepts = 0;     /* trampoline fired (substrate hook) */
atomic_ulong g_machete_substituted = 0;    /* dispatched to Marlin (W.2.x) */
atomic_ulong g_machete_passthrough = 0;    /* forwarded to original Machete */

/* Naked trampoline. Preserves all arg-passing registers (rdi/rsi/rdx/rcx/
 * r8/r9 + xmm0-7) and the stack; uses only r11 (SysV scratch) for the
 * counter increment + load of the original function ptr. Inline-asm
 * sequence:
 *   1. leaq g_machete_intercepts(%rip), %r11
 *   2. lock incq (%r11)                   (atomic ++ on the counter)
 *   3. leaq g_machete_passthrough(%rip), %r11
 *   4. lock incq (%r11)                   (we forward = passthrough)
 *   5. movq g_machete_original_fn(%rip), %r11
 *   6. jmpq *%r11                         (tail-call original; ret addr untouched)
 *
 * The lock-inc-rip-relative pattern is well-defined under PIC; r11 is
 * never an SysV arg register and is call-clobbered, so the caller does
 * not expect it preserved. */
__attribute__((naked, used, visibility("default")))
void _ZN7machete11mm_dispatchENS_6MMArgsE(void)
{
	/* GOTPCREL-relative addressing for PIC shared-library safety.
	 * @GOTPCREL yields the address of the GOT slot holding &symbol; one
	 * indirection gives &symbol, second gives the value. For variables
	 * (g_machete_intercepts etc) we want &symbol -> one deref. For
	 * g_machete_original_fn (which holds the orig fn pointer) we want
	 * the pointer value -> double deref. */
	__asm__ volatile (
		"movq g_machete_intercepts@GOTPCREL(%%rip), %%r11\n\t"
		"lock incq (%%r11)\n\t"
		"movq g_machete_passthrough@GOTPCREL(%%rip), %%r11\n\t"
		"lock incq (%%r11)\n\t"
		"movq g_machete_original_fn@GOTPCREL(%%rip), %%r11\n\t"
		"movq (%%r11), %%r11\n\t"
		"jmpq *%%r11\n\t"
		: : : "memory"
	);
}

/* Init/retry: dlsym Machete entry, save original, register GOT-patch.
 *
 * At cuInit time (substrate init) the vLLM Machete symbol may NOT yet be
 * visible — _C.abi3.so is lazy-loaded by vLLM later when models actually
 * use AWQ/Machete. This function is therefore RETRYABLE: returns early if
 * already armed; otherwise re-attempts dlsym + dlopen + GOT-register.
 *
 * The retry hook is in cipher_workload_observe_launch (called on every
 * kernel launch). One atomic load + branch when armed — sub-nanosecond.
 *
 * Returns 1 if armed (either freshly or previously), 0 otherwise. */
int cipher_rt_machete_intercept_init(void)
{
	if (atomic_load(&g_machete_present)) return 1;

	void *orig = dlsym(RTLD_NEXT, "_ZN7machete11mm_dispatchENS_6MMArgsE");
	if (!orig) {
		const char *candidates[] = {
			"/usr/local/lib/python3.12/dist-packages/vllm/_C.abi3.so",
			"/usr/lib/python3/dist-packages/vllm/_C.abi3.so",
			NULL
		};
		for (int i = 0; candidates[i] && !orig; ++i) {
			void *h = dlopen(candidates[i], RTLD_LAZY | RTLD_NOLOAD);
			if (h) orig = dlsym(h, "_ZN7machete11mm_dispatchENS_6MMArgsE");
		}
	}
	if (!orig) return 0;

	/* Race: another retry tick may have armed in between. CAS-guard. */
	int expected = 0;
	if (!atomic_compare_exchange_strong(&g_machete_present, &expected, 1))
		return 1;

	g_machete_original_fn = orig;

	int rc = cipher_rt_got_register("_ZN7machete11mm_dispatchENS_6MMArgsE",
	                                (void *)&_ZN7machete11mm_dispatchENS_6MMArgsE,
	                                NULL);
	if (rc != 0) {
		cipher_log("MACHETE: cipher_rt_got_register failed rc=%d", rc);
		atomic_store(&g_machete_present, 0);
		g_machete_original_fn = NULL;
		return 0;
	}
	/* Force re-scan so the freshly registered patch lands in any
	 * currently-loaded module's GOT (including _C.abi3.so if vLLM has
	 * loaded it in the meantime). */
	(void)cipher_rt_got_patch_apply();
	cipher_log("MACHETE: intercept ARMED (retry-hook) — original=%p trampoline=%p",
	           g_machete_original_fn,
	           (void *)&_ZN7machete11mm_dispatchENS_6MMArgsE);
	return 1;
}

/* === Diagnostic accessors === */
unsigned long cipher_rt_machete_intercepts(void)
{
	return (unsigned long)atomic_load(&g_machete_intercepts);
}
unsigned long cipher_rt_machete_substituted(void)
{
	return (unsigned long)atomic_load(&g_machete_substituted);
}
unsigned long cipher_rt_machete_passthrough(void)
{
	return (unsigned long)atomic_load(&g_machete_passthrough);
}
int cipher_rt_machete_is_present(void)
{
	return atomic_load(&g_machete_present);
}
