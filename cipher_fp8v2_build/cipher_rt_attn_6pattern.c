/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_attn_6pattern.c -- W.5 (2026-05-28) FlashAttention 6-pattern intercept.
 *
 * GOT-patch intercept for vLLM 0.21's attention library entry points.
 *
 * W.5 v2 (2026-05-28) self-resolution fix: the trampolines are UNIQUE
 * hidden symbols (cipher_attn_pN_tramp), NOT the mangled target names.
 * v1 defined the trampoline AS the mangled target symbol with default
 * visibility — that shadowed the library's own export, so
 * dlsym(lib_handle, mangled) returned OUR trampoline instead of the real
 * function. The trampoline then tail-jmp'd to itself → unbounded
 * recursion → stack overflow → segfault (observed on E3 cell: attn_p2=1
 * then "Segfault encountered"). Hidden, uniquely-named trampolines do not
 * shadow the lib symbols, so dlsym resolves the real original; the
 * GOT-patcher writes the trampoline ADDRESS (not name) into matching
 * GOT slots, so the patched-name → trampoline redirect still works.
 *
 * 6 patterns enumerated (vLLM 0.21):
 *   P1 FlashAttention-2  flash::run_mha_fwd(Flash_fwd_params&,CUstream*,bool)
 *                        _vllm_fa2_C.abi3.so
 *   P2 FlashAttention-3  run_mha_fwd(Flash_fwd_params&,CUstream*)  [Hopper]
 *                        _vllm_fa3_C.abi3.so
 *   P3 FlashInfer prefill  -- JIT-COMPILED (no static symbol); §D residue
 *   P4 FlashInfer decode   -- JIT-COMPILED (no static symbol); §D residue
 *   P5 FlashMLA          FMHACutlassSM100FwdRun(...)  _flashmla_C.abi3.so
 *   P6 PagedAttention v2 paged_attention_v2(...)      _C.abi3.so
 *
 * v1 scope: intercept + counter + passthrough (byte-identical to baseline,
 * so Memory #11 KL gate trivially holds). Substrate-side substitution
 * kernels per pattern are close-report §D items (v1.x).
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <dlfcn.h>
#include "cipher_v2_internal.h"
#include "cipher_rt_got_patch.h"

/* Mangled target symbol names (what GOT slots reference). */
#define SYM_P1 "_ZN5flash11run_mha_fwdERNS_16Flash_fwd_paramsEP11CUstream_stb"
#define SYM_P2 "_Z11run_mha_fwdR16Flash_fwd_paramsP11CUstream_st"
#define SYM_P5 "_Z22FMHACutlassSM100FwdRunN2at6TensorES0_S0_S0_S0_S0_S0_S0_ifiib"
#define SYM_P6 "_Z18paged_attention_v2RN2at6TensorES1_S1_S1_S1_S1_S1_ldS1_S1_llRKSt8optionalIS0_ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES1_S1_lllll"

/* Resolved original fn ptrs (populated at init). */
void *g_attn_p1_orig = NULL;
void *g_attn_p2_orig = NULL;
void *g_attn_p5_orig = NULL;
void *g_attn_p6_orig = NULL;

/* Per-pattern engagement counters. */
atomic_ulong g_attn_p1_intercepts = 0, g_attn_p1_passthrough = 0;
atomic_ulong g_attn_p2_intercepts = 0, g_attn_p2_passthrough = 0;
atomic_ulong g_attn_p5_intercepts = 0, g_attn_p5_passthrough = 0;
atomic_ulong g_attn_p6_intercepts = 0, g_attn_p6_passthrough = 0;
atomic_int   g_attn_6p_armed = 0;

/* ── Naked GOTPCREL trampolines — UNIQUE HIDDEN symbols ────────────────────
 * Hidden visibility + unique names ⇒ they do NOT shadow the patched library
 * exports, so arm_pattern's dlsym resolves the real original. Each: inc
 * intercepts + passthrough (forward-only in v1), tail-jmp to original.
 * Only r11 (SysV call-clobbered scratch) is touched; arg registers pass
 * through ABI-faithfully for the per-pattern signatures. */

__attribute__((naked, used, visibility("hidden")))
static void cipher_attn_p1_tramp(void)
{
	__asm__ volatile (
		"movq g_attn_p1_intercepts@GOTPCREL(%%rip), %%r11\n\t"
		"lock incq (%%r11)\n\t"
		"movq g_attn_p1_passthrough@GOTPCREL(%%rip), %%r11\n\t"
		"lock incq (%%r11)\n\t"
		"movq g_attn_p1_orig@GOTPCREL(%%rip), %%r11\n\t"
		"movq (%%r11), %%r11\n\t"
		"jmpq *%%r11\n\t"
		: : : "memory"
	);
}

__attribute__((naked, used, visibility("hidden")))
static void cipher_attn_p2_tramp(void)
{
	__asm__ volatile (
		"movq g_attn_p2_intercepts@GOTPCREL(%%rip), %%r11\n\t"
		"lock incq (%%r11)\n\t"
		"movq g_attn_p2_passthrough@GOTPCREL(%%rip), %%r11\n\t"
		"lock incq (%%r11)\n\t"
		"movq g_attn_p2_orig@GOTPCREL(%%rip), %%r11\n\t"
		"movq (%%r11), %%r11\n\t"
		"jmpq *%%r11\n\t"
		: : : "memory"
	);
}

__attribute__((naked, used, visibility("hidden")))
static void cipher_attn_p5_tramp(void)
{
	__asm__ volatile (
		"movq g_attn_p5_intercepts@GOTPCREL(%%rip), %%r11\n\t"
		"lock incq (%%r11)\n\t"
		"movq g_attn_p5_passthrough@GOTPCREL(%%rip), %%r11\n\t"
		"lock incq (%%r11)\n\t"
		"movq g_attn_p5_orig@GOTPCREL(%%rip), %%r11\n\t"
		"movq (%%r11), %%r11\n\t"
		"jmpq *%%r11\n\t"
		: : : "memory"
	);
}

__attribute__((naked, used, visibility("hidden")))
static void cipher_attn_p6_tramp(void)
{
	__asm__ volatile (
		"movq g_attn_p6_intercepts@GOTPCREL(%%rip), %%r11\n\t"
		"lock incq (%%r11)\n\t"
		"movq g_attn_p6_passthrough@GOTPCREL(%%rip), %%r11\n\t"
		"lock incq (%%r11)\n\t"
		"movq g_attn_p6_orig@GOTPCREL(%%rip), %%r11\n\t"
		"movq (%%r11), %%r11\n\t"
		"jmpq *%%r11\n\t"
		: : : "memory"
	);
}

/* Resolve one pattern: dlopen the specific .so + dlsym the mangled name
 * (real original — our trampolines are hidden, no shadow) + register the
 * GOT-patch (mangled name → trampoline address). Returns 1 if armed. */
static int arm_pattern(const char *so_path, const char *mangled,
                       void **orig_slot, void *trampoline, const char *tag)
{
	if (*orig_slot) return 1;   /* already armed */
	void *h = dlopen(so_path, RTLD_LAZY | RTLD_NOLOAD);
	if (!h) return 0;            /* lib not loaded yet — retry later */
	void *orig = dlsym(h, mangled);
	if (!orig) return 0;
	/* Defensive: never accept our own trampoline as the "original". */
	if (orig == trampoline) {
		cipher_log("ATTN-6P: %s dlsym returned trampoline (shadow!) — skip", tag);
		return 0;
	}
	*orig_slot = orig;
	int rc = cipher_rt_got_register(mangled, trampoline, NULL);
	if (rc != 0) {
		cipher_log("ATTN-6P: %s got_register failed rc=%d", tag, rc);
		*orig_slot = NULL;
		return 0;
	}
	cipher_log("ATTN-6P: %s ARMED orig=%p tramp=%p", tag, orig, trampoline);
	return 1;
}

int cipher_rt_attn_6pattern_init(void)
{
	int armed = 0;
	armed += arm_pattern(
		"/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
		SYM_P1, &g_attn_p1_orig, (void *)&cipher_attn_p1_tramp, "P1-FA2");
	armed += arm_pattern(
		"/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so",
		SYM_P2, &g_attn_p2_orig, (void *)&cipher_attn_p2_tramp, "P2-FA3");
	armed += arm_pattern(
		"/usr/local/lib/python3.12/dist-packages/vllm/_flashmla_C.abi3.so",
		SYM_P5, &g_attn_p5_orig, (void *)&cipher_attn_p5_tramp, "P5-FlashMLA");
	armed += arm_pattern(
		"/usr/local/lib/python3.12/dist-packages/vllm/_C.abi3.so",
		SYM_P6, &g_attn_p6_orig, (void *)&cipher_attn_p6_tramp, "P6-PagedAttnV2");
	int prev = atomic_load(&g_attn_6p_armed);
	if (armed > prev) {
		atomic_store(&g_attn_6p_armed, armed);
		(void)cipher_rt_got_patch_apply();
	}
	return armed;
}

/* === Diagnostic accessors === */
unsigned long cipher_rt_attn_p1_intercepts(void) { return atomic_load(&g_attn_p1_intercepts); }
unsigned long cipher_rt_attn_p2_intercepts(void) { return atomic_load(&g_attn_p2_intercepts); }
unsigned long cipher_rt_attn_p5_intercepts(void) { return atomic_load(&g_attn_p5_intercepts); }
unsigned long cipher_rt_attn_p6_intercepts(void) { return atomic_load(&g_attn_p6_intercepts); }
int cipher_rt_attn_6pattern_armed(void) { return atomic_load(&g_attn_6p_armed); }
