/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_cublaslt_variants.c -- F-B.3.6.1 Path B' assembly trampolines
 * for the 17 dtype-specialized cuBLAS-Lt matmul variants that libcublasLt
 * exports but cublasLt.h does NOT publicly declare.
 *
 * Variants: ACC, BII, BSB, BSS, CCC, DDD, HHH, HSH, HSS, KCC, KCK, SSS,
 * TSS, TST, VCC, VCV, ZZZ. Modern torch routes default fp32/fp16/bf16/
 * tf32 matmul through these specialized entries via dlsym at runtime;
 * the umbrella cublasLtMatmul intercept is bypassed.
 *
 * Trampoline pattern (x86-64 SysV ABI):
 *
 *   cipher_lt_<V>_trampoline:
 *       lock incq g_lt_<V>_calls(%rip)
 *       jmpq *g_real_lt_<V>(%rip)
 *
 * Arg registers (rdi/rsi/rdx/rcx/r8/r9 + stack) pass through unchanged
 * from caller to real fn. No signature decoding required. The lock-prefixed
 * incq makes the counter SMP-safe without ldrex/strex equivalents (x86-64
 * cache coherency does the work). The jmpq is a tail call -- the real
 * fn's return goes directly back to the original caller, bypassing the
 * trampoline's stack frame entirely.
 *
 * Telemetry is per-variant (16 separate counters) for diagnostics +
 * dispatch-routing decisions; cipher_rt_cublaslt_shim_calls() aggregates
 * across all variants (extends the existing umbrella cublasLtMatmul
 * counter from cipher_rt_cublas_shim.c).
 *
 * Init: cipher_rt_cublaslt_variants_init() resolves each real fn via
 * dlvsym (libcublasLt.so.13 + fallback .so.12) and registers each
 * trampoline with the dlsym hook so torch's runtime dlsym() returns the
 * trampoline pointer.
 *
 * v1 scope: telemetry + passthrough. Actuator dispatch on cuBLAS-Lt
 * variants requires decoding opaque cublasLtMatrixLayout_t descriptors
 * via cublasLtMatrixLayoutGetAttribute; lands in B.6''.3.6 Step 4
 * (separate file cipher_rt_cublaslt_layout.c + variant-specific full C
 * shims that supersede selected trampolines).
 */
#define _GNU_SOURCE
#include "cipher_rt_dlsym_hook.h"
#include "cipher_v2_internal.h"
#include "cipher_rt_matmul_dispatch.h"   /* D.10 try_actuators */
#include "cipher_rt_cublaslt_layout.h"   /* D.10 descriptor decode */

#include <dlfcn.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>                        /* getenv (D.10 CIPHER_LT_ROUTE gate) */

/* 17 variant cell symbols. Each cell is one (counter, real-fn ptr) pair
 * with naked trampoline. The trampoline references its own counter +
 * real-fn ptr via %rip-relative addressing; no register clobbering.
 *
 * Convention: counter is an atomic_ulong (sequential consistency by
 * default on x86; explicit `lock incq` matches that). Real-fn ptr is a
 * plain void* updated only at init time before any call site can fire.
 *
 * GCC emits the symbol with name from the function declaration; we use
 * __attribute__((naked)) to suppress the function prologue/epilogue and
 * write the entire body in inline asm. used + visibility=default keeps
 * the symbol exported so dlsym hook can substitute its address. */

/* static globals so PC-rel addressing in the trampoline works in a -fPIC
 * shared lib without GOT fixup. Trampoline itself stays external (default
 * visibility) so the dlsym hook can return its address. */
#define CIPHER_LT_VARIANT(VAR) \
	static atomic_ulong g_lt_##VAR##_calls; \
	static void *g_real_lt_##VAR; \
	__attribute__((naked, used, visibility("default"))) \
	void cipher_lt_##VAR##_trampoline(void) \
	{ \
		__asm__ volatile( \
			"lock incq g_lt_" #VAR "_calls(%%rip)\n\t" \
			"jmpq *g_real_lt_" #VAR "(%%rip)\n\t" \
			::: "memory"); \
	}

CIPHER_LT_VARIANT(ACC)
CIPHER_LT_VARIANT(BII)
CIPHER_LT_VARIANT(BSB)
CIPHER_LT_VARIANT(BSS)
CIPHER_LT_VARIANT(CCC)
CIPHER_LT_VARIANT(DDD)
CIPHER_LT_VARIANT(HHH)
CIPHER_LT_VARIANT(HSH)
CIPHER_LT_VARIANT(HSS)
CIPHER_LT_VARIANT(KCC)
CIPHER_LT_VARIANT(KCK)
CIPHER_LT_VARIANT(SSS)
CIPHER_LT_VARIANT(TSS)
CIPHER_LT_VARIANT(TST)
CIPHER_LT_VARIANT(VCC)
CIPHER_LT_VARIANT(VCV)
CIPHER_LT_VARIANT(ZZZ)

#undef CIPHER_LT_VARIANT

/* ── D.10 LT-ROUTE: full C shims that supersede selected naked trampolines ──
 * Same 16-arg cublasLtMatmul ABI as the typed variants. Gated CIPHER_LT_ROUTE;
 * OFF ⇒ count + passthrough to the real fn = byte-identical to the trampoline
 * (vanilla). ON ⇒ decode descriptors → actuator chain; on any decode failure or
 * no actuator handling, pass through (Mem #11: never route on a bad decode). */
typedef int (*cipher_lt_matmul_fn)(
	void *lightHandle, void *computeDesc, const void *alpha,
	const void *A, void *Adesc, const void *B, void *Bdesc, const void *beta,
	const void *C, void *Cdesc, void *D, void *Ddesc,
	const void *algo, void *workspace, unsigned long workspaceSizeInBytes,
	void *stream);

static int lt_route_enabled(void)
{
	static int cached = -1;
	if (cached < 0) {
		const char *e = getenv("CIPHER_LT_ROUTE");
		cached = (e && e[0] == '1') ? 1 : 0;
	}
	return cached;
}

/* One routed shim per selected variant: own real-fn ptr + counter. */
#define CIPHER_LT_ROUTED_SHIM(VAR) \
	static int cipher_lt_##VAR##_full_shim( \
		void *lh, void *cd, const void *al, \
		const void *A, void *Ad, const void *B, void *Bd, const void *be, \
		const void *C, void *Cd, void *D, void *Dd, \
		const void *algo, void *ws, unsigned long wss, void *st) \
	{ \
		atomic_fetch_add(&g_lt_##VAR##_calls, 1); \
		cipher_lt_matmul_fn real = (cipher_lt_matmul_fn)g_real_lt_##VAR; \
		if (!real) return 15; /* CUBLAS_STATUS_NOT_SUPPORTED */ \
		if (lt_route_enabled()) { \
			struct cipher_rt_matmul_call call; \
			if (cipher_rt_cublaslt_decode(cd, A, Ad, B, Bd, C, Cd, D, Dd, \
			                              al, be, st, &call) == 0) { \
				int status = 0; \
				if (cipher_rt_matmul_try_actuators(&call, &status) \
				    == CIPHER_RT_MATMUL_HANDLED) \
					return status; /* actuator substituted into D */ \
			} \
		} \
		return real(lh, cd, al, A, Ad, B, Bd, be, C, Cd, D, Dd, \
		            algo, ws, wss, st); \
	}

/* Routed set (verified via CUBLASLT_LOG_LEVEL=4): HSH = fp16 hot path
 * (Mistral/Llama/Qwen/TinyLlama/phi loaded fp16). bf16/tf32 variants added
 * once confirmed per-family. */
CIPHER_LT_ROUTED_SHIM(HSH)

#undef CIPHER_LT_ROUTED_SHIM

/* Variant descriptor for init-time lookup + dispatch registration. */
struct lt_variant {
	const char *symname;
	void      **real_slot;
	void       *trampoline;
	atomic_ulong *counter;
	void       *routed_shim;  /* D.10: non-NULL ⇒ register this instead of the trampoline */
};

#define CIPHER_LT_VAR_ENTRY(VAR) { \
	"cublasLt" #VAR "Matmul", \
	&g_real_lt_##VAR, \
	(void *)cipher_lt_##VAR##_trampoline, \
	&g_lt_##VAR##_calls, \
	NULL, \
}

/* D.10: routed variant — registers the full shim instead of the naked trampoline. */
#define CIPHER_LT_VAR_ENTRY_ROUTED(VAR) { \
	"cublasLt" #VAR "Matmul", \
	&g_real_lt_##VAR, \
	(void *)cipher_lt_##VAR##_trampoline, \
	&g_lt_##VAR##_calls, \
	(void *)cipher_lt_##VAR##_full_shim, \
}

static struct lt_variant g_variants[] = {
	CIPHER_LT_VAR_ENTRY(ACC),
	CIPHER_LT_VAR_ENTRY(BII),
	CIPHER_LT_VAR_ENTRY(BSB),
	CIPHER_LT_VAR_ENTRY(BSS),
	CIPHER_LT_VAR_ENTRY(CCC),
	CIPHER_LT_VAR_ENTRY(DDD),
	CIPHER_LT_VAR_ENTRY(HHH),
	CIPHER_LT_VAR_ENTRY_ROUTED(HSH),
	CIPHER_LT_VAR_ENTRY(HSS),
	CIPHER_LT_VAR_ENTRY(KCC),
	CIPHER_LT_VAR_ENTRY(KCK),
	CIPHER_LT_VAR_ENTRY(SSS),
	CIPHER_LT_VAR_ENTRY(TSS),
	CIPHER_LT_VAR_ENTRY(TST),
	CIPHER_LT_VAR_ENTRY(VCC),
	CIPHER_LT_VAR_ENTRY(VCV),
	CIPHER_LT_VAR_ENTRY(ZZZ),
};

#undef CIPHER_LT_VAR_ENTRY

#define N_VARIANTS ((int)(sizeof(g_variants) / sizeof(g_variants[0])))

static void *resolve_lt_real(const char *sym)
{
	void *h = dlopen("libcublasLt.so.13", RTLD_NOW | RTLD_NOLOAD);
	if (!h) h = dlopen("libcublasLt.so.13", RTLD_NOW);
	if (!h) {
		h = dlopen("libcublasLt.so.12", RTLD_NOW | RTLD_NOLOAD);
		if (!h) h = dlopen("libcublasLt.so.12", RTLD_NOW);
	}
	if (!h) return NULL;
	void *p = dlvsym(h, sym, "libcublasLt.so.13");
	if (!p) p = dlvsym(h, sym, "libcublasLt.so.12");
	if (!p) p = dlsym(h, sym);
	return p;
}

int cipher_rt_cublaslt_variants_init(void)
{
	int i, resolved = 0, routed = 0;

	(void)cipher_rt_cublaslt_layout_init();  /* D.10: resolve descriptor-decode APIs */

	for (i = 0; i < N_VARIANTS; i++) {
		void *real = resolve_lt_real(g_variants[i].symname);
		if (real) {
			*g_variants[i].real_slot = real;
			/* D.10: register the full routing shim if this variant has one
			 * (it self-gates on CIPHER_LT_ROUTE; OFF == trampoline behavior),
			 * else the telemetry-only naked trampoline. */
			void *hook = g_variants[i].routed_shim
			           ? g_variants[i].routed_shim
			           : g_variants[i].trampoline;
			if (g_variants[i].routed_shim) routed++;
			(void)cipher_rt_dlsym_register(g_variants[i].symname, hook);
			resolved++;
		} else {
			cipher_dbg("LT-VARIANT: '%s' not resolved (skipping)",
			           g_variants[i].symname);
		}
	}
	cipher_log("LT-VARIANT: %d/%d armed (%d routed full-shim, route=%s)",
	           resolved, N_VARIANTS, routed,
	           lt_route_enabled() ? "ON" : "OFF");
	return resolved;
}

unsigned long cipher_rt_cublaslt_variants_aggregate(void)
{
	unsigned long sum = 0;
	int i;
	for (i = 0; i < N_VARIANTS; i++)
		sum += atomic_load(g_variants[i].counter);
	return sum;
}

/* Per-variant counter accessor (diagnostic; cipher-platform verify and
 * test harnesses may read these for per-dtype telemetry). Returns 0 for
 * an unknown variant name. */
unsigned long cipher_rt_cublaslt_variant_calls(const char *variant_name)
{
	int i;
	if (!variant_name) return 0;
	for (i = 0; i < N_VARIANTS; i++) {
		const char *s = g_variants[i].symname;
		/* Match suffix: cublasLt<VAR>Matmul -- check VAR substring. */
		if (s && __builtin_strstr(s, variant_name) == s + 8 &&
		    s[8 + __builtin_strlen(variant_name)] == 'M')
			return atomic_load(g_variants[i].counter);
	}
	return 0;
}
