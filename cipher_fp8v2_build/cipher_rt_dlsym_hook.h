/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_dlsym_hook.h -- F-B.3.6.1 dlsym/dlvsym interception.
 *
 * Substrate cannot reach symbols resolved via runtime dlsym() into a specific
 * library handle (e.g. torch dlsym-resolving cublasLtSSSMatmul out of
 * libcublasLt.so.13). GOT patching catches only DT_NEEDED-linked refs. This
 * hook intercepts dlsym + dlvsym at the libdl boundary, consults a registry
 * of (symbol-name -> CIPHER-shim-address) mappings, and substitutes the
 * shim's address when caller dlsym's a known symbol. Non-matched lookups
 * fall through to real dlsym unchanged.
 *
 * Mechanism:
 *  - cipher_rt_got_register("dlsym", cipher_rt_dlsym_impl, NULL)
 *  - cipher_rt_got_register("dlvsym", cipher_rt_dlvsym_impl, NULL)
 *  - Both impls share a TLS recursion guard so the impl can safely call
 *    real dlsym to do its own resolution + initial registry lookups.
 *
 * x86-64 only.
 */
#ifndef CIPHER_RT_DLSYM_HOOK_H
#define CIPHER_RT_DLSYM_HOOK_H

#ifdef __cplusplus
extern "C" {
#endif

/* Register a (symbol name -> shim function pointer) mapping. Exact match;
 * no wildcards in v1. Called from substrate inits before first dlsym.
 * Returns 0 on success; -1 if registry full or args invalid. */
int cipher_rt_dlsym_register(const char *symname, void *shim_fn);

/* Install GOT patches for dlsym + dlvsym targets. Idempotent; called from
 * the substrate inject chain after cipher_rt_got_patch_init. */
int cipher_rt_dlsym_hook_init(void);

/* Diagnostic counters. */
unsigned long cipher_rt_dlsym_intercepts(void);   /* registry-matched */
unsigned long cipher_rt_dlsym_passthrough(void);  /* registry-miss; real */

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_DLSYM_HOOK_H */
