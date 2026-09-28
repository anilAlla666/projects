/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_got_patch.h -- CP 2.5 GOT/PLT patcher.
 *
 * Replaces LD_PRELOAD link-order symbol interposition with runtime GOT
 * patching, driven from InitializeInjection2. A library loaded by the
 * CUDA driver via CUDA_INJECTION64_PATH lands LAST in the link map, so it
 * cannot shadow libcublas / libtorch symbols the LD_PRELOAD way. Patching
 * the caller's GOT slots directly is load-order-independent and works
 * from the injection lib.
 *
 * x86-64 / glibc.
 */
#ifndef CIPHER_RT_GOT_PATCH_H
#define CIPHER_RT_GOT_PATCH_H

#ifdef __cplusplus
extern "C" {
#endif

/* Register an interception target. Call before cipher_rt_got_patch_apply().
 *   symname    -- exact .dynsym name (mangled, for C++ symbols).
 *   trampoline -- address written into every matching GOT slot.
 *   save_real  -- optional; if non-NULL, *save_real receives the original
 *                 value of the first matching slot (diagnostic only --
 *                 substrates resolve the real fn via dlopen+dlsym, which
 *                 is robust against lazy-PLT re-resolution; see the .c).
 * Returns 0 on success, -1 if the target table is full. */
int cipher_rt_got_register(const char *symname, void *trampoline,
                           void **save_real);

/* Scan every loaded ELF module and install all registered patches.
 * Idempotent: a slot already holding the trampoline is skipped, and a
 * trampoline value is never saved as the "real" pointer. Returns the
 * number of GOT slots NEWLY patched on this call. */
int cipher_rt_got_patch_apply(void);

/* Inject-chain entry: apply() + a one-line operator-log summary.
 * Always returns 0 -- a patch miss degrades that substrate, it does not
 * block cuInit. */
int cipher_rt_got_patch_init(void);

/* Diagnostics. */
unsigned long cipher_rt_got_slots_patched(void);   /* cumulative */
unsigned long cipher_rt_got_modules_scanned(void); /* last apply() */

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_GOT_PATCH_H */
