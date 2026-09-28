/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_dlsym_hook.c -- F-B.3.6.1 dlsym/dlvsym interception (Path B').
 *
 * See cipher_rt_dlsym_hook.h.
 *
 * TLS recursion guard: dlsym_impl calls real dlsym to resolve its own
 * pointers (and to fall through on registry miss). Without the guard,
 * the substrate's own dlsym calls would re-enter the hook and risk
 * infinite recursion. The guard is set on hook entry and cleared on exit;
 * while set, the hook short-circuits to the cached real dlsym pointer.
 */
#define _GNU_SOURCE
#include "cipher_rt_dlsym_hook.h"
#include "cipher_rt_got_patch.h"
#include "cipher_v2_internal.h"

#include <dlfcn.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#define CIPHER_DLSYM_MAX_TARGETS 32

struct dlsym_entry {
	const char *symname;
	void       *shim_fn;
};

static struct dlsym_entry g_table[CIPHER_DLSYM_MAX_TARGETS];
static atomic_int         g_n_entries;
static atomic_int         g_init_done;
static atomic_ulong       g_intercepts;
static atomic_ulong       g_passthrough;

typedef void *(*dlsym_fn_t)(void *, const char *);
typedef void *(*dlvsym_fn_t)(void *, const char *, const char *);

static dlsym_fn_t  g_real_dlsym;
static dlvsym_fn_t g_real_dlvsym;

/* Thread-local recursion guard: any dlsym() call originating from inside
 * cipher_rt_dlsym_impl (or any code it calls) skips the registry lookup
 * to avoid re-entry. */
static __thread int t_in_hook;

/* Resolve real dlsym + dlvsym via libdl's own symbols at startup.
 * Uses libc's dlsym via RTLD_NEXT pattern -- but RTLD_NEXT itself calls
 * dlsym, so we resolve via dlvsym(RTLD_DEFAULT, "dlsym", "GLIBC_2.34")
 * which uses a known versioned symbol in glibc 2.34+ libdl. Fallback to
 * earlier versions if GLIBC_2.34 not present. */
static int resolve_real_dlsym(void)
{
	/* Bootstrap: use dlvsym to find dlsym, since dlsym recursion would
	 * trip our hook. dlvsym is itself in libdl; we need a way to call
	 * it without going through any hook. The trick: dlvsym's symbol
	 * lookup at link time resolves to libdl at startup via DT_NEEDED of
	 * libcipher_rt.so. libcipher_rt.so does NOT directly link libdl
	 * (it uses libc.so.6 which integrated dlfcn since glibc 2.34), so
	 * dlvsym is satisfied at libc level. The first call to dlvsym
	 * resolves & caches; subsequent libdl-pointer cache uses it.
	 *
	 * Simplest mechanism: dlvsym(RTLD_DEFAULT, "dlsym", ...) walks
	 * search order including libc.so.6 where the real dlsym lives.
	 * We MUST NOT route through the GOT patched slots. dlvsym calls
	 * resolve at libcipher_rt.so's own GOT, which we never patch (the
	 * patcher excludes its own module by program-header walk semantics).
	 */
	if (g_real_dlsym) return 0;

	g_real_dlvsym = (dlvsym_fn_t)dlvsym;
	if (!g_real_dlvsym)
		return -1;
	g_real_dlsym = (dlsym_fn_t)g_real_dlvsym(
		RTLD_DEFAULT, "dlsym", "GLIBC_2.34");
	if (!g_real_dlsym)
		g_real_dlsym = (dlsym_fn_t)g_real_dlvsym(
			RTLD_DEFAULT, "dlsym", "GLIBC_2.2.5");
	if (!g_real_dlsym)
		g_real_dlsym = (dlsym_fn_t)dlsym;  /* last resort */
	return g_real_dlsym ? 0 : -1;
}

int cipher_rt_dlsym_register(const char *symname, void *shim_fn)
{
	int idx;

	if (!symname || !shim_fn)
		return -1;
	idx = atomic_fetch_add(&g_n_entries, 1);
	if (idx >= CIPHER_DLSYM_MAX_TARGETS) {
		atomic_fetch_sub(&g_n_entries, 1);
		cipher_log("DLSYM-HOOK: registry full -- '%s' not registered",
		           symname);
		return -1;
	}
	g_table[idx].symname = symname;
	g_table[idx].shim_fn = shim_fn;
	return 0;
}

static void *registry_lookup(const char *sym)
{
	int n = atomic_load(&g_n_entries);
	int i;

	if (!sym) return NULL;
	for (i = 0; i < n; i++) {
		if (g_table[i].symname && strcmp(sym, g_table[i].symname) == 0)
			return g_table[i].shim_fn;
	}
	return NULL;
}

/* GOT-patched dlsym trampoline. Caller passes (handle, symname); we
 * consult the registry and either return CIPHER's shim or fall through
 * to real dlsym. */
void *cipher_rt_dlsym_impl(void *handle, const char *symname)
{
	void *r;

	if (t_in_hook) {
		/* Recursion: any dlsym() invoked from inside the hook (during
		 * shim_lazy_init real-fn resolution, registry lookups, etc.)
		 * skips the registry and uses real dlsym directly. */
		return g_real_dlsym ? g_real_dlsym(handle, symname)
		                    : dlsym(handle, symname);
	}

	t_in_hook = 1;
	if (!g_real_dlsym && resolve_real_dlsym() < 0) {
		t_in_hook = 0;
		return NULL;
	}
	r = registry_lookup(symname);
	if (r) {
		atomic_fetch_add(&g_intercepts, 1);
		t_in_hook = 0;
		return r;
	}
	atomic_fetch_add(&g_passthrough, 1);
	r = g_real_dlsym(handle, symname);
	t_in_hook = 0;
	return r;
}

void *cipher_rt_dlvsym_impl(void *handle, const char *symname,
                            const char *version)
{
	void *r;

	if (t_in_hook) {
		return g_real_dlvsym ? g_real_dlvsym(handle, symname, version)
		                     : dlvsym(handle, symname, version);
	}

	t_in_hook = 1;
	if (!g_real_dlvsym) {
		if (resolve_real_dlsym() < 0) {
			t_in_hook = 0;
			return NULL;
		}
	}
	r = registry_lookup(symname);
	if (r) {
		atomic_fetch_add(&g_intercepts, 1);
		t_in_hook = 0;
		return r;
	}
	atomic_fetch_add(&g_passthrough, 1);
	r = g_real_dlvsym(handle, symname, version);
	t_in_hook = 0;
	return r;
}

int cipher_rt_dlsym_hook_init(void)
{
	int rc = 0;

	if (atomic_exchange(&g_init_done, 1)) return 0;

	if (resolve_real_dlsym() < 0) {
		cipher_log("DLSYM-HOOK: failed to resolve real dlsym; hook inactive");
		return -1;
	}

	if (cipher_rt_got_register("dlsym",
	                           (void *)cipher_rt_dlsym_impl,
	                           NULL) != 0) {
		cipher_log("DLSYM-HOOK: GOT registration failed (dlsym)");
		rc = -1;
	}
	if (cipher_rt_got_register("dlvsym",
	                           (void *)cipher_rt_dlvsym_impl,
	                           NULL) != 0) {
		cipher_log("DLSYM-HOOK: GOT registration failed (dlvsym)");
		rc = -1;
	}
	cipher_log("DLSYM-HOOK: registered (real dlsym=%p dlvsym=%p; %d shims in registry)",
	           (void *)g_real_dlsym, (void *)g_real_dlvsym,
	           atomic_load(&g_n_entries));
	return rc;
}

unsigned long cipher_rt_dlsym_intercepts(void)
{
	return atomic_load(&g_intercepts);
}

unsigned long cipher_rt_dlsym_passthrough(void)
{
	return atomic_load(&g_passthrough);
}
