/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_got_patch.c -- CP 2.5 GOT/PLT patcher.
 *
 * dl_iterate_phdr walks every loaded ELF object; for each we parse the
 * PT_DYNAMIC relocation tables (.rela.plt JUMP_SLOT + .rela.dyn GLOB_DAT),
 * find GOT slots whose relocation symbol matches a registered target, and
 * overwrite the slot with our trampoline. Patching the call site directly
 * makes interception independent of dlopen order -- which is exactly why
 * it works from a CUDA_INJECTION64_PATH-loaded library (loaded last, so
 * LD_PRELOAD-style symbol shadowing is impossible).
 *
 * RELRO: the GOT page may be read-only (full RELRO, -z now). write_got_slot
 * reads the page's current protection from /proc/self/maps, mprotect's it
 * writable, writes, and restores the original protection -- never downgrades
 * a lazy-binding-writable page to read-only on a guess.
 *
 * Real-fn resolution is deliberately NOT done from the saved slot value:
 * a lazy (unresolved) JUMP_SLOT holds the PLT resolver stub, and calling
 * that stub re-resolves and rewrites the slot -- which would clobber our
 * patch. Substrates resolve the real function via dlopen(soname)+dlsym
 * instead (see cipher_rt_cublas_shim.c / cipher_rt_attn_dispatch.cpp).
 * save_real here is diagnostic.
 *
 * x86-64 / glibc only (RELA relocations; glibc relocates DT_* d_ptr
 * entries to absolute addresses at load time -- used directly).
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "cipher_rt_got_patch.h"
#include "cipher_v2_internal.h"

#include <link.h>
#include <elf.h>
#include <sys/mman.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <pthread.h>

#define CIPHER_GOT_MAX_TARGETS 16

struct got_target {
	const char *symname;
	void       *trampoline;
	void      **save_real;
	int         saved;       /* save_real already written once */
};

static struct got_target g_targets[CIPHER_GOT_MAX_TARGETS];
static int               g_n_targets;
static pthread_mutex_t   g_mu = PTHREAD_MUTEX_INITIALIZER;
static unsigned long     g_slots_patched;    /* cumulative */
static unsigned long     g_modules_scanned;  /* last apply() */

int cipher_rt_got_register(const char *symname, void *trampoline,
                           void **save_real)
{
	if (!symname || !trampoline)
		return -1;
	pthread_mutex_lock(&g_mu);
	if (g_n_targets >= CIPHER_GOT_MAX_TARGETS) {
		pthread_mutex_unlock(&g_mu);
		cipher_log("GOT: target table full -- '%s' not registered", symname);
		return -1;
	}
	g_targets[g_n_targets].symname    = symname;
	g_targets[g_n_targets].trampoline = trampoline;
	g_targets[g_n_targets].save_real  = save_real;
	g_targets[g_n_targets].saved      = 0;
	g_n_targets++;
	pthread_mutex_unlock(&g_mu);
	return 0;
}

/* Protection flags of the mapped region containing addr, or -1 if not
 * found (then the caller leaves the page writable -- safe). */
static int region_prot(uintptr_t addr)
{
	FILE *f = fopen("/proc/self/maps", "r");
	if (!f)
		return -1;
	char line[512];
	int prot = -1;
	while (fgets(line, sizeof(line), f)) {
		unsigned long lo, hi;
		char r, w, x, p;
		if (sscanf(line, "%lx-%lx %c%c%c%c", &lo, &hi, &r, &w, &x, &p) != 6)
			continue;
		if (addr >= lo && addr < hi) {
			prot = 0;
			if (r == 'r') prot |= PROT_READ;
			if (w == 'w') prot |= PROT_WRITE;
			if (x == 'x') prot |= PROT_EXEC;
			break;
		}
	}
	fclose(f);
	return prot;
}

/* Overwrite *slot with val, handling RELRO. Returns 0 ok, -1 on failure. */
static int write_got_slot(void **slot, void *val)
{
	long pg = sysconf(_SC_PAGESIZE);
	if (pg <= 0)
		pg = 4096;
	uintptr_t page = (uintptr_t)slot & ~((uintptr_t)pg - 1);
	int orig = region_prot((uintptr_t)slot);
	int want = (orig >= 0 ? orig : PROT_READ) | PROT_WRITE;

	if (mprotect((void *)page, (size_t)pg, want) != 0)
		return -1;
	*slot = val;
	/* Restore only if we actually changed the protection. If orig is
	 * unknown, leave the page writable -- a writable GOT is the lazy-
	 * binding default; never downgrade to read-only on a guess (a later
	 * lazy resolution of another slot on this page would fault). */
	if (orig >= 0 && orig != want)
		(void)mprotect((void *)page, (size_t)pg, orig);
	return 0;
}

struct scan_ctx { int patched; };

static int phdr_cb(struct dl_phdr_info *info, size_t size, void *data)
{
	(void)size;
	struct scan_ctx *ctx = (struct scan_ctx *)data;
	g_modules_scanned++;

	/* Locate PT_DYNAMIC. */
	const ElfW(Dyn) *dyn = NULL;
	for (int i = 0; i < info->dlpi_phnum; i++) {
		if (info->dlpi_phdr[i].p_type == PT_DYNAMIC) {
			dyn = (const ElfW(Dyn) *)(info->dlpi_addr +
			                          info->dlpi_phdr[i].p_vaddr);
			break;
		}
	}
	if (!dyn)
		return 0;

	/* Walk the dynamic array. glibc relocates DT_* d_ptr to absolute. */
	const ElfW(Sym)  *symtab = NULL;
	const char       *strtab = NULL;
	const ElfW(Rela) *jmprel = NULL, *rela = NULL;
	size_t jmprelsz = 0, relasz = 0;
	for (const ElfW(Dyn) *d = dyn; d->d_tag != DT_NULL; d++) {
		switch (d->d_tag) {
		case DT_SYMTAB:   symtab   = (const ElfW(Sym) *)d->d_un.d_ptr;  break;
		case DT_STRTAB:   strtab   = (const char *)d->d_un.d_ptr;       break;
		case DT_JMPREL:   jmprel   = (const ElfW(Rela) *)d->d_un.d_ptr; break;
		case DT_PLTRELSZ: jmprelsz = d->d_un.d_val;                     break;
		case DT_RELA:     rela     = (const ElfW(Rela) *)d->d_un.d_ptr; break;
		case DT_RELASZ:   relasz   = d->d_un.d_val;                     break;
		default: break;
		}
	}
	if (!symtab || !strtab)
		return 0;

	/* .rela.plt (JUMP_SLOT) + .rela.dyn (GLOB_DAT). Both Elf*_Rela on
	 * x86-64 (DT_PLTREL == DT_RELA). */
	const ElfW(Rela) *tabs[2] = { jmprel, rela };
	size_t            szs[2]  = { jmprelsz, relasz };
	for (int t = 0; t < 2; t++) {
		if (!tabs[t])
			continue;
		size_t count = szs[t] / sizeof(ElfW(Rela));
		for (size_t i = 0; i < count; i++) {
			const ElfW(Rela) *r = &tabs[t][i];
			unsigned long rtype = ELF64_R_TYPE(r->r_info);
			if (rtype != R_X86_64_JUMP_SLOT &&
			    rtype != R_X86_64_GLOB_DAT)
				continue;
			const char *name = strtab + symtab[ELF64_R_SYM(r->r_info)].st_name;
			if (!name || !name[0])
				continue;

			for (int g = 0; g < g_n_targets; g++) {
				if (strcmp(name, g_targets[g].symname) != 0)
					continue;
				void **slot = (void **)(info->dlpi_addr + r->r_offset);
				if (*slot == g_targets[g].trampoline)
					break;  /* already patched -- idempotent */
				if (g_targets[g].save_real && !g_targets[g].saved) {
					*g_targets[g].save_real = *slot;
					g_targets[g].saved = 1;
				}
				if (write_got_slot(slot, g_targets[g].trampoline) == 0) {
					ctx->patched++;
					g_slots_patched++;
					cipher_dbg("GOT: patched '%s' slot %p in %s",
					   name, (void *)slot,
					   (info->dlpi_name && info->dlpi_name[0])
					     ? info->dlpi_name : "(main)");
				} else {
					cipher_log("GOT: mprotect failed for '%s' in %s",
					   name, (info->dlpi_name && info->dlpi_name[0])
					     ? info->dlpi_name : "(main)");
				}
				break;
			}
		}
	}
	return 0;
}

int cipher_rt_got_patch_apply(void)
{
	struct scan_ctx ctx = { 0 };
	pthread_mutex_lock(&g_mu);
	g_modules_scanned = 0;
	dl_iterate_phdr(phdr_cb, &ctx);
	pthread_mutex_unlock(&g_mu);
	return ctx.patched;
}

int cipher_rt_got_patch_init(void)
{
	int n = cipher_rt_got_patch_apply();
	cipher_log("GOT: patch applied -- %d slot(s) across %lu module(s); "
	           "%d target(s) registered",
	           n, g_modules_scanned, g_n_targets);
	return 0;
}

unsigned long cipher_rt_got_slots_patched(void)   { return g_slots_patched; }
unsigned long cipher_rt_got_modules_scanned(void) { return g_modules_scanned; }
