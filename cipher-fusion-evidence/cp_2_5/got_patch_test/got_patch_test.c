/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * CP 2.5 -- GOT patcher standalone test. No CUDA.
 *
 * Proves cipher_rt_got_patch can intercept a real PLT call (puts) from
 * the test's own GOT: baseline (un-patched) misses the fake, post-patch
 * routes through it, re-apply is idempotent (0 new slots), interception
 * survives re-apply. Built with full RELRO (-z now -z relro) so the GOT
 * page is read-only and the mprotect path is exercised.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "cipher_rt_got_patch.h"
#include <stdio.h>
#include <dlfcn.h>

static int g_fake_hits;
typedef int (*puts_fn)(const char *);
static puts_fn g_real_puts;

/* The interception trampoline. Calls the real puts resolved via dlsym --
 * NOT via the saved GOT slot (a lazy stub would re-resolve and clobber
 * the patch). */
static int fake_puts(const char *s)
{
	g_fake_hits++;
	return g_real_puts ? g_real_puts(s) : 0;
}

int main(void)
{
	int fails = 0;

	g_real_puts = (puts_fn)dlsym(RTLD_DEFAULT, "puts");
	if (!g_real_puts) {
		fprintf(stderr, "TEST FAIL: cannot resolve real puts\n");
		return 1;
	}

	/* baseline: an un-patched puts call must not hit the fake. */
	puts("got_patch_test: baseline line (pre-patch, NOT intercepted)");
	if (g_fake_hits != 0) {
		fprintf(stderr, "TEST FAIL: fake hit before patch (%d)\n", g_fake_hits);
		fails++;
	}

	void *saved = NULL;
	if (cipher_rt_got_register("puts", (void *)fake_puts, &saved) != 0) {
		fprintf(stderr, "TEST FAIL: register\n");
		return 1;
	}
	int n = cipher_rt_got_patch_apply();
	fprintf(stderr, "patch apply #1: %d slot(s), %lu module(s) scanned, "
	        "saved_orig=%p\n", n, cipher_rt_got_modules_scanned(), saved);
	if (n < 1) {
		fprintf(stderr, "TEST FAIL: 0 slots patched\n");
		fails++;
	}

	/* post-patch: puts must route through fake_puts. */
	int before = g_fake_hits;
	puts("got_patch_test: post-patch line (SHOULD be intercepted)");
	if (g_fake_hits != before + 1) {
		fprintf(stderr, "TEST FAIL: puts not intercepted (hits %d -> %d)\n",
		        before, g_fake_hits);
		fails++;
	}

	/* idempotency: a second apply patches 0 new slots. */
	int n2 = cipher_rt_got_patch_apply();
	fprintf(stderr, "patch apply #2 (idempotency): %d slot(s)\n", n2);
	if (n2 != 0) {
		fprintf(stderr, "TEST FAIL: re-apply patched %d slot(s), want 0\n", n2);
		fails++;
	}

	/* interception survives re-apply. */
	before = g_fake_hits;
	puts("got_patch_test: after re-apply (still intercepted)");
	if (g_fake_hits != before + 1) {
		fprintf(stderr, "TEST FAIL: interception lost after re-apply\n");
		fails++;
	}

	fprintf(stderr, "got_patch_test: fake_puts hits total = %d\n", g_fake_hits);
	fprintf(stderr, "GOT_PATCH_TEST: %s%s\n", fails ? "FAIL" : "PASS",
	        fails ? "" : " (baseline-miss, patch-hit, idempotent, durable)");
	return fails ? 1 : 0;
}
