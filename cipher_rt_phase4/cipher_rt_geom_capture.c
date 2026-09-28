/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_geom_capture.c -- TEST-ONLY launch-geometry capture (default-OFF).
 * See header. Additive: a new TU; one env-gated call from cipher_cupti.c.
 * Default-OFF (CIPHER_GEOM_CAPTURE unset) => early-out, byte-identical.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include "cipher_v2_internal.h"   /* cipher_log macro */
#include "cipher_rt_geom_capture.h"

static atomic_int        g_on = -1;     /* -1 uninit, 0 off, 1 on */
static FILE             *g_f  = NULL;
static pthread_mutex_t   g_mu = PTHREAD_MUTEX_INITIALIZER;

#define NSEEN 4096
static atomic_uintptr_t  g_seen[NSEEN]; /* fn-pointers already dumped (dedup) */

static int geom_enabled(void)
{
	int v = atomic_load_explicit(&g_on, memory_order_relaxed);
	if (v >= 0) return v;
	const char *e = getenv("CIPHER_GEOM_CAPTURE");
	v = (e && e[0] && e[0] != '0') ? 1 : 0;
	atomic_store_explicit(&g_on, v, memory_order_relaxed);
	return v;
}

/* ── Geometry-keyed attention INTERCEPT (Step 3) ─────────────────────────
 * Framework-agnostic attention DETECTION: counts launches the geometry
 * classifier (cipher::classify_launch) calls ATTENTION (OpClass==1) — NO
 * symbol/path hardcode (the de-coupling from cipher_rt_attn_6pattern.c's
 * _vllm_fa3_C GOT-patch). Per-call count (parity with the 6-pattern's
 * per-call intercept counters). Observe-only — NEVER substitutes/redirects
 * (substitution is v1.5, W.5). Gated by CIPHER_GEOM_ATTN, default-OFF =>
 * byte-identical; even ON it only increments a counter (output-neutral). */
#define CIPHER_OPCLASS_ATTENTION 1
static atomic_int   g_attn_on = -1;
static atomic_ulong g_geom_attn_intercepts = 0;  /* ATTENTION-classified launches */
static atomic_ulong g_geom_launches        = 0;  /* total launches seen (denominator) */

static void geom_attn_atexit(void)
{
	cipher_log("GEOM-ATTN: exit totals — geom_attn_intercepts=%lu launches=%lu "
	           "(geometry-keyed attention detection; compare vs 6-pattern attn_p2)",
	           atomic_load(&g_geom_attn_intercepts), atomic_load(&g_geom_launches));
}

static int attn_detect_enabled(void)
{
	int v = atomic_load_explicit(&g_attn_on, memory_order_relaxed);
	if (v >= 0) return v;
	const char *e = getenv("CIPHER_GEOM_ATTN");
	v = (e && e[0] && e[0] != '0') ? 1 : 0;
	atomic_store_explicit(&g_attn_on, v, memory_order_relaxed);
	if (v) atexit(geom_attn_atexit);   /* worker-subprocess-visible count dump */
	return v;
}

unsigned long cipher_rt_geom_attn_intercepts(void)
{ return atomic_load_explicit(&g_geom_attn_intercepts, memory_order_relaxed); }
unsigned long cipher_rt_geom_launches(void)
{ return atomic_load_explicit(&g_geom_launches, memory_order_relaxed); }

void cipher_rt_geom_capture(const void *fn,
                            unsigned gx, unsigned gy, unsigned gz,
                            unsigned bx, unsigned by, unsigned bz,
                            unsigned shmem, int geom_class)
{
	int cap = geom_enabled();         /* CIPHER_GEOM_CAPTURE — per-fn dump */
	int det = attn_detect_enabled();  /* CIPHER_GEOM_ATTN — attention detect counter */
	if ((!cap && !det) || !fn) return;   /* both OFF => byte-identical early-out */

	/* Geometry-keyed attention detection (per-call; parity with the 6-pattern). */
	if (det) {
		atomic_fetch_add_explicit(&g_geom_launches, 1, memory_order_relaxed);
		if (geom_class == CIPHER_OPCLASS_ATTENTION)
			atomic_fetch_add_explicit(&g_geom_attn_intercepts, 1, memory_order_relaxed);
	}
	if (!cap) return;   /* detection-only: counted; skip the per-fn dump */

	/* dedup: dump each unique fn once (bounded linear probe). */
	uintptr_t k = (uintptr_t)fn;
	unsigned  h = (unsigned)((k >> 4) % NSEEN);
	for (unsigned i = 0; i < 8; ++i) {
		unsigned idx = (h + i) % NSEEN;
		uintptr_t cur = atomic_load_explicit(&g_seen[idx], memory_order_relaxed);
		if (cur == k) return;                       /* already dumped */
		if (cur == 0) {
			uintptr_t zero = 0;
			if (atomic_compare_exchange_strong(&g_seen[idx], &zero, k))
				break;                              /* claimed this slot */
			if (atomic_load_explicit(&g_seen[idx], memory_order_relaxed) == k)
				return;                             /* raced; another thread did it */
		}
	}

	pthread_mutex_lock(&g_mu);
	if (!g_f) g_f = fopen("/tmp/cipher_geom_capture.jsonl", "w");
	if (g_f) {
		fprintf(g_f,
		    "{\"fn\":\"%p\",\"gx\":%u,\"gy\":%u,\"gz\":%u,"
		    "\"bx\":%u,\"by\":%u,\"bz\":%u,\"shmem\":%u,\"geom_class\":%d}\n",
		    fn, gx, gy, gz, bx, by, bz, shmem, geom_class);
		fflush(g_f);
	}
	pthread_mutex_unlock(&g_mu);
}
