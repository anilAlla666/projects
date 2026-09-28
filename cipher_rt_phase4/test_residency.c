/* test_residency.c -- N-region co-residence concurrent-serve coherence (STEP 2-arch).
 * Proves the per-region eviction-during-use invariant HOLDS across N regions concurrently, under a residency
 * manager that evicts cold models to page in demanded ones (more registered than fit).
 *   GATE A: N regions, M threads serve_demand(random region) under budget pressure -> every HIT reads correct
 *           data (never half-evicted, never cross-region), evictions happen, resident HBM <= budget, no wasted
 *           page_ins (livelock-free).
 *   WATCHDOG: a timeout thread fires (_exit 3) if the test hangs -> detects deadlock.
 * Negative controls (compile-time, in cipher_rt_pager.c):
 *   -DCIPHER_RESIDENCY_BREAK_LIVELOCK : release mgr lock before taking ref -> wasted page_ins > 0 (DETECTED).
 *   -DCIPHER_RESIDENCY_BREAK_LOCKORDER: reverse lock order -> AB-BA deadlock -> watchdog fires (DETECTED). */
#include "cipher_rt_pager.h"
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#define NREG     6
#define REGION   (4u*1024*1024)
#define NTHREADS 8
#define RUN_SEC  5
#define WD_SEC   30

static int       g_id[NREG];
static uint64_t  g_warm_cksum[NREG];
static volatile int g_corrupt = 0, g_memerr = 0, g_xregion = 0;
static volatile long g_hits = 0, g_miss = 0;
static volatile int g_stop = 0, g_done = 0;

static uint64_t cksum(const unsigned char *p) {
	uint64_t s = 0; size_t offs[3] = {0, REGION/2, REGION-65536};
	for (int o = 0; o < 3; o++) for (size_t i = 0; i < 65536; i++) s = s*131 + p[offs[o]+i];
	return s;
}

static void *serve_worker(void *arg) {
	unsigned int seed = (unsigned int)(uintptr_t)arg * 2654435761u + 1;
	cudaStream_t st; cudaStreamCreate(&st);
	unsigned char *hbuf; cudaMallocHost((void**)&hbuf, REGION);
	while (!g_stop) {
		int k = rand_r(&seed) % NREG;
		unsigned long long va = 0;
		if (cipher_pager_serve_demand(g_id[k], &va) == CIPHER_PAGER_HIT) {
			/* GPU read in-flight; serve_end (ref--) BEFORE the sync -> exercises eviction-during-use */
			cudaError_t e = cudaMemcpyAsync(hbuf, (void*)va, REGION, cudaMemcpyDeviceToHost, st);
			cipher_pager_serve_end(g_id[k]);
			cudaError_t e2 = cudaStreamSynchronize(st);
			__sync_fetch_and_add(&g_hits, 1);
			if (e != cudaSuccess || e2 != cudaSuccess) { __sync_fetch_and_add(&g_memerr, 1); cudaGetLastError(); }
			else { uint64_t c = cksum(hbuf);
				if (c != g_warm_cksum[k]) {
					__sync_fetch_and_add(&g_corrupt, 1);
					for (int j = 0; j < NREG; j++) if (c == g_warm_cksum[j]) __sync_fetch_and_add(&g_xregion, 1);
				}
			}
		} else { __sync_fetch_and_add(&g_miss, 1); }
	}
	cudaFreeHost(hbuf); cudaStreamDestroy(st);
	return NULL;
}

static void *watchdog(void *arg) {
	(void)arg;
	for (int i = 0; i < WD_SEC && !g_done; i++) sleep(1);
	if (!g_done) {
		fprintf(stderr, "WATCHDOG: test did not finish in %ds -> DEADLOCK/LIVELOCK DETECTED\n", WD_SEC);
		fflush(stderr); _exit(3);
	}
	return NULL;
}

int main(void) {
	if (cipher_pager_init() != 0) { fprintf(stderr, "init FAILED\n"); return 1; }
	pthread_t wd; pthread_create(&wd, NULL, watchdog, NULL);

	/* N distinct regions, distinct content */
	for (int i = 0; i < NREG; i++) {
		unsigned char *warm = malloc(REGION);
		for (size_t j = 0; j < REGION; j++) warm[j] = (unsigned char)((j * 2654435761u + i * 40503u) >> 13);
		g_warm_cksum[i] = cksum(warm);
		g_id[i] = cipher_pager_register(REGION, warm);
		if (g_id[i] < 0 || cipher_pager_page_in(g_id[i]) != 0) { fprintf(stderr, "region %d setup FAILED\n", i); return 1; }
		free(warm);
	}
	/* budget = only 3 of 6 regions fit -> forced eviction under demand */
	cipher_pager_mgr_init((size_t)3 * REGION);

	size_t free_start = 0, total_hbm = 0; cudaMemGetInfo(&free_start, &total_hbm);  /* physical-HBM leak baseline */

	pthread_t serves[NTHREADS];
	for (int i = 0; i < NTHREADS; i++) pthread_create(&serves[i], NULL, serve_worker, (void*)(uintptr_t)(i+1));
	sleep(RUN_SEC);
	g_stop = 1;
	for (int i = 0; i < NTHREADS; i++) pthread_join(serves[i], NULL);
	g_done = 1;

	/* tally evictions across regions */
	unsigned long evicts = 0, pageins = 0;
	for (int i = 0; i < NREG; i++) { struct cipher_pager_stats s; cipher_pager_get_stats(g_id[i], &s); evicts += s.evict_cnt; pageins += s.pagein_cnt; }
	unsigned long long resident = cipher_pager_mgr_resident_bytes();
	unsigned long wasted = cipher_pager_mgr_wasted();
	size_t free_end = 0; cudaMemGetInfo(&free_end, &total_hbm);
	/* PHYSICAL leak check (advisor): after 13k+ evict/pagein cycles, free HBM must not have dropped (a per-cycle
	 * cuMemRelease leak of REGION bytes would compound to GiB). Allow 64 MiB slack for driver/stream scratch. */
	long long phys_delta = (long long)free_start - (long long)free_end;
	printf("GATE-A N=%d budget=%uMiB: hits=%ld miss=%ld corrupt=%d xregion=%d memerr=%d | evicts=%lu pageins=%lu resident=%lluMiB wasted=%lu | freeHBM start=%zuMiB end=%zuMiB drift=%lldMiB\n",
	       NREG, (3*REGION)>>20, g_hits, g_miss, g_corrupt, g_xregion, g_memerr, evicts, pageins, resident>>20, wasted,
	       free_start>>20, free_end>>20, phys_delta>>20);

	int coherent  = (g_corrupt == 0 && g_memerr == 0 && g_xregion == 0);
	int pressured = (evicts > 0);
	int budget_ok = (resident <= (size_t)3 * REGION);
	int no_leak   = (phys_delta < (long long)(64u<<20));   /* physical HBM stable across 13k cycles */
#ifdef CIPHER_RESIDENCY_BREAK_LIVELOCK
	printf("NEGATIVE-CONTROL (livelock): expect wasted>0 -> %s\n", (wasted > 0) ? "DETECTED GOOD" : "NOT DETECTED (blind!)");
	return (wasted > 0) ? 0 : 2;
#else
	int livelock_free = (wasted == 0);
	printf("GATE-A verdict: coherent=%d pressured=%d budget_ok=%d livelock_free=%d no_phys_leak=%d -> %s\n",
	       coherent, pressured, budget_ok, livelock_free, no_leak,
	       (coherent && pressured && budget_ok && livelock_free && no_leak) ? "PASS (N-region co-residence concurrent-serve coherent)" : "FAIL");
	return (coherent && pressured && budget_ok && livelock_free && no_leak) ? 0 : 1;
#endif
}
