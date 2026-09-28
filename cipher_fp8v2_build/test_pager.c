/* test_pager.c -- correctness gates for the residency pager (cipher_rt_pager.c).
 * Gate 1: bit-identical after page-out -> page-in (extends the proven sequential case).
 * Gate 2: EVICTION-DURING-USE under contention -- N serve threads (real GPU read + checksum on per-thread
 *         streams) racing a page-out/in churner. Every HIT must read correct data, never fault, never read
 *         mid-eviction. hits>0 AND misses>0 proves the churner actually overlapped.
 * Negative control: build with -DCIPHER_PAGER_NO_GPU_SYNC (in cipher_rt_pager.c) -> MUST corrupt/fault,
 *         proving the test can detect a coherence violation. */
#include "cipher_rt_pager.h"
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#define REGION (16u*1024*1024)
#define NTHREADS 8
#define ITERS 1500
static int g_id;
static unsigned char *g_warm;
static uint64_t g_warm_cksum;

/* sample-checksum at 3 offsets (detects garbage / partial / unmapped reads cheaply) */
static uint64_t cksum(const unsigned char *p) {
	uint64_t s = 0; size_t offs[3] = {0, REGION/2, REGION-65536};
	for (int o = 0; o < 3; o++) for (size_t i = 0; i < 65536; i++) s = s*131 + p[offs[o]+i];
	return s;
}

static volatile int g_corrupt = 0, g_memerr = 0;
static volatile long g_hits = 0, g_miss = 0;
static volatile int g_stop = 0;

static void *serve_worker(void *arg) {
	(void)arg;
	cudaStream_t st; cudaStreamCreate(&st);
	unsigned char *hbuf; cudaMallocHost((void**)&hbuf, REGION);
	while (!g_stop) {
		unsigned long long va = 0;
		if (cipher_pager_serve_begin(g_id, &va) == CIPHER_PAGER_HIT) {
			/* real GPU read of the resident region; serve_end (ref--) BEFORE the sync, so the read is
			 * IN FLIGHT when the evictor may drain ref==0 -> exercises the GPU-async eviction race. */
			cudaError_t e = cudaMemcpyAsync(hbuf, (void*)va, REGION, cudaMemcpyDeviceToHost, st);
			cipher_pager_serve_end(g_id);
			cudaError_t e2 = cudaStreamSynchronize(st);
			__sync_fetch_and_add(&g_hits, 1);
			if (e != cudaSuccess || e2 != cudaSuccess) { __sync_fetch_and_add(&g_memerr, 1); cudaGetLastError(); }
			else if (cksum(hbuf) != g_warm_cksum) __sync_fetch_and_add(&g_corrupt, 1);
		} else {
			__sync_fetch_and_add(&g_miss, 1);
			usleep(20);   /* brief backoff on MISS (region being paged) */
		}
	}
	cudaFreeHost(hbuf); cudaStreamDestroy(st);
	return NULL;
}

static void *churn_worker(void *arg) {
	(void)arg;
	/* hold a RESIDENT dwell (serves HIT + issue ~600us reads) then evict (catches in-flight reads -> the
	 * eviction-during-use race). dwell 3ms > read time so most page_outs overlap an in-flight serve read. */
	while (!g_stop) {
		if (cipher_pager_page_in(g_id) == CIPHER_PAGER_OK) usleep(3000);  /* resident window */
		cipher_pager_page_out(g_id);
		usleep(300);                                                      /* evicted window (serves MISS) */
	}
	return NULL;
}

int main(void) {
	if (cipher_pager_init() != 0) { fprintf(stderr, "pager_init FAILED\n"); return 1; }
	g_warm = malloc(REGION);
	for (size_t i = 0; i < REGION; i++) g_warm[i] = (unsigned char)((i * 2654435761u) >> 13);
	g_warm_cksum = cksum(g_warm);
	g_id = cipher_pager_register(REGION, g_warm);
	if (g_id < 0) { fprintf(stderr, "register FAILED\n"); return 1; }

	/* ---- Gate 1: sequential bit-identical across a page cycle ---- */
	unsigned char *hbuf; cudaMallocHost((void**)&hbuf, REGION);
	int g1 = 1;
	if (cipher_pager_page_in(g_id) != 0) g1 = 0;
	{ unsigned long long va; cipher_pager_serve_begin(g_id, &va);
	  cudaMemcpy(hbuf, (void*)va, REGION, cudaMemcpyDeviceToHost); cipher_pager_serve_end(g_id);
	  if (cksum(hbuf) != g_warm_cksum) g1 = 0; }
	cipher_pager_page_out(g_id);
	if (cipher_pager_page_in(g_id) != 0) g1 = 0;
	{ unsigned long long va; cipher_pager_serve_begin(g_id, &va);
	  cudaMemcpy(hbuf, (void*)va, REGION, cudaMemcpyDeviceToHost); cipher_pager_serve_end(g_id);
	  if (cksum(hbuf) != g_warm_cksum) g1 = 0; }
	printf("GATE1 bit-identical after page-out->page-in: %s\n", g1 ? "PASS" : "FAIL");
	cudaFreeHost(hbuf);

	/* ---- Gate 2: eviction-during-use under contention (duration-based) ---- */
	cipher_pager_page_in(g_id);          /* start RESIDENT so serves HIT immediately */
	pthread_t serves[NTHREADS], churn;
	pthread_create(&churn, NULL, churn_worker, NULL);
	for (int i = 0; i < NTHREADS; i++) pthread_create(&serves[i], NULL, serve_worker, NULL);
	sleep(4);                            /* run the contention for 4s */
	g_stop = 1;
	for (int i = 0; i < NTHREADS; i++) pthread_join(serves[i], NULL);
	pthread_join(churn, NULL);

	struct cipher_pager_stats st; cipher_pager_get_stats(g_id, &st);
	printf("GATE2 contention: hits=%ld miss=%ld corrupt=%d memerr=%d | evicts=%lu pageins=%lu cold_miss=%lu\n",
	       g_hits, g_miss, g_corrupt, g_memerr, st.evict_cnt, st.pagein_cnt, st.cold_miss);
	int race_exercised = (g_hits > 0 && g_miss > 0);
	int coherent = (g_corrupt == 0 && g_memerr == 0);
#ifdef CIPHER_PAGER_NO_GPU_SYNC
	printf("NEGATIVE-CONTROL (no GPU sync): expect corruption -> %s\n",
	       (!coherent) ? "DETECTED (test can fail) GOOD" : "NO CORRUPTION (test blind! window too narrow)");
	return (!coherent) ? 0 : 2;
#else
	printf("GATE2 verdict: race_exercised=%d coherent=%d -> %s\n", race_exercised, coherent,
	       (race_exercised && coherent) ? "PASS (never served half-evicted under contention)" : "FAIL");
	return (race_exercised && coherent) ? 0 : 1;
#endif
}
