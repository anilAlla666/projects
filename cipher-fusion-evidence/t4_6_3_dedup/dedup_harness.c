/* T4.6.3 Phase 2 harness — drives cipher_rt_kv_alloc's content-hash
 * dedup on synthesized known-content pages and runs the three binding
 * indicators.
 *
 * For each flavor: replay {flavor}_pageseq.txt. Each line is a page-id;
 * identical id => byte-identical synthesized 2 MiB content. So the
 * allocator's dedup ratio MUST equal the file's sim ratio — any
 * deviation is an allocator bug (indicator a). Every put is read back
 * and SHA-256-checked against the content put (indicator b, strict).
 * After freeing all pages, refcount accounting must zero out (c).
 *
 * Build: gcc -O2 -I<phase4> -o dedup_harness dedup_harness.c \
 *          <phase4>/cipher_rt_kv_alloc.c -lcuda -lcrypto -lpthread
 */
#include "cipher_rt_kv_alloc.h"
#include <cuda.h>
#include <openssl/sha.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define PAGE (2 * 1024 * 1024)
#define ODIR "/home/ubuntu/cipher-fusion-evidence/t4_6_3_dedup"

/* Deterministic 2 MiB pseudo-random content for a page-id (splitmix64).
 * Identical id -> identical bytes; distinct id -> distinct bytes. */
static void synth(uint64_t pid, uint8_t *buf)
{
	uint64_t x = pid * 0x9E3779B97F4A7C15ULL + 0x123456789ULL;
	uint64_t *w = (uint64_t *)buf;
	for (size_t i = 0; i < PAGE / 8; i++) {
		x += 0x9E3779B97F4A7C15ULL;
		uint64_t z = x;
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
		z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
		w[i] = z ^ (z >> 31);
	}
}

static int run_flavor(const char *flavor, int k)
{
	char path[256];
	snprintf(path, sizeof path, "%s/pageseq_%s_%d.txt", ODIR, flavor, k);
	FILE *f = fopen(path, "r");
	if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
	int M = 0;
	double sim_ratio = 0.0;
	if (fscanf(f, "%d %lf", &M, &sim_ratio) != 2) { fclose(f); return 1; }
	uint64_t *seq = malloc((size_t)M * sizeof(uint64_t));
	for (int i = 0; i < M; i++)
		if (fscanf(f, "%lu", &seq[i]) != 1) { fclose(f); return 1; }
	fclose(f);

	unsigned long long *dev = malloc((size_t)M * sizeof(*dev));
	uint8_t *content = malloc(PAGE);
	uint8_t *readback = malloc(PAGE);

	printf("\n=== flavor=%s window=%d  M=%d  sim_ratio=%.4f%% ===\n",
	       flavor, k, M, sim_ratio * 100.0);

	/* dedup stats are process-cumulative counters; snapshot at flavor
	 * start and delta, since this process replays 3 flavors. */
	struct cipher_rt_kv_dedup_stats st0;
	cipher_rt_kv_dedup_get_stats(&st0);

	int put_fail = 0, verify_fail = 0;
	for (int i = 0; i < M; i++) {
		synth(seq[i], content);
		if (cipher_rt_kv_dedup_put(content, &dev[i]) != 0) {
			put_fail++; dev[i] = 0; continue;
		}
		/* (b) byte-correctness: read the page CIPHER handed back
		 * (deduped or not) and SHA-256 it against what we put. */
		if (cuMemcpyDtoH(readback, (CUdeviceptr)dev[i], PAGE)
		    != CUDA_SUCCESS) { verify_fail++; continue; }
		unsigned char s_put[32], s_back[32];
		SHA256(content, PAGE, s_put);
		SHA256(readback, PAGE, s_back);
		if (memcmp(s_put, s_back, 32) != 0)
			verify_fail++;
	}

	struct cipher_rt_kv_dedup_stats st1;
	cipher_rt_kv_dedup_get_stats(&st1);
	uint64_t f_puts = st1.puts - st0.puts;
	uint64_t f_hits = st1.hits - st0.hits;
	uint64_t f_misses = st1.misses - st0.misses;
	uint64_t f_coll = st1.hash_collisions - st0.hash_collisions;
	double real_ratio = f_puts ? (double)f_hits / f_puts : 0.0;

	printf("(a) counters : puts=%lu hits=%lu misses=%lu  "
	       "real_ratio=%.4f%%  real/sim=%.4f\n",
	       f_puts, f_hits, f_misses, real_ratio * 100.0,
	       sim_ratio > 0 ? real_ratio / sim_ratio : 0.0);
	printf("    physical_pages=%lu virtual_pages=%lu hash_collisions=%lu  "
	       "put_fail=%d\n", st1.physical_pages, st1.virtual_pages,
	       f_coll, put_fail);
	printf("(b) byte-id  : %d/%d pages mismatched (SHA-256 readback)\n",
	       verify_fail, M);

	/* (c) refcount integrity: free everything, expect a clean zero. */
	for (int i = 0; i < M; i++)
		if (dev[i]) cipher_rt_kv_dedup_free(dev[i]);
	struct cipher_rt_kv_dedup_stats st2;
	cipher_rt_kv_dedup_get_stats(&st2);
	uint64_t f_releases = st2.refcount_releases - st1.refcount_releases;
	printf("(c) refcount : after free-all  physical_pages=%lu "
	       "virtual_pages=%lu  this-flavor releases=%lu (misses was %lu)\n",
	       st2.physical_pages, st2.virtual_pages, f_releases, f_misses);

	int ind_a = (f_puts == (uint64_t)(M - put_fail)) &&
	            (real_ratio >= sim_ratio - 0.005) &&
	            (real_ratio <= sim_ratio + 0.005);
	int ind_b = (verify_fail == 0 && put_fail == 0);
	int ind_c = (st2.physical_pages == 0 && st2.virtual_pages == 0 &&
	             f_releases == f_misses);
	printf("INDICATORS %s: (a)=%s (b)=%s (c)=%s\n",
	       (ind_a && ind_b && ind_c) ? "PASS" : "FAIL",
	       ind_a ? "pass" : "FAIL", ind_b ? "pass" : "FAIL",
	       ind_c ? "pass" : "FAIL");

	free(seq); free(dev); free(content); free(readback);
	return (ind_a && ind_b && ind_c) ? 0 : 1;
}

int main(void)
{
	if (cipher_rt_kv_alloc_init(20ULL << 30) != 0) {
		fprintf(stderr, "alloc_init failed\n"); return 1;
	}
	if (cipher_rt_kv_dedup_init() != 0) {
		fprintf(stderr, "dedup_init failed\n"); return 1;
	}
	int fails = 0, runs = 0;
	const char *flavors[3] = { "conversation", "toolagent", "synthetic" };
	for (int i = 0; i < 3; i++)
		for (int k = 0; k < 5; k++) {
			fails += run_flavor(flavors[i], k);
			runs++;
		}
	printf("\n=== T4.6.3 Phase 2: %d/%d subset runs PASS — %s ===\n",
	       runs - fails, runs, fails == 0 ? "ALL PASS" : "FAIL");
	return fails ? 1 : 0;
}
