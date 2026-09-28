/* T4.6.4 memo-B input: measure the dedup hash-table bucket-walk cost,
 * to choose mutex vs spinlock for the kmod table lock.
 *
 * Models the kmod's chained content-hash table: 65536 buckets, ~150K
 * entries (load factor ~2.3, a heavy multi-tenant steady state). Times
 * the per-lookup bucket walk (hash -> bucket -> chain compare).
 *
 * Build: gcc -O2 -o bw t4_6_4_bucketwalk_bench.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define BUCKETS 65536
#define ENTRIES 150000
#define LOOKUPS 2000000

struct ent { uint64_t hash; struct ent *next; };

static uint64_t splitmix(uint64_t *s) {
	uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

int main(void) {
	struct ent **buk = calloc(BUCKETS, sizeof(*buk));
	struct ent *pool = malloc(ENTRIES * sizeof(struct ent));
	uint64_t s = 12345;
	uint64_t *keys = malloc(ENTRIES * sizeof(uint64_t));
	for (int i = 0; i < ENTRIES; i++) {
		uint64_t h = splitmix(&s);
		keys[i] = h;
		pool[i].hash = h;
		pool[i].next = buk[h & (BUCKETS - 1)];
		buk[h & (BUCKETS - 1)] = &pool[i];
	}
	/* chain-length stats */
	int maxchain = 0; long total = 0, nonempty = 0;
	for (int b = 0; b < BUCKETS; b++) {
		int n = 0; for (struct ent *e = buk[b]; e; e = e->next) n++;
		total += n; if (n) nonempty++; if (n > maxchain) maxchain = n;
	}
	printf("table: %d buckets, %d entries, load=%.2f, "
	       "max chain=%d, mean nonempty chain=%.2f\n",
	       BUCKETS, ENTRIES, (double)ENTRIES / BUCKETS, maxchain,
	       (double)total / nonempty);

	/* time lookups: 50%% present (hit) / 50%% absent (full-chain miss) */
	volatile uint64_t sink = 0;
	struct timespec t0, t1;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	uint64_t q = 999;
	for (int i = 0; i < LOOKUPS; i++) {
		uint64_t key;
		if (i & 1) key = keys[splitmix(&q) % ENTRIES];   /* present */
		else       key = splitmix(&q);                   /* absent  */
		struct ent *e = buk[key & (BUCKETS - 1)];
		while (e) { if (e->hash == key) { sink += 1; break; } e = e->next; }
	}
	clock_gettime(CLOCK_MONOTONIC, &t1);
	double ns = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec))
	            / (double)LOOKUPS;
	printf("bucket walk: %.2f ns / lookup  (%d lookups, sink=%lu)\n",
	       ns, LOOKUPS, (unsigned long)sink);
	return 0;
}
