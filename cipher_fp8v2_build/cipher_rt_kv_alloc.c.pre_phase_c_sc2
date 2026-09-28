/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_kv_alloc.c - Phase 4.6.2 KV page allocator (CUDA VMM).
 *
 * See cipher_rt_kv_alloc.h for the design.
 */
#include "cipher_rt_kv_alloc.h"

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <stdint.h>
#include "cipher_kvdedup.h"   /* T4.6.4 kmod ioctl ABI */

/* Per-page slot state. */
enum page_state {
	PAGE_FREE     = 0,  /* not part of any slab */
	PAGE_RESERVED = 1,  /* VA belongs to a slab, no physical backing yet */
	PAGE_MAPPED   = 2,  /* physical 2 MiB page mapped + access set */
};

struct page_slot {
	uint8_t                       state;
	CUmemGenericAllocationHandle  handle;  /* valid iff state == MAPPED */
};

struct cipher_rt_kv_slab {
	struct cipher_rt_kv_page_tag tag;
	size_t      base_page;    /* index of first page in the pool */
	size_t      n_pages;      /* VA span in pages */
	size_t      mapped_pages; /* leading pages physically backed */
	int         in_use;
};

#define MAX_SLABS 4096

static struct {
	int                init_done;
	CUdevice           dev;
	CUcontext          primary_ctx;
	CUdeviceptr        pool_base;
	size_t             pool_bytes;
	size_t             page_size;
	size_t             n_pages;
	struct page_slot  *pages;       /* n_pages entries */
	struct cipher_rt_kv_slab slabs[MAX_SLABS];
	pthread_mutex_t    mu;
	struct cipher_rt_kv_alloc_stats stats;
} g = {
	.mu = PTHREAD_MUTEX_INITIALIZER,
};

static double now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

#define CK(call, onerr) do { \
	CUresult _r = (call); \
	if (_r != CUDA_SUCCESS) { \
		const char *_s = NULL; cuGetErrorString(_r, &_s); \
		fprintf(stderr, "[cipher-kvalloc] %s -> %d %s\n", \
		        #call, _r, _s ? _s : "?"); \
		onerr; \
	} \
} while (0)

int cipher_rt_kv_alloc_init(size_t va_pool_bytes)
{
	pthread_mutex_lock(&g.mu);
	if (g.init_done) {
		pthread_mutex_unlock(&g.mu);
		return 0;
	}

	/* cuInit is idempotent; PyTorch will already have called it, but a
	 * standalone unit test needs it too. */
	CK(cuInit(0), { pthread_mutex_unlock(&g.mu); return -1; });
	CK(cuDeviceGet(&g.dev, 0), { pthread_mutex_unlock(&g.mu); return -1; });
	CK(cuDevicePrimaryCtxRetain(&g.primary_ctx, g.dev),
	   { pthread_mutex_unlock(&g.mu); return -1; });
	CK(cuCtxSetCurrent(g.primary_ctx),
	   { pthread_mutex_unlock(&g.mu); return -1; });

	CUmemAllocationProp prop = {0};
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = g.dev;
	size_t gran = 0;
	CK(cuMemGetAllocationGranularity(&gran, &prop,
	                                 CU_MEM_ALLOC_GRANULARITY_MINIMUM),
	   { pthread_mutex_unlock(&g.mu); return -1; });
	g.page_size = gran;

	/* Round the pool up to a whole number of pages. */
	size_t pool = (va_pool_bytes + gran - 1) / gran * gran;
	CK(cuMemAddressReserve(&g.pool_base, pool, 0, 0, 0),
	   { pthread_mutex_unlock(&g.mu); return -1; });
	g.pool_bytes = pool;
	g.n_pages = pool / gran;

	g.pages = calloc(g.n_pages, sizeof(*g.pages));
	if (!g.pages) {
		cuMemAddressFree(g.pool_base, pool);
		pthread_mutex_unlock(&g.mu);
		return -1;
	}

	memset(&g.stats, 0, sizeof(g.stats));
	g.stats.bytes_va_reserved = pool;
	g.stats.page_size = gran;
	g.init_done = 1;
	pthread_mutex_unlock(&g.mu);

	fprintf(stderr, "[cipher-kvalloc] init: pool=%.1f GiB page=%zu KiB "
	        "n_pages=%zu (verified %s)\n",
	        pool / (1024.0 * 1024 * 1024), gran / 1024, g.n_pages,
	        CIPHER_RT_KV_ALLOC_VERIFIED_CUDA);
	return 0;
}

/* Caller holds g.mu. Map one pool page `gi` physically.
 *
 *   shared == 0  -> MISS: create a fresh physical page, map it, and
 *                   cuMemsetD8-ZERO it (op #1 invariant: a 2 MiB page
 *                   may be physical memory just freed by another
 *                   tenant; zeroing prevents a cross-tenant leak and
 *                   satisfies StaticLayer zero-init).
 *   shared != 0  -> HIT:  map the EXISTING shared physical page and
 *                   SKIP the zero — the page already holds the
 *                   deduped content; zeroing it would destroy it.
 *
 * The zero is a single conditional local to this function, gated on
 * the miss/hit decision this function itself makes — never a parameter
 * threaded from a caller (phase2_design.md, A3). */
static int map_one(size_t gi, CUmemGenericAllocationHandle shared)
{
	CUdeviceptr va = g.pool_base + (CUdeviceptr)gi * g.page_size;
	int fresh = (shared == 0);
	CUmemGenericAllocationHandle h = shared;

	if (fresh) {
		CUmemAllocationProp prop = {0};
		prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
		prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		prop.location.id = g.dev;
		/* T4.6.4: make every page exportable as a POSIX fd so the
		 * dedup path can hand its handle to the kmod. Harmless for
		 * the slab path — create/map/use/free are unchanged. */
		prop.requestedHandleTypes =
			CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
		CK(cuMemCreate(&h, g.page_size, &prop, 0), return -1);
	}
	CK(cuMemMap(va, g.page_size, 0, h, 0),
	   { if (fresh) cuMemRelease(h); return -1; });

	CUmemAccessDesc acc = {0};
	acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	acc.location.id = g.dev;
	acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	CK(cuMemSetAccess(va, g.page_size, &acc, 1),
	   { cuMemUnmap(va, g.page_size); if (fresh) cuMemRelease(h); return -1; });

	if (fresh)               /* A3: zero ONLY freshly-created physical */
		CK(cuMemsetD8(va, 0, g.page_size),
		   { cuMemUnmap(va, g.page_size); cuMemRelease(h); return -1; });

	g.pages[gi].handle = h;
	g.pages[gi].state = PAGE_MAPPED;
	return 0;
}

/* Caller holds g.mu. Map slab pages [from_page, to_page) physically.
 * The slab path is always MISS (fresh, exclusively-owned, zeroed). */
static int map_pages(struct cipher_rt_kv_slab *s, size_t from_page,
                     size_t to_page)
{
	double t0 = now_ms();
	for (size_t p = from_page; p < to_page; p++) {
		if (map_one(s->base_page + p, 0) != 0)
			return -1;
		g.stats.pages_mapped++;
		g.stats.pages_resident++;
	}
	g.stats.map_ms_total += now_ms() - t0;
	if (g.stats.pages_resident > g.stats.pages_resident_peak)
		g.stats.pages_resident_peak = g.stats.pages_resident;
	return 0;
}

cipher_rt_kv_slab_t *cipher_rt_kv_slab_create(
	const struct cipher_rt_kv_page_tag *tag,
	size_t max_bytes, size_t initial_bytes,
	unsigned long long *out_devptr)
{
	if (!g.init_done || !tag || max_bytes == 0)
		return NULL;

	pthread_mutex_lock(&g.mu);

	size_t want = (max_bytes + g.page_size - 1) / g.page_size;
	if (initial_bytes > max_bytes)
		initial_bytes = max_bytes;
	size_t init_pages = (initial_bytes + g.page_size - 1) / g.page_size;
	if (init_pages == 0)
		init_pages = 1;

	/* First-fit: find `want` consecutive FREE pages. */
	size_t run = 0, start = 0;
	int found = 0;
	for (size_t i = 0; i < g.n_pages; i++) {
		if (g.pages[i].state == PAGE_FREE) {
			if (run == 0) start = i;
			if (++run == want) { found = 1; break; }
		} else {
			run = 0;
		}
	}
	if (!found) {
		pthread_mutex_unlock(&g.mu);
		fprintf(stderr, "[cipher-kvalloc] slab_create: no VA run of "
		        "%zu pages\n", want);
		return NULL;
	}

	/* Find a free slab record. */
	struct cipher_rt_kv_slab *s = NULL;
	for (int i = 0; i < MAX_SLABS; i++) {
		if (!g.slabs[i].in_use) { s = &g.slabs[i]; break; }
	}
	if (!s) {
		pthread_mutex_unlock(&g.mu);
		fprintf(stderr, "[cipher-kvalloc] slab_create: slab table full\n");
		return NULL;
	}

	s->tag = *tag;
	s->base_page = start;
	s->n_pages = want;
	s->mapped_pages = 0;
	s->in_use = 1;
	for (size_t i = 0; i < want; i++)
		g.pages[start + i].state = PAGE_RESERVED;

	if (map_pages(s, 0, init_pages) != 0) {
		/* roll back */
		for (size_t i = 0; i < want; i++)
			g.pages[start + i].state = PAGE_FREE;
		s->in_use = 0;
		pthread_mutex_unlock(&g.mu);
		return NULL;
	}
	s->mapped_pages = init_pages;
	g.stats.slabs_created++;

	CUdeviceptr base = g.pool_base + (CUdeviceptr)start * g.page_size;
	if (out_devptr)
		*out_devptr = (unsigned long long)base;
	pthread_mutex_unlock(&g.mu);
	return s;
}

int cipher_rt_kv_slab_ensure(cipher_rt_kv_slab_t *s, size_t needed_bytes)
{
	if (!s || !s->in_use)
		return -1;
	pthread_mutex_lock(&g.mu);
	size_t need_pages = (needed_bytes + g.page_size - 1) / g.page_size;
	if (need_pages > s->n_pages)
		need_pages = s->n_pages;
	if (need_pages <= s->mapped_pages) {
		pthread_mutex_unlock(&g.mu);
		return 0;
	}
	int rc = map_pages(s, s->mapped_pages, need_pages);
	if (rc == 0)
		s->mapped_pages = need_pages;
	pthread_mutex_unlock(&g.mu);
	return rc;
}

void cipher_rt_kv_slab_free(cipher_rt_kv_slab_t *s)
{
	if (!s || !s->in_use)
		return;
	pthread_mutex_lock(&g.mu);
	for (size_t p = 0; p < s->n_pages; p++) {
		size_t gi = s->base_page + p;
		struct page_slot *slot = &g.pages[gi];
		if (slot->state == PAGE_MAPPED) {
			CUdeviceptr va = g.pool_base +
			    (CUdeviceptr)gi * g.page_size;
			CK(cuMemUnmap(va, g.page_size), {});
			CK(cuMemRelease(slot->handle), {});
			g.stats.pages_unmapped++;
			if (g.stats.pages_resident > 0)
				g.stats.pages_resident--;
		}
		slot->state = PAGE_FREE;
		slot->handle = 0;
	}
	s->in_use = 0;
	g.stats.slabs_freed++;
	pthread_mutex_unlock(&g.mu);
}

int cipher_rt_kv_page_info(unsigned long long devptr,
                           struct cipher_rt_kv_page_tag *out_tag)
{
	if (!g.init_done || !out_tag)
		return -1;
	pthread_mutex_lock(&g.mu);
	CUdeviceptr p = (CUdeviceptr)devptr;
	if (p < g.pool_base || p >= g.pool_base + g.pool_bytes) {
		pthread_mutex_unlock(&g.mu);
		return -1;  /* not ours — customer-managed */
	}
	size_t page = (size_t)((p - g.pool_base) / g.page_size);
	if (g.pages[page].state == PAGE_FREE) {
		pthread_mutex_unlock(&g.mu);
		return -1;
	}
	/* Find the owning slab. */
	for (int i = 0; i < MAX_SLABS; i++) {
		struct cipher_rt_kv_slab *s = &g.slabs[i];
		if (s->in_use && page >= s->base_page &&
		    page < s->base_page + s->n_pages) {
			*out_tag = s->tag;
			pthread_mutex_unlock(&g.mu);
			return 0;
		}
	}
	pthread_mutex_unlock(&g.mu);
	return -1;
}

void cipher_rt_kv_alloc_get_stats(struct cipher_rt_kv_alloc_stats *out)
{
	if (!out)
		return;
	pthread_mutex_lock(&g.mu);
	*out = g.stats;
	pthread_mutex_unlock(&g.mu);
}

void cipher_rt_kv_alloc_shutdown(void)
{
	pthread_mutex_lock(&g.mu);
	if (!g.init_done) {
		pthread_mutex_unlock(&g.mu);
		return;
	}
	for (int i = 0; i < MAX_SLABS; i++) {
		struct cipher_rt_kv_slab *s = &g.slabs[i];
		if (!s->in_use)
			continue;
		for (size_t p = 0; p < s->n_pages; p++) {
			size_t gi = s->base_page + p;
			if (g.pages[gi].state == PAGE_MAPPED) {
				CUdeviceptr va = g.pool_base +
				    (CUdeviceptr)gi * g.page_size;
				cuMemUnmap(va, g.page_size);
				cuMemRelease(g.pages[gi].handle);
			}
		}
		s->in_use = 0;
	}
	cuMemAddressFree(g.pool_base, g.pool_bytes);
	free(g.pages);
	g.pages = NULL;
	g.init_done = 0;
	pthread_mutex_unlock(&g.mu);
}

/* ===================================================================
 * Phase 4.6.3 — content-hash deduplication.
 * Additive path; shares g.mu (single mutex — phase2_design.md).
 * =================================================================== */

/* --- xxhash64 (seed 0; input length is always a multiple of 32 since
 *     a CIPHER page is 2 MiB, so the <32-byte tail path is omitted). --- */
#define XXH_P1 0x9E3779B185EBCA87ULL
#define XXH_P2 0xC2B2AE3D27D4EB4FULL
#define XXH_P3 0x165667B19E3779F9ULL
#define XXH_P4 0x85EBCA77C2B2AE63ULL

static inline uint64_t xxh_rotl(uint64_t x, int r)
{ return (x << r) | (x >> (64 - r)); }

static inline uint64_t xxh_round(uint64_t acc, uint64_t in)
{ acc += in * XXH_P2; acc = xxh_rotl(acc, 31); acc *= XXH_P1; return acc; }

static inline uint64_t xxh_merge(uint64_t acc, uint64_t v)
{ v = xxh_round(0, v); acc ^= v; acc = acc * XXH_P1 + XXH_P4; return acc; }

static uint64_t xxh64(const void *data, size_t len)  /* len % 32 == 0 */
{
	const uint8_t *p = (const uint8_t *)data;
	const uint8_t *end = p + len;
	uint64_t v1 = XXH_P1 + XXH_P2, v2 = XXH_P2, v3 = 0;
	uint64_t v4 = (uint64_t)0 - XXH_P1;
	uint64_t b;
	while (p < end) {
		memcpy(&b, p, 8); v1 = xxh_round(v1, b); p += 8;
		memcpy(&b, p, 8); v2 = xxh_round(v2, b); p += 8;
		memcpy(&b, p, 8); v3 = xxh_round(v3, b); p += 8;
		memcpy(&b, p, 8); v4 = xxh_round(v4, b); p += 8;
	}
	uint64_t h = xxh_rotl(v1, 1) + xxh_rotl(v2, 7) +
	             xxh_rotl(v3, 12) + xxh_rotl(v4, 18);
	h = xxh_merge(h, v1); h = xxh_merge(h, v2);
	h = xxh_merge(h, v3); h = xxh_merge(h, v4);
	h += (uint64_t)len;
	h ^= h >> 33; h *= XXH_P2; h ^= h >> 29; h *= XXH_P3; h ^= h >> 32;
	return h;
}

/* T4.6.4 — the dedup refcount table now lives in cipher_kmod
 * (cross-tenant). cipher_rt is the client: it computes xxhash64,
 * carves page VAs from the pool, exports/imports cuIpc POSIX handles,
 * does the memcmp-verify; the kmod owns refcounts + handle lifetime. */

static struct {
	int          init_done;
	int          kvd_fd;        /* /dev/cipher_kvdedup */
	uint32_t     tenant_id;
	CUdeviceptr  scratch_va;    /* HIT memcmp-verify scratch map */
	void        *pin_buf;       /* page_size pinned host buffer */
	uint64_t    *pooloff;       /* g.n_pages: pool page idx -> kmod offset */
} d;

/* Caller holds g.mu. Find a FREE pool page, mark RESERVED, return index. */
static long carve_free_page(void)
{
	for (size_t i = 0; i < g.n_pages; i++) {
		if (g.pages[i].state == PAGE_FREE) {
			g.pages[i].state = PAGE_RESERVED;
			return (long)i;
		}
	}
	return -1;
}

/* Caller holds g.mu. True iff `content` byte-equals the page behind
 * `stored` — full 2 MiB host memcmp (strict; never a hash compare). */
static int dedup_verify(const void *content,
                        CUmemGenericAllocationHandle stored)
{
	if (cuMemMap(d.scratch_va, g.page_size, 0, stored, 0) != CUDA_SUCCESS)
		return 0;
	CUmemAccessDesc acc = {0};
	acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	acc.location.id = g.dev;
	acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	int eq = 0;
	if (cuMemSetAccess(d.scratch_va, g.page_size, &acc, 1) == CUDA_SUCCESS &&
	    cuMemcpyDtoH(d.pin_buf, d.scratch_va, g.page_size) == CUDA_SUCCESS)
		eq = (memcmp(content, d.pin_buf, g.page_size) == 0);
	cuMemUnmap(d.scratch_va, g.page_size);
	return eq;
}

int cipher_rt_kv_dedup_init(void)
{
	pthread_mutex_lock(&g.mu);
	if (d.init_done) { pthread_mutex_unlock(&g.mu); return 0; }
	if (!g.init_done) {
		pthread_mutex_unlock(&g.mu);
		fprintf(stderr, "[cipher-kvalloc] dedup_init: alloc not initialized\n");
		return -1;
	}
	d.kvd_fd = open("/dev/" CIPHER_KVDEDUP_DEV_NAME, O_RDWR | O_CLOEXEC);
	if (d.kvd_fd < 0) {
		pthread_mutex_unlock(&g.mu);
		fprintf(stderr, "[cipher-kvalloc] dedup_init: open /dev/%s: %s\n",
		        CIPHER_KVDEDUP_DEV_NAME, strerror(errno));
		return -1;
	}
	struct cipher_kvdedup_init ini;
	memset(&ini, 0, sizeof(ini));
	if (ioctl(d.kvd_fd, CIPHER_KVDEDUP_INIT, &ini) != 0) {
		close(d.kvd_fd); d.kvd_fd = -1;
		pthread_mutex_unlock(&g.mu);
		return -1;
	}
	d.tenant_id = ini.tenant_id;
	CK(cuMemAddressReserve(&d.scratch_va, g.page_size, 0, 0, 0),
	   { close(d.kvd_fd); pthread_mutex_unlock(&g.mu); return -1; });
	CK(cuMemHostAlloc(&d.pin_buf, g.page_size, 0),
	   { close(d.kvd_fd); pthread_mutex_unlock(&g.mu); return -1; });
	d.pooloff = calloc(g.n_pages, sizeof(*d.pooloff));
	if (!d.pooloff) {
		close(d.kvd_fd);
		pthread_mutex_unlock(&g.mu);
		return -1;
	}
	d.init_done = 1;
	pthread_mutex_unlock(&g.mu);
	fprintf(stderr, "[cipher-kvalloc] dedup init: kmod-backed, tenant_id=%u\n",
	        d.tenant_id);
	return 0;
}

/* Caller holds g.mu. Unwind a freshly-mapped page after a failure. */
static void dedup_unwind_page(size_t gi, CUdeviceptr va)
{
	cuMemUnmap(va, g.page_size);
	cuMemRelease(g.pages[gi].handle);
	g.pages[gi].state = PAGE_FREE;
	g.pages[gi].handle = 0;
}

int cipher_rt_kv_dedup_put(const void *content, unsigned long long *out_devptr)
{
	if (!content)
		return -1;
	pthread_mutex_lock(&g.mu);
	if (!d.init_done) { pthread_mutex_unlock(&g.mu); return -1; }

	uint64_t h = xxh64(content, g.page_size);

	long gi = carve_free_page();
	if (gi < 0) {
		pthread_mutex_unlock(&g.mu);
		fprintf(stderr, "[cipher-kvalloc] dedup_put: pool full\n");
		return -1;
	}
	CUdeviceptr va = g.pool_base + (CUdeviceptr)gi * g.page_size;

	/* fresh exportable physical page; write the content */
	if (map_one((size_t)gi, 0) != 0) {
		g.pages[gi].state = PAGE_FREE;
		pthread_mutex_unlock(&g.mu);
		return -1;
	}
	if (cuMemcpyHtoD(va, content, g.page_size) != CUDA_SUCCESS) {
		dedup_unwind_page((size_t)gi, va);
		pthread_mutex_unlock(&g.mu);
		return -1;
	}

	int export_fd = -1;
	if (cuMemExportToShareableHandle(&export_fd, g.pages[gi].handle,
	      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) != CUDA_SUCCESS) {
		dedup_unwind_page((size_t)gi, va);
		pthread_mutex_unlock(&g.mu);
		return -1;
	}

	struct cipher_kvdedup_put p;
	memset(&p, 0, sizeof(p));
	p.content_hash = h;
	p.export_fd = export_fd;
	if (ioctl(d.kvd_fd, CIPHER_KVDEDUP_PUT, &p) != 0) {
		close(export_fd);
		dedup_unwind_page((size_t)gi, va);
		pthread_mutex_unlock(&g.mu);
		return -1;
	}

	if (p.result == CIPHER_KVDEDUP_RESULT_MISS) {
		close(export_fd);              /* kmod holds its own ref now */
		d.pooloff[gi] = p.pool_offset;
		if (out_devptr) *out_devptr = (unsigned long long)va;
		pthread_mutex_unlock(&g.mu);
		return 0;
	}

	/* HIT: import the candidate handle, memcmp-verify the content */
	CUmemGenericAllocationHandle shared = 0;
	int matched = 0;
	if (cuMemImportFromShareableHandle(&shared,
	      (void *)(intptr_t)p.candidate_fd,
	      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == CUDA_SUCCESS)
		matched = dedup_verify(content, shared);
	close(p.candidate_fd);

	if (matched) {
		/* swap this VA onto the shared physical page; drop our own */
		cuMemUnmap(va, g.page_size);
		cuMemRelease(g.pages[gi].handle);
		g.pages[gi].state = PAGE_RESERVED;
		if (map_one((size_t)gi, shared) != 0) {
			cuMemRelease(shared);
			g.pages[gi].state = PAGE_FREE;
			close(export_fd);
			pthread_mutex_unlock(&g.mu);
			return -1;
		}
		close(export_fd);
		struct cipher_kvdedup_confirm cf;
		cf.pool_offset = p.pool_offset;
		if (ioctl(d.kvd_fd, CIPHER_KVDEDUP_CONFIRM, &cf) != 0) {
			dedup_unwind_page((size_t)gi, va);
			pthread_mutex_unlock(&g.mu);
			return -1;
		}
		d.pooloff[gi] = p.pool_offset;
		if (out_devptr) *out_devptr = (unsigned long long)va;
		pthread_mutex_unlock(&g.mu);
		return 0;
	}

	/* xxhash64 collision: keep our own page, re-register FORCE_NEW */
	if (shared)
		cuMemRelease(shared);
	struct cipher_kvdedup_put p2;
	memset(&p2, 0, sizeof(p2));
	p2.content_hash = h;
	p2.export_fd = export_fd;
	p2.flags = CIPHER_KVDEDUP_FLAG_FORCE_NEW;
	if (ioctl(d.kvd_fd, CIPHER_KVDEDUP_PUT, &p2) != 0) {
		close(export_fd);
		dedup_unwind_page((size_t)gi, va);
		pthread_mutex_unlock(&g.mu);
		return -1;
	}
	close(export_fd);
	d.pooloff[gi] = p2.pool_offset;
	if (out_devptr) *out_devptr = (unsigned long long)va;
	pthread_mutex_unlock(&g.mu);
	return 0;
}

void cipher_rt_kv_dedup_free(unsigned long long devptr)
{
	pthread_mutex_lock(&g.mu);
	if (!d.init_done) { pthread_mutex_unlock(&g.mu); return; }
	CUdeviceptr p = (CUdeviceptr)devptr;
	if (p < g.pool_base || p >= g.pool_base + g.pool_bytes) {
		pthread_mutex_unlock(&g.mu);
		return;
	}
	size_t gi = (size_t)((p - g.pool_base) / g.page_size);
	if (g.pages[gi].state != PAGE_MAPPED) {
		pthread_mutex_unlock(&g.mu);
		return;
	}
	struct cipher_kvdedup_free f;
	f.pool_offset = d.pooloff[gi];
	(void)ioctl(d.kvd_fd, CIPHER_KVDEDUP_FREE, &f);
	CUdeviceptr va = g.pool_base + (CUdeviceptr)gi * g.page_size;
	cuMemUnmap(va, g.page_size);
	cuMemRelease(g.pages[gi].handle);
	g.pages[gi].state = PAGE_FREE;
	g.pages[gi].handle = 0;
	d.pooloff[gi] = 0;
	pthread_mutex_unlock(&g.mu);
}

void cipher_rt_kv_dedup_get_stats(struct cipher_rt_kv_dedup_stats *out)
{
	if (!out)
		return;
	memset(out, 0, sizeof(*out));
	pthread_mutex_lock(&g.mu);
	if (d.init_done) {
		struct cipher_kvdedup_stats s;
		memset(&s, 0, sizeof(s));
		if (ioctl(d.kvd_fd, CIPHER_KVDEDUP_STATS, &s) == 0) {
			out->puts             = s.puts;
			out->hits             = s.hits;
			out->misses           = s.misses;
			out->physical_pages   = s.entries;
			out->virtual_pages    = s.virtual_refs;
			out->hash_collisions  = s.collisions_forced;
			out->refcount_releases = s.refcount_releases;
		}
	}
	pthread_mutex_unlock(&g.mu);
}
