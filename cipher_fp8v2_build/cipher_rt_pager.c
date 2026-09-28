/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_pager.c -- distinct-model tiered-residence weight pager, STEP 1 (see cipher_rt_pager.h).
 *
 * Correctness-first cut: per-region mutex serializes {state, ref} (advisor: a lock-free acq_rel protocol here
 * is a StoreLoad/Dekker race; mutex held O(1), never across GPU work, is the correct first cut). Eviction
 * coherence via cuCtxSynchronize before cuMemUnmap (caller-side sync before unmap is mandatory anyway; the
 * NSLOTS event-ring is a correctness hole under fan-out collision -> deferred to per-stream events in STEP 2).
 *
 * Default-OFF, additive, NOT wired into cipher_v2_init_body -> OFF byte-identical.
 */
#define _GNU_SOURCE
#include "cipher_rt_pager.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* NEGATIVE CONTROL (test only): -DCIPHER_PAGER_NO_GPU_SYNC removes the eviction GPU-sync so the contention
 * test MUST corrupt/fault -- proving the test can detect a coherence violation. Never defined in production. */

enum { PG_EVICTED = 0, PG_PAGING_IN, PG_RESIDENT, PG_PAGING_OUT };
#define PG_MAX 256

struct pg_region {
	CUdeviceptr                  va;
	size_t                       size;      /* mapped/reserved region size (aligned) */
	size_t                       bump;      /* sub-alloc high-water (= weight bytes routed in); guarded by mu */
	size_t                       cur_live;  /* currently-live sub-alloc bytes (malloc +, free -); guarded by mu */
	size_t                       peak_live; /* max cur_live over the load = min reserve needed; guarded by mu */
	size_t                       warm_bytes;/* bytes to restore on page_in (== bump for alloc-fed, size for register) */
	unsigned long long           model_key; /* registry key (distinct-model identity) */
	void                        *warm;     /* pinned RAM warm copy */
	CUmemGenericAllocationHandle hbm;       /* valid while mapped (state >= RESIDENT or PAGING_OUT) */
	int                          state;     /* guarded by mu */
	int                          ref;       /* in-flight serve count; guarded by mu */
	pthread_mutex_t              mu;
	unsigned long                cold_miss, pagein_cnt, evict_cnt;  /* guarded by mu */
	unsigned long long           last_used;  /* LRU tick (residency manager); atomic-bumped */
	unsigned long long           freq;       /* cumulative access count (LFU-DA); atomic-bumped */
	unsigned long long           key;        /* LFU-DA key = freq + age-at-last-access (heuristic; aligned-64 racy-benign) */
	int                          used;
};

static struct pg_region g_rg[PG_MAX];
static int               g_npg = 0;
static pthread_mutex_t   g_reg_mu = PTHREAD_MUTEX_INITIALIZER;
static size_t            g_gran = 0;
static int               g_dev = 0;
static CUmemAllocationProp g_prop;
static int               g_ready = 0;
static int               g_pages_in_flight = 0;   /* PCIe queue depth; guarded by g_reg_mu */
static int               g_loading = -1;          /* region currently armed for weight routing; guarded by g_reg_mu */

static size_t alignup(size_t s) { return ((s + g_gran - 1) / g_gran) * g_gran; }

int cipher_pager_init(void)
{
	if (g_ready) return 0;
	if (cuInit(0) != CUDA_SUCCESS) return CIPHER_PAGER_ERR;
	cuCtxGetDevice(&g_dev);
	memset(&g_prop, 0, sizeof(g_prop));
	g_prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
	g_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	g_prop.location.id   = g_dev;
	if (cuMemGetAllocationGranularity(&g_gran, &g_prop,
	        CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
		return CIPHER_PAGER_ERR;
	g_ready = 1;
	fprintf(stderr, "[cipher-pager] init: dev=%d granularity=%zu KiB\n", g_dev, g_gran / 1024);
	return 0;
}

/* map: cuMemCreate + cuMemMap + cuMemSetAccess (mirror cipher_rt_kv_alloc.c:185-204). */
static int pg_map(struct pg_region *r)
{
	if (cuMemCreate(&r->hbm, r->size, &g_prop, 0) != CUDA_SUCCESS) return CIPHER_PAGER_ERR;
	if (cuMemMap(r->va, r->size, 0, r->hbm, 0) != CUDA_SUCCESS) { cuMemRelease(r->hbm); return CIPHER_PAGER_ERR; }
	CUmemAccessDesc acc; memset(&acc, 0, sizeof(acc));
	acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	acc.location.id   = g_dev;
	acc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	if (cuMemSetAccess(r->va, r->size, &acc, 1) != CUDA_SUCCESS) {
		cuMemUnmap(r->va, r->size); cuMemRelease(r->hbm); return CIPHER_PAGER_ERR;
	}
	return 0;
}

int cipher_pager_register(size_t size, const void *warm_data)
{
	if (!g_ready && cipher_pager_init() != 0) return -1;
	pthread_mutex_lock(&g_reg_mu);
	if (g_npg >= PG_MAX) { pthread_mutex_unlock(&g_reg_mu); return -1; }
	int id = g_npg++;
	pthread_mutex_unlock(&g_reg_mu);
	struct pg_region *r = &g_rg[id];
	memset(r, 0, sizeof(*r));
	r->size = alignup(size ? size : 1);
	pthread_mutex_init(&r->mu, NULL);
	if (cuMemAddressReserve(&r->va, r->size, 0, 0, 0) != CUDA_SUCCESS) return -1;
	if (cudaMallocHost(&r->warm, r->size) != cudaSuccess) { cuMemAddressFree(r->va, r->size); return -1; }
	if (warm_data) memcpy(r->warm, warm_data, size);
	r->warm_bytes = r->size;     /* register() path restores the whole region */
	r->state = PG_EVICTED;
	r->used = 1;
	return id;
}

int cipher_pager_page_in(int id)
{
	if (id < 0 || id >= g_npg || !g_rg[id].used) return CIPHER_PAGER_ERR;
	struct pg_region *r = &g_rg[id];
	pthread_mutex_lock(&r->mu);
	if (r->state != PG_EVICTED) { pthread_mutex_unlock(&r->mu); return CIPHER_PAGER_BUSY; }
	r->state = PG_PAGING_IN;
	pthread_mutex_unlock(&r->mu);

	pthread_mutex_lock(&g_reg_mu); g_pages_in_flight++; pthread_mutex_unlock(&g_reg_mu);
	int rc = pg_map(r);
	if (rc == 0) {
		/* restore weights from the RAM warm copy; sync so RESIDENT means data is in place.
		 * warm_bytes (== bump for alloc-fed regions) -- only the populated prefix holds weights. */
		if (cudaMemcpy((void *)r->va, r->warm, r->warm_bytes, cudaMemcpyHostToDevice) != cudaSuccess) rc = CIPHER_PAGER_ERR;
		else if (cudaDeviceSynchronize() != cudaSuccess) rc = CIPHER_PAGER_ERR;
	}
	pthread_mutex_lock(&g_reg_mu); g_pages_in_flight--; pthread_mutex_unlock(&g_reg_mu);

	pthread_mutex_lock(&r->mu);
	if (rc == 0) { r->state = PG_RESIDENT; r->pagein_cnt++; }
	else         { r->state = PG_EVICTED; }   /* map failed -> stay evicted */
	pthread_mutex_unlock(&r->mu);
	return rc;
}

int cipher_pager_page_out(int id)
{
	if (id < 0 || id >= g_npg || !g_rg[id].used) return CIPHER_PAGER_ERR;
	struct pg_region *r = &g_rg[id];
	pthread_mutex_lock(&r->mu);
	if (r->state != PG_RESIDENT) { pthread_mutex_unlock(&r->mu); return CIPHER_PAGER_BUSY; }
	r->state = PG_PAGING_OUT;          /* publish: new serve_begin now MISSes */
	pthread_mutex_unlock(&r->mu);

	/* drain: wait until all serves that passed serve_begin have called serve_end (finished ISSUING). */
	for (;;) {
		pthread_mutex_lock(&r->mu);
		int ref = r->ref;
		pthread_mutex_unlock(&r->mu);
		if (ref == 0) break;
		sched_yield();
	}
#ifndef CIPHER_PAGER_NO_GPU_SYNC
	/* eviction coherence: all ISSUED GPU reads of va must COMPLETE before unmap. cuMemUnmap does not
	 * implicitly synchronize. (STEP 2: replace this global sync with per-stream completion events.) */
	cuCtxSynchronize();
#endif
	cuMemUnmap(r->va, r->size);
	cuMemRelease(r->hbm);
	pthread_mutex_lock(&r->mu);
	r->state = PG_EVICTED; r->evict_cnt++;
	pthread_mutex_unlock(&r->mu);
	return CIPHER_PAGER_OK;
}

int cipher_pager_serve_begin(int id, unsigned long long *va_out)
{
	if (id < 0 || id >= g_npg || !g_rg[id].used) return CIPHER_PAGER_MISS;
	struct pg_region *r = &g_rg[id];
	pthread_mutex_lock(&r->mu);
	if (r->state == PG_RESIDENT) {
		r->ref++;
		if (va_out) *va_out = (unsigned long long)r->va;
		pthread_mutex_unlock(&r->mu);
		return CIPHER_PAGER_HIT;
	}
	r->cold_miss++;
	pthread_mutex_unlock(&r->mu);
	return CIPHER_PAGER_MISS;
}

void cipher_pager_serve_end(int id)
{
	if (id < 0 || id >= g_npg || !g_rg[id].used) return;
	struct pg_region *r = &g_rg[id];
	pthread_mutex_lock(&r->mu);
	if (r->ref > 0) r->ref--;
	pthread_mutex_unlock(&r->mu);
}

int cipher_pager_state(int id)
{
	if (id < 0 || id >= g_npg || !g_rg[id].used) return -1;
	struct pg_region *r = &g_rg[id];
	pthread_mutex_lock(&r->mu); int s = r->state; pthread_mutex_unlock(&r->mu);
	return s;
}

void cipher_pager_get_stats(int id, struct cipher_pager_stats *out)
{
	if (!out || id < 0 || id >= g_npg || !g_rg[id].used) return;
	struct pg_region *r = &g_rg[id];
	pthread_mutex_lock(&r->mu);
	out->cold_miss = r->cold_miss; out->pagein_cnt = r->pagein_cnt; out->evict_cnt = r->evict_cnt;
	out->state = r->state; out->ref = r->ref;
	out->used_bytes = r->bump; out->base_va = (unsigned long long)r->va;
	out->peak_live = r->peak_live; out->cur_live = r->cur_live;
	pthread_mutex_unlock(&r->mu);
	pthread_mutex_lock(&g_reg_mu); out->pages_in_flight = g_pages_in_flight; pthread_mutex_unlock(&g_reg_mu);
}

/* ===================== per-allocation routing (STEP 1.5) ===================== *
 * The torch MemPool installs cipher_pager_malloc/free; weight allocations made inside the load window
 * (begin_load..end_load, expressed in Python as `with torch.cuda.use_mem_pool(pager_pool)`) bump from the
 * armed region's cuMemMap VA, so the model's weights live behind a pageable region. Everything else (KV /
 * activation / scratch, allocated OUTSIDE the pool) uses torch's default allocator -- coexistence intact. */

int cipher_pager_begin_load(unsigned long long model_key, size_t reserve_bytes)
{
	if (!g_ready && cipher_pager_init() != 0) return -1;
	pthread_mutex_lock(&g_reg_mu);
	if (g_loading >= 0)      { pthread_mutex_unlock(&g_reg_mu); fprintf(stderr, "[cipher-pager] begin_load: a load is already armed (loads are serial)\n"); return -1; }
	if (g_npg >= PG_MAX)     { pthread_mutex_unlock(&g_reg_mu); return -1; }
	int id = g_npg++;
	pthread_mutex_unlock(&g_reg_mu);
	struct pg_region *r = &g_rg[id];
	memset(r, 0, sizeof(*r));
	r->size = alignup(reserve_bytes ? reserve_bytes : g_gran);
	r->model_key = model_key;
	pthread_mutex_init(&r->mu, NULL);
	if (cuMemAddressReserve(&r->va, r->size, 0, 0, 0) != CUDA_SUCCESS) return -1;
	if (pg_map(r) != 0) { cuMemAddressFree(r->va, r->size); return -1; }   /* HBM mapped; weights write here during load */
	r->bump = 0;
	r->state = PG_PAGING_IN;   /* mapped but warm not yet captured -> serve MISSes until end_load */
	r->used = 1;
	pthread_mutex_lock(&g_reg_mu); g_loading = id; pthread_mutex_unlock(&g_reg_mu);
	fprintf(stderr, "[cipher-pager] begin_load id=%d key=0x%llx reserve=%zu MiB\n", id, model_key, r->size >> 20);
	return id;
}

int cipher_pager_end_load(int id)
{
	if (id < 0 || id >= g_npg || !g_rg[id].used) return CIPHER_PAGER_ERR;
	struct pg_region *r = &g_rg[id];
	pthread_mutex_lock(&r->mu);
	size_t used = r->bump;
	pthread_mutex_unlock(&r->mu);
	/* capture the warm copy of the populated [0,used) bytes (the model's weights, just written by the load) */
	if (cudaMallocHost(&r->warm, used ? used : 1) != cudaSuccess) return CIPHER_PAGER_ERR;
	if (used && cudaMemcpy(r->warm, (void *)r->va, used, cudaMemcpyDeviceToHost) != cudaSuccess) return CIPHER_PAGER_ERR;
	if (cudaDeviceSynchronize() != cudaSuccess) return CIPHER_PAGER_ERR;
	pthread_mutex_lock(&r->mu);
	r->warm_bytes = used;
	r->state = PG_RESIDENT;
	pthread_mutex_unlock(&r->mu);
	pthread_mutex_lock(&g_reg_mu); if (g_loading == id) g_loading = -1; pthread_mutex_unlock(&g_reg_mu);
	fprintf(stderr, "[cipher-pager] end_load id=%d weights=%zu MiB -> RESIDENT\n", id, used >> 20);
	return 0;
}

void *cipher_pager_malloc(size_t size, int device, void *stream)
{
	(void)device; (void)stream;
	int id;
	pthread_mutex_lock(&g_reg_mu); id = g_loading; pthread_mutex_unlock(&g_reg_mu);
	if (id >= 0 && id < g_npg && g_rg[id].used) {
		struct pg_region *r = &g_rg[id];
		pthread_mutex_lock(&r->mu);
		size_t off = (r->bump + 511) & ~((size_t)511);    /* 512B align (torch tensor alignment) */
		if (off + size > r->size) {                       /* under-reserved -> surface as OOM, never silent passthrough */
			pthread_mutex_unlock(&r->mu);
			fprintf(stderr, "[cipher-pager] malloc OOM id=%d need=%zu off=%zu cap=%zu (raise reserve)\n", id, size, off, r->size);
			return NULL;
		}
		void *p = (void *)(uintptr_t)(r->va + off);
		r->bump = off + size;
		r->cur_live += size;                                  /* live tracking (compaction sizing) */
		if (r->cur_live > r->peak_live) r->peak_live = r->cur_live;
		pthread_mutex_unlock(&r->mu);
		return p;
	}
	/* no load armed -> passthrough (defensive; MemPool scoping normally prevents this) */
	void *p = NULL;
	if (cudaMalloc(&p, size) != cudaSuccess) return NULL;
	return p;
}

void cipher_pager_free(void *ptr, size_t size, int device, void *stream)
{
	(void)device; (void)stream;
	if (!ptr) return;
	CUdeviceptr p = (CUdeviceptr)(uintptr_t)ptr;
	/* in-region pointer -> bump sub-alloc, freed wholesale at page_out/release (space not reclaimed in [0,bump),
	 * but track cur_live so compaction knows the true live footprint vs the bump high-water). */
	for (int i = 0; i < g_npg; i++) {
		struct pg_region *r = &g_rg[i];
		if (r->used && p >= r->va && p < r->va + r->size) {
			pthread_mutex_lock(&r->mu);
			if (r->cur_live >= size) r->cur_live -= size;
			pthread_mutex_unlock(&r->mu);
			return;
		}
	}
	cudaFree(ptr);   /* passthrough free of a non-pager allocation */
}

int cipher_pager_find(unsigned long long model_key)
{
	for (int i = 0; i < g_npg; i++)
		if (g_rg[i].used && g_rg[i].model_key == model_key) return i;
	return -1;
}

/* ===================== multi-model co-residence residency manager (STEP 2-arch) ===================== *
 * N distinct models' regions in ONE process/HBM, each pageable via the proven per-region state machine. A
 * resident-HBM budget; serve_demand pages in an evicted model on demand, evicting LRU ref==0 victims to fit.
 * Lock order: g_mgr_mu -> region mu (the HIT fast path takes only the region lock). The page_in-then-serve_begin
 * is performed ATOMICALLY under g_mgr_mu so the demanded model holds a ref before the manager lock drops --
 * otherwise a concurrent victim-selection could evict it in the gap (livelock; advisor catch #1). */
static pthread_mutex_t   g_mgr_mu = PTHREAD_MUTEX_INITIALIZER;
static size_t            g_mgr_budget = 0;       /* 0 = unlimited (manager inactive) */
static size_t            g_mgr_resident = 0;     /* sum of resident region sizes; guarded by g_mgr_mu */
static unsigned long long g_mgr_clock = 0;       /* LRU tick (atomic) */
static unsigned long     g_mgr_wasted = 0;       /* page_ins whose serve then MISSed (livelock detector) */

/* frequency-aware residency policy (DEFAULT-OFF: LRU). CIPHER_PAGER_POLICY=lfu|lfuda selects LFU with Dynamic Aging:
 * evict the min-key RESIDENT region, key = freq + age, where age is the key of the last-evicted victim (a monotone
 * non-decreasing floor under g_mgr_mu). The aging floor lets a newly-popular model overtake a stale-hot one --
 * curing pure-LFU's no-aging pin under a popularity shift; on a STATIONARY workload age advances slowly so it
 * behaves as plain LFU. The policy is a HEURISTIC over WHICH ref==0/RESIDENT region to evict -- it is NEVER a
 * correctness input (any ref==0 RESIDENT region is safe to evict), so it cannot perturb the eviction-during-use
 * coherence invariant (state/ref/cuCtxSynchronize). Fixed at mgr_init from env; LRU is the byte-identical fallback. */
enum { CIPHER_POLICY_LRU = 0, CIPHER_POLICY_LFUDA = 1 };
static int                g_mgr_policy = CIPHER_POLICY_LRU;  /* set in mgr_init from env; default LRU */
static unsigned long long g_mgr_age    = 0;                  /* LFU-DA aging factor; written under g_mgr_mu */

/* record an access for the residency policy. last_used (LRU) + freq/key (LFU-DA). The g_mgr_age read here is a
 * benign-racy aligned-64 load on the HIT fast path (no g_mgr_mu) -- identical discipline to last_used; the policy
 * value is a heuristic, never a correctness input, so a slightly-stale age only nudges a future victim choice. */
static inline void mgr_touch(struct pg_region *r)
{
	r->last_used = __sync_add_and_fetch(&g_mgr_clock, 1);
	unsigned long long f = __sync_add_and_fetch(&r->freq, 1);
	r->key = f + g_mgr_age;
}

/* caller holds g_mgr_mu. Evict a victim (RESIDENT, ref==0, id != keep) until resident+need <= budget; -1 if stuck.
 * LRU (default): smallest last_used. LFU-DA: smallest key; on eviction the age floor rises to the victim's key. */
static int mgr_make_room(size_t need, int keep)
{
	while (g_mgr_budget && g_mgr_resident + need > g_mgr_budget) {
		int victim = -1; unsigned long long best = ~0ULL;
		for (int i = 0; i < g_npg; i++) {
			struct pg_region *r = &g_rg[i];
			if (i == keep || !r->used) continue;
			pthread_mutex_lock(&r->mu);
			int cand = (r->state == PG_RESIDENT && r->ref == 0);
			unsigned long long score = (g_mgr_policy == CIPHER_POLICY_LFUDA) ? r->key : r->last_used;
			pthread_mutex_unlock(&r->mu);
			if (cand && score < best) { best = score; victim = i; }
		}
		if (victim < 0) return -1;                 /* nothing evictable (all in use / referenced) */
		size_t vb = g_rg[victim].size;
		if (cipher_pager_page_out(victim) == CIPHER_PAGER_OK) {
			g_mgr_resident -= vb;
			if (g_mgr_policy == CIPHER_POLICY_LFUDA) g_mgr_age = best;  /* aging: floor rises to evicted victim's key */
		}
		/* else: victim raced out of RESIDENT (e.g. just took a ref); loop re-selects */
	}
	return 0;
}

/* caller holds g_mgr_mu. Select the eviction policy from CIPHER_PAGER_POLICY (default LRU). */
static void mgr_policy_init(void)
{
	const char *p = getenv("CIPHER_PAGER_POLICY");
	if (p && (strcmp(p, "lfu") == 0 || strcmp(p, "lfuda") == 0)) g_mgr_policy = CIPHER_POLICY_LFUDA;
	else                                                          g_mgr_policy = CIPHER_POLICY_LRU;
}

int cipher_pager_mgr_init(size_t budget_bytes)
{
	if (!g_ready && cipher_pager_init() != 0) return CIPHER_PAGER_ERR;
	pthread_mutex_lock(&g_mgr_mu);
	mgr_policy_init();
	g_mgr_budget = budget_bytes;
	g_mgr_resident = 0;
	for (int i = 0; i < g_npg; i++) {
		if (!g_rg[i].used) continue;
		pthread_mutex_lock(&g_rg[i].mu);
		if (g_rg[i].state == PG_RESIDENT) g_mgr_resident += g_rg[i].size;
		pthread_mutex_unlock(&g_rg[i].mu);
	}
	mgr_make_room(0, -1);                            /* evict down to budget (policy-selected victims) */
	pthread_mutex_unlock(&g_mgr_mu);
	fprintf(stderr, "[cipher-pager] mgr_init budget=%zu MiB resident=%zu MiB policy=%s\n",
	        budget_bytes >> 20, g_mgr_resident >> 20, (g_mgr_policy == CIPHER_POLICY_LFUDA) ? "lfu-da" : "lru");
	return 0;
}

int cipher_pager_serve_demand(int id, unsigned long long *va_out)
{
	if (id < 0 || id >= g_npg || !g_rg[id].used) return CIPHER_PAGER_MISS;
	/* fast HIT path: region lock only (no manager lock) */
	int rc = cipher_pager_serve_begin(id, va_out);
	if (rc == CIPHER_PAGER_HIT) {
#ifdef CIPHER_RESIDENCY_BREAK_LOCKORDER
		/* NEGATIVE CONTROL: take region lock then manager lock = REVERSED order -> AB-BA deadlock under contention */
		pthread_mutex_lock(&g_rg[id].mu); pthread_mutex_lock(&g_mgr_mu);
		g_rg[id].last_used = ++g_mgr_clock;
		g_rg[id].freq++; g_rg[id].key = g_rg[id].freq + g_mgr_age;
		pthread_mutex_unlock(&g_mgr_mu); pthread_mutex_unlock(&g_rg[id].mu);
#else
		mgr_touch(&g_rg[id]);
#endif
		return CIPHER_PAGER_HIT;
	}
	/* MISS -> residency change under manager lock */
	pthread_mutex_lock(&g_mgr_mu);
	if (cipher_pager_state(id) != PG_RESIDENT) {     /* re-check: another thread may have paged it in */
		size_t need = g_rg[id].size;
		if (mgr_make_room(need, id) != 0) { pthread_mutex_unlock(&g_mgr_mu); return CIPHER_PAGER_MISS; }
		if (cipher_pager_page_in(id) == CIPHER_PAGER_OK) g_mgr_resident += need;
		else { pthread_mutex_unlock(&g_mgr_mu); return CIPHER_PAGER_ERR; }
	}
#ifdef CIPHER_RESIDENCY_BREAK_LIVELOCK
	/* NEGATIVE CONTROL: release the manager lock BEFORE taking the ref -> eviction window -> livelock/wasted page_ins */
	pthread_mutex_unlock(&g_mgr_mu);
	rc = cipher_pager_serve_begin(id, va_out);
	if (rc != CIPHER_PAGER_HIT) __sync_fetch_and_add(&g_mgr_wasted, 1);
#else
	/* CORRECT (advisor #1): take the ref while still holding g_mgr_mu (g_mgr_mu->region order) so victim
	 * selection cannot evict id in the gap; the demanded model is ref-protected before the lock drops. */
	rc = cipher_pager_serve_begin(id, va_out);
	if (rc != CIPHER_PAGER_HIT) g_mgr_wasted++;       /* should be impossible: ref taken under the lock */
	else mgr_touch(&g_rg[id]);                         /* g_mgr_age read is fully synchronized here (hold g_mgr_mu) */
	pthread_mutex_unlock(&g_mgr_mu);
#endif
	return rc;
}

unsigned long long cipher_pager_mgr_resident_bytes(void)
{
	pthread_mutex_lock(&g_mgr_mu); unsigned long long b = g_mgr_resident; pthread_mutex_unlock(&g_mgr_mu); return b;
}

unsigned long cipher_pager_mgr_wasted(void)
{
	pthread_mutex_lock(&g_mgr_mu); unsigned long w = g_mgr_wasted; pthread_mutex_unlock(&g_mgr_mu); return w;
}

/* diagnostic: cumulative access count for a region (proves the LFU-DA hot set accumulated frequency). */
unsigned long long cipher_pager_mgr_freq(int id)
{
	if (id < 0 || id >= g_npg || !g_rg[id].used) return 0;
	struct pg_region *r = &g_rg[id];
	pthread_mutex_lock(&r->mu); unsigned long long f = r->freq; pthread_mutex_unlock(&r->mu);
	return f;
}

/* determinism-independent restore proof: full chunked D2H checksum of the live region [0,warm_bytes). */
unsigned long long cipher_pager_live_cksum(int id)
{
	if (id < 0 || id >= g_npg || !g_rg[id].used) return 0;
	struct pg_region *r = &g_rg[id];
	pthread_mutex_lock(&r->mu);
	int resident = (r->state == PG_RESIDENT);
	size_t n = r->warm_bytes ? r->warm_bytes : r->bump;
	CUdeviceptr va = r->va;
	pthread_mutex_unlock(&r->mu);
	if (!resident || n == 0) return 0;
	const size_t CH = 16u * 1024 * 1024;
	unsigned char *buf = NULL;
	if (cudaMallocHost((void **)&buf, CH) != cudaSuccess) return 0;
	unsigned long long h = 1469598103934665603ULL;   /* FNV-1a 64 */
	for (size_t off = 0; off < n; off += CH) {
		size_t m = (n - off < CH) ? (n - off) : CH;
		if (cudaMemcpy(buf, (void *)(uintptr_t)(va + off), m, cudaMemcpyDeviceToHost) != cudaSuccess) { cudaFreeHost(buf); return 0; }
		for (size_t i = 0; i < m; i++) { h ^= buf[i]; h *= 1099511628211ULL; }
	}
	cudaFreeHost(buf);
	return h;
}
