/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_pager.h -- distinct-model tiered-residence weight pager (ORCHESTRATE G-O2/G-O8/G-O9), STEP 1.
 *
 * Residency state machine (EVICTED/PAGING_IN/RESIDENT/PAGING_OUT) + the eviction-during-use coherence
 * invariant: NEVER serve a half-evicted region, NEVER unmap a region with an in-flight GPU read, without a
 * per-serve sync on the hot path. Mechanism (STEP 1, correctness-first):
 *   - {state, ref} transitions serialized by a per-region mutex (held O(1), NEVER across GPU work).
 *   - serve hot path: serve_begin (lock; check RESIDENT; ref++; unlock) -> [caller issues GPU read of va] ->
 *     serve_end (lock; ref--; unlock). NO sync on this path.
 *   - page_out (the evictor, off the serve hot path): CAS RESIDENT->PAGING_OUT; drain ref==0; cuCtxSynchronize
 *     (all in-flight GPU reads complete); cuMemUnmap+cuMemRelease; ->EVICTED.
 * (Per-stream completion-event-gated unmap, to replace the global cuCtxSynchronize, is the STEP-2 optimization.)
 *
 * Default-OFF, additive: not wired into cipher_v2_init_body; a library for the correctness test + the future
 * vLLM weight-allocation routing op. NOT yet serving real model weights (routing + vLLM-allocator coexistence
 * is the next op). Mirrors the proven cuMemMap pattern in cipher_rt_kv_alloc.c:127-208.
 */
#ifndef CIPHER_RT_PAGER_H
#define CIPHER_RT_PAGER_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

enum {
	CIPHER_PAGER_HIT   = 0,  /* serve_begin: region RESIDENT, va valid, ref taken */
	CIPHER_PAGER_OK    = 0,
	CIPHER_PAGER_BUSY  = 1,  /* state was not the required precondition (another transition in flight) */
	CIPHER_PAGER_ERR   = 2,  /* CUDA/VMM error */
	CIPHER_PAGER_MISS  = 3,  /* serve_begin: region not RESIDENT (being paged) -> caller must page_in */
};

struct cipher_pager_stats {
	unsigned long cold_miss;     /* serve_begin MISS count (would-be cold pulls) */
	unsigned long pagein_cnt;    /* completed page_ins */
	unsigned long evict_cnt;     /* completed page_outs */
	int state;                   /* current state */
	int ref;                     /* current in-flight serve count */
	int pages_in_flight;         /* global: regions currently PAGING_IN (PCIe queue depth) */
	unsigned long used_bytes;    /* bump high-water = bytes routed into this region (== weight bytes) */
	unsigned long long base_va;  /* region base VA (for residency .data_ptr() in-range checks) */
	unsigned long peak_live;     /* max simultaneously-live sub-alloc bytes during load (min reserve needed) */
	unsigned long cur_live;      /* currently-live sub-alloc bytes (bump - freed; the true tight footprint) */
};

int  cipher_pager_init(void);                                  /* resolve VMM + granularity; 0 ok */
int  cipher_pager_register(size_t size, const void *warm_data);/* -> region id >=0, or -1 */
int  cipher_pager_page_in(int id);                             /* EVICTED -> RESIDENT */
int  cipher_pager_page_out(int id);                            /* RESIDENT -> EVICTED (eviction-safe) */
int  cipher_pager_serve_begin(int id, unsigned long long *va_out); /* HIT(0)+va, or MISS(3) */
void cipher_pager_serve_end(int id);
int  cipher_pager_state(int id);
void cipher_pager_get_stats(int id, struct cipher_pager_stats *out);

/* ---- per-allocation routing (STEP 1.5): a per-model pageable region fed by a torch pluggable allocator ---- *
 * begin_load reserves+maps a region and arms routing; cipher_pager_malloc/free are the torch MemPool allocator
 * (weight allocs bump from the region's cuMemMap VA -> pageable); end_load captures the warm copy + -> RESIDENT.
 * g_loading is a single global -> loads are SERIAL (STEP-1 boundary; concurrent begin_loads would mis-route). */
int  cipher_pager_begin_load(unsigned long long model_key, size_t reserve_bytes); /* -> region id, arms routing */
int  cipher_pager_end_load(int id);                            /* capture warm(bump) + PAGING_IN->RESIDENT */
void *cipher_pager_malloc(size_t size, int device, void *stream);       /* torch CUDAPluggableAllocator malloc */
void  cipher_pager_free(void *ptr, size_t size, int device, void *stream);/* torch CUDAPluggableAllocator free */
int  cipher_pager_find(unsigned long long model_key);          /* registry lookup -> region id, or -1 */
unsigned long long cipher_pager_live_cksum(int id);            /* full chunked D2H checksum of live [0,bump) */

/* ---- multi-model co-residence residency manager (STEP 2-arch): N distinct models in ONE process/HBM ---- *
 * Builds on the proven per-region state machine + eviction-during-use coherence. A resident-HBM budget; on a
 * serve to an EVICTED model, demand-page-in (evicting LRU ref==0 victims to make room). Lock order g_mgr_mu ->
 * region mu (HIT fast-path takes the region lock only); the page_in-then-serve_begin is ATOMIC under g_mgr_mu so
 * the just-paged model holds a ref before the manager lock drops (no livelock). Default-OFF (budget 0 = inactive). */
int  cipher_pager_mgr_init(size_t budget_bytes);               /* set resident-HBM budget; evict down to it (policy below) */
int  cipher_pager_serve_demand(int id, unsigned long long *va_out); /* HIT (ref held) or demand-page-in then HIT; MISS if unfittable */
unsigned long long cipher_pager_mgr_resident_bytes(void);      /* current sum of resident region sizes */
unsigned long      cipher_pager_mgr_wasted(void);              /* page_ins whose serve then MISSed (==0 iff no livelock) */
unsigned long long cipher_pager_mgr_freq(int id);              /* diagnostic: cumulative access count for a region */

/* Eviction policy (default-OFF=LRU): env CIPHER_PAGER_POLICY=lfu|lfuda selects LFU with Dynamic Aging (evict the
 * least-frequently-used cold region, with an aging floor that lets a newly-popular model overtake a stale-hot one).
 * Frequency-skewed (Zipfian) traffic keeps the hot set pinned, so a cold-model burst can't evict an about-to-be-hit
 * hot model -> fewer cold misses. The policy chooses WHICH ref==0/RESIDENT region to evict; it is never a correctness
 * input, so the eviction-during-use coherence invariant is unaffected. Fixed at mgr_init time; LRU is the fallback. */

#ifdef __cplusplus
}
#endif
#endif
