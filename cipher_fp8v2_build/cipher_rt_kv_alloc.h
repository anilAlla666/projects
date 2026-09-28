/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_kv_alloc.h - Phase 4.6.2 KV page allocator (CUDA VMM).
 *
 * The KV memory layer. A single large virtual-address pool is reserved
 * once; per-(tenant,seq,layer,head_kv,role) "slabs" carve VA sub-ranges
 * out of it, and physical 2 MiB pages are mapped on demand as a
 * sequence's KV cache grows during decode.
 *
 * Each physical page carries a tag identifying what it holds. This is
 * the structural prerequisite for T4.6.3 (content-hash dedup) and
 * T4.6.4 (cuIpc cross-tenant export) — both sit on top of this.
 *
 * Design decisions (see PHASE_4_T4_6_2 prompt D1-D7, revised):
 *  - 2 MiB physical page granularity (CUDA VMM minimum on H100/CUDA13).
 *  - One per-process VA pool, slabs are first-fit sub-ranges.
 *  - INLINE allocation, no background thread: T4.6.2 S1 measured VMM
 *    APIs at ~0.04 ms/page on this stack (vs vAttention paper's ~40 ms);
 *    the background-overlap machinery the paper needed is unnecessary
 *    here.
 *  - content_hash field reserved, left 0 by T4.6.2 (T4.6.3 fills it).
 *
 * No driver fork. Pure user-space CUDA driver API.
 */
#ifndef CIPHER_RT_KV_ALLOC_H
#define CIPHER_RT_KV_ALLOC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum cipher_rt_kv_role {
	CIPHER_RT_KV_ROLE_K = 0,
	CIPHER_RT_KV_ROLE_V = 1,
};

/* Identifies what a slab (and its pages) holds. Stable layout —
 * append-only. */
struct cipher_rt_kv_page_tag {
	uint32_t tenant_id;     /* from GREEN_CTX / kmod tenant registry */
	uint32_t seq_id;        /* sequence / request id within the tenant */
	uint16_t layer;         /* transformer layer index */
	uint16_t head_kv;       /* KV head index (GQA): 0xFFFF = whole-tensor slab */
	uint8_t  role;          /* enum cipher_rt_kv_role */
	uint8_t  _pad[3];
	uint64_t content_hash;  /* FNV-1a of contents; 0 until T4.6.3 sets it */
};

/* Opaque slab handle. */
typedef struct cipher_rt_kv_slab cipher_rt_kv_slab_t;

/* Initialize the allocator: reserve a `va_pool_bytes` virtual-address
 * pool (rounded up to page granularity). Idempotent — safe to call from
 * a lazy first-use path. Returns 0 on success, -1 on failure (failure
 * is non-fatal to the host: callers degrade to customer-managed KV). */
int cipher_rt_kv_alloc_init(size_t va_pool_bytes);

/* Tear down: free all slabs and the VA pool. */
void cipher_rt_kv_alloc_shutdown(void);

/* Create a slab: reserve `max_bytes` of VA (rounded to page), physically
 * map `initial_bytes` worth of pages. The slab's pages are tagged with
 * *tag. On success returns the slab handle and writes the base device
 * pointer to *out_devptr. Returns NULL on failure. */
cipher_rt_kv_slab_t *cipher_rt_kv_slab_create(
	const struct cipher_rt_kv_page_tag *tag,
	size_t max_bytes, size_t initial_bytes,
	unsigned long long *out_devptr);

/* Ensure [base, base + needed_bytes) of the slab is physically backed.
 * Maps additional 2 MiB pages as required. Cheap/no-op if already
 * backed. Returns 0 on success, -1 on failure. */
int cipher_rt_kv_slab_ensure(cipher_rt_kv_slab_t *slab, size_t needed_bytes);

/* Free a slab: unmap + release all physical pages, return the VA range
 * to the pool free list. */
void cipher_rt_kv_slab_free(cipher_rt_kv_slab_t *slab);

/* Reverse lookup: given any device pointer that falls inside a managed
 * slab, write its page tag to *out_tag. Returns 0 if found and managed,
 * -1 otherwise (pointer is customer-managed / not ours). */
int cipher_rt_kv_page_info(unsigned long long devptr,
                           struct cipher_rt_kv_page_tag *out_tag);

/* Telemetry snapshot. */
struct cipher_rt_kv_alloc_stats {
	uint64_t slabs_created;
	uint64_t slabs_freed;
	uint64_t pages_mapped;       /* cumulative cuMemMap count */
	uint64_t pages_unmapped;     /* cumulative cuMemUnmap count */
	uint64_t pages_resident;     /* currently mapped */
	uint64_t pages_resident_peak;
	uint64_t bytes_va_reserved;
	uint64_t page_size;
	double   map_ms_total;       /* cumulative create+map+setaccess time */
};
void cipher_rt_kv_alloc_get_stats(struct cipher_rt_kv_alloc_stats *out);

/* ===================================================================
 * Phase 4.6.3 — content-hash deduplication.
 *
 * Additive, separate from the slab API above. A dedup page is mapped
 * through a refcount-aware path; the slab API (create/ensure/free) is
 * unchanged and only ever sees exclusively-owned pages — so the A1
 * "shared handle released too early" hazard is structurally avoided
 * and the op #1 unit test holds by construction. Merging dedup into
 * the live slab path is T4.6.4+.
 *
 * Mechanism: xxhash64 over the 2 MiB content keys a chained table;
 * every hash hit is verified by a full-page host memcmp (a collision
 * degrades to a missed dedup, never corruption). See phase2_design.md.
 * =================================================================== */

/* Reserve the dedup scratch VA + pinned host buffer. Idempotent.
 * Call after cipher_rt_kv_alloc_init(). Returns 0 on success. */
int cipher_rt_kv_dedup_init(void);

/* Place a 2 MiB page holding `content` (host buffer, exactly page_size
 * bytes). If a content-identical page already exists, the new virtual
 * page is mapped onto the SAME physical page (no new physical memory)
 * and *out_devptr returns its address. Returns 0 on success. */
int cipher_rt_kv_dedup_put(const void *content,
                           unsigned long long *out_devptr);

/* Release a dedup page. Decrements the physical page's refcount;
 * frees physical memory only when the last reference is dropped. */
void cipher_rt_kv_dedup_free(unsigned long long devptr);

/* W10-12 Step 2 G3 — test/inspection helper. Computes the
 * model-keyed content hash exactly as the production put-path does
 * (xxh64(content, len) XOR golden-ratio-mixed model_uuid). Exposed
 * for the synthetic cross-model algebraic gate; takes uuid as
 * caller-supplied so the test can sweep without manipulating the
 * snapshot table. */
uint64_t cipher_rt_kv_dedup_model_hash(const void *content,
                                       unsigned long content_len,
                                       uint64_t model_uuid_lo,
                                       uint64_t model_uuid_hi);

/* Week 5 Step 1b — in-place rebind of caller's existing VA onto a deduped
 * physical (per WEEK_5_STEP_1_DESIGN_MEMO.md Part I.1).
 *
 * Caller already owns a CIPHER-VMM page at `existing_devptr` with content
 * written. This function reads the content (DtoH copy through the pinned
 * scratch), PUTs to the kmod, and on HIT does cuMemUnmap + cuMemRelease +
 * cuMemMap to swap the underlying physical onto the deduped shared page
 * WITHOUT changing the caller's VA. Returns 0 on success; sets
 * *was_deduped to 1 if HIT+memcmp-matched (caller's VA now points at
 * shared physical), 0 otherwise (MISS, or HIT-then-hash-collision —
 * caller's VA points at its own original physical, now registered).
 *
 * Errors: -EINVAL if existing_devptr is not in this bridge's pool or not
 * page-aligned or not currently mapped; -EIO if rebind fails mid-swap (a
 * recovery attempt is made to keep the VA mapped, but the original
 * content is lost on this path — defensive log + return). Idempotent on
 * second call against the same page (no-op; was_deduped=0). */
int cipher_rt_kv_dedup_alias(unsigned long long existing_devptr,
                             int *was_deduped);

struct cipher_rt_kv_dedup_stats {
	uint64_t puts;             /* total dedup_put calls */
	uint64_t hits;             /* content matched an existing page */
	uint64_t misses;           /* new unique content */
	uint64_t physical_pages;   /* unique physical pages currently live */
	uint64_t virtual_pages;    /* virtual pages currently mapped */
	uint64_t hash_collisions;  /* xxh64 matched but memcmp differed */
	uint64_t refcount_releases;/* physical pages freed at refcount 0 */
};
void cipher_rt_kv_dedup_get_stats(struct cipher_rt_kv_dedup_stats *out);

/* ===================================================================
 * Phase C SC2 — producer-side weight arena.
 *
 * A weight arena holds one model's weights as ONE large VMM physical
 * allocation. Unlike KV slabs (which grow page-by-page as a sequence
 * decodes), model weights are load-once and fixed-size — so the arena
 * is a single cuMemCreate: one physical handle, one exportable POSIX
 * file descriptor, mapped over one contiguous VA range. Individual
 * weight tensors are bump-allocated sub-ranges (the bridge does the
 * sub-allocation; the C side owns the physical allocation + export).
 *
 * Separate from the KV slab / dedup paths above — it shares only the
 * CUDA device + primary context established by cipher_rt_kv_alloc_init,
 * which must be called first.
 * =================================================================== */

/* Create a weight arena of `bytes` (rounded up to page granularity):
 * one exportable cuMemCreate allocation, mapped + zeroed. Writes the
 * base device pointer to *out_base. Returns 0 on success, -1 on
 * failure. cipher_rt_kv_alloc_init() must have been called first. */
int cipher_rt_weight_arena_create(uint32_t tenant_id, size_t bytes,
                                  unsigned long long *out_base);

/* Export the arena (identified by its base devptr) as a POSIX file
 * descriptor — the producer-side shareable handle a peer process
 * imports in SC3. Writes the fd to *out_fd; the caller owns it.
 * Returns 0 on success, -1 if `base` is unknown or the export fails. */
int cipher_rt_weight_arena_export(unsigned long long base, int *out_fd);

/* Reverse lookup: if `devptr` falls inside a weight arena, write the
 * arena's tenant id and VMM physical handle and return 0; else -1. */
int cipher_rt_weight_arena_info(unsigned long long devptr,
                                uint32_t *out_tenant,
                                unsigned long long *out_handle);

/* Release a weight arena: unmap, release physical, free the VA. */
void cipher_rt_weight_arena_free(unsigned long long base);

/* Track 2 SC3 — CONSUMER side. Import a weight arena exported by a peer
 * (`fd` from cipher_rt_weight_arena_export, passed cross-process). Imports
 * the VMM handle, reserves VA — preferring `want_base` (the producer's base,
 * for a same-VA mapping; pass 0 to skip), falling back to any VA — maps it,
 * and sets READ-only access (a consumer must never write shared weights).
 * `bytes` is the producer arena size (page-rounded internally). Writes the
 * mapped base to *out_base. Returns 0 on success, -1 on failure (the caller
 * then falls back to an independent model load). The imported arena is
 * tracked like a created one — _info and _free work on it. */
int cipher_rt_weight_arena_import(int fd, unsigned long long want_base,
                                  size_t bytes,
                                  unsigned long long *out_base);

/* Verified-against environment. */
#define CIPHER_RT_KV_ALLOC_VERIFIED_CUDA "13.0 / driver 580.105.08 / H100"

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_RT_KV_ALLOC_H */
