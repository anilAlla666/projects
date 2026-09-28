/* SPDX-License-Identifier: GPL-2.0
 *
 * cipher_stream_registry — W7-9 Step 5: multi-tenant CUDA-stream resolver.
 *
 * Per WEEK_7_9_SCOPE_LOCK.md §7 Step 5 + design freeze 2026-05-23. Maps
 * (tgid, stream_handle) -> tenant_id. Plugin registers tenant's named CUDA
 * streams via CIPHER_REGISTER_STREAMS (NR 29). cipher_rt_phase4 reads the
 * mmap'd view slots lock-free in the cublas/SDPA shim hot path.
 *
 * Key design point per advisor 2026-05-23: stream_handle is process-local.
 * The kmod MUST key on (tgid, stream_handle), not stream_handle alone — two
 * processes can hold identical cudaStream_t values referring to different
 * streams.
 *
 * View-slot layout (mmap'd to userspace):
 *   8192 slots × 24 B = 192 KiB total, vmalloc'd at init. Open-addressing
 *   hashtable, linear probing on collision. Userspace lookup is a lock-free
 *   linear probe with generation-counter pre/post check (Step 4 reader
 *   pattern). Kernel insertion takes a spinlock and bumps the generation.
 */
#ifndef CIPHER_STREAM_REGISTRY_H
#define CIPHER_STREAM_REGISTRY_H

#include <linux/types.h>
#include <linux/fs.h>

#define CIPHER_STREAM_MAX_PER_TENANT 16
#define CIPHER_STREAM_VIEW_SLOTS     8192      /* must be power of 2 */
#define CIPHER_STREAM_VIEW_SIZE      (CIPHER_STREAM_VIEW_SLOTS * sizeof(struct cipher_stream_view_slot))
#define CIPHER_STREAM_VIEW_MMAP_PGOFF 0x100000 /* offset above the audit ring */

/* View slot layout. Must be packed for stable userspace ABI. */
struct cipher_stream_view_slot {
	__u64 stream_handle;   /* 0 = empty slot */
	__u32 tgid;            /* process group id of the registering tenant */
	__u32 tenant_id;
	__u32 generation;      /* bumped on every update; ABA detection */
	__u32 _pad;
} __packed;

/* Reserved value returned to userspace when no entry matches. */
#define CIPHER_STREAM_TENANT_NONE 0xFFFFFFFFu

/* Lifecycle. */
int  cipher_stream_registry_init(void);
void cipher_stream_registry_exit(void);

/* Registration. Called from CIPHER_REGISTER_STREAMS ioctl handler with the
 * caller's current->tgid. Inserts num_streams (tgid, stream_handles[i]) ->
 * tenant_id mappings. Returns 0 on success.
 *
 * Errors:
 *   -EINVAL   num_streams == 0 or > CIPHER_STREAM_MAX_PER_TENANT, or any
 *             stream_handle == 0 (default-stream sentinel rejected per
 *             v1 named-streams policy)
 *   -EBUSY    one of the (tgid, stream_handle) entries is already mapped to
 *             a DIFFERENT tenant_id (stream collision within process)
 *   -ENOSPC   the view-slot table is full (>= CIPHER_STREAM_VIEW_SLOTS active)
 */
int cipher_stream_register(u32 tgid, u32 tenant_id,
			   const u64 *stream_handles, u32 num_streams);

/* Unregister all streams owned by tgid. Called from release handler when a
 * tenant's /dev/cipher fd closes. Returns the number of slots cleared. */
int cipher_stream_unregister_tgid(u32 tgid);

/* Kernel-side lookup (for in-kernel callers; userspace uses the mmap'd view).
 * Returns tenant_id on match or CIPHER_STREAM_TENANT_NONE on miss. */
u32 cipher_stream_lookup(u32 tgid, u64 stream_handle);

/* mmap handler — maps the view-slot table read-only into userspace. Called
 * from the multiplexed cipher_dev mmap dispatcher when
 * vma->vm_pgoff >= CIPHER_STREAM_VIEW_MMAP_PGOFF. */
int cipher_stream_registry_mmap(struct file *filp, struct vm_area_struct *vma);

#endif /* CIPHER_STREAM_REGISTRY_H */
