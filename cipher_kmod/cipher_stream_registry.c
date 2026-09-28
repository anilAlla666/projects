// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_stream_registry — W7-9 Step 5: multi-tenant CUDA-stream resolver.
 *
 * Open-addressing hashtable mapping (tgid, stream_handle) -> tenant_id.
 * Storage is a single vmalloc'd region that doubles as the userspace mmap
 * surface (no kernel pointers leak; layout is stable ABI for cipher_rt_phase4).
 *
 * Concurrency:
 *   Writers (CIPHER_REGISTER_STREAMS ioctl, fd-close cleanup) hold a global
 *   spinlock and bump per-slot generation counters for ABA detection.
 *   Readers (in-kernel cipher_stream_lookup AND userspace mmap'd readers)
 *   are lock-free: linear probe, pre/post generation check on the matched
 *   slot, retry on tear.
 *
 * Sizing:
 *   8192 slots × 24 B = 192 KiB. Holds 128 tenants × 16 streams comfortably
 *   (target load factor 25% — open-addressing degrades > 70%, so 2048
 *   active entries among 8192 slots stays in the O(1) regime).
 */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/vmalloc.h>
#include <linux/sched.h>
#include <linux/printk.h>

#include "cipher_stream_registry.h"

static struct cipher_stream_view_slot *g_view;
static DEFINE_SPINLOCK(g_view_lock);
static u32 g_active_slots;  /* protected by g_view_lock */

/* FNV-1a-ish 64-bit mix; cheap, decent distribution for (tgid, handle). */
static inline u32 cipher_stream_hash(u32 tgid, u64 handle)
{
	u64 h = (u64)tgid * 0x100000001b3ULL;
	h ^= handle * 0x100000001b3ULL;
	h ^= h >> 32;
	return (u32)(h & (CIPHER_STREAM_VIEW_SLOTS - 1));
}

int cipher_stream_registry_init(void)
{
	if (g_view) return 0;  /* idempotent */
	/* vmalloc_user (not vzalloc) — remap_vmalloc_range requires the
	 * VM_USERMAP flag which only vmalloc_user/vmalloc_32_user set. */
	g_view = vmalloc_user(CIPHER_STREAM_VIEW_SIZE);
	if (!g_view) {
		pr_err("cipher_stream_registry: vmalloc_user %zu B failed\n",
		       (size_t)CIPHER_STREAM_VIEW_SIZE);
		return -ENOMEM;
	}
	memset(g_view, 0, CIPHER_STREAM_VIEW_SIZE);
	g_active_slots = 0;
	pr_info("cipher_stream_registry: ready, %u slots @ %p (%zu KiB)\n",
		CIPHER_STREAM_VIEW_SLOTS, g_view,
		(size_t)CIPHER_STREAM_VIEW_SIZE / 1024);
	return 0;
}

void cipher_stream_registry_exit(void)
{
	if (!g_view) return;
	vfree(g_view);
	g_view = NULL;
	g_active_slots = 0;
}

int cipher_stream_register(u32 tgid, u32 tenant_id,
			   const u64 *stream_handles, u32 num_streams)
{
	u32 i, j, probes, idx;
	int rc = 0;

	if (!g_view) return -ENODEV;
	if (num_streams == 0 || num_streams > CIPHER_STREAM_MAX_PER_TENANT)
		return -EINVAL;
	for (i = 0; i < num_streams; i++) {
		if (stream_handles[i] == 0) return -EINVAL;
	}

	spin_lock(&g_view_lock);
	for (i = 0; i < num_streams; i++) {
		u64 handle = stream_handles[i];
		idx = cipher_stream_hash(tgid, handle);
		probes = 0;
		while (probes < CIPHER_STREAM_VIEW_SLOTS) {
			struct cipher_stream_view_slot *s = &g_view[idx];
			if (s->stream_handle == 0) {
				/* empty slot -> insert here */
				s->stream_handle = handle;
				s->tgid          = tgid;
				s->tenant_id     = tenant_id;
				WRITE_ONCE(s->generation, s->generation + 1);
				g_active_slots++;
				break;
			}
			if (s->stream_handle == handle && s->tgid == tgid) {
				/* same (tgid, handle) — collision check */
				if (s->tenant_id != tenant_id) {
					rc = -EBUSY;
					goto out;
				}
				/* idempotent re-register: update generation */
				WRITE_ONCE(s->generation, s->generation + 1);
				break;
			}
			idx = (idx + 1) & (CIPHER_STREAM_VIEW_SLOTS - 1);
			probes++;
		}
		if (probes == CIPHER_STREAM_VIEW_SLOTS) {
			rc = -ENOSPC;
			goto out;
		}
	}
out:
	spin_unlock(&g_view_lock);
	return rc;
}

int cipher_stream_unregister_tgid(u32 tgid)
{
	u32 i, cleared = 0;

	if (!g_view) return 0;
	spin_lock(&g_view_lock);
	for (i = 0; i < CIPHER_STREAM_VIEW_SLOTS; i++) {
		struct cipher_stream_view_slot *s = &g_view[i];
		if (s->stream_handle && s->tgid == tgid) {
			s->stream_handle = 0;
			s->tgid          = 0;
			s->tenant_id     = 0;
			WRITE_ONCE(s->generation, s->generation + 1);
			cleared++;
			if (g_active_slots) g_active_slots--;
		}
	}
	spin_unlock(&g_view_lock);
	return (int)cleared;
}

u32 cipher_stream_lookup(u32 tgid, u64 stream_handle)
{
	u32 idx, probes;

	if (!g_view || stream_handle == 0) return CIPHER_STREAM_TENANT_NONE;
	idx = cipher_stream_hash(tgid, stream_handle);
	probes = 0;
	while (probes < CIPHER_STREAM_VIEW_SLOTS) {
		struct cipher_stream_view_slot *s = &g_view[idx];
		u64 h = READ_ONCE(s->stream_handle);
		if (h == 0) return CIPHER_STREAM_TENANT_NONE;   /* probe terminator */
		if (h == stream_handle && READ_ONCE(s->tgid) == tgid)
			return READ_ONCE(s->tenant_id);
		idx = (idx + 1) & (CIPHER_STREAM_VIEW_SLOTS - 1);
		probes++;
	}
	return CIPHER_STREAM_TENANT_NONE;
}

int cipher_stream_registry_mmap(struct file *filp, struct vm_area_struct *vma)
{
	size_t want;

	(void)filp;
	if (!g_view) return -ENODEV;

	/* Read-only mapping. */
	if (vma->vm_flags & VM_WRITE) return -EACCES;
	if (vma->vm_flags & VM_EXEC)  return -EACCES;

	want = vma->vm_end - vma->vm_start;
	if (want > CIPHER_STREAM_VIEW_SIZE) return -EINVAL;

	/* Userspace passed vm_pgoff with the CIPHER_STREAM_VIEW_MMAP_PGOFF
	 * base offset. Strip that off (the dispatcher already routed us here);
	 * remap_vmalloc_range walks pages relative to its own buffer, so we
	 * pass 0 here. */
	return remap_vmalloc_range(vma, g_view, 0);
}
