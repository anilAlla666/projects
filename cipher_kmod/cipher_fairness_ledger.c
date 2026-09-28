// SPDX-License-Identifier: GPL-2.0-or-later
//
// cipher_fairness_ledger — D.8 FAIRNESS + SHIELD cross-tenant work-ledger.
// See cipher_fairness_ledger.h for the design rationale (kmod-owned shared
// region replacing the per-container /dev/shm table; timing-only enforcement
// lives in libcipher_rt; this file only provides shared memory + slot
// arbitration). ADDITIVE: NR 32 + a new mmap window only.

#include "cipher_fairness_ledger.h"

#include <linux/vmalloc.h>
#include <linux/mm.h>
#include <linux/sched.h>
#include <linux/ktime.h>
#include <linux/atomic.h>
#include <linux/printk.h>

/* The one global region. NULL until init succeeds; register/mmap no-op when
 * NULL so a vmalloc failure degrades to "no enforcement", never a crash. */
static struct cipher_fair_slot *g_ledger;

/* Same FNV-style mix as the stream registry, masked to the slot count. */
static inline u32 fair_hash(u32 tgid)
{
	u64 h = (u64)tgid * 0x100000001b3ULL;
	h ^= h >> 32;
	return (u32)(h & (CIPHER_FAIR_SLOTS - 1));
}

int cipher_fairness_ledger_init(void)
{
	/* vmalloc_user (not vzalloc) — remap_vmalloc_range into userspace
	 * requires the VM_USERMAP flag only vmalloc_user sets. Zero-filled. */
	g_ledger = vmalloc_user(CIPHER_FAIR_REGION_BYTES);
	if (!g_ledger) {
		pr_err("cipher_fairness_ledger: vmalloc_user %u B failed; "
		       "FAIRNESS/SHIELD disabled\n", CIPHER_FAIR_REGION_BYTES);
		return -ENOMEM;
	}
	pr_info("cipher_fairness_ledger: %u slots, %u B region (D.8 FAIRNESS+SHIELD, inert until armed)\n",
	        CIPHER_FAIR_SLOTS, CIPHER_FAIR_REGION_BYTES);
	return 0;
}

void cipher_fairness_ledger_exit(void)
{
	struct cipher_fair_slot *p = g_ledger;

	g_ledger = NULL;
	if (p)
		vfree(p);
}

int cipher_fairness_ledger_register(u32 band, u32 *out_slot)
{
	u32 tgid = (u32)current->tgid;
	u32 idx, probes;

	if (!g_ledger)
		return -ENOMEM;
	if (tgid == 0)
		return -EINVAL;

	idx = fair_hash(tgid);
	for (probes = 0; probes < CIPHER_FAIR_SLOTS; probes++) {
		struct cipher_fair_slot *s = &g_ledger[idx];
		u64 cur = READ_ONCE(s->key);

		/* Already mine — re-arm in place. */
		if (cur == (u64)tgid)
			goto claim;

		/* Empty — try to claim atomically. */
		if (cur == 0) {
			u64 prev = cmpxchg(&s->key, 0ULL, (u64)tgid);

			if (prev == 0 || prev == (u64)tgid)
				goto claim;
			/* Lost the race; if the winner is us, take it. */
			if (READ_ONCE(s->key) == (u64)tgid)
				goto claim;
		}
		idx = (idx + 1) & (CIPHER_FAIR_SLOTS - 1);
	}
	return -ENOSPC;

claim:
	{
		struct cipher_fair_slot *s = &g_ledger[idx];

		/* Fresh arm: zero the self-accounted counters and set the band.
		 * The owning tenant writes gemm_calls/last_active_ns from here on
		 * via the RW mmap; we never touch them again. */
		WRITE_ONCE(s->gemm_calls, 0ULL);
		WRITE_ONCE(s->last_active_ns, (u64)ktime_get_ns());
		WRITE_ONCE(s->yield_count, 0u);
		WRITE_ONCE(s->band, band);
		smp_wmb();
		if (out_slot)
			*out_slot = idx;
	}
	return 0;
}

int cipher_fairness_ledger_mmap(struct file *filp, struct vm_area_struct *vma)
{
	size_t want = vma->vm_end - vma->vm_start;

	(void)filp;
	if (!g_ledger)
		return -ENODEV;
	/* Reject EXEC; PERMIT WRITE (tenants self-account into their slot) —
	 * this is the deliberate difference from the read-only audit/stream
	 * windows. The page offset has already been rebased to 0 by the dev.c
	 * multiplex dispatcher. */
	if (vma->vm_flags & VM_EXEC)
		return -EACCES;
	if (want > CIPHER_FAIR_REGION_BYTES)
		return -EINVAL;
	return remap_vmalloc_range(vma, g_ledger, 0);
}
