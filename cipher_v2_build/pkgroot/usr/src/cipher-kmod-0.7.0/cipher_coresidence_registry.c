// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_coresidence_registry — W.6 sub-C.
 *
 * Per-tgid GPU co-residence registry. A tenant process registers its
 * substrate model fingerprint (cipher_workload_model_fingerprint, W.6 sub-B)
 * and periodically queries which OTHER tenant processes share this GPU right
 * now, with each peer's fingerprint. W.4 POOL consumes (co-resident pids +
 * per-pid fingerprint) to form legal cross-tenant GEMM-coalescing groups
 * (same fingerprint => same model => coalescable).
 *
 * Table: kernel hashtable keyed on tgid, 256 buckets (CIPHER_COHORT_HASH_BITS).
 * Entries hold {tgid, model_fingerprint, last_seen_ns}. N <= 128 tenants
 * (G1+G2 cap), so 256 buckets keeps the table in the O(1) regime.
 *
 * Liveness: an entry is live iff (ktime_get_ns() - last_seen_ns) <=
 * CIPHER_COHORT_STALE_NS (30 s). cipher_cohort_register() refreshes
 * last_seen on every call (heartbeat-or-insert); cipher_cohort_snapshot()
 * prunes stale entries lazily during the walk. Monotonic clock so wall-time
 * adjustments do not perturb liveness.
 *
 * Concurrency: cipher_cohort_lock serialises register + snapshot. Allocation
 * happens outside the lock (GFP_KERNEL may sleep) via the model_registry
 * two-phase race-recheck idiom. snapshot()'s kfree-under-spinlock is safe
 * (model_registry_exit precedent; kfree does not sleep).
 *
 * ABI: ioctl NR 30 (REGISTER) + NR 31 (QUERY), MODULE_VERSION 0.6.5 -> 0.6.6.
 */

#include <linux/module.h>
#include <linux/slab.h>
#include <linux/hashtable.h>
#include <linux/spinlock.h>
#include <linux/ktime.h>
#include <linux/uaccess.h>
#include <linux/string.h>
#include <linux/sched.h>

#include "cipher_internal.h"
#include "cipher_ioctl.h"

#define CIPHER_COHORT_HASH_BITS 8          /* 256 buckets */

struct cipher_cohort_node {
	u32                tgid;
	u64                model_fingerprint;
	u64                last_seen_ns;
	struct hlist_node  hnode;
};

static DEFINE_HASHTABLE(cipher_cohort_hash, CIPHER_COHORT_HASH_BITS);
static DEFINE_SPINLOCK(cipher_cohort_lock);
static bool cipher_cohort_ready;

/* Caller holds cipher_cohort_lock. */
static struct cipher_cohort_node *cipher_cohort_find_locked(u32 tgid)
{
	struct cipher_cohort_node *e;

	hash_for_each_possible(cipher_cohort_hash, e, hnode, tgid)
		if (e->tgid == tgid)
			return e;
	return NULL;
}

int cipher_cohort_registry_init(void)
{
	hash_init(cipher_cohort_hash);
	cipher_cohort_ready = true;
	pr_info("cipher_kmod: coresidence_registry ready (W.6 sub-C) — "
	        "%u-bucket hashtable, %llu s liveness\n",
	        1u << CIPHER_COHORT_HASH_BITS,
	        (unsigned long long)(CIPHER_COHORT_STALE_NS / 1000000000ULL));
	return 0;
}

void cipher_cohort_registry_exit(void)
{
	struct cipher_cohort_node *e;
	struct hlist_node *tmp;
	int bkt;
	int freed = 0;

	cipher_cohort_ready = false;

	spin_lock(&cipher_cohort_lock);
	hash_for_each_safe(cipher_cohort_hash, bkt, tmp, e, hnode) {
		hash_del(&e->hnode);
		kfree(e);
		freed++;
	}
	spin_unlock(&cipher_cohort_lock);

	pr_info("cipher_kmod: coresidence_registry torn down; freed %d entries\n",
	        freed);
}

/* Heartbeat-or-insert: refresh last_seen for tgid (updating fingerprint if
 * changed), inserting a fresh entry if absent. Returns 0, or -ENOMEM. */
int cipher_cohort_register(u32 tgid, u64 model_fingerprint)
{
	struct cipher_cohort_node *e, *neu;
	u64 now = ktime_get_ns();

	if (!cipher_cohort_ready)
		return -ENOSYS;

	spin_lock(&cipher_cohort_lock);
	e = cipher_cohort_find_locked(tgid);
	if (e) {
		e->model_fingerprint = model_fingerprint;
		e->last_seen_ns      = now;
		spin_unlock(&cipher_cohort_lock);
		return 0;
	}
	spin_unlock(&cipher_cohort_lock);

	/* Allocate outside the lock (GFP_KERNEL may sleep). */
	neu = kzalloc(sizeof(*neu), GFP_KERNEL);
	if (!neu)
		return -ENOMEM;
	neu->tgid              = tgid;
	neu->model_fingerprint = model_fingerprint;
	neu->last_seen_ns      = now;

	/* Race-check: re-take lock, double-check, insert. */
	spin_lock(&cipher_cohort_lock);
	e = cipher_cohort_find_locked(tgid);
	if (e) {
		/* Lost the race; refresh the existing entry, drop ours. */
		e->model_fingerprint = model_fingerprint;
		e->last_seen_ns      = now;
		spin_unlock(&cipher_cohort_lock);
		kfree(neu);
		return 0;
	}
	hash_add(cipher_cohort_hash, &neu->hnode, tgid);
	spin_unlock(&cipher_cohort_lock);
	return 0;
}

/* Prune stale entries (> CIPHER_COHORT_STALE_NS since last_seen) and snapshot
 * the live set into out[0..min(live,max_entries)). *n_resident_out receives
 * the TRUE live count (which may exceed max_entries => snapshot truncated).
 * Returns the number of entries actually written. */
int cipher_cohort_snapshot(u32 max_entries,
                           struct cipher_cohort_entry_abi *out,
                           u32 *n_resident_out)
{
	struct cipher_cohort_node *e;
	struct hlist_node *tmp;
	u64 now = ktime_get_ns();
	u32 live = 0, filled = 0;
	int bkt;

	if (!cipher_cohort_ready) {
		if (n_resident_out)
			*n_resident_out = 0;
		return 0;
	}

	spin_lock(&cipher_cohort_lock);
	hash_for_each_safe(cipher_cohort_hash, bkt, tmp, e, hnode) {
		if (now - e->last_seen_ns > CIPHER_COHORT_STALE_NS) {
			hash_del(&e->hnode);
			kfree(e);
			continue;
		}
		live++;
		if (out && filled < max_entries) {
			out[filled].tgid              = e->tgid;
			out[filled]._pad              = 0;
			out[filled].model_fingerprint = e->model_fingerprint;
			filled++;
		}
	}
	spin_unlock(&cipher_cohort_lock);

	if (n_resident_out)
		*n_resident_out = live;
	return (int)filled;
}

/* ioctl NR 30 handler — explicit register/heartbeat. */
long cipher_dev_cohort_register(unsigned long arg)
{
	struct cipher_cohort_register req;

	if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
		return -EFAULT;

	/* Anti-spoof: key on current->tgid, ignore req.tgid. */
	return cipher_cohort_register(current->tgid, req.model_fingerprint);
}

/* ioctl NR 31 handler — heartbeat-or-insert self, prune, return live snapshot.
 * Heap-allocates the ~2 KiB payload per the model_registry kzalloc precedent
 * (stack budget 1024 B). */
long cipher_dev_cohort_query(unsigned long arg)
{
	struct cipher_cohort_query *q;
	u32 max, n = 0;
	long rc = 0;

	q = kzalloc(sizeof(*q), GFP_KERNEL);
	if (!q)
		return -ENOMEM;

	if (copy_from_user(q, (void __user *)arg, sizeof(*q))) {
		rc = -EFAULT;
		goto out;
	}

	max = q->max_entries;
	if (max > CIPHER_COHORT_MAX)
		max = CIPHER_COHORT_MAX;

	/* Heartbeat-or-insert the caller (process context, may sleep). */
	if (q->caller_fingerprint)
		cipher_cohort_register(current->tgid, q->caller_fingerprint);

	/* Prune stale peers + snapshot the live set (incl. self). */
	cipher_cohort_snapshot(max, q->entries, &n);
	q->n_resident = n;

	if (copy_to_user((void __user *)arg, q, sizeof(*q)))
		rc = -EFAULT;
out:
	kfree(q);
	return rc;
}
