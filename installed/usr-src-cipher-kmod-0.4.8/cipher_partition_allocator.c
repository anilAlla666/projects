// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_partition_allocator.c — Phase 4.2 T4.2.1 SM partition allocator.
 *
 * Version 0.4.5 — 0.4.4 base + B10: nr 9 takes a `flags` field with
 *                 CIPHER_PARTITION_FLAG_FIT_HINT for "fit-to-N" semantic
 *                 (release excess slots back to the pool on smaller
 *                 hint). Enables cooperative fair-share in libcipher_rt's
 *                 ARBITRATE poll thread. See PHASE_4_BACKLOG.md B10.
 *
 * H100 has 132 SMs. Partitioned into blocks of 4 SMs = 33 slots.
 * Slots are an array of atomic_t state. State encoding:
 *
 *   state == 0                                  → slot free
 *   state == (BIT(31) | pid_low_31)             → slot held by pid
 *
 * Request path is a single linear scan with per-slot atomic_cmpxchg:
 * O(slots) = O(37 max). No global lock. ABA-safe because the packed
 * state includes the tenant's pid — a release-then-reacquire from a
 * different tenant produces a different state value, so cmpxchg on
 * the old (pid-A, occupied) word never accidentally clobbers a
 * (pid-B, occupied) slot.
 *
 * Per-tenant `sm_partition_mask` on cipher_pid_stats is now a CACHE
 * for snapshot readers (exporter, ioctl nr 8). The slot array is the
 * source of truth. The cache is updated WRITE_ONCE under RCU after
 * each CAS sequence.
 *
 * Same FCFS + quartile-rebalance + 30 s idle-reclaim policy as 0.4.1.
 * Lock-free progress guarantee: in any N-way contended request burst,
 * at least one tenant completes per slot-scan pass.
 *
 * Driven by:
 *   - cipher_dev_request_sm_partition handler (T4.2.1 ioctl nr 9)
 *   - cipher_state_updater kthread (5 s rebalance + idle-reclaim tick)
 *
 * EXPORT_SYMBOL_GPL on request/release so future cipher_rt_km can call.
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/atomic.h>
#include <linux/rcupdate.h>
#include <linux/hashtable.h>
#include <linux/sched.h>
#include <linux/sort.h>
#include <linux/jiffies.h>
#include <linux/bitops.h>
#include <linux/slab.h>

#include "cipher_internal.h"

/* SM topology — see 0.4.1 comments. sm_count is a module parameter so
 * non-H100-SXM5 pods can override (B100=144, B200=148, H100 PCIe=114). */
#define CIPHER_SMS_PER_PARTITION     4
#define CIPHER_SM_COUNT_DEFAULT      132
#define CIPHER_PARTITION_HINT_CAP    8
#define CIPHER_PARTITION_IDLE_NS     (30ULL * NSEC_PER_SEC)
#define CIPHER_REBALANCE_INTERVAL_NS (5ULL * NSEC_PER_SEC)
/* The sm_partition_mask is u32 (stable ABI in struct cipher_partition_request
 * nr 9). One bit per slot caps the slot count at 32 to keep (1U << i) inside
 * unsigned-int width. Without this cap, slot 32 triggers UBSAN
 * shift-out-of-bounds and bit-0 aliasing — see PHASE_4_BACKLOG.md B6.
 * For pods with >32 SMs/4 partitions (H100 SXM5 = 33, B100 = 36, B200 = 37),
 * one to five slots of capacity are sacrificed in exchange for ABI stability.
 * Lifting this requires a u64 mask via a new ioctl nr 10 (deferred). */
#define CIPHER_PARTITION_SLOTS_MAX   32   /* u32 mask width — see B6 above */

#define CIPHER_SLOT_OCCUPIED         0x80000000U
#define CIPHER_SLOT_PACK(pid) \
	(CIPHER_SLOT_OCCUPIED | ((u32)(pid) & 0x7FFFFFFFU))

static int sm_count = CIPHER_SM_COUNT_DEFAULT;
module_param(sm_count, int, 0444);
MODULE_PARM_DESC(sm_count,
	"SM count for partition allocator. H100 SXM5=132 (default), "
	"H100 PCIe=114, B100=144, B200=148.");

static int cipher_partition_slots;    /* set in _init from sm_count */
static u64 cipher_last_rebalance_jiffies;

/* The slot array. Static storage, never freed; lifetime is module lifetime.
 *
 * Each slot is cache-line aligned to eliminate false sharing under heavy
 * concurrent contention. With 37 slots × 64 B = 2368 B, the array fits in
 * L1d on any modern x86 core, but no two slots share a line — so 33
 * concurrent CAS requesters each acquire their own line exclusively
 * instead of bouncing one shared line among 33 cores. */
struct cipher_partition_slot {
	atomic_t state;
} ____cacheline_aligned;
static struct cipher_partition_slot cipher_slots[CIPHER_PARTITION_SLOTS_MAX];

/* Find the cipher_pid_stats entry for `target_pid` under RCU.
 * Returns NULL if not registered. Caller holds rcu_read_lock(). */
static struct cipher_pid_stats *cipher_partition_find_tenant_rcu(pid_t target_pid)
{
	struct cipher_pid_stats *q;
	hash_for_each_possible_rcu(cipher_pid_table, q, node, (u32)target_pid) {
		if (q->pid == target_pid)
			return q;
	}
	return NULL;
}

/* Write back the per-tenant cache so snapshot readers see fresh values. */
static void cipher_partition_writeback_cache(pid_t target_pid, u32 mask)
{
	struct cipher_pid_stats *e;

	rcu_read_lock();
	e = cipher_partition_find_tenant_rcu(target_pid);
	if (e) {
		WRITE_ONCE(e->sm_partition_mask,  mask);
		WRITE_ONCE(e->sm_partition_count, hweight32(mask));
	}
	rcu_read_unlock();
}

/* Public API: request partitions for `target_pid`.
 * Idempotent: re-requesting from the same tenant keeps slots already held
 * and tries to expand up to hint if free slots exist.
 * Returns 0 on success, -ENOSPC if no slots could be claimed,
 * -EINVAL on bad arguments. */
int cipher_partition_request(pid_t target_pid, u32 hint,
                              u32 *out_mask, u32 *out_count)
{
	return cipher_partition_request_v2(target_pid, hint, 0,
	                                   out_mask, out_count);
}
EXPORT_SYMBOL_GPL(cipher_partition_request);

/* B10 (kmod 0.4.5): cipher_partition_request_v2 with flags. Implements
 * fit-to-N semantic when CIPHER_PARTITION_FLAG_FIT_HINT is set.
 *
 * Default (flags=0): identical to T4.2.1 behavior. Caller asks for hint
 * slots; if currently holds more, mask is unchanged.
 *
 * FIT_HINT semantic:
 *   - have > hint_clamp -> walk slot array, release `have - hint_clamp`
 *     slots via atomic_cmpxchg(my_state → 0). Release highest-indexed
 *     slots first (preserves low-index slot warm-cache).
 *   - have < hint_clamp -> grow same as default (claim free slots up to
 *     hint_clamp via cmpxchg).
 *   - have == hint_clamp -> no-op.
 *   - hint == 0 -> release all (only valid with FIT_HINT).
 */
int cipher_partition_request_v2(pid_t target_pid, u32 hint, u32 flags,
                                 u32 *out_mask, u32 *out_count)
{
	u32 my_state;
	u32 cur_mask = 0;
	int have = 0;
	int claims = 0;
	int releases = 0;
	int hint_clamp;
	int fit = !!(flags & CIPHER_PARTITION_FLAG_FIT_HINT);
	int i;

	if (!out_mask || !out_count)
		return -EINVAL;
	/* B10: reject unknown flag bits. */
	if (flags & ~CIPHER_PARTITION_FLAGS_ALL)
		return -EINVAL;

	if (fit && hint == 0) {
		hint_clamp = 0;          /* release-all */
	} else if (hint < 1) {
		hint_clamp = 1;
	} else if (hint > CIPHER_PARTITION_HINT_CAP) {
		hint_clamp = CIPHER_PARTITION_HINT_CAP;
	} else {
		hint_clamp = (int)hint;
	}

	my_state = CIPHER_SLOT_PACK(target_pid);

	/* Idempotent fast path: tenant cache already satisfies hint.
	 *
	 * NOTE (B10): the fast path can only be taken WITHOUT FIT_HINT.
	 * With FIT_HINT, we cannot short-circuit on cached_count >= hint
	 * because we may need to RELEASE down to hint, which the cache
	 * fast-path does not do. */
	if (!fit) {
		struct cipher_pid_stats *e;
		u32 cached_mask = 0, cached_count = 0;

		rcu_read_lock();
		e = cipher_partition_find_tenant_rcu(target_pid);
		if (e) {
			cached_mask  = READ_ONCE(e->sm_partition_mask);
			cached_count = READ_ONCE(e->sm_partition_count);
		}
		rcu_read_unlock();
		if (cached_count >= (u32)hint_clamp && cached_mask != 0) {
			*out_mask  = cached_mask;
			*out_count = cached_count;
			return 0;
		}
	}

	/* First pass: count current allocations + claim free slots up to
	 * hint_clamp via per-slot atomic_cmpxchg. FCFS for grows. */
	for (i = 0; i < cipher_partition_slots; i++) {
		u32 s = atomic_read(&cipher_slots[i].state);
		if (s == my_state) {
			cur_mask |= (1U << i);
			have++;
		} else if (s == 0 && have < hint_clamp) {
			if (atomic_cmpxchg(&cipher_slots[i].state, 0, my_state) == 0) {
				cur_mask |= (1U << i);
				have++;
				claims++;
			}
		}
	}

	/* B10: with FIT_HINT, if we still hold MORE than hint_clamp, release
	 * the highest-indexed slots back to the pool. We walk the slot array
	 * in REVERSE so the lowest-indexed slots (more cache-warm) survive.
	 * Each release is cmpxchg(my_state → 0); a slot that has been stolen
	 * by another tenant (different state) is left alone. */
	if (fit && have > hint_clamp) {
		for (i = cipher_partition_slots - 1; i >= 0 && have > hint_clamp; i--) {
			u32 s = atomic_read(&cipher_slots[i].state);
			if (s != my_state)
				continue;
			if (!(cur_mask & (1U << i)))
				continue;
			if (atomic_cmpxchg(&cipher_slots[i].state,
			                   my_state, 0) == my_state) {
				cur_mask &= ~(1U << i);
				have--;
				releases++;
			}
		}
	}

	/* Writeback cache whenever the granted set changed. */
	if (claims > 0 || releases > 0)
		cipher_partition_writeback_cache(target_pid, cur_mask);

	*out_mask  = cur_mask;
	*out_count = (u32)have;

	/* Return semantics:
	 *   - have == 0 && we asked for some slots -> -ENOSPC (existing
	 *     behavior preserved when hint>=1).
	 *   - have == 0 && FIT_HINT && hint==0 -> success (explicit release-all).
	 *   - otherwise -> 0.
	 */
	if (have == 0 && hint_clamp > 0)
		return -ENOSPC;
	return 0;
}
EXPORT_SYMBOL_GPL(cipher_partition_request_v2);

/* Public API: release all partitions held by `target_pid`.
 * Uses cmpxchg(my_state → 0) so concurrent (different-tenant) writers
 * cannot be clobbered. */
int cipher_partition_release(pid_t target_pid)
{
	u32 my_state = CIPHER_SLOT_PACK(target_pid);
	int i, released = 0;

	for (i = 0; i < cipher_partition_slots; i++) {
		u32 s = atomic_read(&cipher_slots[i].state);
		if (s == my_state) {
			if (atomic_cmpxchg(&cipher_slots[i].state, my_state, 0)
			    == my_state)
				released++;
		}
	}

	cipher_partition_writeback_cache(target_pid, 0);
	return released > 0 ? 0 : -ESRCH;
}
EXPORT_SYMBOL_GPL(cipher_partition_release);

/* Slot-only release: hot path for cipher_do_exit_pre. Skips the cache
 * writeback because the caller is about to delete the cipher_pid_stats
 * entry from cipher_pid_table. */
void cipher_partition_release_slots_only(pid_t target_pid)
{
	u32 my_state = CIPHER_SLOT_PACK(target_pid);
	int i;

	for (i = 0; i < cipher_partition_slots; i++) {
		u32 s = atomic_read(&cipher_slots[i].state);
		if (s == my_state)
			atomic_cmpxchg(&cipher_slots[i].state, my_state, 0);
	}
}
EXPORT_SYMBOL_GPL(cipher_partition_release_slots_only);

/* Idle reclaim — called by cipher_state_updater every 5 s. Walks the
 * per-PID table; any tenant idle > 30 s releases its slots via cmpxchg.
 * Tenants that have never received telemetry are left alone. */
static int cipher_partition_reap_idle(void)
{
	struct cipher_pid_stats *e;
	int bkt, reaped = 0;
	u64 now = get_jiffies_64();

	rcu_read_lock();
	hash_for_each_rcu(cipher_pid_table, bkt, e, node) {
		u64 last = READ_ONCE(e->last_telemetry_jiffies);
		u64 idle_ns;
		u32 mask = READ_ONCE(e->sm_partition_mask);
		pid_t pid = e->pid;
		u32 my_state;
		int i;

		if (mask == 0 || last == 0)
			continue;
		idle_ns = jiffies_to_nsecs((unsigned long)(now - last));
		if (idle_ns < CIPHER_PARTITION_IDLE_NS)
			continue;

		my_state = CIPHER_SLOT_PACK(pid);
		for (i = 0; i < cipher_partition_slots; i++) {
			if (mask & (1U << i))
				atomic_cmpxchg(&cipher_slots[i].state,
				               my_state, 0);
		}
		WRITE_ONCE(e->sm_partition_mask,  0);
		WRITE_ONCE(e->sm_partition_count, 0);
		reaped++;
	}
	rcu_read_unlock();
	return reaped;
}

/* Rebalance — reweights per-tenant caps by launches_total quartiles.
 * Over-allocated tenants release their highest-numbered slots via
 * cmpxchg(my_state → 0). Under-allocated tenants are not forcibly
 * expanded here; they'll grow on their next CIPHER_REQUEST_SM_PARTITION
 * call (the allocator is request-driven). */
struct cipher_tenant_load {
	pid_t pid;
	u64   launches;
};

static int cipher_load_cmp_desc(const void *a, const void *b)
{
	u64 la = ((const struct cipher_tenant_load *)a)->launches;
	u64 lb = ((const struct cipher_tenant_load *)b)->launches;
	if (lb > la) return  1;
	if (lb < la) return -1;
	return 0;
}

static int cipher_partition_rebalance(void)
{
	struct cipher_tenant_load *loads;
	struct cipher_pid_stats *e;
	int bkt, n = 0, adjusted = 0;
	const int MAX = 256;
	int i;

	loads = kmalloc_array(MAX, sizeof(*loads), GFP_ATOMIC);
	if (!loads)
		return -ENOMEM;

	rcu_read_lock();
	hash_for_each_rcu(cipher_pid_table, bkt, e, node) {
		u32 mask;
		if (n >= MAX) break;
		mask = READ_ONCE(e->sm_partition_mask);
		if (mask == 0) continue;
		loads[n].pid      = e->pid;
		loads[n].launches = READ_ONCE(e->launches_total);
		n++;
	}
	rcu_read_unlock();

	if (n == 0) {
		kfree(loads);
		return 0;
	}
	sort(loads, n, sizeof(*loads), cipher_load_cmp_desc, NULL);

	for (i = 0; i < n; i++) {
		int q = (i * 4) / n;
		u32 cap;
		u32 cur_mask;

		switch (q) {
		case 0:  cap = 8; break;
		case 1:  cap = 4; break;
		case 2:  cap = 2; break;
		default: cap = 1; break;
		}

		rcu_read_lock();
		e = cipher_partition_find_tenant_rcu(loads[i].pid);
		if (!e) {
			rcu_read_unlock();
			continue;
		}
		cur_mask = READ_ONCE(e->sm_partition_mask);
		if (hweight32(cur_mask) > cap) {
			u32 my_state = CIPHER_SLOT_PACK(e->pid);
			int k;
			int to_free = hweight32(cur_mask) - cap;
			/* Free highest bits first — preserve low-end continuity. */
			for (k = cipher_partition_slots - 1;
			     k >= 0 && to_free > 0; k--) {
				if (cur_mask & (1U << k)) {
					if (atomic_cmpxchg(&cipher_slots[k].state,
					                   my_state, 0) == my_state) {
						cur_mask &= ~(1U << k);
						to_free--;
					}
				}
			}
			WRITE_ONCE(e->sm_partition_mask,  cur_mask);
			WRITE_ONCE(e->sm_partition_count, hweight32(cur_mask));
			adjusted++;
		}
		rcu_read_unlock();
	}

	kfree(loads);
	return adjusted;
}

/* Public hook called by cipher_state_updater every tick.
 * Gated to 5 s rebalance + reap interval. */
void cipher_partition_tick(void)
{
	u64 now = get_jiffies_64();
	u64 since_ns;

	if (cipher_last_rebalance_jiffies == 0) {
		cipher_last_rebalance_jiffies = now;
		return;
	}
	since_ns = jiffies_to_nsecs(
		(unsigned long)(now - cipher_last_rebalance_jiffies));
	if (since_ns < CIPHER_REBALANCE_INTERVAL_NS)
		return;

	(void)cipher_partition_reap_idle();
	(void)cipher_partition_rebalance();
	cipher_last_rebalance_jiffies = now;
}

int cipher_partition_allocator_init(void)
{
	int i;

	if (sm_count < CIPHER_SMS_PER_PARTITION) {
		pr_warn("cipher_partition_allocator: sm_count=%d below minimum %d; "
		        "falling back to default %d\n",
		        sm_count, CIPHER_SMS_PER_PARTITION,
		        CIPHER_SM_COUNT_DEFAULT);
		sm_count = CIPHER_SM_COUNT_DEFAULT;
	}

	cipher_partition_slots = sm_count / CIPHER_SMS_PER_PARTITION;
	if (cipher_partition_slots > CIPHER_PARTITION_SLOTS_MAX)
		cipher_partition_slots = CIPHER_PARTITION_SLOTS_MAX;

	for (i = 0; i < CIPHER_PARTITION_SLOTS_MAX; i++)
		atomic_set(&cipher_slots[i].state, 0);

	cipher_last_rebalance_jiffies = 0;
	pr_info("cipher_partition_allocator: lock-free atomic slots; "
	        "sm_count=%d slots=%d (%d SMs each), cap %d per tenant, "
	        "30 s idle reclaim, 5 s rebalance\n",
	        sm_count, cipher_partition_slots,
	        CIPHER_SMS_PER_PARTITION, CIPHER_PARTITION_HINT_CAP);
	return 0;
}

void cipher_partition_allocator_exit(void)
{
	int i;

	for (i = 0; i < CIPHER_PARTITION_SLOTS_MAX; i++)
		atomic_set(&cipher_slots[i].state, 0);
}
