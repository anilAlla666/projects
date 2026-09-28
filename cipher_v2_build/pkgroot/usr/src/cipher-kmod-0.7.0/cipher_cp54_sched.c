// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_cp54_sched.c — CP 5.4 per-tenant arbitration: 8-SM-group ledger.
 *
 * The cross-process SM-allocation authority for CP 5.4. CUDA green contexts
 * on H100 partition in multiples of 8 SMs; this ledger allocates the device
 * as 15 × 8-SM groups (120 of 132 SMs — the 12-SM remainder is unallocatable).
 * CP 5.4 Step 1.3a empirically measured cuDevSmResourceSplitByCount(minCount=8)
 * on this H100 80GB SXM5: it returns 15 groups + 12 remainder (GPC clustering),
 * NOT the 16 + 4 originally assumed — see step1_3/PHASE_1_3A_PROBE.md.
 *
 *   group g  ↔  the g-th 8-SM resource from cuDevSmResourceSplitByCount
 *               (libcipher_rt enacts the placement — CP 5.4 Step 1.3)
 *
 * Three tenant classes (cipher_ioctl.h CIPHER_CP54_QOS_*):
 *   PARTITION — owns ceil(sm_count/8) groups exclusively.
 *   POOL      — the batch-pool owner (one PID); owns the residual groups.
 *   SHARED    — a batch-pool member; owns no groups, uses the pool.
 *
 * Concurrency (CP 5.4 scope memo V1 §5, adjudicated Option ii):
 *   - The 15-group atomic array is the source of truth; claim/release is
 *     per-group atomic_cmpxchg.
 *   - ALLOCATE takes cipher_cp54_lock (mutex) for the whole operation.
 *     ALLOCATE is a COLD path — per-session tenant registration, not
 *     per-request (v1 scoping) — so a whole-operation mutex is bounded and
 *     simple. NOTE: the V1 memo's literal text kept a "lock-free fast path"
 *     for non-resize allocates; this build takes the mutex for the whole
 *     ALLOCATE because the metadata-table slot claim also needs
 *     serialisation, and splitting it reintroduces exactly the subtle
 *     reasoning Option (ii) was chosen to avoid. Flagged for the Step 1.2
 *     adjudication checkpoint.
 *   - FREE / release(pid) are LOCK-FREE — release(pid) is called from the
 *     do_exit kprobe reaper (atomic context, cannot sleep), so it must not
 *     take a mutex. It only ever touches the dying PID's own groups +
 *     metadata entry; atomic_cmpxchg makes that safe against a concurrent
 *     mutex-held ALLOCATE.
 *   - QUERY is lock-free atomic reads.
 *
 * The legacy nr-9 4-SM-slot allocator (cipher_partition_allocator.c) is
 * untouched but its userspace path is deactivated (cipher_dev.c returns
 * -ENOSYS for nr 9) — the two allocators address the same physical SMs and
 * must not both grant. One allocator, one source of truth.
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/atomic.h>
#include <linux/mutex.h>
#include <linux/bitops.h>
#include <linux/build_bug.h>
#include <linux/sched.h>
#include <linux/uaccess.h>
#include <linux/printk.h>
#include <linux/ktime.h>
#include <linux/seq_file.h>

#include "cipher_internal.h"
#include "cipher_ioctl.h"

/* CP 5.4 Step 1.3a: 15, not 16 — cuDevSmResourceSplitByCount(minCount=8) on
 * this H100 yields 15 × 8-SM groups + 12 remainder (PHASE_1_3A_PROBE.md). */
#define CIPHER_CP54_NUM_GROUPS     15
#define CIPHER_CP54_SMS_PER_GROUP  8

/* Per-group atomic state encoding. Three states (Track 3 SC2):
 *   0                       free
 *   GRP_PACK(pid)           owned by pid          — bit 31 set
 *   GRP_RSVD(pid)           reserved for pid's    — bit 30 set
 *                           pending migration
 * Track 3 SC2 narrows the packed pid from 31 to 30 bits to free bit 30 as
 * the RSVD flag. Linux PID_MAX_LIMIT is 2^22 (4.19 M); 30 bits hold 1.07 B —
 * 256× headroom. This encoding is kmod-internal: userspace sees pid_t
 * unchanged through the ioctl ABI. For any real pid (< 2^22) GRP_PACK is
 * byte-identical to the pre-SC2 value, so the existing ledger is unchanged. */
#define CIPHER_CP54_GRP_OCC        0x80000000U   /* bit 31 — OWNED */
#define CIPHER_CP54_GRP_RSVD_BIT   0x40000000U   /* bit 30 — RESERVED */
#define CIPHER_CP54_GRP_PID_MASK   0x3FFFFFFFU   /* bits 0..29 — packed pid */
#define CIPHER_CP54_GRP_PACK(pid)  \
	(CIPHER_CP54_GRP_OCC | ((u32)(pid) & CIPHER_CP54_GRP_PID_MASK))
#define CIPHER_CP54_GRP_RSVD(pid)  \
	(CIPHER_CP54_GRP_RSVD_BIT | ((u32)(pid) & CIPHER_CP54_GRP_PID_MASK))

/* Track 3 SC2 migration tunables. SC2 ships a deliberately CONSERVATIVE
 * default policy (item-1 PUSH): the automatic (FREE-triggered) path proposes
 * a migration only for a large, sustained, POOL-unreachable gap — so all-
 * pinned regression workloads (W1/W2/W3) behave as if migration did not
 * exist. CIPHER_CP54_COMPACT_MIGRATE bypasses the size/sustain gates (operator
 * / test forcing). SC4 reshapes these for production realism. */
/* Track 3 SC4 (folded into SC5) — migration policy is operator-tunable via
 * module parameters at /sys/module/cipher_kmod/parameters/ (mode 0644 — boot-
 * time via insmod args, runtime via a sysfs write). cp54_eval_migration reads
 * them live. Production defaults: SC5 design memo §0. The PROPOSED->ABORT
 * timeout stays a compile-time constant — not in the adjudicated param set. */
#define CIPHER_CP54_MIG_TIMEOUT_NS     (30ULL * NSEC_PER_SEC) /* PROPOSED->ABORT */

static int  cipher_cp54_mig_gap_min_grps = 1;     /* propose at gap >= N groups */
static uint cipher_cp54_mig_sustain_ms   = 2000;  /* gap must persist this long */
static uint cipher_cp54_mig_ratelimit_ms = 10000; /* >=1 migration/tenant/this */
static int  cipher_cp54_mig_verbose      = 1;     /* 1 = log PROPOSE/COMMIT */
module_param(cipher_cp54_mig_gap_min_grps, int,  0644);
module_param(cipher_cp54_mig_sustain_ms,   uint, 0644);
module_param(cipher_cp54_mig_ratelimit_ms, uint, 0644);
module_param(cipher_cp54_mig_verbose,      int,  0644);
MODULE_PARM_DESC(cipher_cp54_mig_gap_min_grps,
	"Track 3 DSM: min POOL-unreachable gap (8-SM groups) to propose a migration");
MODULE_PARM_DESC(cipher_cp54_mig_sustain_ms,
	"Track 3 DSM: a gap must persist this many ms before a migration is proposed");
MODULE_PARM_DESC(cipher_cp54_mig_ratelimit_ms,
	"Track 3 DSM: minimum ms between migration proposals to one tenant");
MODULE_PARM_DESC(cipher_cp54_mig_verbose,
	"Track 3 DSM: 1 = log migration PROPOSE/COMMIT to dmesg, 0 = quiet");

/* grp_mask (cipher_cp54_allocate / _query) is a u32 — one bit per group. */
static_assert(CIPHER_CP54_NUM_GROUPS <= 32,
              "CIPHER_CP54_NUM_GROUPS must fit a u32 grp_mask");

/* 15 × 8-SM groups. Per-group state: 0 free, GRP_PACK(pid) owned,
 * GRP_RSVD(pid) reserved for pid's pending migration (Track 3 SC2). */
static atomic_t cipher_cp54_groups[CIPHER_CP54_NUM_GROUPS];

/* Per-PID CP 5.4 metadata. Slot claim is serialised by cipher_cp54_lock
 * (ALLOCATE path); release clears in_use lock-free (WRITE_ONCE) — each PID
 * only ever writes its own entry, so there is no same-entry write race. */
#define CIPHER_CP54_MAX_ALLOCS  128 /* W6 G1: 64 -> 128 per WEEK_6_ARCHITECTURE_GAP_AUDIT.md; A.4 confirms all 8 cap-iterating loops are single O(N), no nested CP54-iteration O(N^2); array static .bss grows ~4 KiB. ABI bump 0.4.8 -> 0.5.0. */
struct cipher_cp54_alloc {
	pid_t pid;
	u8    qos_class;
	u8    in_use;
	/* Track 3 SC2 — per-tenant migration state. All fields are mutated
	 * only under cipher_cp54_lock; POLL_MIGRATE reads them lock-free
	 * (a torn read is benign for a diagnostic poll — QUERY discipline). */
	u8    migratable;       /* SUBSCRIBE_MIGRATE; 0 = pinned (default) */
	u8    migrate_state;    /* CIPHER_CP54_MIG_* */
	u8    last_outcome;     /* CIPHER_CP54_MIGOUT_* */
	u32   target_mask;      /* proposed new_mask while PROPOSED/MIGRATING */
	u64   proposed_ns;      /* ktime at PROPOSED entry — timeout base */
	u64   last_migrate_ns;  /* ktime of last PROPOSE — rate-limit base */
	u32   migration_count;  /* Track 3 SC4 — committed migrations by this pid */
};
static struct cipher_cp54_alloc cipher_cp54_allocs[CIPHER_CP54_MAX_ALLOCS];

/* ktime at which the current POOL-unreachable gap was first observed; 0 = no
 * qualifying gap. Touched only under cipher_cp54_lock (cp54_eval_migration). */
static u64 cipher_cp54_gap_since_ns;

/* Track 3 SC4 — aggregate migration counters for /proc/cipher/migrations.
 * Mutated under cipher_cp54_lock; read lock-free by the /proc printer. */
static atomic_t cipher_cp54_stat_proposals;
static atomic_t cipher_cp54_stat_commits;
static atomic_t cipher_cp54_stat_abort_timeout;
static atomic_t cipher_cp54_stat_abort_nack;
static atomic_t cipher_cp54_stat_abort_refused;

/* The batch-pool owner PID, 0 = none. atomic_t so the lock-free release()
 * path can cmpxchg(pid→0) without clobbering a concurrent ALLOCATE(POOL)
 * that has just registered a new owner. */
static atomic_t cipher_cp54_pool_pid;

static DEFINE_MUTEX(cipher_cp54_lock);

/* ---- group helpers (atomic_cmpxchg; safe lock-free) ------------------- */

/* Mask of groups currently owned by `pid`. */
static u32 cp54_mask_of(pid_t pid)
{
	u32 want = CIPHER_CP54_GRP_PACK(pid), mask = 0;
	int g;

	for (g = 0; g < CIPHER_CP54_NUM_GROUPS; g++)
		if ((u32)atomic_read(&cipher_cp54_groups[g]) == want)
			mask |= (1U << g);
	return mask;
}

static int cp54_free_count(void)
{
	int g, n = 0;

	for (g = 0; g < CIPHER_CP54_NUM_GROUPS; g++)
		if (atomic_read(&cipher_cp54_groups[g]) == 0)
			n++;
	return n;
}

/* Claim up to `n` free groups for `pid`. Returns the count claimed. */
static int cp54_claim(pid_t pid, int n)
{
	u32 want = CIPHER_CP54_GRP_PACK(pid);
	int g, got = 0;

	for (g = 0; g < CIPHER_CP54_NUM_GROUPS && got < n; g++)
		if (atomic_cmpxchg(&cipher_cp54_groups[g], 0, (int)want) == 0)
			got++;
	return got;
}

/* Release every group held by `pid` in ANY state — owned (PACK) or reserved
 * (RSVD). Two cmpxchg attempts per group: this is the Track 3 SC2 reaper
 * sweep — a tenant crashing in any migration phase (PROPOSED, MIGRATING,
 * mid-swap, post-swap-pre-ACK) leaks no group, because both its old owned
 * set and its reserved set return to free. Lock-free; safe in the do_exit
 * reaper's atomic context. Returns the count released. */
static int cp54_release_groups(pid_t pid)
{
	u32 own = CIPHER_CP54_GRP_PACK(pid);
	u32 rsv = CIPHER_CP54_GRP_RSVD(pid);
	int g, n = 0;

	for (g = 0; g < CIPHER_CP54_NUM_GROUPS; g++) {
		if (atomic_cmpxchg(&cipher_cp54_groups[g], (int)own, 0)
		    == (int)own)
			n++;
		else if (atomic_cmpxchg(&cipher_cp54_groups[g], (int)rsv, 0)
		         == (int)rsv)
			n++;
	}
	return n;
}

/* Release only the RSVD groups held by `pid` (an aborted migration's
 * reservation); the tenant keeps its owned PACK groups. Caller holds the
 * mutex. */
static void cp54_release_rsvd(pid_t pid)
{
	u32 rsv = CIPHER_CP54_GRP_RSVD(pid);
	int g;

	for (g = 0; g < CIPHER_CP54_NUM_GROUPS; g++)
		atomic_cmpxchg(&cipher_cp54_groups[g], (int)rsv, 0);
}

/* Reserve the free groups named by `mask` as RSVD(pid). Caller holds the
 * mutex. Returns the count reserved — == hweight32(mask) on full success;
 * a short count means a free group was lost to a concurrent lock-free FREE
 * and the caller must roll back. */
static int cp54_reserve(pid_t pid, u32 mask)
{
	u32 rsv = CIPHER_CP54_GRP_RSVD(pid);
	int g, got = 0;

	for (g = 0; g < CIPHER_CP54_NUM_GROUPS; g++)
		if ((mask & (1U << g)) &&
		    atomic_cmpxchg(&cipher_cp54_groups[g], 0, (int)rsv) == 0)
			got++;
	return got;
}

/* Shrink the batch pool by up to `n` groups — release its highest-indexed
 * groups back to free (low groups stay warm). Caller holds the mutex.
 * Returns the count freed. */
static int cp54_pool_shrink(int n)
{
	pid_t pool = atomic_read(&cipher_cp54_pool_pid);
	u32 want;
	int g, freed = 0;

	if (pool == 0)
		return 0;
	want = CIPHER_CP54_GRP_PACK(pool);
	for (g = CIPHER_CP54_NUM_GROUPS - 1; g >= 0 && freed < n; g--)
		if (atomic_cmpxchg(&cipher_cp54_groups[g], (int)want, 0)
		    == (int)want)
			freed++;
	return freed;
}

/* Track 3 SC1 fact-2 / SC5 — the constrained POOL grant. Claim the maximal
 * contiguous LOW PREFIX [0, K-1] for `pid` (the POOL): walk groups from 0 up,
 * claiming each that is free or already POOL-owned, and STOP at the first
 * group that is partition-owned or RESERVED. A free group above that K is
 * left free — POOL-unreachable (its count-only green ctx cannot place there);
 * Dynamic SM Migration compacts it back. Replaces the prior "claim every free
 * group" — which let the POOL ledger-own scattered groups its green context
 * could not reach (the SC5 design memo §1 fragmentation finding). Caller holds
 * the mutex. Returns the count newly claimed. */
static int cp54_pool_claim_low_prefix(pid_t pid)
{
	u32 self = CIPHER_CP54_GRP_PACK(pid);
	int g, got = 0;

	for (g = 0; g < CIPHER_CP54_NUM_GROUPS; g++) {
		int v = atomic_read(&cipher_cp54_groups[g]);

		if (v == 0) {
			if (atomic_cmpxchg(&cipher_cp54_groups[g], 0,
			                   (int)self) != 0)
				break;     /* lost a race — the prefix ends here */
			got++;
		} else if (v != (int)self) {
			break;             /* partition-owned or RSVD — prefix ends */
		}
		/* v == self: already ours — the prefix continues */
	}
	return got;
}

/* ---- metadata helpers ------------------------------------------------- */

/* Caller holds the mutex. */
static struct cipher_cp54_alloc *cp54_meta_find(pid_t pid)
{
	int i;

	for (i = 0; i < CIPHER_CP54_MAX_ALLOCS; i++)
		if (READ_ONCE(cipher_cp54_allocs[i].in_use) &&
		    cipher_cp54_allocs[i].pid == pid)
			return &cipher_cp54_allocs[i];
	return NULL;
}

/* Caller holds the mutex. Returns NULL only if the table is full. */
static struct cipher_cp54_alloc *cp54_meta_get_or_create(pid_t pid)
{
	struct cipher_cp54_alloc *m = cp54_meta_find(pid);
	int i;

	if (m)
		return m;
	for (i = 0; i < CIPHER_CP54_MAX_ALLOCS; i++) {
		if (!READ_ONCE(cipher_cp54_allocs[i].in_use)) {
			m = &cipher_cp54_allocs[i];
			m->pid = pid;
			m->qos_class = CIPHER_CP54_QOS_SHARED;
			/* Track 3 SC2: a freed slot keeps its stale fields (release
			 * only clears in_use) — zero the migration state on reuse so
			 * a new tenant never inherits a prior tenant's migration. */
			m->migratable     = 0;
			m->migrate_state  = CIPHER_CP54_MIG_IDLE;
			m->last_outcome   = CIPHER_CP54_MIGOUT_NONE;
			m->target_mask    = 0;
			m->proposed_ns    = 0;
			m->last_migrate_ns = 0;
			m->migration_count = 0;
			WRITE_ONCE(m->in_use, 1);
			return m;
		}
	}
	return NULL;
}

/* ---- release (lock-free; FREE ioctl + do_exit reaper) ----------------- */

static void cp54_release_impl(pid_t pid)
{
	struct cipher_cp54_alloc *m;
	int i;

	cp54_release_groups(pid);
	/* cmpxchg, not a plain store: clear the pool owner only if it is
	 * still this dying pid — never clobber a concurrent ALLOCATE(POOL)
	 * that has just registered a new owner. */
	atomic_cmpxchg(&cipher_cp54_pool_pid, pid, 0);
	for (i = 0; i < CIPHER_CP54_MAX_ALLOCS; i++) {
		m = &cipher_cp54_allocs[i];
		if (READ_ONCE(m->in_use) && m->pid == pid)
			WRITE_ONCE(m->in_use, 0);
	}
}

/* do_exit reaper hook — atomic context, must not sleep. */
void cipher_cp54_release(pid_t pid)
{
	cp54_release_impl(pid);
}

/* ---- Track 3 SC2: migration state machine ----------------------------
 *
 * IDLE --[PROPOSE]--> PROPOSED --[START]--> MIGRATING --[ACK(ok)]--> COMMIT
 *                         |                     |
 *                [timeout, no START]   [ACK(!ok) tenant-NACK]
 *                         v                     v
 *                       ABORT  <----------------+
 *
 * Every transition that touches the ledger runs under cipher_cp54_lock.
 * The kmod never unilaterally aborts a MIGRATING migration; only the tenant
 * may (ACK ok=0). PROPOSED migrations time out lazily (cp54_abort_timeouts,
 * evaluated on the next FREE/COMPACT) — no timer/workqueue.
 */

/* COMMIT — caller holds the mutex; `m` is MIGRATING. Atomically free the
 * tenant's old owned groups and promote its reservation to owned. After
 * this the tenant owns exactly target_mask. old_mask and target_mask are
 * disjoint by construction (target ⊆ free, old ⊆ owned at PROPOSE). */
static void cp54_commit_migration(struct cipher_cp54_alloc *m)
{
	u32 own = CIPHER_CP54_GRP_PACK(m->pid);
	u32 rsv = CIPHER_CP54_GRP_RSVD(m->pid);
	int g;

	for (g = 0; g < CIPHER_CP54_NUM_GROUPS; g++)
		atomic_cmpxchg(&cipher_cp54_groups[g], (int)own, 0);
	for (g = 0; g < CIPHER_CP54_NUM_GROUPS; g++)
		atomic_cmpxchg(&cipher_cp54_groups[g], (int)rsv, (int)own);
	m->migrate_state = CIPHER_CP54_MIG_IDLE;
	m->target_mask   = 0;
	m->last_outcome  = CIPHER_CP54_MIGOUT_COMMITTED;
	m->migration_count++;
	atomic_inc(&cipher_cp54_stat_commits);
}

/* ABORT — caller holds the mutex. Release the reservation; the tenant keeps
 * its owned groups untouched (the B-floor fallback). `outcome` records why. */
static void cp54_abort_migration(struct cipher_cp54_alloc *m, u8 outcome)
{
	cp54_release_rsvd(m->pid);
	m->migrate_state = CIPHER_CP54_MIG_IDLE;
	m->target_mask   = 0;
	m->last_outcome  = outcome;
	if (outcome == CIPHER_CP54_MIGOUT_ABORTED_TIMEOUT)
		atomic_inc(&cipher_cp54_stat_abort_timeout);
	else if (outcome == CIPHER_CP54_MIGOUT_ABORTED_TENANT_NACK)
		atomic_inc(&cipher_cp54_stat_abort_nack);
	else if (outcome == CIPHER_CP54_MIGOUT_ABORTED_KMOD_REFUSED)
		atomic_inc(&cipher_cp54_stat_abort_refused);
}

/* Lazy timeout — caller holds the mutex. Abort every PROPOSED migration the
 * tenant never STARTed within the window. MIGRATING migrations are NOT
 * timed out by the kmod (SC1 §3d: safe-but-stuck; a hung MIGRATING tenant
 * is reclaimed only by the do_exit reaper). */
static void cp54_abort_timeouts(u64 now)
{
	struct cipher_cp54_alloc *m;
	int i;

	for (i = 0; i < CIPHER_CP54_MAX_ALLOCS; i++) {
		m = &cipher_cp54_allocs[i];
		if (READ_ONCE(m->in_use) &&
		    m->migrate_state == CIPHER_CP54_MIG_PROPOSED &&
		    now - m->proposed_ns > CIPHER_CP54_MIG_TIMEOUT_NS)
			cp54_abort_migration(m,
				CIPHER_CP54_MIGOUT_ABORTED_TIMEOUT);
	}
}

/* Compaction evaluation — caller holds the mutex. Detects a POOL-unreachable
 * gap and, if a `migratable` PARTITION is in the way, PROPOSEs a count-
 * preserving migration that moves it up to the highest free groups (its old
 * low groups then free, extending the POOL's contiguous low prefix). One
 * partition per pass — incremental compaction; the next FREE/COMPACT re-runs.
 *
 * SC2 ships the MECHANISM with a fixed conservative default policy (item-1
 * split: policy framework is SC4). `forced` (COMPACT_MIGRATE) bypasses the
 * size/sustain gates; the automatic FREE path honours them, so all-pinned
 * workloads — every regression smoke test — never PROPOSE. */
static void cp54_eval_migration(bool forced)
{
	u64 now = ktime_get_ns();
	pid_t pool = atomic_read(&cipher_cp54_pool_pid);
	u32 free_mask = 0, stranded_mask, new_mask;
	int g, lowest_part_g = CIPHER_CP54_NUM_GROUPS;
	int stranded_count, k, cnt;
	struct cipher_cp54_alloc *cand;
	pid_t cand_pid;
	u32 own_mask;

	cp54_abort_timeouts(now);

	/* Scan the ledger: free-group mask + the lowest group index owned by
	 * any PARTITION (a group is partition-owned if OCC is set and the
	 * owner pid is not the POOL). */
	for (g = 0; g < CIPHER_CP54_NUM_GROUPS; g++) {
		u32 v = (u32)atomic_read(&cipher_cp54_groups[g]);

		if (v == 0) {
			free_mask |= (1U << g);
		} else if ((v & CIPHER_CP54_GRP_OCC) &&
			   (pid_t)(v & CIPHER_CP54_GRP_PID_MASK) != pool) {
			if (g < lowest_part_g)
				lowest_part_g = g;
		}
	}

	/* A free group strictly above the lowest partition group is POOL-
	 * unreachable: the POOL is a count-only contiguous-low-prefix placement
	 * (Step 1.3b'), so a partition below blocks it from growing into it. */
	if (lowest_part_g >= CIPHER_CP54_NUM_GROUPS) {
		cipher_cp54_gap_since_ns = 0;
		return;
	}
	stranded_mask  = free_mask & ~((1U << (lowest_part_g + 1)) - 1U);
	stranded_count = hweight32(stranded_mask);
	if (stranded_count == 0) {
		cipher_cp54_gap_since_ns = 0;
		return;
	}
	if (cipher_cp54_gap_since_ns == 0)
		cipher_cp54_gap_since_ns = now;

	/* Policy gates — automatic path only; COMPACT_MIGRATE (forced) bypasses
	 * them. Operator-tunable via the module parameters (Track 3 SC4). */
	if (!forced) {
		if (stranded_count < cipher_cp54_mig_gap_min_grps)
			return;
		if (now - cipher_cp54_gap_since_ns <
		    (u64)cipher_cp54_mig_sustain_ms * NSEC_PER_MSEC)
			return;
	}

	/* The blocker is the partition owning lowest_part_g. Migrate it (and
	 * only it this pass) if it is migratable, IDLE, and not rate-limited. */
	cand_pid = (pid_t)((u32)atomic_read(&cipher_cp54_groups[lowest_part_g])
			   & CIPHER_CP54_GRP_PID_MASK);
	cand = cp54_meta_find(cand_pid);
	if (!cand || cand->qos_class != CIPHER_CP54_QOS_PARTITION)
		return;
	if (!cand->migratable ||
	    cand->migrate_state != CIPHER_CP54_MIG_IDLE)
		return;                                  /* pinned / busy */
	if (cand->last_migrate_ns != 0 &&
	    now - cand->last_migrate_ns <
	    (u64)cipher_cp54_mig_ratelimit_ms * NSEC_PER_MSEC)
		return;                                  /* rate-limited */

	own_mask = cp54_mask_of(cand_pid);
	k = hweight32(own_mask);
	if (k == 0)
		return;

	/* Count-preserving target: the k highest-indexed free groups. */
	new_mask = 0;
	cnt = 0;
	for (g = CIPHER_CP54_NUM_GROUPS - 1; g >= 0 && cnt < k; g--)
		if (free_mask & (1U << g)) {
			new_mask |= (1U << g);
			cnt++;
		}
	if (cnt < k)                       /* not enough free room this pass */
		return;
	if (new_mask == own_mask)          /* no-op */
		return;
	if ((int)__ffs(new_mask) <= lowest_part_g)
		return;                    /* move would not free the blocker */

	if (cp54_reserve(cand_pid, new_mask) != k) {
		cp54_release_rsvd(cand_pid);   /* lost a race — roll back */
		return;
	}
	cand->migrate_state  = CIPHER_CP54_MIG_PROPOSED;
	cand->target_mask    = new_mask;
	cand->proposed_ns    = now;
	cand->last_migrate_ns = now;       /* rate-limit gates from PROPOSE */
	atomic_inc(&cipher_cp54_stat_proposals);
	if (cipher_cp54_mig_verbose)
		pr_info("cipher_kmod: CP54 migrate PROPOSE pid=%d %#x -> %#x "
			"(%d grps)\n", cand_pid, own_mask, new_mask, k);
}

/* ---- ioctl entry points (called from cipher_dev.c) -------------------- */

long cipher_cp54_ioctl_allocate(unsigned long arg)
{
	struct cipher_cp54_allocate p;
	struct cipher_cp54_alloc *m;
	pid_t pid = current->pid;
	u32 mask;
	int need, have, shortfall;
	long rc = 0;

	if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
		return -EFAULT;
	if (p.qos_class != CIPHER_CP54_QOS_PARTITION &&
	    p.qos_class != CIPHER_CP54_QOS_SHARED &&
	    p.qos_class != CIPHER_CP54_QOS_POOL)
		return -EINVAL;

	mutex_lock(&cipher_cp54_lock);

	m = cp54_meta_get_or_create(pid);
	if (!m) {
		mutex_unlock(&cipher_cp54_lock);
		return -ENOMEM;          /* metadata table full */
	}

	/* A tenant already classified PARTITION or POOL cannot switch class
	 * by re-calling ALLOCATE — it must FREE first. A fresh or SHARED
	 * tenant may still be classified (SHARED owns no groups). Same-class
	 * re-calls (idempotent grow) are allowed. */
	if (m->qos_class != CIPHER_CP54_QOS_SHARED &&
	    m->qos_class != p.qos_class) {
		mutex_unlock(&cipher_cp54_lock);
		return -EINVAL;
	}

	switch (p.qos_class) {
	case CIPHER_CP54_QOS_POOL:
		/* Singleton: a second distinct PID cannot register POOL. */
		{
			int cur = atomic_read(&cipher_cp54_pool_pid);

			if (cur != 0 && cur != pid) {
				rc = -EEXIST;
				break;
			}
		}
		atomic_set(&cipher_cp54_pool_pid, pid);
		m->qos_class = CIPHER_CP54_QOS_POOL;
		/* Track 3 SC1 fact-2 / SC5: the POOL claims the maximal
		 * contiguous LOW PREFIX, not every free group — a free group
		 * above a live partition is POOL-unreachable (the count-only
		 * green ctx cannot place there). The constrained grant keeps
		 * the POOL valid + disjoint; Dynamic SM Migration compacts the
		 * stranded capacity back. Idempotent re-call (POOL.check_resize)
		 * grows the prefix after a partition frees. */
		cp54_pool_claim_low_prefix(pid);
		mask = cp54_mask_of(pid);
		break;

	case CIPHER_CP54_QOS_PARTITION:
		need = DIV_ROUND_UP(p.sm_count, CIPHER_CP54_SMS_PER_GROUP);
		if (need < 1)
			need = 1;
		if (need > CIPHER_CP54_NUM_GROUPS)
			need = CIPHER_CP54_NUM_GROUPS;
		m->qos_class = CIPHER_CP54_QOS_PARTITION;
		have = hweight32(cp54_mask_of(pid));      /* idempotent re-call */
		if (have < need) {
			have += cp54_claim(pid, need - have); /* free groups */
			if (have < need) {                    /* shrink the pool */
				shortfall = need - have;
				cp54_pool_shrink(shortfall);
				have += cp54_claim(pid, shortfall);
			}
		}
		mask = cp54_mask_of(pid);
		if (mask == 0)
			rc = -ENOSPC;
		break;

	default: /* CIPHER_CP54_QOS_SHARED — batch-pool member, owns no groups */
		m->qos_class = CIPHER_CP54_QOS_SHARED;
		mask = 0;
		break;
	}

	mutex_unlock(&cipher_cp54_lock);
	if (rc != 0)
		return rc;

	p.grp_mask_out  = mask;
	p.grp_count_out = hweight32(mask);
	if (copy_to_user((void __user *)arg, &p, sizeof(p)))
		return -EFAULT;
	return 0;
}

long cipher_cp54_ioctl_free(unsigned long arg)
{
	(void)arg;                          /* _IO — no payload */
	cp54_release_impl(current->pid);
	/* Track 3 SC2: a FREE changes ledger occupancy — re-evaluate
	 * compaction. The FREE *ioctl* is process context and may take the
	 * mutex (unlike the do_exit reaper, which shares cp54_release_impl
	 * but must stay lock-free). With the conservative default policy and
	 * zero migratable tenants this is a side-effect-free scan, so all-
	 * pinned workloads keep byte-identical FREE behaviour. */
	mutex_lock(&cipher_cp54_lock);
	cp54_eval_migration(false);
	mutex_unlock(&cipher_cp54_lock);
	return 0;
}

long cipher_cp54_ioctl_query(unsigned long arg)
{
	struct cipher_cp54_query q;
	pid_t pid = current->pid;
	pid_t pool = atomic_read(&cipher_cp54_pool_pid);
	struct cipher_cp54_alloc *m;
	int i, n_part = 0;

	memset(&q, 0, sizeof(q));

	/* n_partitions: registered PARTITION-class tenants. Lock-free scan —
	 * a torn read at worst mis-counts by one for one snapshot; benign for
	 * a diagnostic query. */
	for (i = 0; i < CIPHER_CP54_MAX_ALLOCS; i++) {
		m = &cipher_cp54_allocs[i];
		if (READ_ONCE(m->in_use) &&
		    READ_ONCE(m->qos_class) == CIPHER_CP54_QOS_PARTITION)
			n_part++;
	}
	q.n_partitions   = n_part;
	q.pool_grp_count = pool ? hweight32(cp54_mask_of(pool)) : 0;
	q.free_grp_count = cp54_free_count();
	q.my_grp_mask    = cp54_mask_of(pid);

	q.my_qos_class = 0xFFFFFFFFU;       /* sentinel: caller not registered */
	for (i = 0; i < CIPHER_CP54_MAX_ALLOCS; i++) {
		m = &cipher_cp54_allocs[i];
		if (READ_ONCE(m->in_use) && m->pid == pid) {
			q.my_qos_class = READ_ONCE(m->qos_class);
			break;
		}
	}

	if (copy_to_user((void __user *)arg, &q, sizeof(q)))
		return -EFAULT;
	return 0;
}

/* ---- Track 3 SC2 migration ioctls (nrs 16-20) ------------------------- */

/* nr 16 — SUBSCRIBE_MIGRATE: declare the caller `migratable` (1) or `pinned`
 * (0). Default is pinned; a tenant that never calls this is never proposed
 * to. Creates a metadata slot if the caller has none yet. */
long cipher_cp54_ioctl_subscribe_migrate(unsigned long arg)
{
	struct cipher_cp54_alloc *m;
	u32 val;

	if (copy_from_user(&val, (void __user *)arg, sizeof(val)))
		return -EFAULT;

	mutex_lock(&cipher_cp54_lock);
	m = cp54_meta_get_or_create(current->pid);
	if (!m) {
		mutex_unlock(&cipher_cp54_lock);
		return -ENOMEM;
	}
	m->migratable = val ? 1 : 0;
	mutex_unlock(&cipher_cp54_lock);
	return 0;
}

/* nr 17 — POLL_MIGRATE: read the caller's migration state. Lock-free reads
 * (QUERY discipline — a torn read at worst costs one re-poll). */
long cipher_cp54_ioctl_poll_migrate(unsigned long arg)
{
	struct cipher_cp54_migrate_poll out;
	pid_t pid = current->pid;
	struct cipher_cp54_alloc *m;
	int i;

	memset(&out, 0, sizeof(out));
	out.cur_mask = cp54_mask_of(pid);

	for (i = 0; i < CIPHER_CP54_MAX_ALLOCS; i++) {
		m = &cipher_cp54_allocs[i];
		if (READ_ONCE(m->in_use) && m->pid == pid) {
			out.migrate_state = READ_ONCE(m->migrate_state);
			out.target_mask   = READ_ONCE(m->target_mask);
			out.last_outcome  = READ_ONCE(m->last_outcome);
			break;
		}
	}

	if (copy_to_user((void __user *)arg, &out, sizeof(out)))
		return -EFAULT;
	return 0;
}

/* nr 18 — START_MIGRATE: PROPOSED -> MIGRATING (the tenant's commit of
 * intent). After this the kmod will not ABORT the migration; only the
 * tenant may, via ACK(ok=0). -EINVAL if the caller is not PROPOSED. */
long cipher_cp54_ioctl_start_migrate(unsigned long arg)
{
	struct cipher_cp54_alloc *m;
	long rc = 0;

	(void)arg;
	mutex_lock(&cipher_cp54_lock);
	m = cp54_meta_find(current->pid);
	if (!m || m->migrate_state != CIPHER_CP54_MIG_PROPOSED)
		rc = -EINVAL;
	else
		m->migrate_state = CIPHER_CP54_MIG_MIGRATING;
	mutex_unlock(&cipher_cp54_lock);
	return rc;
}

/* nr 19 — ACK_MIGRATE: MIGRATING -> COMMIT (ok=1) or -> ABORT (ok=0, the
 * tenant-NACK path — the tenant reports its migrate() did not complete, so
 * it still runs on its old groups). -EINVAL if the caller is not MIGRATING. */
long cipher_cp54_ioctl_ack_migrate(unsigned long arg)
{
	struct cipher_cp54_alloc *m;
	long rc = 0;
	u32 ok;

	if (copy_from_user(&ok, (void __user *)arg, sizeof(ok)))
		return -EFAULT;

	mutex_lock(&cipher_cp54_lock);
	m = cp54_meta_find(current->pid);
	if (!m || m->migrate_state != CIPHER_CP54_MIG_MIGRATING)
		rc = -EINVAL;
	else if (ok)
		cp54_commit_migration(m);
	else
		cp54_abort_migration(m, CIPHER_CP54_MIGOUT_ABORTED_TENANT_NACK);
	mutex_unlock(&cipher_cp54_lock);
	return rc;
}

/* nr 20 — COMPACT_MIGRATE: force a compaction-evaluation pass (operator
 * control / deterministic test trigger). Bypasses the conservative
 * size/sustain gates; still honours opt-in, IDLE, and rate-limit. */
long cipher_cp54_ioctl_compact_migrate(unsigned long arg)
{
	(void)arg;
	mutex_lock(&cipher_cp54_lock);
	cp54_eval_migration(true);
	mutex_unlock(&cipher_cp54_lock);
	return 0;
}

/* ---- Track 3 SC4: /proc/cipher/migrations emitter --------------------- */

/* Read-only operator + measurement view: policy params in effect, current
 * POOL-unreachable gap state, aggregate counters, and a row per migratable
 * tenant. Lock-free reads (diagnostic — QUERY discipline). */
int cipher_cp54_migration_proc_show(struct seq_file *sf, void *v)
{
	u64 now = ktime_get_ns();
	u64 gap_since = cipher_cp54_gap_since_ns;
	pid_t pool = atomic_read(&cipher_cp54_pool_pid);
	u32 free_mask = 0;
	int g, i, lowest_part_g = CIPHER_CP54_NUM_GROUPS, stranded = 0;

	(void)v;
	seq_printf(sf, "policy:   gap_min_grps=%d sustain_ms=%u "
	           "ratelimit_ms=%u verbose=%d\n",
	           cipher_cp54_mig_gap_min_grps, cipher_cp54_mig_sustain_ms,
	           cipher_cp54_mig_ratelimit_ms, cipher_cp54_mig_verbose);

	for (g = 0; g < CIPHER_CP54_NUM_GROUPS; g++) {
		u32 gv = (u32)atomic_read(&cipher_cp54_groups[g]);

		if (gv == 0)
			free_mask |= (1U << g);
		else if ((gv & CIPHER_CP54_GRP_OCC) &&
			 (pid_t)(gv & CIPHER_CP54_GRP_PID_MASK) != pool &&
			 g < lowest_part_g)
			lowest_part_g = g;
	}
	if (lowest_part_g < CIPHER_CP54_NUM_GROUPS)
		stranded = hweight32(free_mask &
			   ~((1U << (lowest_part_g + 1)) - 1U));
	seq_printf(sf, "gap:      stranded_groups=%d gap_age_ms=%llu\n",
	           stranded,
	           gap_since ? (now - gap_since) / NSEC_PER_MSEC : 0ULL);

	seq_printf(sf, "counters: proposals=%d commits=%d\n",
	           atomic_read(&cipher_cp54_stat_proposals),
	           atomic_read(&cipher_cp54_stat_commits));
	seq_printf(sf, "          aborts: timeout=%d tenant_nack=%d "
	           "kmod_refused=%d\n",
	           atomic_read(&cipher_cp54_stat_abort_timeout),
	           atomic_read(&cipher_cp54_stat_abort_nack),
	           atomic_read(&cipher_cp54_stat_abort_refused));

	for (i = 0; i < CIPHER_CP54_MAX_ALLOCS; i++) {
		struct cipher_cp54_alloc *m = &cipher_cp54_allocs[i];

		if (READ_ONCE(m->in_use) && READ_ONCE(m->migratable))
			seq_printf(sf, "tenant:   pid=%d qos=%u cur_mask=%#06x "
			           "migrate_state=%u last_outcome=%u "
			           "migration_count=%u\n",
			           m->pid, m->qos_class, cp54_mask_of(m->pid),
			           m->migrate_state, m->last_outcome,
			           m->migration_count);
	}
	return 0;
}

/* ---- init / exit ------------------------------------------------------ */

int cipher_cp54_sched_init(void)
{
	int g;

	for (g = 0; g < CIPHER_CP54_NUM_GROUPS; g++)
		atomic_set(&cipher_cp54_groups[g], 0);
	memset(cipher_cp54_allocs, 0, sizeof(cipher_cp54_allocs));
	atomic_set(&cipher_cp54_pool_pid, 0);
	cipher_cp54_gap_since_ns = 0;
	atomic_set(&cipher_cp54_stat_proposals, 0);
	atomic_set(&cipher_cp54_stat_commits, 0);
	atomic_set(&cipher_cp54_stat_abort_timeout, 0);
	atomic_set(&cipher_cp54_stat_abort_nack, 0);
	atomic_set(&cipher_cp54_stat_abort_refused, 0);
	pr_info("cipher_kmod: CP 5.4 arbitration ledger — %d × %d-SM groups "
	        "(%d SMs); legacy nr-9 4-SM allocator deactivated\n",
	        CIPHER_CP54_NUM_GROUPS, CIPHER_CP54_SMS_PER_GROUP,
	        CIPHER_CP54_NUM_GROUPS * CIPHER_CP54_SMS_PER_GROUP);
	return 0;
}

void cipher_cp54_sched_exit(void)
{
	int g;

	for (g = 0; g < CIPHER_CP54_NUM_GROUPS; g++)
		atomic_set(&cipher_cp54_groups[g], 0);
}
