// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_flops.c — CP 3.3 continuous FLOP telemetry + per-tenant MFU.
 *
 * The GH100 SM perfmon PRI register map is not published in the open
 * kernel-module tree, so cipher_kmod cannot read hardware FLOP counters
 * directly. The cipher_flopd root daemon does the hardware-counter read
 * via CUPTI PM Sampling and submits a device-wide achieved-FLOP sample
 * (~10 Hz) through CIPHER_SUBMIT_FLOP_SAMPLE. This TU:
 *
 *   - owns the continuous FLOP-series ring (the kmod owns the series),
 *   - derives per-tenant attributed FLOPs from the per-tenant launch
 *     deltas the kmod already keeps (launches_total, fed by the CUPTI
 *     launch-callback path) — "MFU as a free byproduct of telemetry",
 *   - exposes /proc/cipher/flops and the CIPHER_QUERY_FLOPS ioctl.
 *
 * Per-tenant FLOPs are ATTRIBUTED (device FLOP/s × launch-share), not a
 * hardware per-context measurement — PM Sampling is device-wide. This is
 * proportional attribution; gate-sufficient for real-time MFU.
 *
 * None of this is on the kprobe hot path: SUBMIT runs ~10 Hz from the
 * daemon, QUERY and the proc file are cold reads.
 */
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/capability.h>
#include <linux/hashtable.h>
#include <linux/rcupdate.h>
#include <linux/spinlock.h>
#include <linux/jiffies.h>
#include <linux/seq_file.h>
#include <linux/math64.h>
#include <linux/string.h>
#include <linux/sched.h>

#include "cipher_internal.h"
#include "cipher_ioctl.h"

/* H100 FP16 dense sparse spec — the MFU denominator CIPHER uses elsewhere
 * (cipher_metrics.py PEAK_FLOPS_SPEC = 989e12). */
#define CIPHER_PEAK_FLOPS_PER_S   989000000000000ULL
/* peak / 100000 — divisor turning FLOP/s into MFU milli-percent. */
#define CIPHER_MFU_DIVISOR        9890000000ULL

#define CIPHER_FLOP_RING_SIZE     256
/* a sample older than this is reported STALE. */
#define CIPHER_FLOP_STALE_JIFFIES (2 * HZ)

struct cipher_flop_ring_ent {
	u64 timestamp_ns;
	u64 interval_ns;
	u64 device_tensor_flops_per_s;
	u32 tensor_pipe_milli_pct;
	u32 sm_clock_mhz;
	u32 sm_util_pct;
	u32 power_mw;
	u32 source_seq;
	u64 recv_jiffies;
};

static DEFINE_SPINLOCK(cipher_flop_lock);
static struct cipher_flop_ring_ent cipher_flop_ring[CIPHER_FLOP_RING_SIZE];
static u32 cipher_flop_head;            /* next write index */
static u32 cipher_flop_count;           /* populated, capped at RING_SIZE */
static u64 cipher_flop_total_samples;   /* lifetime count */

/* Copy out the most recent ring entry. Returns false if none yet. */
static bool cipher_flop_latest(struct cipher_flop_ring_ent *out)
{
	bool have;

	spin_lock(&cipher_flop_lock);
	have = (cipher_flop_count > 0);
	if (have) {
		u32 idx = (cipher_flop_head + CIPHER_FLOP_RING_SIZE - 1)
			  % CIPHER_FLOP_RING_SIZE;
		*out = cipher_flop_ring[idx];
	}
	spin_unlock(&cipher_flop_lock);
	return have;
}

/*
 * CIPHER_SUBMIT_FLOP_SAMPLE (nr 11) — cipher_flopd -> kmod, CAP_SYS_ADMIN.
 * Stores the device sample in the ring and re-derives per-tenant
 * attributed FLOPs from per-tenant launch deltas.
 */
long cipher_flops_submit(unsigned long arg)
{
	struct cipher_flop_sample s;
	struct cipher_flop_ring_ent ent;
	struct cipher_pid_stats *e;
	int bkt;
	u64 total_delta = 0;
	u64 device_rate;

	if (!capable(CAP_SYS_ADMIN))
		return -EPERM;
	if (copy_from_user(&s, (void __user *)arg, sizeof(s)))
		return -EFAULT;

	/* Validation — reject obviously-corrupt samples. */
	if (s.timestamp_ns == 0)
		return -EINVAL;
	if (s.tensor_pipe_milli_pct > 100000)   /* 100.000 % */
		return -EINVAL;
	if (s.sm_util_pct > 100)
		return -EINVAL;

	device_rate = s.device_tensor_flops_per_s;

	ent.timestamp_ns              = s.timestamp_ns;
	ent.interval_ns               = s.interval_ns;
	ent.device_tensor_flops_per_s = device_rate;
	ent.tensor_pipe_milli_pct     = s.tensor_pipe_milli_pct;
	ent.sm_clock_mhz              = s.sm_clock_mhz;
	ent.sm_util_pct               = s.sm_util_pct;
	ent.power_mw                  = s.power_mw;
	ent.source_seq                = s.source_seq;
	ent.recv_jiffies              = get_jiffies_64();

	spin_lock(&cipher_flop_lock);
	cipher_flop_ring[cipher_flop_head] = ent;
	cipher_flop_head = (cipher_flop_head + 1) % CIPHER_FLOP_RING_SIZE;
	if (cipher_flop_count < CIPHER_FLOP_RING_SIZE)
		cipher_flop_count++;
	cipher_flop_total_samples++;
	spin_unlock(&cipher_flop_lock);

	/* Per-tenant attribution. Pass 1 sums launch deltas; pass 2 writes
	 * attributed FLOP/s and advances each tenant's launch baseline.
	 * launches_at_last_flop / attributed_flops_per_s are written ONLY
	 * here (single-writer — SUBMIT is serialised by the one daemon). */
	rcu_read_lock();
	hash_for_each_rcu(cipher_pid_table, bkt, e, node) {
		u64 lt   = READ_ONCE(e->launches_total);
		u64 base = READ_ONCE(e->launches_at_last_flop);

		if (lt > base)
			total_delta += (lt - base);
	}
	hash_for_each_rcu(cipher_pid_table, bkt, e, node) {
		u64 lt   = READ_ONCE(e->launches_total);
		u64 base = READ_ONCE(e->launches_at_last_flop);
		u64 delta = (lt > base) ? (lt - base) : 0;
		u32 share_ppm = 0;
		u64 attributed = 0;

		if (total_delta > 0) {
			share_ppm = (u32)div64_u64(delta * 1000000ULL,
						   total_delta);
			/* device_rate/1e6 (~1e9) × share_ppm (<=1e6) — no
			 * u64 overflow; sub-1e6 precision loss is negligible
			 * against a ~1e15 FLOP/s rate. */
			attributed = div_u64(device_rate, 1000000ULL)
				     * share_ppm;
		}
		WRITE_ONCE(e->attributed_flops_per_s, attributed);
		WRITE_ONCE(e->launches_at_last_flop, lt);
	}
	rcu_read_unlock();

	return 0;
}

/* device MFU, milli-percent — attributed FLOP/s vs the 989 TFLOP/s peak.
 * div64_u64: the divisor (9.89e9) exceeds 32 bits, so div_u64 cannot be used. */
static u32 cipher_flop_mfu_milli(u64 flops_per_s)
{
	return (u32)div64_u64(flops_per_s, CIPHER_MFU_DIVISOR);
}

/*
 * CIPHER_QUERY_FLOPS (nr 12) — unprivileged read. Fills the device header
 * + up to max_tenants per-tenant rows.
 */
long cipher_flops_query(unsigned long arg)
{
	struct cipher_flop_query *q;
	struct cipher_flop_ring_ent latest;
	struct cipher_pid_stats *e;
	bool have;
	int bkt;
	u32 cap, n = 0;
	long rc = 0;

	q = kzalloc(sizeof(*q), GFP_KERNEL);
	if (!q)
		return -ENOMEM;
	if (copy_from_user(q, (void __user *)arg, sizeof(*q))) {
		rc = -EFAULT;
		goto out;
	}

	cap = q->max_tenants;
	if (cap > CIPHER_FLOP_QUERY_MAX_TENANTS)
		cap = CIPHER_FLOP_QUERY_MAX_TENANTS;

	memset(q, 0, sizeof(*q));
	q->max_tenants = cap;

	have = cipher_flop_latest(&latest);
	if (!have) {
		q->status = CIPHER_FLOP_STATUS_NO_SOURCE;
	} else {
		u64 now = get_jiffies_64();
		u64 age_j = now - latest.recv_jiffies;

		q->latest_timestamp_ns       = latest.timestamp_ns;
		q->device_tensor_flops_per_s = latest.device_tensor_flops_per_s;
		q->device_mfu_milli_pct      = latest.tensor_pipe_milli_pct;
		q->sm_clock_mhz              = latest.sm_clock_mhz;
		q->sm_util_pct               = latest.sm_util_pct;
		q->sample_age_ms             = (u32)jiffies64_to_msecs(age_j);
		q->status = (age_j > CIPHER_FLOP_STALE_JIFFIES)
			     ? CIPHER_FLOP_STATUS_STALE
			     : CIPHER_FLOP_STATUS_LIVE;
	}
	spin_lock(&cipher_flop_lock);
	q->ring_samples = cipher_flop_count;
	spin_unlock(&cipher_flop_lock);

	rcu_read_lock();
	hash_for_each_rcu(cipher_pid_table, bkt, e, node) {
		struct cipher_flop_tenant *t;
		u64 lt, base, delta, attributed;

		if (n >= cap)
			break;
		attributed = READ_ONCE(e->attributed_flops_per_s);
		lt   = READ_ONCE(e->launches_total);
		base = READ_ONCE(e->launches_at_last_flop);
		delta = (lt > base) ? (lt - base) : 0;

		t = &q->tenants[n];
		t->pid  = e->pid;
		t->tgid = e->tgid;
		memcpy(t->tenant_id, e->tenant_id, CIPHER_TENANT_ID_LEN);
		t->tenant_id[CIPHER_TENANT_ID_LEN - 1] = '\0';
		t->launches_delta        = delta;
		t->attributed_flops_per_s = attributed;
		t->mfu_milli_pct          = cipher_flop_mfu_milli(attributed);
		/* share recovered from attributed vs device rate. */
		if (have && latest.device_tensor_flops_per_s > 0)
			t->launch_share_ppm = (u32)div64_u64(attributed,
				max_t(u64, div_u64(
					latest.device_tensor_flops_per_s,
					1000000ULL), 1ULL));
		n++;
	}
	rcu_read_unlock();
	q->n_tenants = n;

	if (copy_to_user((void __user *)arg, q, sizeof(*q)))
		rc = -EFAULT;
out:
	kfree(q);
	return rc;
}

/* /proc/cipher/flops — human-readable device + per-tenant FLOP telemetry. */
int cipher_flops_proc_show(struct seq_file *m, void *v)
{
	struct cipher_flop_ring_ent latest;
	struct cipher_pid_stats *e;
	u64 tf, total_samples;
	u32 count;
	int bkt;
	bool have;

	seq_puts(m, "# CIPHER FLOP telemetry — schema 1 (CP 3.3)\n");

	spin_lock(&cipher_flop_lock);
	count = cipher_flop_count;
	total_samples = cipher_flop_total_samples;
	spin_unlock(&cipher_flop_lock);

	have = cipher_flop_latest(&latest);
	if (!have || total_samples == 0) {
		seq_puts(m, "device  status=no-flop-source "
			    "(cipher_flopd not running / no sample yet)\n");
		seq_puts(m, "per-tenant FLOPs/MFU: n/a\n");
		return 0;
	}

	{
		u64 age_j = get_jiffies_64() - latest.recv_jiffies;
		u32 age_ms = (u32)jiffies64_to_msecs(age_j);
		const char *st = (age_j > CIPHER_FLOP_STALE_JIFFIES)
				 ? "stale" : "live";

		tf = latest.device_tensor_flops_per_s;
		seq_printf(m,
			"device  status=%s  age_ms=%u  tensor_tflops=%llu.%03llu  "
			"mfu_pct=%u.%03u  sm_clock_mhz=%u  sm_util_pct=%u  "
			"power_w=%u.%03u\n",
			st, age_ms,
			div64_u64(tf, 1000000000000ULL),
			div_u64(tf % 1000000000000ULL, 1000000000ULL),
			latest.tensor_pipe_milli_pct / 1000U,
			latest.tensor_pipe_milli_pct % 1000U,
			latest.sm_clock_mhz, latest.sm_util_pct,
			latest.power_mw / 1000U, latest.power_mw % 1000U);
		seq_printf(m,
			"device  series_samples=%u  total_samples=%llu  "
			"sample_interval_ns=%llu   # the continuous ring\n",
			count, total_samples, latest.interval_ns);
	}

	seq_puts(m, "\n# per-tenant  (attributed: device FLOP/s x launch-share)\n");
	seq_puts(m, "PID      TGID     TENANT           LAUNCH_DELTA  "
		    "FLOPS_PER_S       MFU%\n");

	rcu_read_lock();
	hash_for_each_rcu(cipher_pid_table, bkt, e, node) {
		u64 attributed = READ_ONCE(e->attributed_flops_per_s);
		u64 lt   = READ_ONCE(e->launches_total);
		u64 base = READ_ONCE(e->launches_at_last_flop);
		u64 delta = (lt > base) ? (lt - base) : 0;
		u32 mfu = cipher_flop_mfu_milli(attributed);
		const char *tnt = e->tenant_id[0] ? e->tenant_id : "-";

		seq_printf(m,
			"%-8u %-8u %-16.16s %-13llu %-17llu %u.%03u\n",
			e->pid, e->tgid, tnt,
			(unsigned long long)delta,
			(unsigned long long)attributed,
			mfu / 1000U, mfu % 1000U);
	}
	rcu_read_unlock();

	return 0;
}

int cipher_flops_init(void)
{
	spin_lock_init(&cipher_flop_lock);
	cipher_flop_head = 0;
	cipher_flop_count = 0;
	cipher_flop_total_samples = 0;
	pr_info("cipher_kmod: CP 3.3 FLOP telemetry ready "
		"(/proc/cipher/flops, ioctl nr 11/12)\n");
	return 0;
}

void cipher_flops_exit(void)
{
	/* Nothing dynamically allocated; the ring is static storage. */
}
