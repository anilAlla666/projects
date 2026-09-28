// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_proc.c -- /proc/cipher/stats seq_file emitter.
 *
 * Output structure (Phase 2):
 *   - header (uptime, hooked addr, totals)
 *   - Section A: Frontend ioctls (nr 200..218)
 *   - Section B: RM ioctls (nr 0x27..0x5e)
 *   - OTHER bucket (decode-miss residual)
 *   - Per-PID summary with TENANT column (top 16 by total)
 *   - Per-TGID summary with TENANT column; main-thread tenant wins,
 *     worker thread tenant used as fallback only
 */
#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/hashtable.h>
#include <linux/jiffies.h>
#include <linux/sort.h>
#include <linux/slab.h>
#include <linux/rcupdate.h>
#include <linux/string.h>
#include <linux/uaccess.h>

#include "cipher_internal.h"
#include "cipher_ioctl.h"

#define CIPHER_PROC_TOPN  16

static struct proc_dir_entry *cipher_proc_dir;
static struct proc_dir_entry *cipher_proc_stats_entry;
static struct proc_dir_entry *cipher_proc_bar0_entry;
static struct proc_dir_entry *cipher_proc_gpu_state_entry;
static struct proc_dir_entry *cipher_proc_flops_entry;
static struct proc_dir_entry *cipher_proc_migrations_entry;
static struct proc_dir_entry *cipher_proc_arenas_entry;
static struct proc_dir_entry *cipher_proc_classify_stats_entry;  /* W2 Step 5 */
static struct proc_dir_entry *cipher_proc_dsm_proposals_entry;   /* W3 S4 II-a */
static struct proc_dir_entry *cipher_proc_sense_session_entry;   /* W4 Step 5 */

struct cipher_topn_row {
	pid_t pid;
	pid_t tgid;
	char  comm[TASK_COMM_LEN];
	char  tenant_id[CIPHER_TENANT_ID_LEN];
	s64   total;
	s64   rm_ctrl;
	s64   rm_alloc;
	s64   rm_free;
	s64   map_mem;
	s64   other;
	s64   errors;
	u64   first_seen_jiffies;
	/* Phase 3 Task 3 telemetry snapshot, READ_ONCE-d at the same point
	 * as the atomic counters. enc/dec util are stored on the kernel
	 * entry but intentionally not surfaced here -- column count is
	 * already wide; they're available via future Prometheus export. */
	u32   sm_util_pct;
	u32   mem_util_pct;
	u64   launches_total;
	u64   last_telemetry_jiffies;
};

struct cipher_tgid_row {
	pid_t tgid;
	char  comm[TASK_COMM_LEN];
	char  tenant_id[CIPHER_TENANT_ID_LEN];
	bool  tenant_from_main;   /* set when filled from PID==TGID row */
	s64   total;
	s64   rm_ctrl;
	s64   rm_alloc;
	s64   rm_free;
	s64   map_mem;
	s64   other;
	s64   errors;
	int   thread_count;
	/* Phase 3 Task 3 rollup: sum of launches across LWPs, max SM% seen. */
	u64   launches_sum;
	u32   max_sm_util_pct;
};

static int cipher_topn_cmp(const void *a, const void *b)
{
	const struct cipher_topn_row *ra = a;
	const struct cipher_topn_row *rb = b;

	if (rb->total > ra->total) return  1;
	if (rb->total < ra->total) return -1;
	return 0;
}

static int cipher_tgid_cmp(const void *a, const void *b)
{
	const struct cipher_tgid_row *ra = a;
	const struct cipher_tgid_row *rb = b;

	if (rb->total > ra->total) return  1;
	if (rb->total < ra->total) return -1;
	return 0;
}

static void cipher_emit_family(struct seq_file *m, u8 family,
			       const char *header,
			       const char *nr_label,
			       const char *nr_underline,
			       const char *nr_fmt,
			       int name_width)
{
	int i;
	s64 sub_count = 0, sub_err = 0;
	char fmt_buf[96];

	seq_printf(m, "%s\n", header);
	seq_printf(m, "  %s  %-*s  %-11s  %-11s\n",
		   nr_label, name_width, "name", "count", "errors");
	seq_printf(m, "  %s  %.*s  %s  %s\n",
		   nr_underline, name_width,
		   "----------------------------------------",
		   "-----------", "-----------");

	scnprintf(fmt_buf, sizeof(fmt_buf),
		  "  %s  %%-%ds  %%11lld  %%11lld%%s\n", nr_fmt, name_width);

	for (i = 0; i < CIPHER_NV_IOCTL_COUNT; i++) {
		s64 c, ec;

		if (cipher_nv_ioctls[i].family != family)
			continue;
		c  = atomic64_read(&cipher_global_cmd_counts[i]);
		ec = atomic64_read(&cipher_global_cmd_errors[i]);
		if (c == 0 && ec == 0)
			continue;
		seq_printf(m, fmt_buf,
			   cipher_nv_ioctls[i].nr,
			   cipher_nv_ioctls[i].name,
			   (long long)c, (long long)ec,
			   (cipher_nv_ioctls[i].nr == 0x2a) ? "  *dominant*" : "");
		sub_count += c;
		sub_err   += ec;
	}
	seq_printf(m, "  %s  %-*s  %11lld  %11lld  (subtotal)\n",
		   nr_underline, name_width, "",
		   (long long)sub_count, (long long)sub_err);
}

static int cipher_proc_show(struct seq_file *m, void *v)
{
	u64 now = get_jiffies_64();
	u64 uptime_jiffies = now - cipher_load_jiffies;
	int i, bkt, npids = 0, n;
	struct cipher_pid_stats *e;
	struct cipher_topn_row *rows;
	s64 total_observed, total_errors, oc, oe;

	seq_printf(m, "cipher_kmod 0.6.5  uptime=%llu jiffies (%llu.%02llu s)\n",
		   (unsigned long long)uptime_jiffies,
		   (unsigned long long)(uptime_jiffies / HZ),
		   (unsigned long long)((uptime_jiffies * 100ULL / HZ) % 100));
	seq_printf(m, "nvidia_unlocked_ioctl @ 0x%lx [hooked]\n",
		   cipher_hooked_addr);

	total_observed =
		atomic64_read(&cipher_global_cmd_counts[CIPHER_NV_IOCTL_TOTAL_SLOT]);
	total_errors =
		atomic64_read(&cipher_global_cmd_errors[CIPHER_NV_IOCTL_TOTAL_SLOT]);
	seq_printf(m,
		   "total_observed=%lld  total_errors=%lld  alloc_failures=%lld  reaped=%lld\n\n",
		   (long long)total_observed,
		   (long long)total_errors,
		   (long long)atomic64_read(&cipher_alloc_failures),
		   (long long)atomic64_read(&cipher_reaped_count));

	cipher_emit_family(m, CIPHER_FAM_FRONTEND,
			   "Frontend ioctls (nr 200..218):",
			   "nr ", "---", "%3u", 32);
	seq_putc(m, '\n');
	cipher_emit_family(m, CIPHER_FAM_RM,
			   "RM ioctls (nr 0x27..0x5e):",
			   "nr  ", "----", "0x%02x", 36);

	oc = atomic64_read(&cipher_global_cmd_counts[CIPHER_NV_IOCTL_OTHER_SLOT]);
	oe = atomic64_read(&cipher_global_cmd_errors[CIPHER_NV_IOCTL_OTHER_SLOT]);
	seq_printf(m, "\nOTHER (cmd outside known NV_ESC ranges):\n");
	seq_printf(m, "                                              %11lld  %11lld%s\n",
		   (long long)oc, (long long)oe,
		   (oc > 0) ? "  <- examine" : "");

	rcu_read_lock();
	hash_for_each_rcu(cipher_pid_table, bkt, e, node)
		npids++;
	rcu_read_unlock();

	rows = kmalloc_array(npids ? npids : 1, sizeof(*rows), GFP_KERNEL);
	if (!rows) {
		seq_printf(m, "\n(per-PID snapshot kmalloc failed: npids=%d)\n",
			   npids);
		return 0;
	}

	n = 0;
	rcu_read_lock();
	hash_for_each_rcu(cipher_pid_table, bkt, e, node) {
		if (n >= npids)
			break;
		rows[n].pid  = e->pid;
		rows[n].tgid = e->tgid;
		memcpy(rows[n].comm, e->comm, TASK_COMM_LEN);
		rows[n].comm[TASK_COMM_LEN - 1] = '\0';
		memcpy(rows[n].tenant_id, e->tenant_id, CIPHER_TENANT_ID_LEN);
		rows[n].tenant_id[CIPHER_TENANT_ID_LEN - 1] = '\0';
		rows[n].total    = atomic64_read(&e->total);
		rows[n].rm_ctrl  = atomic64_read(&e->per_slot[CIPHER_SLOT_RM_CONTROL]);
		rows[n].rm_alloc = atomic64_read(&e->per_slot[CIPHER_SLOT_RM_ALLOC]);
		rows[n].rm_free  = atomic64_read(&e->per_slot[CIPHER_SLOT_RM_FREE]);
		rows[n].map_mem  = atomic64_read(&e->per_slot[CIPHER_SLOT_RM_MAP_MEMORY]);
		rows[n].other    = atomic64_read(&e->other_count);
		rows[n].errors   = atomic64_read(&e->errors);
		rows[n].first_seen_jiffies     = e->first_seen_jiffies;
		rows[n].sm_util_pct            = READ_ONCE(e->sm_util_pct);
		rows[n].mem_util_pct           = READ_ONCE(e->mem_util_pct);
		rows[n].launches_total         = READ_ONCE(e->launches_total);
		rows[n].last_telemetry_jiffies = READ_ONCE(e->last_telemetry_jiffies);
		n++;
	}
	rcu_read_unlock();

	sort(rows, n, sizeof(*rows), cipher_topn_cmp, NULL);

	seq_printf(m, "\nPer-PID summary (top %d by total, %d PIDs total):\n",
		   CIPHER_PROC_TOPN, n);
	seq_puts(m,
		 "  PID      TGID     COMM             TENANT          TOTAL    RM_CTRL  RM_ALLOC  RM_FREE  MAP_MEM  OTHER    ERR    AGE_S   SM%  MEM%  LAUNCHES    AGE_TM\n");

	for (i = 0; i < n && i < CIPHER_PROC_TOPN; i++) {
		u64 age = now - rows[i].first_seen_jiffies;
		const char *tnt = rows[i].tenant_id[0] ? rows[i].tenant_id : "-";
		bool have_tm = (rows[i].last_telemetry_jiffies != 0);
		u64 age_tm   = have_tm ? (now - rows[i].last_telemetry_jiffies) : 0;

		seq_printf(m,
			   "  %-7u  %-7u  %-16s %-15.15s %-8lld %-8lld %-9lld %-8lld %-8lld %-8lld %-6lld %llu.%02llu",
			   rows[i].pid, rows[i].tgid, rows[i].comm, tnt,
			   (long long)rows[i].total,
			   (long long)rows[i].rm_ctrl,
			   (long long)rows[i].rm_alloc,
			   (long long)rows[i].rm_free,
			   (long long)rows[i].map_mem,
			   (long long)rows[i].other,
			   (long long)rows[i].errors,
			   (unsigned long long)(age / HZ),
			   (unsigned long long)((age * 100ULL / HZ) % 100));
		if (have_tm)
			seq_printf(m, "   %3u   %3u  %-10llu  %llu.%02llu\n",
				   rows[i].sm_util_pct,
				   rows[i].mem_util_pct,
				   (unsigned long long)rows[i].launches_total,
				   (unsigned long long)(age_tm / HZ),
				   (unsigned long long)((age_tm * 100ULL / HZ) % 100));
		else
			seq_puts(m, "     -     -  -           -\n");
	}

	/* Per-TGID rollup. Main thread (pid==tgid) wins for tenant_id;
	 * worker threads' tenant_id is fallback only. */
	{
		struct cipher_tgid_row *trows;
		int t = 0, j;

		trows = kmalloc_array(n ? n : 1, sizeof(*trows), GFP_KERNEL);
		if (!trows) {
			seq_printf(m, "\n(per-TGID kmalloc failed: n=%d)\n", n);
			kfree(rows);
			return 0;
		}

		for (i = 0; i < n; i++) {
			bool is_main = (rows[i].pid == rows[i].tgid);
			bool has_tenant = (rows[i].tenant_id[0] != '\0');

			for (j = 0; j < t; j++)
				if (trows[j].tgid == rows[i].tgid) break;
			if (j == t) {
				trows[t].tgid = rows[i].tgid;
				memcpy(trows[t].comm, rows[i].comm, TASK_COMM_LEN);
				trows[t].comm[TASK_COMM_LEN - 1] = '\0';
				trows[t].tenant_id[0] = '\0';
				trows[t].tenant_from_main = false;
				trows[t].total = trows[t].rm_ctrl = 0;
				trows[t].rm_alloc = trows[t].rm_free = 0;
				trows[t].map_mem = trows[t].other = 0;
				trows[t].errors = 0;
				trows[t].thread_count = 0;
				trows[t].launches_sum = 0;
				trows[t].max_sm_util_pct = 0;
				t++;
			}
			trows[j].total      += rows[i].total;
			trows[j].rm_ctrl    += rows[i].rm_ctrl;
			trows[j].rm_alloc   += rows[i].rm_alloc;
			trows[j].rm_free    += rows[i].rm_free;
			trows[j].map_mem    += rows[i].map_mem;
			trows[j].other      += rows[i].other;
			trows[j].errors     += rows[i].errors;
			trows[j].thread_count++;
			trows[j].launches_sum += rows[i].launches_total;
			if (rows[i].sm_util_pct > trows[j].max_sm_util_pct)
				trows[j].max_sm_util_pct = rows[i].sm_util_pct;

			/* Tenant resolution: main thread always wins; worker
			 * threads only fill when nothing better is set. */
			if (has_tenant) {
				if (is_main) {
					memcpy(trows[j].tenant_id,
					       rows[i].tenant_id,
					       CIPHER_TENANT_ID_LEN);
					trows[j].tenant_id[CIPHER_TENANT_ID_LEN - 1] = '\0';
					trows[j].tenant_from_main = true;
				} else if (!trows[j].tenant_from_main &&
					   !trows[j].tenant_id[0]) {
					memcpy(trows[j].tenant_id,
					       rows[i].tenant_id,
					       CIPHER_TENANT_ID_LEN);
					trows[j].tenant_id[CIPHER_TENANT_ID_LEN - 1] = '\0';
				}
			}
		}

		sort(trows, t, sizeof(*trows), cipher_tgid_cmp, NULL);

		seq_printf(m,
			   "\nPer-TGID summary (top %d by total, %d TGIDs total):\n",
			   CIPHER_PROC_TOPN, t);
		seq_puts(m,
			 "  TGID     COMM             TENANT          THR  TOTAL    RM_CTRL  RM_ALLOC  RM_FREE  MAP_MEM  OTHER    ERR    LAUNCHES    MAX_SM%\n");

		for (i = 0; i < t && i < CIPHER_PROC_TOPN; i++) {
			const char *tnt = trows[i].tenant_id[0] ? trows[i].tenant_id : "-";

			seq_printf(m,
				   "  %-7u  %-16s %-15.15s %-3d  %-8lld %-8lld %-9lld %-8lld %-8lld %-8lld %-6lld %-10llu    %u\n",
				   trows[i].tgid, trows[i].comm, tnt,
				   trows[i].thread_count,
				   (long long)trows[i].total,
				   (long long)trows[i].rm_ctrl,
				   (long long)trows[i].rm_alloc,
				   (long long)trows[i].rm_free,
				   (long long)trows[i].map_mem,
				   (long long)trows[i].other,
				   (long long)trows[i].errors,
				   (unsigned long long)trows[i].launches_sum,
				   trows[i].max_sm_util_pct);
		}

		kfree(trows);
	}

	kfree(rows);
	return 0;
}

static int cipher_proc_open(struct inode *inode, struct file *file)
{
	return single_open(file, cipher_proc_show, NULL);
}

static const struct proc_ops cipher_proc_fops = {
	.proc_open    = cipher_proc_open,
	.proc_read    = seq_read,
	.proc_lseek   = seq_lseek,
	.proc_release = single_release,
};

static int cipher_proc_bar0_open(struct inode *inode, struct file *file)
{
	return single_open(file, cipher_bar0_proc_show, NULL);
}

static const struct proc_ops cipher_proc_bar0_fops = {
	.proc_open    = cipher_proc_bar0_open,
	.proc_read    = seq_read,
	.proc_lseek   = seq_lseek,
	.proc_release = single_release,
};

static int cipher_gpu_state_proc_show(struct seq_file *m, void *v)
{
	struct cipher_gpu_state_kern snap;
	u64 now = get_jiffies_64();
	u64 age;

	spin_lock(&cipher_gpu_state.lock);
	snap = cipher_gpu_state;
	spin_unlock(&cipher_gpu_state.lock);

	if (snap.timestamp_jiffies == 0) {
		seq_puts(m, "cipher_gpu_state: no sample yet (cipher-gpustate daemon not running?)\n");
		return 0;
	}

	age = now - snap.timestamp_jiffies;
	seq_printf(m, "power=%u mW (%u.%03u W)\n",
		   snap.power_mw,
		   snap.power_mw / 1000U,
		   snap.power_mw % 1000U);
	seq_printf(m, "temp=%u C\n", snap.temp_c);
	seq_printf(m, "sm_clock=%u MHz, mem_clock=%u MHz\n",
		   snap.sm_clock_mhz, snap.mem_clock_mhz);
	seq_printf(m, "sm_util=%u%%, mem_util=%u%%\n",
		   snap.sm_util_pct, snap.mem_util_pct);
	seq_printf(m, "fb_used=%u MB\n", snap.fb_used_mb);
	seq_printf(m, "age=%llu.%02llu s since last update\n",
		   (unsigned long long)(age / HZ),
		   (unsigned long long)((age * 100ULL / HZ) % 100));
	return 0;
}

static int cipher_proc_gpu_state_open(struct inode *inode, struct file *file)
{
	return single_open(file, cipher_gpu_state_proc_show, NULL);
}

static const struct proc_ops cipher_proc_gpu_state_fops = {
	.proc_open    = cipher_proc_gpu_state_open,
	.proc_read    = seq_read,
	.proc_lseek   = seq_lseek,
	.proc_release = single_release,
};

/* CP 3.3 — /proc/cipher/flops. cipher_flops_proc_show lives in
 * cipher_flops.c (it owns the FLOP-series ring + per-tenant attribution). */
static int cipher_proc_flops_open(struct inode *inode, struct file *file)
{
	return single_open(file, cipher_flops_proc_show, NULL);
}

static const struct proc_ops cipher_proc_flops_fops = {
	.proc_open    = cipher_proc_flops_open,
	.proc_read    = seq_read,
	.proc_lseek   = seq_lseek,
	.proc_release = single_release,
};

/* Track 3 SC4 — /proc/cipher/migrations. cipher_cp54_migration_proc_show
 * lives in cipher_cp54_sched.c (it owns the migration ledger + counters). */
static int cipher_proc_migrations_open(struct inode *inode, struct file *file)
{
	return single_open(file, cipher_cp54_migration_proc_show, NULL);
}

static const struct proc_ops cipher_proc_migrations_fops = {
	.proc_open    = cipher_proc_migrations_open,
	.proc_read    = seq_read,
	.proc_lseek   = seq_lseek,
	.proc_release = single_release,
};

/* Track 2 SC5 — /proc/cipher/arenas. cipher_wa_proc_show lives in
 * cipher_weight_arena.c (it owns the weight-arena fd-custodian registry). */
static int cipher_proc_arenas_open(struct inode *inode, struct file *file)
{
	return single_open(file, cipher_wa_proc_show, NULL);
}

static const struct proc_ops cipher_proc_arenas_fops = {
	.proc_open    = cipher_proc_arenas_open,
	.proc_read    = seq_read,
	.proc_lseek   = seq_lseek,
	.proc_release = single_release,
};

/* Week 2 Step 5 — /proc/cipher/classify_stats. Counters in
 * cipher_main.c (struct cipher_classify_stats cipher_classify_stats).
 * Populated by Step 6 userspace-to-kmod ioctl bridge from
 * libcipher_rt.so's cipher_rt_classify_observer_snapshot. Stays zero
 * until that wiring lands; emitter prints zeros faithfully. */
static const char * const cipher_classify_op_names[16] = {
	"GEMM",             /*  0 */
	"ATTENTION",        /*  1 */
	"CONVOLUTION",      /*  2 */
	"ELEMENTWISE",      /*  3 */
	"REDUCTION",        /*  4 */
	"MEMCPY_TRANSPOSE", /*  5 */
	"ITERATIVE_CUSTOM", /*  6 */
	"reserved_7",       /*  7 */
	"reserved_8",       /*  8 */
	"reserved_9",       /*  9 */
	"reserved_10",      /* 10 */
	"reserved_11",      /* 11 */
	"reserved_12",      /* 12 */
	"reserved_13",      /* 13 */
	"reserved_14",      /* 14 */
	"UNCLASSIFIED",     /* 15 (matches observer convention) */
};

/* Week 3 Step 4 Option II-a — DSM PROPOSE ioctl handler.
 * Pushes a proposal into the bounded ring (drop-oldest-on-overflow
 * via atomic monotonic head modulo RING_SIZE). Observability-only;
 * no auto-action. */
long cipher_dsm_propose_ioctl(unsigned long arg)
{
	struct cipher_dsm_propose_push push;
	struct cipher_dsm_proposal_entry *e;
	u32 idx;

	if (copy_from_user(&push, (void __user *)arg, sizeof(push)))
		return -EFAULT;

	idx = (u32)atomic_fetch_add(1, &cipher_dsm_proposals.head)
	      % CIPHER_DSM_PROPOSAL_RING_SIZE;
	e = &cipher_dsm_proposals.buf[idx];

	/* Write fields. A concurrent /proc reader may see a torn entry
	 * in the worst case; acceptable for Pattern (a) observability. */
	e->timestamp_jiffies = get_jiffies_64();
	e->tenant_id  = push.tenant_id;
	e->source_slot = push.source_slot;
	e->target_slot = push.target_slot;
	e->reason     = push.reason;
	e->confidence = push.confidence;
	e->reserved   = 0;

	atomic64_inc(&cipher_dsm_proposals.total_received);
	return 0;
}

/* Week 2 Step 6 — ioctl handler for CIPHER_PUSH_CLASSIFY_STATS.
 * SET semantics: userspace observer snapshot becomes the kmod-side
 * counter values. Called from cipher_dev.c dispatch on nr=25. */
long cipher_push_classify_stats_ioctl(unsigned long arg)
{
	struct cipher_classify_stats_push push;
	int i;

	if (copy_from_user(&push, (void __user *)arg, sizeof(push)))
		return -EFAULT;

	atomic64_set(&cipher_classify_stats.total,       push.total);
	atomic64_set(&cipher_classify_stats.handled,     push.handled);
	atomic64_set(&cipher_classify_stats.passthrough, push.passthrough);
	for (i = 0; i < 16; i++) {
		atomic64_set(&cipher_classify_stats.per_op_class[i],
		             push.per_op_class[i]);
	}
	return 0;
}

static int cipher_classify_stats_proc_show(struct seq_file *m, void *v)
{
	u64 total       = atomic64_read(&cipher_classify_stats.total);
	u64 handled     = atomic64_read(&cipher_classify_stats.handled);
	u64 passthrough = atomic64_read(&cipher_classify_stats.passthrough);
	int i;

	seq_puts(m, "classify_stats:\n");
	seq_printf(m, "  total:        %llu\n", total);
	seq_printf(m, "  handled:      %llu\n", handled);
	seq_printf(m, "  passthrough:  %llu\n", passthrough);
	seq_puts(m, "  per_op_class:\n");
	for (i = 0; i < 16; i++) {
		u64 c = atomic64_read(&cipher_classify_stats.per_op_class[i]);
		if (c == 0)
			continue;
		seq_printf(m, "    %-18s %llu\n",
		           cipher_classify_op_names[i], c);
	}
	if (total == 0) {
		seq_puts(m, "  (no producer wired yet — Step 6 will populate via ioctl bridge)\n");
	}
	return 0;
}

/* Week 3 Step 4 Option II-a — /proc/cipher/dsm_proposals emitter.
 * Walks the bounded ring buffer backwards from the head, emits the
 * most-recent entries (capped at 64 for readability). Reader sees a
 * snapshot of ring state; concurrent writers may produce torn entries
 * in the worst case (acceptable for observability). */
static const char * const cipher_dsm_reason_names[] = {
	"UNKNOWN",                  /* 0 */
	"TRANSITION_DETECTED",      /* 1 */
	"AGENT_IDLE",               /* 2 */
};

static int cipher_dsm_proposals_proc_show(struct seq_file *m, void *v)
{
	u64 total       = (u64)atomic64_read(&cipher_dsm_proposals.total_received);
	u32 head        = (u32)atomic_read(&cipher_dsm_proposals.head);
	u32 emit_count  = total < 64 ? (u32)total : 64;
	u64 now_jiffies = get_jiffies_64();
	u32 i;

	seq_puts(m, "dsm_proposals:\n");
	seq_printf(m, "  total_received: %llu\n", total);
	seq_printf(m, "  ring_size:      %u\n", CIPHER_DSM_PROPOSAL_RING_SIZE);
	seq_printf(m, "  emitting:       %u (most recent)\n", emit_count);
	if (emit_count == 0) {
		seq_puts(m, "  (no proposals yet — userspace transition wrapper has not pushed)\n");
		return 0;
	}
	seq_puts(m, "  entries (newest first):\n");
	for (i = 0; i < emit_count; i++) {
		u32 ring_idx;
		struct cipher_dsm_proposal_entry e;
		u64 age_ms;
		const char *reason_name = "?";

		ring_idx = (head - 1 - i) % CIPHER_DSM_PROPOSAL_RING_SIZE;
		e = cipher_dsm_proposals.buf[ring_idx];
		age_ms = (now_jiffies - e.timestamp_jiffies) * 1000ULL / HZ;
		if (e.reason < ARRAY_SIZE(cipher_dsm_reason_names))
			reason_name = cipher_dsm_reason_names[e.reason];
		seq_printf(m,
		           "    tenant=%u  %u->%u  reason=%s  conf=%u  age=%llu ms\n",
		           e.tenant_id, e.source_slot, e.target_slot,
		           reason_name, e.confidence, age_ms);
	}
	return 0;
}

static int cipher_proc_dsm_proposals_open(struct inode *inode, struct file *file)
{
	return single_open(file, cipher_dsm_proposals_proc_show, NULL);
}

static const struct proc_ops cipher_proc_dsm_proposals_fops = {
	.proc_open    = cipher_proc_dsm_proposals_open,
	.proc_read    = seq_read,
	.proc_lseek   = seq_lseek,
	.proc_release = single_release,
};

static int cipher_proc_classify_stats_open(struct inode *inode, struct file *file)
{
	return single_open(file, cipher_classify_stats_proc_show, NULL);
}

static const struct proc_ops cipher_proc_classify_stats_fops = {
	.proc_open    = cipher_proc_classify_stats_open,
	.proc_read    = seq_read,
	.proc_lseek   = seq_lseek,
	.proc_release = single_release,
};

/* Week 4 Step 5 — /proc/cipher/sense_session. Counters in cipher_main.c
 * (struct cipher_sense_session_stats cipher_sense_session_stats).
 * Populated by Step 6+ userspace-to-kmod ioctl bridge from
 * libcipher_rt.so's cipher_rt_sense_transition flusher. Stays zero
 * until that wiring lands; emitter prints zeros faithfully.
 *
 * Class indices match the canonical CipherSessionType enum in
 * cipher-may13-evidence/include/cipher_sense.h:
 *   0 = UNKNOWN, 1 = HUMAN, 2 = AGENT, 3 = BATCH. */
static const char * const cipher_sense_class_names[4] = {
	"UNKNOWN", /* 0 */
	"HUMAN",   /* 1 */
	"AGENT",   /* 2 */
	"BATCH",   /* 3 */
};

static int cipher_sense_session_proc_show(struct seq_file *m, void *v)
{
	u64 total_sessions    = atomic64_read(&cipher_sense_session_stats.total_sessions);
	u64 total_transitions = atomic64_read(&cipher_sense_session_stats.total_transitions);
	int i;

	seq_puts(m, "sense_session:\n");
	seq_printf(m, "  total_sessions:    %llu\n", total_sessions);
	seq_printf(m, "  total_transitions: %llu\n", total_transitions);
	seq_puts(m, "  per_class:\n");
	for (i = 0; i < 4; i++) {
		u64 c = atomic64_read(&cipher_sense_session_stats.per_class[i]);
		seq_printf(m, "    %-8s %llu\n", cipher_sense_class_names[i], c);
	}
	if (total_sessions == 0 && total_transitions == 0) {
		seq_puts(m, "  (no producer wired yet — Step 6+ will populate via ioctl bridge)\n");
	}
	return 0;
}

static int cipher_proc_sense_session_open(struct inode *inode, struct file *file)
{
	return single_open(file, cipher_sense_session_proc_show, NULL);
}

static const struct proc_ops cipher_proc_sense_session_fops = {
	.proc_open    = cipher_proc_sense_session_open,
	.proc_read    = seq_read,
	.proc_lseek   = seq_lseek,
	.proc_release = single_release,
};

int cipher_proc_init(void)
{
	cipher_proc_dir = proc_mkdir(CIPHER_PROC_DIR, NULL);
	if (!cipher_proc_dir)
		return -ENOMEM;

	cipher_proc_stats_entry = proc_create(CIPHER_PROC_STATS, 0444,
					      cipher_proc_dir,
					      &cipher_proc_fops);
	if (!cipher_proc_stats_entry) {
		proc_remove(cipher_proc_dir);
		cipher_proc_dir = NULL;
		return -ENOMEM;
	}

	cipher_proc_bar0_entry = proc_create(CIPHER_PROC_BAR0, 0444,
					     cipher_proc_dir,
					     &cipher_proc_bar0_fops);
	if (!cipher_proc_bar0_entry) {
		proc_remove(cipher_proc_stats_entry);
		cipher_proc_stats_entry = NULL;
		proc_remove(cipher_proc_dir);
		cipher_proc_dir = NULL;
		return -ENOMEM;
	}

	cipher_proc_gpu_state_entry = proc_create(CIPHER_PROC_GPU_STATE, 0444,
						  cipher_proc_dir,
						  &cipher_proc_gpu_state_fops);
	if (!cipher_proc_gpu_state_entry) {
		proc_remove(cipher_proc_bar0_entry);
		cipher_proc_bar0_entry = NULL;
		proc_remove(cipher_proc_stats_entry);
		cipher_proc_stats_entry = NULL;
		proc_remove(cipher_proc_dir);
		cipher_proc_dir = NULL;
		return -ENOMEM;
	}

	cipher_proc_flops_entry = proc_create(CIPHER_PROC_FLOPS, 0444,
					      cipher_proc_dir,
					      &cipher_proc_flops_fops);
	if (!cipher_proc_flops_entry) {
		proc_remove(cipher_proc_gpu_state_entry);
		cipher_proc_gpu_state_entry = NULL;
		proc_remove(cipher_proc_bar0_entry);
		cipher_proc_bar0_entry = NULL;
		proc_remove(cipher_proc_stats_entry);
		cipher_proc_stats_entry = NULL;
		proc_remove(cipher_proc_dir);
		cipher_proc_dir = NULL;
		return -ENOMEM;
	}

	cipher_proc_migrations_entry = proc_create(CIPHER_PROC_MIGRATIONS, 0444,
						   cipher_proc_dir,
						   &cipher_proc_migrations_fops);
	if (!cipher_proc_migrations_entry) {
		proc_remove(cipher_proc_flops_entry);
		cipher_proc_flops_entry = NULL;
		proc_remove(cipher_proc_gpu_state_entry);
		cipher_proc_gpu_state_entry = NULL;
		proc_remove(cipher_proc_bar0_entry);
		cipher_proc_bar0_entry = NULL;
		proc_remove(cipher_proc_stats_entry);
		cipher_proc_stats_entry = NULL;
		proc_remove(cipher_proc_dir);
		cipher_proc_dir = NULL;
		return -ENOMEM;
	}

	cipher_proc_arenas_entry = proc_create(CIPHER_PROC_ARENAS, 0444,
					       cipher_proc_dir,
					       &cipher_proc_arenas_fops);
	if (!cipher_proc_arenas_entry) {
		proc_remove(cipher_proc_migrations_entry);
		cipher_proc_migrations_entry = NULL;
		proc_remove(cipher_proc_flops_entry);
		cipher_proc_flops_entry = NULL;
		proc_remove(cipher_proc_gpu_state_entry);
		cipher_proc_gpu_state_entry = NULL;
		proc_remove(cipher_proc_bar0_entry);
		cipher_proc_bar0_entry = NULL;
		proc_remove(cipher_proc_stats_entry);
		cipher_proc_stats_entry = NULL;
		proc_remove(cipher_proc_dir);
		cipher_proc_dir = NULL;
		return -ENOMEM;
	}

	cipher_proc_classify_stats_entry = proc_create(
		CIPHER_PROC_CLASSIFY_STATS, 0444,
		cipher_proc_dir,
		&cipher_proc_classify_stats_fops);
	if (!cipher_proc_classify_stats_entry) {
		proc_remove(cipher_proc_arenas_entry);
		cipher_proc_arenas_entry = NULL;
		proc_remove(cipher_proc_migrations_entry);
		cipher_proc_migrations_entry = NULL;
		proc_remove(cipher_proc_flops_entry);
		cipher_proc_flops_entry = NULL;
		proc_remove(cipher_proc_gpu_state_entry);
		cipher_proc_gpu_state_entry = NULL;
		proc_remove(cipher_proc_bar0_entry);
		cipher_proc_bar0_entry = NULL;
		proc_remove(cipher_proc_stats_entry);
		cipher_proc_stats_entry = NULL;
		proc_remove(cipher_proc_dir);
		cipher_proc_dir = NULL;
		return -ENOMEM;
	}

	cipher_proc_dsm_proposals_entry = proc_create(
		CIPHER_PROC_DSM_PROPOSALS, 0444,
		cipher_proc_dir,
		&cipher_proc_dsm_proposals_fops);
	if (!cipher_proc_dsm_proposals_entry) {
		proc_remove(cipher_proc_classify_stats_entry);
		cipher_proc_classify_stats_entry = NULL;
		proc_remove(cipher_proc_arenas_entry);
		cipher_proc_arenas_entry = NULL;
		proc_remove(cipher_proc_migrations_entry);
		cipher_proc_migrations_entry = NULL;
		proc_remove(cipher_proc_flops_entry);
		cipher_proc_flops_entry = NULL;
		proc_remove(cipher_proc_gpu_state_entry);
		cipher_proc_gpu_state_entry = NULL;
		proc_remove(cipher_proc_bar0_entry);
		cipher_proc_bar0_entry = NULL;
		proc_remove(cipher_proc_stats_entry);
		cipher_proc_stats_entry = NULL;
		proc_remove(cipher_proc_dir);
		cipher_proc_dir = NULL;
		return -ENOMEM;
	}

	cipher_proc_sense_session_entry = proc_create(
		CIPHER_PROC_SENSE_SESSION, 0444,
		cipher_proc_dir,
		&cipher_proc_sense_session_fops);
	if (!cipher_proc_sense_session_entry) {
		proc_remove(cipher_proc_dsm_proposals_entry);
		cipher_proc_dsm_proposals_entry = NULL;
		proc_remove(cipher_proc_classify_stats_entry);
		cipher_proc_classify_stats_entry = NULL;
		proc_remove(cipher_proc_arenas_entry);
		cipher_proc_arenas_entry = NULL;
		proc_remove(cipher_proc_migrations_entry);
		cipher_proc_migrations_entry = NULL;
		proc_remove(cipher_proc_flops_entry);
		cipher_proc_flops_entry = NULL;
		proc_remove(cipher_proc_gpu_state_entry);
		cipher_proc_gpu_state_entry = NULL;
		proc_remove(cipher_proc_bar0_entry);
		cipher_proc_bar0_entry = NULL;
		proc_remove(cipher_proc_stats_entry);
		cipher_proc_stats_entry = NULL;
		proc_remove(cipher_proc_dir);
		cipher_proc_dir = NULL;
		return -ENOMEM;
	}
	return 0;
}

void cipher_proc_exit(void)
{
	if (cipher_proc_sense_session_entry) {
		proc_remove(cipher_proc_sense_session_entry);
		cipher_proc_sense_session_entry = NULL;
	}
	if (cipher_proc_dsm_proposals_entry) {
		proc_remove(cipher_proc_dsm_proposals_entry);
		cipher_proc_dsm_proposals_entry = NULL;
	}
	if (cipher_proc_classify_stats_entry) {
		proc_remove(cipher_proc_classify_stats_entry);
		cipher_proc_classify_stats_entry = NULL;
	}
	if (cipher_proc_arenas_entry) {
		proc_remove(cipher_proc_arenas_entry);
		cipher_proc_arenas_entry = NULL;
	}
	if (cipher_proc_migrations_entry) {
		proc_remove(cipher_proc_migrations_entry);
		cipher_proc_migrations_entry = NULL;
	}
	if (cipher_proc_flops_entry) {
		proc_remove(cipher_proc_flops_entry);
		cipher_proc_flops_entry = NULL;
	}
	if (cipher_proc_gpu_state_entry) {
		proc_remove(cipher_proc_gpu_state_entry);
		cipher_proc_gpu_state_entry = NULL;
	}
	if (cipher_proc_bar0_entry) {
		proc_remove(cipher_proc_bar0_entry);
		cipher_proc_bar0_entry = NULL;
	}
	if (cipher_proc_stats_entry) {
		proc_remove(cipher_proc_stats_entry);
		cipher_proc_stats_entry = NULL;
	}
	if (cipher_proc_dir) {
		proc_remove(cipher_proc_dir);
		cipher_proc_dir = NULL;
	}
}
