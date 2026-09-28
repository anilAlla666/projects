// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_proc.c — /proc/cipher/stats seq_file emitter.
 *
 * Reader path:
 *   1. count entries under RCU
 *   2. kmalloc snapshot array (GFP_KERNEL — process context)
 *   3. fill snapshot under RCU
 *   4. sort by total
 *   5. emit globals + top-N PIDs
 *
 * Snapshot avoids holding RCU lock across seq_printf (which can sleep on user
 * copy). Race with concurrent inserts is benign: snapshot may miss PIDs added
 * after the count walk, but we cap iteration at the counted size.
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

#include "cipher_internal.h"

#define CIPHER_PROC_TOPN  16

static struct proc_dir_entry *cipher_proc_dir;
static struct proc_dir_entry *cipher_proc_stats_entry;

struct cipher_topn_row {
	pid_t pid;
	pid_t tgid;
	char  comm[TASK_COMM_LEN];
	s64   total;
	s64   xfer_cmd;
	s64   card_info;
	s64   other;
	s64   errors;
	u64   first_seen_jiffies;
};

static int cipher_topn_cmp(const void *a, const void *b)
{
	const struct cipher_topn_row *ra = a;
	const struct cipher_topn_row *rb = b;

	if (rb->total > ra->total) return  1;
	if (rb->total < ra->total) return -1;
	return 0;
}

static int cipher_proc_show(struct seq_file *m, void *v)
{
	u64 now = get_jiffies_64();
	u64 uptime_jiffies = now - cipher_load_jiffies;
	int i, bkt, npids = 0, n;
	struct cipher_pid_stats *e;
	struct cipher_topn_row *rows;
	s64 total_observed, total_errors, oc, oe;

	seq_printf(m, "cipher_kmod 0.1.0  uptime=%llu jiffies (%llu.%02llu s)\n",
		   (unsigned long long)uptime_jiffies,
		   (unsigned long long)(uptime_jiffies / HZ),
		   (unsigned long long)((uptime_jiffies * 100ULL / HZ) % 100));
	seq_printf(m, "nvidia_unlocked_ioctl @ 0x%lx [hooked]\n",
		   cipher_hooked_addr);

	total_observed =
		atomic64_read(&cipher_global_cmd_counts[CIPHER_NV_ESC_TOTAL_SLOT]);
	total_errors =
		atomic64_read(&cipher_global_cmd_errors[CIPHER_NV_ESC_TOTAL_SLOT]);
	seq_printf(m,
		   "total_observed=%lld  total_errors=%lld  alloc_failures=%lld\n\n",
		   (long long)total_observed,
		   (long long)total_errors,
		   (long long)atomic64_read(&cipher_alloc_failures));

	seq_puts(m, "Global ioctl counts by NV_ESC nr:\n");
	seq_puts(m, "  nr  name                              count        errors\n");
	seq_puts(m, "  --  --------------------------------  -----------  -----------\n");
	for (i = 0; i < CIPHER_NV_ESC_COUNT; i++) {
		s64 c  = atomic64_read(&cipher_global_cmd_counts[i]);
		s64 ec = atomic64_read(&cipher_global_cmd_errors[i]);

		if (c == 0 && ec == 0)
			continue;
		seq_printf(m, "  %2d  %-32s  %11lld  %11lld%s\n",
			   i, cipher_decode_nv_esc_name(i),
			   (long long)c, (long long)ec,
			   (i == 11) ? "  *dominant*" : "");
	}
	oc = atomic64_read(&cipher_global_cmd_counts[CIPHER_NV_ESC_OTHER_SLOT]);
	oe = atomic64_read(&cipher_global_cmd_errors[CIPHER_NV_ESC_OTHER_SLOT]);
	seq_printf(m, "      %-32s  %11lld  %11lld\n",
		   "OTHER (cmd outside NV_ESC_*)",
		   (long long)oc, (long long)oe);

	/* Pass 1: count under RCU */
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

	/* Pass 2: snapshot under RCU. Cap at npids in case more entries
	 * appeared between passes. */
	n = 0;
	rcu_read_lock();
	hash_for_each_rcu(cipher_pid_table, bkt, e, node) {
		if (n >= npids)
			break;
		rows[n].pid  = e->pid;
		rows[n].tgid = e->tgid;
		memcpy(rows[n].comm, e->comm, TASK_COMM_LEN);
		rows[n].comm[TASK_COMM_LEN - 1] = '\0';
		rows[n].total      = atomic64_read(&e->total);
		rows[n].xfer_cmd   = atomic64_read(&e->per_cmd[11]);
		rows[n].card_info  = atomic64_read(&e->per_cmd[0]);
		rows[n].other      = atomic64_read(&e->other_count);
		rows[n].errors     = atomic64_read(&e->errors);
		rows[n].first_seen_jiffies = e->first_seen_jiffies;
		n++;
	}
	rcu_read_unlock();

	sort(rows, n, sizeof(*rows), cipher_topn_cmp, NULL);

	seq_printf(m, "\nPer-PID summary (top %d by total, %d PIDs total):\n",
		   CIPHER_PROC_TOPN, n);
	seq_puts(m,
		 "  PID      TGID     COMM             TOTAL        XFER_CMD     CARD_INFO    OTHER     ERR     AGE_S\n");

	for (i = 0; i < n && i < CIPHER_PROC_TOPN; i++) {
		u64 age = now - rows[i].first_seen_jiffies;

		seq_printf(m,
			   "  %-7u  %-7u  %-16s %-12lld %-12lld %-12lld %-9lld %-7lld %llu.%02llu\n",
			   rows[i].pid, rows[i].tgid, rows[i].comm,
			   (long long)rows[i].total,
			   (long long)rows[i].xfer_cmd,
			   (long long)rows[i].card_info,
			   (long long)rows[i].other,
			   (long long)rows[i].errors,
			   (unsigned long long)(age / HZ),
			   (unsigned long long)((age * 100ULL / HZ) % 100));
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
	return 0;
}

void cipher_proc_exit(void)
{
	if (cipher_proc_stats_entry) {
		proc_remove(cipher_proc_stats_entry);
		cipher_proc_stats_entry = NULL;
	}
	if (cipher_proc_dir) {
		proc_remove(cipher_proc_dir);
		cipher_proc_dir = NULL;
	}
}
