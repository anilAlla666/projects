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

#include "cipher_internal.h"

#define CIPHER_PROC_TOPN  16

static struct proc_dir_entry *cipher_proc_dir;
static struct proc_dir_entry *cipher_proc_stats_entry;

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

	seq_printf(m, "cipher_kmod 0.2.0  uptime=%llu jiffies (%llu.%02llu s)\n",
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
		rows[n].first_seen_jiffies = e->first_seen_jiffies;
		n++;
	}
	rcu_read_unlock();

	sort(rows, n, sizeof(*rows), cipher_topn_cmp, NULL);

	seq_printf(m, "\nPer-PID summary (top %d by total, %d PIDs total):\n",
		   CIPHER_PROC_TOPN, n);
	seq_puts(m,
		 "  PID      TGID     COMM             TENANT          TOTAL    RM_CTRL  RM_ALLOC  RM_FREE  MAP_MEM  OTHER    ERR    AGE_S\n");

	for (i = 0; i < n && i < CIPHER_PROC_TOPN; i++) {
		u64 age = now - rows[i].first_seen_jiffies;
		const char *tnt = rows[i].tenant_id[0] ? rows[i].tenant_id : "-";

		seq_printf(m,
			   "  %-7u  %-7u  %-16s %-15.15s %-8lld %-8lld %-9lld %-8lld %-8lld %-8lld %-6lld %llu.%02llu\n",
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
			 "  TGID     COMM             TENANT          THR  TOTAL    RM_CTRL  RM_ALLOC  RM_FREE  MAP_MEM  OTHER    ERR\n");

		for (i = 0; i < t && i < CIPHER_PROC_TOPN; i++) {
			const char *tnt = trows[i].tenant_id[0] ? trows[i].tenant_id : "-";

			seq_printf(m,
				   "  %-7u  %-16s %-15.15s %-3d  %-8lld %-8lld %-9lld %-8lld %-8lld %-8lld %-6lld\n",
				   trows[i].tgid, trows[i].comm, tnt,
				   trows[i].thread_count,
				   (long long)trows[i].total,
				   (long long)trows[i].rm_ctrl,
				   (long long)trows[i].rm_alloc,
				   (long long)trows[i].rm_free,
				   (long long)trows[i].map_mem,
				   (long long)trows[i].other,
				   (long long)trows[i].errors);
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
