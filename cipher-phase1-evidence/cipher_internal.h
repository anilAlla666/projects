/* SPDX-License-Identifier: GPL-2.0
 *
 * cipher_internal.h — types, constants, externs shared across cipher_kmod TUs.
 * Phase 1: observation only. No transformation, no MMIO, no PMU.
 */
#ifndef CIPHER_INTERNAL_H
#define CIPHER_INTERNAL_H

#include <linux/types.h>
#include <linux/atomic.h>
#include <linux/spinlock.h>
#include <linux/hashtable.h>
#include <linux/sched.h>

/* NV_ESC nr layout. See cipher_ioctl_decode.c for the name table.
 * Slots 0..18 = known NV_ESC nrs; 19 = OTHER (cmd outside NV_ESC); 20 = TOTAL.
 */
#define CIPHER_NV_ESC_COUNT       19
#define CIPHER_NV_ESC_OTHER_SLOT  19
#define CIPHER_NV_ESC_TOTAL_SLOT  20
#define CIPHER_GLOBAL_SLOTS       21

/* Per-PID hashtable: 1024 buckets. Key = task->pid (LWP, per-thread). */
#define CIPHER_PID_HASH_BITS  10
#define CIPHER_PID_HASH_SIZE  (1 << CIPHER_PID_HASH_BITS)

#define CIPHER_PROC_DIR    "cipher"
#define CIPHER_PROC_STATS  "stats"
#define CIPHER_DEV_NAME    "cipher"

struct cipher_pid_stats {
	struct hlist_node node;
	struct rcu_head   rcu;
	pid_t             pid;
	pid_t             tgid;
	char              comm[TASK_COMM_LEN];   /* refreshed every observation */
	atomic64_t        per_cmd[CIPHER_NV_ESC_COUNT];
	atomic64_t        other_count;
	atomic64_t        total;
	atomic64_t        errors;
	u64               first_seen_jiffies;
	u64               last_seen_jiffies;
};

/* Storage lives in cipher_probe.c (table) and cipher_main.c (the rest). */
extern struct hlist_head cipher_pid_table[CIPHER_PID_HASH_SIZE];
extern spinlock_t        cipher_pid_insert_lock;
extern atomic64_t        cipher_global_cmd_counts[CIPHER_GLOBAL_SLOTS];
extern atomic64_t        cipher_global_cmd_errors[CIPHER_GLOBAL_SLOTS];
extern atomic64_t        cipher_alloc_failures;
extern u64               cipher_load_jiffies;
extern unsigned long     cipher_hooked_addr;

/* cipher_ioctl_decode.c */
int         cipher_decode_nv_esc_nr(unsigned int cmd);    /* 0..18 or -1 */
const char *cipher_decode_nv_esc_name(int nr);            /* never NULL */

/* cipher_probe.c */
int  cipher_probe_init(void);
void cipher_probe_exit(void);

/* cipher_proc.c */
int  cipher_proc_init(void);
void cipher_proc_exit(void);

/* cipher_dev.c */
int  cipher_dev_init(void);
void cipher_dev_exit(void);

#endif /* CIPHER_INTERNAL_H */
