/* SPDX-License-Identifier: GPL-2.0
 *
 * cipher_internal.h — types, constants, externs shared across cipher_kmod TUs.
 * Phase 1.5.1: NV_ESC_RM_* family added to the decoder.
 *
 * Slot model (Option Z): one sorted-by-nr table of named ioctls.
 * Per-PID counters use a flat atomic64_t array indexed by table slot.
 * OTHER and TOTAL live in two extra global slots beyond the table.
 */
#ifndef CIPHER_INTERNAL_H
#define CIPHER_INTERNAL_H

#include <linux/types.h>
#include <linux/atomic.h>
#include <linux/spinlock.h>
#include <linux/hashtable.h>
#include <linux/sched.h>

#include "cipher_ioctl.h"   /* CIPHER_TENANT_ID_LEN, ioctl ABI */

/* ---- ioctl decode table ----------------------------------------------- */

#define CIPHER_FAM_FRONTEND  0   /* original NV_ESC_*, nr 200..218 */
#define CIPHER_FAM_RM        1   /* NV_ESC_RM_*,        nr 0x27..0x5e */

struct cipher_nv_ioctl_def {
	unsigned int  nr;
	const char   *name;
	u8            family;
};

#define CIPHER_NV_IOCTL_COUNT       24
#define CIPHER_NV_IOCTL_OTHER_SLOT  24
#define CIPHER_NV_IOCTL_TOTAL_SLOT  25
#define CIPHER_GLOBAL_SLOTS         26

/* Slot indices. MUST match designated initialiser positions in
 * cipher_nv_ioctls[] (cipher_ioctl_decode.c). cipher_decode_self_check()
 * verifies the binding at module init.
 */
enum {
	CIPHER_SLOT_RM_ALLOC_MEMORY = 0,
	CIPHER_SLOT_RM_FREE,
	CIPHER_SLOT_RM_CONTROL,
	CIPHER_SLOT_RM_ALLOC,
	CIPHER_SLOT_RM_DUP_OBJECT,
	CIPHER_SLOT_RM_SHARE,
	CIPHER_SLOT_RM_VID_HEAP_CONTROL,
	CIPHER_SLOT_RM_MAP_MEMORY,
	CIPHER_SLOT_RM_UNMAP_MEMORY,
	CIPHER_SLOT_RM_UPDATE_DEVICE_MAPPING_INFO,
	CIPHER_SLOT_CARD_INFO,
	CIPHER_SLOT_REGISTER_FD,
	CIPHER_SLOT_ALLOC_OS_EVENT,
	CIPHER_SLOT_FREE_OS_EVENT,
	CIPHER_SLOT_STATUS_CODE,
	CIPHER_SLOT_CHECK_VERSION_STR,
	CIPHER_SLOT_IOCTL_XFER_CMD,
	CIPHER_SLOT_ATTACH_GPUS_TO_FD,
	CIPHER_SLOT_QUERY_DEVICE_INTR,
	CIPHER_SLOT_SYS_PARAMS,
	CIPHER_SLOT_NUMA_INFO,
	CIPHER_SLOT_SET_NUMA_STATUS,
	CIPHER_SLOT_EXPORT_TO_DMABUF_FD,
	CIPHER_SLOT_WAIT_OPEN_COMPLETE,
};

extern const struct cipher_nv_ioctl_def
	cipher_nv_ioctls[CIPHER_NV_IOCTL_COUNT];

/* ---- per-PID hashtable ------------------------------------------------ */

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
	char              tenant_id[CIPHER_TENANT_ID_LEN];  /* Phase 2: "" until REGISTER_TENANT */
	atomic64_t        per_slot[CIPHER_NV_IOCTL_COUNT];   /* 192 B */
	atomic64_t        other_count;
	atomic64_t        total;
	atomic64_t        errors;
	u64               first_seen_jiffies;
	u64               last_seen_jiffies;
};

/* Storage in cipher_probe.c (table) and cipher_main.c (the rest). */
extern struct hlist_head cipher_pid_table[CIPHER_PID_HASH_SIZE];
extern spinlock_t        cipher_pid_insert_lock;
extern atomic64_t        cipher_global_cmd_counts[CIPHER_GLOBAL_SLOTS];
extern atomic64_t        cipher_global_cmd_errors[CIPHER_GLOBAL_SLOTS];
extern atomic64_t        cipher_alloc_failures;
extern atomic64_t        cipher_reaped_count;
extern u64               cipher_load_jiffies;
extern unsigned long     cipher_hooked_addr;

/* cipher_ioctl_decode.c */
int cipher_decode_nv_ioctl_slot(unsigned int cmd);  /* slot 0..23 or -1 */
int cipher_decode_self_check(void);                  /* 0 or -EINVAL */

/* cipher_probe.c */
int  cipher_probe_init(void);
void cipher_probe_exit(void);
/* exposed for cipher_dev.c (REGISTER_TENANT handler needs to upsert an entry) */
struct cipher_pid_stats *cipher_pid_get_or_create(pid_t pid, pid_t tgid);

/* cipher_proc.c */
int  cipher_proc_init(void);
void cipher_proc_exit(void);

/* cipher_dev.c */
int  cipher_dev_init(void);
void cipher_dev_exit(void);

#endif /* CIPHER_INTERNAL_H */
