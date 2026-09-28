// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_kmod main: module init/exit, cross-TU global storage.
 *
 * Phase 1 — observe-only kprobe + kretprobe on nvidia_unlocked_ioctl.
 * No transformation. No interference with running CUDA workloads.
 */
#include <linux/module.h>
#include <linux/init.h>
#include <linux/jiffies.h>
#include <linux/spinlock.h>
#include <linux/atomic.h>

#include "cipher_internal.h"

spinlock_t    cipher_pid_insert_lock;
atomic64_t    cipher_global_cmd_counts[CIPHER_GLOBAL_SLOTS];
atomic64_t    cipher_global_cmd_errors[CIPHER_GLOBAL_SLOTS];
atomic64_t    cipher_alloc_failures;
atomic64_t    cipher_reaped_count;
u64           cipher_load_jiffies;
unsigned long cipher_hooked_addr;

static int __init cipher_init(void)
{
	int rc;

	rc = cipher_decode_self_check();
	if (rc) {
		pr_err("cipher_kmod: decoder self-check failed: %d\n", rc);
		return rc;
	}

	spin_lock_init(&cipher_pid_insert_lock);
	cipher_load_jiffies = get_jiffies_64();

	pr_info("cipher_kmod: loading (Phase 2 — tenant identity flow via /dev/cipher)\n");

	rc = cipher_dev_init();
	if (rc) {
		pr_err("cipher_kmod: dev_init failed: %d\n", rc);
		return rc;
	}

	rc = cipher_proc_init();
	if (rc) {
		pr_err("cipher_kmod: proc_init failed: %d\n", rc);
		cipher_dev_exit();
		return rc;
	}

	rc = cipher_probe_init();
	if (rc) {
		pr_err("cipher_kmod: probe_init failed: %d\n", rc);
		cipher_proc_exit();
		cipher_dev_exit();
		return rc;
	}

	pr_info("cipher_kmod: loaded ok; nvidia_unlocked_ioctl @ 0x%lx hooked\n",
		cipher_hooked_addr);
	return 0;
}

static void __exit cipher_exit(void)
{
	pr_info("cipher_kmod: unloading\n");
	cipher_probe_exit();   /* stops hot path, drains RCU, frees per-PID table */
	cipher_proc_exit();
	cipher_dev_exit();
	pr_info("cipher_kmod: unloaded cleanly\n");
}

module_init(cipher_init);
module_exit(cipher_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("CIPHER");
MODULE_DESCRIPTION("CIPHER kmod — Phase 2: kernel-side observation + tenant identity via /dev/cipher");
MODULE_VERSION("0.2.0");
