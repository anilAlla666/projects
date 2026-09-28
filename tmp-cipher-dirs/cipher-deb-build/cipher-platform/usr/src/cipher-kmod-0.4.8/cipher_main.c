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

spinlock_t                   cipher_pid_insert_lock;
atomic64_t                   cipher_global_cmd_counts[CIPHER_GLOBAL_SLOTS];
atomic64_t                   cipher_global_cmd_errors[CIPHER_GLOBAL_SLOTS];
atomic64_t                   cipher_alloc_failures;
atomic64_t                   cipher_reaped_count;
u64                          cipher_load_jiffies;
unsigned long                cipher_hooked_addr;
struct cipher_gpu_state_kern cipher_gpu_state;

static int __init cipher_init(void)
{
	int rc;

	rc = cipher_decode_self_check();
	if (rc) {
		pr_err("cipher_kmod: decoder self-check failed: %d\n", rc);
		return rc;
	}

	spin_lock_init(&cipher_pid_insert_lock);
	spin_lock_init(&cipher_gpu_state.lock);
	cipher_load_jiffies = get_jiffies_64();

	pr_info("cipher_kmod: loading (Phase 4.2 T4.2.1 — lock-free SM partition allocator)\n");

	cipher_partition_allocator_init();
	cipher_flops_init();   /* CP 3.3 FLOP-series ring; cannot fail */

	rc = cipher_dev_init();
	if (rc) {
		pr_err("cipher_kmod: dev_init failed: %d\n", rc);
		return rc;
	}

	rc = cipher_kvdedup_init();
	if (rc) {
		pr_err("cipher_kmod: kvdedup_init failed: %d\n", rc);
		goto out_dev;
	}

	rc = cipher_bar0_init();
	if (rc < 0 && rc != -ENODEV) {
		pr_err("cipher_kmod: BAR0 init failed: %d\n", rc);
		goto out_kvdedup;
	}
	/* -ENODEV is acceptable: no NVIDIA GPU present.
	 * Module still loads with BAR0 layer disabled. */

	rc = cipher_proc_init();
	if (rc) {
		pr_err("cipher_kmod: proc_init failed: %d\n", rc);
		goto out_bar0;
	}

	rc = cipher_probe_init();
	if (rc) {
		pr_err("cipher_kmod: probe_init failed: %d\n", rc);
		goto out_proc;
	}

	/* Phase 4.1 — derived-state kthread. Best-effort: kthread failure
	 * does not block module load (derived fields stay zero; readers
	 * tolerate). */
	rc = cipher_state_updater_init();
	if (rc) {
		pr_warn("cipher_kmod: cipher_state_updater_init failed: %d; "
		        "derived snapshot fields will remain zero\n", rc);
		/* not fatal */
	}

	pr_info("cipher_kmod: loaded ok; nvidia_unlocked_ioctl @ 0x%lx hooked\n",
		cipher_hooked_addr);
	return 0;

out_proc:
	cipher_proc_exit();
out_bar0:
	cipher_bar0_exit();
out_kvdedup:
	cipher_kvdedup_exit();
out_dev:
	cipher_dev_exit();
	return rc;
}

static void __exit cipher_exit(void)
{
	pr_info("cipher_kmod: unloading\n");
	cipher_state_updater_exit();   /* stops kthread before probe_exit drains */
	cipher_flops_exit();
	cipher_partition_allocator_exit();
	cipher_probe_exit();   /* stops hot path, drains RCU, frees per-PID table */
	cipher_proc_exit();
	cipher_bar0_exit();
	cipher_kvdedup_exit();
	cipher_dev_exit();
	pr_info("cipher_kmod: unloaded cleanly\n");
}

module_init(cipher_init);
module_exit(cipher_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("CIPHER");
MODULE_DESCRIPTION("CIPHER kmod — GPU-state spine + BAR0 reads + CP3.3 FLOP telemetry (CUPTI-fed, per-tenant MFU)");
MODULE_VERSION("0.4.8");
