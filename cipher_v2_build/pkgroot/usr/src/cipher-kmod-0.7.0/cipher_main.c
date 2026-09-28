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
#include "cipher_stream_registry.h"
#include "cipher_fairness_ledger.h"

spinlock_t                   cipher_pid_insert_lock;
atomic64_t                   cipher_global_cmd_counts[CIPHER_GLOBAL_SLOTS];
atomic64_t                   cipher_global_cmd_errors[CIPHER_GLOBAL_SLOTS];
atomic64_t                   cipher_alloc_failures;
atomic64_t                   cipher_reaped_count;
u64                          cipher_load_jiffies;
unsigned long                cipher_hooked_addr;
struct cipher_gpu_state_kern cipher_gpu_state;

/* Week 2 Step 5 — classify-stats counters (BSS zero-init). Populated by
 * Step 6 userspace-to-kmod ioctl bridge; stays zero until wiring lands. */
struct cipher_classify_stats cipher_classify_stats;

/* Week 3 Step 4 Option II-a — DSM proposal ring (BSS zero-init).
 * Populated by CIPHER_DSM_PROPOSE ioctl pushes from libcipher_rt.so's
 * cipher_rt_sense_transition_flush(); /proc/cipher/dsm_proposals
 * emits recent entries. */
struct cipher_dsm_proposal_ring cipher_dsm_proposals;

/* Week 4 Step 5 — sense-session classification counters (BSS zero-init).
 * Populated by a future Step 6+ userspace-to-kmod ioctl bridge; stays
 * zero until wiring lands. /proc/cipher/sense_session emits faithfully. */
struct cipher_sense_session_stats cipher_sense_session_stats;

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

	pr_info("cipher_kmod: loading (Phase 4 — CP 5.4 8-SM-group arbitration; W4 Step 4 retired LP-8)\n");

	cipher_cp54_sched_init();   /* CP 5.4 8-SM-group arbitration ledger */
	cipher_wa_init();           /* Track 2 SC5 weight-arena fd custodian */
	cipher_model_registry_init(); /* W7-9 Step 1 G10 model-identity registry */
	rc = cipher_audit_chain_init(); /* W7-9 Step 3 G6 kmod-resident AUDIT chain */
	if (rc) {
		pr_err("cipher_kmod: audit_chain_init failed: %d\n", rc);
		return rc;
	}
	rc = cipher_stream_registry_init(); /* W7-9 Step 5: multi-tenant resolver view slots */
	if (rc) {
		pr_err("cipher_kmod: stream_registry_init failed: %d\n", rc);
		cipher_audit_chain_exit();
		return rc;
	}
	cipher_cohort_registry_init(); /* W.6 sub-C: GPU co-residence registry; cannot fail */
	cipher_fairness_ledger_init(); /* D.8 FAIRNESS+SHIELD ledger; degrades to disabled on -ENOMEM */
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
	cipher_probe_exit();   /* stops hot path, drains RCU, frees per-PID table */
	/* CP 5.4: tear down the arbitration ledger only AFTER cipher_probe_exit
	 * has unregistered the do_exit kprobe — otherwise a process exiting
	 * mid-rmmod could fire cipher_cp54_release() against torn-down state. */
	cipher_cp54_sched_exit();
	cipher_cohort_registry_exit();/* W.6 sub-C: free co-residence registry entries */
	cipher_fairness_ledger_exit();/* D.8 FAIRNESS+SHIELD: free vmalloc'd ledger region */
	cipher_stream_registry_exit();/* W7-9 Step 5: free vmalloc'd view-slot table */
	cipher_audit_chain_exit();    /* W7-9 Step 3 G6: free vmalloc'd ring buffer */
	cipher_model_registry_exit(); /* W7-9 Step 1 G10: free model_registry entries before wa */
	cipher_wa_exit();           /* Track 2 SC5: cancel reaper, fput held fds */
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
MODULE_VERSION("0.7.0"); /* D.8 FAIRNESS+SHIELD: CIPHER_FAIRNESS_REGISTER (NR 32) + cross-tenant work-ledger (cipher_fairness_ledger.c, vmalloc_user region mapped RW at CIPHER_FAIR_VIEW_MMAP_PGOFF 0x200000) replacing the per-container /dev/shm fairness table. Additive — existing NRs 1-31 unchanged, reserved NRs (2/3/4) still -ENOSYS. Region is inert at init; timing-only enforcement lives in libcipher_rt, default-OFF (worst case == 0.6.6 no-enforcement). 0.7.0 supersedes 0.6.6. PRIOR: W.6 sub-C: GPU co-residence registry. CIPHER_COHORT_REGISTER (NR 30) + CIPHER_COHORT_QUERY (NR 31) — per-tgid table {tgid, model_fingerprint, last_seen_ns}; QUERY heartbeat-or-inserts the caller (current->tgid, caller_fingerprint), prunes peers stale > 30 s, returns the live snapshot (incl. self). libcipher_rt registers cipher_workload_model_fingerprint() (W.6 sub-B) and sets multi_tenant_detected = (n_resident >= 2) authoritatively. W.4 POOL consumes (co-resident pids + per-pid fingerprint) for legal cross-tenant GEMM-coalescing groups. Additive — existing NRs 1, 27, 28, 29 unchanged; single-GPU-per-host assumption (v1.x: multi-GPU keying). 0.6.6 supersedes 0.6.5. */
