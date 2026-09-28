// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_state_updater.c — Phase 4 derived-state computation kthread.
 *
 * STAGED: this file is NOT in Kbuild's cipher_kmod-y for cipher_kmod 0.3.1.
 * Gets wired into the live build in T4.1.6 when the module bumps to 0.4.0.
 *
 * Runs at 1 kHz. For each entry in cipher_pid_table, computes derived
 * fields from:
 *   - cipher_gpu_state (Phase 3 device-wide telemetry)
 *   - cipher_pid_stats (Phase 3 per-PID telemetry)
 *   - Phase 4 actuator write-back paths (e.g., sm_partition_mask)
 *
 * Writes are WRITE_ONCE per field; no struct-wide locking. Cross-field
 * invariants (e.g., partition_count == popcount(partition_mask)) are
 * weakly enforced — readers tolerate brief inconsistency. Freshness via
 * snapshot_jiffies field.
 *
 * Skipped silently if cipher_gpu_state.timestamp_jiffies is stale
 * (> 1 s old). Prevents derived fields from drifting on stale data.
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/delay.h>
#include <linux/sched.h>
#include <linux/spinlock.h>
#include <linux/jiffies.h>
#include <linux/hashtable.h>
#include <linux/rcupdate.h>

#include "cipher_internal.h"

#define CIPHER_STATE_UPDATER_HZ          1000
#define CIPHER_GPU_STATE_STALE_NS        (1ULL * NSEC_PER_SEC)
#define CIPHER_SUSTAINED_CLOCK_EMA_NUM   95
#define CIPHER_SUSTAINED_CLOCK_EMA_DEN   100

/* H100 SXM5 envelope constants (from BAR0_PMC observation + nvidia-smi). */
#define CIPHER_H100_THERMAL_LIMIT_C      90
#define CIPHER_H100_POWER_TDP_W          700

static struct task_struct *cipher_state_updater_task;
static atomic_t            cipher_state_updater_running;

/* Helper: convert jiffies-delta to nanoseconds. */
static inline u64 cipher_jiffies_to_ns(u64 delta_jiffies)
{
	return jiffies_to_nsecs((unsigned long)delta_jiffies);
}

/* Compute thermal headroom percentage from device temperature.
 * Returns 0 if temp >= limit, 100 if temp <= 0. */
static u32 cipher_compute_thermal_headroom(u32 temp_c)
{
	if (temp_c >= CIPHER_H100_THERMAL_LIMIT_C)
		return 0;
	return (CIPHER_H100_THERMAL_LIMIT_C - temp_c) * 100
	        / CIPHER_H100_THERMAL_LIMIT_C;
}

/* Compute power headroom in watts from device power_mw. */
static u32 cipher_compute_power_headroom(u32 power_mw)
{
	u32 power_w = power_mw / 1000;
	if (power_w >= CIPHER_H100_POWER_TDP_W)
		return 0;
	return CIPHER_H100_POWER_TDP_W - power_w;
}

/* EMA of sm_clock. Reads prior sustained_clock_mhz, blends with current. */
static u32 cipher_compute_sustained_clock(u32 prior, u32 current_mhz)
{
	if (prior == 0)
		return current_mhz;
	/* prior * 0.95 + current * 0.05 = (prior * 95 + current * 5) / 100 */
	return (prior * CIPHER_SUSTAINED_CLOCK_EMA_NUM
	      + current_mhz * (CIPHER_SUSTAINED_CLOCK_EMA_DEN
	                       - CIPHER_SUSTAINED_CLOCK_EMA_NUM))
	       / CIPHER_SUSTAINED_CLOCK_EMA_DEN;
}

/* Voltage envelope estimate (mv) from clock + power.
 * Coarse linear model: v = base + clock_term + power_term.
 * Refined in T4.3.x with calibration data. */
static u32 cipher_compute_voltage_envelope(u32 sustained_clock_mhz, u32 power_mw)
{
	/* Base 700 mV, +0.2 mV/MHz, +0.5 mV/W. Within H100 envelope ~700-1100 mV. */
	u32 v = 700 + (sustained_clock_mhz * 2) / 10 + (power_mw / 1000) / 2;
	if (v > 1100) v = 1100;
	return v;
}

/* Update derived fields for a single cipher_pid_stats entry.
 * Reader must hold rcu_read_lock(). Writer side uses WRITE_ONCE. */
static void cipher_update_tenant_derived(struct cipher_pid_stats *e,
                                         const struct cipher_gpu_state_kern *gs_snap)
{
	u32 prior_clock, new_clock;
	u32 thermal, power_hr, vmv;

	if (!e)
		return;

	/* Thermal / power headroom (device-wide, but stored per tenant for
	 * actuator convenience). */
	thermal = cipher_compute_thermal_headroom(gs_snap->temp_c);
	power_hr = cipher_compute_power_headroom(gs_snap->power_mw);

	WRITE_ONCE(e->thermal_headroom_pct, thermal);
	WRITE_ONCE(e->power_headroom_w, power_hr);

	/* Sustained clock EMA. Read prior, blend, write back. */
	prior_clock = READ_ONCE(e->sustained_clock_mhz);
	new_clock = cipher_compute_sustained_clock(prior_clock,
	                                            gs_snap->sm_clock_mhz);
	WRITE_ONCE(e->sustained_clock_mhz, new_clock);

	/* Voltage envelope from clock + power. */
	vmv = cipher_compute_voltage_envelope(new_clock, gs_snap->power_mw);
	WRITE_ONCE(e->voltage_envelope_mv, vmv);

	/* L2 / KV / weight / fusion / agentic clusters: populated by their
	 * respective actuators in P4.4 / P4.5 / P4.6 / P4.7. The kthread
	 * reads gpu_state for the device-wide fields and leaves the
	 * actuator-owned fields alone. sm_partition_mask is written by
	 * P4.2 allocator, not here.
	 *
	 * fairness_quota_remaining_pct comes from FAIRNESS table (P4.7);
	 * the kthread is the right place to compute it once FAIRNESS is
	 * fused. For now: leave at 100 (default to "full quota"). */
	if (READ_ONCE(e->fairness_quota_remaining_pct) == 0)
		WRITE_ONCE(e->fairness_quota_remaining_pct, 100);

	/* Update freshness. */
	WRITE_ONCE(e->snapshot_jiffies, get_jiffies_64());
}

static int cipher_state_updater_fn(void *unused)
{
	int bkt;
	struct cipher_pid_stats *e;
	struct cipher_gpu_state_kern gs_snap;
	u64 now, gs_ts;

	(void)unused;

	pr_info("cipher_state_updater: kthread starting (cadence %d Hz)\n",
		CIPHER_STATE_UPDATER_HZ);

	while (!kthread_should_stop()) {
		/* Snapshot gpu_state under its spinlock — brief copy out. */
		spin_lock(&cipher_gpu_state.lock);
		gs_snap = cipher_gpu_state;
		spin_unlock(&cipher_gpu_state.lock);

		gs_ts = gs_snap.timestamp_jiffies;
		now = get_jiffies_64();

		/* Skip if gpu_state stale > 1 s. Don't propagate stale data
		 * into derived fields. */
		if (gs_ts == 0
		    || cipher_jiffies_to_ns(now - gs_ts) > CIPHER_GPU_STATE_STALE_NS) {
			goto sleep_and_continue;
		}

		/* RCU walk of cipher_pid_table. Update derived fields. */
		rcu_read_lock();
		hash_for_each_rcu(cipher_pid_table, bkt, e, node) {
			cipher_update_tenant_derived(e, &gs_snap);
		}
		rcu_read_unlock();

sleep_and_continue:
		/* W4 Step 4: cipher_partition_tick() removed with LP-8 retirement.
		 * The legacy 32-slot array had no live writes since CP 5.4 deactivated
		 * nr 9, so the tick had nothing to reap or rebalance. */
		/* 1 kHz cadence: sleep 1 ms. */
		msleep_interruptible(1000 / CIPHER_STATE_UPDATER_HZ);
	}

	pr_info("cipher_state_updater: kthread stopping\n");
	return 0;
}

int cipher_state_updater_init(void)
{
	struct task_struct *t;

	if (atomic_xchg(&cipher_state_updater_running, 1) == 1) {
		pr_warn("cipher_state_updater: already running\n");
		return -EBUSY;
	}

	t = kthread_create(cipher_state_updater_fn, NULL, "cipher_state");
	if (IS_ERR(t)) {
		atomic_set(&cipher_state_updater_running, 0);
		pr_err("cipher_state_updater: kthread_create failed: %ld\n",
		       PTR_ERR(t));
		return PTR_ERR(t);
	}
	cipher_state_updater_task = t;
	wake_up_process(t);
	return 0;
}

void cipher_state_updater_exit(void)
{
	if (cipher_state_updater_task) {
		kthread_stop(cipher_state_updater_task);
		cipher_state_updater_task = NULL;
	}
	atomic_set(&cipher_state_updater_running, 0);
}
