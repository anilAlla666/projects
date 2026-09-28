// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_tenant_snapshot.c — Phase 4.1 contract implementation.
 *
 * Implements the three lookup functions specified in PHASE_4_CONTRACT.md:
 *   - cipher_get_current_tenant_snapshot   (T4.1.3)
 *   - cipher_get_tenant_snapshot_by_id     (T4.1.4)
 *   - cipher_enumerate_tenants             (T4.1.5)
 *
 * Plus the partition-mask write-back path used by P4.2:
 *   - cipher_set_sm_partition_mask
 *   - cipher_get_sm_partition_mask
 *
 * Implementation notes:
 *
 * The snapshot is a per-CPU thread-local struct assembled on demand from
 * cipher_pid_stats fields. This avoids storing a separate full-size snapshot
 * per tenant (which would be 336 B * 150 = 50 KB cold-cache footprint) and
 * lets the assembly stay in register cache during the read. Total target
 * for assembly + bucket walk: < 200 ns on H100 host CPU.
 *
 * Writers (REGISTER_TENANT for identity fields; cipher_state_updater
 * kthread for derived fields; P4.2 allocator for partition_mask) update
 * individual u32/u64 fields on cipher_pid_stats via WRITE_ONCE. No struct-
 * wide locking. Readers tolerate brief cross-field inconsistency.
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/string.h>
#include <linux/spinlock.h>
#include <linux/hashtable.h>
#include <linux/rcupdate.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/percpu.h>
#include <linux/jiffies.h>

#include "cipher_internal.h"

/* Per-CPU assembled snapshot. Each CPU has one slot reused across reads
 * by tasks running on that CPU. The caller's rcu_read_lock guarantees no
 * preemption migration mid-read on PREEMPT_RCU; on classic RCU we get the
 * same via the implicit non-preempt section. */
static DEFINE_PER_CPU(struct cipher_tenant_snapshot, cipher_snapshot_tls);

/* ---- assembly helper ------------------------------------------------- */

/* Copy fields from a cipher_pid_stats entry into the per-CPU snapshot slot.
 * Caller must be inside rcu_read_lock(). The entry pointer is valid for
 * the duration of the read-side section. */
static void cipher_assemble_snapshot(struct cipher_tenant_snapshot *out,
                                     const struct cipher_pid_stats *e)
{
	/* Identity. */
	memcpy(out->tenant_id_str, e->tenant_id, CIPHER_TENANT_ID_LEN);
	out->tenant_id_str[CIPHER_TENANT_ID_LEN - 1] = '\0';
	out->tenant_session_fp = READ_ONCE(e->tenant_session_fp);
	out->tenant_handle_u32 = READ_ONCE(e->tenant_handle_u32);
	out->pid               = (u32)e->pid;
	out->tgid              = (u32)e->tgid;

	/* Live telemetry (Phase 3 fields). */
	out->sm_util_pct       = READ_ONCE(e->sm_util_pct);
	out->mem_util_pct      = READ_ONCE(e->mem_util_pct);
	out->launches_total    = READ_ONCE(e->launches_total);
	out->grid_ops_total    = READ_ONCE(e->grid_ops_total);
	out->ioctls_total      = (u64)atomic64_read(&e->total);

	/* SM partition cluster (P4.2). */
	out->sm_partition_mask  = READ_ONCE(e->sm_partition_mask);
	out->sm_partition_count = READ_ONCE(e->sm_partition_count);

	/* DVFS / thermal cluster (P4.3). */
	out->thermal_headroom_pct = READ_ONCE(e->thermal_headroom_pct);
	out->power_headroom_w     = READ_ONCE(e->power_headroom_w);
	out->sustained_clock_mhz  = READ_ONCE(e->sustained_clock_mhz);
	out->voltage_envelope_mv  = READ_ONCE(e->voltage_envelope_mv);

	/* L2 cluster (P4.4). */
	out->l2_residency_kb   = READ_ONCE(e->l2_residency_kb);
	out->hot_region_count  = READ_ONCE(e->hot_region_count);
	memcpy(out->predicted_hot_regions, e->predicted_hot_regions,
	       sizeof(out->predicted_hot_regions));

	/* KV cluster (P4.6). */
	out->kv_cache_size_mb        = READ_ONCE(e->kv_cache_size_mb);
	out->kv_compression_ratio_pct = READ_ONCE(e->kv_compression_ratio_pct);

	/* Weight cluster (P4.5). */
	out->weight_dedup_savings_mb    = READ_ONCE(e->weight_dedup_savings_mb);
	out->weight_content_hash_count  = READ_ONCE(e->weight_content_hash_count);

	/* Fusion cluster (P4.7). */
	out->fusion_recipe_id     = READ_ONCE(e->fusion_recipe_id);
	out->fusion_eligible_flag = READ_ONCE(e->fusion_eligible_flag);

	/* Agentic cluster (P4.7). */
	out->session_band                 = READ_ONCE(e->session_band);
	out->slo_priority                 = READ_ONCE(e->slo_priority);
	out->fairness_quota_remaining_pct = READ_ONCE(e->fairness_quota_remaining_pct);

	/* Attention cluster (P4.6 secondary). */
	out->graph_capture_state           = READ_ONCE(e->graph_capture_state);
	out->koopman_substitution_eligibility =
		READ_ONCE(e->koopman_substitution_eligibility);

	out->snapshot_jiffies = READ_ONCE(e->snapshot_jiffies);
}

/* ---- lookup helper (RCU read-side hashtable walk) -------------------- */

static struct cipher_pid_stats *cipher_pid_lookup_rcu_local(pid_t pid)
{
	struct cipher_pid_stats *e;
	hash_for_each_possible_rcu(cipher_pid_table, e, node, (u32)pid) {
		if (e->pid == pid)
			return e;
	}
	return NULL;
}

/* ---- T4.1.3: cipher_get_current_tenant_snapshot ---------------------- */

const struct cipher_tenant_snapshot *cipher_get_current_tenant_snapshot(void)
{
	struct cipher_pid_stats *e;
	struct cipher_tenant_snapshot *out;

	e = cipher_pid_lookup_rcu_local(current->pid);
	if (!e)
		return NULL;

	out = this_cpu_ptr(&cipher_snapshot_tls);
	cipher_assemble_snapshot(out, e);
	return out;
}
EXPORT_SYMBOL_GPL(cipher_get_current_tenant_snapshot);

/* Helper for ioctl nr 8 cross-process by-pid path. RCU read-side. */
const struct cipher_tenant_snapshot *cipher_get_tenant_snapshot_by_pid(pid_t pid)
{
	struct cipher_pid_stats *e;
	struct cipher_tenant_snapshot *out;

	e = cipher_pid_lookup_rcu_local(pid);
	if (!e)
		return NULL;

	out = this_cpu_ptr(&cipher_snapshot_tls);
	cipher_assemble_snapshot(out, e);
	return out;
}
EXPORT_SYMBOL_GPL(cipher_get_tenant_snapshot_by_pid);

/* ---- T4.1.4: cipher_get_tenant_snapshot_by_id ------------------------ */

const struct cipher_tenant_snapshot *
cipher_get_tenant_snapshot_by_id(const char *tenant_id)
{
	struct cipher_pid_stats *e;
	struct cipher_tenant_snapshot *out;
	int bkt;
	size_t len;

	if (!tenant_id || !tenant_id[0])
		return NULL;

	len = strnlen(tenant_id, CIPHER_TENANT_ID_LEN);

	hash_for_each_rcu(cipher_pid_table, bkt, e, node) {
		if (e->tenant_id[0] &&
		    strncmp(e->tenant_id, tenant_id, len) == 0 &&
		    (e->tenant_id[len] == '\0' || len == CIPHER_TENANT_ID_LEN)) {
			out = this_cpu_ptr(&cipher_snapshot_tls);
			cipher_assemble_snapshot(out, e);
			return out;
		}
	}
	return NULL;
}
EXPORT_SYMBOL_GPL(cipher_get_tenant_snapshot_by_id);

/* ---- T4.1.5: cipher_enumerate_tenants -------------------------------- */

int cipher_enumerate_tenants(struct cipher_tenant_snapshot *out, int max)
{
	struct cipher_pid_stats *e;
	int bkt, n = 0;

	if (!out || max <= 0)
		return 0;

	rcu_read_lock();
	hash_for_each_rcu(cipher_pid_table, bkt, e, node) {
		if (n >= max)
			break;
		cipher_assemble_snapshot(&out[n], e);
		n++;
	}
	rcu_read_unlock();
	return n;
}
EXPORT_SYMBOL_GPL(cipher_enumerate_tenants);

/* ---- partition allocator write-back paths (P4.2) --------------------- */

int cipher_set_sm_partition_mask(pid_t target_pid, u32 mask)
{
	struct cipher_pid_stats *e;
	u32 cnt = hweight32(mask);
	int rc = -ESRCH;

	rcu_read_lock();
	e = cipher_pid_lookup_rcu_local(target_pid);
	if (e) {
		WRITE_ONCE(e->sm_partition_mask, mask);
		WRITE_ONCE(e->sm_partition_count, cnt);
		rc = 0;
	}
	rcu_read_unlock();
	return rc;
}

int cipher_get_sm_partition_mask(pid_t target_pid, u32 *out_mask)
{
	struct cipher_pid_stats *e;
	int rc = -ESRCH;

	if (!out_mask)
		return -EINVAL;

	rcu_read_lock();
	e = cipher_pid_lookup_rcu_local(target_pid);
	if (e) {
		*out_mask = READ_ONCE(e->sm_partition_mask);
		rc = 0;
	}
	rcu_read_unlock();
	return rc;
}
