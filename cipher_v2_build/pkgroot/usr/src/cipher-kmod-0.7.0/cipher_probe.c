// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_probe.c — kprobe + kretprobe on nvidia_unlocked_ioctl.
 *
 * x86_64 SysV ABI:
 *   regs->di = struct file *      (1st arg)
 *   regs->si = unsigned int  cmd  (2nd arg, low 32 bits of rsi)
 *   regs->dx = unsigned long arg  (3rd arg)
 *   regs->ax = long return value  (in kretprobe handler)
 *
 * Per-PID stats live in a 1024-bucket RCU hashtable keyed by task->pid.
 * Hot path: rcu_read_lock + hash_for_each_possible_rcu + atomic64_inc.
 * Slow path (new PID): GFP_ATOMIC alloc, spin_lock, double-check, hash_add_rcu.
 */
#include <linux/module.h>
#include <linux/kprobes.h>
#include <linux/hashtable.h>
#include <linux/slab.h>
#include <linux/sched.h>
#include <linux/rcupdate.h>
#include <linux/jiffies.h>
#include <linux/ptrace.h>
#include <linux/string.h>

#include "cipher_internal.h"

/* The hashtable. cipher_proc.c walks it via the extern in the header. */
DEFINE_HASHTABLE(cipher_pid_table, CIPHER_PID_HASH_BITS);

static struct cipher_pid_stats *cipher_pid_lookup_rcu(pid_t pid)
{
	struct cipher_pid_stats *e;

	hash_for_each_possible_rcu(cipher_pid_table, e, node, pid) {
		if (e->pid == pid)
			return e;
	}
	return NULL;
}

struct cipher_pid_stats *cipher_pid_get_or_create(pid_t pid, pid_t tgid)
{
	struct cipher_pid_stats *e, *neu;

	rcu_read_lock();
	e = cipher_pid_lookup_rcu(pid);
	rcu_read_unlock();
	if (likely(e))
		return e;

	neu = kmalloc(sizeof(*neu), GFP_ATOMIC | __GFP_NOWARN);
	if (!neu) {
		atomic64_inc(&cipher_alloc_failures);
		return NULL;
	}

	memset(neu, 0, sizeof(*neu));
	INIT_HLIST_NODE(&neu->node);
	neu->pid  = pid;
	neu->tgid = tgid;
	get_task_comm(neu->comm, current);
	neu->first_seen_jiffies = get_jiffies_64();
	neu->last_seen_jiffies  = neu->first_seen_jiffies;

	spin_lock(&cipher_pid_insert_lock);
	/* Re-check under lock: another CPU may have inserted same pid. */
	e = cipher_pid_lookup_rcu(pid);
	if (e) {
		spin_unlock(&cipher_pid_insert_lock);
		kfree(neu);
		return e;
	}
	hash_add_rcu(cipher_pid_table, &neu->node, pid);
	spin_unlock(&cipher_pid_insert_lock);
	return neu;
}

static int cipher_kprobe_pre(struct kprobe *p, struct pt_regs *regs)
{
	struct cipher_pid_stats *e;
	unsigned int cmd;
	pid_t pid, tgid;
	int slot;

	cmd  = (unsigned int)regs->si;
	pid  = current->pid;
	tgid = current->tgid;
	slot = cipher_decode_nv_ioctl_slot(cmd);

	/* Globals first — succeed even if per-PID alloc fails. */
	if (slot >= 0)
		atomic64_inc(&cipher_global_cmd_counts[slot]);
	else
		atomic64_inc(&cipher_global_cmd_counts[CIPHER_NV_IOCTL_OTHER_SLOT]);
	atomic64_inc(&cipher_global_cmd_counts[CIPHER_NV_IOCTL_TOTAL_SLOT]);

	e = cipher_pid_get_or_create(pid, tgid);
	if (e) {
		/* Modification A: refresh comm every observation. ~16 B memcpy
		 * + brief task->alloc_lock acquisition per ioctl. */
		get_task_comm(e->comm, current);
		if (slot >= 0)
			atomic64_inc(&e->per_slot[slot]);
		else
			atomic64_inc(&e->other_count);
		atomic64_inc(&e->total);
		e->last_seen_jiffies = get_jiffies_64();
	}

	return 0;
}

/* Kretprobe entry handler stashes (cmd, pid) so the return handler can
 * decode the command and locate the per-PID slot without re-reading regs->si
 * (which the trampoline may have clobbered). */
struct cipher_ret_data {
	unsigned int cmd;
	pid_t        pid;
};

static int cipher_kretprobe_entry(struct kretprobe_instance *ri,
				  struct pt_regs *regs)
{
	struct cipher_ret_data *d = (struct cipher_ret_data *)ri->data;

	d->cmd = (unsigned int)regs->si;
	d->pid = current->pid;
	return 0;
}

static int cipher_kretprobe_handler(struct kretprobe_instance *ri,
				    struct pt_regs *regs)
{
	struct cipher_ret_data *d = (struct cipher_ret_data *)ri->data;
	long ret = (long)regs_return_value(regs);
	struct cipher_pid_stats *e;
	int slot;

	if (ret >= 0)
		return 0;

	slot = cipher_decode_nv_ioctl_slot(d->cmd);
	if (slot >= 0)
		atomic64_inc(&cipher_global_cmd_errors[slot]);
	else
		atomic64_inc(&cipher_global_cmd_errors[CIPHER_NV_IOCTL_OTHER_SLOT]);
	atomic64_inc(&cipher_global_cmd_errors[CIPHER_NV_IOCTL_TOTAL_SLOT]);

	rcu_read_lock();
	e = cipher_pid_lookup_rcu(d->pid);
	if (e)
		atomic64_inc(&e->errors);
	rcu_read_unlock();
	return 0;
}

static struct kprobe cipher_kp = {
	.symbol_name = "nvidia_unlocked_ioctl",
	.pre_handler = cipher_kprobe_pre,
};

static struct kretprobe cipher_krp = {
	.kp.symbol_name = "nvidia_unlocked_ioctl",
	.entry_handler  = cipher_kretprobe_entry,
	.handler        = cipher_kretprobe_handler,
	.data_size      = sizeof(struct cipher_ret_data),
	.maxactive      = 0,   /* kprobes default: 10 * num_possible_cpus() */
};

/*
 * do_exit reaper (Phase 1.5.3).
 *
 * Fires for every thread/task exit on the system, not just nvidia
 * consumers, so the fast path must be O(1) for not-tracked PIDs.
 * Pattern:
 *   1. RCU lookup. If miss (the common case), return immediately.
 *   2. If hit, take cipher_pid_insert_lock, re-check under lock,
 *      hash_del_rcu + kfree_rcu.
 *
 * Bypassing the lock on the miss path is safe because hash_del_rcu
 * and hash_add_rcu are themselves the only writers, and both happen
 * under the same lock.
 */
static int cipher_do_exit_pre(struct kprobe *p, struct pt_regs *regs)
{
	pid_t pid = current->pid;
	struct cipher_pid_stats *e;

	/* CP 5.4: release any 8-SM-group allocation held by this tenant —
	 * UNCONDITIONALLY, before the cipher_pid_stats fast-path guard below.
	 * The CP 5.4 ledger is an independent subsystem: a process can hold
	 * CP 5.4 groups without ever acquiring a cipher_pid_stats entry (it
	 * need not be an observed nvidia consumer nor a REGISTER_TENANT
	 * caller). Lock-free (atomic_cmpxchg + WRITE_ONCE) — safe in this
	 * kprobe atomic context — and a cheap bounded no-op when this pid
	 * owns no CP 5.4 groups (the common case). Touches only the CP 5.4
	 * ledger, so it is order-independent of the legacy reaper paths. */
	cipher_cp54_release(pid);

	rcu_read_lock();
	e = cipher_pid_lookup_rcu(pid);
	rcu_read_unlock();
	if (likely(!e))
		return 0;

	/* W4 Step 4: cipher_partition_release_slots_only() removed with LP-8
	 * retirement. The legacy 32-slot array was never written after CP 5.4
	 * deactivated nr 9, so the per-pid slot release was a no-op walk. The
	 * live partition state — CP 5.4 8-SM-group ownership — was already
	 * released at L198 above via cipher_cp54_release(). */

	spin_lock(&cipher_pid_insert_lock);
	e = cipher_pid_lookup_rcu(pid);
	if (e) {
		hash_del_rcu(&e->node);
		kfree_rcu(e, rcu);
		atomic64_inc(&cipher_reaped_count);
	}
	spin_unlock(&cipher_pid_insert_lock);
	return 0;
}

static struct kprobe cipher_kp_exit = {
	.symbol_name = "do_exit",
	.pre_handler = cipher_do_exit_pre,
};

int cipher_probe_init(void)
{
	int rc;

	rc = register_kprobe(&cipher_kp);
	if (rc < 0) {
		pr_err("cipher_kmod: register_kprobe failed: %d\n", rc);
		return rc;
	}
	cipher_hooked_addr = (unsigned long)cipher_kp.addr;

	rc = register_kretprobe(&cipher_krp);
	if (rc < 0) {
		pr_err("cipher_kmod: register_kretprobe failed: %d\n", rc);
		unregister_kprobe(&cipher_kp);
		return rc;
	}

	rc = register_kprobe(&cipher_kp_exit);
	if (rc < 0) {
		pr_err("cipher_kmod: register_kprobe(do_exit) failed: %d\n", rc);
		unregister_kretprobe(&cipher_krp);
		unregister_kprobe(&cipher_kp);
		return rc;
	}

	pr_info("cipher_kmod: kprobes attached at 0x%lx; do_exit reaper at 0x%lx\n",
		cipher_hooked_addr, (unsigned long)cipher_kp_exit.addr);
	return 0;
}

void cipher_probe_exit(void)
{
	struct cipher_pid_stats *e;
	struct hlist_node *tmp;
	int bkt;

	unregister_kprobe(&cipher_kp_exit);
	unregister_kretprobe(&cipher_krp);
	unregister_kprobe(&cipher_kp);

	/* Drain any pre-handler that was mid-flight when probes detached. */
	synchronize_rcu();

	spin_lock(&cipher_pid_insert_lock);
	hash_for_each_safe(cipher_pid_table, bkt, tmp, e, node) {
		hash_del_rcu(&e->node);
		kfree_rcu(e, rcu);
	}
	spin_unlock(&cipher_pid_insert_lock);

	rcu_barrier();   /* wait for all kfree_rcu callbacks to run */

	pr_info("cipher_kmod: kprobes detached, alloc_failures=%lld reaped=%lld\n",
		(long long)atomic64_read(&cipher_alloc_failures),
		(long long)atomic64_read(&cipher_reaped_count));
}
