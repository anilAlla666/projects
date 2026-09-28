// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_weight_arena.c — Track 2 SC5: kmod weight-arena fd custodian.
 *
 * Cross-tenant weight sharing (Track 2): a producer tenant packs a model's
 * weights into one CUDA VMM allocation and exports it as a POSIX fd; peer
 * tenants import that fd and run the model on the shared physical bytes.
 *
 * SC5 design-memo §1 (probe-confirmed, TRACK_2_SC5_PROBE_LOG.md): CUDA VMM
 * POSIX-fd shareable handles are ALREADY cross-process reference-counted by
 * the driver — the physical memory survives while any handle holder is alive,
 * and process exit auto-releases that process's handles. So the kmod does NOT
 * manage the physical memory and NEVER calls the CUDA driver.
 *
 * The kmod's role is the fd CUSTODIAN / rendezvous:
 *   - REGISTER — the producer hands the kmod its exported fd; the kmod takes
 *     an independent `struct file *` reference (fget). That reference keeps
 *     the cuMem shareable handle alive even after the producer exits.
 *   - IMPORT  — a consumer asks for the arena by id; the kmod dup's its held
 *     fd into the consumer (get_file + fd_install). The producer need not be
 *     alive. The consumer then cuMemImportFromShareableHandle's that fd.
 *   - the arena's metadata blob (the SC3 layout manifest + SC4 fingerprint)
 *     is stored opaque and returned by IMPORT — the kmod never interprets it.
 *
 * Lifetime: the kmod holds its fd reference while any participant (producer
 * or consumer) is alive. A periodic 5 s workqueue checks participant
 * liveness; an arena with zero live participants is reaped — the kmod fput's
 * its `struct file *` (dropping the last in-kmod reference; the CUDA driver
 * frees the physical once all process handles are also gone). A periodic
 * workqueue, not a do_exit kprobe hook, is used deliberately: arena
 * reclamation is not latency-critical (the physical is held safely by the
 * fd reference meanwhile) — unlike CP 5.4 SM groups — and this keeps SC5
 * off the delicate do_exit path. (SC5 design memo §e flagged this option.)
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/mutex.h>
#include <linux/atomic.h>
#include <linux/fs.h>
#include <linux/file.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/workqueue.h>
#include <linux/uaccess.h>
#include <linux/printk.h>
#include <linux/seq_file.h>

#include "cipher_internal.h"
#include "cipher_ioctl.h"

/* CIPHER_WA_MAX_ARENAS + CIPHER_WA_BLOB_MAX come from cipher_ioctl.h (ABI). */
#define CIPHER_WA_MAX_CONSUMERS  64
#define CIPHER_WA_REAP_SECS      5

struct cipher_wa_arena {
	int           in_use;
	u32           arena_id;
	pid_t         producer_pid;                       /* 0 = producer gone */
	pid_t         consumers[CIPHER_WA_MAX_CONSUMERS];  /* 0 = empty slot */
	u64           base;
	u64           size;
	struct file  *fdfile;                             /* the kmod-held ref */
	u32           blob_len;
	char          blob[CIPHER_WA_BLOB_MAX];           /* opaque metadata */
};

static struct cipher_wa_arena cipher_wa_arenas[CIPHER_WA_MAX_ARENAS];
static DEFINE_MUTEX(cipher_wa_lock);
static atomic_t cipher_wa_next_id = ATOMIC_INIT(0);
static struct delayed_work cipher_wa_reap_work;

/* ---- helpers (caller holds cipher_wa_lock) ---------------------------- */

static struct cipher_wa_arena *wa_find(u32 arena_id)
{
	int i;

	for (i = 0; i < CIPHER_WA_MAX_ARENAS; i++)
		if (cipher_wa_arenas[i].in_use &&
		    cipher_wa_arenas[i].arena_id == arena_id)
			return &cipher_wa_arenas[i];
	return NULL;
}

/* Is pid `nr` still a live task? (init-namespace pid, as stored.) */
static bool wa_pid_alive(pid_t nr)
{
	struct pid *p;
	struct task_struct *t;

	if (nr == 0)
		return false;
	p = find_get_pid(nr);
	if (!p)
		return false;
	t = get_pid_task(p, PIDTYPE_PID);
	if (t)
		put_task_struct(t);
	put_pid(p);
	return t != NULL;
}

static int wa_live_participants(const struct cipher_wa_arena *a)
{
	int i, n = 0;

	if (a->producer_pid)
		n++;
	for (i = 0; i < CIPHER_WA_MAX_CONSUMERS; i++)
		if (a->consumers[i])
			n++;
	return n;
}

/* Release an arena: drop the kmod's fd reference, clear the slot.
 * Caller holds the lock. The CUDA driver frees the physical once every
 * process handle is also gone — the kmod never calls CUDA. */
static void wa_release(struct cipher_wa_arena *a)
{
	if (a->fdfile)
		fput(a->fdfile);
	a->fdfile = NULL;
	a->in_use = 0;
}

/* ---- periodic liveness reaper ----------------------------------------- */

static void cipher_wa_reap_fn(struct work_struct *w)
{
	int i, j;

	mutex_lock(&cipher_wa_lock);
	for (i = 0; i < CIPHER_WA_MAX_ARENAS; i++) {
		struct cipher_wa_arena *a = &cipher_wa_arenas[i];

		if (!a->in_use)
			continue;
		if (a->producer_pid && !wa_pid_alive(a->producer_pid))
			a->producer_pid = 0;
		for (j = 0; j < CIPHER_WA_MAX_CONSUMERS; j++)
			if (a->consumers[j] && !wa_pid_alive(a->consumers[j]))
				a->consumers[j] = 0;
		if (wa_live_participants(a) == 0) {
			pr_info("cipher_kmod: weight-arena id=%u reaped (all "
			        "participants exited)\n", a->arena_id);
			wa_release(a);
		}
	}
	mutex_unlock(&cipher_wa_lock);
	schedule_delayed_work(&cipher_wa_reap_work,
	                      CIPHER_WA_REAP_SECS * HZ);
}

/* ---- ioctl entry points (nrs 21-24) ----------------------------------- */

/* nr 21 — ARENA_REGISTER: a producer hands over its exported fd + metadata. */
long cipher_wa_ioctl_register(unsigned long arg)
{
	struct cipher_arena_register p;
	struct cipher_wa_arena *a = NULL;
	struct file *f;
	int i;

	if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
		return -EFAULT;
	if (p.blob_len > CIPHER_WA_BLOB_MAX)
		return -EINVAL;

	f = fget(p.fd);
	if (!f)
		return -EBADF;

	mutex_lock(&cipher_wa_lock);
	for (i = 0; i < CIPHER_WA_MAX_ARENAS; i++)
		if (!cipher_wa_arenas[i].in_use) {
			a = &cipher_wa_arenas[i];
			break;
		}
	if (!a) {
		mutex_unlock(&cipher_wa_lock);
		fput(f);
		return -ENOSPC;
	}
	if (p.blob_len &&
	    copy_from_user(a->blob, (void __user *)p.blob_ptr, p.blob_len)) {
		mutex_unlock(&cipher_wa_lock);
		fput(f);
		return -EFAULT;
	}
	a->arena_id     = (u32)atomic_inc_return(&cipher_wa_next_id);
	a->producer_pid = current->pid;
	memset(a->consumers, 0, sizeof(a->consumers));
	a->base     = p.base;
	a->size     = p.size;
	a->fdfile   = f;
	a->blob_len = p.blob_len;
	a->in_use   = 1;
	p.arena_id_out = a->arena_id;
	mutex_unlock(&cipher_wa_lock);

	if (copy_to_user((void __user *)arg, &p, sizeof(p))) {
		/* The arena was created + the fd fget'd, but the caller never
		 * learns its id — roll the registration back so it does not
		 * sit live-but-orphaned until the reaper notices. */
		mutex_lock(&cipher_wa_lock);
		if (a->in_use && a->arena_id == p.arena_id_out)
			wa_release(a);
		mutex_unlock(&cipher_wa_lock);
		return -EFAULT;
	}
	pr_info("cipher_kmod: weight-arena id=%u REGISTER pid=%d size=%llu\n",
	        p.arena_id_out, current->pid, p.size);
	return 0;
}

/* nr 22 — ARENA_IMPORT: a consumer obtains a dup'd fd + metadata by id.
 * Ordering: reserve the fd number, copy the blob out, and only then
 * fd_install — so an abort before install just put_unused_fd's a reserved
 * (uninstalled) number, no fd to revoke. */
long cipher_wa_ioctl_import(unsigned long arg)
{
	struct cipher_arena_import p;
	struct cipher_wa_arena *a;
	int slot = -1, newfd, i;

	if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
		return -EFAULT;

	mutex_lock(&cipher_wa_lock);
	a = wa_find(p.arena_id);
	if (!a) {
		mutex_unlock(&cipher_wa_lock);
		return -ENOENT;
	}
	for (i = 0; i < CIPHER_WA_MAX_CONSUMERS; i++)
		if (a->consumers[i] == 0) {
			slot = i;
			break;
		}
	if (slot < 0) {
		mutex_unlock(&cipher_wa_lock);
		return -ENOSPC;
	}
	newfd = get_unused_fd_flags(O_CLOEXEC);     /* reserve, not yet install */
	if (newfd < 0) {
		mutex_unlock(&cipher_wa_lock);
		return newfd;
	}
	if (a->blob_len) {
		if (p.blob_cap < a->blob_len) {
			put_unused_fd(newfd);
			mutex_unlock(&cipher_wa_lock);
			return -ENOSPC;             /* caller buffer too small */
		}
		if (copy_to_user((void __user *)p.blob_ptr,
		                 a->blob, a->blob_len)) {
			put_unused_fd(newfd);
			mutex_unlock(&cipher_wa_lock);
			return -EFAULT;
		}
	}
	/* commit: claim the consumer slot, dup the kmod-held fd into the caller */
	a->consumers[slot] = current->pid;
	get_file(a->fdfile);
	fd_install(newfd, a->fdfile);
	p.fd_out   = newfd;
	p.base     = a->base;
	p.size     = a->size;
	p.blob_len = a->blob_len;
	mutex_unlock(&cipher_wa_lock);

	if (copy_to_user((void __user *)arg, &p, sizeof(p)))
		return -EFAULT;     /* fd is installed; caller's bad arg ptr */
	pr_info("cipher_kmod: weight-arena id=%u IMPORT pid=%d -> fd=%d\n",
	        p.arena_id, current->pid, newfd);
	return 0;
}

/* nr 23 — ARENA_LEAVE: explicit participant departure (the clean-exit
 * fast path; the periodic reaper is the crash path). */
long cipher_wa_ioctl_leave(unsigned long arg)
{
	u32 arena_id;
	struct cipher_wa_arena *a;
	pid_t me = current->pid;
	int i;

	if (copy_from_user(&arena_id, (void __user *)arg, sizeof(arena_id)))
		return -EFAULT;

	mutex_lock(&cipher_wa_lock);
	a = wa_find(arena_id);
	if (!a) {
		mutex_unlock(&cipher_wa_lock);
		return -ENOENT;
	}
	if (a->producer_pid == me)
		a->producer_pid = 0;
	for (i = 0; i < CIPHER_WA_MAX_CONSUMERS; i++)
		if (a->consumers[i] == me)
			a->consumers[i] = 0;
	if (wa_live_participants(a) == 0)
		wa_release(a);
	mutex_unlock(&cipher_wa_lock);
	return 0;
}

/* nr 24 — ARENA_QUERY: read-only operator snapshot. */
long cipher_wa_ioctl_query(unsigned long arg)
{
	/* W6 G2 follow-on: struct cipher_arena_query grows 392 B -> 2408 B
	 * with CIPHER_WA_MAX_ARENAS 16 -> 100, exceeding the 1024 B kernel
	 * stack budget (gcc -Wframe-larger-than=1024). Heap-allocate. */
	struct cipher_arena_query *q;
	int i, j, n = 0;
	long rc = 0;

	q = kzalloc(sizeof(*q), GFP_KERNEL);
	if (!q)
		return -ENOMEM;
	mutex_lock(&cipher_wa_lock);
	for (i = 0; i < CIPHER_WA_MAX_ARENAS; i++) {
		struct cipher_wa_arena *a = &cipher_wa_arenas[i];
		int nc = 0;

		if (!a->in_use)
			continue;
		for (j = 0; j < CIPHER_WA_MAX_CONSUMERS; j++)
			if (a->consumers[j])
				nc++;
		q->arenas[n].arena_id     = a->arena_id;
		q->arenas[n].producer_pid = a->producer_pid;
		q->arenas[n].n_consumers  = nc;
		q->arenas[n].size         = a->size;
		n++;
	}
	mutex_unlock(&cipher_wa_lock);
	q->n_arenas = n;

	if (copy_to_user((void __user *)arg, q, sizeof(*q)))
		rc = -EFAULT;
	kfree(q);
	return rc;
}

/* ---- /proc/cipher/arenas --------------------------------------------- */

int cipher_wa_proc_show(struct seq_file *sf, void *v)
{
	int i, j;

	(void)v;
	mutex_lock(&cipher_wa_lock);
	seq_puts(sf, "Track 2 SC5 — weight-arena registry (kmod fd custodian)\n");
	for (i = 0; i < CIPHER_WA_MAX_ARENAS; i++) {
		struct cipher_wa_arena *a = &cipher_wa_arenas[i];
		int nc = 0;

		if (!a->in_use)
			continue;
		for (j = 0; j < CIPHER_WA_MAX_CONSUMERS; j++)
			if (a->consumers[j])
				nc++;
		seq_printf(sf, "arena id=%u producer_pid=%d consumers=%d "
		           "base=0x%llx size=%llu blob=%u\n",
		           a->arena_id, a->producer_pid, nc,
		           a->base, a->size, a->blob_len);
	}
	mutex_unlock(&cipher_wa_lock);
	return 0;
}

/* ---- init / exit ------------------------------------------------------ */

void cipher_wa_init(void)
{
	memset(cipher_wa_arenas, 0, sizeof(cipher_wa_arenas));
	atomic_set(&cipher_wa_next_id, 0);
	INIT_DELAYED_WORK(&cipher_wa_reap_work, cipher_wa_reap_fn);
	schedule_delayed_work(&cipher_wa_reap_work, CIPHER_WA_REAP_SECS * HZ);
	pr_info("cipher_kmod: Track 2 SC5 weight-arena registry — %d slots, "
	        "%ds liveness reaper\n", CIPHER_WA_MAX_ARENAS,
	        CIPHER_WA_REAP_SECS);
}

void cipher_wa_exit(void)
{
	int i;

	cancel_delayed_work_sync(&cipher_wa_reap_work);
	mutex_lock(&cipher_wa_lock);
	for (i = 0; i < CIPHER_WA_MAX_ARENAS; i++)
		if (cipher_wa_arenas[i].in_use)
			wa_release(&cipher_wa_arenas[i]);
	mutex_unlock(&cipher_wa_lock);
}
