// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * cipher_kvdedup.c — T4.6.4 kmod-owned cross-tenant KV page dedup.
 *
 * /dev/cipher_kvdedup. The dedup refcount table lives in kernel memory;
 * tenant processes ioctl in. The kernel outlives tenant processes and
 * is notified of tenant death via the release fop — that is what makes
 * cross-tenant resource ownership defensible.
 *
 * Userspace (cipher_rt) computes the xxhash64 of the 2 MiB page and the
 * memcmp-verify on a hit. The kmod owns the table, the refcounts, and
 * the cuIpc POSIX-FD handles (held as struct file *). See
 * t4_6_4_design.md.
 *
 * Locking: a single mutex (kvd_lock) — dedup is a per-request cold path;
 * the bucket walk measured 33 ns; the handler does sleepable work
 * (kzalloc, fd_install, copy_to_user) which a mutex permits cleanly.
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/file.h>
#include <linux/uaccess.h>
#include <linux/list.h>
#include <linux/xarray.h>
#include <linux/atomic.h>

#include "cipher_internal.h"   /* cipher_kvdedup_init/_exit prototypes */
#include "cipher_kvdedup.h"

#define KVD_BUCKETS 65536u   /* power of 2 */

/* One unique physical page registered in the table. */
struct kvd_entry {
	u64               content_hash;
	struct file      *handle_file;   /* kmod's ref to the cuIpc POSIX fd */
	u32               refcount;
	u32               pool_offset;   /* == its xarray id */
	struct hlist_node node;          /* kvd_buckets chain */
};

/* A page a given tenant fd holds a reference to. */
struct kvd_track {
	u32              pool_offset;
	struct list_head node;
};

/* Per-fd tenant state — file->private_data. */
struct kvd_tenant {
	u32              tenant_id;
	u32              n_tracked;
	struct list_head tracked;        /* list of kvd_track */
};

static DEFINE_MUTEX(kvd_lock);
static struct hlist_head *kvd_buckets;          /* KVD_BUCKETS heads */
static DEFINE_XARRAY_ALLOC1(kvd_xa);            /* pool_offset -> kvd_entry* */
static u32 kvd_next_tenant = 1;

/* stats — all under kvd_lock except kvd_tenants_open */
static u64 kvd_entries, kvd_virtual_refs;
static u64 kvd_puts, kvd_hits, kvd_misses, kvd_forced, kvd_releases;
static atomic_t kvd_tenants_open = ATOMIC_INIT(0);

/* device */
static int           kvd_major;
static struct cdev   kvd_cdev;
static struct class *kvd_class;
static struct device *kvd_device;

/* ---- table helpers (caller holds kvd_lock) ---- */

static struct kvd_entry *kvd_find_by_hash(u64 h)
{
	struct kvd_entry *e;

	hlist_for_each_entry(e, &kvd_buckets[h & (KVD_BUCKETS - 1)], node)
		if (e->content_hash == h)
			return e;
	return NULL;
}

/* Drop one reference to an entry; free physical + table slot at zero. */
static void kvd_entry_deref(struct kvd_entry *e)
{
	if (WARN_ON_ONCE(e->refcount == 0))
		return;
	if (kvd_virtual_refs)
		kvd_virtual_refs--;
	if (--e->refcount == 0) {
		hlist_del(&e->node);
		xa_erase(&kvd_xa, e->pool_offset);
		fput(e->handle_file);          /* releases the cuIpc handle */
		kfree(e);
		if (kvd_entries)
			kvd_entries--;
		kvd_releases++;
	}
}

/* Append a tracking record to a tenant. Caller holds kvd_lock.
 * Returns 0, or -ENOSPC if the per-tenant cap is hit. */
static int kvd_track_add(struct kvd_tenant *t, u32 pool_offset)
{
	struct kvd_track *tr;

	if (t->n_tracked >= CIPHER_KVDEDUP_MAX_PER_TENANT)
		return -ENOSPC;
	tr = kzalloc(sizeof(*tr), GFP_KERNEL);
	if (!tr)
		return -ENOMEM;
	tr->pool_offset = pool_offset;
	list_add(&tr->node, &t->tracked);
	t->n_tracked++;
	return 0;
}

/* Find + remove a tenant's tracking record. Caller holds kvd_lock.
 * Returns 0 if found (anti-spoof: a tenant may only free what it holds),
 * -EPERM otherwise. */
static int kvd_track_remove(struct kvd_tenant *t, u32 pool_offset)
{
	struct kvd_track *tr;

	list_for_each_entry(tr, &t->tracked, node) {
		if (tr->pool_offset == pool_offset) {
			list_del(&tr->node);
			kfree(tr);
			t->n_tracked--;
			return 0;
		}
	}
	return -EPERM;
}

/* ---- ioctl handlers ---- */

static long kvd_ioctl_init(struct kvd_tenant *t, unsigned long arg)
{
	struct cipher_kvdedup_init out = { .tenant_id = t->tenant_id };

	if (copy_to_user((void __user *)arg, &out, sizeof(out)))
		return -EFAULT;
	return 0;
}

static long kvd_ioctl_put(struct kvd_tenant *t, unsigned long arg)
{
	struct cipher_kvdedup_put p;
	struct file *exp_file;
	struct kvd_entry *e;
	long ret = 0;
	int newfd;

	if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
		return -EFAULT;
	if (p.export_fd < 0)
		return -EBADF;

	exp_file = fget(p.export_fd);     /* one ref; transferred or dropped */
	if (!exp_file)
		return -EBADF;

	p.result = CIPHER_KVDEDUP_RESULT_MISS;
	p.candidate_fd = -1;
	p.pool_offset = 0;

	mutex_lock(&kvd_lock);
	kvd_puts++;

	e = (p.flags & CIPHER_KVDEDUP_FLAG_FORCE_NEW)
		? NULL : kvd_find_by_hash(p.content_hash);

	if (e) {
		/* HIT: hand the requester a fresh fd to the existing handle
		 * so userspace can memcmp-verify. Do NOT bump refcount or
		 * track yet — that is CONFIRM's job. */
		newfd = get_unused_fd_flags(O_CLOEXEC);
		if (newfd < 0) {
			ret = newfd;
			goto out;
		}
		get_file(e->handle_file);
		fd_install(newfd, e->handle_file);
		p.result = CIPHER_KVDEDUP_RESULT_HIT;
		p.candidate_fd = newfd;
		p.pool_offset = e->pool_offset;
		kvd_hits++;
	} else {
		/* MISS (or FORCE_NEW): register a new entry; the kmod keeps
		 * the exporter's handle (transfer the fget ref). */
		u32 id;

		e = kzalloc(sizeof(*e), GFP_KERNEL);
		if (!e) {
			ret = -ENOMEM;
			goto out;
		}
		ret = xa_alloc(&kvd_xa, &id, e,
		               XA_LIMIT(1, CIPHER_KVDEDUP_MAX_ENTRIES),
		               GFP_KERNEL);
		if (ret) {
			kfree(e);
			ret = -ENOSPC;          /* global entry cap */
			goto out;
		}
		e->content_hash = p.content_hash;
		e->handle_file  = exp_file;     /* keep the fget ref */
		e->refcount     = 1;
		e->pool_offset  = id;
		hlist_add_head(&e->node,
		               &kvd_buckets[p.content_hash & (KVD_BUCKETS - 1)]);

		ret = kvd_track_add(t, id);
		if (ret) {
			/* roll the registration back fully */
			hlist_del(&e->node);
			xa_erase(&kvd_xa, id);
			kfree(e);
			goto out;               /* exp_file fput below */
		}
		exp_file = NULL;                /* ref transferred to the entry */
		kvd_entries++;
		kvd_virtual_refs++;
		kvd_misses++;
		if (p.flags & CIPHER_KVDEDUP_FLAG_FORCE_NEW)
			kvd_forced++;
		p.result = CIPHER_KVDEDUP_RESULT_MISS;
		p.pool_offset = id;
	}

out:
	mutex_unlock(&kvd_lock);
	if (exp_file)                       /* HIT, or error: not transferred */
		fput(exp_file);
	if (ret == 0 && copy_to_user((void __user *)arg, &p, sizeof(p)))
		return -EFAULT;
	return ret;
}

static long kvd_ioctl_confirm(struct kvd_tenant *t, unsigned long arg)
{
	struct cipher_kvdedup_confirm c;
	struct kvd_entry *e;
	long ret;

	if (copy_from_user(&c, (void __user *)arg, sizeof(c)))
		return -EFAULT;

	mutex_lock(&kvd_lock);
	e = xa_load(&kvd_xa, (unsigned long)c.pool_offset);
	if (!e || e->refcount == 0) {
		mutex_unlock(&kvd_lock);
		return -ENOENT;
	}
	ret = kvd_track_add(t, e->pool_offset);
	if (ret == 0) {
		e->refcount++;
		kvd_virtual_refs++;
	}
	mutex_unlock(&kvd_lock);
	return ret;
}

static long kvd_ioctl_free(struct kvd_tenant *t, unsigned long arg)
{
	struct cipher_kvdedup_free f;
	struct kvd_entry *e;
	int ret;

	if (copy_from_user(&f, (void __user *)arg, sizeof(f)))
		return -EFAULT;

	mutex_lock(&kvd_lock);
	/* anti-spoof: a tenant may only free a page it holds */
	ret = kvd_track_remove(t, (u32)f.pool_offset);
	if (ret) {
		mutex_unlock(&kvd_lock);
		return ret;             /* -EPERM */
	}
	e = xa_load(&kvd_xa, (unsigned long)f.pool_offset);
	if (e)
		kvd_entry_deref(e);
	mutex_unlock(&kvd_lock);
	return 0;
}

static long kvd_ioctl_stats(unsigned long arg)
{
	struct cipher_kvdedup_stats s;

	memset(&s, 0, sizeof(s));
	mutex_lock(&kvd_lock);
	s.entries           = kvd_entries;
	s.virtual_refs      = kvd_virtual_refs;
	s.puts              = kvd_puts;
	s.hits              = kvd_hits;
	s.misses            = kvd_misses;
	s.collisions_forced = kvd_forced;
	s.refcount_releases = kvd_releases;
	mutex_unlock(&kvd_lock);
	s.tenants_open = (u32)atomic_read(&kvd_tenants_open);

	if (copy_to_user((void __user *)arg, &s, sizeof(s)))
		return -EFAULT;
	return 0;
}

/* ---- file operations ---- */

static int kvd_open(struct inode *inode, struct file *file)
{
	struct kvd_tenant *t;

	t = kzalloc(sizeof(*t), GFP_KERNEL);
	if (!t)
		return -ENOMEM;
	INIT_LIST_HEAD(&t->tracked);

	mutex_lock(&kvd_lock);
	t->tenant_id = kvd_next_tenant++;
	mutex_unlock(&kvd_lock);

	file->private_data = t;
	atomic_inc(&kvd_tenants_open);
	return 0;
}

/* The teardown guarantee: fires on close() AND on process death
 * (SIGKILL included) — the kernel runs the release fop either way.
 * Every page this tenant still holds is dereferenced here. */
static int kvd_release(struct inode *inode, struct file *file)
{
	struct kvd_tenant *t = file->private_data;
	struct kvd_track *tr, *tmp;
	struct kvd_entry *e;

	if (!t)
		return 0;

	mutex_lock(&kvd_lock);
	list_for_each_entry_safe(tr, tmp, &t->tracked, node) {
		e = xa_load(&kvd_xa, tr->pool_offset);
		if (e)
			kvd_entry_deref(e);
		list_del(&tr->node);
		kfree(tr);
	}
	mutex_unlock(&kvd_lock);

	kfree(t);
	atomic_dec(&kvd_tenants_open);
	return 0;
}

static long kvd_unlocked_ioctl(struct file *file, unsigned int cmd,
                               unsigned long arg)
{
	struct kvd_tenant *t = file->private_data;

	if (!t)
		return -EINVAL;
	switch (cmd) {
	case CIPHER_KVDEDUP_INIT:    return kvd_ioctl_init(t, arg);
	case CIPHER_KVDEDUP_PUT:     return kvd_ioctl_put(t, arg);
	case CIPHER_KVDEDUP_CONFIRM: return kvd_ioctl_confirm(t, arg);
	case CIPHER_KVDEDUP_FREE:    return kvd_ioctl_free(t, arg);
	case CIPHER_KVDEDUP_STATS:   return kvd_ioctl_stats(arg);
	default:                     return -ENOTTY;
	}
}

static const struct file_operations kvd_fops = {
	.owner          = THIS_MODULE,   /* rmmod with open fds -> EBUSY */
	.open           = kvd_open,
	.release        = kvd_release,
	.unlocked_ioctl = kvd_unlocked_ioctl,
};

/* ---- module init / exit (called from cipher_main.c) ---- */

/*
 * devnode callback — codifies /dev/cipher_kvdedup at 0666 root:root at node
 * creation so the mode survives every module reload (kmod 0.4.8). Matches the
 * CIPHER ABI trust model (cipher_ioctl.h): cross-tenant KV-dedup clients run
 * non-root. Operator deployments may tighten via a udev rule (0660 + group).
 */
static char *kvd_devnode(const struct device *dev, umode_t *mode)
{
	if (mode)
		*mode = 0666;
	return NULL;
}

int cipher_kvdedup_init(void)
{
	dev_t devno;
	int rc;

	kvd_buckets = kvcalloc(KVD_BUCKETS, sizeof(*kvd_buckets), GFP_KERNEL);
	if (!kvd_buckets)
		return -ENOMEM;

	rc = alloc_chrdev_region(&devno, 0, 1, CIPHER_KVDEDUP_DEV_NAME);
	if (rc < 0)
		goto out_buckets;
	kvd_major = MAJOR(devno);

	cdev_init(&kvd_cdev, &kvd_fops);
	kvd_cdev.owner = THIS_MODULE;
	rc = cdev_add(&kvd_cdev, devno, 1);
	if (rc < 0)
		goto out_unreg;

	kvd_class = class_create(CIPHER_KVDEDUP_DEV_NAME);
	if (IS_ERR(kvd_class)) {
		rc = PTR_ERR(kvd_class);
		kvd_class = NULL;
		goto out_cdev;
	}

	/* Codify the node mode before device_create(); see kvd_devnode(). */
	kvd_class->devnode = kvd_devnode;

	kvd_device = device_create(kvd_class, NULL, devno, NULL,
	                           "%s", CIPHER_KVDEDUP_DEV_NAME);
	if (IS_ERR(kvd_device)) {
		rc = PTR_ERR(kvd_device);
		kvd_device = NULL;
		goto out_class;
	}

	pr_info("cipher_kmod: /dev/%s ready (major=%d, T4.6.4 cross-tenant KV dedup)\n",
		CIPHER_KVDEDUP_DEV_NAME, kvd_major);
	return 0;

out_class:
	class_destroy(kvd_class);
	kvd_class = NULL;
out_cdev:
	cdev_del(&kvd_cdev);
out_unreg:
	unregister_chrdev_region(MKDEV(kvd_major, 0), 1);
out_buckets:
	kvfree(kvd_buckets);
	kvd_buckets = NULL;
	return rc;
}

void cipher_kvdedup_exit(void)
{
	unsigned long idx;
	struct kvd_entry *e;

	/* Reachable only when no tenant fd is open (.owner pins the module
	 * while any fd lives), so the table should already be empty.
	 * Defensive drain regardless. */
	if (kvd_device)
		device_destroy(kvd_class, MKDEV(kvd_major, 0));
	if (kvd_class)
		class_destroy(kvd_class);
	cdev_del(&kvd_cdev);
	unregister_chrdev_region(MKDEV(kvd_major, 0), 1);

	xa_for_each(&kvd_xa, idx, e) {
		fput(e->handle_file);
		kfree(e);
	}
	xa_destroy(&kvd_xa);
	kvfree(kvd_buckets);
	kvd_buckets = NULL;
}
