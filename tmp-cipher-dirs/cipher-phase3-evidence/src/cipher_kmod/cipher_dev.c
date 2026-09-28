// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_dev.c -- /dev/cipher control surface.
 *
 * Phase 2: CIPHER_REGISTER_TENANT is live. The handler validates that
 * the caller's pid/tgid match the payload's claimed identity (no
 * spoofing tenant attribution across tasks), then stamps tenant_id on
 * the per-PID hashtable entry.
 *
 * Phase 3 Task 3: SUBMIT_GPU_STATE / SUBMIT_PROCESS_UTIL (CAP_SYS_ADMIN,
 * for the cipher-gpustate daemon) and SUBMIT_LAUNCH_STATS (anti-spoof,
 * for workloads' own CUPTI counters) are live.
 *
 * Phase 6 ioctls (SNAPSHOT, RESET, GET_VERSION) return -ENOSYS until
 * implemented.
 */
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/err.h>
#include <linux/uaccess.h>
#include <linux/string.h>
#include <linux/sched.h>
#include <linux/spinlock.h>

#include "cipher_internal.h"
#include "cipher_ioctl.h"

static int            cipher_dev_major;
static struct cdev    cipher_cdev;
static struct class  *cipher_class;
static struct device *cipher_device;

static int cipher_dev_open(struct inode *inode, struct file *file)
{
	return 0;
}

static int cipher_dev_release(struct inode *inode, struct file *file)
{
	return 0;
}

static long cipher_dev_register_tenant(unsigned long arg)
{
	struct cipher_register_tenant payload;
	struct cipher_pid_stats *e;
	pid_t cur_pid  = current->pid;
	pid_t cur_tgid = current->tgid;

	if (copy_from_user(&payload, (void __user *)arg, sizeof(payload)))
		return -EFAULT;

	/* Anti-spoof: caller can only register itself. */
	if ((pid_t)payload.pid != cur_pid || (pid_t)payload.tgid != cur_tgid)
		return -EPERM;

	/* Ensure NUL-termination of the tenant string. */
	payload.tenant_id[CIPHER_TENANT_ID_LEN - 1] = '\0';

	e = cipher_pid_get_or_create(cur_pid, cur_tgid);
	if (!e)
		return -ENOMEM;

	/* strscpy under the insert lock: same lock serialises hashtable
	 * inserts and tenant_id writes. Brief; readers in cipher_proc.c
	 * use memcpy without the lock, accepting that a concurrent
	 * tenant change yields a possibly-torn 64-byte string for one
	 * snapshot. Diagnostic output only; benign. */
	spin_lock(&cipher_pid_insert_lock);
	strscpy(e->tenant_id, payload.tenant_id, sizeof(e->tenant_id));
	spin_unlock(&cipher_pid_insert_lock);

	return 0;
}

static long cipher_dev_submit_gpu_state(unsigned long arg)
{
	struct cipher_gpu_state payload;

	if (!capable(CAP_SYS_ADMIN))
		return -EPERM;

	if (copy_from_user(&payload, (void __user *)arg, sizeof(payload)))
		return -EFAULT;

	if (payload.timestamp_ns == 0)
		return -EINVAL;
	if (payload.sm_util_pct > 100 || payload.mem_util_pct > 100)
		return -EINVAL;
	/* Sanity bound on temperature: any real thermal shutdown
	 * threshold is far below 200 C, so a value above that is
	 * almost certainly a daemon bug or wire corruption. */
	if (payload.temp_c > 200)
		return -EINVAL;

	spin_lock(&cipher_gpu_state.lock);
	cipher_gpu_state.timestamp_jiffies = get_jiffies_64();
	cipher_gpu_state.power_mw      = payload.power_mw;
	cipher_gpu_state.temp_c        = payload.temp_c;
	cipher_gpu_state.sm_clock_mhz  = payload.sm_clock_mhz;
	cipher_gpu_state.mem_clock_mhz = payload.mem_clock_mhz;
	cipher_gpu_state.sm_util_pct   = payload.sm_util_pct;
	cipher_gpu_state.mem_util_pct  = payload.mem_util_pct;
	cipher_gpu_state.fb_used_mb    = payload.fb_used_mb;
	spin_unlock(&cipher_gpu_state.lock);

	return 0;
}

static long cipher_dev_submit_process_util(unsigned long arg)
{
	struct cipher_process_util payload;
	struct cipher_pid_stats *e;

	if (!capable(CAP_SYS_ADMIN))
		return -EPERM;

	if (copy_from_user(&payload, (void __user *)arg, sizeof(payload)))
		return -EFAULT;

	if (payload.sm_util_pct  > 100 || payload.mem_util_pct > 100 ||
	    payload.enc_util_pct > 100 || payload.dec_util_pct > 100)
		return -EINVAL;

	e = cipher_pid_get_or_create((pid_t)payload.pid, (pid_t)payload.tgid);
	if (!e)
		return -ENOMEM;

	WRITE_ONCE(e->sm_util_pct,            payload.sm_util_pct);
	WRITE_ONCE(e->mem_util_pct,           payload.mem_util_pct);
	WRITE_ONCE(e->fb_used_mb,             payload.fb_used_mb);
	WRITE_ONCE(e->enc_util_pct,           payload.enc_util_pct);
	WRITE_ONCE(e->dec_util_pct,           payload.dec_util_pct);
	WRITE_ONCE(e->last_telemetry_jiffies, get_jiffies_64());

	return 0;
}

static long cipher_dev_submit_launch_stats(unsigned long arg)
{
	struct cipher_launch_stats payload;
	struct cipher_pid_stats *e;
	pid_t cur_pid  = current->pid;
	pid_t cur_tgid = current->tgid;

	/* No CAP_SYS_ADMIN: workloads call this themselves. Anti-spoof
	 * (caller pid/tgid must match payload) is the access control. */
	if (copy_from_user(&payload, (void __user *)arg, sizeof(payload)))
		return -EFAULT;

	if ((pid_t)payload.pid != cur_pid || (pid_t)payload.tgid != cur_tgid)
		return -EPERM;

	e = cipher_pid_get_or_create(cur_pid, cur_tgid);
	if (!e)
		return -ENOMEM;

	WRITE_ONCE(e->launches_total,         payload.launches_total);
	WRITE_ONCE(e->grid_ops_total,         payload.grid_ops_total);
	WRITE_ONCE(e->last_telemetry_jiffies, get_jiffies_64());

	return 0;
}

static long cipher_dev_unlocked_ioctl(struct file *file, unsigned int cmd,
				      unsigned long arg)
{
	switch (cmd) {
	case CIPHER_REGISTER_TENANT:
		return cipher_dev_register_tenant(arg);
	case CIPHER_SUBMIT_GPU_STATE:
		return cipher_dev_submit_gpu_state(arg);
	case CIPHER_SUBMIT_PROCESS_UTIL:
		return cipher_dev_submit_process_util(arg);
	case CIPHER_SUBMIT_LAUNCH_STATS:
		return cipher_dev_submit_launch_stats(arg);
	case CIPHER_SNAPSHOT:
	case CIPHER_RESET:
	case CIPHER_GET_VERSION:
		return -ENOSYS;
	default:
		return -ENOTTY;
	}
}

static const struct file_operations cipher_dev_fops = {
	.owner          = THIS_MODULE,
	.open           = cipher_dev_open,
	.release        = cipher_dev_release,
	.unlocked_ioctl = cipher_dev_unlocked_ioctl,
};

int cipher_dev_init(void)
{
	dev_t devno;
	int rc;

	rc = alloc_chrdev_region(&devno, 0, 1, CIPHER_DEV_NAME);
	if (rc < 0)
		return rc;
	cipher_dev_major = MAJOR(devno);

	cdev_init(&cipher_cdev, &cipher_dev_fops);
	cipher_cdev.owner = THIS_MODULE;
	rc = cdev_add(&cipher_cdev, devno, 1);
	if (rc < 0)
		goto out_unreg;

	cipher_class = class_create(CIPHER_DEV_NAME);
	if (IS_ERR(cipher_class)) {
		rc = PTR_ERR(cipher_class);
		cipher_class = NULL;
		goto out_cdev;
	}

	cipher_device = device_create(cipher_class, NULL, devno, NULL,
				      "%s", CIPHER_DEV_NAME);
	if (IS_ERR(cipher_device)) {
		rc = PTR_ERR(cipher_device);
		cipher_device = NULL;
		goto out_class;
	}

	pr_info("cipher_kmod: /dev/%s ready (major=%d, REGISTER_TENANT + GPU_STATE/PROCESS_UTIL/LAUNCH_STATS live)\n",
		CIPHER_DEV_NAME, cipher_dev_major);
	return 0;

out_class:
	class_destroy(cipher_class);
	cipher_class = NULL;
out_cdev:
	cdev_del(&cipher_cdev);
out_unreg:
	unregister_chrdev_region(MKDEV(cipher_dev_major, 0), 1);
	return rc;
}

void cipher_dev_exit(void)
{
	dev_t devno = MKDEV(cipher_dev_major, 0);

	if (cipher_device) {
		device_destroy(cipher_class, devno);
		cipher_device = NULL;
	}
	if (cipher_class) {
		class_destroy(cipher_class);
		cipher_class = NULL;
	}
	cdev_del(&cipher_cdev);
	unregister_chrdev_region(devno, 1);
}

/* The /dev/cipher node defaults to mode 0600 root:root via udev. For
 * Phase 2 testing we want non-root userspace to be able to register
 * tenants (so workloads don't need sudo). The udev rule is owner-
 * configurable; for in-pod testing we relax via chmod after insmod.
 * Production deployments will set the correct mode via udev rules
 * shipped in the .deb/.rpm package. */
