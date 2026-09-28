// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_dev.c — placeholder /dev/cipher.
 *
 * Phase 1: open/release succeed; ioctl returns -ENOTTY. The control surface
 * (snapshot/reset/observe-only attach) lands in Phase 2+.
 *
 * Created via cdev + class_create + device_create so udev gets a node.
 */
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/err.h>

#include "cipher_internal.h"

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

static long cipher_dev_unlocked_ioctl(struct file *file, unsigned int cmd,
				      unsigned long arg)
{
	return -ENOTTY;
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

	pr_info("cipher_kmod: /dev/%s ready (major=%d)\n",
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
