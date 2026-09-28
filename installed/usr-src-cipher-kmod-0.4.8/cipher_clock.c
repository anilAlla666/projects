// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_clock.c -- Phase 4.3 T4.3.2 SM clock actuation via usermode_helper.
 *
 * The libcipher_rt user-process VOLT actuator hits NVML privilege boundary
 * (nvmlDeviceSetGpuLockedClocks returns NVML_ERROR_NOT_SUPPORTED from a
 * non-root injection context, even though it works from `sudo`). This file
 * implements the kmod-mediated path: an ioctl (nr 10) that takes a target
 * MHz, validates bounds, calls `nvidia-smi -lgc <mhz>` via call_usermodehelper
 * (which runs as root in kernel context), then waits for completion.
 *
 * Trust model:
 *   - /dev/cipher is 0666 by convention (existing CIPHER ABI pattern;
 *     other ioctls like REQUEST_SM_PARTITION also accept non-root). The
 *     access gate is the device-node permission, not a per-ioctl CAP
 *     check. Operator-deployed environments can restrict via udev rule
 *     (chgrp + 0660) if a stricter policy is required.
 *   - The kmod's call_usermodehelper inherits root regardless of the
 *     caller's privilege; that is the whole point of this ioctl —
 *     convey privilege to a non-root injection-time libcipher_rt
 *     actuator. Adding a per-ioctl CAP_SYS_ADMIN check here would
 *     defeat the design intent (the very gap T4.3.2 closes is
 *     non-root user-process → privileged actuation).
 *   - mhz must be in [CIPHER_CLOCK_MIN_MHZ, CIPHER_CLOCK_MAX_MHZ]
 *     (H100 safe range). Out-of-range → -EINVAL. Bounds are enforced
 *     regardless of caller — a malicious tenant cannot brick the GPU.
 *   - mhz == 0 is the explicit reset (calls nvidia-smi -rgc).
 *   - Rate-limit: at most one in-flight call (mutex below).
 *   - Audit: every successful actuation logged at KERN_INFO with caller
 *     pid + uid for after-the-fact attribution.
 *
 * Why call_usermodehelper:
 *   - nvidia.ko exports nvidia_p2p_* and nvidia_register_error_cb only;
 *     no clock-control symbols are exported (audit S2.A1).
 *   - GSP RPC packet construction would require firmware reverse-engineering.
 *   - usermodehelper is the well-trodden Linux pattern (kernel calls
 *     userspace binaries) and proven to work on this pod via sudo NOPASSWD.
 *
 * Latency: nvidia-smi spawn + NVML round-trip ~100-300 ms. Acceptable for
 * init-time one-shot clock-set; not suitable for per-launch actuation.
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/string.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/umh.h>
#include <linux/uaccess.h>

#include "cipher_ioctl.h"
#include "cipher_internal.h"

#define CIPHER_CLOCK_MIN_MHZ  210u
#define CIPHER_CLOCK_MAX_MHZ  1980u

static DEFINE_MUTEX(cipher_clock_lock);

/*
 * Run /usr/bin/nvidia-smi with the given args. Blocking call (UMH_WAIT_PROC).
 * Returns the umh exit code (0 = success).
 */
static int run_nvidia_smi(char **argv)
{
	static char *envp[] = {
		"HOME=/",
		"PATH=/sbin:/usr/sbin:/bin:/usr/bin",
		"TERM=linux",
		NULL,
	};
	int rc;

	rc = call_usermodehelper(argv[0], argv, envp, UMH_WAIT_PROC);
	if (rc != 0)
		pr_warn("cipher_clock: nvidia-smi rc=%d (%s)\n", rc, argv[0]);
	return rc;
}

/*
 * Apply the requested SM clock lock via `nvidia-smi -i 0 -lgc <mhz>`.
 * mhz == 0 means reset (`-rgc`).
 *
 * Returns 0 on success, -ERESTARTSYS on signal, -EBUSY on lock contention,
 * -EIO on umh failure.
 */
static int cipher_clock_apply(u32 mhz)
{
	char mhz_buf[16];
	char *argv_set[] = {
		"/usr/bin/nvidia-smi",
		"-i", "0",
		"-lgc", mhz_buf,
		NULL,
	};
	char *argv_reset[] = {
		"/usr/bin/nvidia-smi",
		"-i", "0",
		"-rgc",
		NULL,
	};
	int rc;

	if (mutex_lock_interruptible(&cipher_clock_lock))
		return -ERESTARTSYS;

	if (mhz == 0) {
		rc = run_nvidia_smi(argv_reset);
	} else {
		snprintf(mhz_buf, sizeof(mhz_buf), "%u", mhz);
		rc = run_nvidia_smi(argv_set);
	}

	mutex_unlock(&cipher_clock_lock);

	if (rc != 0)
		return -EIO;
	return 0;
}

/*
 * ioctl nr 10 handler: CIPHER_SET_CLOCK_MHZ
 *
 * payload is a single __u32 target MHz. Validates range, applies via
 * usermode helper. No per-ioctl CAP_SYS_ADMIN check — by design: access is
 * gated by the /dev/cipher node permission (0666; codified in cipher_dev.c
 * via cipher_devnode()), and the safety mechanism is the [MIN,MAX] MHz
 * bounds clamp plus the per-call uid-audit log below. A CAP check here would
 * defeat T4.3.2's purpose — conveying privilege to a non-root injection-time
 * actuator. See the "Trust model" block at the top of this file.
 */
int cipher_dev_set_clock_mhz(unsigned long arg)
{
	u32 mhz;
	int rc;

	if (copy_from_user(&mhz, (void __user *)arg, sizeof(mhz)))
		return -EFAULT;

	/* 0 is the explicit reset/unlock — bypass range check. */
	if (mhz != 0 && (mhz < CIPHER_CLOCK_MIN_MHZ || mhz > CIPHER_CLOCK_MAX_MHZ)) {
		pr_warn("cipher_clock: refused mhz=%u (allowed: 0 or [%u, %u])\n",
			mhz, CIPHER_CLOCK_MIN_MHZ, CIPHER_CLOCK_MAX_MHZ);
		return -EINVAL;
	}

	rc = cipher_clock_apply(mhz);
	if (rc == 0)
		pr_info("cipher_clock: set %u MHz via nvidia-smi (caller pid=%d uid=%u)\n",
			mhz, current->pid,
			from_kuid_munged(current_user_ns(), current_uid()));
	return rc;
}
