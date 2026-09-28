/* SPDX-License-Identifier: GPL-2.0
 *
 * cipher_ioctl.h -- public ABI between cipher_kmod and userspace clients.
 *
 * IMPORTANT: this is a stable ABI. Once shipped, do not change struct
 * layouts or ioctl nrs. Add new ioctls via reserved nrs only. Existing
 * fields are never reordered, renamed, or repurposed.
 *
 * Phase 2: CIPHER_REGISTER_TENANT (nr 1) is the only live ioctl.
 * Phase 6: nrs 2/3/4 are reserved -- the kmod returns -ENOSYS for them.
 */
#ifndef CIPHER_IOCTL_H
#define CIPHER_IOCTL_H

#include <linux/types.h>
#include <linux/ioctl.h>

#define CIPHER_IOCTL_MAGIC    'C'

#define CIPHER_TENANT_ID_LEN  64

/*
 * CIPHER_REGISTER_TENANT (nr 1, _IOW)
 *
 * Caller declares its tenant identity. cipher_kmod stamps tenant_id
 * onto the per-PID hashtable entry for the calling task (the task's
 * `pid` is its LWP; `tgid` is its thread-group id).
 *
 * Security: the kernel verifies that payload.pid == current->pid and
 * payload.tgid == current->tgid before accepting. Spoofing another
 * task's identity returns -EPERM.
 */
struct cipher_register_tenant {
	__u32 pid;
	__u32 tgid;
	char  tenant_id[CIPHER_TENANT_ID_LEN];
};

#define CIPHER_REGISTER_TENANT \
	_IOW(CIPHER_IOCTL_MAGIC, 1, struct cipher_register_tenant)

/*
 * Reserved for Phase 6. ABI layout below is provisional; do NOT depend
 * on it. cipher_kmod currently returns -ENOSYS for these nrs.
 */
struct cipher_snapshot {
	__u32 size_used;
	char  buf[4096];
};

#define CIPHER_SNAPSHOT      _IOR(CIPHER_IOCTL_MAGIC, 2, struct cipher_snapshot)
#define CIPHER_RESET         _IOW(CIPHER_IOCTL_MAGIC, 3, __u32)
#define CIPHER_GET_VERSION   _IOR(CIPHER_IOCTL_MAGIC, 4, __u32)

#endif /* CIPHER_IOCTL_H */
