// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_ioctl_decode.c — sorted nr->slot table + binary search.
 *
 * NVIDIA encodes ioctls as _IO(magic='F', nr) with nr taken from
 * NV_ESC_* (frontend, 200..218) or NV_ESC_RM_* (resource manager,
 * 0x27..0x5e). Both pass through nvidia_unlocked_ioctl. The cmd
 * passed to the handler is the encoded value; _IOC_TYPE/_IOC_NR
 * extract the magic byte and the nr.
 *
 * Sources:
 *   - kernel-open NVIDIA driver: nv_escape.h
 *   - gvisor abi/nvgpu (Go bindings, cross-checked)
 *
 * Phase 1 surfaced that on driver 580.105.08, NV_ESC_RM_* dominates
 * 90+% of CUDA ioctl traffic. NV_ESC_IOCTL_XFER_CMD is empty.
 */
#include <linux/ioctl.h>
#include <linux/types.h>
#include <linux/build_bug.h>
#include <linux/printk.h>

#include "cipher_internal.h"

#define NV_IOCTL_MAGIC  'F'

const struct cipher_nv_ioctl_def
cipher_nv_ioctls[CIPHER_NV_IOCTL_COUNT] = {
	/* RM family — ascending by nr */
	[CIPHER_SLOT_RM_ALLOC_MEMORY]                =
		{ 0x27, "NV_ESC_RM_ALLOC_MEMORY",                CIPHER_FAM_RM       },
	[CIPHER_SLOT_RM_FREE]                        =
		{ 0x29, "NV_ESC_RM_FREE",                        CIPHER_FAM_RM       },
	[CIPHER_SLOT_RM_CONTROL]                     =
		{ 0x2a, "NV_ESC_RM_CONTROL",                     CIPHER_FAM_RM       },
	[CIPHER_SLOT_RM_ALLOC]                       =
		{ 0x2b, "NV_ESC_RM_ALLOC",                       CIPHER_FAM_RM       },
	[CIPHER_SLOT_RM_DUP_OBJECT]                  =
		{ 0x34, "NV_ESC_RM_DUP_OBJECT",                  CIPHER_FAM_RM       },
	[CIPHER_SLOT_RM_SHARE]                       =
		{ 0x35, "NV_ESC_RM_SHARE",                       CIPHER_FAM_RM       },
	[CIPHER_SLOT_RM_VID_HEAP_CONTROL]            =
		{ 0x4a, "NV_ESC_RM_VID_HEAP_CONTROL",            CIPHER_FAM_RM       },
	[CIPHER_SLOT_RM_MAP_MEMORY]                  =
		{ 0x4e, "NV_ESC_RM_MAP_MEMORY",                  CIPHER_FAM_RM       },
	[CIPHER_SLOT_RM_UNMAP_MEMORY]                =
		{ 0x4f, "NV_ESC_RM_UNMAP_MEMORY",                CIPHER_FAM_RM       },
	[CIPHER_SLOT_RM_UPDATE_DEVICE_MAPPING_INFO]  =
		{ 0x5e, "NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO",  CIPHER_FAM_RM       },

	/* Frontend family — ascending by nr */
	[CIPHER_SLOT_CARD_INFO]              =
		{ 200, "NV_ESC_CARD_INFO",            CIPHER_FAM_FRONTEND },
	[CIPHER_SLOT_REGISTER_FD]            =
		{ 201, "NV_ESC_REGISTER_FD",          CIPHER_FAM_FRONTEND },
	[CIPHER_SLOT_ALLOC_OS_EVENT]         =
		{ 206, "NV_ESC_ALLOC_OS_EVENT",       CIPHER_FAM_FRONTEND },
	[CIPHER_SLOT_FREE_OS_EVENT]          =
		{ 207, "NV_ESC_FREE_OS_EVENT",        CIPHER_FAM_FRONTEND },
	[CIPHER_SLOT_STATUS_CODE]            =
		{ 209, "NV_ESC_STATUS_CODE",          CIPHER_FAM_FRONTEND },
	[CIPHER_SLOT_CHECK_VERSION_STR]      =
		{ 210, "NV_ESC_CHECK_VERSION_STR",    CIPHER_FAM_FRONTEND },
	[CIPHER_SLOT_IOCTL_XFER_CMD]         =
		{ 211, "NV_ESC_IOCTL_XFER_CMD",       CIPHER_FAM_FRONTEND },
	[CIPHER_SLOT_ATTACH_GPUS_TO_FD]      =
		{ 212, "NV_ESC_ATTACH_GPUS_TO_FD",    CIPHER_FAM_FRONTEND },
	[CIPHER_SLOT_QUERY_DEVICE_INTR]      =
		{ 213, "NV_ESC_QUERY_DEVICE_INTR",    CIPHER_FAM_FRONTEND },
	[CIPHER_SLOT_SYS_PARAMS]             =
		{ 214, "NV_ESC_SYS_PARAMS",           CIPHER_FAM_FRONTEND },
	[CIPHER_SLOT_NUMA_INFO]              =
		{ 215, "NV_ESC_NUMA_INFO",            CIPHER_FAM_FRONTEND },
	[CIPHER_SLOT_SET_NUMA_STATUS]        =
		{ 216, "NV_ESC_SET_NUMA_STATUS",      CIPHER_FAM_FRONTEND },
	[CIPHER_SLOT_EXPORT_TO_DMABUF_FD]    =
		{ 217, "NV_ESC_EXPORT_TO_DMABUF_FD",  CIPHER_FAM_FRONTEND },
	[CIPHER_SLOT_WAIT_OPEN_COMPLETE]     =
		{ 218, "NV_ESC_WAIT_OPEN_COMPLETE",   CIPHER_FAM_FRONTEND },
};

static_assert(ARRAY_SIZE(cipher_nv_ioctls) == CIPHER_NV_IOCTL_COUNT,
	      "cipher_nv_ioctls table size drift");

/*
 * Binary search by nr. Table is sorted ascending; designated
 * initialisers + cipher_decode_self_check() guarantee that.
 */
int cipher_decode_nv_ioctl_slot(unsigned int cmd)
{
	unsigned int nr;
	int lo, hi;

	if (_IOC_TYPE(cmd) != NV_IOCTL_MAGIC)
		return -1;
	nr = _IOC_NR(cmd);

	lo = 0;
	hi = CIPHER_NV_IOCTL_COUNT - 1;
	while (lo <= hi) {
		int mid = (lo + hi) >> 1;
		unsigned int mnr = cipher_nv_ioctls[mid].nr;

		if (mnr == nr)
			return mid;
		if (mnr < nr)
			lo = mid + 1;
		else
			hi = mid - 1;
	}
	return -1;
}

/*
 * Run-time guard: the table must be strictly ascending by nr and every
 * entry must have a name. Catches enum/init drift introduced by a
 * sloppy edit. Called from cipher_init() before kprobe registration.
 */
int cipher_decode_self_check(void)
{
	int i;

	for (i = 0; i < CIPHER_NV_IOCTL_COUNT; i++) {
		if (!cipher_nv_ioctls[i].name) {
			pr_err("cipher_kmod: ioctl table slot %d has no name\n",
			       i);
			return -EINVAL;
		}
		if (i > 0 &&
		    cipher_nv_ioctls[i].nr <= cipher_nv_ioctls[i - 1].nr) {
			pr_err("cipher_kmod: ioctl table not ascending at slot %d (nr 0x%x after 0x%x)\n",
			       i,
			       cipher_nv_ioctls[i].nr,
			       cipher_nv_ioctls[i - 1].nr);
			return -EINVAL;
		}
	}
	return 0;
}
