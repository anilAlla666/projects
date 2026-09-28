// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_ioctl_decode.c — NV_ESC nr → name resolution.
 *
 * NVIDIA encodes ioctls as _IO(NV_IOCTL_MAGIC='F', NV_ESC_xxx) where NV_ESC
 * numbers are in [200..218] for the 14 known commands. _IOC_NR(cmd) returns
 * the nr field (200..218); we subtract NV_IOCTL_NR_BASE to index the table.
 *
 * Anything with type != 'F' or nr outside [200..218] is treated as OTHER and
 * counted in CIPHER_NV_ESC_OTHER_SLOT — covers UVM/modeset/DRM ioctls plus
 * any nrs we haven't catalogued.
 */
#include <linux/ioctl.h>
#include <linux/types.h>

#include "cipher_internal.h"

#define NV_IOCTL_MAGIC    'F'
#define NV_IOCTL_NR_BASE  200
#define NV_IOCTL_NR_LAST  218   /* NV_ESC_WAIT_OPEN_COMPLETE */

static const char * const cipher_nv_esc_names[CIPHER_NV_ESC_COUNT] = {
	[0]  = "NV_ESC_CARD_INFO",            /* 200 */
	[1]  = "NV_ESC_REGISTER_FD",          /* 201 */
	[2]  = "(nr 202 unused)",
	[3]  = "(nr 203 unused)",
	[4]  = "(nr 204 unused)",
	[5]  = "(nr 205 unused)",
	[6]  = "NV_ESC_ALLOC_OS_EVENT",       /* 206 */
	[7]  = "NV_ESC_FREE_OS_EVENT",        /* 207 */
	[8]  = "(nr 208 unused)",
	[9]  = "NV_ESC_STATUS_CODE",          /* 209 */
	[10] = "NV_ESC_CHECK_VERSION_STR",    /* 210 */
	[11] = "NV_ESC_IOCTL_XFER_CMD",       /* 211 — dominant */
	[12] = "NV_ESC_ATTACH_GPUS_TO_FD",    /* 212 */
	[13] = "NV_ESC_QUERY_DEVICE_INTR",    /* 213 */
	[14] = "NV_ESC_SYS_PARAMS",           /* 214 */
	[15] = "NV_ESC_NUMA_INFO",            /* 215 */
	[16] = "NV_ESC_SET_NUMA_STATUS",      /* 216 */
	[17] = "NV_ESC_EXPORT_TO_DMABUF_FD",  /* 217 */
	[18] = "NV_ESC_WAIT_OPEN_COMPLETE",   /* 218 */
};

int cipher_decode_nv_esc_nr(unsigned int cmd)
{
	unsigned int type = _IOC_TYPE(cmd);
	unsigned int nr   = _IOC_NR(cmd);

	if (type != NV_IOCTL_MAGIC)
		return -1;
	if (nr < NV_IOCTL_NR_BASE || nr > NV_IOCTL_NR_LAST)
		return -1;
	return (int)(nr - NV_IOCTL_NR_BASE);
}

const char *cipher_decode_nv_esc_name(int nr)
{
	if (nr < 0 || nr >= CIPHER_NV_ESC_COUNT)
		return "UNKNOWN";
	return cipher_nv_esc_names[nr];
}
