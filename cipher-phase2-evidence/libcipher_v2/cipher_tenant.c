/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_tenant.c -- tenant identity registration with cipher_kmod.
 *
 * Reads CIPHER_TENANT_ID, opens /dev/cipher, issues
 * CIPHER_REGISTER_TENANT, closes. Single function, no global state.
 */
#define _GNU_SOURCE   /* gettid() from glibc */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/types.h>

#include "cipher_ioctl.h"
#include "cipher_v2_internal.h"

int cipher_v2_tenant_register(void)
{
	const char *tenant = getenv(CIPHER_V2_TENANT_ENV);
	struct cipher_register_tenant payload;
	int fd, rc;

	if (!tenant || !tenant[0]) {
		cipher_dbg("no %s set; skipping registration",
			   CIPHER_V2_TENANT_ENV);
		return 0;
	}

	fd = open(CIPHER_V2_DEV, O_RDWR);
	if (fd < 0) {
		cipher_log("open(%s) failed: %s -- tenant '%s' NOT registered",
			   CIPHER_V2_DEV, strerror(errno), tenant);
		return -1;
	}

	memset(&payload, 0, sizeof(payload));
	payload.pid  = (__u32)gettid();   /* kernel task->pid is LWP */
	payload.tgid = (__u32)getpid();   /* kernel task->tgid is process */
	strncpy(payload.tenant_id, tenant, sizeof(payload.tenant_id) - 1);
	payload.tenant_id[sizeof(payload.tenant_id) - 1] = '\0';

	rc = ioctl(fd, CIPHER_REGISTER_TENANT, &payload);
	if (rc < 0) {
		cipher_log("ioctl REGISTER_TENANT failed: %s (errno=%d) -- tenant '%s' NOT registered",
			   strerror(errno), errno, payload.tenant_id);
		close(fd);
		return -2;
	}

	cipher_log("tenant '%s' registered for pid=%u tgid=%u",
		   payload.tenant_id, payload.pid, payload.tgid);
	close(fd);
	return 0;
}
