/* SPDX-License-Identifier: GPL-2.0
 * cipher_test_happy.c — exercise REGISTER_TENANT + SUBMIT_LAUNCH_STATS
 * as the current (non-root) user. All four ioctls run with valid payloads.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>

#include "/home/ubuntu/cipher_kmod/cipher_ioctl.h"

static pid_t mytid(void)
{
	return (pid_t)syscall(SYS_gettid);
}

static __u64 monotonic_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (__u64)ts.tv_sec * 1000000000ULL + (__u64)ts.tv_nsec;
}

int main(void)
{
	int fd, rc, failures = 0;

	fd = open("/dev/cipher", O_RDWR);
	if (fd < 0) {
		fprintf(stderr, "FAIL  open /dev/cipher: %s\n", strerror(errno));
		return 1;
	}
	printf("PASS  open /dev/cipher fd=%d uid=%u tid=%d pid=%d\n",
	       fd, geteuid(), mytid(), getpid());

	/* Step 2: REGISTER_TENANT (Phase 2 ABI regression) */
	{
		struct cipher_register_tenant rt;
		memset(&rt, 0, sizeof(rt));
		rt.pid  = mytid();
		rt.tgid = getpid();
		strncpy(rt.tenant_id, "happy-test-tenant", CIPHER_TENANT_ID_LEN - 1);
		errno = 0;
		rc = ioctl(fd, CIPHER_REGISTER_TENANT, &rt);
		if (rc == 0) {
			printf("PASS  CIPHER_REGISTER_TENANT pid=%u tgid=%u tenant=%s\n",
			       rt.pid, rt.tgid, rt.tenant_id);
		} else {
			fprintf(stderr, "FAIL  CIPHER_REGISTER_TENANT rc=%d errno=%d %s\n",
				rc, errno, strerror(errno));
			failures++;
		}
	}

	/* Step 3: SUBMIT_LAUNCH_STATS (no CAP_SYS_ADMIN, anti-spoof matches) */
	{
		struct cipher_launch_stats ls;
		memset(&ls, 0, sizeof(ls));
		ls.pid            = mytid();
		ls.tgid           = getpid();
		ls.timestamp_ns   = monotonic_ns();
		ls.launches_total = 1000;
		ls.grid_ops_total = 50000;
		errno = 0;
		rc = ioctl(fd, CIPHER_SUBMIT_LAUNCH_STATS, &ls);
		if (rc == 0) {
			printf("PASS  CIPHER_SUBMIT_LAUNCH_STATS launches=%llu grid_ops=%llu\n",
			       (unsigned long long)ls.launches_total,
			       (unsigned long long)ls.grid_ops_total);
		} else {
			fprintf(stderr, "FAIL  CIPHER_SUBMIT_LAUNCH_STATS rc=%d errno=%d %s\n",
				rc, errno, strerror(errno));
			failures++;
		}
	}

	close(fd);
	if (failures) {
		fprintf(stderr, "happy: %d FAILURE(S)\n", failures);
		return 1;
	}
	printf("happy: all PASS\n");
	return 0;
}
