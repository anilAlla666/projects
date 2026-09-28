/* SPDX-License-Identifier: GPL-2.0
 * cipher_test_root.c — exercise the CAP_SYS_ADMIN-gated ioctls.
 * Must be run via sudo.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <time.h>

#include "/home/ubuntu/cipher_kmod/cipher_ioctl.h"

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
	printf("PASS  open /dev/cipher fd=%d uid=%u pid=%d\n",
	       fd, geteuid(), getpid());

	/* Step 2: SUBMIT_GPU_STATE with realistic H100 values. */
	{
		struct cipher_gpu_state gs;
		memset(&gs, 0, sizeof(gs));
		gs.timestamp_ns  = monotonic_ns();
		gs.power_mw      = 350000;   /* 350 W */
		gs.temp_c        = 65;
		gs.sm_clock_mhz  = 1980;
		gs.mem_clock_mhz = 1593;
		gs.sm_util_pct   = 75;
		gs.mem_util_pct  = 40;
		gs.fb_used_mb    = 8192;
		errno = 0;
		rc = ioctl(fd, CIPHER_SUBMIT_GPU_STATE, &gs);
		if (rc == 0) {
			printf("PASS  CIPHER_SUBMIT_GPU_STATE power=%u mW temp=%u C sm=%u%% mem=%u%%\n",
			       gs.power_mw, gs.temp_c, gs.sm_util_pct, gs.mem_util_pct);
		} else {
			fprintf(stderr, "FAIL  CIPHER_SUBMIT_GPU_STATE rc=%d errno=%d %s\n",
				rc, errno, strerror(errno));
			failures++;
		}
	}

	/* Step 3: SUBMIT_PROCESS_UTIL targeting ourselves. */
	{
		struct cipher_process_util pu;
		memset(&pu, 0, sizeof(pu));
		pu.pid           = getpid();
		pu.tgid          = getpid();
		pu.timestamp_ns  = monotonic_ns();
		pu.sm_util_pct   = 60;
		pu.mem_util_pct  = 30;
		pu.fb_used_mb    = 4096;
		pu.enc_util_pct  = 0;
		pu.dec_util_pct  = 0;
		errno = 0;
		rc = ioctl(fd, CIPHER_SUBMIT_PROCESS_UTIL, &pu);
		if (rc == 0) {
			printf("PASS  CIPHER_SUBMIT_PROCESS_UTIL pid=%u sm=%u%% mem=%u%% fb=%u MB\n",
			       pu.pid, pu.sm_util_pct, pu.mem_util_pct, pu.fb_used_mb);
		} else {
			fprintf(stderr, "FAIL  CIPHER_SUBMIT_PROCESS_UTIL rc=%d errno=%d %s\n",
				rc, errno, strerror(errno));
			failures++;
		}
	}

	close(fd);
	if (failures) {
		fprintf(stderr, "root: %d FAILURE(S)\n", failures);
		return 1;
	}
	printf("root: all PASS\n");
	return 0;
}
