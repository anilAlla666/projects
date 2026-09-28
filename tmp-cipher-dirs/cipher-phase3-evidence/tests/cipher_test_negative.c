/* SPDX-License-Identifier: GPL-2.0
 * cipher_test_negative.c — verify every reject path returns the
 * correct errno. Run twice: once unprivileged, once via sudo.
 * geteuid() switches between the two test sets.
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
#include <linux/ioctl.h>

#include "/home/ubuntu/cipher_kmod/cipher_ioctl.h"

static int failures;

static __u64 monotonic_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (__u64)ts.tv_sec * 1000000000ULL + (__u64)ts.tv_nsec;
}

#define EXPECT_FAIL(name, expr, expected_errno) do {                              \
	errno = 0;                                                                \
	int _rc = (expr);                                                         \
	int _saved = errno;                                                       \
	if (_rc == -1 && _saved == (expected_errno)) {                            \
		printf("PASS  %-52s (got errno=%d %s)\n",                         \
		       name, _saved, strerror(_saved));                           \
	} else {                                                                  \
		fprintf(stderr,                                                   \
			"FAIL  %s: expected rc=-1 errno=%d (%s), got rc=%d errno=%d (%s)\n", \
			name, (expected_errno), strerror(expected_errno),         \
			_rc, _saved, strerror(_saved));                           \
		failures++;                                                       \
	}                                                                         \
} while (0)

int main(void)
{
	int fd;
	uid_t uid = geteuid();

	fd = open("/dev/cipher", O_RDWR);
	if (fd < 0) {
		fprintf(stderr, "FAIL  open /dev/cipher: %s\n", strerror(errno));
		return 1;
	}
	printf("=== negative tests (uid=%u pid=%d) ===\n", uid, getpid());

	if (uid != 0) {
		/* ---- non-root tests (1-6) ---- */

		/* Test 1: SUBMIT_GPU_STATE → EPERM */
		{
			struct cipher_gpu_state gs;
			memset(&gs, 0, sizeof(gs));
			gs.timestamp_ns = monotonic_ns();
			gs.sm_util_pct  = 50;
			gs.mem_util_pct = 50;
			EXPECT_FAIL("Test 1  SUBMIT_GPU_STATE as non-root",
				    ioctl(fd, CIPHER_SUBMIT_GPU_STATE, &gs),
				    EPERM);
		}

		/* Test 2: SUBMIT_PROCESS_UTIL → EPERM */
		{
			struct cipher_process_util pu;
			memset(&pu, 0, sizeof(pu));
			pu.pid  = getpid();
			pu.tgid = getpid();
			pu.timestamp_ns = monotonic_ns();
			EXPECT_FAIL("Test 2  SUBMIT_PROCESS_UTIL as non-root",
				    ioctl(fd, CIPHER_SUBMIT_PROCESS_UTIL, &pu),
				    EPERM);
		}

		/* Test 3: SUBMIT_LAUNCH_STATS with pid+1 → EPERM (anti-spoof) */
		{
			struct cipher_launch_stats ls;
			memset(&ls, 0, sizeof(ls));
			ls.pid  = getpid() + 1;
			ls.tgid = getpid();
			ls.timestamp_ns = monotonic_ns();
			EXPECT_FAIL("Test 3  SUBMIT_LAUNCH_STATS pid+1 spoof",
				    ioctl(fd, CIPHER_SUBMIT_LAUNCH_STATS, &ls),
				    EPERM);
		}

		/* Test 4: SUBMIT_LAUNCH_STATS with tgid+99 → EPERM */
		{
			struct cipher_launch_stats ls;
			memset(&ls, 0, sizeof(ls));
			ls.pid  = getpid();
			ls.tgid = getpid() + 99;
			ls.timestamp_ns = monotonic_ns();
			EXPECT_FAIL("Test 4  SUBMIT_LAUNCH_STATS tgid+99 spoof",
				    ioctl(fd, CIPHER_SUBMIT_LAUNCH_STATS, &ls),
				    EPERM);
		}

		/* Test 5: CIPHER_SNAPSHOT (reserved nr 2) → ENOSYS */
		{
			struct cipher_snapshot snap;
			memset(&snap, 0, sizeof(snap));
			EXPECT_FAIL("Test 5  CIPHER_SNAPSHOT reserved nr 2",
				    ioctl(fd, CIPHER_SNAPSHOT, &snap),
				    ENOSYS);
		}

		/* Test 6: _IO('C', 99) unknown nr → ENOTTY */
		{
			unsigned int unknown_cmd = _IO('C', 99);
			EXPECT_FAIL("Test 6  unknown ioctl _IO('C',99)",
				    ioctl(fd, unknown_cmd, NULL),
				    ENOTTY);
		}
	} else {
		/* ---- sudo tests (7-11) ---- */

		/* Test 7: SUBMIT_GPU_STATE with timestamp_ns=0 → EINVAL */
		{
			struct cipher_gpu_state gs;
			memset(&gs, 0, sizeof(gs));
			gs.timestamp_ns = 0;   /* explicit invalid */
			gs.sm_util_pct  = 50;
			gs.mem_util_pct = 50;
			gs.temp_c       = 65;
			EXPECT_FAIL("Test 7  SUBMIT_GPU_STATE timestamp_ns=0",
				    ioctl(fd, CIPHER_SUBMIT_GPU_STATE, &gs),
				    EINVAL);
		}

		/* Test 8: SUBMIT_GPU_STATE with sm_util_pct=101 → EINVAL */
		{
			struct cipher_gpu_state gs;
			memset(&gs, 0, sizeof(gs));
			gs.timestamp_ns = monotonic_ns();
			gs.sm_util_pct  = 101;   /* invalid */
			gs.mem_util_pct = 50;
			gs.temp_c       = 65;
			EXPECT_FAIL("Test 8  SUBMIT_GPU_STATE sm_util_pct=101",
				    ioctl(fd, CIPHER_SUBMIT_GPU_STATE, &gs),
				    EINVAL);
		}

		/* Test 9: SUBMIT_GPU_STATE with temp_c=300 → EINVAL (new check) */
		{
			struct cipher_gpu_state gs;
			memset(&gs, 0, sizeof(gs));
			gs.timestamp_ns = monotonic_ns();
			gs.sm_util_pct  = 50;
			gs.mem_util_pct = 50;
			gs.temp_c       = 300;   /* invalid */
			EXPECT_FAIL("Test 9  SUBMIT_GPU_STATE temp_c=300",
				    ioctl(fd, CIPHER_SUBMIT_GPU_STATE, &gs),
				    EINVAL);
		}

		/* Test 10: SUBMIT_PROCESS_UTIL sm_util_pct=101 → EINVAL */
		{
			struct cipher_process_util pu;
			memset(&pu, 0, sizeof(pu));
			pu.pid  = getpid();
			pu.tgid = getpid();
			pu.timestamp_ns = monotonic_ns();
			pu.sm_util_pct  = 101;   /* invalid */
			EXPECT_FAIL("Test 10 SUBMIT_PROCESS_UTIL sm_util_pct=101",
				    ioctl(fd, CIPHER_SUBMIT_PROCESS_UTIL, &pu),
				    EINVAL);
		}

		/* Test 11: SUBMIT_PROCESS_UTIL enc_util_pct=200 → EINVAL */
		{
			struct cipher_process_util pu;
			memset(&pu, 0, sizeof(pu));
			pu.pid  = getpid();
			pu.tgid = getpid();
			pu.timestamp_ns = monotonic_ns();
			pu.sm_util_pct  = 50;
			pu.enc_util_pct = 200;   /* invalid */
			EXPECT_FAIL("Test 11 SUBMIT_PROCESS_UTIL enc_util_pct=200",
				    ioctl(fd, CIPHER_SUBMIT_PROCESS_UTIL, &pu),
				    EINVAL);
		}
	}

	close(fd);
	if (failures) {
		fprintf(stderr, "negative: %d FAILURE(S)\n", failures);
		return 1;
	}
	printf("negative: all PASS\n");
	return 0;
}
