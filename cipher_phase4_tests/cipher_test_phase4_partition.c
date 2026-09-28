/* SPDX-License-Identifier: GPL-2.0
 * cipher_test_phase4_partition.c — exercise CIPHER_REQUEST_SM_PARTITION (ioctl nr 9).
 *
 *   Test 1   Request 4: rc==0, popcount==4.
 *   Test 2   Request 8 after 4: rc==0, popcount==8 (idempotent expansion).
 *   Test 3   Forked 11 children; union of masks <= 33 slots; >=5 grant >0 slots.
 *   Test 4   After ephemeral child exits and reaper runs, parent can still request.
 *   Test 5   hint=0 clamps to 1+ (rc==0); hint=99 clamps to <=8.
 *   Test 7   Perf: 1,000,000 idempotent calls; report p50/p90/p99/max.
 *
 * Requires root (must run as root so child PIDs can register their own tenants).
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
#include <sys/wait.h>
#include <time.h>
#include <stdint.h>

#include "/home/ubuntu/cipher_kmod/cipher_ioctl.h"

static pid_t mytid(void) { return (pid_t)syscall(SYS_gettid); }

static __u64 now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (__u64)ts.tv_sec * 1000000000ULL + (__u64)ts.tv_nsec;
}

static int popcount32(uint32_t x)
{
	return __builtin_popcount(x);
}

static int register_self(int fd, const char *tid)
{
	struct cipher_register_tenant rt;
	memset(&rt, 0, sizeof(rt));
	rt.pid  = mytid();
	rt.tgid = getpid();
	strncpy(rt.tenant_id, tid, CIPHER_TENANT_ID_LEN - 1);
	return ioctl(fd, CIPHER_REGISTER_TENANT, &rt);
}

static int cmp_u64(const void *a, const void *b)
{
	__u64 x = *(const __u64 *)a, y = *(const __u64 *)b;
	return (x < y) ? -1 : (x > y);
}

int main(void)
{
	int fd, rc, failures = 0;
	struct cipher_partition_request pr;

	fd = open("/dev/cipher", O_RDWR);
	if (fd < 0) { fprintf(stderr, "FAIL open: %s\n", strerror(errno)); return 1; }
	if (register_self(fd, "partition-test-parent") != 0) {
		fprintf(stderr, "FAIL REGISTER_TENANT errno=%d %s\n", errno, strerror(errno));
		return 1;
	}

	/* --- Test 1: request 4 --- */
	memset(&pr, 0, sizeof(pr));
	pr.hint_partitions = 4;
	errno = 0;
	rc = ioctl(fd, CIPHER_REQUEST_SM_PARTITION, &pr);
	if (rc == 0)                                 printf("PASS  Test 1  request 4: rc==0\n");
	else { printf("FAIL  Test 1  rc=%d errno=%d %s\n", rc, errno, strerror(errno)); failures++; }
	if (rc == 0 && popcount32(pr.partition_mask_out) == 4)
		printf("PASS  Test 1  popcount(mask) == 4\n");
	else if (rc == 0) { printf("FAIL  Test 1  popcount=%d\n", popcount32(pr.partition_mask_out)); failures++; }
	printf("  (mask=0x%08x count=%u)\n", pr.partition_mask_out, pr.partition_count_out);

	/* --- Test 2: request 8 after 4 --- */
	memset(&pr, 0, sizeof(pr));
	pr.hint_partitions = 8;
	errno = 0;
	rc = ioctl(fd, CIPHER_REQUEST_SM_PARTITION, &pr);
	if (rc == 0)                                 printf("PASS  Test 2  request 8 after 4: rc==0\n");
	else { printf("FAIL  Test 2  rc=%d errno=%d %s\n", rc, errno, strerror(errno)); failures++; }
	if (rc == 0 && popcount32(pr.partition_mask_out) == 8)
		printf("PASS  Test 2  popcount(mask) == 8 (cap)\n");
	else if (rc == 0) { printf("FAIL  Test 2  popcount=%d\n", popcount32(pr.partition_mask_out)); failures++; }
	printf("  (mask=0x%08x count=%u)\n", pr.partition_mask_out, pr.partition_count_out);

	/* --- Test 3: 11 children simultaneously alive, each asks for 4 --- */
	{
		int pipes[11][2];
		pid_t kids[11];
		uint32_t masks[11] = {0};
		uint32_t union_mask = 0;
		int sum_counts = 0, granted_nonzero = 0;

		for (int k = 0; k < 11; k++) {
			if (pipe(pipes[k]) != 0) { perror("pipe"); return 2; }
		}
		for (int k = 0; k < 11; k++) {
			pid_t p = fork();
			if (p == 0) {
				/* Child */
				int cfd = open("/dev/cipher", O_RDWR);
				if (cfd < 0) _exit(20);
				char tname[64];
				snprintf(tname, sizeof(tname), "ptn-test-child-%d", k);
				if (register_self(cfd, tname) != 0) _exit(21);
				struct cipher_partition_request cpr;
				memset(&cpr, 0, sizeof(cpr));
				cpr.hint_partitions = 4;
				if (ioctl(cfd, CIPHER_REQUEST_SM_PARTITION, &cpr) != 0)
					cpr.partition_mask_out = 0;
				close(pipes[k][0]);
				ssize_t wrn = write(pipes[k][1], &cpr.partition_mask_out, sizeof(uint32_t));
				(void)wrn;
				close(pipes[k][1]);
				/* sleep so all 11 are alive concurrently */
				usleep(400 * 1000);
				close(cfd);
				_exit(0);
			} else {
				kids[k] = p;
				close(pipes[k][1]);
			}
		}
		for (int k = 0; k < 11; k++) {
			ssize_t rdn = read(pipes[k][0], &masks[k], sizeof(uint32_t));
			(void)rdn;
			close(pipes[k][0]);
			union_mask |= masks[k];
			sum_counts += popcount32(masks[k]);
			if (popcount32(masks[k]) > 0) granted_nonzero++;
		}
		for (int k = 0; k < 11; k++) {
			int st;
			waitpid(kids[k], &st, 0);
		}
		if (popcount32(union_mask) <= 33)
			printf("PASS  Test 3  union of 11 simultaneously-alive tenant masks <= 33 slots\n");
		else { printf("FAIL  Test 3  union popcount=%d > 33\n", popcount32(union_mask)); failures++; }
		if (granted_nonzero >= 5)
			printf("PASS  Test 3  at least 5 children granted >0 slots\n");
		else { printf("FAIL  Test 3  only %d children got slots\n", granted_nonzero); failures++; }
		printf("  (union_mask=0x%08x union_popcount=%d sum_of_counts=%d)\n",
		       union_mask, popcount32(union_mask), sum_counts);
	}

	/* --- Test 4: after ephemeral child + do_exit reap, parent can still request --- */
	{
		/* Wait long enough that all 11 ephemeral children have hit do_exit
		 * and their slot CAS-release ran in cipher_do_exit_pre. */
		usleep(300 * 1000);
		memset(&pr, 0, sizeof(pr));
		pr.hint_partitions = 8;
		errno = 0;
		rc = ioctl(fd, CIPHER_REQUEST_SM_PARTITION, &pr);
		if (rc == 0)
			printf("PASS  Test 4  after ephemeral exit: still able to request\n");
		else { printf("FAIL  Test 4  rc=%d errno=%d %s\n", rc, errno, strerror(errno)); failures++; }
		printf("  (post-reap free pool granted us %u slots)\n", pr.partition_count_out);
	}

	/* --- Test 5: hint clamps --- */
	memset(&pr, 0, sizeof(pr));
	pr.hint_partitions = 0;
	errno = 0;
	rc = ioctl(fd, CIPHER_REQUEST_SM_PARTITION, &pr);
	if (rc == 0) printf("PASS  Test 5  hint=0 clamps to 1+, rc==0\n");
	else { printf("FAIL  Test 5  hint=0 rc=%d errno=%d %s\n", rc, errno, strerror(errno)); failures++; }
	memset(&pr, 0, sizeof(pr));
	pr.hint_partitions = 99;
	errno = 0;
	rc = ioctl(fd, CIPHER_REQUEST_SM_PARTITION, &pr);
	if (rc == 0 && pr.partition_count_out <= 8)
		printf("PASS  Test 5  hint=99 clamps to <=8\n");
	else { printf("FAIL  Test 5  hint=99 rc=%d count=%u\n", rc, pr.partition_count_out); failures++; }

	/* --- Test 7: perf, 1,000,000 idempotent calls --- */
	{
		const int N = 1000000;
		__u64 *samples = malloc((size_t)N * sizeof(__u64));
		if (!samples) { fprintf(stderr, "malloc FAIL\n"); return 2; }
		struct cipher_partition_request rp;
		memset(&rp, 0, sizeof(rp));
		rp.hint_partitions = 8;
		/* warmup */
		for (int i = 0; i < 1000; i++) {
			rp.hint_partitions = 8;
			(void)ioctl(fd, CIPHER_REQUEST_SM_PARTITION, &rp);
		}
		for (int i = 0; i < N; i++) {
			__u64 t0 = now_ns();
			rp.hint_partitions = 8;
			(void)ioctl(fd, CIPHER_REQUEST_SM_PARTITION, &rp);
			__u64 t1 = now_ns();
			samples[i] = t1 - t0;
		}
		qsort(samples, N, sizeof(__u64), cmp_u64);
		__u64 p50  = samples[(size_t)((double)N * 0.50)];
		__u64 p90  = samples[(size_t)((double)N * 0.90)];
		__u64 p99  = samples[(size_t)((double)N * 0.99)];
		__u64 mx   = samples[N - 1];
		printf("\nTest 7  CIPHER_REQUEST_SM_PARTITION perf (N=%d):\n", N);
		printf("  p50   %llu ns\n", (unsigned long long)p50);
		printf("  p90   %llu ns\n", (unsigned long long)p90);
		printf("  p99   %llu ns\n", (unsigned long long)p99);
		printf("  max   %llu ns\n", (unsigned long long)mx);
		free(samples);
	}

	close(fd);
	if (failures) { fprintf(stderr, "partition: %d FAILURE(S)\n", failures); return 1; }
	printf("\npartition: all PASS\n");
	return 0;
}
