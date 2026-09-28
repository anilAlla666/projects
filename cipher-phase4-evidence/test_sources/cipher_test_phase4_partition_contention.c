/* SPDX-License-Identifier: GPL-2.0
 * cipher_test_phase4_partition_contention.c
 *
 * Drives CIPHER_REQUEST_SM_PARTITION from N threads concurrently. Each thread
 * has already grabbed its full 8-slot allocation, so every per-iteration call
 * hits the idempotent fast path (cache hit). This measures the COST OF THE
 * SHARED-PATH lookup itself under heavy contention — that's the metric the
 * lock-free design has to beat.
 *
 * Output: aggregate wall_time, ops/s, p50/p90/p99/max/mean (ns per ioctl), and
 * a ratio against the single-thread p99 baseline supplied via CIPHER_SINGLE_NS
 * env var (defaults to 310 ns from the Phase 4.1 perf test).
 *
 * Gate the user supplied in T4.2.1: p99 contended <= 5x single-thread p99.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <stdint.h>

#include "/home/ubuntu/cipher_kmod/cipher_ioctl.h"

#define THREADS_DEFAULT  33
#define ITERS_DEFAULT    30000

static __u64 now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (__u64)ts.tv_sec * 1000000000ULL + (__u64)ts.tv_nsec;
}

static int cmp_u64(const void *a, const void *b)
{
	__u64 x = *(const __u64 *)a, y = *(const __u64 *)b;
	return (x < y) ? -1 : (x > y);
}

struct worker_arg {
	int        fd;
	int        iters;
	int        hint;
	pthread_barrier_t *bar;
	__u64     *samples;
	const char *tid_str;
};

static pid_t mytid(void) { return (pid_t)syscall(SYS_gettid); }

static int register_self(int fd, const char *tid)
{
	struct cipher_register_tenant rt;
	memset(&rt, 0, sizeof(rt));
	rt.pid  = mytid();
	rt.tgid = getpid();
	strncpy(rt.tenant_id, tid, CIPHER_TENANT_ID_LEN - 1);
	return ioctl(fd, CIPHER_REGISTER_TENANT, &rt);
}

static void *worker(void *arg)
{
	struct worker_arg *w = (struct worker_arg *)arg;
	int fd;
	struct cipher_partition_request pr;
	int i;

	/* Per-thread fd opens (CIPHER_PER_THREAD_FD=1) eliminate kernel-side
	 * struct file serialization on a single shared fd. */
	if (w->fd >= 0) {
		fd = w->fd;
	} else {
		fd = open("/dev/cipher", O_RDWR);
		if (fd < 0)
			return NULL;
	}

	/* Each thread has its own kernel task->pid (tid) so it registers + claims
	 * its own slots first. After this, the ioctl hits the idempotent fast
	 * path on every call. */
	if (register_self(fd, w->tid_str) != 0)
		goto out_close;
	memset(&pr, 0, sizeof(pr));
	pr.hint_partitions = w->hint;
	(void)ioctl(fd, CIPHER_REQUEST_SM_PARTITION, &pr);

	pthread_barrier_wait(w->bar);

	for (i = 0; i < w->iters; i++) {
		__u64 t0 = now_ns();
		pr.hint_partitions = w->hint;
		(void)ioctl(fd, CIPHER_REQUEST_SM_PARTITION, &pr);
		__u64 t1 = now_ns();
		w->samples[i] = t1 - t0;
	}
out_close:
	if (w->fd < 0)
		close(fd);
	return NULL;
}

int main(int argc, char **argv)
{
	int threads = THREADS_DEFAULT;
	int iters   = ITERS_DEFAULT;
	int hint    = 8;
	int fd;

	if (argc > 1) threads = atoi(argv[1]);
	if (argc > 2) iters   = atoi(argv[2]);
	if (argc > 3) hint    = atoi(argv[3]);
	{
		const char *envh = getenv("CIPHER_HINT");
		if (envh && *envh) hint = atoi(envh);
	}
	if (threads < 1 || iters < 1 || hint < 1 || hint > 8) {
		fprintf(stderr, "usage: %s [threads] [iters] [hint(1..8)]\n", argv[0]);
		return 2;
	}

	__u64 single_ns_baseline = 310;
	const char *env = getenv("CIPHER_SINGLE_NS");
	if (env && *env) single_ns_baseline = strtoull(env, NULL, 10);

	int per_thread_fd = 0;
	const char *envf = getenv("CIPHER_PER_THREAD_FD");
	if (envf && *envf && atoi(envf) != 0) per_thread_fd = 1;

	if (per_thread_fd) {
		fd = -1;  /* worker opens its own */
	} else {
		fd = open("/dev/cipher", O_RDWR);
		if (fd < 0) { fprintf(stderr, "FAIL open: %s\n", strerror(errno)); return 1; }
	}

	pthread_t          *ts   = calloc(threads, sizeof(*ts));
	struct worker_arg  *args = calloc(threads, sizeof(*args));
	__u64             **bufs = calloc(threads, sizeof(*bufs));
	if (!ts || !args || !bufs) { fprintf(stderr, "alloc FAIL\n"); return 2; }
	for (int k = 0; k < threads; k++) {
		bufs[k] = malloc((size_t)iters * sizeof(__u64));
		if (!bufs[k]) { fprintf(stderr, "alloc FAIL\n"); return 2; }
	}

	pthread_barrier_t bar;
	pthread_barrier_init(&bar, NULL, threads + 1);

	for (int k = 0; k < threads; k++) {
		args[k].fd      = fd;
		args[k].iters   = iters;
		args[k].hint    = hint;
		args[k].bar     = &bar;
		args[k].samples = bufs[k];
		char *name = malloc(64);
		snprintf(name, 64, "cont-%d", k);
		args[k].tid_str = name;
		pthread_create(&ts[k], NULL, worker, &args[k]);
	}

	pthread_barrier_wait(&bar);
	__u64 t_start = now_ns();
	for (int k = 0; k < threads; k++) pthread_join(ts[k], NULL);
	__u64 t_end = now_ns();
	double wall_s = (double)(t_end - t_start) / 1.0e9;
	long long total = (long long)threads * (long long)iters;

	/* aggregate */
	__u64 *all = malloc((size_t)total * sizeof(__u64));
	if (!all) { fprintf(stderr, "alloc FAIL\n"); return 2; }
	size_t off = 0;
	for (int k = 0; k < threads; k++) {
		memcpy(all + off, bufs[k], (size_t)iters * sizeof(__u64));
		off += iters;
		free(bufs[k]);
	}
	qsort(all, (size_t)total, sizeof(__u64), cmp_u64);
	__u64 p50  = all[(size_t)((double)total * 0.50)];
	__u64 p90  = all[(size_t)((double)total * 0.90)];
	__u64 p99  = all[(size_t)((double)total * 0.99)];
	__u64 mx   = all[total - 1];
	double sum = 0;
	for (long long i = 0; i < total; i++) sum += (double)all[i];
	double mean = sum / (double)total;

	double ratio = (double)p99 / (double)single_ns_baseline;

	printf("CIPHER_REQUEST_SM_PARTITION contended  (threads=%d, iters/thread=%d, hint=%d, per_thread_fd=%d, total=%lld)\n",
	       threads, iters, hint, per_thread_fd, total);
	printf("  wall_time    %.2f s\n", wall_s);
	printf("  ops/s        %.0f\n", (double)total / wall_s);
	printf("  p50          %llu ns\n", (unsigned long long)p50);
	printf("  p90          %llu ns\n", (unsigned long long)p90);
	printf("  p99          %llu ns\n", (unsigned long long)p99);
	printf("  max          %llu ns\n", (unsigned long long)mx);
	printf("  mean         %.0f ns\n", mean);
	printf("\nGate: p99 contended vs p99 single-thread baseline (%llu ns).\n",
	       (unsigned long long)single_ns_baseline);
	printf("  ratio        %.1fx\n", ratio);
	if (ratio <= 5.0)
		printf("  verdict      PASS (<= 5x)\n");
	else if (ratio <= 10.0)
		printf("  verdict      MARGINAL (5x..10x)\n");
	else
		printf("  verdict      FAIL — needs further work\n");

	free(all);
	free(ts); free(args); free(bufs);
	close(fd);
	return 0;
}
