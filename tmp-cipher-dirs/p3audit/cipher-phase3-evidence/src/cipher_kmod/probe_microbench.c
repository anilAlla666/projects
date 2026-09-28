/* SPDX-License-Identifier: GPL-2.0
 *
 * probe_microbench.c — Phase 1.5.2 Workload D
 *
 * Tight loop calling NV_ESC_CHECK_VERSION_STR through /dev/nvidiactl.
 * Measures wall-clock per ioctl with CLOCK_MONOTONIC. We do NOT care
 * whether the ioctl returns success — we care that it reaches
 * nvidia_unlocked_ioctl in the kernel so cipher_kmod's kprobe fires.
 *
 * Build: gcc -O2 -o probe_microbench probe_microbench.c
 * Run:   ./probe_microbench [iterations]   (default 100000)
 * Out:   one JSON line on stdout
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <time.h>

#define NV_IOCTL_MAGIC               'F'
#define NV_ESC_CHECK_VERSION_STR_NR  210

/* Match NVIDIA's nv_ioctl_rm_api_version_t (typical layout). Exact size
 * doesn't change the measurement: even if the driver rejects internally,
 * the kprobe fires on entry. */
struct nv_version_payload {
	char vstr[64];
	int  reply;
	int  cmp;
};

#define NV_ESC_CHECK_VERSION_STR \
	_IOWR(NV_IOCTL_MAGIC, NV_ESC_CHECK_VERSION_STR_NR, struct nv_version_payload)

#define WARMUP 200

int main(int argc, char **argv)
{
	int n = 100000;
	int fd, i;
	long long ns;
	double ns_per;
	struct nv_version_payload data;
	struct timespec t0, t1;

	if (argc > 1)
		n = atoi(argv[1]);
	if (n <= 0) {
		fprintf(stderr, "iterations must be > 0\n");
		return 2;
	}

	fd = open("/dev/nvidiactl", O_RDWR);
	if (fd < 0) {
		fprintf(stderr, "open /dev/nvidiactl failed: %s\n",
			strerror(errno));
		return 1;
	}

	memset(&data, 0, sizeof(data));
	strncpy(data.vstr, "0.0.0", sizeof(data.vstr) - 1);

	for (i = 0; i < WARMUP; i++)
		(void)ioctl(fd, NV_ESC_CHECK_VERSION_STR, &data);

	clock_gettime(CLOCK_MONOTONIC, &t0);
	for (i = 0; i < n; i++)
		(void)ioctl(fd, NV_ESC_CHECK_VERSION_STR, &data);
	clock_gettime(CLOCK_MONOTONIC, &t1);

	ns = (long long)(t1.tv_sec - t0.tv_sec) * 1000000000LL +
	     (t1.tv_nsec - t0.tv_nsec);
	ns_per = (double)ns / (double)n;

	printf("{\"iterations\": %d, \"total_ns\": %lld, \"ns_per_ioctl\": %.3f}\n",
	       n, ns, ns_per);

	close(fd);
	return 0;
}
