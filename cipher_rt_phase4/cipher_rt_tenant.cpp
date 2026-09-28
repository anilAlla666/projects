/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_rt_tenant.cpp — Phase 4 actuator-side API implementation.
 *
 * STAGED: not yet in cipher_rt build. Wired in T4.1+ after cipher_kmod
 * 0.4.0 ships ioctl nr 8.
 *
 * Mode 3 cache lives in a __thread struct so each actuator thread gets
 * its own copy without locking. Refresh is one ioctl call; reads are
 * pointer return + age check.
 */
#define _GNU_SOURCE
#include "cipher_rt_tenant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>

/* IOCTL number — keep in sync with cipher_kmod/cipher_ioctl.h nr 8 (T4.1.2). */
#define CIPHER_IOCTL_MAGIC  'C'
#define CIPHER_GET_TENANT_SNAPSHOT \
	_IOWR(CIPHER_IOCTL_MAGIC, 8, struct cipher_tenant_snapshot_query)

/* ----- module-local state -----
 *
 * Per-thread fd to /dev/cipher (PHASE_4_BACKLOG.md B1, fixed 2026-05-13).
 *
 * Each thread lazily opens its own /dev/cipher fd on first ioctl. This is
 * binding per PHASE_4_ARCHITECTURE.md "Binding deployment requirement:
 * per-thread /dev/cipher fd (T4.0.9.D)": sharing a single fd across
 * threads inflates p99 by ~50x at 33-thread concurrency due to the VFS
 * single-struct-file throughput cap (~8 M ops/s). Per-thread fds give
 * near-linear scaling to ~76 M ops/s at 33 threads.
 *
 * Lifecycle:
 *   - t_cipher_fd is zero-initialized per thread (TLS) -> set to -1 on
 *     first observation via t_cipher_fd_initialized so we don't open
 *     unconditionally on threads that never call us.
 *   - cipher_rt_tenant_open() is idempotent per thread.
 *   - cipher_rt_tenant_close() closes the calling thread's fd only.
 *   - Threads that don't explicitly close leak one fd at thread-exit
 *     time; acceptable for the actuator's per-stream lifetime (CUDA
 *     stream lifetime is typically the process). A pthread_cleanup
 *     wrapper can be added if a specific consumer needs deterministic
 *     close on thread exit.
 */
static __thread int t_cipher_fd             = -1;
static __thread int t_cipher_fd_initialized = 0;  /* 0 means "fresh TLS,
                                                     interpret t_cipher_fd
                                                     as not-yet-opened" */

/* Per-thread cached snapshot (Mode 3). */
struct cipher_rt_tls_cache {
	struct cipher_tenant_snapshot snap;
	int                           valid;
	uint64_t                      last_refresh_ns;
};

static __thread struct cipher_rt_tls_cache g_tls_cache;

/* ----- helpers ----- */

static uint64_t monotonic_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static pid_t my_tid(void)
{
	return (pid_t)syscall(SYS_gettid);
}

/* ----- Mode 1: ioctl path ----- */

extern "C" int cipher_rt_tenant_open(void)
{
	if (t_cipher_fd_initialized && t_cipher_fd >= 0) return 0;
	t_cipher_fd = open("/dev/cipher", O_RDWR);
	t_cipher_fd_initialized = 1;
	if (t_cipher_fd < 0) {
		fprintf(stderr, "[cipher_rt:tenant] open(/dev/cipher) failed: %s\n",
		        strerror(errno));
		return -errno;
	}
	return 0;
}

extern "C" void cipher_rt_tenant_close(void)
{
	if (t_cipher_fd_initialized && t_cipher_fd >= 0) {
		close(t_cipher_fd);
		t_cipher_fd = -1;
	}
	t_cipher_fd_initialized = 0;
}

/* Lazy per-thread open: query functions auto-open the calling thread's fd
 * if cipher_rt_tenant_open() hasn't been called explicitly on this thread.
 * Matches the original "open once, query many" semantic but per-thread.
 * Returns the (possibly newly opened) fd, or -1 on failure. */
static inline int cipher_rt_tenant_fd_ensure(void)
{
	if (t_cipher_fd_initialized && t_cipher_fd >= 0)
		return t_cipher_fd;
	if (cipher_rt_tenant_open() != 0)
		return -1;
	return t_cipher_fd;
}

extern "C" int cipher_rt_tenant_query_by_pid(pid_t pid,
                                  struct cipher_tenant_snapshot *out)
{
	struct cipher_tenant_snapshot_query q;
	int fd, rc;

	if (!out)            return -EINVAL;
	fd = cipher_rt_tenant_fd_ensure();
	if (fd < 0)          return -EBADF;

	memset(&q, 0, sizeof(q));
	q.target_pid = (uint32_t)pid;

	errno = 0;
	rc = ioctl(fd, CIPHER_GET_TENANT_SNAPSHOT, &q);
	if (rc != 0) return -errno;

	memcpy(out, &q.snapshot, sizeof(*out));
	return 0;
}

extern "C" int cipher_rt_tenant_query_by_id(const char *tenant_id,
                                 struct cipher_tenant_snapshot *out)
{
	struct cipher_tenant_snapshot_query q;
	int fd, rc;

	if (!out || !tenant_id) return -EINVAL;
	fd = cipher_rt_tenant_fd_ensure();
	if (fd < 0)             return -EBADF;

	memset(&q, 0, sizeof(q));
	q.target_pid = 0;  /* signal: use tenant_id */
	strncpy(q.target_tenant_id, tenant_id, sizeof(q.target_tenant_id) - 1);

	errno = 0;
	rc = ioctl(fd, CIPHER_GET_TENANT_SNAPSHOT, &q);
	if (rc != 0) return -errno;

	memcpy(out, &q.snapshot, sizeof(*out));
	return 0;
}

/* ----- Mode 2: /proc poll ----- */

/* Light parser for /proc/cipher/stats. Returns count of tenants populated.
 * Reads the per-PID table (top 16 by total), copies what we can directly
 * (tenant_id, pid, tgid, sm_util_pct, mem_util_pct, launches_total).
 * Other fields fall back to 0 — caller can issue Mode 1 query for full. */
extern "C" int cipher_rt_tenant_enumerate(struct cipher_tenant_snapshot *out,
                                          int max)
{
	FILE *fp;
	char  line[1024];
	int   n = 0;
	int   in_per_pid = 0;

	if (!out || max <= 0) return -EINVAL;

	fp = fopen("/proc/cipher/stats", "r");
	if (!fp) return -errno;

	while (fgets(line, sizeof(line), fp) && n < max) {
		if (strstr(line, "Per-PID summary")) { in_per_pid = 1; continue; }
		if (strstr(line, "Per-TGID summary")) { in_per_pid = 0; continue; }
		if (!in_per_pid)                       continue;
		if (line[0] == '\n' || line[0] == ' ' &&
		    (strstr(line, "PID ") || strstr(line, "---")))
			continue;

		/* Rows look like:
		 *   28474  28474  python3  tenant-A  475 ... 80 7 768 0.49
		 * Split on whitespace, expect 16 fields. */
		struct cipher_tenant_snapshot *s = &out[n];
		char  tenant_id[CIPHER_TENANT_ID_LEN] = {0};
		char  comm[32] = {0};
		unsigned pid, tgid;
		unsigned long long total, launches;
		unsigned sm, mem;
		float    age_tm;

		int f = sscanf(line, " %u %u %31s %63s %llu %*u %*u %*u %*u %*u %*u %*f %u %u %llu %f",
		               &pid, &tgid, comm, tenant_id, &total,
		               &sm, &mem, &launches, &age_tm);
		if (f < 8) continue;
		if (tenant_id[0] == '-') continue;  /* skip cipher-gpustate row */

		memset(s, 0, sizeof(*s));
		strncpy(s->tenant_id_str, tenant_id, sizeof(s->tenant_id_str) - 1);
		s->pid             = pid;
		s->tgid            = tgid;
		s->sm_util_pct     = sm;
		s->mem_util_pct    = mem;
		s->launches_total  = launches;
		/* Other fields filled by Mode 1 ioctl if needed. */
		n++;
	}
	fclose(fp);
	return n;
}

/* ----- Mode 3: thread-local cache ----- */

extern "C" int cipher_rt_tenant_refresh_cached(void)
{
	int rc;

	rc = cipher_rt_tenant_query_by_pid(my_tid(), &g_tls_cache.snap);
	if (rc != 0) {
		g_tls_cache.valid = 0;
		return rc;
	}
	g_tls_cache.valid           = 1;
	g_tls_cache.last_refresh_ns = monotonic_ns();
	return 0;
}

extern "C" const struct cipher_tenant_snapshot *cipher_rt_tenant_cached(void)
{
	if (!g_tls_cache.valid) return NULL;
	return &g_tls_cache.snap;
}

extern "C" uint64_t cipher_rt_tenant_cached_age_ns(void)
{
	if (!g_tls_cache.valid) return UINT64_MAX;
	return monotonic_ns() - g_tls_cache.last_refresh_ns;
}
