/* SPDX-License-Identifier: GPL-2.0
 * cipher_test_phase4_snapshot.c — exercise CIPHER_GET_TENANT_SNAPSHOT (ioctl nr 8).
 *
 * Scenarios:
 *   Test 1  Lookup by target_pid: rc==0, tenant_id_str matches, snapshot.pid==target_pid,
 *           tenant_session_fp != 0, tenant_handle_u32 != 0.
 *   Test 2  Lookup by target_tenant_id (non-empty string wins): rc==0, snapshot.pid matches.
 *   Test 3  Unknown pid -> -ENOENT.
 *   Test 4  No selector (both empty / zero) -> -ENOENT.
 *   Test 5  Unknown tenant_id -> -ENOENT.
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

#include "/home/ubuntu/cipher_kmod/cipher_ioctl.h"

static pid_t mytid(void)
{
	return (pid_t)syscall(SYS_gettid);
}

int main(void)
{
	int fd, rc, failures = 0, passes = 0;
	struct cipher_register_tenant rt;
	struct cipher_tenant_snapshot_query *q;
	pid_t target_pid;
	const char *tid_str = "snapshot-test-tenant";

	q = calloc(1, sizeof(*q));
	if (!q) { fprintf(stderr, "calloc FAIL\n"); return 2; }

	fd = open("/dev/cipher", O_RDWR);
	if (fd < 0) {
		fprintf(stderr, "FAIL  open /dev/cipher: %s\n", strerror(errno));
		free(q);
		return 1;
	}
	target_pid = mytid();

	memset(&rt, 0, sizeof(rt));
	rt.pid  = target_pid;
	rt.tgid = getpid();
	strncpy(rt.tenant_id, tid_str, CIPHER_TENANT_ID_LEN - 1);
	if (ioctl(fd, CIPHER_REGISTER_TENANT, &rt) != 0) {
		fprintf(stderr, "FAIL  REGISTER_TENANT errno=%d %s\n", errno, strerror(errno));
		close(fd); free(q);
		return 1;
	}

	/* Also push a SUBMIT_LAUNCH_STATS so cipher_pid_stats has activity. */
	{
		struct cipher_launch_stats ls;
		memset(&ls, 0, sizeof(ls));
		ls.pid  = target_pid;
		ls.tgid = getpid();
		ls.timestamp_ns = 1;
		ls.launches_total = 7;
		ls.grid_ops_total = 49;
		(void)ioctl(fd, CIPHER_SUBMIT_LAUNCH_STATS, &ls);
	}

	/* --- Test 1: by pid --- */
	memset(q, 0, sizeof(*q));
	q->target_pid = target_pid;
	errno = 0;
	rc = ioctl(fd, CIPHER_GET_TENANT_SNAPSHOT, q);
	if (rc == 0)            { printf("PASS  Test 1  by pid: ioctl rc==0\n"); passes++; }
	else                    { printf("FAIL  Test 1  ioctl rc=%d errno=%d %s\n", rc, errno, strerror(errno)); failures++; }
	if (rc == 0 && strncmp(q->snapshot.tenant_id_str, tid_str, strlen(tid_str)) == 0)
				{ printf("PASS  Test 1  tenant_id_str matches\n"); passes++; }
	else if (rc == 0)       { printf("FAIL  Test 1  tenant_id_str='%s' want='%s'\n", q->snapshot.tenant_id_str, tid_str); failures++; }
	if (rc == 0 && (pid_t)q->snapshot.pid == target_pid)
				{ printf("PASS  Test 1  snapshot.pid == target_pid\n"); passes++; }
	else if (rc == 0)       { printf("FAIL  Test 1  snapshot.pid=%u target=%d\n", q->snapshot.pid, target_pid); failures++; }
	if (rc == 0 && q->snapshot.tenant_session_fp != 0)
				{ printf("PASS  Test 1  tenant_session_fp non-zero\n"); passes++; }
	else if (rc == 0)       { printf("FAIL  Test 1  tenant_session_fp == 0\n"); failures++; }
	if (rc == 0 && q->snapshot.tenant_handle_u32 != 0)
				{ printf("PASS  Test 1  tenant_handle_u32 non-zero\n"); passes++; }
	else if (rc == 0)       { printf("FAIL  Test 1  tenant_handle_u32 == 0\n"); failures++; }

	/* --- Test 2: by tenant_id (non-empty string wins over pid) --- */
	memset(q, 0, sizeof(*q));
	q->target_pid = 0xDEADBEEFu; /* should be ignored */
	strncpy(q->target_tenant_id, tid_str, CIPHER_TENANT_ID_LEN - 1);
	errno = 0;
	rc = ioctl(fd, CIPHER_GET_TENANT_SNAPSHOT, q);
	if (rc == 0)            { printf("PASS  Test 2  by id: ioctl rc==0\n"); passes++; }
	else                    { printf("FAIL  Test 2  ioctl rc=%d errno=%d %s\n", rc, errno, strerror(errno)); failures++; }
	if (rc == 0 && (pid_t)q->snapshot.pid == target_pid)
				{ printf("PASS  Test 2  snapshot.pid matches our pid\n"); passes++; }
	else if (rc == 0)       { printf("FAIL  Test 2  snapshot.pid=%u target=%d\n", q->snapshot.pid, target_pid); failures++; }

	/* --- Test 3: unknown pid -> ENOENT --- */
	memset(q, 0, sizeof(*q));
	q->target_pid = 0x7fffffff;
	errno = 0;
	rc = ioctl(fd, CIPHER_GET_TENANT_SNAPSHOT, q);
	if (rc != 0 && errno == ENOENT)
		{ printf("PASS  Test 3  unknown pid -> ENOENT                      (errno=%d %s)\n", errno, strerror(errno)); passes++; }
	else
		{ printf("FAIL  Test 3  rc=%d errno=%d %s\n", rc, errno, strerror(errno)); failures++; }

	/* --- Test 4: no selector -> ENOENT --- */
	memset(q, 0, sizeof(*q));
	errno = 0;
	rc = ioctl(fd, CIPHER_GET_TENANT_SNAPSHOT, q);
	if (rc != 0 && errno == ENOENT)
		{ printf("PASS  Test 4  no selector -> ENOENT                      (errno=%d %s)\n", errno, strerror(errno)); passes++; }
	else
		{ printf("FAIL  Test 4  rc=%d errno=%d %s\n", rc, errno, strerror(errno)); failures++; }

	/* --- Test 5: unknown tenant_id -> ENOENT --- */
	memset(q, 0, sizeof(*q));
	strncpy(q->target_tenant_id, "no-such-tenant-anywhere", CIPHER_TENANT_ID_LEN - 1);
	errno = 0;
	rc = ioctl(fd, CIPHER_GET_TENANT_SNAPSHOT, q);
	if (rc != 0 && errno == ENOENT)
		{ printf("PASS  Test 5  unknown tenant_id -> ENOENT                (errno=%d %s)\n", errno, strerror(errno)); passes++; }
	else
		{ printf("FAIL  Test 5  rc=%d errno=%d %s\n", rc, errno, strerror(errno)); failures++; }

	close(fd);
	free(q);
	if (failures) {
		fprintf(stderr, "snapshot: %d FAILURE(S) (%d PASS)\n", failures, passes);
		return 1;
	}
	printf("snapshot: all PASS\n");
	return 0;
}
