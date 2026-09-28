// CP 5.4 Step 1.2 Phase 2 — kmod isolation tests.
// Direct ioctl() exercise of /dev/cipher against the CP 5.4 kmod (90103f82).
// 6 tests: legacy nr-9 deactivation, ALLOCATE/FREE/QUERY, pool resize,
// do_exit reaper, disjointness, concurrent stress. Distinct "tenants" are
// distinct PIDs (fork) — the kmod ledger keys on current->pid.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/wait.h>

#include "cipher_ioctl.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, name) do { \
	if (cond) { printf("  PASS  %s\n", name); g_pass++; } \
	else      { printf("  FAIL  %s\n", name); g_fail++; } \
} while (0)

static int dev_open(void)
{
	int fd = open("/dev/cipher", O_RDWR);
	if (fd < 0) { perror("open /dev/cipher"); exit(2); }
	return fd;
}

static int do_alloc(int fd, unsigned qos, unsigned sm_count,
                    unsigned *mask, unsigned *count)
{
	struct cipher_cp54_allocate a;
	int rc;
	memset(&a, 0, sizeof(a));
	a.qos_class = qos;
	a.sm_count = sm_count;
	rc = ioctl(fd, CIPHER_CP54_ALLOCATE, &a);
	if (mask)  *mask  = a.grp_mask_out;
	if (count) *count = a.grp_count_out;
	return rc;
}

static int do_query(int fd, struct cipher_cp54_query *q)
{
	memset(q, 0, sizeof(*q));
	return ioctl(fd, CIPHER_CP54_QUERY, q);
}

/* Fork a holder: child opens /dev/cipher, ALLOCATEs, reports (mask,count)
 * up the pipe, then blocks until released. Returns child pid; *rel is the
 * write-fd to release it. */
static pid_t spawn_holder(unsigned qos, unsigned sm_count,
                          unsigned *out_mask, unsigned *out_count, int *rel)
{
	int up[2], down[2];
	pid_t p;
	unsigned res[2];

	if (pipe(up) || pipe(down)) { perror("pipe"); exit(2); }
	p = fork();
	if (p == 0) {
		char c;
		int fd;
		unsigned m = 0xFFFFFFFFu, n = 0;
		close(up[0]); close(down[1]);
		fd = dev_open();
		if (do_alloc(fd, qos, sm_count, &m, &n) != 0) { m = 0xFFFFFFFFu; n = 0; }
		res[0] = m; res[1] = n;
		if (write(up[1], res, sizeof(res)) < 0) _exit(3);
		(void)!read(down[0], &c, 1);   /* block until released */
		ioctl(fd, CIPHER_CP54_FREE);
		_exit(0);
	}
	close(up[1]); close(down[0]);
	if (read(up[0], res, sizeof(res)) != (ssize_t)sizeof(res)) {
		res[0] = 0xFFFFFFFFu; res[1] = 0;
	}
	close(up[0]);
	*out_mask = res[0]; *out_count = res[1];
	*rel = down[1];
	return p;
}

static void release_holder(int rel, pid_t p)
{
	char c = 'x';
	(void)!write(rel, &c, 1);
	close(rel);
	waitpid(p, NULL, 0);
}

int main(void)
{
	int fd;
	struct cipher_cp54_query q;
	int rc;

	printf("CP 5.4 Phase 2 — kmod isolation tests (/dev/cipher)\n\n");

	/* ---- Test 1: legacy nr-9 deactivation ---- */
	printf("Test 1 — legacy nr-9 deactivated:\n");
	{
		struct cipher_partition_request pr;
		memset(&pr, 0, sizeof(pr));
		pr.hint_partitions = 4;
		fd = dev_open();
		errno = 0;
		rc = ioctl(fd, CIPHER_REQUEST_SM_PARTITION, &pr);
		CHECK(rc < 0 && errno == ENOSYS,
		      "nr-9 REQUEST_SM_PARTITION returns -ENOSYS");
		close(fd);
	}

	/* ---- Test 2: ALLOCATE / FREE / QUERY basic correctness ---- */
	printf("Test 2 — ALLOCATE / FREE / QUERY:\n");
	{
		unsigned mask = 0, count = 0;
		fd = dev_open();
		rc = do_alloc(fd, CIPHER_CP54_QOS_PARTITION, 16, &mask, &count);
		CHECK(rc == 0 && count == 2, "ALLOCATE PARTITION 16 SMs -> 2 groups");
		rc = do_query(fd, &q);
		CHECK(rc == 0 && q.my_grp_mask == mask &&
		      q.my_qos_class == CIPHER_CP54_QOS_PARTITION,
		      "QUERY reflects the allocation");
		rc = ioctl(fd, CIPHER_CP54_FREE);
		CHECK(rc == 0, "FREE returns 0");
		rc = do_query(fd, &q);
		CHECK(rc == 0 && q.my_grp_mask == 0 && q.free_grp_count == 16,
		      "QUERY after FREE: no groups held, all 16 free");
		close(fd);
	}

	/* ---- Test 3: pool resize via partition allocation ---- */
	printf("Test 3 — pool resize:\n");
	{
		unsigned pmask = 0, pcount = 0, amask = 0, acount = 0;
		int rel;
		pid_t pool = spawn_holder(CIPHER_CP54_QOS_POOL, 0, &pmask, &pcount, &rel);
		CHECK(pcount == 16, "POOL holder claims all 16 residual groups");
		fd = dev_open();
		rc = do_alloc(fd, CIPHER_CP54_QOS_PARTITION, 16, &amask, &acount);
		CHECK(rc == 0 && acount == 2,
		      "PARTITION 16 SMs allocates -> 2 groups (pool shrunk)");
		rc = do_query(fd, &q);
		CHECK(rc == 0 && q.pool_grp_count == 14 && q.n_partitions == 1 &&
		      q.my_grp_mask == amask,
		      "QUERY: pool now 14 groups, 1 partition tenant, mask matches");
		ioctl(fd, CIPHER_CP54_FREE);
		close(fd);
		release_holder(rel, pool);
		fd = dev_open();
		do_query(fd, &q);
		CHECK(q.free_grp_count == 16, "after release: all 16 groups free again");
		close(fd);
	}

	/* ---- Test 4: do_exit reaper ---- */
	printf("Test 4 — do_exit reaper:\n");
	{
		pid_t c = fork();
		if (c == 0) {
			int cfd = dev_open();
			do_alloc(cfd, CIPHER_CP54_QOS_PARTITION, 24, NULL, NULL);
			_exit(0);            /* exit WITHOUT FREE — reaper must reclaim */
		}
		waitpid(c, NULL, 0);
		fd = dev_open();
		rc = do_query(fd, &q);
		CHECK(rc == 0 && q.free_grp_count == 16 && q.n_partitions == 0,
		      "child exited without FREE -> reaper reclaimed its 3 groups");
		close(fd);
	}

	/* ---- Test 5: disjointness invariant ---- */
	printf("Test 5 — disjointness:\n");
	{
		unsigned m1 = 0, c1 = 0, m2 = 0, c2 = 0;
		int r1, r2;
		pid_t h1 = spawn_holder(CIPHER_CP54_QOS_PARTITION, 16, &m1, &c1, &r1);
		pid_t h2 = spawn_holder(CIPHER_CP54_QOS_PARTITION, 16, &m2, &c2, &r2);
		CHECK(c1 == 2 && c2 == 2, "two PARTITION tenants each get 2 groups");
		CHECK((m1 & m2) == 0, "their group masks are DISJOINT (no shared SM)");
		release_holder(r1, h1);
		release_holder(r2, h2);
		fd = dev_open();
		do_query(fd, &q);
		CHECK(q.free_grp_count == 16, "after release: all 16 groups free");
		close(fd);
	}

	/* ---- Test 6: concurrent stress (light) ---- */
	printf("Test 6 — concurrent stress:\n");
	{
		const int NK = 4, ITERS = 5;
		int i, allok = 1;
		for (i = 0; i < NK; i++) {
			pid_t c = fork();
			if (c == 0) {
				int cfd = dev_open(), j;
				for (j = 0; j < ITERS; j++) {
					unsigned m, n;
					do_alloc(cfd, CIPHER_CP54_QOS_PARTITION, 8, &m, &n);
					ioctl(cfd, CIPHER_CP54_FREE);
				}
				_exit(0);
			}
		}
		for (i = 0; i < NK; i++) {
			int st = 0;
			wait(&st);
			if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) allok = 0;
		}
		CHECK(allok, "4 tenants x 5 ALLOCATE/FREE concurrent — no crash");
		fd = dev_open();
		rc = do_query(fd, &q);
		CHECK(rc == 0 && q.free_grp_count == 16 && q.n_partitions == 0,
		      "ledger consistent after stress — all 16 free, 0 partitions");
		close(fd);
	}

	printf("\n=== Phase 2 result: %d PASS, %d FAIL ===\n", g_pass, g_fail);
	return g_fail == 0 ? 0 : 1;
}
