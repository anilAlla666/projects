/* T4.6.4 cross-process dedup harness — Phase 4 closure gate.
 *
 * Forks independent tenant processes that each use the kmod-backed
 * cipher_rt_kv_dedup API against the shared /dev/cipher_kvdedup. Runs
 * indicators (a) cross-process byte-correct, (b) tenant teardown /
 * release-fop, (d) refcount integrity under churn. (c) is the slab
 * unit test (separate); (e) is the module-unload shell test.
 *
 * Children replay explicit page-id lists (Mooncake windows for (a),
 * an interleaved toolagent window for (d)); identical id => identical
 * synthesized 2 MiB content, so a cross-process dedup must reproduce
 * the content byte-for-byte.
 *
 * Build: gcc -O2 -I<phase4> -I<kmod> -o kvdedup_xproc kvdedup_xproc.c \
 *          <phase4>/cipher_rt_kv_alloc.c -lcuda -lcrypto -lpthread
 */
#include "cipher_rt_kv_alloc.h"
#include "cipher_kvdedup.h"
#include <cuda.h>
#include <openssl/sha.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <signal.h>

#define PAGE (2 * 1024 * 1024)
#define POOL (20ULL << 30)
#define MAXP 8192

enum { CMD_PUT, CMD_VERIFY, CMD_FREEALL, CMD_EXIT };
struct cmd { int op; int id_count; };   /* CMD_PUT: id_count ints follow */
struct res { int ok; int fail; };

static void synth(uint64_t pid, uint8_t *buf)
{
	uint64_t x = pid * 0x9E3779B97F4A7C15ULL + 0x123456789ULL;
	uint64_t *w = (uint64_t *)buf;
	for (size_t i = 0; i < PAGE / 8; i++) {
		x += 0x9E3779B97F4A7C15ULL;
		uint64_t z = x;
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
		z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
		w[i] = z ^ (z >> 31);
	}
}

static ssize_t readn(int fd, void *buf, size_t n)
{
	size_t got = 0;
	while (got < n) {
		ssize_t r = read(fd, (char *)buf + got, n - got);
		if (r <= 0) return r;
		got += r;
	}
	return (ssize_t)got;
}

/* ---- tenant child: long-lived, driven by cmd/res pipes ---- */
static int child_main(int cfd, int rfd)
{
	CUcontext ctx;
	CUdevice dev;
	if (cuInit(0) || cuDeviceGet(&dev, 0) ||
	    cuDevicePrimaryCtxRetain(&ctx, dev) || cuCtxSetCurrent(ctx))
		_exit(20);
	if (cipher_rt_kv_alloc_init(POOL) != 0) _exit(21);
	if (cipher_rt_kv_dedup_init() != 0) _exit(22);

	static unsigned long long dev_ptr[MAXP];
	static int dev_id[MAXP];
	static int idbuf[MAXP];
	int n = 0;
	uint8_t *content = malloc(PAGE), *readback = malloc(PAGE);
	struct cmd c;

	while (readn(cfd, &c, sizeof c) == (ssize_t)sizeof c) {
		struct res r = {0, 0};
		if (c.op == CMD_PUT) {
			int k = c.id_count;
			if (k > MAXP) k = MAXP;
			if (readn(cfd, idbuf, (size_t)k * sizeof(int))
			    != (ssize_t)((size_t)k * sizeof(int)))
				_exit(24);
			for (int i = 0; i < k && n < MAXP; i++) {
				synth((uint64_t)idbuf[i], content);
				unsigned long long dp;
				if (cipher_rt_kv_dedup_put(content, &dp) != 0) {
					r.fail++;
					continue;
				}
				dev_ptr[n] = dp;
				dev_id[n] = idbuf[i];
				n++;
				r.ok++;
			}
		} else if (c.op == CMD_VERIFY) {
			for (int i = 0; i < n; i++) {
				synth((uint64_t)dev_id[i], content);
				if (cuMemcpyDtoH(readback, (CUdeviceptr)dev_ptr[i],
				                 PAGE) != CUDA_SUCCESS) {
					r.fail++;
					continue;
				}
				unsigned char a[32], b[32];
				SHA256(content, PAGE, a);
				SHA256(readback, PAGE, b);
				if (memcmp(a, b, 32) == 0) r.ok++;
				else r.fail++;
			}
		} else if (c.op == CMD_FREEALL) {
			for (int i = 0; i < n; i++)
				cipher_rt_kv_dedup_free(dev_ptr[i]);
			n = 0;
			r.ok = 1;
		} else if (c.op == CMD_EXIT) {
			ssize_t w = write(rfd, &r, sizeof r); (void)w;
			_exit(0);   /* graceful close -> release fop fires */
		}
		if (write(rfd, &r, sizeof r) != (ssize_t)sizeof r)
			_exit(23);
	}
	_exit(0);
}

struct child { pid_t pid; int cfd; int rfd; };

static struct child spawn_child(void)
{
	int c2[2], r2[2];
	struct child ch;
	if (pipe(c2) || pipe(r2)) { perror("pipe"); exit(1); }
	ch.pid = fork();
	if (ch.pid == 0) {
		close(c2[1]); close(r2[0]);
		_exit(child_main(c2[0], r2[1]));
	}
	close(c2[0]); close(r2[1]);
	ch.cfd = c2[1]; ch.rfd = r2[0];
	return ch;
}

/* CMD_PUT with an explicit id list */
static struct res tell_put(struct child *ch, const int *ids, int n)
{
	struct cmd c = { CMD_PUT, n };
	struct res r = { -1, -1 };
	if (write(ch->cfd, &c, sizeof c) != (ssize_t)sizeof c) return r;
	if (write(ch->cfd, ids, (size_t)n * sizeof(int))
	    != (ssize_t)((size_t)n * sizeof(int))) return r;
	if (readn(ch->rfd, &r, sizeof r) != (ssize_t)sizeof r) r.ok = -1;
	return r;
}

/* CMD_VERIFY / CMD_FREEALL / CMD_EXIT */
static struct res tell(struct child *ch, int op)
{
	struct cmd c = { op, 0 };
	struct res r = { -1, -1 };
	if (write(ch->cfd, &c, sizeof c) != (ssize_t)sizeof c) return r;
	if (readn(ch->rfd, &r, sizeof r) != (ssize_t)sizeof r) r.ok = -1;
	return r;
}

/* parent opens the device directly as a stats observer */
static struct cipher_kvdedup_stats kmod_stats(void)
{
	struct cipher_kvdedup_stats s;
	memset(&s, 0, sizeof s);
	int fd = open("/dev/" CIPHER_KVDEDUP_DEV_NAME, O_RDWR);
	if (fd < 0) return s;
	struct cipher_kvdedup_init ini;
	ioctl(fd, CIPHER_KVDEDUP_INIT, &ini);
	ioctl(fd, CIPHER_KVDEDUP_STATS, &s);
	close(fd);
	return s;
}

/* read up to `cap` page-ids from a pageseq window file */
static int load_ids(const char *path, int *ids, int cap)
{
	FILE *f = fopen(path, "r");
	if (!f) return 0;
	int m; double ratio;
	if (fscanf(f, "%d %lf", &m, &ratio) != 2) { fclose(f); return 0; }
	int n = 0;
	while (n < cap && n < m && fscanf(f, "%d", &ids[n]) == 1) n++;
	fclose(f);
	return n;
}

int main(void)
{
	int fails = 0;
	const char *PD = "/home/ubuntu/cipher-fusion-evidence/t4_6_3_dedup";
	const char *flav[3] = { "conversation", "toolagent", "synthetic" };
	static int ids[MAXP];

	/* ===== Indicator (a): cross-process byte-correct ===== */
	printf("=== indicator (a): cross-process byte-correct ===\n");
	{
		struct child A = spawn_child(), B = spawn_child();
		int run = 0, a_ok = 0, a_fail = 0;
		/* explicit run: 500 distinct ids, A registers, B dedups+verifies */
		for (int i = 0; i < 500; i++) ids[i] = 900000 + i;
		tell_put(&A, ids, 500);
		struct res rb = tell_put(&B, ids, 500);
		struct res rv = tell(&B, CMD_VERIFY);
		printf("  run %2d explicit  : B put ok=%d, verified ok=%d fail=%d\n",
		       run++, rb.ok, rv.ok, rv.fail);
		a_ok += rv.ok; a_fail += rv.fail + (rb.ok != 500);
		tell(&A, CMD_FREEALL); tell(&B, CMD_FREEALL);
		/* 15 Mooncake windows replayed cross-process (cap 1000/window) */
		for (int fi = 0; fi < 3; fi++)
			for (int k = 0; k < 5; k++) {
				char p[256];
				snprintf(p, sizeof p, "%s/pageseq_%s_%d.txt",
				         PD, flav[fi], k);
				int n = load_ids(p, ids, 1000);
				tell_put(&A, ids, n);
				tell_put(&B, ids, n);
				rv = tell(&B, CMD_VERIFY);
				printf("  run %2d %-12s: n=%d  B verified ok=%d fail=%d\n",
				       run++, flav[fi], n, rv.ok, rv.fail);
				a_ok += rv.ok; a_fail += rv.fail;
				tell(&A, CMD_FREEALL); tell(&B, CMD_FREEALL);
			}
		tell(&A, CMD_EXIT); tell(&B, CMD_EXIT);
		waitpid(A.pid, NULL, 0); waitpid(B.pid, NULL, 0);
		printf("  (a) TOTAL: %d runs, cross-process verified ok=%d fail=%d -> %s\n",
		       run, a_ok, a_fail, a_fail == 0 ? "PASS" : "FAIL");
		if (a_fail) fails++;
	}

	/* ===== Indicator (b): tenant teardown / release fop ===== */
	printf("=== indicator (b): tenant teardown (release fop) ===\n");
	{
		int b_fail = 0;
		for (int i = 0; i < 200; i++) ids[i] = 910000 + i;
		/* scenario 1: graceful close() */
		{
			struct child A = spawn_child(), B = spawn_child();
			tell_put(&A, ids, 200);
			tell_put(&B, ids, 200);
			struct res r1 = tell(&B, CMD_VERIFY);
			tell(&A, CMD_EXIT);                 /* graceful */
			waitpid(A.pid, NULL, 0);
			struct res r2 = tell(&B, CMD_VERIFY);
			printf("  graceful: B before=%d/%d, after A close()=%d/%d\n",
			       r1.ok, r1.ok + r1.fail, r2.ok, r2.ok + r2.fail);
			if (r2.fail || r2.ok != 200) b_fail++;
			tell(&B, CMD_EXIT); waitpid(B.pid, NULL, 0);
		}
		/* scenario 2: SIGKILL mid-life */
		{
			struct child A = spawn_child(), B = spawn_child();
			tell_put(&A, ids, 200);
			tell_put(&B, ids, 200);
			kill(A.pid, SIGKILL);
			waitpid(A.pid, NULL, 0);
			struct res r = tell(&B, CMD_VERIFY);
			printf("  sigkill : B after A SIGKILL=%d/%d\n",
			       r.ok, r.ok + r.fail);
			if (r.fail || r.ok != 200) b_fail++;
			tell(&B, CMD_EXIT); waitpid(B.pid, NULL, 0);
		}
		/* scenario 3: 4 tenants share, SIGKILL 3, survivor intact */
		{
			struct child c[4];
			for (int i = 0; i < 4; i++) c[i] = spawn_child();
			for (int i = 0; i < 4; i++) tell_put(&c[i], ids, 200);
			for (int i = 0; i < 3; i++) {
				kill(c[i].pid, SIGKILL);
				waitpid(c[i].pid, NULL, 0);
			}
			struct res r = tell(&c[3], CMD_VERIFY);
			printf("  3-of-4  : survivor verified=%d/%d after 3 SIGKILL\n",
			       r.ok, r.ok + r.fail);
			if (r.fail || r.ok != 200) b_fail++;
			tell(&c[3], CMD_EXIT); waitpid(c[3].pid, NULL, 0);
		}
		struct cipher_kvdedup_stats s = kmod_stats();
		printf("  post-(b) kmod stats: entries=%llu virtual_refs=%llu\n",
		       (unsigned long long)s.entries,
		       (unsigned long long)s.virtual_refs);
		if (s.entries != 0) { printf("  LEAK: entries != 0\n"); b_fail++; }
		printf("  (b) -> %s\n", b_fail == 0 ? "PASS" : "FAIL");
		if (b_fail) fails++;
	}

	/* ===== Indicator (d): refcount integrity, 4-tenant churn ===== */
	printf("=== indicator (d): refcount integrity, 4-tenant churn ===\n");
	{
		struct cipher_kvdedup_stats s0 = kmod_stats();
		char p[256];
		snprintf(p, sizeof p, "%s/pageseq_toolagent_0.txt", PD);
		int n = load_ids(p, ids, 4000);   /* real toolagent window */
		/* interleave-split: child i replays ids[i], ids[i+4], ...
		 * -> all 4 tenants draw the same id pool -> cross-tenant churn */
		struct child c[4];
		for (int i = 0; i < 4; i++) c[i] = spawn_child();
		static int q[4][MAXP];
		int qn[4] = {0, 0, 0, 0};
		for (int i = 0; i < n; i++) {
			int t = i & 3;
			q[t][qn[t]++] = ids[i];
		}
		for (int i = 0; i < 4; i++)
			tell_put(&c[i], q[i], qn[i]);
		for (int i = 0; i < 4; i++) {
			tell(&c[i], CMD_FREEALL);
			tell(&c[i], CMD_EXIT);
			waitpid(c[i].pid, NULL, 0);
		}
		struct cipher_kvdedup_stats s1 = kmod_stats();
		unsigned long long misses = s1.misses - s0.misses;
		unsigned long long rel = s1.refcount_releases - s0.refcount_releases;
		printf("  4 tenants, %d-page toolagent window interleaved; "
		       "misses=%llu releases=%llu entries=%llu virtual_refs=%llu\n",
		       n, misses, rel, (unsigned long long)s1.entries,
		       (unsigned long long)s1.virtual_refs);
		int d_ok = (s1.entries == 0 && s1.virtual_refs == 0 &&
		            rel == misses);
		printf("  (d) -> %s\n", d_ok ? "PASS" : "FAIL");
		if (!d_ok) fails++;
	}

	printf("\n=== T4.6.4 harness: %s ===\n",
	       fails == 0 ? "ALL INDICATORS (a)(b)(d) PASS" : "FAIL");
	return fails ? 1 : 0;
}
