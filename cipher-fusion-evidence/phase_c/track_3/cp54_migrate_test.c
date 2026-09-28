// SPDX-License-Identifier: GPL-2.0
/*
 * cp54_migrate_test.c — Track 3 SC2 unit tests for the kmod migration state
 * machine (CIPHER_CP54_* nrs 16-20) + 5-scenario PROPOSE->COMMIT latency
 * capture (SC2 plan item 6 / item-2 PUSH).
 *
 * Build: gcc -O2 -I/home/ubuntu/cipher_kmod -o cp54_migrate_test cp54_migrate_test.c
 *
 * SC2 measures the KMOD-SIDE coordination latency only — PROPOSE (inside
 * COMPACT) to COMMIT (inside ACK). The tenant-side green-context rebuild
 * (~1.7 ms, Step 1.5) is SC3 and is NOT part of this number.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <time.h>
#include "cipher_ioctl.h"

static int g_pass, g_fail;
#define CHECK(c, name) do {                                            \
	if (c) { printf("  PASS  %s\n", name); g_pass++; }             \
	else   { printf("  FAIL  %s\n", name); g_fail++; }             \
} while (0)

static long now_ns(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (long)t.tv_sec * 1000000000L + t.tv_nsec;
}

static int dev_open(void)
{
	int fd = open("/dev/cipher", O_RDWR);
	if (fd < 0) { perror("open /dev/cipher"); exit(1); }
	return fd;
}

static int cp_alloc(int fd, int sm)
{
	struct cipher_cp54_allocate a;
	memset(&a, 0, sizeof a);
	a.qos_class = CIPHER_CP54_QOS_PARTITION;
	a.sm_count  = sm;
	if (ioctl(fd, CIPHER_CP54_ALLOCATE, &a) < 0) return -1;
	return (int)a.grp_mask_out;
}
static void cp_free(int fd)        { ioctl(fd, CIPHER_CP54_FREE, 0); }
static int  cp_subscribe(int fd, unsigned v)
{ return ioctl(fd, CIPHER_CP54_SUBSCRIBE_MIGRATE, &v); }
static int  cp_start(int fd)       { return ioctl(fd, CIPHER_CP54_START_MIGRATE, 0); }
static int  cp_ack(int fd, unsigned ok)
{ return ioctl(fd, CIPHER_CP54_ACK_MIGRATE, &ok); }
static int  cp_compact(int fd)     { return ioctl(fd, CIPHER_CP54_COMPACT_MIGRATE, 0); }
static int  cp_poll(int fd, struct cipher_cp54_migrate_poll *p)
{ memset(p, 0, sizeof *p); return ioctl(fd, CIPHER_CP54_POLL_MIGRATE, p); }
static int  cp_query_free(int fd)
{
	struct cipher_cp54_query q;
	memset(&q, 0, sizeof q);
	ioctl(fd, CIPHER_CP54_QUERY, &q);
	return (int)q.free_grp_count;
}

/* ---- state-machine correctness tests --------------------------------- */

static void test_state_machine(void)
{
	struct cipher_cp54_migrate_poll p;
	int fd, m;
	unsigned tgt;

	printf("Test SM-1 — happy path PROPOSE->START->MIGRATING->ACK->COMMIT:\n");
	fd = dev_open();
	m = cp_alloc(fd, 8);
	CHECK(m > 0 && __builtin_popcount(m) == 1,
	      "ALLOCATE PARTITION 8SM -> 1 group");
	CHECK(cp_subscribe(fd, 1) == 0, "SUBSCRIBE_MIGRATE(1) ok");
	CHECK(cp_compact(fd) == 0, "COMPACT_MIGRATE ok");
	cp_poll(fd, &p);
	CHECK(p.migrate_state == CIPHER_CP54_MIG_PROPOSED, "POLL -> PROPOSED");
	CHECK(p.target_mask != 0 &&
	      __builtin_popcount(p.target_mask) == __builtin_popcount(m),
	      "target_mask is count-preserving");
	CHECK((p.target_mask & m) == 0, "target_mask disjoint from current");
	tgt = p.target_mask;
	CHECK(cp_start(fd) == 0, "START_MIGRATE ok (PROPOSED->MIGRATING)");
	cp_poll(fd, &p);
	CHECK(p.migrate_state == CIPHER_CP54_MIG_MIGRATING, "POLL -> MIGRATING");
	CHECK(cp_ack(fd, 1) == 0, "ACK_MIGRATE(1) ok (MIGRATING->COMMIT)");
	cp_poll(fd, &p);
	CHECK(p.migrate_state == CIPHER_CP54_MIG_IDLE, "POLL -> IDLE after COMMIT");
	CHECK(p.last_outcome == CIPHER_CP54_MIGOUT_COMMITTED,
	      "last_outcome == COMMITTED");
	CHECK(p.cur_mask == tgt, "cur_mask == target — migration applied");
	cp_free(fd); close(fd);

	printf("Test SM-2 — pinned tenant is never proposed to:\n");
	fd = dev_open();
	cp_alloc(fd, 8);                    /* no SUBSCRIBE -> pinned default */
	cp_compact(fd);
	cp_poll(fd, &p);
	CHECK(p.migrate_state == CIPHER_CP54_MIG_IDLE,
	      "pinned: COMPACT -> still IDLE (no PROPOSE)");
	cp_free(fd); close(fd);

	printf("Test SM-3 — START without PROPOSE rejected:\n");
	fd = dev_open();
	cp_alloc(fd, 8); cp_subscribe(fd, 1);
	CHECK(cp_start(fd) < 0, "START in IDLE -> -EINVAL");
	cp_free(fd); close(fd);

	printf("Test SM-4 — ACK without START rejected:\n");
	fd = dev_open();
	cp_alloc(fd, 8); cp_subscribe(fd, 1); cp_compact(fd);
	cp_poll(fd, &p);
	CHECK(p.migrate_state == CIPHER_CP54_MIG_PROPOSED, "setup: PROPOSED");
	CHECK(cp_ack(fd, 1) < 0, "ACK in PROPOSED -> -EINVAL");
	cp_start(fd); cp_ack(fd, 1);         /* clean up: commit it */
	cp_free(fd); close(fd);

	printf("Test SM-5 — tenant NACK (ACK ok=0) + rate-limit:\n");
	fd = dev_open();
	cp_alloc(fd, 8); cp_subscribe(fd, 1);
	cp_compact(fd); cp_poll(fd, &p);
	CHECK(p.migrate_state == CIPHER_CP54_MIG_PROPOSED, "setup: PROPOSED");
	cp_start(fd);
	CHECK(cp_ack(fd, 0) == 0, "ACK_MIGRATE(0) NACK ok");
	cp_poll(fd, &p);
	CHECK(p.migrate_state == CIPHER_CP54_MIG_IDLE, "NACK -> IDLE");
	CHECK(p.last_outcome == CIPHER_CP54_MIGOUT_ABORTED_TENANT_NACK,
	      "last_outcome == ABORTED_TENANT_NACK");
	CHECK(p.cur_mask != 0, "NACK: tenant keeps its owned groups (B-floor)");
	cp_compact(fd); cp_poll(fd, &p);     /* re-COMPACT within 10s */
	CHECK(p.migrate_state == CIPHER_CP54_MIG_IDLE,
	      "rate-limit: re-COMPACT within 10s -> no re-PROPOSE");
	cp_free(fd); close(fd);
}

static void test_reaper(void)
{
	struct cipher_cp54_migrate_poll p;
	int fd, st;
	pid_t c;

	printf("Test SM-6 — do_exit reaper frees PACK+RSVD on crash:\n");

	c = fork();
	if (c == 0) {
		int f = dev_open();
		cp_alloc(f, 8); cp_subscribe(f, 1); cp_compact(f);
		cp_poll(f, &p);
		_exit(p.migrate_state == CIPHER_CP54_MIG_PROPOSED ? 0 : 1);
	}
	waitpid(c, &st, 0);
	CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0,
	      "child reached PROPOSED then exited without FREE");
	usleep(150000);
	fd = dev_open();
	CHECK(cp_query_free(fd) == 15,
	      "crash-in-PROPOSED: all 15 groups free (PACK + RSVD reaped)");
	close(fd);

	c = fork();
	if (c == 0) {
		int f = dev_open();
		cp_alloc(f, 8); cp_subscribe(f, 1); cp_compact(f); cp_start(f);
		cp_poll(f, &p);
		_exit(p.migrate_state == CIPHER_CP54_MIG_MIGRATING ? 0 : 1);
	}
	waitpid(c, &st, 0);
	CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0,
	      "child reached MIGRATING then exited without FREE");
	usleep(150000);
	fd = dev_open();
	CHECK(cp_query_free(fd) == 15,
	      "crash-in-MIGRATING: all 15 groups free (PACK + RSVD reaped)");
	close(fd);
}

/* returns observed PROPOSE->ABORT wall time (s) for the quiet-period test */
static double test_timeout(void)
{
	struct cipher_cp54_migrate_poll p;
	int fd;
	long t0, t1;

	printf("Test SM-7 — PROPOSED timeout -> ABORTED_TIMEOUT (quiet, ~32s):\n");
	fd = dev_open();
	cp_alloc(fd, 8); cp_subscribe(fd, 1);
	cp_compact(fd); cp_poll(fd, &p);
	CHECK(p.migrate_state == CIPHER_CP54_MIG_PROPOSED, "setup: PROPOSED");
	t0 = now_ns();
	sleep(32);                           /* quiet ledger, never START */
	cp_compact(fd);                      /* lazy timeout fires here */
	t1 = now_ns();
	cp_poll(fd, &p);
	CHECK(p.last_outcome == CIPHER_CP54_MIGOUT_ABORTED_TIMEOUT,
	      "timed-out migration recorded as ABORTED_TIMEOUT");
	cp_free(fd); close(fd);
	return (t1 - t0) / 1e9;
}

/* ---- latency capture (item 6 PUSH — 5 scenarios) --------------------- */

static int cmp_double(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;
	return (x > y) - (x < y);
}

/* One PROPOSE->COMMIT cycle on `fd`; returns wall us, or -1 if not PROPOSED. */
static double one_cycle(int fd)
{
	struct cipher_cp54_migrate_poll p;
	long t0, t1;

	cp_alloc(fd, 8); cp_subscribe(fd, 1);
	cp_compact(fd);
	t0 = now_ns();
	cp_poll(fd, &p);
	if (p.migrate_state != CIPHER_CP54_MIG_PROPOSED) { cp_free(fd); return -1; }
	cp_start(fd);
	cp_ack(fd, 1);
	t1 = now_ns();
	cp_free(fd);
	return (t1 - t0) / 1000.0;
}

int main(void)
{
	struct cipher_cp54_migrate_poll p;
	double quiet_abort_s, *arr, *carr;
	int fd, i, n, miss = 0, cn;
	long t0, t1;
	FILE *j;

	printf("CP 5.4 Track 3 SC2 — migration state-machine tests (/dev/cipher)\n\n");

	test_state_machine();
	test_reaper();
	quiet_abort_s = test_timeout();

	printf("\n=== state-machine result: %d PASS, %d FAIL ===\n\n",
	       g_pass, g_fail);

	/* --- latency scenario 1+2: median / p95 / p99 PROPOSE->COMMIT --- */
	printf("Latency — scenario 1+2: PROPOSE->COMMIT x200\n");
	n = 200;
	arr = calloc(n, sizeof(double));
	fd = dev_open();
	for (i = 0; i < n; i++) {
		double us = one_cycle(fd);
		if (us < 0) { miss++; arr[i] = 0; } else arr[i] = us;
	}
	close(fd);
	CHECK(miss == 0, "all 200 cycles reached PROPOSED");
	qsort(arr, n, sizeof(double), cmp_double);
	double med = arr[n / 2], p95 = arr[(int)(n * 0.95)],
	       p99 = arr[(int)(n * 0.99)], mn = arr[0], mx = arr[n - 1];
	printf("  median=%.1fus p95=%.1fus p99=%.1fus min=%.1fus max=%.1fus\n",
	       med, p95, p99, mn, mx);

	/* --- scenario 3: PROPOSE->COMMIT under contention --- */
	printf("Latency — scenario 3: PROPOSE->COMMIT under 7-process contention\n");
	for (i = 0; i < 7; i++) {
		pid_t c = fork();
		if (c == 0) {
			int f = dev_open();
			long end = now_ns() + 4L * 1000000000L;
			while (now_ns() < end) cp_compact(f);  /* lock contention */
			close(f);
			_exit(0);
		}
	}
	cn = 50;
	carr = calloc(cn, sizeof(double));
	fd = dev_open();
	for (i = 0; i < cn; i++) {
		double us = one_cycle(fd);
		carr[i] = (us < 0) ? 0 : us;
	}
	close(fd);
	for (i = 0; i < 7; i++) wait(NULL);
	qsort(carr, cn, sizeof(double), cmp_double);
	double c_med = carr[cn / 2], c_max = carr[cn - 1];
	printf("  contention median=%.1fus max=%.1fus\n", c_med, c_max);

	/* --- scenario 4: worst-case under tenant slow-response --- */
	printf("Latency — scenario 4: slow tenant response (delay between START/ACK)\n");
	int delays[3] = { 50, 200, 500 };       /* ms */
	double slow_us[3];
	for (i = 0; i < 3; i++) {
		fd = dev_open();
		cp_alloc(fd, 8); cp_subscribe(fd, 1);
		cp_compact(fd);
		t0 = now_ns();
		cp_poll(fd, &p);
		cp_start(fd);
		usleep(delays[i] * 1000);            /* simulate slow migrate() */
		cp_ack(fd, 1);
		t1 = now_ns();
		slow_us[i] = (t1 - t0) / 1000.0;
		cp_free(fd); close(fd);
		printf("  injected %dms -> PROPOSE->COMMIT %.1fus\n",
		       delays[i], slow_us[i]);
	}

	/* --- scenario 5: PROPOSED->ABORT under a quiet ledger --- */
	printf("Latency — scenario 5: PROPOSED->ABORT quiet-period = %.1fs\n",
	       quiet_abort_s);

	/* --- emit JSON --- */
	j = fopen("/home/ubuntu/cipher-fusion-evidence/phase_c/track_3/"
		  "TRACK_3_SC2_LATENCY.json", "w");
	fprintf(j,
	"{\n"
	"  \"_note\": \"SC2 measures kmod-side coordination latency PROPOSE(in COMPACT)->COMMIT(in ACK). Tenant-side green-ctx rebuild (~1.7ms, Step 1.5) is SC3, not included.\",\n"
	"  \"scenario_1_2_propose_commit\": {\n"
	"    \"n\": %d, \"misses\": %d,\n"
	"    \"median_us\": %.1f, \"p95_us\": %.1f, \"p99_us\": %.1f,\n"
	"    \"min_us\": %.1f, \"max_us\": %.1f\n"
	"  },\n"
	"  \"scenario_3_contention\": {\n"
	"    \"n\": %d, \"noise_procs\": 7, \"noise_op\": \"COMPACT_MIGRATE spin (cipher_cp54_lock contention)\",\n"
	"    \"median_us\": %.1f, \"max_us\": %.1f\n"
	"  },\n"
	"  \"scenario_4_slow_response\": [\n"
	"    {\"injected_ms\": 50,  \"propose_commit_us\": %.1f},\n"
	"    {\"injected_ms\": 200, \"propose_commit_us\": %.1f},\n"
	"    {\"injected_ms\": 500, \"propose_commit_us\": %.1f}\n"
	"  ],\n"
	"  \"scenario_5_quiet_abort\": {\n"
	"    \"timeout_ns\": 30000000000,\n"
	"    \"observed_abort_s\": %.2f,\n"
	"    \"note\": \"Lazy timeout: ABORT fires on the first FREE/COMPACT after PROPOSE+30s. Under a permanently quiet ledger the latency is UNBOUNDED until the next ledger activity — the item-4 trade-off accepted at adjudication. SC6 must weigh this for the v1.5 explicit-timer decision.\"\n"
	"  }\n"
	"}\n",
	n, miss, med, p95, p99, mn, mx,
	cn, c_med, c_max,
	slow_us[0], slow_us[1], slow_us[2],
	quiet_abort_s);
	fclose(j);
	free(arr); free(carr);

	printf("\nwrote TRACK_3_SC2_LATENCY.json\n");
	printf("\n=== SC2 UNIT TEST TOTAL: %d PASS, %d FAIL ===\n", g_pass, g_fail);
	return g_fail ? 1 : 0;
}
