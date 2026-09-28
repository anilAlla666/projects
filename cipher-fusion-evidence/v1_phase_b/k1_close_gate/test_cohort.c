/* W.6 sub-C — kmod co-residence registry functional test.
 *
 * Deterministic, no GPU. Drives CIPHER_COHORT_REGISTER (nr 30) +
 * CIPHER_COHORT_QUERY (nr 31) directly to verify:
 *   T1 single-tenant : self registers; query returns self with its fp.
 *   T2 multi same-mdl: a peer with the SAME fingerprint is co-resident
 *                      (W.4 would coalesce). [M1 analog]
 *   T3 multi diff-mdl: a peer with a DIFFERENT fingerprint is co-resident
 *                      (W.4 would NOT coalesce). [M2 analog]
 *   T4 heartbeat     : peers that only QUERY (never explicit REGISTER) stay
 *                      live via the heartbeat-or-insert path.
 *   T5 stale-prune   : a killed peer ages out within the 30 s window; the
 *                      survivor's count drops and its fingerprint disappears.
 *
 * Build: gcc -O2 -o test_cohort test_cohort.c
 * Run:   ./test_cohort       (~35 s — T5 needs the 30 s liveness window)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <stdint.h>
#include <linux/ioctl.h>

#define CIPHER_IOCTL_MAGIC 'C'
#define CIPHER_COHORT_MAX  128

struct cohort_register { uint32_t tgid; uint32_t _pad; uint64_t fp; };
struct cohort_peer     { uint32_t tgid; uint32_t _pad; uint64_t fp; };
struct cohort_query {
	uint64_t caller_fp;
	uint32_t max_entries;
	uint32_t n_resident;
	struct cohort_peer entries[CIPHER_COHORT_MAX];
};

#define COHORT_REGISTER _IOW(CIPHER_IOCTL_MAGIC, 30, struct cohort_register)
#define COHORT_QUERY    _IOWR(CIPHER_IOCTL_MAGIC, 31, struct cohort_query)

#define FP_LLAMA   0xd40e456727923bceULL
#define FP_MISTRAL 0x9fde4562bbbb73ceULL

static int g_fail = 0;
#define CHECK(cond, msg, ...) do { \
	if (cond) { printf("  PASS: " msg "\n", ##__VA_ARGS__); } \
	else      { printf("  FAIL: " msg "\n", ##__VA_ARGS__); g_fail++; } \
} while (0)

static int reg(int fd, uint64_t fp)
{
	struct cohort_register r = { .tgid = 0, ._pad = 0, .fp = fp };
	return ioctl(fd, COHORT_REGISTER, &r);
}

/* Query (heartbeat-or-insert self with caller_fp). Returns n_resident; if
 * find_tgid != 0, writes that tgid's fingerprint to *found_fp (or sets
 * *found_fp = 0 if absent). */
static uint32_t query(int fd, uint64_t caller_fp, uint32_t find_tgid, uint64_t *found_fp)
{
	struct cohort_query q;
	memset(&q, 0, sizeof(q));
	q.caller_fp = caller_fp;
	q.max_entries = CIPHER_COHORT_MAX;
	if (ioctl(fd, COHORT_QUERY, &q) < 0) {
		printf("  ioctl QUERY errno=%d\n", errno);
		return 0;
	}
	if (found_fp) {
		*found_fp = 0;
		for (uint32_t i = 0; i < q.n_resident && i < CIPHER_COHORT_MAX; i++)
			if (q.entries[i].tgid == find_tgid) { *found_fp = q.entries[i].fp; break; }
	}
	return q.n_resident;
}

/* Child: register fp, then heartbeat via QUERY every 1 s for `secs`. */
static void child_body(uint64_t fp, int use_explicit_reg, int secs)
{
	int fd = open("/dev/cipher", O_RDWR | O_CLOEXEC);
	if (fd < 0) _exit(2);
	if (use_explicit_reg) reg(fd, fp);
	for (int i = 0; i < secs; i++) { query(fd, fp, 0, NULL); sleep(1); }
	close(fd);
	_exit(0);
}

int main(void)
{
	int fd = open("/dev/cipher", O_RDWR | O_CLOEXEC);
	if (fd < 0) { printf("open /dev/cipher failed errno=%d\n", errno); return 2; }
	pid_t self = getpid();

	printf("[T1] single-tenant: self register + query\n");
	CHECK(reg(fd, FP_LLAMA) == 0, "REGISTER self (fp=0x%llx)", (unsigned long long)FP_LLAMA);
	uint64_t sfp = 0;
	uint32_t n = query(fd, FP_LLAMA, self, &sfp);
	CHECK(n >= 1, "query n_resident=%u (>=1)", n);
	CHECK(sfp == FP_LLAMA, "self fingerprint readback = 0x%llx", (unsigned long long)sfp);

	printf("[T2/T3] multi-tenant: spawn same-model + diff-model peers\n");
	/* child1: SAME fingerprint as self (M1, coalescable) via heartbeat-only. */
	pid_t c1 = fork();
	if (c1 == 0) child_body(FP_LLAMA, 0 /*query-only heartbeat*/, 45);
	/* child2: DIFFERENT fingerprint (M2, not coalescable) via explicit reg. */
	pid_t c2 = fork();
	if (c2 == 0) child_body(FP_MISTRAL, 1 /*explicit register*/, 45);

	sleep(3);  /* let children register */
	uint64_t f1 = 0, f2 = 0;
	n = query(fd, FP_LLAMA, c1, &f1);
	query(fd, FP_LLAMA, c2, &f2);
	CHECK(n >= 3, "co_resident n_resident=%u (>=3: self + 2 peers)", n);
	CHECK(f1 == FP_LLAMA, "[T2 same-model] child1 fp=0x%llx == self (W.4 coalesces)", (unsigned long long)f1);
	CHECK(f2 == FP_MISTRAL, "[T3 diff-model] child2 fp=0x%llx != self (W.4 splits)", (unsigned long long)f2);
	CHECK(f1 != f2, "[T4 heartbeat] query-only peer (child1) is live alongside reg peer (child2)");

	printf("[T5] stale-prune: kill diff-model peer, wait past 30 s window\n");
	kill(c2, SIGKILL);
	waitpid(c2, NULL, 0);
	/* Keep self heartbeating; child1 heartbeats itself. Poll until child2
	 * ages out (last_seen > 30 s) or 38 s cap. */
	uint64_t fdead = 1; uint32_t npruned = 0;
	for (int t = 0; t < 38; t++) {
		sleep(2);
		npruned = query(fd, FP_LLAMA, c2, &fdead);
		if (fdead == 0) break;   /* child2 pruned */
	}
	CHECK(fdead == 0, "killed diff-model peer pruned from registry");
	uint64_t still = 0;
	query(fd, FP_LLAMA, c1, &still);
	CHECK(still == FP_LLAMA, "same-model peer (child1, still heartbeating) survives prune");
	CHECK(npruned >= 2, "post-prune n_resident=%u (self + surviving peer)", npruned);

	kill(c1, SIGKILL); waitpid(c1, NULL, 0);
	close(fd);
	printf("\n=== test_cohort: %s (%d failures) ===\n", g_fail ? "FAIL" : "PASS", g_fail);
	return g_fail ? 1 : 0;
}
