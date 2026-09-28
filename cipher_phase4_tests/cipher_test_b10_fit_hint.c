/* SPDX-License-Identifier: GPL-2.0
 * cipher_test_b10_fit_hint.c — verify kmod 0.4.5 FIT_HINT semantic.
 *
 * Test matrix:
 *   1) hint=4 flags=0           -> 4 slots
 *   2) hint=8 flags=0           -> grows to 8
 *   3) hint=4 flags=0           -> STAYS at 8 (old asymmetric behavior)
 *   4) hint=4 flags=FIT_HINT    -> SHRINKS to 4 (B10 new behavior)
 *   5) hint=2 flags=FIT_HINT    -> SHRINKS to 2
 *   6) hint=6 flags=FIT_HINT    -> GROWS to 6
 *   7) hint=6 flags=FIT_HINT    -> STAYS at 6 (no-op same)
 *   8) hint=0 flags=FIT_HINT    -> RELEASE ALL (mask=0)
 *   9) hint=4 flags=0           -> grows to 4 (post-release)
 *  10) hint=0 flags=0           -> clamps to 1 (old behavior)
 *  11) hint=4 flags=0x2 (bad)   -> -EINVAL
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

#include "/home/ubuntu/cipher_kmod/cipher_ioctl.h"

static int popcount32(unsigned int x) { return __builtin_popcount(x); }

static int do_req(int fd, unsigned hint, unsigned flags,
                  unsigned *out_mask, unsigned *out_count, int *out_errno)
{
	struct cipher_partition_request req;
	int rc;
	memset(&req, 0, sizeof(req));
	req.hint_partitions = hint;
	req.flags = flags;
	errno = 0;
	rc = ioctl(fd, CIPHER_REQUEST_SM_PARTITION, &req);
	*out_errno = errno;
	if (rc != 0) {
		*out_mask = 0; *out_count = 0;
		return rc;
	}
	*out_mask  = req.partition_mask_out;
	*out_count = req.partition_count_out;
	return 0;
}

static void register_self(int fd, const char *tname)
{
	struct cipher_register_tenant rt;
	memset(&rt, 0, sizeof(rt));
	rt.pid  = getpid();
	rt.tgid = getpid();
	strncpy(rt.tenant_id, tname, CIPHER_TENANT_ID_LEN - 1);
	(void)ioctl(fd, CIPHER_REGISTER_TENANT, &rt);
}

int main(void)
{
	int fd = open("/dev/cipher", O_RDWR);
	if (fd < 0) { perror("open"); return 1; }
	register_self(fd, "b10_fit_hint_test");

	unsigned mask, count;
	int err;
	int pass = 0, fail = 0;

#define CHECK(label, cond) do { \
	if (cond) { printf("  PASS  %s\n", label); pass++; } \
	else      { printf("  FAIL  %s\n", label); fail++; } \
} while (0)

	/* 1 */
	do_req(fd, 4, 0, &mask, &count, &err);
	printf("[1] hint=4 flags=0 -> mask=0x%08x count=%u\n", mask, count);
	CHECK("popcount==4", popcount32(mask) == 4);

	/* 2 */
	do_req(fd, 8, 0, &mask, &count, &err);
	printf("[2] hint=8 flags=0 -> mask=0x%08x count=%u\n", mask, count);
	CHECK("popcount==8 (grew)", popcount32(mask) == 8);
	unsigned mask2 = mask;

	/* 3 */
	do_req(fd, 4, 0, &mask, &count, &err);
	printf("[3] hint=4 flags=0 -> mask=0x%08x count=%u\n", mask, count);
	CHECK("mask STAYS at 8 (old asymmetric)", mask == mask2);

	/* 4 — the B10 headline test */
	do_req(fd, 4, CIPHER_PARTITION_FLAG_FIT_HINT, &mask, &count, &err);
	printf("[4] hint=4 FLAG_FIT_HINT -> mask=0x%08x count=%u\n", mask, count);
	CHECK("popcount==4 (B10 shrink)", popcount32(mask) == 4);

	/* 5 */
	do_req(fd, 2, CIPHER_PARTITION_FLAG_FIT_HINT, &mask, &count, &err);
	printf("[5] hint=2 FIT_HINT -> mask=0x%08x count=%u\n", mask, count);
	CHECK("popcount==2 (shrink further)", popcount32(mask) == 2);

	/* 6 */
	do_req(fd, 6, CIPHER_PARTITION_FLAG_FIT_HINT, &mask, &count, &err);
	printf("[6] hint=6 FIT_HINT -> mask=0x%08x count=%u\n", mask, count);
	CHECK("popcount==6 (FIT_HINT grow)", popcount32(mask) == 6);
	unsigned mask6 = mask;

	/* 7 */
	do_req(fd, 6, CIPHER_PARTITION_FLAG_FIT_HINT, &mask, &count, &err);
	printf("[7] hint=6 FIT_HINT repeat -> mask=0x%08x count=%u\n", mask, count);
	CHECK("identical (no-op same)", mask == mask6 && popcount32(mask) == 6);

	/* 8 — release all */
	int rc8 = do_req(fd, 0, CIPHER_PARTITION_FLAG_FIT_HINT, &mask, &count, &err);
	printf("[8] hint=0 FIT_HINT -> rc=%d mask=0x%08x count=%u (errno=%d %s)\n",
	       rc8, mask, count, err, strerror(err));
	CHECK("release-all returns mask=0", mask == 0 && count == 0);

	/* 9 */
	do_req(fd, 4, 0, &mask, &count, &err);
	printf("[9] hint=4 flags=0 -> mask=0x%08x count=%u\n", mask, count);
	CHECK("re-grow after release", popcount32(mask) == 4);

	/* 10 */
	do_req(fd, 0, 0, &mask, &count, &err);
	printf("[10] hint=0 flags=0 -> mask=0x%08x count=%u (clamps to 1)\n", mask, count);
	CHECK("hint=0 no-FIT clamps to old behavior", count >= 1);

	/* 11 — bad flag bit */
	int rc11 = do_req(fd, 4, 0x2, &mask, &count, &err);
	printf("[11] hint=4 flags=0x2 (bad) -> rc=%d errno=%d %s\n", rc11, err, strerror(err));
	CHECK("unknown flag rejected with EINVAL", rc11 != 0 && err == EINVAL);

	close(fd);

	printf("\n=== Summary ===\n");
	printf("pass: %d  fail: %d\n", pass, fail);
	return fail > 0 ? 1 : 0;
}
