/* SPDX-License-Identifier: GPL-2.0
 * cipher_test_b7_idempotency.c — verify kmod allocator behavior on
 * repeat CIPHER_REQUEST_SM_PARTITION from the same pid on the same fd.
 *
 * B7 advisor pre-design (binding): the poll-thread fix depends on the
 * kmod returning predictable masks when re-issued with the same or
 * different hint. Three behaviors to discriminate:
 *
 *   hint=4 -> hint=8  : mask should GROW (more bits set)
 *   hint=8 -> hint=4  : mask should SHRINK or STAY (code comment says stay)
 *   hint=4 -> hint=4  : mask should be IDENTICAL
 *
 * If shrinking does NOT work, B7's poll thread must be asymmetric:
 * re-issue on rank improvement (grow), skip on rank degradation (leak).
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

static int do_req(int fd, unsigned hint, unsigned *out_mask, unsigned *out_count)
{
	struct cipher_partition_request req;
	int rc;
	memset(&req, 0, sizeof(req));
	req.hint_partitions = hint;
	rc = ioctl(fd, CIPHER_REQUEST_SM_PARTITION, &req);
	if (rc != 0) {
		fprintf(stderr, "ioctl hint=%u rc=%d errno=%d %s\n",
		        hint, rc, errno, strerror(errno));
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
	if (fd < 0) { perror("open /dev/cipher"); return 1; }
	register_self(fd, "b7_idempotency_test");

	unsigned mask, count;
	int pass = 0, fail = 0;

	/* 1) initial hint=4 */
	if (do_req(fd, 4, &mask, &count) != 0) return 2;
	printf("[step 1] hint=4 -> mask=0x%08x count=%u (popcount=%d)\n",
	       mask, count, popcount32(mask));
	if (popcount32(mask) == 4) { printf("  PASS popcount==4\n"); pass++; }
	else                       { printf("  FAIL popcount=%d expected 4\n", popcount32(mask)); fail++; }
	unsigned mask1 = mask;

	/* 2) repeat hint=4 — expect IDENTICAL */
	if (do_req(fd, 4, &mask, &count) != 0) return 2;
	printf("[step 2] hint=4 repeat -> mask=0x%08x count=%u\n", mask, count);
	if (mask == mask1) { printf("  PASS mask identical to step 1\n"); pass++; }
	else               { printf("  FAIL mask=0x%08x step1=0x%08x\n", mask, mask1); fail++; }

	/* 3) hint=8 — expect GROW */
	if (do_req(fd, 8, &mask, &count) != 0) return 2;
	printf("[step 3] hint=8 -> mask=0x%08x count=%u (popcount=%d)\n",
	       mask, count, popcount32(mask));
	if (popcount32(mask) == 8) { printf("  PASS popcount==8 (grew)\n"); pass++; }
	else                       { printf("  FAIL popcount=%d expected 8\n", popcount32(mask)); fail++; }
	unsigned mask3 = mask;

	/* 4) hint=4 after hint=8 — does it SHRINK or STAY? */
	if (do_req(fd, 4, &mask, &count) != 0) return 2;
	printf("[step 4] hint=4 after hint=8 -> mask=0x%08x count=%u (popcount=%d)\n",
	       mask, count, popcount32(mask));
	if (popcount32(mask) == 4) {
		printf("  BEHAVIOR: SHRINK (kmod releases slots on smaller hint)\n");
		pass++;
	} else if (mask == mask3) {
		printf("  BEHAVIOR: STAY (kmod retains larger mask; matches code comment)\n");
		pass++;
	} else {
		printf("  FAIL unexpected popcount=%d (not 4 and not equal to step3 0x%08x)\n",
		       popcount32(mask), mask3);
		fail++;
	}
	unsigned mask4 = mask;

	/* 5) hint=2 — exercise further shrink direction */
	if (do_req(fd, 2, &mask, &count) != 0) return 2;
	printf("[step 5] hint=2 -> mask=0x%08x count=%u (popcount=%d)\n",
	       mask, count, popcount32(mask));
	if (popcount32(mask) == 2) {
		printf("  BEHAVIOR: SHRINK to 2 (confirms shrink works at all levels)\n");
		pass++;
	} else if (mask == mask4) {
		printf("  BEHAVIOR: STAY at previous mask (matches step 4 STAY semantic)\n");
		pass++;
	} else {
		printf("  FAIL unexpected popcount=%d\n", popcount32(mask));
		fail++;
	}

	/* 6) hint=8 again — verify grow works after STAY */
	if (do_req(fd, 8, &mask, &count) != 0) return 2;
	printf("[step 6] hint=8 again -> mask=0x%08x count=%u (popcount=%d)\n",
	       mask, count, popcount32(mask));
	if (popcount32(mask) == 8) { printf("  PASS grow back to 8 works\n"); pass++; }
	else                       { printf("  FAIL popcount=%d expected 8\n", popcount32(mask)); fail++; }

	close(fd);

	printf("\n=== Summary ===\n");
	printf("passes: %d, fails: %d\n", pass, fail);
	if (fail > 0) {
		printf("VERDICT: idempotency BROKEN somewhere; B7 poll-thread design must adapt\n");
		return 1;
	}
	/* Print the cardinal finding for B7 design selection: */
	printf("\nIDEMPOTENCY MAP (binding for B7 design):\n");
	printf("  grow on larger hint:  YES (steps 1->3 and 5->6)\n");
	printf("  identical on same:    YES (step 2)\n");
	printf("  on smaller hint:      see steps 4 and 5 BEHAVIOR lines above\n");
	return 0;
}
