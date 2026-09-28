/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * libcipher_v2 internal header.
 *
 * CUDA injection library loaded by CUDA_INJECTION64_PATH. At cuInit:
 *   (1) reads CIPHER_TENANT_ID and stamps it via /dev/cipher
 *       (REGISTER_TENANT, ioctl nr 1) -- Phase 2.
 *   (2) subscribes a CUPTI kernel-launch callback that counts
 *       launches and periodically flushes via SUBMIT_LAUNCH_STATS
 *       (ioctl nr 7) -- Phase 3 Task 5.
 *
 * Linking footprint: libc.so.6 + libcupti.so.12 (transitive libcuda).
 */
#ifndef CIPHER_V2_INTERNAL_H
#define CIPHER_V2_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>

#define CIPHER_V2_VERSION  "0.2.0"
#define CIPHER_V2_DEV      "/dev/cipher"
#define CIPHER_V2_TENANT_ENV  "CIPHER_TENANT_ID"
#define CIPHER_V2_DEBUG_ENV   "CIPHER_V2_DEBUG"

/* Logging policy:
 *   cipher_log()    -> always to stderr (load confirmation + errors)
 *   cipher_dbg()    -> stderr only if CIPHER_V2_DEBUG env var is set
 */
#define cipher_log(fmt, ...) \
	fprintf(stderr, "[cipher_v2] " fmt "\n", ##__VA_ARGS__)

#define cipher_dbg(fmt, ...) do { \
	if (getenv(CIPHER_V2_DEBUG_ENV)) \
		fprintf(stderr, "[cipher_v2:dbg] " fmt "\n", ##__VA_ARGS__); \
} while (0)

int cipher_v2_tenant_register(void);
int cipher_v2_cupti_init(void);
int cipher_rt_pr_init(void);    /* Phase 4 T4.2.2 — partition router */
/* cipher_rt_arb_init — RETIRED in CP 5.4 Step 1.3 (ARB / ioctl nr 9). */
int cipher_rt_smp_init(void);   /* Phase 4 T4.2.3 — SM_PACKER */
int cipher_rt_green_ctx_ensure(void);
int cipher_rt_green_ctx_push(void);
void cipher_rt_green_ctx_pop(void);

#endif /* CIPHER_V2_INTERNAL_H */
