/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * libcipher_v2 internal header.
 *
 * Pure-libc injection library loaded by CUDA_INJECTION64_PATH. Reads
 * CIPHER_TENANT_ID at cuInit time and stamps it onto cipher_kmod's
 * per-PID hashtable via /dev/cipher. Nothing else.
 *
 * No CUDA dependency, no CUPTI dependency, no libcuda dependency.
 * Linking footprint: libc.so.6 only.
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

#endif /* CIPHER_V2_INTERNAL_H */
