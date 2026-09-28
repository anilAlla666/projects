/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_kvdedup.h — T4.6.4 /dev/cipher_kvdedup ioctl ABI.
 *
 * Shared by the kmod (cipher_kvdedup.c) and userspace (cipher_rt).
 * Cross-tenant KV page deduplication: the refcount table lives in
 * kernel memory; tenant processes ioctl in. See t4_6_4_design.md.
 *
 * Userspace computes the xxhash64 of the 2 MiB page and the
 * memcmp-verify on a hit; the kmod owns the table, the refcounts,
 * and the cuIpc POSIX-FD handles.
 */
#ifndef CIPHER_KVDEDUP_H
#define CIPHER_KVDEDUP_H

#include <linux/types.h>

#define CIPHER_KVDEDUP_MAGIC     'K'
#define CIPHER_KVDEDUP_DEV_NAME  "cipher_kvdedup"

/* PUT flags */
#define CIPHER_KVDEDUP_FLAG_FORCE_NEW  0x1u  /* skip the bucket walk —
                                              * register a fresh entry
                                              * (used after a userspace
                                              * memcmp collision) */
/* PUT result codes */
#define CIPHER_KVDEDUP_RESULT_MISS  0u  /* new content — entry registered */
#define CIPHER_KVDEDUP_RESULT_HIT   1u  /* hash hit — verify candidate_fd */

struct cipher_kvdedup_init {
	__u32 tenant_id;     /* out */
	__u32 _pad;
};

struct cipher_kvdedup_put {
	__u64 content_hash;  /* in : xxhash64 of the 2 MiB page */
	__s32 export_fd;     /* in : cuMem POSIX shareable fd of the caller's
	                      *      freshly-created physical page */
	__u32 flags;         /* in : CIPHER_KVDEDUP_FLAG_* */
	__u32 result;        /* out: CIPHER_KVDEDUP_RESULT_* */
	__u32 _pad;
	__u64 pool_offset;   /* out: id of the (new or existing) entry */
	__s32 candidate_fd;  /* out: HIT only — fresh fd to the existing
	                      *      entry's handle for userspace memcmp;
	                      *      -1 on MISS */
	__u32 _pad2;
};

struct cipher_kvdedup_confirm {
	__u64 pool_offset;   /* in : pool_offset from a HIT_CANDIDATE put,
	                      *      after userspace memcmp confirmed match */
};

struct cipher_kvdedup_free {
	__u64 pool_offset;   /* in */
};

struct cipher_kvdedup_stats {
	__u64 entries;            /* unique physical pages registered */
	__u64 virtual_refs;       /* sum of refcounts across all entries */
	__u64 puts;
	__u64 hits;
	__u64 misses;
	__u64 collisions_forced;  /* FORCE_NEW puts */
	__u64 refcount_releases;  /* physical pages freed at refcount 0 */
	__u32 tenants_open;
	__u32 _pad;
};

#define CIPHER_KVDEDUP_INIT \
	_IOR(CIPHER_KVDEDUP_MAGIC, 1, struct cipher_kvdedup_init)
#define CIPHER_KVDEDUP_PUT \
	_IOWR(CIPHER_KVDEDUP_MAGIC, 2, struct cipher_kvdedup_put)
#define CIPHER_KVDEDUP_CONFIRM \
	_IOW(CIPHER_KVDEDUP_MAGIC, 3, struct cipher_kvdedup_confirm)
#define CIPHER_KVDEDUP_FREE \
	_IOW(CIPHER_KVDEDUP_MAGIC, 4, struct cipher_kvdedup_free)
#define CIPHER_KVDEDUP_STATS \
	_IOR(CIPHER_KVDEDUP_MAGIC, 5, struct cipher_kvdedup_stats)

/* Per-tenant tracked-page cap (design memo C). */
#define CIPHER_KVDEDUP_MAX_PER_TENANT  8192
/* Global registered-entry cap (design memo G). */
#define CIPHER_KVDEDUP_MAX_ENTRIES     (1u << 20)

#endif /* CIPHER_KVDEDUP_H */
