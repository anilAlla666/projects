// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_audit_chain — W7-9 Step 3 G6: kmod-resident AUDIT chain.
 *
 * Per CIPHER_REENGINEERING_PLAN.md v1.2.3 §4.8 Step 1 (AUDIT chain HMAC
 * entry). Closes architecture-gap-audit §3.1 G6 (chain was userspace-only +
 * per-process; v1.2.3 G6 makes it kmod-resident + per-tenant; survives
 * tenant process crashes; externally verifiable from the mmap'd ring).
 *
 * Storage:
 *   Per-tenant ring buffer of `cipher_audit_entry` (64 B each, single
 *   cache line). CIPHER_AUDIT_RING_ENTRIES = 4096 -> 256 KiB per tenant.
 *   CIPHER_RT_MAX_TENANTS (= CIPHER_CP54_MAX_ALLOCS post-G1 = 128) tenants
 *   -> 32 MiB total. Backing memory is vmalloc'd at init (one-shot, no
 *   physical pinning until first write per page).
 *
 * Mapping to userspace:
 *   Read-only mmap via cipher_dev_fops.mmap (Path X from W7-9 scope-lock —
 *   extend existing /dev/cipher fops rather than new /dev/cipher_audit
 *   device). vma->vm_pgoff = tenant_id picks the per-tenant view; userspace
 *   passes the desired tenant_id as the offset to mmap.
 *
 * Cryptography:
 *   HMAC-SHA256(per-tenant key, prev_chain_head || payload). Key derived at
 *   first audit_chain_record via get_random_bytes (lazily; persisted in the
 *   ring's key field). Chain head is the latest HMAC; stored in
 *   cipher_pid_stats.hmac_chain_head (Step 3 struct grow) for fast access
 *   via the existing GET_TENANT_SNAPSHOT ioctl + via the mmap'd ring.
 *
 *   Verification (offline): /tmp/step3_baseline/verify_audit_chain.c walks
 *   the mmap'd ring, re-computes the HMAC chain from the seed using each
 *   tenant's key (also in the ring), and compares each entry's payload_hash
 *   to the recomputed value. PASS = real chain integrity.
 *
 * Concurrency:
 *   Per-tenant single-writer (COMMIT caller on tenant's thread) /
 *   multi-reader (offline verifier mmap reader). atomic head index.
 *   No cross-tenant ordering — each tenant's ring is independent.
 *
 * Hot-path cost: HMAC-SHA256 over (32 B prev + N B payload). With N=0
 * (no payload from COMMIT Step 1 placeholder use), this is a single block
 * SHA-256 (~200 ns on Hopper-era x86). Budget per spec: 1 µs.
 */

#include <linux/module.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/random.h>
#include <linux/atomic.h>
#include <linux/string.h>
#include <linux/sched.h>
#include <crypto/sha2.h>

#include "cipher_internal.h"
#include "cipher_ioctl.h"

#define CIPHER_AUDIT_RING_ENTRIES 4096
#define CIPHER_AUDIT_ENTRY_BYTES  64
#define CIPHER_AUDIT_RING_MASK   (CIPHER_AUDIT_RING_ENTRIES - 1)

/* Compile-time sanity: ring_entries must be a power of two for cheap
 * mask-indexing. */
_Static_assert((CIPHER_AUDIT_RING_ENTRIES & CIPHER_AUDIT_RING_MASK) == 0,
               "ring entries must be power of two");

struct cipher_audit_entry {
	__u64 timestamp_ns;
	__u64 commit_seq;
	__u8  payload_hash[32];   /* HMAC-SHA256(prev_chain_head || payload) */
	__u32 actuator_id;
	__u32 flags;
	__u8  _pad[8];
};
_Static_assert(sizeof(struct cipher_audit_entry) == CIPHER_AUDIT_ENTRY_BYTES,
               "audit entry must be exactly 64 B");

/* Per-tenant ring header — 128 B (two cache lines). Layout chosen so the
 * `chain_head[32]` mirror is in the FIRST cache line (writer-hot) and the
 * `hmac_key[32]` is in the second (write-once at key init). */
struct cipher_audit_ring {
	atomic_t head;                            /* writer-only, monotonic; modulo RING_ENTRIES */
	atomic_t tail;                            /* reserved for consumer cursor */
	__u32    key_initialised;                 /* 0 until first record (u32 for alignment) */
	__u32    _pad_a;
	__u64    record_count;                    /* relaxed-atomic via WRITE_ONCE; for /proc + verifier */
	__u8     chain_head[32];                  /* latest HMAC; mirrored to cipher_tenant_snapshot.hmac_chain_head by Step 4 */
	__u8     _pad_b[8];                        /* fill first cache line (64 B) */
	__u8     hmac_key[32];                    /* per-tenant key; get_random_bytes at first record */
	__u8     _pad_c[32];                       /* fill second cache line */
	struct cipher_audit_entry entries[CIPHER_AUDIT_RING_ENTRIES];
};
_Static_assert(sizeof(struct cipher_audit_ring)
               == (128 + CIPHER_AUDIT_RING_ENTRIES * 64),
               "audit ring header must be exactly 128 B + entries");

#define CIPHER_AUDIT_RING_BYTES sizeof(struct cipher_audit_ring)
#define CIPHER_AUDIT_TOTAL_BYTES (CIPHER_AUDIT_RING_BYTES * CIPHER_RT_MAX_TENANTS)

/* Backing storage, vmalloc'd at init. Each tenant's ring is contiguous;
 * the whole vmalloc'd region is mmap'able by userspace consumers
 * (read-only). */
static struct cipher_audit_ring *cipher_audit_rings;
static atomic_t cipher_audit_initialised;

/* HMAC-SHA256 of (msg) keyed by (key, key_len). key_len must be <= 64 B
 * (block size). Output written to out[32]. Manual construction because
 * Linux kernel doesn't expose a one-shot HMAC-SHA256 inline; we use the
 * sha256_init/update/final primitives from <crypto/sha2.h>. */
static void cipher_hmac_sha256(const __u8 *key, size_t key_len,
                               const __u8 *msg, size_t msg_len,
                               __u8 out[32])
{
	struct sha256_state ctx;
	__u8 ipad[SHA256_BLOCK_SIZE];
	__u8 opad[SHA256_BLOCK_SIZE];
	__u8 inner_digest[32];
	__u8 k_padded[SHA256_BLOCK_SIZE];
	size_t i;

	/* Key padding: zero-pad to block size (64 B). RFC 2104 says keys
	 * longer than block_size are SHA256'd first; we always pass 32-B
	 * keys so the zero-pad branch is taken. */
	memset(k_padded, 0, sizeof(k_padded));
	if (key_len > SHA256_BLOCK_SIZE) {
		sha256_init(&ctx);
		sha256_update(&ctx, key, key_len);
		sha256_final(&ctx, k_padded);   /* only fills first 32 B; rest zero */
	} else {
		memcpy(k_padded, key, key_len);
	}

	for (i = 0; i < SHA256_BLOCK_SIZE; i++) {
		ipad[i] = k_padded[i] ^ 0x36;
		opad[i] = k_padded[i] ^ 0x5c;
	}

	/* inner = SHA256(ipad || msg) */
	sha256_init(&ctx);
	sha256_update(&ctx, ipad, SHA256_BLOCK_SIZE);
	if (msg && msg_len)
		sha256_update(&ctx, msg, msg_len);
	sha256_final(&ctx, inner_digest);

	/* outer = SHA256(opad || inner) */
	sha256_init(&ctx);
	sha256_update(&ctx, opad, SHA256_BLOCK_SIZE);
	sha256_update(&ctx, inner_digest, 32);
	sha256_final(&ctx, out);
}

/* Ensure the per-tenant HMAC key is initialised. Lazily on first record. */
static void cipher_audit_ensure_key(struct cipher_audit_ring *r)
{
	if (READ_ONCE(r->key_initialised))
		return;
	get_random_bytes(r->hmac_key, sizeof(r->hmac_key));
	smp_wmb();
	WRITE_ONCE(r->key_initialised, 1);
}

int cipher_audit_chain_init(void)
{
	if (atomic_cmpxchg(&cipher_audit_initialised, 0, 1) != 0)
		return 0;                                       /* already initialised */

	cipher_audit_rings = vmalloc_user(CIPHER_AUDIT_TOTAL_BYTES);
	if (!cipher_audit_rings) {
		atomic_set(&cipher_audit_initialised, 0);
		pr_err("cipher_kmod: audit_chain_init: vmalloc(%llu) failed\n",
		       (unsigned long long)CIPHER_AUDIT_TOTAL_BYTES);
		return -ENOMEM;
	}
	/* vmalloc_user returns zero-filled memory. atomic_t head/tail and
	 * key_initialised are already zero; entries[] are zero. */
	pr_info("cipher_kmod: audit_chain ready (W7-9 Step 3 G6) — %u tenants, "
	        "%u entries/tenant, %llu MiB vmalloc'd\n",
	        CIPHER_RT_MAX_TENANTS, CIPHER_AUDIT_RING_ENTRIES,
	        (unsigned long long)(CIPHER_AUDIT_TOTAL_BYTES >> 20));
	return 0;
}

void cipher_audit_chain_exit(void)
{
	if (atomic_cmpxchg(&cipher_audit_initialised, 1, 0) != 1)
		return;
	vfree(cipher_audit_rings);
	cipher_audit_rings = NULL;
	pr_info("cipher_kmod: audit_chain torn down\n");
}

int cipher_audit_chain_record(u32 tenant_id, u64 commit_seq,
                              u32 actuator_id,
                              const u8 *payload, size_t payload_len)
{
	struct cipher_audit_ring *r;
	struct cipher_audit_entry entry;
	__u8 buf[32 + 128];                                  /* prev_head + payload (cap 128 B) */
	size_t msg_len;
	int head_idx;

	if (!atomic_read(&cipher_audit_initialised) || !cipher_audit_rings)
		return -ENOSYS;
	if (tenant_id >= CIPHER_RT_MAX_TENANTS)
		return -EINVAL;
	if (payload_len > 128)
		return -EINVAL;

	r = &cipher_audit_rings[tenant_id];
	cipher_audit_ensure_key(r);

	/* Read prev chain head from the per-tenant ring. First record uses
	 * an implicit zero seed (r->chain_head is zero-initialised by
	 * vmalloc_user). The chain is keyed by the per-tenant hmac_key, so
	 * an attacker who doesn't know the key cannot forge entries even
	 * with a known seed. */
	memcpy(buf, r->chain_head, 32);
	msg_len = 32;
	if (payload && payload_len) {
		memcpy(buf + 32, payload, payload_len);
		msg_len += payload_len;
	}

	/* Compute new chain head: HMAC-SHA256(key, prev_head || payload). */
	cipher_hmac_sha256(r->hmac_key, sizeof(r->hmac_key),
	                   buf, msg_len,
	                   entry.payload_hash);

	/* Fill remaining entry fields. */
	entry.timestamp_ns = ktime_get_ns();
	entry.commit_seq   = commit_seq;
	entry.actuator_id  = actuator_id;
	entry.flags        = 0;
	memset(entry._pad, 0, sizeof(entry._pad));

	/* Single-writer-per-tenant: atomic_inc + modulo. The COMMIT caller
	 * on this tenant is the sole writer; head atomicity is for the
	 * userspace mmap reader (offline verifier) to see a coherent
	 * monotonic count. */
	head_idx = atomic_fetch_inc(&r->head) & CIPHER_AUDIT_RING_MASK;
	r->entries[head_idx] = entry;

	/* Publish the new chain head. memcpy is sufficient: the matching
	 * atomic_inc(&r->head) above orders writes before this update for
	 * any reader that loads head first (acquire) then chain_head. */
	memcpy(r->chain_head, entry.payload_hash, 32);
	WRITE_ONCE(r->record_count, r->record_count + 1);
	return 0;
}

/* fops.mmap handler. Read-only mapping of the entire vmalloc'd ring region.
 * vma->vm_pgoff = 0 -> map from offset 0; userspace indexes per-tenant by
 * computing tenant_id * CIPHER_AUDIT_RING_BYTES. */
int cipher_audit_chain_mmap(struct file *filp, struct vm_area_struct *vma)
{
	unsigned long vsize = vma->vm_end - vma->vm_start;

	(void)filp;

	if (!atomic_read(&cipher_audit_initialised) || !cipher_audit_rings)
		return -ENOSYS;

	/* Read-only: refuse VM_WRITE. */
	if (vma->vm_flags & VM_WRITE)
		return -EPERM;

	/* Allow up to the full ring region; partial maps OK. */
	if (vsize == 0 || vsize > CIPHER_AUDIT_TOTAL_BYTES)
		return -EINVAL;

	/* vma->vm_pgoff is the page offset into the region the caller wants
	 * to start at. remap_vmalloc_range honours this directly. */
	return remap_vmalloc_range(vma, cipher_audit_rings, vma->vm_pgoff);
}

/* For test_audit_chain.c and other in-process consumers: return the total
 * mmap-able region size. */
u64 cipher_audit_chain_region_bytes(void)
{
	return CIPHER_AUDIT_TOTAL_BYTES;
}

/* For verify_audit_chain.c offline reproduction: a helper that wraps the
 * HMAC-SHA256 primitive. Not exported as an ioctl (verifier reproduces the
 * HMAC in userspace using the same key from the mmap'd ring). */
void cipher_audit_chain_hmac(const __u8 *key, size_t key_len,
                             const __u8 *msg, size_t msg_len,
                             __u8 out[32])
{
	cipher_hmac_sha256(key, key_len, msg, msg_len, out);
}

/* ioctl NR 28 CIPHER_AUDIT_RECORD handler. Heap-allocates the payload
 * buffer per the W6 G1+G2 cipher_arena_query kzalloc precedent (payload
 * is up to 128 B but the request struct is well under 1 KiB; we still
 * kzalloc for safety + future growth). */
long cipher_dev_audit_record(unsigned long arg)
{
	struct cipher_audit_record_req *req;
	long rc;

	req = kzalloc(sizeof(*req), GFP_KERNEL);
	if (!req)
		return -ENOMEM;
	if (copy_from_user(req, (void __user *)arg, sizeof(*req))) {
		rc = -EFAULT;
		goto out;
	}
	if (req->payload_len > sizeof(req->payload)) {
		rc = -EINVAL;
		goto out;
	}
	rc = cipher_audit_chain_record(req->tenant_id, req->commit_seq,
	                               req->actuator_id,
	                               req->payload, req->payload_len);
out:
	kfree(req);
	return rc;
}
