// SPDX-License-Identifier: GPL-2.0
/*
 * cipher_model_registry — W7-9 Step 1 G10.
 *
 * Per-model identity registry. Tenants register a (model_path, hf_config_hash,
 * model_arch) tuple via ioctl NR 27 CIPHER_REGISTER_MODEL; this file hands back
 * a stable 128-bit model_uuid keyed on hf_config_hash. Idempotent: same hash
 * always returns the same uuid (lookup-before-insert under spinlock).
 *
 * Hashtable: 256 buckets keyed on the FNV-32 of hf_config_hash[0..7] (cheap
 * for our typical N=10-20 distinct models). Entries hold the 32-byte
 * hf_config_hash for collision-safe key matching.
 *
 * Concurrency: cipher_model_lock serialises register + lookup_by_uuid.
 * Hot-path callers (none in Step 1 — Step 1 is engine-init-only) would
 * read via RCU; Step 1 builds the spinlock-protected API and downstream
 * Steps 3 (G6 AUDIT chain) / W10-12 G3 (KV-dedup) / W10-12 G4 (Marlin) /
 * W13-14 G12 (Koopman) consume.
 *
 * Lifecycle: cipher_model_registry_init() at module load; entries live
 * until cipher_model_registry_exit() at unload (frees all entries). No
 * per-tenant ref counting in Step 1 — entries persist for the lifetime of
 * the kmod load. W10+ may add ref counting for reaper-driven cleanup.
 *
 * ABI: ioctl NR 27, MODULE_VERSION bumped 0.5.0 -> 0.5.5.
 */

#include <linux/module.h>
#include <linux/slab.h>
#include <linux/hashtable.h>
#include <linux/spinlock.h>
#include <linux/random.h>
#include <linux/uaccess.h>
#include <linux/string.h>
#include <linux/limits.h>

#include "cipher_internal.h"
#include "cipher_ioctl.h"

#define CIPHER_MODEL_HASH_BITS 8           /* 256 buckets */

struct cipher_model_entry {
	u8                 hf_config_hash[32];
	u8                 model_uuid[16];
	u32                model_arch;
	struct hlist_node  hnode;
};

static DEFINE_HASHTABLE(cipher_model_hash, CIPHER_MODEL_HASH_BITS);
static DEFINE_SPINLOCK(cipher_model_lock);
static bool cipher_model_registry_ready;

/* FNV-32 of the first 8 bytes of the sha256 hash — sha256 is already a
 * uniform random oracle, so any 4 contiguous bytes is a good bucket index. */
static u32 cipher_model_bucket_key(const u8 hash[32])
{
	return ((u32)hash[0] << 24) | ((u32)hash[1] << 16) |
	       ((u32)hash[2] << 8)  |  (u32)hash[3];
}

/* Caller holds cipher_model_lock. */
static struct cipher_model_entry *
cipher_model_find_locked(const u8 hash[32])
{
	struct cipher_model_entry *e;
	u32 key = cipher_model_bucket_key(hash);

	hash_for_each_possible(cipher_model_hash, e, hnode, key)
		if (memcmp(e->hf_config_hash, hash, 32) == 0)
			return e;
	return NULL;
}

int cipher_model_registry_init(void)
{
	hash_init(cipher_model_hash);
	cipher_model_registry_ready = true;
	pr_info("cipher_kmod: model_registry ready (W7-9 Step 1 G10) — "
	        "%u-bucket hashtable\n", 1u << CIPHER_MODEL_HASH_BITS);
	return 0;
}

void cipher_model_registry_exit(void)
{
	struct cipher_model_entry *e;
	struct hlist_node *tmp;
	int bkt;
	int freed = 0;

	cipher_model_registry_ready = false;

	spin_lock(&cipher_model_lock);
	hash_for_each_safe(cipher_model_hash, bkt, tmp, e, hnode) {
		hash_del(&e->hnode);
		kfree(e);
		freed++;
	}
	spin_unlock(&cipher_model_lock);

	pr_info("cipher_kmod: model_registry torn down; freed %d entries\n",
	        freed);
}

/* Idempotent register: same hf_config_hash always returns the same uuid.
 * Returns 0 on success and writes out_uuid; returns -ENOMEM if kalloc
 * fails; returns -EINVAL if model_arch is unknown. */
int cipher_model_register(const struct cipher_register_model *req,
                          u8 out_uuid[16])
{
	struct cipher_model_entry *e, *neu = NULL;
	u32 key;

	if (!cipher_model_registry_ready)
		return -ENOSYS;

	switch (req->model_arch) {
	case CIPHER_MODEL_ARCH_MISTRAL:
	case CIPHER_MODEL_ARCH_QWEN:
	case CIPHER_MODEL_ARCH_LLAMA:
	case CIPHER_MODEL_ARCH_GPT_NEOX:
	case CIPHER_MODEL_ARCH_OTHER:
		break;
	default:
		return -EINVAL;
	}

	key = cipher_model_bucket_key(req->hf_config_hash);

	/* Optimistic path: lookup under lock. */
	spin_lock(&cipher_model_lock);
	e = cipher_model_find_locked(req->hf_config_hash);
	if (e) {
		memcpy(out_uuid, e->model_uuid, 16);
		spin_unlock(&cipher_model_lock);
		return 0;
	}
	spin_unlock(&cipher_model_lock);

	/* Allocate outside the lock (GFP_KERNEL may sleep). */
	neu = kzalloc(sizeof(*neu), GFP_KERNEL);
	if (!neu)
		return -ENOMEM;
	memcpy(neu->hf_config_hash, req->hf_config_hash, 32);
	neu->model_arch = req->model_arch;
	get_random_bytes(neu->model_uuid, 16);

	/* Race-check: re-take lock, double-check, insert. */
	spin_lock(&cipher_model_lock);
	e = cipher_model_find_locked(req->hf_config_hash);
	if (e) {
		/* Lost the race; consume the existing entry, drop ours. */
		memcpy(out_uuid, e->model_uuid, 16);
		spin_unlock(&cipher_model_lock);
		kfree(neu);
		return 0;
	}
	hash_add(cipher_model_hash, &neu->hnode, key);
	memcpy(out_uuid, neu->model_uuid, 16);
	spin_unlock(&cipher_model_lock);
	return 0;
}

/* Reverse lookup: given a model_uuid, return its hf_config_hash + model_arch.
 * Linear scan (acceptable at N≤256 entries). Returns 0 on hit, -ENOENT on
 * miss. Step 1 builds the API; consumed by W10-12 G3 + G4 / W13-14 G12. */
int cipher_model_lookup_by_uuid(const u8 uuid[16],
                                u8 hf_config_hash_out[32],
                                u32 *model_arch_out)
{
	struct cipher_model_entry *e;
	int bkt;
	int rc = -ENOENT;

	if (!cipher_model_registry_ready)
		return -ENOSYS;

	spin_lock(&cipher_model_lock);
	hash_for_each(cipher_model_hash, bkt, e, hnode) {
		if (memcmp(e->model_uuid, uuid, 16) == 0) {
			if (hf_config_hash_out)
				memcpy(hf_config_hash_out, e->hf_config_hash, 32);
			if (model_arch_out)
				*model_arch_out = e->model_arch;
			rc = 0;
			break;
		}
	}
	spin_unlock(&cipher_model_lock);
	return rc;
}

/* ioctl NR 27 handler. Heap-allocates the ~4 KiB payload per the W6 G1+G2
 * cipher_arena_query kzalloc precedent (stack budget 1024 B). */
long cipher_dev_register_model(unsigned long arg)
{
	struct cipher_register_model *req;
	u8 uuid[16];
	long rc;

	req = kzalloc(sizeof(*req), GFP_KERNEL);
	if (!req)
		return -ENOMEM;

	if (copy_from_user(req, (void __user *)arg, sizeof(*req))) {
		rc = -EFAULT;
		goto out;
	}

	rc = cipher_model_register(req, uuid);
	if (rc)
		goto out;

	memcpy(req->model_uuid, uuid, 16);

	if (copy_to_user((void __user *)arg, req, sizeof(*req)))
		rc = -EFAULT;
out:
	kfree(req);
	return rc;
}
