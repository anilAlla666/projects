/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_stream_resolver — W7-9 Step 5 userspace mirror of the kmod
 * stream-view-slot table. mmap'd into the process address space at startup;
 * provides cipher_v2_current_tenant_id_from_stream() for the cublas/SDPA
 * shim hot path.
 *
 * Lookup is lock-free linear probe over the mmap'd table with a generation
 * counter pre/post check for ABA detection. Cost target: < 100 ns p99
 * uncontended.
 */
#ifndef CIPHER_STREAM_RESOLVER_H
#define CIPHER_STREAM_RESOLVER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Mirror of the kmod's struct cipher_stream_view_slot. Layout MUST stay in
 * sync with cipher_kmod/cipher_stream_registry.h. */
struct cipher_stream_view_slot_user {
    uint64_t stream_handle;
    uint32_t tgid;
    uint32_t tenant_id;
    uint32_t generation;
    uint32_t _pad;
} __attribute__((packed));

#define CIPHER_STREAM_VIEW_SLOTS_USER       8192
#define CIPHER_STREAM_VIEW_MMAP_PGOFF_USER  0x100000  /* must match kmod */
#define CIPHER_STREAM_TENANT_NONE_USER      0xFFFFFFFFu
#define CIPHER_STREAM_MAX_PER_TENANT_USER   16

/* CIPHER_REGISTER_STREAMS payload mirror (NR 29). */
struct cipher_register_streams_user {
    uint32_t tenant_id;
    uint32_t num_streams;
    uint64_t stream_handles[CIPHER_STREAM_MAX_PER_TENANT_USER];
};

/* Initialise the resolver. Opens /dev/cipher (or reuses an existing fd
 * via lazy CAS) and mmaps the view-slot region. Idempotent. Returns 0 on
 * success, -errno on failure. If init fails, lookup degrades to a constant
 * CIPHER_STREAM_TENANT_NONE return (callers default to tenant_id=0).
 *
 * Called from cipher_v2_init_body after cipher_rt_commit_init. */
int cipher_stream_resolver_init(void);

/* Look up (current-process-tgid, stream_handle) -> tenant_id via the mmap'd
 * view-slot table. Returns CIPHER_STREAM_TENANT_NONE_USER on miss or if the
 * resolver isn't initialised.
 *
 * Cost: ~50 ns uncontended (hash + 1-2 cache-line loads). */
uint32_t cipher_v2_current_tenant_id_from_stream(uintptr_t stream_handle);

/* Issue CIPHER_REGISTER_STREAMS ioctl (NR 29) to install the caller's
 * named streams into the kmod registry. Returns 0 on success, -errno on
 * failure. Idempotent on retries of the same (tgid, stream_handle,
 * tenant_id) tuple. */
int cipher_stream_resolver_register(uint32_t tenant_id,
                                    const uintptr_t *stream_handles,
                                    uint32_t num_streams);

#ifdef __cplusplus
}
#endif

#endif /* CIPHER_STREAM_RESOLVER_H */
