/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * cipher_stream_resolver — userspace side of W7-9 Step 5 multi-tenant
 * resolver. Mirrors the kmod's vmalloc'd view-slot table via mmap and
 * provides a < 100 ns linear-probe lookup for the cublas/SDPA shim hot
 * path.
 */
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>

#include "cipher_stream_resolver.h"

/* CIPHER_REGISTER_STREAMS ioctl NR 29 — _IOW('C', 29, struct cipher_register_streams). */
#define CIPHER_IOCTL_MAGIC_R 'C'
#define _IOC_NRBITS_R    8
#define _IOC_TYPEBITS_R  8
#define _IOC_SIZEBITS_R  14
#define _IOC_DIRBITS_R   2
#define _IOC_NRSHIFT_R   0
#define _IOC_TYPESHIFT_R (_IOC_NRSHIFT_R + _IOC_NRBITS_R)
#define _IOC_SIZESHIFT_R (_IOC_TYPESHIFT_R + _IOC_TYPEBITS_R)
#define _IOC_DIRSHIFT_R  (_IOC_SIZESHIFT_R + _IOC_SIZEBITS_R)
#define _IOC_NONE_R   0u
#define _IOC_WRITE_R  1u
#define _IOC_R(dir, nr, sz) (((dir) << _IOC_DIRSHIFT_R) | \
                             (CIPHER_IOCTL_MAGIC_R << _IOC_TYPESHIFT_R) | \
                             ((nr) << _IOC_NRSHIFT_R) | \
                             ((sz) << _IOC_SIZESHIFT_R))
#define CIPHER_REGISTER_STREAMS_R \
    _IOC_R(_IOC_WRITE_R, 29, sizeof(struct cipher_register_streams_user))

/* Page size on Linux x86 H100 pods is 4 KiB. */
#define CIPHER_PAGE_SIZE_USER 4096

static atomic_int g_fd = -1;
static struct cipher_stream_view_slot_user *g_view = NULL;
static uint32_t g_self_tgid = 0;

/* FNV-1a-ish 64-bit mix — must match kmod cipher_stream_hash exactly. */
static inline uint32_t resolver_hash(uint32_t tgid, uint64_t handle)
{
    uint64_t h = (uint64_t)tgid * 0x100000001b3ULL;
    h ^= handle * 0x100000001b3ULL;
    h ^= h >> 32;
    return (uint32_t)(h & (CIPHER_STREAM_VIEW_SLOTS_USER - 1));
}

static int resolver_open_fd(void)
{
    int fd = atomic_load_explicit(&g_fd, memory_order_acquire);
    if (fd >= 0) return fd;
    fd = open("/dev/cipher", O_RDWR | O_CLOEXEC);
    if (fd < 0) return -errno;
    int expected = -1;
    if (!atomic_compare_exchange_strong_explicit(
            &g_fd, &expected, fd,
            memory_order_acq_rel, memory_order_acquire)) {
        close(fd);
        fd = expected;
    }
    return fd;
}

int cipher_stream_resolver_init(void)
{
    if (g_view) return 0;                               /* idempotent */
    int fd = resolver_open_fd();
    if (fd < 0) return fd;
    /* mmap the view-slot region read-only. Userspace asks for it at the
     * high offset CIPHER_STREAM_VIEW_MMAP_PGOFF * PAGE_SIZE; the kmod's
     * multiplex dispatcher strips that base before delegating to
     * remap_vmalloc_range. */
    size_t size = (size_t)CIPHER_STREAM_VIEW_SLOTS_USER
                  * sizeof(struct cipher_stream_view_slot_user);
    /* round up to page boundary */
    size = (size + CIPHER_PAGE_SIZE_USER - 1) & ~(size_t)(CIPHER_PAGE_SIZE_USER - 1);
    off_t off = (off_t)CIPHER_STREAM_VIEW_MMAP_PGOFF_USER * CIPHER_PAGE_SIZE_USER;
    void *p = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, off);
    if (p == MAP_FAILED) return -errno;
    g_view = (struct cipher_stream_view_slot_user *)p;
    g_self_tgid = (uint32_t)getpid();                   /* tgid == getpid() for single-threaded leader */
    return 0;
}

uint32_t cipher_v2_current_tenant_id_from_stream(uintptr_t stream_handle)
{
    if (!g_view || stream_handle == 0) return CIPHER_STREAM_TENANT_NONE_USER;
    uint32_t tgid = g_self_tgid;
    uint32_t idx = resolver_hash(tgid, (uint64_t)stream_handle);
    uint32_t probes = 0;
    while (probes < CIPHER_STREAM_VIEW_SLOTS_USER) {
        const struct cipher_stream_view_slot_user *s = &g_view[idx];
        uint32_t g1 = __atomic_load_n(&s->generation, __ATOMIC_ACQUIRE);
        uint64_t h  = __atomic_load_n(&s->stream_handle, __ATOMIC_RELAXED);
        if (h == 0) return CIPHER_STREAM_TENANT_NONE_USER;  /* probe terminator */
        if (h == (uint64_t)stream_handle &&
            __atomic_load_n(&s->tgid, __ATOMIC_RELAXED) == tgid) {
            uint32_t tid = __atomic_load_n(&s->tenant_id, __ATOMIC_RELAXED);
            uint32_t g2 = __atomic_load_n(&s->generation, __ATOMIC_ACQUIRE);
            if (g1 == g2) return tid;
            /* generation churn during read — retry this slot */
            continue;
        }
        idx = (idx + 1) & (CIPHER_STREAM_VIEW_SLOTS_USER - 1);
        probes++;
    }
    return CIPHER_STREAM_TENANT_NONE_USER;
}

int cipher_stream_resolver_register(uint32_t tenant_id,
                                    const uintptr_t *stream_handles,
                                    uint32_t num_streams)
{
    if (num_streams == 0 || num_streams > CIPHER_STREAM_MAX_PER_TENANT_USER)
        return -EINVAL;
    int fd = resolver_open_fd();
    if (fd < 0) return fd;
    struct cipher_register_streams_user req;
    memset(&req, 0, sizeof(req));
    req.tenant_id   = tenant_id;
    req.num_streams = num_streams;
    for (uint32_t i = 0; i < num_streams; i++) {
        req.stream_handles[i] = (uint64_t)stream_handles[i];
    }
    if (ioctl(fd, CIPHER_REGISTER_STREAMS_R, &req) < 0) {
        return -errno;
    }
    return 0;
}
