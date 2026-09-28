# Track 2 — /dev/cipher ABI additions

Companion to `TRACK_3_ABI.md`. Records the ioctl numbers Track 2 SC5 adds to
the `/dev/cipher` interface. Per [[cipher-abi-rule]]: ioctl `nr`s are
**additive only**; assigned `nr`s are permanent; reserved `nr`s 2/3/4 return
`-ENOSYS`; new ioctls take fresh `nr`s.

## NR allocation map (magic `'C'`)

| nr | owner | ioctl |
|---|---|---|
| 1 | Phase 2 | `CIPHER_REGISTER_TENANT` |
| 2,3,4 | — | reserved → `-ENOSYS` |
| 5–12 | Phase 3/4 | tenant snapshot, GPU state, FLOP series, clock, … |
| 13,14 | CP 5.4 | SM-group allocator (ALLOCATE / FREE / QUERY family) |
| 15 | CP 5.4 | (CP54 scheduler) |
| 16–20 | **Track 3** | DSM migration state machine (`TRACK_3_ABI.md`) |
| **21** | **Track 2 SC5** | **`CIPHER_ARENA_REGISTER`** |
| **22** | **Track 2 SC5** | **`CIPHER_ARENA_IMPORT`** |
| **23** | **Track 2 SC5** | **`CIPHER_ARENA_LEAVE`** |
| **24** | **Track 2 SC5** | **`CIPHER_ARENA_QUERY`** |

NRs 21-24 are **permanently** assigned to these four ioctls. Future
maintainers must not collapse the numbering or repurpose them.

## SC5 weight-arena fd custodian — NRs 21-24

Cross-tenant weight sharing: a producer REGISTERs its exported CUDA VMM
POSIX fd with the kmod; peer tenants IMPORT a dup'd fd by arena id — whether
or not the producer is still alive. The kmod is the fd **custodian**: it
holds the `struct file *`, never calls the CUDA driver (the driver already
cross-process-refcounts the physical memory). Opaque metadata blobs (the SC3
layout manifest + SC4 fingerprint) pass by a userspace pointer so the ioctl
structs stay small and within the `_IOC` 14-bit size field.

### NR 21 — `CIPHER_ARENA_REGISTER` — `_IOWR('C',21,struct cipher_arena_register)`

```c
struct cipher_arena_register {
	__s32 fd;            /* in : exported VMM POSIX fd                  */
	__u32 arena_id_out;  /* out: assigned arena id (>= 1)               */
	__u64 base;          /* in : arena base devptr                      */
	__u64 size;          /* in : arena byte size                        */
	__u64 blob_ptr;      /* in : userspace ptr to the metadata blob     */
	__u32 blob_len;      /* in : blob length, <= CIPHER_WA_BLOB_MAX     */
	__u32 reserved[5];
};   /* sizeof == 56 */
```
The kmod `fget`s `fd` (an independent reference), copies in the blob,
allocates a registry slot, sets `producer_pid = current->pid`, returns the
new `arena_id_out`. Errors: `-EBADF` (bad fd), `-ENOSPC` (16 slots full),
`-EINVAL` (blob too large), `-EFAULT` (bad userspace ptr).

### NR 22 — `CIPHER_ARENA_IMPORT` — `_IOWR('C',22,struct cipher_arena_import)`

```c
struct cipher_arena_import {
	__u32 arena_id;      /* in : arena to import                        */
	__s32 fd_out;        /* out: a fd in the caller, dup'd from kmod's  */
	__u64 base;          /* out: arena base devptr                      */
	__u64 size;          /* out: arena byte size                        */
	__u64 blob_ptr;      /* in : userspace OUT buffer for the blob      */
	__u32 blob_cap;      /* in : blob_ptr buffer capacity               */
	__u32 blob_len;      /* out: actual blob length                     */
	__u32 reserved[4];
};   /* sizeof == 56 */
```
The kmod reserves an fd (`get_unused_fd_flags(O_CLOEXEC)`), copies the blob
out, then `get_file` + `fd_install`s its held file into the caller, and adds
`current->pid` to the arena's consumer set. Works with **no producer
process alive**. Errors: `-ENOENT` (no such arena), `-ENOSPC` (64 consumer
slots full, or `blob_cap < blob_len`), `-EFAULT`.

### NR 23 — `CIPHER_ARENA_LEAVE` — `_IOW('C',23,__u32)`

Argument: the `__u32` arena id. Removes `current->pid` from the arena's
participant set (producer or consumer). When the set empties the kmod
`fput`s its held file and frees the slot. The clean-exit fast path; the
periodic reaper is the crash path. Errors: `-ENOENT`, `-EFAULT`.

### NR 24 — `CIPHER_ARENA_QUERY` — `_IOR('C',24,struct cipher_arena_query)`

```c
struct cipher_arena_query {
	__u32 n_arenas;       /* out */
	__u32 reserved;
	struct {
		__u32 arena_id;
		__u32 producer_pid;   /* 0 once the producer has exited */
		__u32 n_consumers;
		__u32 _pad;
		__u64 size;
	} arenas[16];         /* CIPHER_WA_MAX_ARENAS */
};   /* sizeof == 392 */
```
Read-only operator snapshot. Also surfaced human-readable at
`/proc/cipher/arenas`.

## Lifetime semantics (design memo §b)

An arena is **alive while any participant (producer + consumers) is
registered**. The kmod drops its held `struct file *` (`fput`) when the
participant count reaches **0**, reaper-driven. A producer that REGISTERs
and then exits with **no consumer ever having joined** is reaped on the next
5 s cycle — `IMPORT`-after-producer-death requires at least one consumer to
have joined while the producer was alive (i.e. count never hit 0). This is
the SC5 v1 boundary; a registration grace window is a possible v2 hardening.
