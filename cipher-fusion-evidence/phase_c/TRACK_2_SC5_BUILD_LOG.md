# Track 2 SC5-2 — BUILD LOG

**Date:** 2026-05-19. **Phase B of SC5-2** (design memo §a/§d/§e). Builds the
kmod weight-arena fd custodian. Phase A (the VMM cross-process refcount probe,
the load-bearing crux) PASSED — see `TRACK_2_SC5_PROBE_LOG.md`.

## Crux confirmed (Phase A) → the kmod is an fd custodian, never calls CUDA

The probe confirmed on this H100 / CUDA 13: a VMM POSIX-fd shareable handle is
cross-process refcounted by the driver — physical memory survives while any
handle holder is alive; process exit auto-releases its handles. So the kmod
holds a `struct file *` on the exported fd (`fget`) and that reference *is* a
handle keeping the physical alive. The kmod does pure kernel fd-table work
(`fget`/`get_file`/`fd_install`/`fput`); it never calls the CUDA driver.

## New source unit — `cipher_weight_arena.c`

The arena registry. 16 slots (`CIPHER_WA_MAX_ARENAS`), 64 consumer pids each
(`CIPHER_WA_MAX_CONSUMERS`), a 64 KiB opaque metadata blob per arena
(`CIPHER_WA_BLOB_MAX`, stored verbatim, never interpreted — design memo §f).

- `cipher_wa_ioctl_register` (NR 21) — `fget` the producer's exported fd,
  copy in the blob, allocate a slot, return a fresh arena id.
- `cipher_wa_ioctl_import` (NR 22) — by arena id: reserve an fd
  (`get_unused_fd_flags(O_CLOEXEC)`), copy the blob out, then `get_file` +
  `fd_install` the kmod-held file into the caller. Ordering is deliberate —
  on any pre-install abort the reserved number is just `put_unused_fd`'d, no
  installed fd to revoke.
- `cipher_wa_ioctl_leave` (NR 23) / `cipher_wa_ioctl_query` (NR 24).
- **Reaper:** a periodic 5 s `delayed_work` (`cipher_wa_reap_fn`) — checks
  participant liveness via `find_get_pid`/`get_pid_task`, `fput`s the held
  file and frees the slot when an arena's participant count reaches 0. A
  workqueue, **not** a `do_exit` kprobe hook: arena reclamation is not
  latency-critical (the fd ref holds the physical safely meanwhile), and
  this keeps SC5 off the delicate `do_exit` path (design memo §e — the memo
  flagged the workqueue as the permitted option).

## Files touched (all additive — no existing kmod path modified)

| file | change |
|---|---|
| `cipher_weight_arena.c` | **new** — the arena registry (382 lines) |
| `cipher_ioctl.h` | **+** SC5 ABI block — structs + `CIPHER_ARENA_{REGISTER,IMPORT,LEAVE,QUERY}` (NRs 21-24); `CIPHER_WA_BLOB_MAX`, `CIPHER_WA_MAX_ARENAS` |
| `cipher_internal.h` | **+** `cipher_wa_*` prototypes; `CIPHER_PROC_ARENAS` |
| `cipher_dev.c` | **+** 4 ioctl `case`s after `CIPHER_CP54_COMPACT_MIGRATE` |
| `cipher_proc.c` | **+** `/proc/cipher/arenas` node (`cipher_wa_proc_show`) |
| `cipher_main.c` | **+** `cipher_wa_init()` / `cipher_wa_exit()` calls |
| `Kbuild` | **+** `cipher_weight_arena.o` |

## Pre-tarball fix (advisor review)

`cipher_wa_ioctl_register` originally returned `-EFAULT` if the final
writeback `copy_to_user(&p)` failed — but the arena was already created and
the fd `fget`'d, so it sat live-but-orphaned (caller never learned its id)
until the reaper noticed. Fixed: on that failure path, re-take the lock and
`wa_release` the just-created arena before returning `-EFAULT`.

## Build & load

```
make -C /lib/modules/6.8.0-1046-nvidia/build M=/home/ubuntu/cipher_kmod modules
  CC [M]  cipher_weight_arena.o      # clean — no warnings
  LD [M]  cipher_kmod.ko
```

Clean build (only the pre-existing benign "compiler differs" kernel-build
note). `rmmod` + `insmod` of the new `.ko`: loads clean.

```
cipher_kmod: Track 2 SC5 weight-arena registry — 16 slots, 5s liveness reaper
/proc/cipher/{stats,bar0_state,gpu_state,flops,migrations,arenas}   # arenas live
```

## Anchor rotation

| artifact | pre-SC5 | post-SC5 | preserved as |
|---|---|---|---|
| `cipher_kmod.ko` | `285d102e` | **`008b3c66`** | pre: `cipher_kmod.ko.pre_track2_sc5`; post: `cipher_kmod.ko.track2_sc5` |
| `libcipher_rt.so` | `83afd1ca` | `83afd1ca` | byte-identical (untouched) |
| `cipher_kv_bridge.so` | `c04b0c39` | `c04b0c39` | byte-identical (untouched) |
| `libcipher_v2.so` | `cc0479b8` | `cc0479b8` | byte-identical (untouched) |

Source tarballed: `cipher_kmod_fallback/cipher_kmod_src_track2_sc5.tar.gz`.
**New campaign kmod anchor: `008b3c66`** ([[cipher-abi-rule]] honored — NRs
21-24 are fresh; no reserved nr repurposed).
