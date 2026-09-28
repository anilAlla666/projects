# Phase C / Track 2 — Weight-Sharing — SC5 DESIGN MEMO

**Date:** 2026-05-19. **Type:** design/scope — **paperwork only**, no source
modified, no GPU. STOP for adjudication before SC5-2. **Predecessor:** Track 2
SC4 closed (`TRACK_2_SC4_CLOSEOUT.md`).

SC5 makes the shared arena's lifetime **not depend on the producer process** —
the safety primitive for production: a producer must be able to crash/exit
without stranding the arena, and a consumer must be able to join after the
producer is gone. SC3/SC4 assumed "producer stays alive"; SC4's
`model._cipher_weight_arena` pin is an interim user-space fix SC5 supersedes.

---

## §1 — THE CRUX: CUDA VMM already refcounts the *physical* memory

Before designing a kmod arena manager, one fact decides the whole shape of
SC5. **CUDA VMM POSIX-fd shareable handles are cross-process reference-counted
by the driver.** `cuMemCreate` makes a physical allocation;
`cuMemExportToShareableHandle(POSIX_FD)` + `cuMemImportFromShareableHandle` in
a peer create independent handles referencing the *same* physical; the
physical is freed only when **every** handle (producer's + all consumers') is
released — and process exit (crash or clean) auto-releases that process's
handles. So the **physical-memory lifetime is already "alive while any handle
holder is alive"**, and a producer crash already does **not** free the
physical out from under a live consumer.

What SC3/SC4 actually hit was *not* a cross-process lifetime failure — it was
a consumer **prematurely releasing its own handle** (the SC4 GC bug). And what
SC3's "producer stays alive" assumption actually protected was **not** the
memory — it was the **fd hand-off**: SC3 passes the fd over a UNIX socket from
a *live* producer (`SCM_RIGHTS`); once the producer exits, that socket is gone,
so a *new* consumer has no way to get the fd.

**Therefore SC5's real job is the fd RENDEZVOUS, not memory management.** The
kmod becomes the **fd custodian** — it holds the exported fd so (a) the
physical stays alive while the kmod holds it, and (b) consumers can obtain the
fd from the kmod whether or not the producer is alive. The kmod **never calls
the CUDA driver** — it `fget`/`fput`/`dup`s a kernel file object; the fd's
existence *is* a handle reference. This **sidesteps item (c) entirely** (no
userspace cleanup daemon, no kmod→CUDA calls).

**SC5-2 MUST begin with an empirical probe** confirming this VMM behavior on
this H100/CUDA 13 — specifically: (1) a consumer keeps a working mapping after
the producer process exits; (2) the physical frees only after the last handle
(incl. the kmod's `struct file`) is dropped. If the probe contradicts §1, the
design is revisited *before* any kmod code. The T4.6.4 precedent ("POSIX-FD
handles pass through the kmod cleanly — it already stores/`fput`s them") is
strong supporting evidence the custodian model works.

## §a — Arena registry in the kmod

A new kmod arena registry (a fresh source unit, e.g. `cipher_weight_arena.c`).
Per-arena entry: producer pid, **consumer pid set** (multi-consumer — SC3/SC4
were single-consumer), `base`, `size`, the kmod-held **`struct file *`** for
the exported fd, and an **opaque metadata blob** (the producer's JSON layout
manifest + SC4 fingerprint — the kmod stores bytes, never interprets them,
§f). New ioctls (§d). Reaper integration (§e).

## §b — Lifetime semantics — "alive while any participant is registered"

The kmod holds **one `struct file *`** reference to the exported fd from arena
registration onward. While the kmod holds it, the cuMem shareable handle's
underlying object stays alive → the physical stays alive — *independent of the
producer*. The kmod **drops** that reference (`fput`) when the arena's
participant count (producer + consumers) reaches **0**, reaper-driven (§e).
- *Producer crash mid-consumer-import:* the kmod already holds the fd ref
  (taken at REGISTER); the producer crashing only removes its pid from the set
  — consumers and the import are unaffected. The race SC3 could not handle is
  closed by the kmod-held ref.
- *Consumer crash mid-rebind:* the reaper decrements; the physical is
  unaffected while others remain; freed when the set empties.

## §c — cuMemRelease / cuMemAddressFree — the kmod never calls CUDA

Resolved by §1's custodian model. The kmod manages the **fd** (`fget` at
REGISTER, `dup` into a consumer's table at IMPORT, `fput` when the participant
set empties) — all pure kernel fd-table operations, no CUDA. Each importing
process still does its own `cuMemRelease`/`cuMemUnmap`/`cuMemAddressFree` for
its *own* mapping on exit (the driver does this automatically on process
teardown). The physical frees when the last reference — every process handle
**plus** the kmod's `fput` — is gone. **No userspace cleanup daemon; no
kmod→CUDA path.**

## §d — ABI — 4 new ioctls, NRs 21-24, additive

Per [[cipher-abi-rule]] / the Track 3 `TRACK_3_ABI.md` pattern (nrs 1-20 are
assigned; 21+ are fresh). Extend `cipher_ioctl.h`:

| nr | name | dir | role |
|----|------|-----|------|
| 21 | `CIPHER_ARENA_REGISTER` | `_IOWR` | producer: pass the exported fd + metadata blob; kmod `fget`s the fd, creates the registry entry, returns an arena id |
| 22 | `CIPHER_ARENA_IMPORT` | `_IOWR` | consumer: by arena id, kmod `dup`s its held fd into the caller + returns base/size/metadata; adds caller pid to the set |
| 23 | `CIPHER_ARENA_LEAVE` | `_IOW` | explicit participant leave (clean-exit fast path; the reaper is the crash path) |
| 24 | `CIPHER_ARENA_QUERY` | `_IOR` | operator visibility — arena ids, producer/consumer pids, sizes |

Not extending nrs 13/14 (those are the CP54 SM-group allocator — unrelated).
`TRACK_2_ABI.md` will document NRs 21-24 (mirrors `TRACK_3_ABI.md`).

## §e — Reaper extension

Track 3 SC2's `do_exit` reaper (`cipher_cp54_release`) frees SM groups. SC5
adds an **arena-registry reaper hook**: when any registered participant pid
exits, remove it from every arena's participant set; when a set reaches 0,
`fput` that arena's held `struct file *` and free the registry slot. `fput`
from the `do_exit` atomic context is safe — it defers the final close to task
work — but to stay strictly lock-free like the CP54 reaper, SC5-2 may instead
have the reaper flag the slot and a `cipher_partition_tick`/workqueue do the
`fput` (SC5-2 build detail). Either way: kernel fd-table work, no CUDA.

## §f — Fingerprint integration — kmod stores it opaque

The SC4 fingerprint stays a **Python concern** (`cipher_model_fingerprint.py`,
verify-before-import). The kmod arena entry carries the producer's metadata
(JSON layout manifest + fingerprint) as an **opaque blob** — stored and
returned by IMPORT, never interpreted by the kmod. This lets the kmod be the
**single rendezvous** (fd + layout + fingerprint delivered atomically by
IMPORT) — SC5 thus **subsumes the SC3 JSON-sidecar / SCM_RIGHTS transport**:
no socket, no sidecar file. The consumer still runs `verify_fingerprint`
(SC4) on the blob before `weight_arena_import`.

## §g — Failure modes

| failure | handling |
|---|---|
| producer crash during arena creation | REGISTER is atomic — either the entry (with the `fget`'d fd) exists or it does not; a crash before REGISTER returns leaves no entry, no leak |
| consumer crash during import / rebind | reaper removes the pid; physical unaffected while others remain; freed when the set empties |
| kmod unload with active arenas | the kmod `fput`s all held fds on module exit → physical frees once process handles also drop; a live consumer keeps its *existing* mapping, new IMPORTs fail — **acceptable v1, documented** |
| "multiple producers, same arena" | a non-issue — each `cuMemCreate` is a distinct physical; each REGISTER is a distinct arena id; there is no shared-arena collision. Noted, not handled. |

## §2 — SC5-2 / SC5-3 / SC5-4 plan & anchors

| phase | scope | est. |
|---|---|---|
| SC5-1 | this design memo | done |
| SC5-2 | **probe first** (§1 — VMM cross-process refcount behavior); then build the kmod arena registry (`cipher_weight_arena.c`, ioctls 21-24, reaper hook); switch the producer/consumer harness from `SCM_RIGHTS` to the kmod rendezvous | ~1.5–2 d |
| SC5-3 | verify: producer-crash survival (consumer keeps working after producer killed), last-consumer-exit frees, multi-consumer (N=3), reaper correctness; regression smoke | ~0.5 d |
| SC5-4 | `TRACK_2_SC5_CLOSEOUT.md` + `TRACK_2_ABI.md`; memory | ~0.5 d |

**Anchors — recommend KMOD-ONLY rotation.** The fd custodian is pure kmod
fd-table work; the producer/consumer issue the new ioctls **directly from
Python** (an ioctl that passes an fd, one that returns a dup'd fd — exactly
SC3's fd-passing, now via the kmod instead of a socket). `cipher_kv_bridge`'s
`weight_arena_import(fd,…)` is **unchanged** — the consumer simply gets the fd
from the kmod. `libcipher_rt` is **unchanged**. So SC5 rotates **kmod
`285d102e` → SC5 only**; libcipher_rt `83afd1ca`, cipher_kv_bridge `c04b0c39`,
libcipher_v2 `86618c30` stay. (The user's authorization allowed for all three
rotating — the custodian design avoids the other two.) Preserve
`cipher_kmod.ko.pre_sc5` before any kmod edit. Pre-SC5 baseline: the full
regression set on the current substrate.

## §3 — Adjudication ask

**STOPPING — no source modified, no build, no GPU.** Decisions:

1. **The §1 crux** — accept the **fd-custodian** model (kmod holds the fd,
   never calls CUDA; VMM already refcounts the physical) and the **probe-first**
   SC5-2 step that empirically confirms the VMM behavior before kmod code.
2. **Registry + lifetime (§a/§b)** — accept the kmod arena registry with a
   participant set; "alive while any participant registered"; reaper-driven
   `fput` at count 0.
3. **ABI (§d)** — accept 4 new ioctls NRs 21-24 (`ARENA_REGISTER/IMPORT/LEAVE/
   QUERY`), additive; `TRACK_2_ABI.md`.
4. **Fingerprint (§f)** — accept the kmod storing the SC4 fingerprint + layout
   as an opaque blob; the kmod becomes the single rendezvous, subsuming the
   SC3 SCM_RIGHTS/sidecar transport; verify stays Python-side.
5. **Anchors (§2)** — accept **kmod-only** rotation (libcipher_rt,
   cipher_kv_bridge, libcipher_v2 unchanged).

On adjudication: proceed to **SC5-2** — the VMM-refcount probe first, then the
kmod arena registry.
