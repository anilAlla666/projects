# Track 2 SC5 — CLOSEOUT

**Date:** 2026-05-19. **Status: SC5-2 COMPLETE — STOP for adjudication
before SC6.** Predecessor: SC4 closed (`TRACK_2_SC4_CLOSEOUT.md`).

SC5 makes a shared weight arena's lifetime **independent of the producer
process** — the production safety primitive. SC3/SC4 assumed "producer stays
alive" and used an interim `model._cipher_weight_arena` user-space pin; SC5
supersedes both.

## What SC5-2 delivered

| phase | deliverable | result |
|---|---|---|
| A — probe | VMM cross-process refcount probe (`sc5_probe*.py`) | **PASS** — `TRACK_2_SC5_PROBE_LOG.md`; the crux holds on this H100/CUDA 13 |
| B — build | `cipher_weight_arena.c` + ioctls 21-24 + `/proc/cipher/arenas` | **clean build, clean load** — `TRACK_2_SC5_BUILD_LOG.md` |
| C — harness | `sc5_arena_ioctl.py` + producer/consumer/run | producer-dies-first integration |
| D — verify | unit 15/15, integration 12/12, regression smoke | all PASS |

## What is proven

The kmod is an **fd custodian**: at `ARENA_REGISTER` it `fget`s the
producer's exported VMM POSIX fd; that kernel `struct file *` reference *is*
a CUDA handle, so it keeps the physical memory alive — the kmod never calls
the CUDA driver (design memo §1). The producer-dies-first integration
(`sc5_run.py`, 12/12) demonstrates end-to-end:

1. **A producer can crash** (`SIGKILL`, no cleanup) and the shared arena
   survives — `producer_pid` clears, the arena stays.
2. **A consumer keeps working** after the producer's death — consumer-A's
   forward pass runs *after* the `SIGKILL`, bit-identical to the producer's.
3. **A fresh peer can still join** with no producer process in existence —
   consumer-B IMPORTs and runs after the producer is long gone. SC3's
   `SCM_RIGHTS` socket transport structurally could not do this; the kmod
   rendezvous can.
4. **The physical frees on the last participant** — once both consumers
   exit, the reaper `fput`s the held fd and frees the slot.
5. **Sharing is real** — each consumer adds only its CUDA context
   (~0.5–0.7 GiB), not another ~2 GiB weight copy.

kmod unit tests (15/15) verify the ioctl boundary directly: REGISTER/IMPORT/
LEAVE/QUERY, opaque-blob round-trip, multi-consumer counting, and clean
`-ENOENT`/`-ENOSPC` error paths.

## v1 boundary — stated honestly

- **Zero-participant survival is not guaranteed.** A producer that REGISTERs
  and then dies with **no consumer ever having joined** is reaped on the
  next 5 s cycle (participant count hits 0). `IMPORT`-after-producer-death
  works only if ≥1 consumer joined while the producer was alive — which the
  integration enforces by ordering. This is consistent with the adjudicated
  design memo §b ("alive while any participant is registered"). A
  registration grace window (a `never_imported` sticky bit, or a
  `registered_at` + module-param grace) is the natural v2 hardening if a
  "register, then later attach" deployment pattern is wanted. **Not a bug —
  a documented v1 scope line.**
- **Reaper latency** — arena reclamation is workqueue-driven (5 s), not
  `do_exit`-immediate. Deliberate (design memo §e): reclamation is not
  latency-critical and this keeps SC5 off the delicate `do_exit` path. The
  physical is held safely by the fd ref in the meantime.
- **kmod unload with active arenas** — `cipher_wa_exit` `fput`s all held
  fds; a live consumer keeps its existing mapping, new IMPORTs fail.
  Acceptable v1 (design memo §g).

## Anchors

| artifact | before | after |
|---|---|---|
| `cipher_kmod.ko` | `285d102e` | **`008b3c66`** ← new campaign anchor |
| `libcipher_rt.so` | `83afd1ca` | `83afd1ca` (untouched) |
| `cipher_kv_bridge.so` | `c04b0c39` | `c04b0c39` (untouched) |
| `libcipher_v2.so` | `cc0479b8` | `cc0479b8` (untouched) |

Kmod-only rotation, as the design memo §2 recommended. Fallbacks:
`cipher_kmod.ko.pre_track2_sc5` (`285d102e`), `cipher_kmod.ko.track2_sc5`
(`008b3c66`), source `cipher_kmod_src_track2_sc5.tar.gz`. ABI documented in
`TRACK_2_ABI.md` (NRs 21-24, additive, [[cipher-abi-rule]] honored).

## Regression

Smoke PASS — `TRACK_2_SC5_REGRESSION.md`. isolation 15/15, W3 `SC2_PASS`
(KL 0.0), dmesg clean. SC5's kmod changes are wholly additive; no existing
path modified.

## Deliverables

`TRACK_2_SC5_PRE_BASELINE.md`, `TRACK_2_SC5_PROBE_LOG.md`,
`TRACK_2_SC5_BUILD_LOG.md`, `TRACK_2_ABI.md`, `TRACK_2_SC5_UNIT_TESTS.md`,
`TRACK_2_SC5_INTEGRATION.md`, `TRACK_2_SC5_REGRESSION.md`, this closeout.
Code: `sc5_arena_ioctl.py`, `sc5_producer.py`, `sc5_consumer.py`,
`sc5_run.py`, `sc5_unit.py`; kmod `cipher_weight_arena.c` + edits.

## Adjudication ask — STOP before SC6

SC5-2 is complete: the probe confirmed the crux, the kmod fd custodian is
built and loaded, and the producer-dies-first integration proves the
producer-independent lifetime end-to-end (12/12), with unit tests (15/15)
and a clean regression smoke. The interim SC4 `model._cipher_weight_arena`
pin is now backed by a real kmod-owned lifetime.

**Decisions requested:**

1. Accept SC5-2 — fd-custodian kmod registry, producer-dies-first verified,
   kmod anchor `008b3c66`.
2. Accept the **v1 boundary** above (zero-participant survival not
   guaranteed; 5 s reaper latency) — or direct the v2 grace-window
   hardening now.
3. Authorize **SC6** — Track 2's remaining scope (the N-tenant
   weight-sharing measurement / CP 5.5 prerequisite wiring), or redirect.

**STOPPING for adjudication.** No further source modified pending decision.
