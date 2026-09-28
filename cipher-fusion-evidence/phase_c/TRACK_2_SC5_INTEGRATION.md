# Track 2 SC5-2 — INTEGRATION (producer-dies-first)

**Date:** 2026-05-19. Tool: `sc5_run.py`. **Verdict: PASS — 12 / 12 checks.**
Result JSON: `sc5_integration_result.json`.

This is the SC5 crux test: a real model (TinyLlama-1.1B fp16) shared across
tenants through the kmod arena registry, with the producer **SIGKILL'd** —
no cleanup, no `ARENA_LEAVE` — while a consumer holds the arena.

## Harness

- `sc5_arena_ioctl.py` — pure-ctypes `/dev/cipher` ioctl helper (REGISTER /
  IMPORT / LEAVE / QUERY). The SC5 anchor decision is kmod-only rotation, so
  the producer/consumer issue the ioctls **directly** — `cipher_kv_bridge`
  and `libcipher_rt` are unmodified.
- `sc5_producer.py` — loads TinyLlama, packs every weight tensor into a VMM
  weight arena, runs a reference forward pass, REGISTERs the arena fd + a
  metadata blob (SC3 layout manifest + SC4 fingerprint), stays alive.
- `sc5_consumer.py` — IMPORTs by arena id, verifies the SC4 fingerprint,
  maps the shared arena, meta-loads TinyLlama (no safetensors read), rebinds
  every parameter onto a read-only arena view, runs the forward pass.
- `sc5_run.py` — orchestrates the ordering below.

## Sequence executed

| step | event | observed |
|---|---|---|
| 1 | producer REGISTERs arena | `arena_id=1`, `size=2216873984` (2.06 GiB), blob 32–40 KB |
| 2 | consumer-A IMPORTs, rebinds, blocks on a sentinel | `n_consumers=1`, `producer_pid=1685071` |
| 3 | **`SIGKILL` the producer** (crash path) | producer process gone |
| 4 | wait one reaper cycle (8 s), QUERY | **arena survived**; `producer_pid → 0`; `n_consumers=1` |
| 5 | drop the sentinel → consumer-A runs its forward | A forward completes — **after** producer death |
| 6 | consumer-B launched (producer long dead) → IMPORT + forward | B IMPORT succeeds with **no producer process in existence** |
| 7 | compare producer / A / B logits | all bit-identical |
| 8 | both consumers exit, wait a reaper cycle, QUERY | **arena reaped** (`query → null`) |

dmesg trail: `REGISTER pid=1685071` → `IMPORT pid=1685221 → fd=37` →
`IMPORT pid=1685305 → fd=37` → `weight-arena id=1 reaped (all participants
exited)`.

## The 12 checks (all PASS)

| check | result | what it proves |
|---|---|---|
| `arena_survived_producer_sigkill` | ✅ | the kmod's `fget` ref + the consumer's import handle keep the physical alive across an uncleaned producer death |
| `producer_pid_cleared_by_reaper` | ✅ | the 5 s liveness reaper detects the dead producer and clears its pid |
| `consumerA_forward_after_producer_death` | ✅ | a consumer's mapping keeps **working** after the producer is gone |
| `consumerB_imported_with_no_producer` | ✅ | the kmod rendezvous hands a *new* peer the fd with no producer alive — SC3's `SCM_RIGHTS` socket transport could not do this |
| `consumerB_forward_done` | ✅ | consumer-B runs to completion on the shared arena |
| `A_bit_identical` / `B_bit_identical` | ✅ ✅ | `torch.equal(producer, A)` and `(producer, B)` — max abs diff `0.0`; the consumers run on the producer's exact physical bytes |
| `A_weight_arena_backed` / `B_weight_arena_backed` | ✅ ✅ | `page_info` of the largest consumer weight tensor reports `kind=weight` — it is arena memory, not a private copy |
| `A_shared_no_weight_copy` / `B_shared_no_weight_copy` | ✅ ✅ | see memory table below |
| `arena_reaped_after_last_participant` | ✅ | participant count 0 → the kmod `fput`s its held fd → the slot is freed (design memo §b lifetime rule) |

## Memory — sharing is real

| GPU framebuffer (`nvidia-smi memory.used`) | MiB |
|---|---|
| idle | 0 |
| + producer (weights + ctx) | 5001 |
| + consumer-A | 5527 |
| consumer-A added | **526** |
| consumer-B added | **749** |

Each consumer adds only its own CUDA context (~0.5–0.7 GiB) — **not** another
~2 GiB model copy. The weight bytes are counted once in the GPU framebuffer
no matter how many tenants attach. Gate: each consumer's add `<` half the
producer footprint (5001/2 = 2500 MiB) — both well under.

## Verdict

**PASS.** SC5 delivers what SC3/SC4 could not: the shared arena's lifetime no
longer depends on the producer process. A producer may crash; a consumer
keeps working; a fresh consumer can still join — all via the kmod fd
custodian, with the model output bit-identical and zero duplicated weight
memory.
