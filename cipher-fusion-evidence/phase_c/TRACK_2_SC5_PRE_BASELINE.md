# Track 2 SC5-2 — PRE-BASELINE

**Date:** 2026-05-19. Captured before any SC5 source modification, per the
phase discipline ([[cipher-phase-discipline]]) — the reference the SC5
regression smoke is measured against.

## Anchors — pre-SC5 state

| artifact | md5 | preserved as |
|---|---|---|
| `cipher_kmod.ko` | `285d102e` | `cipher_kmod_fallback/cipher_kmod.ko.pre_track2_sc5` |
| `libcipher_rt.so` | `83afd1ca` | (unchanged this phase) |
| `cipher_kv_bridge.so` | `c04b0c39` | (unchanged this phase) |
| `libcipher_v2.so` | `cc0479b8` | (unchanged this phase; = the `v0.3.0-cupti.unpromoted` build) |

The kmod `.ko` was preserved **outside** the kbuild tree
([[cipher-kbuild-clean-wipes-ko]]) before the first SC5 edit.

> Note: earlier Track 2 docs (SC2–SC4) cite the libcipher_v2 anchor as
> `86618c30`. That md5 is the archived `libcipher_v2.so.v0.2.0`; the *active*
> `libcipher_v2.so` has been `cc0479b8` (= `libcipher_v2.so.v0.3.0-cupti.
> unpromoted`) since 2026-05-17 — i.e. the label in the older docs was
> already stale before SC5. SC5 modifies neither file. Recorded here so the
> SC5 regression compares against the true pre-SC5 state.

## Functional baseline (pre-SC5 substrate)

| gate | tool | pre-SC5 result |
|---|---|---|
| isolation | `cp54_isolation_test` | 15 / 15 PASS |
| SC2 unit | `sc2_verify.py` unit set | 29 / 29 PASS |
| W1 decode | TinyLlama B=1 | 491.7 tok/s |
| W3 | `sc2_verify.py` | `SC2_PASS=True`, KL 0.0 |
| dmesg | — | clean |

## SC5-2 scope (what the rotation will cover)

SC5 is **kmod-only** (design memo §2): a new source unit
`cipher_weight_arena.c`, 4 additive ioctls (NRs 21-24), a `/proc/cipher/
arenas` node. `libcipher_rt`, `cipher_kv_bridge`, `libcipher_v2` are **not**
touched — the producer/consumer issue the new ioctls directly from Python
(`sc5_arena_ioctl.py`). Expected post-SC5 anchor state: kmod rotates,
the other three byte-identical.
