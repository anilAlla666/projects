# Track 2 SC5-2 — REGRESSION SMOKE TEST

**Date:** 2026-05-19. **Verdict: SMOKE TEST PASS — no regression.**

SC5 rotates the kmod (`285d102e → 008b3c66`). Unlike SC3/SC4 (pure Python),
SC5 has a real regression surface — but every SC5 kmod change is **additive**
(a new source unit, 4 new ioctl `case`s, a new `/proc` node, new init/exit
calls). No existing kmod code path was modified. The smoke confirms it.

## Anchor verification

| artifact | pre-SC5 | post-SC5 | verdict |
|---|---|---|---|
| `cipher_kmod.ko` | `285d102e` | **`008b3c66`** | rotated — **expected** (additive SC5 build) |
| `libcipher_rt.so` | `83afd1ca` | `83afd1ca` | byte-identical |
| `cipher_kv_bridge.so` | `c04b0c39` | `c04b0c39` | byte-identical |
| `libcipher_v2.so` | `cc0479b8` | `cc0479b8` | byte-identical |

The three userspace libraries are untouched — SC5's producer/consumer issue
the new ioctls directly from Python (`sc5_arena_ioctl.py`). The pre-SC5
`.ko` is preserved at `cipher_kmod_fallback/cipher_kmod.ko.pre_track2_sc5`;
the post-SC5 `.ko` at `cipher_kmod.ko.track2_sc5`; source tarballed at
`cipher_kmod_src_track2_sc5.tar.gz`.

> The `libcipher_v2.so` anchor is recorded as `cc0479b8` (= the
> `v0.3.0-cupti.unpromoted` build, active since 2026-05-17). Earlier Track 2
> docs label it `86618c30`; that md5 is the archived `libcipher_v2.so.v0.2.0`
> — the older label was already stale before SC5 began. SC5 modifies neither.
> Flagged here rather than papered over; it is **not** an SC5 regression (the
> active file is unchanged across the phase).

## Functional gates (post-SC5 substrate)

| gate | tool | pre-SC5 | post-SC5 | verdict |
|---|---|---|---|---|
| isolation | `cp54_isolation_test` | 15/15 | **15/15 PASS** | PASS |
| W3 | `sc2_verify.py` | `SC2_PASS`, KL 0.0 | **`SC2_PASS=True`, KL 0.0** | PASS |
| dmesg | — | clean | **clean** | PASS |

- **isolation 15/15** — the CP 5.4 SM-group allocator (ALLOCATE / FREE /
  QUERY, the do_exit reaper, disjointness, concurrent stress). SC5 adds code
  *beside* it; this proves the additive change disturbed nothing — the kmod's
  existing hot paths are intact.
- **W3** — `sc2_verify.py`: the `cipher_kv_bridge` VMM workload (the SC2
  weight-arena substrate that Track 2 builds on). All five gates pass,
  logits max-abs-diff `0.0`, KL `0.0`.
- **dmesg** — clean across the build, the kmod reload, the unit tests, and
  the producer-dies-first integration; the only weight-arena lines are the
  expected `REGISTER` / `IMPORT` / `reaped` info messages.

## Verdict — SMOKE TEST PASS

SC5 rotated only the kmod, and only additively. The CP 5.4 allocator
(isolation 15/15) and the Track 2 SC2 substrate (W3 `SC2_PASS`) are
unaffected; the three userspace anchors are byte-identical. No rollback is
needed beyond the preserved `cipher_kmod.ko.pre_track2_sc5` fallback.

**New campaign kmod anchor: `008b3c66`.**
