# Track 2 SC6-2 — REGRESSION SMOKE TEST

**Date:** 2026-05-19. **Verdict: SMOKE TEST PASS — no regression.**

SC6 is **pure harness work** (new `sc6_*.py` test scripts; no substrate file
modified). The regression surface is structurally null; the smoke confirms
it, and that the SC6 runs themselves left the substrate healthy.

## Anchor verification — nothing rotated

| artifact | SC5 close | post-SC6 | verdict |
|---|---|---|---|
| `cipher_kmod.ko` | `008b3c66` | **`008b3c66`** | byte-identical |
| `libcipher_rt.so` | `83afd1ca` | **`83afd1ca`** | byte-identical |
| `cipher_kv_bridge.so` | `c04b0c39` | **`c04b0c39`** | byte-identical |
| `libcipher_v2.so` | `cc0479b8` | **`cc0479b8`** | byte-identical |

All four md5s unchanged — the definitive proof SC6 rotated nothing. The
producer/consumer issue the SC5 ioctls directly via `sc5_arena_ioctl.py`;
`cipher_kv_bridge` is used unmodified.

## Functional gates (post-SC6 substrate)

| gate | tool | result | verdict |
|---|---|---|---|
| isolation | `cp54_isolation_test` | **15/15 PASS** | PASS |
| W3 | `sc2_verify.py` | **`SC2_PASS=True`, KL 0.0**, all 5 gates | PASS |
| dmesg | — | **clean** | PASS |

- **isolation 15/15** — the CP 5.4 SM-group allocator (ALLOCATE/FREE/QUERY,
  the do_exit reaper, disjointness, concurrent stress) — intact after the
  full SC6 run.
- **W3** — `sc2_verify.py`, the `cipher_kv_bridge` VMM workload — all five
  gates pass, logits max-abs-diff `0.0`, KL `0.0`.
- **dmesg** — clean across the TinyLlama + Mistral-7B SC6 runs; the only
  weight-arena lines are the expected `REGISTER` / `IMPORT` / `reaped` info
  messages. No warning, no oops, no fault.

## Teardown integrity (from `TRACK_2_SC6_MEMORY.md`)

Every SC6 phase's third framebuffer measurement (`exited`) returned to
`idle` (0 MiB) — no arena, context, or allocator state leaked past process
teardown, on either model, in either the shared or the independent phase.

## Verdict — SMOKE TEST PASS

SC6 added the N=4 verification harness with **zero substrate change** — all
four anchors byte-identical. The kmod allocator (isolation 15/15) and the
Track 2 SC2 substrate (W3 `SC2_PASS`) are unaffected; dmesg clean; teardown
leak-free. No rollback artifact required (no binary changed).
