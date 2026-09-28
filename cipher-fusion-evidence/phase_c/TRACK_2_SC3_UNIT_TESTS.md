# Track 2 SC3-2 — UNIT TESTS

**Date:** 2026-05-19. cipher_kv_bridge `c04b0c39`. The four SC3-2 unit checks,
exercised within the two integration runs (`sc3_run.py` — same-VA and
offset-relative); full detail in `TRACK_2_SC3_INTEGRATION.md`.

| unit check | result | how |
|---|---|---|
| **import primitive** — `cipher_rt_weight_arena_import` / `weight_arena_import` maps a peer's exported arena | **PASS** | consumer imported the producer's `SCM_RIGHTS`-passed fd and mapped a usable arena (both runs) |
| **same-VA path** — `cuMemAddressReserve(want_base)` maps at the producer's base | **PASS** | same-VA run: `same_va=true`, consumer base == producer base |
| **offset-relative path** — `want_base=0` → reserve-without-hint branch | **PASS (path exercised)** | offset-relative run: `SC3_FORCE_OFFSET=1` → the no-hint reserve branch ran, import bit-identical. The VA coincided with the producer's (deterministic fresh-process VA allocation — a different base could not be forced); immaterial — `WeightArena.view` is base-relative (`arena.base+offset`), correct for any base by construction |
| **manifest round-trip** — JSON layout manifest survives `SCM_RIGHTS` transport and drives a correct rebind | **PASS** | 201-tensor manifest sent + parsed + every offset/shape/dtype used; the bit-identical forward proves each entry was correct |

**Bit-identical forward, both runs** — `torch.equal` true, max-abs-diff 0.0:
the consumer's parameters are the producer's exact physical bytes.

**Verdict: 4/4 unit checks PASS.** No substrate change to kmod `285d102e` /
libcipher_rt `83afd1ca` — SC3 touched `cipher_kv_bridge` only.
