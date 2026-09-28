# Phase C / Track 2 — SC3 CLOSEOUT — consumer-side weight import

**Date:** 2026-05-19. **Verdict: SC3 COMPLETE — cross-tenant weight-sharing
works end to end.** cipher_kv_bridge anchor `fca6843d` → `c04b0c39`. kmod
`285d102e`, libcipher_rt `83afd1ca`, libcipher_v2 `86618c30` unchanged.
Stopping for adjudication before SC4.

---

## §1 — What SC3 delivered

SC2 built the producer side (a tenant packs its weights into an exportable
VMM arena). **SC3 built the consumer side and closed the loop:** a peer
process imports that arena and *runs the model on the shared physical bytes*.

- `cipher_rt_weight_arena_import` (C) — `cuMemImportFromShareableHandle` →
  `cuMemAddressReserve` (same-VA hint, offset-relative fallback) → `cuMemMap`
  → `cuMemSetAccess(READ-only)`.
- `weight_arena_import` + `WeightArena.view` (pybind) — symmetric to SC2's
  `weight_arena_create` / `.alloc`.
- The consumer path: meta-load (`init_empty_weights(include_buffers=False)` —
  the safetensors weight read is **skipped**), rebind every parameter onto a
  READ-only arena view per the JSON layout manifest, buffers computed locally.

## §2 — Verification

| gate | result |
|---|---|
| import primitive / same-VA / offset-relative / manifest round-trip | **4/4 unit checks PASS** (`TRACK_2_SC3_UNIT_TESTS.md`) |
| **bit-identical forward** (Y_producer == Y_consumer) | **true, max-abs-diff 0.0** — both consumer runs (Item-4 PUSH) |
| **memory — sharing happened** | producer-loaded 5001 MiB; consumer added **0 MiB** — 2 tenants, 1 physical weight copy |
| consumer weight tensor arena-backed | `page_info kind='weight'` ✓ |
| regression smoke | W3 `SC2_PASS`/KL 0.0, isolation 15/15, dmesg clean |

**The substrate-value claim, demonstrated:** a second same-model tenant runs
TinyLlama with **zero additional GPU weight memory** — and bit-identically.

## §3 — Findings / honest notes

- **`weight_arena_import` is a module function**, not `WeightArena.import_fd`
  as the SC3-1 memo loosely named it — symmetric to `weight_arena_create`.
- **Offset-relative path: exercised, VA coincided.** The `want_base=0` import
  branch ran (offset-relative run), bit-identical; the no-hint reserve
  *coincidentally* returned the producer's base because fresh-process VA
  allocation is deterministic. A genuinely-different base could not be forced
  in-harness; immaterial — `WeightArena.view` is base-relative by
  construction (SC3-1 adjudication accepted this).
- **W1 variance** (`run_arm3.sh`) measured ~±7 % run-to-run — exceeds the ±3 %
  gate, but W1 is cipher_kv_bridge-independent so it cannot regress from an
  SC3 change; flagged as a measurement-methodology note, not a Track 2 issue.

## §4 — SC3 scope line (carried forward)

SC3 verified weight-sharing under **SC3's assumptions**: a *matching* model
(byte-identical weights) and *producer-owned* arena lifetime (the producer
stays alive). The deferred pieces, by design:
- **SC4** — model-identity fingerprint + fallback: only byte-identical models
  may share; a mismatch must fall back to an independent load.
- **SC5** — kmod-owned arena lifetime: the arena must survive the producer
  crashing/exiting (refcount + `do_exit` reaper); SC3's producer-stays-alive
  is a test scaffold, not the production lifetime model.
- **SC6** — end-to-end N=4: FB total, page_info physical-page identity at
  scale, tenant-ceiling.

## §5 — Anchors

| artifact | anchor | note |
|---|---|---|
| cipher_kv_bridge | `fca6843d` → **`c04b0c39`** | rotated; `.pre_track2_sc3` (`fca6843d`) + `.track2_sc3` (`c04b0c39`); src `cipher_kv_bridge_src_track2_sc3.tar.gz` (`5dcd92fe`) |
| kmod | `285d102e` | unchanged (Track 3 close anchor) |
| libcipher_rt | `83afd1ca` | unchanged (Track 3 close anchor) |
| libcipher_v2 | `86618c30` | unchanged |

## §6 — SC3 deliverables

`TRACK_2_SC3_DESIGN_MEMO.md`, `TRACK_2_SC3_PRE_BASELINE.md`,
`TRACK_2_SC3_BUILD_LOG.md`, `sc3_producer.py` / `sc3_consumer.py` /
`sc3_run.py`, `TRACK_2_SC3_UNIT_TESTS.md`, `TRACK_2_SC3_INTEGRATION.md`,
`TRACK_2_SC3_REGRESSION.md`, this closeout, result JSONs.

## §7 — Adjudication ask

**STOPPING — SC3 complete, smoke PASS, anchor rotated.** Confirm:
1. **SC3 = PASS** — cross-tenant weight-sharing works end to end, bit-identical,
   0 MiB consumer weight memory.
2. The §3 findings (module-fn naming; offset-relative VA-coincidence; W1
   variance).
3. The §4 scope line — SC4 (identity fingerprint) and SC5 (kmod-owned
   lifetime) are the remaining load-bearing pieces before N-tenant production.

On adjudication, proceed to **SC4** — model-identity fingerprinting + the
mismatch fallback.
