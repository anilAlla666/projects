# Track 3 (Dynamic SM Migration) — ANCHOR BASELINE

**Date:** 2026-05-19. Recorded at Track 3 authorization, **before any source
modification**. This is the rollback baseline for the whole Track 3 workstream.

---

## Pre-Track-3 anchors (verified live, 2026-05-19)

| artifact | anchor (md5 prefix) | live path | full md5 |
|---|---|---|---|
| kmod `cipher_kmod.ko` | **`8d777dfb`** | `cipher_kmod/cipher_kmod.ko` | `8d777dfb4c60d128b77c49a8082b9ee2` |
| libcipher_rt | **`ebc0baaa`** | `cipher_rt_phase4/libcipher_rt.so` | `ebc0baaa4e771abbdd20f6c4af645040` |
| libcipher_v2 | **`86618c30`** | `libcipher_v2/libcipher_v2.so.v0.2.0` | `86618c30896470b642fcc6985d8dc632` |
| cipher_kv_bridge | **`fca6843d`** | `cipher_rt_phase4/cipher_kv_bridge.so` | `fca6843dcfd82a650e223d513d3b3ca9` |

## Track 3 fallbacks preserved (2026-05-19, verified md5)

| fallback file | md5 | = anchor |
|---|---|---|
| `cipher_kmod_fallback/cipher_kmod.ko.pre_track3` | `8d777dfb4c60d128b77c49a8082b9ee2` | `8d777dfb` ✓ |
| `cipher_rt_fallback/libcipher_rt.so.pre_track3` | `ebc0baaa4e771abbdd20f6c4af645040` | `ebc0baaa` ✓ |

Both `.pre_track3` copies are **outside the kbuild dir** ([[cipher-kbuild-clean-wipes-ko]]
— `kbuild clean` globs `*.ko`). `libcipher_v2` and `cipher_kv_bridge` are not
expected to change in Track 3 (surface + preserve if SC4's opt-in-flag plumbing
needs libcipher_v2 — see SC1 design memo §(f)/§(h)).

## libcipher_v2 anchor note (not a discrepancy — documented)

`libcipher_v2/libcipher_v2.so` (the plain name) is md5 `cc0479b8…` — the
**un-promoted v0.3.0 CUPTI superset**, NOT a campaign anchor, not in the soak
runtime path (`cipher_anchors_manifest.txt`, `D6_LIBCIPHER_V2_ANCHOR_RESOLUTION.md`).
The campaign anchor `86618c30` is `libcipher_v2.so.v0.2.0`. Recorded so a Track 3
smoke test does not mis-flag the plain `.so` as a regressed anchor.

## Per-milestone discipline

Every Track 3 SC closes with: anchor preservation verified (md5 recorded) →
smoke test W1/W2/W3 within ±3% mean-vs-mean → rollback path documented → SC
test → adjudication checkpoint. Any smoke-test regression → STOP, rollback,
surface. Anchor rotations are appended to this file as each SC lands.

## Rotation log

### SC2 — kmod migration state machine (2026-05-19)

kmod anchor rotated **`8d777dfb` → `98da2d1f`**
(`98da2d1fd906a1b58c978a6996021539`). SC2 built the Track 3 migration state
machine (ioctl nrs 16-20, RSVD group state, two-cmpxchg reaper,
`cp54_eval_migration`). Verified: 15/15 isolation tests PASS (no NR 1-15
change), 29/29 SC2 unit-test assertions PASS, regression smoke W1/W2/W3 PASS
within ±3 % of `TRACK_3_SC2_PRE_BASELINE.md`, dmesg clean.

Loaded artifact: `cipher_kmod/cipher_kmod.ko` (`98da2d1f`).
Fallbacks (all md5-verified, outside the kbuild dir per
[[cipher-kbuild-clean-wipes-ko]]):
- `cipher_kmod_fallback/cipher_kmod.ko.track3_sc2` (`98da2d1f` — SC2)
- `cipher_kmod_fallback/cipher_kmod.ko.pre_sc2` (`8d777dfb` — SC2 rollback)
- `cipher_kmod_fallback/cipher_kmod.ko.pre_track3` (`8d777dfb` — Track 3 rollback)
Source: `cipher_kmod_src_track3_sc2.tar.gz` (`414e6e82`).

libcipher_rt `ebc0baaa`, libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d`
— **unchanged** (SC2 is kmod-only). See `TRACK_3_SC2_CLOSEOUT.md`.

### SC3 — tenant-side migration primitive (2026-05-19)

libcipher_rt anchor rotated **`ebc0baaa` → `83afd1ca`**
(`83afd1ca4118dc651854ef751e4ef82d`). SC3 built `cipher_rt_green_ctx_migrate()`
(drain → build → L1 verify → swap → release), `cipher_rt_green_ctx_cur_mask()`,
the `CIPHER_MIGRATABLE` opt-in plumbing, and `cipher_migrate.py` (the
poll-and-migrate handler). Verified: reachability gate PASS, SC3 unit+e2e 7/7
PASS (both e2e tenants migrated end-to-end), regression smoke W1/W2/W3 PASS
within ±3 % of `TRACK_3_SC3_PRE_BASELINE.md`, dmesg clean.

Loaded artifact: `cipher_rt_phase4/libcipher_rt.so` (`83afd1ca`).
Fallbacks (md5-verified): `cipher_rt_fallback/libcipher_rt.so.track3_sc3`
(`83afd1ca` — SC3), `.pre_sc3` (`ebc0baaa` — SC3 rollback), `.pre_track3`
(`ebc0baaa` — Track 3 rollback). Source: `cipher_rt_green_ctx_track3_sc3.tar.gz`
(`bcfcbe38`).

kmod `98da2d1f`, libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d`
— **unchanged** (SC3 is libcipher_rt-only). See `TRACK_3_SC3_CLOSEOUT.md`.

### SC5-2 — SC4 fold + constrained POOL grant (2026-05-19)

kmod anchor rotated **`98da2d1f` → `285d102e`**
(`285d102e86eafb70e95040fd1c2d44d0`). SC5-2 built the folded SC4 policy
(4 module params at `/sys/module/cipher_kmod/parameters/` mode 0644;
`/proc/cipher/migrations`; per-tenant `migration_count` + aggregate counters)
and **`cp54_pool_claim_low_prefix`** — the SC1 fact-2 constrained-low-prefix
POOL grant that was specified but never built (SC5 design memo §1). kmod-only,
no ABI change. Verified: regression smoke PASS — isolation 15/15, SC2-unit
29/29, W1 +0.83 %, W2 clause1=clause2=0, W3 `SC2_PASS`, SC3 7/7, dmesg clean
(`TRACK_3_SC5_REGRESSION.md`).

Loaded artifact: `cipher_kmod/cipher_kmod.ko` (`285d102e`).
Fallbacks (md5-verified): `cipher_kmod_fallback/cipher_kmod.ko.track3_sc5`
(`285d102e`), `.pre_sc5` (`98da2d1f` — SC5 rollback). Source:
`cipher_kmod_src_track3_sc5.tar.gz` (`e57a1a84`).

libcipher_rt `83afd1ca`, libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d`
— **unchanged** (SC5 is kmod-only). The 5-partition churn harness
(`sc5_tenant.py` / `sc5_churn.py`) is built; the measurement run is SC5-3.
See `TRACK_3_SC5_BUILD_LOG.md`. A-vs-B is a config (`gap_min_grps` 1 vs 16) —
no separate anchor.
