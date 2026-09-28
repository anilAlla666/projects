# Track 3 (Dynamic SM Migration) — SC3 CLOSEOUT

**Date:** 2026-05-19. **SC3 scope:** the tenant-side migration primitive in
libcipher_rt + the poll-and-migrate handler. **Verdict: SC3 COMPLETE — all
verification PASS.** libcipher_rt anchor rotated `ebc0baaa` → `83afd1ca`;
kmod `98da2d1f`, libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d`
untouched. Stopping for adjudication before SC4.

SC2 proved the kmod coordination is sub-µs; **SC3 is the load-bearing half** —
and it lands cheap: a migration is a green-context rebuild the tenant runs
between decode rounds, with the kmod state machine (SC2) and the libcipher_rt
primitive (SC3) composing end-to-end.

---

## §1 — What SC3 built

**libcipher_rt `cipher_rt_green_ctx_migrate(new_mask)`** — drain
(`cuCtxSynchronize`) → build the new green context from `new_mask` (the
`ensure()` Steps 1-7, into locals) → **L1 structural verify** → atomic swap of
the four `g_green_*` globals + `cuCtxSetCurrent` → release the old context.
Failure contract: any pre-swap failure returns <0 with the old context intact
and current (B-floor); a post-swap destroy failure is logged-and-leaked
(a committed migration is never unwound).

**`cipher_migrate.py`** — the two-layer poll-and-migrate handler the tenant
runs inline at each decode-round boundary: POLL/START/ACK ioctls + the L2
`%smid` post-swap verify; binds the C primitive via `RTLD_NOLOAD`.

**`CIPHER_MIGRATABLE`** env → `SUBSCRIBE_MIGRATE(1)` in
`cipher_rt_green_ctx_cp54_init()` — opt-in, no libcipher_v2 change (SC1 item 8).

## §2 — Verification (all PASS)

| gate | result |
|---|---|
| reachability gate (`CDLL` to the injected instance) | **PASS** — `RTLD_NOLOAD` reaches it; `CDLL(None)` does not |
| build | clean — no warning/error from `cipher_rt_green_ctx.c` |
| SC3 unit tests | **5/5 PASS** — self_migrate, verify_fault (L1 fail → NACK), destroy_fault (commit-and-leak), optin yes/no |
| SC3 e2e integration | **PASS** — 2 migratable decode tenants both migrated end-to-end (COMPACT → PROPOSE → START → migrate → ACK COMMIT), TFGATE KL ≤ 5.5e-5 |
| regression smoke W1/W2/W3 | **PASS** — W1 +0.11 % (gate ±3 %); W2 clause1=clause2=0; W3 `SC2_PASS` |
| dmesg | clean throughout |

The no-regression guarantee held structurally: all-pinned workloads never set
`CIPHER_MIGRATABLE`, so `cipher_rt_green_ctx_ensure()` runs unchanged and the
migrate path is never entered.

## §3 — Two-layer verification (Item-2 PUSH) — built and exercised

- **L1 structural** (libcipher_rt, pre-swap): `cuGreenCtxGetDevResource` count
  == the old count + descriptor built from exactly `new_mask`'s groups. Gates
  the swap — `verify_fault` confirms a failed L1 → NACK, old context intact.
- **L2 `%smid` probe** (Python handler, post-swap): observed SM set ⊆
  `mask_to_sms(target)`. Every committed e2e migration passed L2.

Both must pass; either fails → `ACK_MIGRATE(0)` NACK. libcipher_rt ships no
embedded CUDA kernel.

## §4 — Deviations / findings (flagged)

1. **Reachability is `RTLD_NOLOAD`, not `CDLL(None)`.** The design memo §3
   listed `CDLL(None)` first; empirically the CUDA driver loads the injection
   library RTLD_LOCAL, so its symbols are not in the global namespace.
   `ctypes.CDLL(path, RTLD_NOLOAD)` reaches the injected instance (verified —
   `is_initialized=1`, matching symbol address). No fallback was needed.
2. **`cipher_rt_green_ctx_migrate` signature is `unsigned`**, not `uint32_t`
   (the file's idiom; 32-bit; Python passes `c_uint`). Equivalent.
3. **`CIPHER_SC3_FAULT` test hook** added to `cipher_rt_green_ctx_migrate()`
   (`verify` / `destroy` fault injection). Env-gated — zero effect unset.
   Retained for SC5 fault injection.
4. **`cur_mask()` returns 0 before the green context is lazily built** — seen
   in the e2e JSON `start_mask` (read before the model's streams triggered
   `ensure()`). Correct behaviour of the accessor, not a fault; the migration
   path itself read the mask correctly (count-preserving check passed, both
   tenants committed).

## §5 — Anchors

| artifact | before | after |
|---|---|---|
| libcipher_rt | `ebc0baaa` | **`83afd1ca`** (loaded; `.track3_sc3` + `.pre_sc3` fallbacks; src `bcfcbe38`) |
| kmod | `98da2d1f` | `98da2d1f` — unchanged |
| libcipher_v2 | `86618c30` | `86618c30` — unchanged |
| cipher_kv_bridge | `fca6843d` | `fca6843d` — unchanged |

Rotation logged in `TRACK_3_ANCHORS.md`; `cipher_anchors_manifest.txt` updated.

## §6 — SC3 deliverables

`TRACK_3_SC3_DESIGN_MEMO.md`, `TRACK_3_SC3_PRE_BASELINE.md`,
`cp54_reachability_test.py` + `sc3_reachability_result.json`,
`TRACK_3_SC3_BUILD_LOG.md`, source (`cipher_rt_green_ctx.{c,h}`,
`cipher_migrate.py`, `sc3_tenant.py`, `sc3_run.py`),
`TRACK_3_SC3_UNIT_TESTS.md` + result JSON, `TRACK_3_SC3_REGRESSION.md`,
this closeout.

## §7 — Adjudication ask

**STOPPING — SC3 complete, smoke PASS, anchor rotated.** Confirm:

1. **SC3 = PASS** on its verification chain (§2).
2. **The §4 findings** — `RTLD_NOLOAD` (not `CDLL(None)`); `CIPHER_SC3_FAULT`
   test hook retained; the cosmetic `cur_mask()`-before-build behaviour.
3. The migration primitive + handler are ready to carry SC4's policy layer.

On adjudication, proceed to **SC4** — the migration policy framework: gap-
detection tuning, rate-limit shaping, the operator interface, production-
realistic defaults (SC2 shipped the conservative-default mechanism; SC4
reshapes it). SC4 is kmod-side policy on top of the SC2 mechanism + the SC3
tenant primitive — both now in place and verified.
