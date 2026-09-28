# Track 3 (Dynamic SM Migration) — SC2 CLOSEOUT

**Date:** 2026-05-19. **SC2 scope:** the kmod-side migration state machine
(ioctl nrs 16-20). **Verdict: SC2 COMPLETE — all verification PASS.**
Kmod anchor rotated `8d777dfb` → `98da2d1f`. Stopping for adjudication
before SC3.

---

## §1 — What SC2 built

The kmod half of Dynamic SM Migration: a cooperative `PROPOSE → START →
MIGRATING → ACK → COMMIT` protocol that lets the ledger compact a `migratable`
PARTITION's 8-SM-group placement under churn. Four files, additive
([[cipher-abi-rule]]):

- **`cipher_ioctl.h`** — 5 ioctls (nrs 16-20), `struct cipher_cp54_migrate_poll`,
  `CIPHER_CP54_MIG_*` / `MIGOUT_*` enums. `TRACK_3_ABI.md` is the manifest.
- **`cipher_cp54_sched.c`** — RSVD group-state encoding (OCC b31 / RSVD b30 /
  pid b0-29); per-tenant migration state; two-cmpxchg reaper sweep;
  `cp54_eval_migration` (conservative gap-detect + rate-limit); the 5 entry
  points; FREE re-evaluates compaction.
- **`cipher_dev.c`** — 5 dispatch cases. **`cipher_internal.h`** — 5 prototypes.

`libcipher_rt`, `libcipher_v2`, `cipher_kv_bridge` — untouched.

## §2 — Verification (all PASS)

| gate | result |
|---|---|
| build | clean — no warning/error from the modified objects |
| existing 6 isolation tests (15-group) | **15/15 PASS** — no NR 1-15 change |
| SC2 state-machine unit tests | **29/29 PASS** (`cp54_migrate_test.c`) |
| — happy path PROPOSE→…→COMMIT | PASS |
| — pinned never proposed; START/ACK ordering rejects | PASS |
| — tenant-NACK → `ABORTED_TENANT_NACK`; rate-limit | PASS |
| — two-cmpxchg reaper, crash in PROPOSED **and** MIGRATING | PASS (15/15 free) |
| — PROPOSED timeout → `ABORTED_TIMEOUT` | PASS |
| regression smoke W1/W2/W3 vs pre-baseline | **PASS** (`TRACK_3_SC2_REGRESSION.md`) |
| — W1 489.342 tok/s (−1.56 %, gate ±3 %) | PASS |
| — W2 disjointness clause1=clause2=0, KL gates | PASS |
| — W3 `SC2_PASS=True`, KL 0.0 | PASS |
| dmesg | clean across build + all tests |

The no-regression guarantee held structurally: with zero `migratable` tenants
`cp54_eval_migration` is a side-effect-free scan, so the smoke workloads
exercise the unchanged ledger path byte-for-byte.

## §3 — 5-scenario latency (`TRACK_3_SC2_LATENCY.json`, item-2/item-6 PUSH)

PROPOSE (inside COMPACT) → COMMIT (inside ACK), **kmod-side coordination only**:

| scenario | result |
|---|---|
| 1+2 — median / p95 / p99 (n=200) | **0.8 / 0.8 / 0.8 µs**; max 1.0 µs |
| 3 — under 7-process lock contention (n=50) | median 12.2 µs, **max 295.7 µs** |
| 4 — slow tenant response (50/200/500 ms) | 50.06 / 200.06 / 500.06 ms — tracks the injected delay + ~64 µs |
| 5 — PROPOSED→ABORT, quiet period | 32.0 s observed (timeout 30 s + slack) |

**The SC6 input this produces.** The kmod coordination cost is **sub-microsecond**
— PROPOSE→COMMIT on the kmod side is negligible. Therefore the *real* migration
latency a tenant experiences is dominated entirely by (a) the tenant's poll
cadence and (b) the SC3 tenant-side green-context rebuild (~1.7 ms prior, Step
1.5) — **not** the kmod. **This means the SC1 item-2 poll-vs-eventfd question
is not a kmod-latency question:** poll is sound, because the kmod side adds
nothing measurable; eventfd would only shave poll-cadence latency. SC6 should
weigh eventfd against poll cadence, not kmod coordination. Scenario 5 confirms
the item-4 trade-off: the lazy timeout is **unbounded until the next
FREE/COMPACT** under a permanently quiet ledger — SC6's explicit-timer call.

## §4 — Deviations from the SC2 plan (flagged)

1. **`ACK_MIGRATE` is `_IOW(__u32)`, not `_IO`.** Necessary consequence of the
   adjudicated item-5 addition: `ABORTED_TENANT_NACK` needs a tenant path to
   report a *failed* migration. NR 19 stays `ACK_MIGRATE`; ABI 16-20 intact.
   The migration invariant is preserved in refined form — the *kmod* never
   unilaterally aborts a MIGRATING migration; the *tenant* may abort its own
   (authoritative: verify-before-swap, SC3). Detail in `TRACK_3_ABI.md`.
2. **SC2/SC4 split honored as adjudicated** — SC2 ships the gap-detect + rate-
   limit *mechanism* with a fixed conservative default (propose only for a
   POOL-unreachable gap > 16 SM sustained > 5 s; COMPACT bypasses the gates).
   SC4 builds the policy framework (tunable thresholds, env-var `migratable`
   plumbing). SC2's `cp54_eval_migration` proposes to the single lowest-group
   blocker per pass — true simultaneous-MIGRATING is precluded by that
   structure and is an SC4/SC5 concern (noted in the latency JSON, scenario 3).
3. **Reaper-triggered re-eval is COMPACT/next-FREE-driven in SC2.** The do_exit
   reaper frees groups (two-cmpxchg) but does not itself trigger a migration
   proposal — that is policy/triggering (SC4). The reaper's correctness job
   (no leaked group in any phase) is done and verified.

## §5 — Anchors

| artifact | before | after |
|---|---|---|
| kmod | `8d777dfb` | **`98da2d1f`** (loaded; `.track3_sc2` + `.pre_sc2` + `.pre_track3` fallbacks; src `414e6e82`) |
| libcipher_rt | `ebc0baaa` | `ebc0baaa` — unchanged |
| libcipher_v2 | `86618c30` | `86618c30` — unchanged |
| cipher_kv_bridge | `fca6843d` | `fca6843d` — unchanged |

Rotation logged in `TRACK_3_ANCHORS.md`.

## §6 — SC2 deliverables

`TRACK_3_SC2_ADJUDICATION.md`, `TRACK_3_SC2_PRE_BASELINE.md`, source patches
(4 files), `TRACK_3_ABI.md`, `TRACK_3_SC2_BUILD_LOG.md`,
`cp54_migrate_test.c`, `TRACK_3_SC2_LATENCY.json`,
`TRACK_3_SC2_REGRESSION.md`, this closeout.

## §7 — Adjudication ask

**STOPPING — SC2 complete, smoke PASS, anchor rotated.** Confirm:

1. **SC2 = PASS** on its verification chain (§2).
2. **The §4 deviations** — `ACK_MIGRATE` as `_IOW(__u32)`; the conservative
   default policy; reaper re-eval deferred to SC4.
3. The **§3 SC6 input** — kmod coordination is sub-µs, so poll-vs-eventfd is a
   poll-cadence question, not a kmod-latency one.

On adjudication, proceed to **SC3** — libcipher_rt
`cipher_rt_green_ctx_migrate(new_mask)`: drain / build / verify-before-swap /
release, the poll-and-migrate handler, the opt-in flag. SC3 rotates the
libcipher_rt anchor `ebc0baaa`.
