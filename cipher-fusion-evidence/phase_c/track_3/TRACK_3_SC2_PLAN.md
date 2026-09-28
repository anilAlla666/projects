# Track 3 (Dynamic SM Migration) — SC2 BUILD PLAN (7 items)

**Date:** 2026-05-19. **Type:** build plan — **paperwork only**. No source
modified, no build, no GPU until this plan is adjudicated.
**Predecessor:** `TRACK_3_SC1_ADJUDICATION.md` (all 8 SC1 items accepted).
**Scope:** SC2 = the **kmod-side** migration state machine, ioctls, and ledger
extension. SC2 rotates the **kmod anchor only** (`8d777dfb` → SC2). libcipher_rt
`ebc0baaa`, libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d` are **untouched
by SC2** — the tenant-side `cipher_rt_green_ctx_migrate()` primitive is SC3.

**Estimate:** ~2–3 days (SC1 memo §4).

---

## Scope reconciliation — read before adjudicating

SC1 memo §4's SC-table put **gap detection + rate limiting** in **SC4**. The
adjudicated SC2 7-item list (user, 2026-05-19) places "per-tenant migration
policy hooks (gap detection, rate limit)" in **SC2** (item 5). This is a
**deliberate, surfaced scope adjustment**, and it is correct: the PROPOSED
state is *untestable* without a trigger that produces a `target_mask`. SC2
therefore builds:

- **the gap-detection *mechanism*** — the function that, given the ledger,
  computes whether a `migratable` partition above a POOL-unreachable free group
  exists, and what `new_mask` would close the gap;
- **the rate-limit *guard*** — the per-tenant `last_migrate_ns` field and the
  `≤ 1/tenant/10 s` check enforced at PROPOSE.

SC4 retains the **policy *framework*** layered on that mechanism: operator
threshold tuning, the `migratable`-flag env-var plumbing in libcipher_rt
(SC1 item 8), and abort-fallback policy. SC2 ships a **minimal fixed default
policy** (propose on any POOL-unreachable gap above a `migratable` partition;
one partition per evaluation pass; hard 10 s rate limit) — enough to drive and
verify the state machine. **Adjudication point: confirm this SC2/SC4 split.**

---

## ABI plan (additive only — [[cipher-abi-rule]])

Current `cipher_ioctl.h` NRs in use: 1–15. **Next free NR = 16.** SC2 adds
**five** ioctls, NRs **16–20**, magic `'C'`. No existing struct layout or NR
changes; reserved NRs 2/3/4 stay `-ENOSYS`.

| NR | name | dir | payload | purpose |
|----|------|-----|---------|---------|
| 16 | `CIPHER_CP54_SUBSCRIBE_MIGRATE` | `_IOW` | `__u32 migratable` | tenant declares `migratable` (1) or `pinned` (0); sets the per-tenant flag in the ledger metadata |
| 17 | `CIPHER_CP54_POLL_MIGRATE` | `_IOR` | `struct cipher_cp54_migrate_poll` | tenant reads its `migrate_state` + `target_mask` |
| 18 | `CIPHER_CP54_START_MIGRATE` | `_IO` | — | `PROPOSED → MIGRATING` (commit of intent; non-abortable after this) |
| 19 | `CIPHER_CP54_ACK_MIGRATE` | `_IO` | — | `MIGRATING → COMMIT`; kmod frees `old_mask`, promotes `RSVD→PACK` |
| 20 | `CIPHER_CP54_COMPACT` | `_IO` | — | operator-forced compaction-evaluation pass (deterministic SC2 test trigger) |

New struct (additive; ends with a `reserved[]` tail per the existing house rule):

```c
struct cipher_cp54_migrate_poll {
    __u32 migrate_state;   /* CIPHER_CP54_MIG_* (IDLE/PROPOSED/MIGRATING) */
    __u32 target_mask;     /* new_mask the kmod proposes; 0 unless PROPOSED+ */
    __u32 cur_mask;        /* caller's current owned mask (cross-check) */
    __u32 reserved[5];
};
#define CIPHER_CP54_MIG_IDLE       0
#define CIPHER_CP54_MIG_PROPOSED   1
#define CIPHER_CP54_MIG_MIGRATING  2
```

`COMMIT`/`ABORT` are kmod-internal terminal transitions back to `IDLE`; they
are not poll-visible states (a tenant that polls `IDLE` after seeing
`MIGRATING` reads `cur_mask` to learn the outcome). Adjudication note: this
keeps the poll enum at three values; an explicit `COMMITTED`/`ABORTED` poll
state is available if the user prefers — flagged, not built.

---

## The 7 build items

### Item 1 — kmod migration state machine
**File:** `cipher_cp54_sched.c`. Per-tenant migration state added to
`struct cipher_cp54_alloc` (kernel-internal struct — not ABI):
`u8 migrate_state`, `u8 migratable`, `u32 target_mask`, `u64 proposed_ns`,
`u64 last_migrate_ns`. Transitions `IDLE→PROPOSED→MIGRATING→(COMMIT|ABORT)→IDLE`
exactly per SC1 §2. **START is the non-abortable point** — once `MIGRATING`,
the kmod never ABORTs. All transitions that mutate the ledger run under
`cipher_cp54_lock` (the existing ALLOCATE-path mutex); `POLL_MIGRATE` is a
lock-free read of `migrate_state`/`target_mask` (the QUERY discipline).
**Timeout = lazy:** `PROPOSED→ABORT` on timeout is evaluated on the next
ledger-mutating event (FREE / COMPACT) and observable by the subject at its
next `POLL_MIGRATE` — no new timer/workqueue infrastructure.

### Item 2 — PROPOSE→COMMIT ioctls
**Files:** `cipher_ioctl.h` (5 NRs + struct above), `cipher_dev.c` (5 new
`case` labels in the dispatch switch, after `CIPHER_CP54_QUERY`),
`cipher_cp54_sched.c` (5 new `cipher_cp54_ioctl_*` entry points),
`cipher_internal.h` (5 new prototypes alongside the existing
`cipher_cp54_ioctl_*` declarations). Existing ioctl semantics unchanged.

### Item 3 — RSVD third group state plumbing
**File:** `cipher_cp54_sched.c`. Group-state encoding extended:

```
  OCC  = bit 31   (0x80000000)  — unchanged
  RSVD = bit 30   (0x40000000)  — NEW
  pid  = bits 0..29 (0x3FFFFFFF mask)  — narrowed from 0..30
```

`CIPHER_CP54_GRP_PACK(pid) = OCC | (pid & 0x3FFFFFFF)`,
`CIPHER_CP54_GRP_RSVD(pid) = RSVD | (pid & 0x3FFFFFFF)`. PID fits 30 bits
safely — Linux `PID_MAX_LIMIT` is 2²² ≪ 2³⁰. This encoding is **kmod-internal**
(the userspace ABI is the 15-bit `grp_mask`), so the narrowing is not an ABI
change. A PROPOSE reserves `new_mask` groups as `RSVD(P)` while `P` still owns
`old_mask` as `PACK(P)`; COMMIT atomically frees `old_mask` and promotes
`RSVD(P)→PACK(P)`. New helper `cp54_reserve(pid, mask)` mirrors `cp54_claim`.

### Item 4 — reaper extension (two-cmpxchg sweep)
**File:** `cipher_cp54_sched.c`, `cp54_release_groups()` / `cp54_release_impl()`.
The `do_exit`-kprobe reaper (`cipher_cp54_release`, atomic context, lock-free)
sweeps each group with **two** cmpxchg attempts — `PACK(pid)→0` **and**
`RSVD(pid)→0` — so a crash in any migration phase leaks no group (SC1 §3d
table). `cp54_release_impl` also clears the dying pid's `migrate_state`.
Still lock-free, still O(15) per release — one extra cmpxchg per group.

### Item 5 — per-tenant migration policy hooks (mechanism)
**File:** `cipher_cp54_sched.c`. New `cp54_eval_migration()` — given the
current ledger, finds the lowest `migratable` PARTITION sitting above a
POOL-unreachable free group and computes the count-preserving `new_mask` that
compacts it downward. Invoked from `CIPHER_CP54_FREE`, from `CIPHER_CP54_COMPACT`,
and (deferred-flag) from the reaper path. **Rate limit:** PROPOSE is suppressed
if `now - last_migrate_ns < 10·NSEC_PER_SEC` for that tenant. One PROPOSE per
evaluation pass (incremental compaction). `popcount(new_mask)==popcount(cur)`
asserted at PROPOSE. (SC4 layers tunable thresholds + env-var `migratable`
plumbing on this mechanism.)

### Item 6 — W1/W2/W3 regression discipline
SC2 closes with the adjudicated per-SC discipline (`TRACK_3_ANCHORS.md` §
"Per-milestone discipline"):
- **anchor preservation** — pre-SC2 md5s recorded; `.pre_track3` fallbacks
  confirmed intact;
- **W1** — Phase B N=8 TinyLlama, mean-vs-mean within **±3 %** of 491.8 tok/s
  (run **all-pinned** — exercises the unchanged path, proving SC1 §3h
  backward-compat);
- **W2** — one PARTITION + POOL concurrent (1.6B-2A pattern): clause1 = 0,
  clause2 = 0, KL gates pass;
- **W3** — Track 2 SC2 VMM weight-arena `page_info` evidence chain re-passes;
- **dmesg clean** — no warning/error from the new ioctls or the reaper change;
- **rollback path documented** — exact `rmmod cipher_kmod && insmod
  cipher_kmod_fallback/cipher_kmod.ko.pre_track3` (`8d777dfb`), then re-pass
  W1/W2/W3 at pre-Track-3 levels.
**Any smoke-test regression → STOP, rollback, surface — no SC3 until resolved.**

### Item 7 — SC2 verification scope (unit test + PROPOSE→COMMIT latency)
- **Unit test** (userspace harness, `/dev/cipher` ioctls — no GPU needed for
  the state-machine test): register 3 PARTITIONs + 1 POOL; FREE the middle
  PARTITION to open a POOL-unreachable gap; the high `migratable` PARTITION
  declares itself via `SUBSCRIBE_MIGRATE`; `COMPACT` forces a pass; assert it
  observes `PROPOSED` with a correct count-preserving `target_mask`; drive
  `START_MIGRATE` → (kmod marks `MIGRATING`) → `ACK_MIGRATE` → assert `COMMIT`
  (old freed, RSVD promoted, ledger compacted). Negative cases: rate-limit
  suppression; `pinned` tenant never proposed to; ACK without START rejected;
  timeout → ABORT, tenant keeps `old_mask`.
- **Reaper test:** crash a tenant in `PROPOSED` and in `MIGRATING`; assert both
  `old_mask` and `RSVD` groups return free (extends isolation-suite Test 4).
- **PROPOSE→COMMIT latency (SC1 item 2 PUSH — hard deliverable):** instrument
  the harness to record wall time PROPOSE→COMMIT across the unit-test runs;
  emit a JSON record for SC6 cost characterization and the v1.5 eventfd
  decision.
- **Smoke test:** W1/W2/W3 + dmesg per item 6.

---

## Files SC2 touches

| file | change |
|---|---|
| `cipher_ioctl.h` | +5 NRs (16–20), +1 struct, +3 state `#define`s — additive |
| `cipher_cp54_sched.c` | state machine, RSVD encoding, reaper sweep, `cp54_eval_migration`, 5 ioctl entry points |
| `cipher_dev.c` | +5 `case` labels in the ioctl dispatch switch |
| `cipher_internal.h` | +5 entry-point prototypes |

**Untouched:** libcipher_rt, libcipher_v2, cipher_kv_bridge, all other kmod
`.c` files, every test harness (`cp54_s16_*`, `run_arm*.sh`).

## Anchor rotation (on SC2 close, not now)

kmod `8d777dfb` → SC2 anchor. Before any build: confirm
`cipher_kmod_fallback/cipher_kmod.ko.pre_track3` (`8d777dfb`) intact; the SC2
`.ko` is stored **outside** the kbuild dir ([[cipher-kbuild-clean-wipes-ko]]).
Source tarball `cipher_kmod_src_track3_sc2.tar.gz`. Rotation logged in
`TRACK_3_ANCHORS.md`.

---

## Adjudication ask

**STOPPING HERE — no source modified, no build, no GPU.** Confirm:

1. The **SC2/SC4 scope split** (gap-detection + rate-limit *mechanism* in SC2;
   *policy framework* + env-var plumbing in SC4).
2. The **5-ioctl ABI** at NRs 16–20 (additive, [[cipher-abi-rule]]).
3. The **RSVD encoding** — OCC bit 31 / RSVD bit 30 / pid bits 0–29.
4. **Lazy timeout** (no new timer/workqueue) for `PROPOSED→ABORT`.
5. The **poll enum** stays 3-valued (IDLE/PROPOSED/MIGRATING; COMMIT/ABORT
   inferred from `cur_mask`) — or request explicit COMMITTED/ABORTED states.
6. SC2 **verification scope** (item 7), including the PROPOSE→COMMIT latency
   JSON as a hard deliverable (SC1 item-2 PUSH).

On adjudication: proceed to the SC2 build, closing with the item-6 regression
discipline and the kmod anchor rotation.
