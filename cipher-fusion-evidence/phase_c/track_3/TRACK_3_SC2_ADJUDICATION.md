# Track 3 (Dynamic SM Migration) — SC2 PLAN ADJUDICATION RECORD

**Date:** 2026-05-19. **Adjudicator:** user (campaign owner).
**Subject:** `TRACK_3_SC2_PLAN.md` — six adjudication asks.
**Verdict:** **ALL SIX ACCEPTED.** Five clean, one (item 5) accepted with an
addition. Four items carry a PUSH (an additional binding requirement).
**SC2 build is UNLOCKED.** Build per the adjudicated plan; close with smoke
test PASS; stop for adjudication before SC3.

---

## Items 1–6 — adjudicated

### Item 1 — SC2/SC4 scope split — **ACCEPTED + PUSH**
SC2 builds the **mechanism** (gap-detection + rate-limit guard); SC4 builds the
**policy** (operator tuning, env-var plumbing, production-realistic defaults).
**PUSH — SC2 ships a CONSERVATIVE default policy:** migration must *almost
never* trigger, so W1/W2/W3 regression smoke tests run as if migration does not
exist. Conservative default: **propose only when a POOL-unreachable gap
> 16 SMs (2 groups) has been sustained > 5 s.** SC4 reshapes for production
realism. Rationale: protect SC2's own regression tests from the mechanism SC2
introduces.

### Item 2 — 5-ioctl ABI at NRs 16–20 — **ACCEPTED + PUSH**
NR 16 `SUBSCRIBE`, 17 `POLL`, 18 `START`, 19 `ACK`, 20 `COMPACT_MIGRATE`.
Additive, no struct relayout, [[cipher-abi-rule]] honored.
**PUSH — write `TRACK_3_ABI.md`** (Track 3 ABI manifest, in
`cipher-fusion-evidence/phase_c/track_3/`). Future maintainers must not
collapse the NR numbering or repurpose NRs 16–20.

### Item 3 — RSVD encoding — **ACCEPTED**
OCC bit 31 / RSVD bit 30 / pid bits 0–29. 30-bit pid safe: `PID_MAX_LIMIT`
2²² = 4.19 M; 30-bit holds 1.07 B (256× headroom). Kmod-internal encoding —
**not ABI** (userspace sees `pid_t` unchanged through the ioctl interface).

### Item 4 — lazy timeout (no new timer/workqueue) — **ACCEPTED + PUSH**
`PROPOSED → ABORT` evaluated on the next FREE/COMPACT. Trade-off accepted: a
stuck migration sits in PROPOSED until other ledger activity triggers the
timeout check. No new concurrency surface.
**PUSH — SC6 must measure "worst-case PROPOSED→ABORT latency under a quiet
period."** If unacceptable for production, v1.5 adds an explicit timer. SC2's
PROPOSE→COMMIT measurement does not catch this — it is a separate worst-case
metric (see item 6).

### Item 5 — poll enum — **ACCEPTED WITH ADDITION**
3-value `migrate_state` enum kept: `IDLE / PROPOSED / MIGRATING` (return to
IDLE on completion regardless of outcome).
**ADDITION — a separate `last_outcome` field** the tenant can query:
`NONE / COMMITTED / ABORTED_TIMEOUT / ABORTED_KMOD_REFUSED / ABORTED_TENANT_NACK`.
Set on transition *out of* MIGRATING (or out of PROPOSED if the migration
aborts before MIGRATING). The tenant inspects `last_outcome` for debugging and
error reporting — this eliminates the bug class where tenant code mis-compares
`cur_mask` and infers the wrong outcome. `struct cipher_cp54_migrate_poll`
gains a `__u32 last_outcome` field (consumes one `reserved[]` slot — still
additive, no relayout); per-tenant kmod storage gains a `u8 last_outcome`.

### Item 6 — SC2 verification scope — **ACCEPTED + PUSH**
PROPOSE→COMMIT latency JSON is a hard deliverable for the SC6 v1.5-eventfd
decision.
**PUSH — the latency JSON must capture all five:**
1. **median** PROPOSE→COMMIT (typical case);
2. **p95 and p99** PROPOSE→COMMIT (tail);
3. worst-case **under contention** (multiple concurrent migrations);
4. worst-case **under tenant slow-response** (tenant ACKs slower than nominal);
5. **PROPOSED→ABORT latency under a quiet period** (item 4 PUSH).
All five surfaced in `TRACK_3_SC2_CLOSEOUT.md` for SC6.

---

## SC2 build — adjudicated procedure

**Source modifications limited to 4 files:** `cipher_ioctl.h`,
`cipher_cp54_sched.c`, `cipher_dev.c`, `cipher_internal.h`.
**Untouched:** libcipher_rt `ebc0baaa`, libcipher_v2 `86618c30`,
cipher_kv_bridge `fca6843d`, all other kmod `.c`, all test harnesses.

**Anchor rotation:** kmod `8d777dfb` → SC2 anchor; preserve `.pre_sc2`
fallback alongside the existing `.pre_track3`.

**Verification chain:** build → existing 6 isolation tests PASS (no NR 1–15
behaviour change) → new SC2 unit tests (NR 16–20 + state machine + gap-detect +
rate-limit + `last_outcome`) → capture 5-scenario latency → smoke test
W1/W2/W3 within ±3 % mean-vs-mean of the pre-baseline → PASS commits the SC2
anchor; FAIL rolls back to `.pre_sc2`/`.pre_track3` and surfaces.

**Deliverables:** SC2 source patches (4 files); `TRACK_3_ABI.md`;
`TRACK_3_SC2_PRE_BASELINE.md`; `TRACK_3_SC2_BUILD_LOG.md`;
`TRACK_3_SC2_LATENCY.json`; `TRACK_3_SC2_REGRESSION.md`;
`TRACK_3_SC2_CLOSEOUT.md`.

**Stop after SC2 closes with smoke test PASS — adjudication before SC3.**
Memory updated at each milestone (adjudication / pre-baseline / build /
smoke / closeout).

**First action:** pre-baseline measurement — re-run W1/W2/W3 once on the
current `8d777dfb` kmod, record in `TRACK_3_SC2_PRE_BASELINE.md`.
