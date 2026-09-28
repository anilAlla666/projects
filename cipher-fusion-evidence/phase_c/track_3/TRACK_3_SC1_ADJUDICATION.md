# Track 3 (Dynamic SM Migration) — SC1 ADJUDICATION RECORD

**Date:** 2026-05-19. **Adjudicator:** user (campaign owner).
**Subject:** `TRACK_3_SC1_DESIGN_MEMO.md` §8 — eight adjudication asks.
**Verdict:** **ALL EIGHT ACCEPTED as recommended in the design memo. No
overrides.** Three items carry an explicit PUSH (an additional requirement on
a *downstream* SC, not a change to the SC1 recommendation).

This file is the durable adjudication of SC1. It unlocks SC2 *planning* only.
SC2 *build* is gated on a separate adjudication of `TRACK_3_SC2_PLAN.md`.
No source modified, no build, no GPU at adjudication time.

---

## Decision header (carried from the authorization)

**Track 3 (Dynamic SM Migration) is the v1 substrate primitive — NOT a v2
deferral.** Option A (live SM migration) is built. Option B
(`cp54_pool_claim_low_prefix`, the Step 1.6X-1 constrained-grant) is **NOT
shipped as the substrate** — it survives only as a *component* of A: the POOL
safety floor during the migration window, and the per-tenant graceful
fallback if migration cost exceeds budget (memo facts 2 + §3e).

---

## §8 items 1–8 — adjudicated

### Item 1 — Architecture — **ACCEPTED**
`PROPOSE → START → MIGRATING → ACK → COMMIT` state machine. Three group
states: **FREE / OWNED / RSVD**. Two-cmpxchg reaper sweep (`PACK(pid)→0` and
`RSVD(pid)→0` per group). **Count-preserving** migration for v1
(`popcount(new_mask) == popcount(old_mask)`, enforced kmod-side at PROPOSE and
client-side at `migrate()`). **B ⊆ A**: the constrained-low-prefix POOL grant
is retained as the POOL's safety floor *during* the migration window.

### Item 2 — Notification — **ACCEPTED + PUSH**
Poll-based delivery via the `CIPHER_CP54_QUERY`-class loop for v1. eventfd is
the scoped v1.5 upgrade path, not v1.
**PUSH:** SC2 **must measure PROPOSE→COMMIT latency end-to-end** so SC6 cost
characterization has empirical data backing any v1.5 eventfd-upgrade decision.
This is a hard SC2 deliverable, not optional instrumentation.

### Item 3 — Non-cooperation — **ACCEPTED**
**Abort-only.** Forced sync after timeout is **rejected**. Destroying a green
context under in-flight kernels is undefined behaviour on H100 — non-negotiable.
An unresponsive `migratable` tenant is treated exactly as a `pinned` tenant
for that proposal; the kmod ABORTs and the tenant keeps `old_mask`.

### Item 4 — Headline metric — **ACCEPTED**
Step 1.6B-4 reports decode-latency variance **per opt-in class** —
`migratable` vs `pinned` measured and reported **separately, not aggregated**.
Track 3 headline claim: **"isolation + bounded, priced migration cost."**

### Item 5 — v1 boundaries — **ACCEPTED + PUSH**
v1 scope: single-GPU H100; 15-group ledger (≤14 active partitions + POOL);
count-preserving migration only; migration rate ≤ 1/tenant/10 s. v1 defers to
v2: multi-GPU/NCCL migration; sub-ms eventfd push; resize-via-migration;
POOL/SHARED-client migration; numeric per-tenant cost budgets; MIGRATING-state
watchdog→reclaim.
**PUSH:** the v1/v2 boundary table must appear **prominently and up front in
`TRACK_3_CLOSEOUT.md`** — production reviewers see the boundaries first, not
buried in fine print.

### Item 6 — SC5 B-baseline — **ACCEPTED**
SC5 builds a **measured throwaway B branch** (`cp54_pool_claim_low_prefix`
alone, ~0.5 d, no anchor rotation) for an honest A-vs-B comparison.
Track 3 closeout claim, as adjudicated: **"A delivers the same correctness as
B with NO stranding under sustained churn, at a bounded migration cost that is
characterized."**

### Item 7 — Sequencing — **ACCEPTED**
**Serial:** Track 3 is the primary Phase 5 workstream; Track 2 SC3–SC6 runs
**after** Track 3 closes. Anchor-lineage clarity is preferred over parallelism.
CP 5.5 timeline impact (~3–4 days) is accepted.

### Item 8 — SC4 opt-in plumbing — **ACCEPTED**
`migratable` is carried as an **env var** (the `CIPHER_QOS_CLASS` /
Step 1.3b' Option 2A pattern). **libcipher_v2 stays at `86618c30` throughout
Track 3** — no rotation. Final SC4 implementation detail is deferred to SC4
design itself.

---

## Anchors at adjudication (unchanged — SC1 is paperwork)

| artifact | anchor | state |
|---|---|---|
| kmod `cipher_kmod.ko` | `8d777dfb` | unchanged; `.pre_track3` fallback preserved ✓ |
| libcipher_rt | `ebc0baaa` | unchanged; `.pre_track3` fallback preserved ✓ |
| libcipher_v2 | `86618c30` | unchanged — and stays unchanged through Track 3 (item 8) |
| cipher_kv_bridge | `fca6843d` | unchanged |

## Next gate

SC2 **planning** is unlocked → `TRACK_3_SC2_PLAN.md` (7-item build plan).
SC2 **build** is gated on adjudication of that plan. **No source modified, no
build, no GPU until the SC2 plan is adjudicated.**
