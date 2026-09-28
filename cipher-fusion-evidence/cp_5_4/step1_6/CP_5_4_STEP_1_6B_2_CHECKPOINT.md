# CP 5.4 — Step 1.6B-2 (mixed-deployment orchestrator) — CHECKPOINT

**Date:** 2026-05-19. **Status: orchestrator BUILT + sanity run — mechanics
PASS, substrate CORRECT, but a measurement-methodology finding STOPS for
adjudication before 1.6B-3.** Anchors unchanged — throwaway harness + two
test-harness edits (preserved).

---

## Built

- `cp54_s16_orchestrator.py` — launches the POOL executor first (claims all 15
  groups on the low prefix), then N PARTITION tenants via
  `CUDA_INJECTION64_PATH` (each `ALLOCATE` shrinks the POOL from the high end),
  then the POOL's batch clients, then `START`s the partitions. Collects
  per-tenant JSON, runs the disjointness check, computes per-tenant decode CV.
- `cp54_pool.py` (session1) — added `PoolBinding.probe_observed_sms()` (per-round
  `%smid` probe). Preserved `.pre_cp5_4_step1_6`.
- `cipher_batch_executor_gen.py` (session1) — emits `POOL_ROUND r SMS …` each
  round. Preserved `.pre_cp5_4_step1_6`. Both are Phase B test harnesses, not
  campaign anchors.

Launch order is load-bearing: POOL-first so `cp54_pool`'s count-only
`torch.cuda.GreenContext.create` low-prefix pick aligns with the kmod POOL
allocation (Step 1.3b' Test B precedent). Partitions-first would put the POOL
off the low prefix → self-verify failure.

## Sanity run — `b2_op5_sanity/` (2 × 16-SM PARTITION + POOL, 5 rounds)

**Orchestration mechanics: all PASS.** POOL-first launch, both partitions
`READY` (part0 `grp_mask=0x1800` groups 11-12; part1 `0x6000` groups 13-14 —
the high end, POOL kept low prefix 0-10), barrier release, all processes
exited rc=0, JSON collected and parsed.

**Substrate: CORRECT.**
- **clause 1 (the runtime enforcement check): 0 fails** — every member's
  kernels stayed within its own kmod grant. Partition self-verify 0/5 each;
  POOL `%smid` self-verify PASS at every green-ctx (build + both rebuilds).
- Per-tenant KL gates PASS (kl_max 5.5e-5). POOL `agg_tok_s` 118.8.
- POOL resized 15→11 (partitions arrived) then 11→15 (partitions exited) —
  correct kmod reaper + reclaim behaviour.

**FINDING — per-round-index disjointness comparison is invalid without
co-extensive tenant runtimes (a harness methodology bug, NOT a substrate
defect).** `clause2_fails=6`. Root cause: the PARTITION tenants finished their
5 rounds in ~10 s; the POOL executor ran ~17 s (warmup + 5 batched rounds).
When the partitions exited, the kmod `do_exit` reaper freed their groups and
the POOL's `check_resize()` correctly reclaimed all 15 — so `POOL_ROUND 2/3/4`
legitimately span all 132 SMs (no partitions left). The orchestrator then
compared `POOL_ROUND 2` (partitions already gone) against partition `ROUND 2`
(captured ~10 s earlier, partitions alive) → 6 false overlaps. No real
disjointness violation occurred at any instant — clause 1 proves it.

**This also affects 1.6B-4:** partitions finishing early means their later
rounds are NOT under POOL contention — the per-tenant variance-under-contention
measurement needs the two classes co-resident for the whole window.

## Decision required before 1.6B-3 (surfaced for adjudication)

- **Option A — per-round lockstep barrier (recommended).** A filesystem
  barrier so all tenants advance round-by-round together: within a round they
  run free (real contention), at the round boundary they sync. Round r becomes
  the same wall-window for everyone → observed-set clause 2 is valid, and no
  tenant exits early → contention is co-extensive (fixes 1.6B-4 too). ~6 lines
  in the partition tenant + the executor (test harness) + orchestrator
  coordination. Matches the user's 1.6B-2 scope ("per-round synchronization
  barrier" + "per-round disjointness verification, continuous").
- **Option B — reframe the gate.** Keep clause 1 (observed ⊆ own kmod grant,
  per round — the real enforcement check, already passing) as the load-bearing
  runtime gate; make clause 2 = kmod grants of concurrently-live tenants are
  pairwise disjoint (from reported `grp_mask`s + tenant lifetimes), not
  observed-set pairwise. No executor change, but departs from the memo §6
  "observed sets" wording and still needs co-extensive runtimes for 1.6B-4.

## 1.6B-2A — per-round lockstep barrier (Option A, adjudicated) — LANDED

Filesystem barrier added: each member, after round r, drops
`/tmp/cp54_s16_<tag>_<who>_r<r>.done` and blocks on `…_all_r<r>.go`; the
orchestrator's barrier-coordinator thread waits for all members' `.done` then
writes `.go`. Partition tenant + executor each gained ~6 lines (test
harnesses, preserved `.pre_cp5_4_step1_6`); the executor's `POOL_ROUND` probe
moved to *after* the round's generate so probes are contemporaneous.

**Re-run `b2_op5_b2a/` (2×16-SM PARTITION + POOL, 5 rounds): VERDICT PASS.**
- All 5 rounds lockstep-synced (`BARRIER round 0..4 — all 3 members done`).
- **disjointness PASS — clause1_fails=0, clause2_fails=0.** POOL held 88 SMs
  (11 groups, low prefix 0–10) every round; partitions on the high groups
  (11–14); zero overlap.
- Partitions + POOL co-resident throughout (exits within 1 s of each other).
- Per-tenant KL gates PASS (kl_max 5.5e-5); POOL agg 111.4 tok/s; per-tenant
  decode CV ≈ 0.31.

Step 1.6B-2 is COMPLETE. The orchestrator is correct and reusable for 1.6B-4.

## Next

Per adjudication: deferral audit (`CP_5_4_DEFERRAL_AUDIT.md`) → STOP for
per-deferral adjudication → then 1.6B-3 / 1.6B-4. OP-2 arithmetic pinned at
2×16 SM + 88-SM pool (memo §11.1).
