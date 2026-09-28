# Track 3 SC6 — MEASUREMENTS (within-A · clause-2 · cost · quiet-period)

**Date:** 2026-05-19. **Substrate:** kmod `285d102e` + libcipher_rt `83afd1ca`
(unchanged — no SC6 anchor rotation). **Harness:** `sc6_churn.py` —
5 seeds, config A, 3 migratable + 2 pinned barriered decode tenants;
`sc6_quiet_period.py`. dmesg clean throughout.

This report closes the two SC5 coverage gaps (within-A latency, clause-2) and
characterizes migration cost + the quiet-period worst case. It does **not**
re-open F1 (the SC5 headline, adjudicated).

---

## §1 — L2 probe scope clarification (resolved before the numbers)

There are **two distinct `%smid` probes** — they must not be conflated:

| probe | site | frequency | in the migration-cost number? |
|---|---|---|---|
| **L2 verification** | `cipher_migrate.py` `step()`, inside the migration branch (`if rc==0 and verify_l2`) | **post-migration only** — matches SC3 design memo §3 | **yes** (`l2_ms`) |
| clause-2 instrumentation | `sc6_tenant.py`, `cm.probe_sms()` each round | every round, all tenants | **no** — pure harness scaffolding for §3 |

The migration-cost `l2_ms` (§4) is therefore a **per-migration** figure, not
per-round. The per-round probe is SC6 measurement instrumentation for the
clause-2 check (§3); a production handler runs no per-round probe.

---

## §2 — Within-A: migratable vs pinned latency (closes SC5 gap 1)

5 seeds, config A, decode-latency (ms), per opt-in class:

| seed | migratable p50/p95/p99 | pinned p50/p95/p99 |
|---|---|---|
| LIFO       | 30.9 / 59.3 / 66.1 | 35.8 / 61.2 / 65.9 |
| FIFO       | 35.3 / 59.9 / 65.8 | 28.6 / 55.8 / 63.9 |
| MIDDLE_OUT | 34.4 / 58.5 / 64.8 | 32.0 / 55.3 / 62.4 |
| RAND_A     | 32.1 / 56.2 / 61.2 | 36.4 / 57.4 / 63.2 |
| RAND_B     | 31.9 / 57.1 / 63.4 | 32.6 / 55.8 / 62.2 |

**Finding — the two classes are statistically indistinguishable.** migratable
p50 spans 30.9–35.3; pinned p50 spans 28.6–36.4 — the ranges overlap and the
sign of (migratable − pinned) is **inconsistent** across seeds (LIFO/RAND_A:
migratable lower; FIFO/MIDDLE_OUT: migratable higher; RAND_B: ~equal). p95/p99
are within ~3 ms across classes. The per-migration cost (§4, ~1.3 ms primitive)
amortized over ~3–4 migrations across hundreds of decode steps is **below the
per-class latency noise floor**. The 1.6B-4 "variance per opt-in class"
headline: **the opt-in class carries no measurable decode-latency penalty** in
this workload — a strong v1 result.

## §3 — Clause-2 disjointness (closes SC5 gap 2)

Per barriered round, every live tenant's `%smid`-observed SM set was collected
(contemporaneous — the round is lockstep-barriered) and checked **pairwise-
disjoint across all tenants**.

**`clause2_fails = 0` — all 5 seeds, every round.** Combined with clause 1
(observed ⊆ allocated — the live L2 check on every migration), Track 3's
disjointness invariant is now **runtime-probed for both clauses** under
multi-tenant churn with live migration. No substrate bug.

## §4 — Migration-cost characterization (SC6-3)

Per-migration cost, bracketed directly with `perf_counter` in `step()`,
pooled over **17 migrations** (5 seeds):

| component | p50 | p95 | p99 | min–max |
|---|---|---|---|---|
| **migrate primitive** (drain→build→L1→swap→release) | **1.26 ms** | 1.59 ms | 1.59 ms | 1.22–1.59 ms |
| L2 `%smid` verification probe | 26.8 ms | 40.1 ms | 40.1 ms | 0.34–40.1 ms |

**The migration mechanism is ~1.3 ms.** Per-seed primitive p50 spans only
1.24–1.28 ms — **flat across all 5 seeds** (which differ in stranding
severity), empirically confirming the count-preserving expectation: cost is a
single green-context rebuild, independent of how far the partition moves.
**This validates the ~1.7 ms Step 1.5 prior** — in fact slightly faster,
because the drain (`cuCtxSynchronize`) happens at a safe point with no
in-flight kernels to wait on. At the v1 rate limit (≤1 migration/tenant/10 s)
the migrate primitive is ~0.013 % overhead.

**The L2 probe (~27 ms) dominates the tenant-observed cost — and is not
intrinsic.** `probe_smid_k<<<4096, 64>>>` launches 4096 blocks to verify SM
placement on an 8-SM partition; the cost is the oversized grid + sync, not the
verification itself. A right-sized probe (enough waves to cover the partition
— hundreds of blocks, not 4096) would cut this by ~10×. **Recommendation
(→ SC6-4 / v1.5):** right-size the L2 probe. Total per-migration tenant-observed
cost today ≈ primitive (~1.3 ms) + L2 (~27 ms) ≈ **~28 ms**, per migration,
rate-limited — operationally bounded (~0.28 % worst-case at ≤1/10 s), and
~10× reducible.

## §5 — Quiet-period PROPOSED→ABORT (SC2 item-4 PUSH)

The kmod's PROPOSED→ABORT timeout is **lazy** — it fires only when an eval
pass runs (on a FREE/COMPACT), never on a timer.

| regime | result |
|---|---|
| **quiesced** — PROPOSE, then only POLL (no eval) | after **75 s** of quiescence the migration was **still PROPOSED**; the abort fired on the **first FREE after**. Latency = **unbounded-until-activity**. |
| **churning** — PROPOSE, then a FREE every 1 s | abort fired at **30.03 s** = the 30 s kmod timeout + ~one churn interval. **Bounded.** |

**v1.5 recommendation (explicit, per the item-4 PUSH):**
- For the **churning multi-tenant target regime** — the regime Track 3 exists
  for — the worst case is **~30–31 s**, well under any reasonable bound.
  **v1's lazy timeout is accepted** as a documented v1 boundary.
- For a deployment that can **quiesce** with a migration pending, the abort is
  **unbounded** — the tenant's reservation strands until activity resumes.
  **If such quiescence is possible, v1.5 should add an explicit timer/eventfd.**
  This is a documented v1 boundary + a named v1.5 deferral, with the empirical
  numbers above backing the decision (→ `TRACK_3_CLOSEOUT.md`).

## §6 — Production deployment guidance (SC6-3)

- **Opt in (`CIPHER_MIGRATABLE=1`)** for high-churn multi-tenant deployments
  (varied partition lifetimes, frequent alloc/free): F1's ~70 % stranding
  reduction is the payoff, and §2 shows **no measurable decode-latency cost**.
- **Stay pinned** for low-churn / long-lived partitions: migration would rarely
  trigger; the opt-in adds nothing.
- **Policy values:** the SC4 defaults (gap 1 / sustain 2 s / rate 10 s) showed
  no cost compounding (§4 cost is flat, below the latency noise floor) —
  **no retune indicated.**
- **Operator view:** `/proc/cipher/migrations` (policy + counters + per-tenant
  state) is the live runbook surface.

## §7 — Verdict (SC6-2 + SC6-3)

| item | result |
|---|---|
| within-A migratable-vs-pinned | **no measurable per-class latency penalty** (SC5 gap 1 closed) |
| clause-2 cross-tenant disjointness | **0 fails, all 5 seeds** (SC5 gap 2 closed) |
| migration primitive cost | **~1.3 ms, flat** — validates the ~1.7 ms prior; count-preserving ⇒ constant |
| L2 probe cost | ~27 ms per migration — oversized grid, ~10× reducible (recommendation) |
| quiet-period | churning ~31 s (bounded, v1 OK); quiesced unbounded (v1.5 timer if quiescence possible) |
| TFGATE KL / dmesg | all pass / clean |

No clause-2 violation, no regression, dmesg clean. **SC6-2 + SC6-3 complete.**
Stopping for adjudication before SC6-4 (`TRACK_3_CLOSEOUT.md`).
