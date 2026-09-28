# Track 3 (Dynamic SM Migration) — SC5 CLOSEOUT

**Date:** 2026-05-19. **SC5 scope:** the folded SC4 policy build + the SC1
fact-2 constrained POOL grant (SC5-2) + the headline measurement (SC5-3).
**Verdict: SC5 COMPLETE — headline confirmed, two coverage gaps surfaced.**
kmod anchor `98da2d1f` → `285d102e` (rotated at SC5-2). libcipher_rt `83afd1ca`,
libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d` unchanged. Stopping for
adjudication before SC6.

---

## §1 — What SC5 delivered

**SC5-2 (build):** SC4 policy folded in — 4 module parameters
(`/sys/module/cipher_kmod/parameters/`, mode 0644), `/proc/cipher/migrations`,
per-tenant `migration_count` + aggregate counters. **`cp54_pool_claim_low_prefix`**
— the SC1 fact-2 constrained-low-prefix POOL grant that the SC5-1 design memo
found was specified but never built — now in place as the shared A/B safety
floor. Regression smoke PASS (`TRACK_3_SC5_REGRESSION.md`).

**SC5-3 (measurement):** 10-scenario A/B sweep + destroy-fault, on kmod
`285d102e` (`TRACK_3_SC5_MEASUREMENTS.md`).

## §2 — Headline result (F1) — CONFIRMED

Live SM migration keeps the POOL **~70 % less stranded** than the no-migration
substrate under sustained partition churn — **mean stranding 2.78 (A) vs 9.49
(B)**, robust across all 5 free-order seeds (LIFO/FIFO/MIDDLE_OUT/RAND_A/RAND_B).
B shows **0 migrations every seed** (clean single-variable switch). **Every
tenant's TFGATE KL gate passes** in all 10 scenarios. Migration cost is **below
the workload noise floor** (A/B latency p50/p95/p99 statistically
indistinguishable — ~7 migrations × ~1.7 ms lost in decode-step variance).
**Destroy-fault at scale PASS** — 3 faulted + 2 clean tenants, all 5 KL gates
pass, faulted tenants commit-and-leak and continue, clean tenants unaffected.

This is the substrate-layer claim Track 3 set out to prove: *a primitive that
keeps a multi-tenant GPU's batch pool compact under arbitrary partition churn,
correctness-preserving, at a cost that vanishes into the workload.*

## §3 — Findings surfaced (not papered over)

- **F2 — seed sensitivity present but small.** The predicted "FIFO/LIFO
  smaller delta" only partly held (FIFO lowest stranding, but LIFO highest —
  prediction wrong for LIFO). Root cause: a harness artifact — 5 single-group
  partitions + no POOL process leave 10/15 groups statically free, dominating
  the stranding metric over churn-induced gaps. Does not weaken F1.
- **Two coverage gaps** (`TRACK_3_SC5_MEASUREMENTS.md §7`): (1) within-A
  migratable-vs-pinned latency was **not** measured — all sc5 tenants ran
  migratable; (2) disjointness **clause 1** is live-verified every migration
  (the L2 `%smid` probe) but **clause 2** (cross-tenant) is not runtime-probed
  by sc5 — it is structural + W2-verified on `285d102e`.

## §4 — Recommendation for adjudication

No disjointness violation, no regression, dmesg clean — nothing forces a STOP.
The two coverage gaps are a **decision for the user**:
- **(i) accept F1 as the headline**, fold the harness refinement (POOL process
  + mixed migratable/pinned tenants + cross-tenant clause-2 `%smid`) into SC6's
  cost-characterization work; or
- **(ii) refine the harness and re-run SC5-3** before SC6.

Recommendation: **(i)** — F1 is the architectural claim and it is solid; the
refinement is naturally a cost-characterization concern, which is SC6's remit.

## §5 — Recommendations for SC6

1. **Production-default validation (Item-c PUSH):** SC5-3 ran the defaults
   (gap=1 / sustain=2 s / rate=10 s); migration cost stayed below the noise
   floor, so the defaults did **not** show cost compounding — no retune
   indicated by this data. SC6 should confirm on a denser workload.
2. **Migration-cost curve:** SC5 latency could not discriminate migration cost
   (below noise). SC6 should measure the migration round directly (the per-
   migration green-ctx rebuild) rather than via aggregate p99.
3. **Leak-rate boundary:** the destroy-fault commit-and-leak held; SC6
   documents the worst-case leaked-handle rate as a v1 boundary (SC3 item-6
   PUSH), prominently in `TRACK_3_CLOSEOUT.md`.
4. **Harness refinement** (§4 option i) — the within-A split + clause-2 probe.
5. The v1/v2 boundary table up-front in `TRACK_3_CLOSEOUT.md` (SC1 item-5 PUSH).

## §6 — Anchors

| artifact | anchor | state |
|---|---|---|
| kmod | `285d102e` | loaded (rotated at SC5-2; `.track3_sc5` / `.pre_sc5` fallbacks; src `e57a1a84`) |
| libcipher_rt | `83afd1ca` | unchanged |
| libcipher_v2 | `86618c30` | unchanged |
| cipher_kv_bridge | `fca6843d` | unchanged |

SC5-3 is measurement-only — no anchor rotation. A-vs-B was a module-param
switch on the one kmod.

## §7 — SC5 deliverables

`TRACK_3_SC5_DESIGN_MEMO.md`, `TRACK_3_SC5_PRE_BASELINE.md`,
`TRACK_3_SC5_BUILD_LOG.md`, `TRACK_3_SC5_REGRESSION.md`,
`sc5_tenant.py` / `sc5_churn.py`, `sc5_churn_results.json` +
`sc5_scenario_*.json`, `TRACK_3_SC5_MEASUREMENTS.md`, this closeout.

## §8 — Adjudication ask

**STOPPING — SC5 complete.** Confirm:
1. **SC5 = PASS** on F1 (the headline) — §2.
2. **The §3 findings** — F2 (seed artifact) and the two coverage gaps.
3. **§4 — choose (i)** accept F1 + refine in SC6 [recommended] **or (ii)**
   refine harness + re-run SC5-3.

On adjudication, proceed to **SC6** — migration-cost characterization +
CP 5.5 integration + `TRACK_3_CLOSEOUT.md` (the final Track 3 closeout).

---

## §9 — Measurement provenance (post-adjudication addendum, 2026-05-19)

SC5 adjudicated **Option (i)** — F1 accepted as the SC5 headline; the two
coverage gaps folded into SC6. To prevent any overclaiming, the explicit
delivered-vs-deferred split:

**SC5 measured (substantiated here):**
- A-vs-B **substrate-level** stranding under churn — F1 — 5 seeds × A/B.
- TFGATE KL correctness, every tenant, every scenario.
- Destroy-fault commit-and-leak at scale (3 faulted + 2 clean).
- Disjointness **clause 1** (observed SMs ⊆ allocated) — live L2 `%smid` on
  every committed migration.

**Deferred to SC6 (NOT substantiated by SC5 — do not cite SC5 for these):**
- Within-A **migratable-vs-pinned** per-class latency (the 1.6B-4 split).
- Disjointness **clause 2** (cross-tenant pairwise-disjoint) runtime probe.
- Migration-**cost curves** (per-migration cost — SC5's aggregate p99 could
  not resolve it; below the noise floor).
- Worst-case **PROPOSED→ABORT under a quiet period** (SC2 item-4 PUSH).

`TRACK_3_CLOSEOUT.md` (SC6-4) carries this as a claim→measurement provenance
table so a reviewer sees exactly which run substantiates which claim.
