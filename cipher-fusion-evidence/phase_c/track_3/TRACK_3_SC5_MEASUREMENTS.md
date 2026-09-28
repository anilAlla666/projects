# Track 3 SC5-3 — HEADLINE MEASUREMENTS

**Date:** 2026-05-19. **Substrate:** kmod `285d102e` + libcipher_rt `83afd1ca`.
**Harness:** `sc5_churn.py` — 5 free-order seeds × {A: migration on,
`gap_min_grps=1`; B: migration off, `gap_min_grps=16`} = 10 scenarios, +
1 destroy-fault scenario. 5 single-group PARTITION decode tenants per scenario,
staggered-lifetime exits = the free order. dmesg clean throughout.

---

## §1 — Expected directions (stated before measurement — Item-3 PUSH)

1. **A-vs-B stranding** — A < B on fragmenting seeds; FIFO/LIFO smaller delta.
2. **Latency** — migratable tenants pay a bounded per-migration cost
   (~1.7 ms, Step 1.5); p50 ≈ unchanged, p99 slightly higher than pinned.
3. **Destroy-fault** — all tenants' KL gates pass; faulted tenants
   commit-and-leak and continue; clean tenants unaffected.
4. **Disjointness** — holds every seed; a violation ⇒ substrate bug ⇒ STOP.

---

## §2 — Results — the 10-scenario A/B sweep

| seed | A stranding mean | B stranding mean | A−B delta | A migrations | B migrations | KL gate (A/B) |
|---|---|---|---|---|---|---|
| LIFO       | 2.95 | 9.84 | 6.89 | 5 | 0 | pass / pass |
| FIFO       | 2.55 | 9.00 | 6.45 | 6 | 0 | pass / pass |
| MIDDLE_OUT | 2.60 | 9.50 | 6.90 | 8 | 0 | pass / pass |
| RAND_A     | 2.96 | 9.52 | 6.56 | 8 | 0 | pass / pass |
| RAND_B     | 2.84 | 9.60 | 6.76 | 9 | 0 | pass / pass |
| **mean**   | **2.78** | **9.49** | **6.71** | — | — | — |

Per-tenant decode latency (ms), all-tenant pooled:

| seed | A p50/p95/p99 | B p50/p95/p99 |
|---|---|---|
| LIFO       | 27.6 / 63.7 / 70.5 | 26.9 / 63.5 / 69.4 |
| FIFO       | 24.0 / 61.5 / 69.3 | 24.1 / 59.6 / 66.6 |
| MIDDLE_OUT | 25.0 / 63.0 / 69.1 | 24.6 / 63.2 / 70.7 |
| RAND_A     | 24.5 / 62.6 / 69.8 | 25.4 / 64.9 / 70.5 |
| RAND_B     | 26.5 / 61.8 / 67.9 | 26.3 / 60.9 / 69.2 |

---

## §3 — F1 (headline) — CONFIRMED, robust to seed

**A reduces POOL stranding ~70 % across all 5 seeds** (mean 2.78 vs 9.49);
**B shows exactly 0 migrations on every seed** (migration disabled via
`gap_min_grps=16` — confirms the config-based A-vs-B is a clean single-variable
switch); **every tenant's TFGATE KL gate passes** in all 10 scenarios. The
substrate-level claim — *"A delivers the same correctness as B with
substantially less stranding under sustained churn, at a bounded migration
cost"* — holds, and holds independent of the free-order pattern.

A's residual stranding (~2.78, not 0): full compaction is rate-limited
(1 migration/tenant/10 s) and incremental; the ~52 s scenario window does not
reach complete compaction. A achieves a **70 % reduction**, not elimination,
in this window — honest claim.

## §4 — F2 — seed sensitivity is PRESENT but SMALL (prediction partly wrong)

The expected "FIFO/LIFO smaller delta" did **not** cleanly appear. Quantitatively:

- B-config stranding ranges only **9.00 (FIFO) … 9.84 (LIFO)** — a ~0.8-unit
  spread on a ~9.5 baseline. The seeds **do** differentiate, but at sub-unit
  scale. FIFO is the lowest (direction-consistent with "FIFO fragments less");
  **LIFO is the highest — the prediction was wrong for LIFO** (LIFO frees the
  high partitions first, accumulating high free groups → more stranding).
- **Root cause — a harness-design artifact.** The harness uses 5 single-group
  partitions and **no POOL process**; groups 5–14 (10 of 15) are statically
  free for the whole run, and the stranding metric (free groups above the
  lowest live partition) is dominated by that static free region — not by the
  churn-induced gaps the free order controls. The free-order signal is real
  but swamped by the ~10-group static baseline.

This is surfaced as a finding, not papered over (Item-3 PUSH). It does **not**
weaken F1 — A-vs-B is dominated by *whether migration runs*, and that signal
(2.78 vs 9.49) is an order of magnitude larger than the seed spread.

## §5 — Destroy-fault at scale — PASS

`MIDDLE_OUT` config A, `CIPHER_SC3_FAULT=destroy` on tenants 0/1/2 (faulted),
tenants 3/4 clean:

| tenant | fault | KL gate | migrations committed |
|---|---|---|---|
| t0 | destroy | **pass** | 1 |
| t1 | destroy | **pass** | 1 |
| t2 | destroy | **pass** | 1 |
| t3 | none | **pass** | 2 |
| t4 | none | **pass** | 3 |

All 5 tenants' KL gates pass. The 3 faulted tenants each committed a migration
*despite* the forced `cuGreenCtxDestroy` failure — the **commit-and-leak path
holds at scale**: a faulted tenant continues correctly on its new context, the
old handle is leaked (bounded). Clean tenants 3/4 are unaffected. dmesg clean.
(SC6 documents the leak rate as a v1 boundary — SC3 item-6 PUSH.)

## §6 — Latency — migration cost is below the workload noise floor

A and B per-tenant latency are **statistically indistinguishable** (p50
~24–28 ms, p99 ~67–71 ms, both configs, every seed). ~5–9 migrations of
~1.7 ms each, spread over hundreds–thousands of decode steps per tenant,
cannot move p99 (~69 ms, dominated by ordinary decode-step variance). This is
a **good** result for Track 3 — migration is cheap enough to vanish into the
noise — but it means the latency axis cannot discriminate migratable from
pinned at this workload's noise level (see §7 gap 1).

## §7 — Coverage gaps (surfaced honestly)

The SC5 harness, as built in SC5-2, does **not** cover two adjudicated items:

1. **Within-A migratable-vs-pinned is NOT measured.** Every sc5 tenant ran
   `CIPHER_MIGRATABLE=1`. The adjudicated metric 3 (the 1.6B-4 per-opt-in-class
   variance split) needs A-config tenants of *both* classes side by side. B is
   the substrate-level all-pinned config, not "pinned tenants inside A" — §6's
   A-vs-B latged indistinguishability is an *ancillary* finding, not a
   substitute for the within-A split.
2. **Disjointness — clause 1 verified, clause 2 not runtime-probed by sc5.**
   Clause 1 (observed SMs ⊆ allocated) **is** live-checked — that is exactly
   the L2 `%smid` probe `cipher_migrate.MigrateHandler.step()` runs on every
   committed migration (~36 across the sweep + 5 in the destroy-fault A run).
   Clause 2 (pairwise-disjoint across tenants) is **not** runtime-probed in
   sc5; it holds structurally (kmod ledger exclusivity + the constrained
   grant) and was verified live by W2 (clauses 1+2, static) on this exact kmod
   `285d102e` in the SC5-2 smoke. Stated honestly: sc5 does not itself prove
   clause 2.
3. **Static-free dominance** (§4) — the harness's 5-single-group / no-POOL
   design makes the stranding metric measure the static free region more than
   churn fragmentation.

**Recommended harness refinement** (for adjudication — SC6 or an SC5-3 re-run):
add a POOL process consuming the low prefix (so "stranding" measures churn
gaps specifically), run A-config with **mixed** migratable/pinned tenants
(the within-A split), and add a cross-tenant clause-2 `%smid` probe.

---

## §8 — Verdict

**F1 — the Track 3 headline — is confirmed and robust:** live SM migration
keeps the POOL ~70 % less stranded than the no-migration substrate under
sustained partition churn, across all 5 free-order patterns, with correctness
(KL) preserved and migration cost below the workload noise floor;
commit-and-leak holds at scale. Disjointness clause 1 verified live on every
migration; clause 2 structural + W2-verified.

**Two adjudicated metrics were not covered by the SC5-2 harness** (§7) —
surfaced for adjudication: accept F1 as the headline and fold the refinement
into SC6, or refine the harness and re-run SC5-3. No disjointness violation,
no regression, dmesg clean — nothing forces a STOP; the gap is a coverage
decision for the user.
