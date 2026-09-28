# Phase 5 — CP 5.3: Song Han advisory engagement — SCOPE DOCUMENT

**Date:** 2026-05-17. **Status:** Engagement scope — complete, ready to share
on warm intro. Companion to `PHASE_5_CP_5_3_DESIGN_MEMO.md`: that memo is the
technical design (the partition-aware Marlin problem); this document is the
commercial / engagement scope it deliberately does not cover.

---

CP 5.3 — partition-aware Marlin — is the Phase 5 **critical path**
(`PHASE_5_PLAN.md` §3). The partition-aware split-K rewrite is specialist
GPU-kernel work; the campaign brings in **Song Han** as advisor rather than
absorbing the full redesign in-house (`PHASE_5_ARCHITECTURE_REVISION.md` §6).
The technical problem statement — Marlin's structurally full-GPU `grid = 132`
persistent-style kernel deadlocking in an 8-SM green context, plus the
two-model INT4 acceptance-collapse axis — lives in the CP 5.3 design memo.
This document defines the **engagement**: structure, commercial terms,
milestones, working arrangement. It is the single artifact to share with Song
Han ahead of the first call.

## §1 — Engagement structure

- **Duration.** 4–12 week scope; realistic mid-point **8 weeks**. The spread is
  the CP 5.3 load-bearing unknown — whether Marlin's split-K / `locks` protocol
  generalises to `grid < 132` with bounded changes (~4 wk) or needs redesign
  (~12 wk) (`PHASE_5_CP_5_3_DESIGN_MEMO.md` §5).
- **Role.** Advisory, deliverable-driven — **not full-time**. Song Han owns the
  kernel design and reference implementation; CIPHER engineering owns substrate
  integration and campaign measurement.
- **Adjudication.** Single point of contact — **Anil** — for technical
  adjudication. Consistent with campaign discipline (`design-memo → approve →
  build`, one adjudicator): each milestone deliverable is accepted or rejected
  by Anil against the §3 gate criteria.

## §2 — Commercial terms

- **Equity.** 0.5–1.0% advisory equity on the **Y Combinator FAST template**
  (Founder/Advisor Standard Template). The range tracks FAST's own tiering —
  0.5% standard advisor, 1.0% expert-tier; the critical-path nature of CP 5.3
  and the specialist scope support the upper half.
- **Vest.** 2-year vest, 6-month cliff — FAST standard, time-based monthly.
- **Cash.** None — equity-only engagement.
- **IP.** Standard advisor IP assignment and confidentiality, per the FAST
  template.
- **Termination.** Mutual termination right at any milestone (§3) — either
  party may end the engagement at a milestone boundary.

**Milestone gating — stated precisely.** FAST vesting is **time-based**, not
milestone-based. The four §3 milestones gate the engagement through the
**mutual termination right**: a missed or rejected milestone is grounds for
either party to terminate, which halts further vesting and forfeits the
unvested balance. That is the mechanism by which milestones gate vesting.
**Decision (2026-05-17, Anil):** the engagement uses the **unmodified FAST
template** with standard time-based vesting — frictionless execution is
preferred over custom milestone-triggered vesting tranches for a 4–12 week
engagement.

This document scopes the engagement; the executed **FAST agreement is the
binding instrument** and should be finalised with legal review before
signature.

## §3 — Deliverables / milestones

Four milestones. Each is adjudicated by Anil (§1) against its gate; each is a
mutual-termination boundary (§2).

| # | Week | Deliverable | Gate |
|---|---|---|---|
| **M1** | 2 | Partition-aware Marlin INT4 GEMM **kernel design memo** — architectural approach to 8-SM partition execution, split-K coordination, warp specialization, shared-memory layout. | Design memo adjudicated by Anil; resolves the load-bearing unknown — does the split-K / `locks` protocol generalise to `grid < 132`. |
| **M2** | 4 | **Reference implementation** passing numerical correctness against full-GPU Marlin. | Bit-exact, or cosine similarity > 0.99, vs full-GPU Marlin reference output. |
| **M3** | 6–8 | **Performance characterization** across all 16 Green Context partitions. | **≥ 75% throughput retained** vs full-GPU Marlin, stable across all 16 partitions. |
| **M4** | 8–12 | **Integration** into CIPHER substrate dispatch, validated under multi-tenant load. | Partition-aware Marlin dispatches correctly through the substrate; multi-tenant correctness and isolation preserved. |

M1 being itself a design memo is deliberate — it slots the advisor's first
deliverable into the campaign's `design-memo → approve → build` discipline, so
the load-bearing unknown is adjudicated before reference-implementation effort
is committed.

## §4 — Working arrangement

- **Async-first**, with a **weekly sync check-in**.
- **Source access.** `libcipher_rt` source under NDA — the Marlin actuator
  lives inside `libcipher_rt` (anchor `c2c5d313`); CP 5.3 is the first Phase 5
  CP to modify a substrate anchor.
- **Code review** on critical-path kernel changes.
- **Co-authorship rights** on any publications resulting from the engagement.

## §5 — Why Song Han

- **AWQ and SmoothQuant authorship** — directly relevant: CP 5.3 is INT4 GEMM
  kernel work, and axis B is a two-model INT4-quantization correctness problem.
- **MIT HAN Lab partition-aware / hardware-efficient kernel research
  precedent** — the closest external match to the partition-aware split-K
  rewrite.
- **Existing relationship** via **Devang Sachdev** — the warm-intro path (§7).
- **No competitive conflicts identified** at scoping time — to be reconfirmed
  before the FAST agreement is executed.

## §6 — Engagement success criteria

- **M3 is the unlock.** Hitting M3 (≥ 75% throughput retained, stable across
  16 partitions) clears the actuator blocker that otherwise forces partitioned
  multi-tenant decode to give up the Marlin INT4 lever — the largest single
  lever in the composed lift. M3 is the gate that lets the **CP 5.5 100-tenant
  integration measurement** route partitioned tenants through Marlin.
- **Strategic payoff.** With partition-aware Marlin in hand, the Phase 5 goal —
  100 concurrent real-decode tenants on one H100 (132 SMs) — can be pursued
  **without trading away spatial SM isolation** (the CIPHER differentiator;
  `PHASE_5_ARCHITECTURE_REVISION.md` §3). CP 4.8 already proved a tenant-density
  ceiling ≥ 100 on synthetic drivers; CP 5.3 keeps the highest-value actuator
  available when that density is delivered against real partitioned tenants.

## §7 — Purpose / how this document is used

- **Ready for the warm intro.** This document is complete and ready to share
  the moment **Devang Sachdev's** warm intro to Song Han lands — a founder
  action item, and the gating dependency on CP 5.3 (`PHASE_5_PLAN.md` §3).
- **Single pre-call artifact.** One document to send Song Han ahead of the
  first call — engagement structure, terms, milestones, and working
  arrangement in one place.
- **Makes the first call productive, not exploratory.** Scope clarity up front
  means the first call can go straight to the technical load-bearing unknown
  (§3 M1) rather than spending the call defining the engagement.

## §8 — Anchors

kmod 0.4.8 `e2f50452`, libcipher_rt `c2c5d313`, libcipher_v2 `86618c30`
(= `libcipher_v2.so.v0.2.0` per the D6 resolution,
`cp_4_8/D6_LIBCIPHER_V2_ANCHOR_RESOLUTION.md`). CP 5.3 modifies the Marlin
actuator inside `libcipher_rt`; the new `libcipher_rt` anchor is recorded when
CP 5.3 ships.
