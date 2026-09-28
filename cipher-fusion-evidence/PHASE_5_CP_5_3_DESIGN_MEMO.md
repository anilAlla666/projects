# Phase 5 — CP 5.3: Partition-aware Marlin — DESIGN MEMO

**Date:** 2026-05-17. **Status:** Phase 5 CP — **critical path**. Not started
— scoped for adjudication. Consolidates `PHASE_5_ARCHITECTURE_REVISION.md` and
`cp_2_4/PHASE_5_MARLIN_PARTITION_CONSTRAINT.md` (preserved as history).

---

## §1 — Scope / what CP 5.3 ships

CP 5.3 makes the **Marlin INT4 GEMM actuator compose with green-context SM
partitioning** — the prerequisite for routing partitioned multi-tenant
decode through the substrate's highest-value lever. It carries **two axes of
the same problem**, both "Marlin validated single-model / full-GPU, breaking
on a new axis":

- **Axis A — partition-aware GEMM.** Marlin is structurally full-GPU:
  `grid = SM count` (132 on this H100), persistent-style, with inter-CTA
  split-K coordination through a `locks` buffer requiring all CTAs
  co-resident. Launched into an 8-SM green context it **deadlocks** (confirmed
  — `cp_2_4/MARLIN_HANG_ROOT_CAUSE.md`). CP 2.4's Fix A pins Marlin to the
  primary context for single-tenant scope; multi-tenant cannot. CP 5.3
  reworks `grid` to track partition SM count and makes the split-K / `locks`
  protocol correct for `grid < 132`.
- **Axis B — two-model acceptance collapse.** With the CP 2.4 model-draft
  speculative-decode hang resolved, draft acceptance collapses **0.490 →
  0.036** with the substrate on (`PHASE_5_ARCHITECTURE_REVISION.md` §10) —
  leading hypothesis: Marlin INT4-quantizing both the 1B draft and 8B target
  so the draft no longer tracks the target's argmax. CP 5.3 root-causes and
  fixes the two-model × per-weight-INT4 interaction.

## §2 — Why this CP / dependencies

**Dependency in.** CP 2.4's Marlin-hang root cause and Fix A established both
constraints; Fix A is correct *only* for CP 2.4's single-tenant scope.

**Dependency out.** Without partition-aware Marlin, the largest single lever
in the composed lift (Marlin INT4) **cannot run on partitioned tenants** —
multi-tenant decode would have to give up that lever or abandon spatial
isolation. CP 5.3 unblocks Marlin in the CP 5.5 100-tenant measurement and is
on the Phase 6 critical path. **SM slicing stays the multi-tenancy primitive;
the actuator is made to fit it** — time-slicing Marlin across the full GPU is
rejected (breaks deterministic billing, reintroduces noisy-neighbour
variance, erases the spatial-isolation differentiator —
`PHASE_5_ARCHITECTURE_REVISION.md` §3).

## §3 — Approach

The partition-aware split-K rewrite is specialist GPU-kernel work. Engage
**Song Han as advisor** (the Devang intro is pending; this memo is the
concrete problem statement to open that engagement). The first task — and the
first question for the engagement — resolves the load-bearing unknown (§5).
If Marlin's split-K protocol does not generalise within bounded effort, fall
back to the architecture-revision **shape 2** (hybrid: Marlin for large
slices, a partition-tolerant fallback GEMM for small slices) — accepting a
fragmented actuator story rather than an unbounded rewrite. Axis B is
diagnosed in parallel: does Marlin's weight quantization / caching stay
correct when two models of different sizes share the actuator (a weight-cache
key collision vs genuine quantization divergence).

## §4 — Gate criteria

CP 5.3 PASSES iff **one** of:

1. **Primary.** The Marlin GEMM runs correct and numerically re-validated on
   `grid < 132` partitions, with the INT4 lift preserved within a declared
   tolerance of its full-GPU envelope; **and** axis B is resolved — draft
   acceptance under the two-model substrate path is restored to within a
   declared tolerance of the substrate-off 0.490 baseline.
2. **Fallback.** The hybrid-actuator decision (shape 2) is taken with an
   explicit, gated partition-tolerant fallback GEMM for small slices — a
   documented descope of Marlin's reach, not a silent one.

Numerical-correctness re-validation is mandatory either way (INT4 GEMM output
vs reference).

## §5 — Calendar

**4–12 weeks** — the longest-lead-time CP in Phase 5, and its critical path.
The spread is driven by **one unknown**: if Marlin's existing split-K
protocol generalises to `grid < 132` with bounded changes, ~4 weeks; if the
persistent-grid assumption is load-bearing and the protocol needs redesign,
~12. Resolving that unknown is the first task. The Song Han engagement should
be opened **immediately** so this CP runs parallel to CP 5.1 / 5.2 / 5.4.

## §6 — Risks / known-unknowns

- **The load-bearing unknown** (§5) — whether the split-K / `locks` protocol
  survives a non-co-resident grid.
- **B≥8 performance envelope** — Marlin's designed regime; the per-partition
  rewrite may shift its performance on small slices (cf. the CP 4.5 TinyLlama
  B=1 regression — Marlin is a B≥8 actuator).
- **Numerical divergence** — an INT4 GEMM that miscomputes is worse than one
  that deadlocks; re-validation is non-negotiable.
- **External-engagement dependency** — the Devang→Song Han intro is pending;
  if it does not land, the rewrite cost is absorbed in-house, pushing toward
  the 12-week end.

## §7 — Anchors at CP start

kmod 0.4.8 `e2f50452`, libcipher_rt `c2c5d313`, libcipher_v2 `86618c30`.
CP 5.3 modifies the Marlin actuator inside libcipher_rt — the first Phase 5
CP to move a substrate anchor; the new libcipher_rt anchor is recorded when
CP 5.3 ships.
