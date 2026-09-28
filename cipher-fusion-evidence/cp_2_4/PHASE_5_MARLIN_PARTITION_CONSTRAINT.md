# Phase 5 — Marlin × SM-partitioning: architectural constraint (TRACKED)

**Date raised:** 2026-05-16. **Status:** open question for the Phase 5 design
memo. **Not a CP 2.4 blocker** (CP 2.4 gate is single-tenant; Fix A pins
Marlin to the full GPU — see `MARLIN_HANG_ROOT_CAUSE.md`).

---

## The constraint

The Marlin INT4 GEMM kernel (`cipher_rt_marlin_engine.cpp`,
`marlin_gemm_launch`) is **structurally full-GPU**:

- It launches `grid = SM count` blocks (132 on this H100), persistent-style —
  one block per SM, all CTAs expected co-resident.
- The blocks coordinate a split-K reduction through an inter-CTA `locks`
  semaphore buffer: block N signals block M.
- If the CTAs are **not** all co-resident, the inter-CTA protocol deadlocks —
  resident CTAs spin on `locks` signals from CTAs that cannot be scheduled.

Confirmed empirically: launched into the 8-SM green context, the 132-block
Marlin GEMM deadlocks; GPU pegs 100 % forever (`MARLIN_HANG_ROOT_CAUSE.md`).

**Therefore: the Marlin actuator and green-context SM-partitioning cannot
compose.** A tenant whose GEMMs route through Marlin must run on the full SM
array. CP 2.4 is single-tenant, so Fix A (pin Marlin to the primary/full
context) is correct and sufficient. **Phase 5 multi-tenancy** — where the
substrate partitions the GPU per tenant *and* actuators run inside those
partitions — must resolve this directly.

## Three candidate shapes for Phase 5

**(1) Time-slice Marlin across the full GPU.** Marlin-routed tenants do not
get a spatial SM partition; instead they time-share the full 132-SM array
(scheduler-level or a CIPHER arbitration token). Pro: no kernel change, keeps
Marlin's performance. Con: abandons spatial isolation for Marlin tenants —
reintroduces the noisy-neighbour variance SM-partitioning exists to remove;
needs a fair time-slice policy.

**(2) Partition-aware Marlin kernel rewrite.** Rework the Marlin kernel so
`grid` tracks the partition's SM count and the split-K / `locks` protocol is
correct for `grid < 132` (no all-CTAs-co-resident assumption). Pro: Marlin
becomes partition-native — full composition. Con: large, deep CUDA work
(IST-DASLab Marlin's design is tied to the persistent-grid assumption);
numerical-correctness re-validation required; the original B≥8 performance
envelope may shift on a small partition.

**(3) Non-Marlin actuator for multi-tenant.** Marlin stays single-tenant /
full-GPU only; partitioned multi-tenant decode uses a different matmul
actuator (e.g. a partition-tolerant INT4 GEMM, or cuBLAS passthrough inside
the partition with a different CIPHER lever). Pro: clean separation, no Marlin
rewrite. Con: gives up Marlin's INT4 lift for partitioned tenants; needs the
substitute actuator built and gated.

## Recommendation for the Phase 5 design memo

Decide (1) vs (2) vs (3) explicitly — it shapes whether SM-partitioning is a
universal substrate or a per-actuator-conditional one. Lowest-risk near-term
is **(3)** (Marlin = full-GPU lever; partitioned tenants use another actuator);
**(2)** is the only path that makes Marlin itself partition-native and should
be scoped as its own task if Marlin INT4 is wanted under partitioning.

Open question — carry into the Phase 5 design memo.
