# Phase 5 — architecture revision: partition-aware Marlin is a P0 prerequisite

**Date:** 2026-05-16. **Trigger:** the CP 2.4 Marlin-hang root cause
(`cp_2_4/MARLIN_HANG_ROOT_CAUSE.md`). **Status:** strategic finding — revises
the Phase 5 plan. Supersedes the candidate-shape list in
`cp_2_4/PHASE_5_MARLIN_PARTITION_CONSTRAINT.md` (which still listed
time-slicing as an option — rejected here).

---

## 1. The original assumption — now falsified

Phase 4 / the Phase 5 plan assumed **green-context SM partitioning composes
with every actuator**, Marlin included: each tenant gets an 8-SM (or N×8)
green-context slice, and all CIPHER actuators run inside that slice.

## 2. The finding (CP 2.4 Fix A)

The Marlin INT4 GEMM kernel is **structurally full-GPU**: `grid = SM count`
(132), persistent-style, with inter-CTA split-K coordination through a `locks`
buffer — its CTAs must be co-resident. Launched into an 8-SM green context it
**deadlocks** (confirmed by instrumentation). Fix A pins Marlin to the primary
(full-GPU) context for CP 2.4's single-tenant scope. But:

**Marlin and green-context SM-partitioning do not compose.** A partitioned
multi-tenant deployment cannot route a tenant's GEMMs through Marlin as built.

## 3. Why SM slicing stays the multi-tenancy primitive

The reflex fix — time-slice Marlin across the full GPU instead of partitioning
— is **rejected**:

- **Billing.** Spatial SM slices are metered capacity; time-slicing makes
  per-tenant GPU cost non-deterministic and hard to bill.
- **Isolation.** Time-slicing reintroduces the noisy-neighbour tail-latency
  variance that spatial partitioning exists to remove (cf. T4.2.4e).
- **Differentiation.** Time-slicing is what vLLM / TGI multi-tenant already
  do. SM slicing — hard spatial isolation with per-slice actuators — is the
  CIPHER product differentiator. Giving it up erases the moat.

SM slicing is the primitive. The actuators must be made to fit it — not the
other way round.

## 4. Implication

**A partition-aware Marlin GEMM is required for the multi-tenant product
story.** Without it, the highest-value actuator (Marlin INT4, the largest
single lever in the 2.96× composition) cannot run on partitioned tenants.

## 5. Three candidate shapes

1. **Rewrite Marlin partition-aware.** Make `grid` track the partition SM
   count and the split-K / `locks` protocol correct for `grid < 132` (drop
   the all-CTAs-co-resident assumption). If the existing protocol generalises,
   moderate; if it needs redesign, large. Deep CUDA work + numerical
   re-validation. This is the path that keeps Marlin's INT4 lift under
   partitioning.
2. **Hybrid actuators per partition size.** Marlin for large slices (≥ the SM
   count its protocol needs), a partition-tolerant fallback GEMM for small
   slices. Avoids the rewrite but fragments the actuator story and caps
   Marlin's reach.
3. **External kernel expertise.** The partition-aware split-K rewrite is
   specialist GPU-kernel work — engage outside expertise rather than absorb
   the full redesign cost in-house.

## 6. Engagement plan

The partition-aware split-K rewrite (shape 1) is exactly the kind of
specialist kernel work to bring in **Song Han** as advisor — the Devang intro
is pending; this finding is the concrete problem statement to open that
engagement with. The §10 constraint (Marlin INT4 correctness when two models
share the actuator) joins the same scope — both are "make the Marlin actuator
correct beyond its original single-model / full-GPU validation envelope."

## 7. Revised Phase 5 prerequisites

**Partition-aware Marlin is a P0 prerequisite of Phase 5 — not a Phase 5
deliverable.** Multi-tenant build work that routes Marlin through partitioned
contexts cannot start until the kernel composes with SM slices.

## 8. Calendar (honest)

**4–12 weeks** for the kernel work, with the spread driven by one unknown: if
Marlin's existing split-K protocol generalises to `grid < SM-count` with
bounded changes, ~4 weeks; if the persistent-grid assumption is load-bearing
and the protocol needs redesign, ~12. Resolving that unknown is the first task
(and the first question for the Song Han engagement).

## 9. What this does NOT affect

- **CP 2.4** — single-tenant scope; Fix A (Marlin full-GPU) is correct. The
  §2 partitioning constraint does not affect the CP 2.4 gate. The §10
  constraint *does* descope the CP 2.4 model-draft spec arm — CP 2.4 ships
  the n-gram draft instead (see §10).
- **The 2026-05-28 Ditlev meeting** — the demo is the T4.6.4 cross-tenant KV
  dedup + the CP 3.3 per-tenant MFU dashboard; neither depends on partitioned
  Marlin. The demo stands.

The §2 finding reshapes Phase 5 sequencing only.

## 10. Second constraint — substrate × model-draft speculative decode (CP 2.4, 2026-05-16)

**Trigger:** the CP 2.4 sub-task (iii) Llama-arm bisection
(`cp_2_4/CUDNN_ATTN_MARLIN_HANG.md`). A second actuator/substrate
incompatibility, distinct from §2:

- **The hang (3a) — closed.** Running the speculative-decode model draft on
  its own CUDA stream hangs cuDNN's runtime-compiled attention engine under
  the v2 substrate (`cuKernelSetAttribute` driver spin). Resolved for CP 2.4
  by running the draft on the default stream — zero benefit lost (the draft
  never overlaps the target; the handoff is fully synchronous). Knob
  `CIPHER_SPEC_DRAFT_STREAM`, default 0.
- **The acceptance collapse (3b) — the Phase 5 item.** With the hang gone,
  the model-draft arm's draft acceptance collapses from **0.490** (substrate
  off) to **0.036** (substrate on) — same draft, prompt, token budget. The
  model-draft Llama arm yields *negative* lift under the full substrate. Not
  yet root-caused; leading hypothesis: Marlin INT4 quantizing **both** the 1B
  draft and the 8B target so the draft no longer tracks the target's argmax —
  a two-model × per-weight-INT4 interaction (a weight-cache key collision, or
  genuine quantization divergence between the two models).

**Same drawer as §2.** Both are the Marlin INT4 actuator, validated
single-model / single-context, breaking on a new axis — §2 on SM
partitioning, §10 on a second concurrent model. The multi-tenant product
runs many models behind one substrate; an actuator that miscomputes when a
second model shares it is a multi-tenancy blocker, exactly like the
partitioning one.

**Phase 5 scope.** Diagnosing 3b — does Marlin's weight quantization /
caching stay correct when two models of different sizes route through the
actuator — joins the partition-aware Marlin rewrite (§5) in the Song Han
engagement scope (§6).

**CP 2.4 impact:** the model-draft spec arm is descoped from CP 2.4. CP 2.4
ships the **n-gram draft** (workload-universal, substrate-clean — bisection
test T2). The model-draft 1.75× primary criterion is documented as not met,
deferred here.
