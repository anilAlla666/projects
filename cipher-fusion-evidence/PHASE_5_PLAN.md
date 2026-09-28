# Phase 5 — campaign plan / CP sequencing

**Date:** 2026-05-17. Written during the post-CP-4.8 paper-work phase (pod
suspended; calendar/cost priority on Phase 5 — founder decision).
**Status:** Phase 4 CLOSED (`cp_4_8/CP_4_8_REPORT.md`). Phase 5 scoped here;
no CP started. Each CP has its own design memo — adjudicated CP-by-CP.

**Anchors at Phase 5 start (held, md5-verified through CP 4.8):**
kmod 0.4.8 `e2f50452`, libcipher_rt `c2c5d313`, libcipher_v2 `86618c30`.

---

## 1. What Phase 5 is

Phase 4 validated the composed substrate on **synthetic / single-stream
drivers** and proved a tenant-density ceiling of ≥100. Phase 5 makes the
substrate **real against the serving stacks operators actually run** and
measures the multi-tenant product story end-to-end. It is the bridge between
the Phase 4 engineering close and the **Phase 6 operator pilot**.

Five CPs, each its own design memo:

| CP | Title | Calendar | Design memo |
|---|---|---|---|
| 5.1 | vLLM / TGI live-decode integration | 2–3 wk | `PHASE_5_CP_5_1_DESIGN_MEMO.md` |
| 5.2 | KV offload hierarchy | 4–6 wk | `PHASE_5_CP_5_2_DESIGN_MEMO.md` |
| 5.3 | Partition-aware Marlin | 4–12 wk | `PHASE_5_CP_5_3_DESIGN_MEMO.md` + `PHASE_5_CP_5_3_SONG_HAN_ENGAGEMENT_SCOPE.md` (advisory engagement scope) |
| 5.4 | Per-tenant arbitration extension | 2–3 wk | `PHASE_5_CP_5_4_DESIGN_MEMO.md` |
| 5.5 | 100-tenant integration measurement | 1–2 wk | `PHASE_5_CP_5_5_DESIGN_MEMO.md` |

These memos supersede the earlier standalone scoping notes by consolidating
them into the per-CP structure: `PHASE_5_CP_5_1_PLAN.md` → CP 5.1 memo;
`PHASE_5_ARCHITECTURE_REVISION.md` → CP 5.3 memo; `PHASE_5_ALLOCATOR_DIAGNOSTIC.md`
and `cp_2_4/PHASE_5_MARLIN_PARTITION_CONSTRAINT.md` → CP 5.3 / 5.4 memos. The
originals are preserved as history; the per-CP memos are the artifacts to
adjudicate.

## 2. Dependency graph

```
  CP 4.8 close ───┬──► CP 5.1 (vLLM/TGI) ───────────────┐
                  │                                     │
  Phase 4.6 KV ───┴──► CP 5.2 (KV offload) ─────────────┤
                                                        ├──► CP 5.5 ──► Phase 5 CLOSE
  CP 2.4 Marlin ──────► CP 5.3 (partition Marlin) ──────┤        (100-tenant
  root cause                                            │     integration measure)
                                                        │
  CP 4.8 density ─────► CP 5.4 (arbitration) ───────────┘
```

- **CP 5.1** — unblocked by CP 4.8 (the WL04/10/16 vLLM env-fails seeded it)
  and the Phase 4.6 KV-dedup primitive. Unblocks the **Phase 6 operator pilot**.
- **CP 5.2** — unblocked by Phase 4.6 (`cp_4_6_5_6` — KV-dedup primitive
  validated 100-proc, cuIpc cross-process verified). Live-KV wiring.
- **CP 5.3** — unblocked by the CP 2.4 Marlin-hang root cause + Fix A.
  Unblocks Marlin routing through partitioned tenants — the largest single
  lever in the composed lift.
- **CP 5.4** — unblocked by the CP 4.8 density ceiling and the
  2026-05-16 allocator-exhaustion diagnostic.
- **CP 5.5** — the Phase 5 close CP; depends on **all four** of 5.1–5.4.

## 3. Sequencing and critical path

- **CP 5.3 is the critical path.** 4–12 wk, the spread driven by a single
  unresolved unknown (does Marlin's split-K protocol generalise to
  `grid < 132` — CP 5.3 memo §5/§6). It must start **first and run parallel**
  to 5.1 / 5.2 / 5.4 — its longest-lead-time external engagement (Song Han)
  should be opened immediately.
- **CP 5.1 / 5.2 / 5.4 run in parallel** with 5.3. Wall-clock for the three
  together ≈ max(2–3, 4–6, 2–3) = **4–6 wk** if staffed concurrently.
- **CP 5.5 closes Phase 5** — 1–2 wk, cannot start until 5.1–5.4 land.
- **Phase 5 wall-clock ≈ critical path 5.3 → 5.5 = 5–14 wk**, the spread
  inherited entirely from CP 5.3's unknown.

## 4. Phase-boundary commitment to flag for adjudication

**CP 5.1 is a stated prerequisite for the Phase 6 operator pilot** — not for
Phase 5 *close*. This creates a deliberate decision point: Phase 6 *can* begin
once CP 5.1 lands (~3 wk in), in parallel with the remainder of Phase 5,
rather than waiting for the full Phase 5 close (5–14 wk). Whether to overlap
Phase 6 with Phase 5 this way, or hold Phase 6 until CP 5.5 closes, is a
sequencing call for adjudication — surfaced here, not pre-decided.

## 5. Adjudication note

Per campaign discipline (`design-memo → approve → build`, one atomic CP at a
time, wait for adjudication between each): these five memos are scoping-level
design memos. Each is independently rejectable. A CP's detailed build memo is
re-elaborated when its STEP starts; nothing here pre-commits build sequencing
beyond the dependency edges in §2.
