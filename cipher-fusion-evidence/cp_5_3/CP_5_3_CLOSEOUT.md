# CP 5.3 — Partition-aware Marlin — CLOSEOUT

**Closed:** 2026-05-18. **Verdict: CLOSED — both axes PASS.** Anchor
`a7ac8e97` (libcipher_rt) unchanged at close.

---

## Scope

CP 5.3 made the Marlin INT4 GEMM actuator compose with multi-tenant
green-context SM partitioning. Per the design memo (`PHASE_5_CP_5_3_DESIGN_MEMO.md`)
it carried **two axes of the same "Marlin validated single-model, breaking on
a new axis" problem**:

- **Axis A** — partition-aware GEMM: Marlin is structurally full-GPU
  (`grid = 132`), deadlocking in an 8-SM green context.
- **Axis B** — two-model speculative-decode acceptance collapse: model-draft
  spec acceptance fell 0.490 → 0.036 with the substrate on.

## Constituent steps and verdicts

| step | report | verdict |
|---|---|---|
| **STEP 1** — split-K diagnostic | `cp_5_3/CP_5_3_STEP_1_REPORT.md` | Resolved the load-bearing unknown: Marlin's split-K / `locks` protocol **does** generalise to `grid < 132` with bounded rework (~4 weeks) — **no escalation** to the design-memo §4 shape-2 hybrid-actuator fallback. |
| **STEP 2A** — partition-aware Marlin wiring (Axis A) | `cp_5_3/CP_5_3_STEP_2_REPORT.md` | **PASS** on the §7 binding gates A-num ∧ B ∧ C, all re-verified: green-ctx GEMM numerically correct, exactly 8 SMs touched (no spill), two concurrent partitions provably disjoint, KU1 deadlock did not recur. Shipped the `grid ← partition SM count` fix in libcipher_rt. |
| **STEP 2B Step 1** — F1 discriminator (Axis B) | `cp_5_3/CP_5_3_STEP_2B_STEP_1_REPORT.md` | **RESOLVED.** Three-arm discriminator: the 0.490→0.036 collapse is **dominated by finding F1** (cross-stream race, degenerate target verify logits) — the F1 fix recovers 74.8% of the gap, restoring the spec arm to a working state (0.380 acceptance, 2.91 tok/round, coherent). Residual ~0.11 is **benign INT4 quantization divergence (H3)** — H2 (weight-cache collision) structurally excluded. No defect; no Step 2 needed. |

## Axis verdicts (adjudicated 2026-05-18)

- **Axis A — PASS.** Partition-aware Marlin runs correct and isolating on
  `grid < 132` partitions. STEP 2 §10 items 1–2 resolved.
- **Axis B — RESOLVED.** The acceptance collapse is root-caused (F1) and
  already fixed in `a7ac8e97`; the residual is a benign accuracy property of
  INT4 + model-draft speculative decode, not a defect. Speculative decode is
  viable on the F1-fixed substrate. STEP 2 §10 items 3–4 resolved.

CP 5.3 design-memo §4 gate met via path 1 (primary): the GEMM is correct on
`grid < 132` **and** Axis B resolved — no hybrid-actuator descope.

## Finding F1 — cross-reference

F1 (shipped full-GPU Marlin degenerate on real decode) was *surfaced* by
CP 5.3 STEP 2 §5, *fixed* in the CP 5.6 detour (anchor `dc804eb3 → a7ac8e97`,
[[cipher-f1-fullgpu-marlin-broken]]), and STEP 2B Step 1 confirmed F1 as the
dominant cause of the Axis B collapse — closing the loop. The Marlin-spec-decode
incompatibility CP 2.4 deferred to Phase 5 is, in the end, the same F1.

## Anchors at close

- **libcipher_rt `a7ac8e97`** — carries STEP 2A's partition-aware Marlin fix
  (built as `dc804eb3`) and the F1 fix. Unchanged by STEP 2B (measurement
  only). This is the CP 5.3 close anchor.
- kmod 0.4.8 `e2f50452`, libcipher_v2 `86618c30` — unchanged.
- Preserved: `libcipher_rt.so.pre_cp5_3_step2` (`c2c5d313`, the CP 5.3 start
  artifact / arm-b pre-F1 substrate).

## Phase 5 status

CP 5.1, 5.2, 5.3, 5.6 CLOSED; CP 5.4 (per-tenant arbitration) and CP 5.5
(100-tenant soak — gated on Track 2 weight-sharing + `FUTURE_SCOPE/A`) pending.

## Artifacts

`cp_5_3/` — the three step reports above, `step2b/` (driver, analyzer,
per-arm JSON + teacher-forced logits), `cp_5_3_step1_evidence.tar.gz`,
`cp_5_3_step2_evidence.tar.gz`.
