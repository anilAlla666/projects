# CP 5.3 — STEP 1: split-K generalisation diagnostic — VERDICT REPORT

**Date:** 2026-05-17. **Status:** STEP 1 executed and complete. **Verdict:
PASS → 4-week branch.** This report applies the pre-registered §5 pass/fail
rule of `CP_5_3_STEP_1_SPLITK_DIAGNOSTIC_SCOPE.md` to the runlogs, declares the
branch, and surfaces for adjudication. It does **not** close CP 5.3 and does
**not** start STEP 2 — `"adjudicated — proceed to STEP 2"` is required first.

---

## §1 — What STEP 1 resolved

CP 5.3 design memo §5 identified **one load-bearing unknown**: *does Marlin's
split-K / `locks` protocol generalise to `grid < 132`*. The calendar spread
(4 wk vs 12 wk) hangs on it. STEP 1 is the diagnostic that resolves it.

**Resolved: YES, it generalises.** The Marlin INT4 GEMM kernel is numerically
correct and deadlock-free at every `grid` value tested — from 132 down to 1 —
across all four committed problem shapes. The split-K protocol is genuinely
grid-parametric, not 132-block-coupled. → **4-week branch**: STEP 2 is the
bounded `grid` rework, not the shape-2 hybrid-actuator descope.

## §2 — Method (as executed)

- **Harness:** `cp53_splitk_diag` (md5 `66a7f1b54fec0a7c0ed14f00394a92ad`), a
  standalone side-build linking the Marlin engine + NVRTC kernel src with
  `grid` exposed as a launch parameter. Shipped `libcipher_rt.so` **not
  touched** (scope memo §6).
- **Reference oracle:** per-run cuBLAS FP16 GEMM on the *dequantized* INT4
  weights (the same int4+scales the Marlin path consumes). Grid-independent —
  the same reference is the correctness target for every `grid`.
- **Sweep:** 4 shapes × `grid ∈ {132, 66, 33, 16, 8}`, G=128, 90 s watchdog
  per cell. Runlog `cp53_splitk_sweep.runlog` (complete `2026-05-17T18:34:49Z`).
- **Falsification probes:** shapes S2/S4 × `grid ∈ {4, 2, 1}` — below the
  committed sweep floor, to force multi-slice-per-block handoff (§3.1) and the
  degenerate single-CTA no-split-K case. Runlog `cp53_splitk_probes.runlog`
  (complete `2026-05-17T18:42:18Z`; the first probe run was cut off by session
  interruption at the final cell and was re-run clean — sweep was unaffected).
- **G=-1 descoped** per scope memo §7a — the shipped actuator is 128-groupwise
  only; `G=-1` is dead code in the production path. Diagnostic is G=128 only.

## §3 — Results — grid sweep (§5 gate set)

Tolerance ε per shape = **2× the `grid=132` Marlin-vs-cuBLAS max-abs-error**
for that shape (the established-correct control, doubled for headroom).

| Shape | M,N,K | grid=132 (control) | ε = 2× | 66 | 33 | 16 | 8 | Verdict |
|---|---|---|---|---|---|---|---|---|
| S1 large-N small-K | 16,4096,512  | 0.031250 | 0.062500 | 0.015625 | 0.015625 | 0.015625 | 0.015625 | **PASS** |
| S2 small-N large-K | 16,128,8192  | 0.250000 | 0.500000 | 0.250000 | 0.250000 | 0.125000 | 0.062500 | **PASS** |
| S3 cp24-hang shape | 16,4096,4096 | 0.062500 | 0.125000 | 0.062500 | 0.062500 | 0.062500 | 0.062500 | **PASS** |
| S4 small × small   | 16,128,512   | 0.015625 | 0.031250 | 0.015625 | 0.015625 | 0.015625 | 0.015625 | **PASS** |

All 20 sweep cells: `status=COMPLETED`, `sync=no error`, `nan=0`. **Zero
deadlocks** (zero 90 s watchdog kills). Every non-132 cell beats ε, mostly by a
wide margin (S2 grid=8 sits at 0.0625 vs ε=0.500).

## §4 — Results — falsification probes (below the gate floor)

| Shape | grid=4 | grid=2 | grid=1 | ε (from §3) | Verdict |
|---|---|---|---|---|---|
| S2 small-N large-K | 0.062500 | 0.062500 | 0.015625 | 0.500000 | **PASS** |
| S4 small × small   | 0.015625 | 0.015625 | 0.007812 | 0.031250 | **PASS** |

All 6 probe cells: `COMPLETED`, `nan=0`, no deadlock — including `grid=1`,
where there is no inter-CTA split-K at all (one CTA does every k-tile
serially).

**§3 falsification surface — genuinely exercised, not merely "it ran":**
- **Slice handoff at small grid (§3.1):** S2 (K=8192, many k-tiles) at
  grid ≤ 8 makes each block cross multiple slices. Correct at every grid.
- **Idle participants / over-subscription (§3.3):** S4 has
  `k_tiles·n_tiles = 8` work tiles; at grid ∈ {132,66,33,16} the split-K
  reduction has idle blocks contributing nothing. Correct — the `locks` wait
  does not stall on idle participants.
- **`locks` buffer sizing (§3.2):** `ws_bytes` is a function of N only; no
  under/over-allocation observed at any grid.
- **Degenerate grid=1:** no split-K, lowest error of all — confirms the
  reduction telescopes cleanly rather than depending on 132-way contention.

**Error decreases as grid shrinks.** This is the expected signature of a
*correct* split-K reduction: fewer CTAs → fewer cross-CTA FP16 partial-sum
accumulations → less reduction-order noise. It is positive evidence the
protocol composes, not a warning sign. INT4 quant error is the floor (grid=132
is non-zero vs the FP16 reference); small grid is at or below that floor.

## §5 — Scope boundary (what STEP 1 does NOT prove)

Stated plainly so it is not softened downstream:

1. **The harness launches `grid` as a knob on the full 132-SM device.** Every
   runlog line reads `device sm_count=132`. STEP 1 proves the
   **kernel / split-K / `locks` protocol is numerically correct and
   deadlock-free when launched with `grid < 132`**. It does **not** prove
   Marlin runs inside a real 8-SM green-context partition.
2. **The CP 2.4 deadlock was a co-residency failure**, not a grid-correctness
   failure: the shipped actuator launches `grid=132` (132 blocks × 96 KiB
   smem); on an 8-SM green context only ~16 are co-resident, 116 starve at the
   split-K barrier → hang. STEP 1 sidesteps co-residency by launching only
   `grid` blocks — which all schedule. STEP 1's contribution is therefore:
   it proves the **fix shape** (`grid ← partition SM count`) yields a *correct*
   kernel, so the rework is bounded and not a protocol redesign.
3. **STEP 2's actual work:** wire `grid = partition_sm_count` into
   `cipher_rt_marlin_engine.cpp:800` (currently `int grid = g_sm_count;`) and
   launch into a real green context, then re-validate numerically.
4. **Axis B untouched.** The two-model INT4 acceptance collapse
   (0.490 → 0.036, design memo §3 / §10) is silent in STEP 1 — a separate STEP.
5. **CP 5.3 is not closed.** Design memo §4 gate needs both (a) grid<132
   correctness *in partitions* and (b) Axis B resolved. STEP 1 partially
   advances (a). The verdict selects the STEP 2 branch; it closes nothing.

## §6 — Verdict

**PASS — split-K generalises → 4-week branch.** Per §5 PASS rule: all non-132
grid values complete with no deadlock AND meet ε for all four shapes at G=128.
The probes extend the evidence two octaves below the gate floor with no change
in verdict. STEP 2 is the **bounded `grid` rework** (design memo §4 path), not
the shape-2 hybrid-actuator descope.

## §7 — Anchors

Held and unchanged through STEP 1: kmod 0.4.8 `e2f50452`, libcipher_rt
`c2c5d313`, libcipher_v2 `86618c30`. STEP 1 is an experimental side-build; no
shipped artifact was modified, so no anchor moved. Per design memo §7 the
libcipher_rt anchor moves only when CP 5.3 *ships* a fix — that is STEP 2+.

## §8 — Evidence

`cipher-fusion-evidence/cp_5_3/`:
- `CP_5_3_STEP_1_SPLITK_DIAGNOSTIC_SCOPE.md` — pre-registered plan + §5 rule
- `cp53_splitk_diag` — harness binary, md5 `66a7f1b54fec0a7c0ed14f00394a92ad`
- `cp53_splitk_diag.cpp`, `cp53_marlin_engine_diag.cpp`,
  `cipher_rt_marlin_kernel_src.cpp` (+ headers) — harness source
- `cp53_splitk_sweep.runlog` — 20-cell sweep, complete
- `cp53_splitk_probes.runlog` — 6-cell falsification probes, complete
- `run_sweep.sh`, `run_probes.sh`, `build_diag.sh` — reproduction scripts
- `CP_5_3_STEP_1_REPORT.md` — this report

## §9 — Adjudication ask

STEP 1 verdict is **PASS → 4-week branch**. Requesting adjudication of this
report. On `"adjudicated — proceed to STEP 2"`, STEP 2 is scoped (design-memo
first, per campaign discipline) as the bounded `grid`-rework: wire
`grid ← partition SM count` into the shipped Marlin actuator, launch into a
real green context, re-validate numerically. Axis B is diagnosed under its own
STEP. No actuator edit until STEP 2 is adjudicated.
