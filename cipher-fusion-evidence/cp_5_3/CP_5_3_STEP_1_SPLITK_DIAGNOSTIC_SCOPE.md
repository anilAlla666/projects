# CP 5.3 — STEP 1: split-K generalisation diagnostic — SCOPE MEMO

**Date:** 2026-05-17. **Status:** STEP 1 scope memo — **plan for adjudication,
not yet executed.** CP 5.3 design memo approved (operator decision this
session: *"Approve CP 5.3 → start STEP 1"*). This memo verifies the design
memo's assumptions against current pod state and lays out the 7-item plan for
the STEP 1 diagnostic. Per campaign discipline (`plan → approve → execute`),
the diagnostic runs only after this memo is adjudicated.

---

## §1 — Design-memo assumptions verified against pod state

The CP 5.3 design memo (§1 Axis A, §6) rests on claims about the Marlin
actuator. Checked against the live tree this session:

| Design-memo claim | Pod-state check | Verdict |
|---|---|---|
| Marlin GEMM launches `grid = SM count` | `cipher_rt_marlin_engine.cpp:800` — `int grid = g_sm_count;`, where `g_sm_count` ← `rt_devattr(MultiProcessorCount)` (line 779), falling back to 132 | **confirmed** — `grid` is the *device* SM count, not the partition's |
| Inter-CTA split-K through a `locks` buffer | `marlin_ws_for_stream` (eng. 706–737) hands each stream a `MARLIN_WS_BYTES` device buffer used as `int *locks` (eng. 797); kernel uses it as a barrier semaphore (`kernel_src.cpp:207`) | **confirmed** |
| Persistent-style / co-residency assumed | kernel work split `iters = ceildiv(k_tiles*n_tiles*parallel, gridDim.x)` (`kernel_src.cpp:227`); per-slice participant count `slice_count` and `slice_idx` derived from `iters` in `init_slice` (249–280) | **confirmed — and see §3: the split is grid-parametric, which is the load-bearing question** |
| Launched into an 8-SM green context it deadlocks | `cp_2_4/MARLIN_HANG_ROOT_CAUSE.md` §2 H3 — instrumented build `b1a3424c`, 132 blocks × 96 KiB smem ⇒ ~16 co-resident on 8 SMs, 116 cannot schedule | **confirmed (prior evidence)** |
| Fix A pins Marlin to the primary context (single-tenant only) | `PrimaryCtxGuard` over quant + `cipher_rt_marlin_engine_dispatch` (eng. 811+); anchor `5e304549` → folded into current `c2c5d313` | **confirmed** |

Pod available for the STEP 1 diagnostic when adjudicated: H100 80GB HBM3,
driver 580.105.08, 0 MiB in use; `cipher_kmod` loaded, `/dev/cipher` mode
0666. Green-context machinery (`cipher_rt_green_ctx.c`) and Fix-A libcipher_rt
present.

**No design-memo assumption was contradicted.** STEP 1 may proceed to plan.

## §2 — The load-bearing unknown (restated, undecided)

CP 5.3 design memo §5: the 4-wk vs 12-wk calendar spread is driven by **one
unknown** — *does Marlin's split-K / `locks` protocol generalise to
`grid < 132`*. STEP 1 is the diagnostic that **resolves** that unknown. It does
not assume an answer. The verdict is whatever the runlog says; this memo
commits, in advance, to the experiment and the pass/fail rule that reads it.

## §3 — Falsification surface

The protocol is **grid-parametric in form** — `iters` divides total work by
`gridDim.x` — but parametric form is necessary, not sufficient, for
correctness at small grid. The diagnostic must hit each of these, not merely
observe "it ran":

1. **Slice handoff at small grid.** `kernel_src.cpp:263–280` — `delta_first =
   iters*blockIdx.x - col_first`, advancing `A`/`C`/`locks` pointers
   (`locks += n_tiles`) as a block crosses slice boundaries. At `grid = 8`
   each block does far more `iters` ⇒ more handoffs per block; verify the
   `locks` indexing stays correct when one block crosses multiple slices
   while peers have not started.
2. **`locks` buffer sizing.** `ws_bytes = (N/128)*max_par*sizeof(int)`,
   `max_par = 16` (eng. 784–789). Confirm this is a function of **N only**,
   not of grid — and that no slot count implicitly assumed 132-block
   contention. If sizing is grid-coupled, small grid may under- or
   over-allocate.
3. **Over-subscription / idle participants.** When `grid >
   k_tiles*n_tiles*parallel` (small partition × small problem), some blocks
   get `slice_iters = 0` (`init_slice`, 251–254). The split-K reduction must
   tolerate idle participants — confirm `slice_count` / `slice_idx` and the
   `locks` wait do not stall on a block that contributes nothing.
4. **The grid value itself.** `eng.:800` `grid = g_sm_count` is the device
   count. The diagnostic varies grid *as an experimental knob* — it does not
   yet change the shipped line (that is a later STEP).

## §4 — STEP 1 plan (7 items)

1. **Build a standalone Marlin-GEMM diagnostic harness** — a side-build that
   links the existing `marlin_gemm_launch` path (or a thin copy exposing
   `grid` as a parameter) with NVRTC kernel compilation unchanged. No edit to
   shipped `libcipher_rt.so`; the harness is an experimental binary.
2. **Reference oracle.** Compute the ground-truth GEMM as cuBLAS FP16 on the
   *dequantized* Marlin weights (INT4→FP16 via the engine's existing
   dequant), at `grid = 132` — the established-correct configuration. Capture
   `C_ref`.
3. **Grid sweep, deadlock axis.** Launch the Marlin GEMM at
   `grid ∈ {8, 16, 33, 66, 132}` for each problem shape (§5), each with a
   watchdog timeout. Record completes / deadlocks per (grid, shape).
4. **Grid sweep, numerical axis.** For every (grid, shape) that completes,
   compare output to `C_ref`; record max-abs-error and the §5 tolerance
   verdict.
5. **Falsification probes.** Targeted runs for §3 items 1–3: a small-partition
   × small-problem shape to force over-subscription; a large-K shape to force
   many slice handoffs per block; both `G ∈ {-1, 128}`.
6. **Write the runlog + STEP 1 verdict report** — `cp_5_3/cp53_splitk_*.runlog`
   and `CP_5_3_STEP_1_REPORT.md` — applying the §5 pass/fail rule, declaring
   the 4-wk or 12-wk branch with the evidence, and (if 12-wk) naming the
   failure mode.
7. **Tarball** the STEP 1 evidence; surface the verdict for adjudication
   (which gates whether CP 5.3 STEP 2 is the bounded `grid` rework or the
   shape-2 hybrid-actuator descope).

## §5 — Gate parameters (committed in advance)

Fixed **now**, before any number is seen, so the diagnostic is not ceremonial:

- **Grid sweep:** `grid ∈ {8, 16, 33, 66, 132}` — 8 = a realistic small
  green-context partition; 132 = the device baseline / known-good control.
- **Problem shapes** (M ≤ 64, N % 64 = 0, K % 128 = 0 — the engine's accepted
  envelope, `eng.:753–761`), each at `G ∈ {-1, 128}`:
  - **S1** large-N small-K: `M=16  N=4096 K=512`  (few k_tiles, many n_tiles)
  - **S2** small-N large-K: `M=16  N=128  K=8192` (many k_tiles — stresses
    slice handoff, §3.1)
  - **S3** square decode shape: `M=16  N=4096 K=4096` (the CP 2.4 hang shape)
  - **S4** small × small: `M=16  N=128  K=512` (forces over-subscription at
    `grid=132` and even `grid=33`, §3.3)
- **Reference:** cuBLAS FP16 GEMM on dequantized weights (§4.2), captured at
  `grid=132`.
- **Tolerance ε:** max-abs-error ≤ **2×** the `grid=132` Marlin-vs-cuBLAS
  error for the *same* shape (i.e. small-grid output must be no worse than the
  established-correct grid=132 output, doubled for headroom). The grid=132
  Marlin-vs-reference error is itself measured in this run — INT4 quant error
  is the floor, not zero.
- **PASS rule — "split-K generalises" (→ 4-wk branch):** *all* non-132 grid
  values complete (no deadlock) **and** meet ε for *all four* shapes at *both*
  `G`. Any deadlock, or any ε miss → **12-wk branch**; the STEP 1 report names
  which grid/shape failed and the failure mode (deadlock vs numerical).
- **Partial result handling:** if some grids pass and some fail, that is a
  12-wk verdict with a characterised boundary (e.g. "generalises for
  grid ≥ 33"), not a pass — reported as such for adjudication.

## §6 — What STEP 1 does NOT do

- Does **not** modify shipped `libcipher_rt.so` — the diagnostic is a side-build.
- Does **not** route Marlin through a real green context yet (STEP 1 isolates
  the *kernel/grid* question; green-context integration is a later STEP, gated
  on the §5 verdict).
- Does **not** address Axis B (two-model INT4 acceptance collapse) — Axis B is
  diagnosed in parallel under its own STEP per design memo §3.
- Does **not** touch the Song Han engagement — STEP 1's verdict is the
  concrete input that engagement would consume.

## §7 — Anchors

Held and unchanged through STEP 1: kmod 0.4.8 `e2f50452`, libcipher_rt
`c2c5d313`, libcipher_v2 `86618c30`. The STEP 1 diagnostic harness is an
experimental side-build and is **not** an anchor. Per design memo §7, CP 5.3
moves the libcipher_rt anchor only when it *ships* a fix — no STEP-1 anchor
change.

## §7a — STEP 1 execution addendum (2026-05-17): G=-1 descoped

During harness build (plan item 1), inspection of the engine quant/repack
pipeline established that `quantize_fp16_to_int4_groupwise_gpu`,
`marlin_repack_host`, and `ensure_weight_quantized_repacked` are
**128-groupwise only** — the shipped actuator (`cipher_rt_marlin_engine_dispatch`)
never emits a `G=-1` weight; the `G=-1` kernel variant is dead code in the
production path. Operator decision this session: **G=-1 descoped**, diagnostic
runs **G=128 only** (the shipped path). G=-1 grid-generalisation is off CP
5.3's critical path; if a future CP adds a `G=-1` quant path it runs its own
grid diagnostic. The §5 PASS rule is otherwise unchanged — "all 4 shapes" now
means all 4 at G=128.

## §8 — Adjudication ask

Approve this 7-item STEP 1 plan to execute the split-K generalisation
diagnostic. The diagnostic's verdict (§5 PASS rule) is itself an adjudication
point: it selects CP 5.3's STEP 2 between the bounded `grid`-rework path and
the shape-2 hybrid-actuator descope (design memo §3/§4).
