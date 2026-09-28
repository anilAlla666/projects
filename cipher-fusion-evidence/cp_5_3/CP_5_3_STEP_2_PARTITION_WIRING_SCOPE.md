# CP 5.3 — STEP 2: partition-aware Marlin wiring — SCOPE MEMO

**Date:** 2026-05-17. **Status:** STEP 2 scope memo — **plan for adjudication,
not yet executed.** STEP 1 verdict adjudicated this session (operator:
*"Step 1 verdict accepted… 4-week branch via bounded grid rework. Proceed to
CP 5.3 Step 2."*). Per campaign discipline (`design-memo → show → approve → no
code until approved → build → gate → report`), this memo identifies what STEP 1
settled vs left open, verifies the wiring points against the live tree, lays
out the 7-item plan, and commits the gate in advance. **No code until this
memo is adjudicated.**

---

## §1 — Known vs known-unknown after STEP 1

**KNOWN (STEP 1 settled it):**
- The Marlin INT4 GEMM kernel and its split-K / `locks` protocol are
  **numerically correct and deadlock-free at any `grid` from 132 down to 1**,
  across 4 problem shapes (`CP_5_3_STEP_1_REPORT.md`). The protocol is genuinely
  grid-parametric. → reducing `grid` does not corrupt the result.
- Error is flat or *decreasing* as `grid` shrinks — fewer split-K partials,
  less FP16 reduction-order noise. The reduction telescopes cleanly.

**KNOWN-UNKNOWN (STEP 2 must resolve — STEP 1 explicitly did not):**
- **KU1 — co-residency in a real partition.** STEP 1 launched `grid=8` on the
  *full 132-SM device*: 8 blocks, trivially co-resident. STEP 2 launches
  `grid=8` inside an actual **8-SM green context** — 8 blocks × 96 KiB dynamic
  smem on 8 SMs. The split-K `locks` barrier requires *all* grid blocks
  co-resident; this is the exact CP 2.4 deadlock surface
  (`cp_2_4/MARLIN_HANG_ROOT_CAUSE.md`: 132 blocks, ~16 co-resident on 8 SMs,
  116 starved). Expected to pass at `grid=8` (1 block/SM, 96 KiB < H100's
  228 KiB) — but **must be verified, not assumed.**
- **KU2 — quant-context vs GEMM-context separation** (the load-bearing design
  question — see §3).
- **KU3 — SM spill.** Does a `grid=8` launch in an 8-SM green ctx stay on
  those 8 physical SMs, or leak onto neighbouring partitions' SMs?
- **KU4 — multi-partition interference.** Two concurrent 8-SM partitions, each
  its own green ctx — do they stay spatially disjoint and non-interfering?
- **KU5 — Axis B.** The two-model INT4 acceptance collapse (0.490 → 0.036) is
  entirely undiagnosed. See §5 for the scoping recommendation.

## §2 — Wiring points verified against the live tree

`cipher_rt_phase4/` this session:

| Claim | Pod-state check | Verdict |
|---|---|---|
| GEMM `grid` is the device SM count | `cipher_rt_marlin_engine.cpp:800` — `int grid = g_sm_count;`; `g_sm_count` (`:677`) ← `rt_devattr(MultiProcessorCount)` (`:779`), fallback 132 (`:780`) | **confirmed** — `:800` is the one line STEP 2 changes |
| The shipped dispatch pins the GEMM to the primary ctx | `cipher_rt_marlin_engine_dispatch` (`:970`) holds a `PrimaryCtxGuard ctx_guard` (`:985`) across the `marlin_gemm_launch` call (`:986`) — comment `:978-980` names the partition deadlock | **confirmed** — the GEMM launch currently runs full-GPU primary ctx |
| Green-context machinery exists and verifies its partition | `cipher_rt_green_ctx.c` — splits 132 SMs into 16×8 (`cuDevSmResourceSplitByCount`, `:139`), `cuGreenCtxCreate` (`:188`), and **verifies** the partition took via `cuGreenCtxGetDevResource` → `g_green_sm_count` (`:209-227`) | **confirmed** — one green ctx per process, 8 SMs, verified |
| The green ctx's verified SM count is reachable | `cipher_rt_green_ctx.c` has the static `g_green_sm_count` (`:57`) but `cipher_rt_green_ctx.h` exports **no accessor** for it | **gap** — STEP 2 adds `cipher_rt_green_ctx_sm_count()` |
| App streams are already green-bound | CUPTI `cuStreamCreate` callback pushes the green ctx (`green_ctx.h` hook strategy; `cipher_rt_green_ctx_push`) | **confirmed** — in the partitioned path the app's launch streams already belong to the green ctx |

**No STEP 1 finding or design-memo assumption was contradicted.** STEP 2 may
proceed to plan.

## §3 — The load-bearing STEP 2 design question (KU2)

The CP 2.4 `PrimaryCtxGuard` exists for a *real* reason: the quant/repack path
(`quantize_fp16_to_int4_groupwise_gpu`) mixes driver-API kernel launches with
runtime-API `cudaMalloc`/`cudaMemcpy`, which is only correct when the **runtime
primary context** is current (`:813-826`). That pin must stay for quant.

But CP 2.4 made the guard cover the **whole dispatch**, GEMM launch included —
so the GEMM runs on the primary (full-GPU) context. That is exactly what STEP 2
must undo. The design:

- **Quant/repack** stays primary-pinned — unchanged, mandatory.
- **The GEMM launch** must run under the **green context** so its `grid` blocks
  are confined to the partition. The app's launch `stream` is already
  green-bound (§2); the fix is to **scope `PrimaryCtxGuard` to the quant phase
  only**, leaving the GEMM `launch_kernel` to execute on the green-bound stream.
- `grid` at `:800` becomes the green ctx's verified SM count when a green ctx
  is active, else `g_sm_count` (preserves the single-tenant / no-partition
  path unchanged).

**KU2 restated:** quant produces `marlin_B` device memory allocated under the
primary context; the GEMM kernel then reads it under the green context. Green
contexts derived from the same device share the device address space, so this
*should* be valid — but it is a known-unknown STEP 2 verifies before relying
on it (a cheap-allocated-primary / read-green smoke check is plan item 2).

## §4 — Falsification surface

STEP 2 must hit each, not merely observe "it ran":
1. **Co-residency deadlock (KU1).** `grid=8` in an 8-SM green ctx, watchdog
   timeout — the CP 2.4 pattern. A hang here = the bounded-rework assumption
   is wrong; report it, do not reframe.
2. **Cross-context memory validity (KU2).** GEMM output under the green ctx
   must be numerically identical to STEP 1's full-device `grid=8` result for
   the same shape — proves the primary-allocated weights are read correctly.
3. **SM spill (KU3).** Per-block `%smid` capture; the partition launch must
   touch **≤ 8 distinct SMs**. >8 = the green ctx is not confining.
4. **Partition disjointness (KU4).** Two concurrent partitions must have
   **disjoint** `%smid` sets — overlap = no spatial isolation, the moat fails.
5. **Token-level correctness.** Partition-aware Marlin in a real decode must
   produce byte-identical token IDs to full-GPU Marlin (gate A).

## §5 — Scoping recommendation: Axis A build vs Axis B diagnosis

The operator's STEP 2 scope lists 5 items. Items 1, 2, 4, 5 are **Axis A** — a
bounded *build* (grid wiring + green-ctx launch + multi-partition + correctness
gate). Item 3 is **Axis B** — an open-ended *root-cause diagnosis* of the
0.490→0.036 acceptance collapse, a different kind of work with a different
gate, and one STEP 1's scope memo §6 and the design memo §3 both already
positioned as "diagnosed in parallel under its own STEP."

**Recommendation:** run STEP 2 = **Axis A** (this memo's plan) and **STEP 2B =
Axis B diagnosis** as a parallel STEP with its own scope memo + report.
Rationale: "one atomic STEP" (campaign discipline) means one atomic *build*;
bundling a multi-week kernel build with an open-ended diagnosis into one report
muddies adjudication. **CP 5.3 closes only when both Axis A (STEP 2) and
Axis B (STEP 2B) land** — design memo §4's gate needs both; neither STEP closes
the CP alone. This is a recommendation — the operator may override and keep
Axis B inside STEP 2 at adjudication. The 7-item plan below is written for
Axis A; if Axis B stays folded in, items 6–7 expand to cover its report.

## §6 — STEP 2 plan (7 items) — Axis A

1. **Add two green-ctx accessors.** `cipher_rt_green_ctx_sm_count()` — returns
   `g_green_sm_count` (verified count) if a green ctx is initialised, else 0 —
   and `cipher_rt_green_ctx_group_id()` — returns `g_green_group_id`, needed by
   item 5 to confirm two test processes did not hash-collide into the same
   partition (`green_ctx.c:158-172`). Both in `cipher_rt_green_ctx.{c,h}`,
   pure accessors, no behaviour change.
2. **KU2 smoke check (gate before the real wiring).** A standalone check:
   allocate device memory under the primary ctx, launch a trivial kernel that
   reads it under the green ctx, verify the read. If it fails, KU2 is real and
   the design in §3 needs revision *before* touching the actuator — reported
   for re-adjudication, no further code.
3. **Wire `grid ← partition SM count`.** `cipher_rt_marlin_engine.cpp:800`:
   `grid = green_ctx_sm_count` when a green ctx is active, else `g_sm_count`.
   Scope `PrimaryCtxGuard` to the quant/repack phase only, so the GEMM
   `launch_kernel` runs on the green-bound stream (§3). Preserve the
   single-tenant path byte-for-byte (no green ctx → identical to today).
4. **`%smid` instrumentation.** A `%smid`-recording variant in a **separate
   copy** of the kernel source (the STEP 1 harness already carries one at
   `cp_5_3/cipher_rt_marlin_kernel_src.cpp`) — each block writes its `%smid` to
   a device buffer, for gates B and C. The **shipped** kernel src is not
   modified; this instrumented kernel is an experimental side-build only.
5. **Real green-context launch + multi-partition test.** Launch partition-aware
   Marlin inside an actual 8-SM green ctx (watchdog for KU1); then 2 concurrent
   8-SM partitions (2 processes) under simultaneous dispatch. **Precondition:
   confirm via `cipher_rt_green_ctx_group_id()` that the two processes selected
   different green groups before measuring — on a hash collision
   (`green_ctx.c:158-172`), respawn with different `(pid, tenant_handle)` until
   disjoint; gate C is measured only on a confirmed non-colliding pair.**
   Capture `%smid` sets, completion, numerical error vs the STEP 1 cuBLAS
   reference.
6. **Correctness gate at partition level** — run the §7 gate A/B/C, write
   `cp53_step2_*.runlog`.
7. **Write `CP_5_3_STEP_2_REPORT.md`** applying the §7 rule + **tarball** the
   STEP 2 evidence; surface for adjudication.

## §7 — Gate parameters (committed in advance)

Fixed **now**, before any number is seen:

- **(A-num) Numerical re-validation — the binding correctness gate.** The
  green-ctx GEMM output must meet the **same ε as STEP 1** (≤ 2× the
  `grid=132` Marlin-vs-cuBLAS error per shape) on the 4 STEP 1 shapes,
  measured against the **independent cuBLAS FP16 reference**. Non-negotiable
  per design memo §4. This is the real correctness oracle.
- **(A-tok) Token comparison — see the precedent caveat below.** Greedy
  decode, **128 generated tokens**, on **TinyLlama-1.1B** (MHA) and
  **Mistral-7B-v0.1** (GQA). Partition-aware Marlin (8-SM green ctx) vs
  full-GPU Marlin.
  **Caveat (must be adjudicated — §10 ask 3):** strict "0 divergent token IDs"
  is *not safe to pre-register as a hard fail.* STEP 1's own runlogs show the
  GEMM's max-abs-error varies with `grid` for the same shape (S2: 0.250 @
  grid=132 vs 0.0625 @ grid=8 — the legitimate FP16 split-K reduction-order
  signature STEP 1 cited as evidence of *correctness*). Greedy argmax can flip
  whenever top-1/top-2 logits sit within that ε — on a fully correct kernel.
  The "same pattern as CP 5.1/5.2" precedent does **not** carry: CP 5.1/5.2
  restored exact KV *bytes* with no FP recomputation; STEP 2 re-runs the GEMM
  under a different reduction order, so byte-identity is not free here.
  Pre-registered options (operator picks one at adjudication):
  - **(i) strict** — 0 divergent token IDs, hard fail. Risks failing a correct
    kernel.
  - **(ii) recommended** — A-num is binding; A-tok is reported as top-1
    agreement rate + divergence-first-index, with a pre-registered expectation
    of **≥ 99% top-1 agreement** over 128 tokens; a miss is investigated, not
    an automatic CP fail.
  - **(iii) tolerant** — A-tok purely observational; A-num alone gates
    correctness.
- **(B) Partition isolation / no spill.** Per-block `%smid` from a `grid=8`
  launch in an 8-SM green ctx must resolve to **≤ 8 distinct SM IDs**. >8 =
  FAIL (green ctx not confining).
- **(C) Multi-partition.** 2 concurrent 8-SM partitions. **Precondition:**
  measured only on a confirmed non-colliding green-group pair (§6 item 5).
  Then: (i) both complete, no deadlock; (ii) both meet A-num, and A-tok per
  the option adjudicated; (iii) the two partitions' `%smid` sets are
  **disjoint** (zero shared SM) — the hard proof of spatial isolation.
- **PASS rule:** STEP 2 (Axis A) PASSES iff A-num **and** B **and** C all pass,
  and A-tok meets whichever option (i/ii/iii) is adjudicated. Any deadlock
  (KU1), any spill (B), or any smid overlap (C) → FAIL; the report names the
  failure mode and whether it invalidates the 4-week bounded-rework verdict
  (escalating toward design memo §4's shape-2 hybrid-actuator fallback).
- **Partial handling:** single-partition passes but multi-partition fails →
  reported as a characterised partial (e.g. "isolated single-tenant, interferes
  at 2 tenants"), not a pass.

## §8 — What STEP 2 does NOT do

- Does **not** address Axis B (per §5 recommendation — STEP 2B).
- Does **not** test beyond 2 concurrent partitions — the 100-tenant scale-out
  is CP 5.5; STEP 2 proves the 2-partition primitive that CP 5.5 builds on.
- Does **not** re-tune Marlin's B≥8 performance envelope — STEP 2 is
  correctness + isolation; the small-slice performance question (design memo
  §6) is noted for a later measurement, not gated here.
- Does **not** touch the kmod or its ABI — STEP 2 is libcipher_rt userspace
  only. The green ctx is selected by `pid ^ tenant_handle` hash, not a kmod
  slot→SM map (`green_ctx.c:158-172`).

## §9 — Anchors

Held entering STEP 2: kmod 0.4.8 `e2f50452`, libcipher_rt `c2c5d313`,
libcipher_v2 `86618c30`, cipher_kv_bridge `8d6ffe3f`. STEP 2 **ships a fix
inside `libcipher_rt.so`** (the `grid` wiring + `PrimaryCtxGuard` scoping) —
per design memo §7 this is the first Phase 5 CP to move the libcipher_rt
anchor. The new libcipher_rt anchor is recorded in the STEP 2 report when the
fix is built; kmod / libcipher_v2 / kv_bridge anchors unchanged. The
`%smid`-instrumented kernel (item 4) is an experimental side-build, not an
anchor. Prior `libcipher_rt.so` preserved as a `.pre_cp5_3_step2` fallback
before the rebuild, per phase discipline.

## §10 — Adjudication ask

Three asks:
1. **Approve the §6 7-item STEP 2 plan** to wire partition-aware Marlin (Axis A)
   and gate it against §7.
2. **Confirm or override the §5 scoping recommendation** — Axis A as STEP 2,
   Axis B as a parallel STEP 2B; or keep Axis B folded into STEP 2.
3. **Pick the §7 A-tok option (i strict / ii recommended / iii tolerant).**
   The original scope said "identical tokens… same pattern as CP 5.1/5.2";
   STEP 1's data shows that strict identity may fail a *correct* kernel and the
   5.1/5.2 precedent does not carry (§7 A-tok caveat). Recommendation: **(ii)**.

No code is written until this memo is adjudicated.
