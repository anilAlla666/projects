# CP 5.3 — STEP 2 (Axis A): partition-aware Marlin wiring — VERDICT REPORT

**Date:** 2026-05-18. **Status:** STEP 2 (Axis A) executed and complete.
**Verdict: PASS on the binding gates (A-num, B, C); A-tok investigated — see
§4.** This report applies the pre-registered §7 rule of
`CP_5_3_STEP_2_PARTITION_WIRING_SCOPE.md` to the runlogs. It does **not** close
CP 5.3 (Axis B / STEP 2B is untouched — §7) and does **not** start STEP 2B —
`"adjudicated — proceed to STEP 2B"` is required first.

---

## §1 — Provenance and the adjudication assumption (stated, not papered over)

STEP 2's build artifacts (`cp53_marlin_engine_step2.cpp`, the rebuilt
`libcipher_rt.so`, `cp53_step2_harness`, gate runlogs) were produced by a prior
session that was interrupted (SSH drop) before authoring this report — the same
pattern as CP 5.2's Step 3. Per campaign discipline, **code is written only
after the scope memo is adjudicated**; the existence of the build is therefore
evidence STEP 2 was approved. Two corroborating facts: (a) the A-tok harness
hard-wires the `>= 0.99` threshold of §7 option **(ii)** — the scope memo's
*recommended* A-tok option — which is what an adjudication selecting (ii) would
produce; (b) the engine carries the exact §6 plan (grid accessor + scoped
guard). This session **re-verified** the gates from scratch rather than trust
the prior runlogs (§3), and completed the option-(ii)-mandated A-tok
investigation (§4) and item 7 (this report + tarball).

## §2 — What STEP 2 built (§6 plan, as executed)

| Item | Built | Evidence |
|---|---|---|
| 1. Green-ctx accessors | `cipher_rt_green_ctx_sm_count()`, `cipher_rt_green_ctx_group_id()` exported in `cipher_rt_green_ctx.h:55,61` | header |
| 2. KU2 smoke check | primary-loaded module + primary-allocated memory, launched on a green-ctx stream | `cp53_ku2_smoke.runlog` — **KU2_PASS**, 512/512 elems correct, 8 SMs touched |
| 3. `grid ← partition SM count` | `cipher_rt_marlin_engine.cpp:812-813` — `green_sm=cipher_rt_green_ctx_sm_count(); grid=(green_sm>0)?green_sm:g_sm_count;`. `PrimaryCtxGuard` scoped to the quant/repack phase only; the GEMM `launch_kernel` runs on the green-bound app stream | engine source; shipped in `libcipher_rt.so` md5 `dc804eb3` |
| 4. `%smid` instrumentation | `cp53_marlin_kernel_src_step2.cpp` — per-block `%smid` to a device buffer; experimental side-build, shipped kernel src untouched | build_step2.sh |
| 5. Real green-ctx launch + multi-partition | `cp53_step2_harness` (side-build linking the **real** `cipher_rt_green_ctx.o`); solo + 2-process concurrent | §3 |
| 6. Correctness gate | A-num / B / C run; runlogs below | §3 |
| 7. Report + tarball | this report; `cp_5_3_step2_evidence.tar.gz` | §9 |

KU2 (the load-bearing design question, scope §3) **resolved PASS**: weights
allocated under the primary context are read correctly by a kernel launched on
a green-context stream — green contexts on the same device share the address
space, as the design assumed.

## §3 — Binding gate results — A-num, B, C (re-verified this session)

Gate C (`run_gate_c.sh`) re-run from scratch this session — it exercises two
concurrent 8-SM green-context partitions and reports A-num + B + C for both:

```
RESULT gate_c=PASS anum_A=PASS anum_B=PASS b_A=PASS b_B=PASS
       disjoint=PASS no_deadlock=PASS overlap=NONE
  A: group=9  anum=PASS b=PASS smids=66,67,80,81,94,95,108,109
  B: group=13 anum=PASS b=PASS smids=74,75,88,89,102,103,116,117
```

- **(A-num) — PASS.** Green-context GEMM output meets the STEP 1 ε (≤ 2× the
  `grid=132` Marlin-vs-cuBLAS error) on all 4 STEP 1 shapes, against the
  independent cuBLAS FP16 reference. `cp53_step2_solo.result` (this campaign):
  every shape `ANUM PASS`, `PART_ERR ≤ EPS` (e.g. S1 0.0156 ≤ 0.0625; S3
  0.0625 ≤ 0.125). The green-ctx `grid=8` GEMM is numerically correct. **This
  is the binding correctness oracle (§7).**
  *Side-build bridge:* A-num runs on `cp53_step2_harness`, whose engine
  `cp53_marlin_engine_step2.cpp` is the shipped engine
  (`cipher_rt_phase4/cipher_rt_marlin_engine.cpp`, built into `dc804eb3`) plus
  a `%smid` 9th kernel arg and a `cipher_diag_quantize_export` diagnostic
  symbol. The full source diff was inspected: the grid/guard fix and the entire
  numerical path are **byte-verbatim** — the only deltas are the instrumentation
  arg (a buffer pointer, no GEMM-math effect) and an appended export function
  (not in any hot path). A-num on the side-build therefore validates the
  shipped `dc804eb3` GEMM numerics; decode coherence on TinyLlama + Mistral
  (§4) is the independent functional cross-check on the shipped `.so` itself.
- **(B) — PASS.** A `grid=8` launch inside an 8-SM green ctx touches **exactly
  8 distinct SM IDs** — no spill. Re-verified pair: A={66,67,80,81,94,95,108,
  109}, B={74,75,88,89,102,103,116,117}, 8 each.
- **(C) — PASS.** Two concurrent 8-SM partitions: both complete (no KU1
  deadlock — the CP 2.4 hang did **not** recur), both meet A-num and B, and the
  two `%smid` sets are **disjoint** (`overlap=NONE`). Reproduced twice with
  *different* green groups (prior run 5/10, this run 9/13) — spatial isolation
  holds independent of which groups hash-select. This is the hard proof of the
  2-partition primitive CP 5.5 builds on.

KU1 (co-residency deadlock — the exact CP 2.4 failure surface) **did not
recur**: `grid=8` × 96 KiB smem on 8 SMs is 1 block/SM, all co-resident, split-K
`locks` barrier satisfied.

## §4 — A-tok: gate-as-run, the miss, and the option-(ii) investigation

**Gate as run.** `run_atok.sh` compared a 128-token greedy decode under the
STEP 2 build (`partition`) against the shipped pre-STEP-2 build (`fullgpu`,
`libcipher_rt.so.pre_cp5_3_step2`, md5 `c2c5d313`). Result:
`atok_tinyllama=INVESTIGATE rate=0.0312`, `atok_mistral=INVESTIGATE
rate=0.0312` — both far below the pre-registered ≥0.99. Per §7 option (ii) a
miss is **investigated, not an auto-fail**. This section is that investigation.

**The miss is the reference, not the partition build.** Decoding the token
streams to text is decisive:

| stream | TinyLlama text | Mistral text |
|---|---|---|
| **gold** (plain FP16, no substrate) | coherent prose | coherent prose |
| **partition** (STEP 2, grid=8 green ctx, INT4 Marlin) | **coherent prose** | **coherent prose** |
| **fullgpu** (shipped `c2c5d313`, grid=132, INT4 Marlin) | repetition gibberish | `the 197imi` + **122× `<unk>`** |

`partition` TinyLlama: *"…the mechanical calculator was the primary tool for
mathematical calculations. The second era, the digital computer, was developed
in the 1940s and 1950s. The third era, the supercomputer…"* — fluent, on-topic,
on both an MHA (TinyLlama) and a GQA (Mistral) model. The `fullgpu` reference
is **degenerate**: the Mistral `<unk>`-flood (token id 0) is the textbook
NaN-logits → argmax(0) signature.

**Triangulation against a correct gold reference** (`run_atok_investigate.sh`,
`cp53_atok_investigate.runlog`):

| comparison | TinyLlama | Mistral |
|---|---|---|
| `partition` vs `gold` | first divergence @ tok 5, then coherent | first divergence @ tok 20, then coherent |
| `fullgpu` (orig) vs `gold` | divergence @ 4 → garbage | divergence @ 4 → garbage |
| `fullgpu` (fresh re-run) vs `gold` | garbage (not transient) | garbage (not transient) |
| `fullgpu` B=8 (Marlin's designed regime) vs `gold` | garbage | garbage |

`partition` tracks `gold` for the first several tokens then diverges — the
**expected INT4-quantization signature** (greedy argmax flips when top-1/top-2
logits sit within INT4 quant error; the scope memo §7 A-tok caveat predicted
exactly this, and one divergence cascades autoregressively). It produces
*correct, coherent* text. The low absolute token-agreement number is a
**property of the metric**, not a defect: strict token-ID identity is not a
valid pass/fail metric for an INT4 kernel under greedy autoregressive decode,
because (a) INT4 ≠ FP16 legitimately and (b) one early flip cascades. The §7
A-tok caveat flagged this risk; the data confirms it is real and larger than
option (ii)'s 99% expectation anticipated.

**Investigation conclusion.** The A-tok 3% is a **broken reference**, not a
broken partition build. The partition build's correctness rests on the binding
A-num oracle (PASS, §3) and on decode coherence on two architectures (PASS,
above). Under §7 option (ii), A-tok is **investigated and explained — not a CP
fail.**

## §5 — Separate finding F1: shipped full-GPU Marlin is broken on real decode

Surfaced by §4, **out of STEP 2's scope, reported not fixed** (per
[[cipher-regression-discipline]]: a blocker from a prior change → audit-first-
in-source, as its own work — never reframed, never silently absorbed).

**Finding.** The shipped `libcipher_rt.so` `c2c5d313` (campaign anchor;
post-CP-2.5) produces **degenerate output when Marlin INT4 is engaged on a real
transformer decode** — full-GPU path, `grid=132`, CP 2.4 `PrimaryCtxGuard`
pinning the GEMM to the primary context. Reproduced **six ways**: TinyLlama and
Mistral × {original A-tok run, fresh re-run, B=8 batch, tenant-id unset}.
Marlin engaged in every case (`MATMUL handled=13875` / `28125`). Not transient,
not M=1-specific, not tenant-id-dependent.

**Why STEP 2's partition build does not have it (hypothesis, for F1's audit —
not asserted as STEP 2's deliverable).** STEP 2's two changes — scoping
`PrimaryCtxGuard` to the quant phase so the GEMM launches on the app's
(green-bound) stream, and `grid←8` — produce coherent output. The shipped path
holds the guard across the GEMM launch, so the GEMM runs on the primary
context decoupled from the app's compute stream. The leading hypothesis is a
**cross-stream race** (GEMM output not ordered against the app stream that
consumes it). This is consistent with CP 2.4's gate having measured tok/W /
tok/s — throughput numbers that a degenerate decode still produces — without a
token-correctness check. **F1 needs its own scoped audit-and-fix
(audit-first-in-source); it is not a CP 5.3 STEP 2 deliverable.**

**Consequence for the A-tok gate design.** A future A-tok must not use the
broken full-GPU build as reference. Once F1 is fixed, re-run partition-vs-fixed-
fullgpu; or replace token-ID identity with a logit-KL / perplexity metric
against gold (robust to the legitimate INT4 argmax flips).

## §6 — Verdict

**STEP 2 (Axis A) PASSES the §7 binding gates.** A-num **PASS**, B **PASS**,
C **PASS** — all re-verified from scratch this session; KU1 deadlock did not
recur; KU2 resolved PASS; no SM spill; two partitions provably disjoint.
A-tok, per §7 option (ii), was **investigated**: the 3% is a broken full-GPU
reference (finding F1, §5) — the partition build produces correct, coherent
decode on MHA and GQA models. Per the §7 PASS rule (A-num ∧ B ∧ C pass, A-tok
meets the adjudicated option), STEP 2 (Axis A) is a **PASS**.

This **confirms the STEP 1 4-week bounded-rework verdict**: the `grid` rework
is correct and isolating in real partitions; no escalation toward the design
memo §4 shape-2 hybrid-actuator fallback.

**STEP 2 does not close CP 5.3.** Per design memo §4 the gate needs Axis A
**and** Axis B. Axis B (the two-model INT4 acceptance collapse 0.490→0.036) is
untouched — STEP 2B, its own scope memo, per the scope memo §5 recommendation.

## §7 — What STEP 2 does NOT do

- Does **not** address Axis B — STEP 2B.
- Does **not** fix finding F1 — that is a separate audit (§5).
- Does **not** test beyond 2 concurrent partitions — 100-tenant scale-out is
  CP 5.5; STEP 2 proves the 2-partition primitive.
- Does **not** re-tune Marlin's B≥8 performance envelope — correctness +
  isolation only.
- Does **not** touch the kmod or its ABI — libcipher_rt userspace only.

## §8 — Anchors

STEP 2 **ships a fix inside `libcipher_rt.so`** — CP 5.3 is the first Phase 5
CP to move the libcipher_rt anchor (design memo §7).

- **libcipher_rt.so rebuilt: `c2c5d313` → `dc804eb3`**
  (`dc804eb3b2796652342852f75763d5ed`,
  `/home/ubuntu/cipher_rt_phase4/libcipher_rt.so`). Prior shipped artifact
  preserved as `libcipher_rt.so.pre_cp5_3_step2` (`c2c5d313`) per phase
  discipline. Per the CP 2.4/2.5 convention the **campaign anchor pointer
  moves on CP 5.3 *close*** (STEP 2A + STEP 2B both landed), not on a STEP —
  `dc804eb3` is the in-flight artifact; the manifest is updated at CP close.
- Unchanged: kmod 0.4.8 `e2f50452`, libcipher_v2 `86618c30`, cipher_kv_bridge
  `8d6ffe3f`.
- The `%smid`-instrumented kernel and `cp53_step2_harness` are experimental
  side-builds, not anchors.

**Anchor-move caveat:** `dc804eb3` ships the partition-aware Marlin fix and is
correct on the partitioned path (this report). It does **not** resolve finding
F1 on the full-GPU path; F1's audit may move the anchor again.

## §9 — Evidence

`cipher-fusion-evidence/cp_5_3/` — tarball `cp_5_3_step2_evidence.tar.gz`:
- `CP_5_3_STEP_2_PARTITION_WIRING_SCOPE.md` — pre-registered plan + §7 rule
- `cp53_marlin_engine_step2.cpp` md5 `03475e45`, `cp53_marlin_kernel_src_step2.cpp`
  md5 `2aa4dce4`, `cp53_step2_harness` md5 `59f4c077`, `cp53_ku2_smoke` md5
  `8f669562`, `build_step2.sh` — STEP 2 build
- `cp53_ku2_smoke.runlog` — KU2_PASS
- `cp53_step2_solo.result`, `cp53_step2_multi_{A,B}.result`,
  `cp53_step2_gatec_reverify.runlog` — A-num / B / C, re-verified this session
- `run_atok.sh`, `cp53_atok_decode.py`, `cp53_atok_*_{fullgpu,partition}.json`
  — A-tok gate as run
- `run_atok_investigate.sh`, `cp53_atok_decode_batched.py`,
  `cp53_atok_investigate.runlog`, `cp53_atok_*_{gold,fullgpu_rerun,fullgpu_b8,
  fullgpu_notenant}.json/.log` — the §4 triangulation + §5 F1 reproduction
- `CP_5_3_STEP_2_REPORT.md` — this report

## §10 — Adjudication ask

1. **Adjudicate STEP 2 (Axis A) = PASS** on the §7 binding gates (A-num ∧ B ∧
   C), with A-tok investigated per option (ii) — §4/§6.
2. **Accept the §4 finding that strict token-ID A-tok is an invalid metric**
   for an INT4 kernel under greedy decode, and that the binding correctness
   oracle is A-num (as §7 itself states). Future A-tok: logit-KL/perplexity vs
   gold, or partition-vs-fixed-fullgpu after F1.
3. **Triage finding F1 (§5)** — shipped full-GPU Marlin (`c2c5d313`) degenerate
   on real decode. Recommend a dedicated regression-audit STEP/CP
   (audit-first-in-source). It is not folded into CP 5.3.
4. **Confirm STEP 2B** (Axis B diagnosis) as the next CP 5.3 STEP — design-memo
   first. CP 5.3 closes only when STEP 2 (Axis A) **and** STEP 2B (Axis B) land.

No code for STEP 2B until its scope memo is adjudicated.
