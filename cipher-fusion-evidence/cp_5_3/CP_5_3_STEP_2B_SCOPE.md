# CP 5.3 — STEP 2B (Axis B): two-model speculative-decode acceptance collapse — SCOPE MEMO

**Date:** 2026-05-18. **Type:** scope/design, no GPU. **Status:** scoped for
adjudication — no code until adjudicated (per STEP 2 §10 discipline).
**Substrate:** `a7ac8e97` (F1-fixed). Anchor unchanged by this memo.

---

## §0 — Correction to the adjudication brief (read first)

DECISION 2 instructed this memo to cover "INT4 + spec-decode composition on
**Mistral-7B** drops acceptance from 0.490 (**FP16 + spec**) to 0.036 (**INT4 +
spec**)." Two parts of that framing are factually wrong against the CP 2.4
evidence, and one hypothesis in the brief targets the wrong arm. Corrected here
because a scope memo built on the wrong premise would mis-scope the diagnosis.

| brief says | evidence says | source |
|---|---|---|
| **Mistral-7B** | **Llama** model-draft arm: target Llama-3.1-8B, draft Llama-3.2-1B-Instruct. Mistral's CP 2.4 spec arm used an *n-gram* draft and ran clean (50 runs, accept cond. 1.0). | `cp_2_4/CUDNN_ATTN_MARLIN_HANG.md` §2 (bisection R1/T3), `cp_2_4/CP_2_4_REPORT.md` §3.4 |
| axis = **FP16 → INT4** | axis = **substrate OFF → substrate ON**. R1 (no v2 substrate, plain FP16) = 0.490; T3 (full v2 substrate) = 0.036. INT4/Marlin is the *leading hypothesis for the cause*, not the measured contrast. | same |
| hypothesis: "INT4 noise breaking **n-gram** pattern matching" | wrong arm — the n-gram Llama arm **worked** (bisection T2: PASS, accept 1.000). The collapse is the **model-draft** arm only. | `CUDNN_ATTN_MARLIN_HANG.md` §2 T2 |

Provenance of the "Mistral" error: it entered in `cp_5_6/FUTURE_SCOPE/F_phase5_resumption.md`
("on Mistral-7B INT4") — authored during CP 5.6 closeout — and propagated into
DECISION 2. The CP 5.3 design memo §1 Axis B definition is correct ("two-model
acceptance collapse", no model named). `FUTURE_SCOPE/F` should be corrected to
"Llama model-draft arm." This memo proceeds on the corrected premise.

## §1 — Observation (corrected)

CP 2.4 sub-task (iii), model-draft Llama arm, 2026-05-16. Target Llama-3.1-8B,
draft Llama-3.2-1B-Instruct, greedy, single prompt, 64 tokens. Bisection
(`CUDNN_ATTN_MARLIN_HANG.md` §2):

| run | config | accept_rate | note |
|---|---|---|---|
| **R1** | model-draft spec, **no v2 substrate** (plain FP16) | **0.490** | byte-identical to stock greedy; 3.2 tok/round; 20 rounds |
| **T3** | model-draft spec, **full v2 substrate** (Marlin INT4 + attn dispatch + green-ctx), draft on default stream | **0.036** | no hang; 51 rounds; 53.9 s — *slower* than stock (T1 36.2 s) → **negative lift** |

Same draft, prompt, token budget. The model-draft engine is correct in
isolation (R1). Under the substrate the draft's proposals almost never match
the target → spec decode yields negative lift. CP 2.4 root-caused the *hang*
(3a, fixed: `CIPHER_SPEC_DRAFT_STREAM=0`); the *acceptance collapse* (3b) was
deferred to Phase 5 = this STEP. The n-gram Llama arm (T2) and the n-gram
Mistral arm (50 runs) are unaffected — STEP 2B is the **model-draft** arm only.

## §2 — Hypotheses

The substrate-ON path (T3) bundles several components changed since R1. The
collapse is attributable to one or more of:

- **H1 — F1 cross-stream race / degenerate target logits (LEADING, post-CP-5.6).**
  The 0.490→0.036 was measured 2026-05-16. F1 — shipped full-GPU Marlin
  producing degenerate decode output via a GEMM-pinned-to-primary-context
  cross-stream race — was discovered 2026-05-18 (CP 5.3 STEP 2 §5) and fixed in
  `a7ac8e97`. T3's substrate had F1. F1-degenerate target logits (NaN /
  repetition / `<unk>` flood) would cause the draft's FP16-trained proposals to
  be rejected wholesale → acceptance ≈ 0. **0.036 is exactly the F1 signature.**
  STEP 2 §5 itself flags that CP 2.4's gate measured tok/W and tok/s — numbers a
  degenerate decode still produces — with no token-correctness check. If H1 is
  the whole story, the F1 fix already resolved Axis B.
- **H2 — Marlin weight-cache key collision between the 1B draft and 8B target.**
  Both models route INT4 GEMMs through one Marlin actuator. If the weight-cache
  key does not disambiguate the two models, one model's quantized/repacked
  weights could be served to the other → garbage logits. (Original CP 2.4
  leading hypothesis.)
- **H3 — genuine two-model INT4 quantization divergence.** No collision, but
  INT4 quantization perturbs the 8B target's argmax enough that the 1B draft
  (trained against the FP16 target) systematically mispredicts. A real
  accuracy limit of INT4 + speculative decode, not a bug.
- **H4 — residual stream/ordering effect** distinct from F1 (e.g. draft↔target
  KV or context-handoff ordering). Lower prior — the handoff is fully
  synchronous (CP 2.4 §3a) — but not excluded until H1 is isolated.

H2/H3/H4 only need to be chased if Step 1 excludes H1.

## §3 — Step 1: three-arm discriminator (the load-bearing measurement)

Step 1 is **not** a bare "reproduce." It is a three-arm contrast that resolves
H1 before any fix work is scoped. All arms: Llama-3.1-8B target +
Llama-3.2-1B-Instruct draft, greedy, the CP 2.4 prompt + 64-token budget
(reproduce R1/T3 conditions exactly), `CIPHER_SPEC_DRAFT_STREAM=0`.

| arm | substrate | binary | expected | role |
|---|---|---|---|---|
| **(a)** | OFF (plain FP16) | — | ≈ 0.490 | sanity floor — reproduces R1 |
| **(b)** | ON, **pre-F1-fix** | preserved pre-F1 `libcipher_rt.so` | ≈ 0.036 | confirms the original collapse on today's stack |
| **(c)** | ON, **F1-fixed** | `a7ac8e97` | **the diagnostic question** | — |

Arm (b) binary: use the immediate pre-F1-fix preserved artifact
(`libcipher_rt.so.pre_cp56`, dc804eb3 per [[cipher-f1-fullgpu-marlin-broken]]).
**Step 1 first verifies the `.pre_*` chain** — confirm which preserved `.so`
genuinely predates the F1 fix and reproduces ≈0.036 — rather than hard-asserting
a filename.

**Decision tree after Step 1:**

| outcome | reading | next |
|---|---|---|
| (c) ≈ (a) ≈ 0.49 | **F1 *was* the collapse.** The F1 fix already resolved Axis B. | STEP 2B closes by verification: Axis B resolved via the existing F1 fix; no new fix. CP 5.3 closes pending Axis-A adjudication. ≈ ½ day total. |
| (c) ≈ (b) ≈ 0.036 | F1 excluded — a **residual two-model INT4 interaction** is real. | Proceed to H2/H3 diagnosis (§3.1). The ~3–5 day path. |
| (c) intermediate | partial — F1 was *part* of it; a residual remains. | Quantify the F1-attributable share; proceed to H2/H3 for the remainder. |

### §3.1 — H2/H3 diagnosis (only if Step 1 ≠ (c)≈(a))

- **H2 test:** instrument the Marlin weight-cache — log the cache key per GEMM
  for a run that interleaves 1B-draft and 8B-target invocations. A collision
  shows as one key serving two distinct weight tensors. If found: the fix is a
  model-discriminating key (cheap, fixable).
- **H3 test:** with H2 excluded, compare the INT4-target argmax sequence
  against the FP16-target argmax on a teacher-forced gold prefix (§4). If the
  INT4 target is *numerically faithful* (KL≈0) yet the draft still mispredicts,
  the collapse is genuine INT4×spec-decode accuracy loss → architectural, scope
  spec-decode out of the INT4 regime with evidence (CP 5.3 §4 fallback).

## §4 — Methodology

- **Primary metric: draft acceptance rate** (`accepted / proposed`) and
  **tok/round** (`gen_tokens / rounds`) — the CP 2.4 §iii metrics, directly
  comparable to R1=0.490 / T3=0.036. Plus wall time for the lift sign.
- **Teacher-forced target-faithfulness gate (the F1 discriminator).** Per arm,
  teacher-force the *target* model on a fixed gold token sequence and measure
  per-step logit KL vs an FP16 gold reference. KL ≤ 0.1 = target numerically
  faithful. This is cascade-free (per [[cipher-cp56-closed]] — free-running KL
  is mechanically broken). Interpretation: arm (c) with KL≈0 but acceptance
  still ≈0.036 → target is fine, collapse is a genuine draft-tracking issue
  (H3). Arm (c) with large KL → target still degenerate (H1 not fully fixed,
  or H2).
- **Power:** logged (arm-level mean, sentinel-windowed) but **secondary** —
  STEP 2B is a correctness/acceptance diagnosis, not a tok/W measurement. The
  CP 2.4 gate measuring only tok/W is what let the degenerate decode through;
  STEP 2B does not repeat that.
- **Matched-pair isolation:** one fresh process per (arm, prompt); no shared KV
  / allocator / weight-cache state across arms.
- **Substrate-attributable:** the (c)−(b) delta is the F1-fix contribution;
  the (a)−(c) gap (if any) is the residual two-model interaction.

## §5 — Decision framework: fixable vs architectural

Maps to the CP 5.3 design memo §4 Axis B gate ("draft acceptance under the
two-model substrate path restored to within a declared tolerance of the
substrate-off 0.490 baseline").

- **Resolved-fixable** — acceptance on `a7ac8e97` is at/near 0.490 (whether via
  the F1 fix alone, or F1 + an H2 key fix). Axis B PASSES; CP 5.3 closes
  (with Axis A) once both are adjudicated.
- **Architectural constraint** — acceptance stays low and is traced to genuine
  INT4×model-draft accuracy loss (H3), not a bug. Then per CP 5.3 §4 fallback
  and FUTURE_SCOPE/F: **explicitly scope speculative decode out of the INT4
  substrate regime, with evidence** — a documented descope, not a silent one.
  The n-gram arms (which work) remain the spec-decode story for that regime.
- **Declared tolerance** to be fixed at adjudication of *this* memo —
  proposed: acceptance ≥ 0.45 (≥ ~92% of the 0.490 baseline) counts as
  resolved; < 0.20 with H3 confirmed counts as architectural; the band between
  is reported with the F1-attributable share quantified.

## §6 — Time estimate (revised)

Step 1's three-arm discriminator collapses the open-ended estimate into a
branch:

| path | scope | est. |
|---|---|---|
| Step 1 | three-arm discriminator + `.pre_*` chain verification | ~0.5 day |
| **branch (c)≈(a)** | verification writeup; Axis B closes via F1 fix | +0.5 day → **~1 day total** |
| **branch (c)≈(b)** | H2/H3 diagnosis + fix-or-descope + verify + writeup | +2.5–4 days → **~3–5 days total** |

This sharpens DECISION 2's "1 day reproduce + 1–2 days fix + 1 day verify": the
fix phase is conditional on Step 1, and the most-likely outcome (H1) is
materially cheaper.

## §7 — Scope boundaries / anchors

- STEP 2B is the **model-draft Llama arm** only. n-gram arms unaffected.
- Does **not** touch Axis A (STEP 2, complete) or the kmod.
- Substrate: `a7ac8e97`. If an H2 fix rebuilds libcipher_rt, the anchor rotates
  and `a7ac8e97` is preserved as `.pre_cp53_step2b` per phase discipline.
- Models required: Llama-3.1-8B + Llama-3.2-1B-Instruct (CP 2.4 used both —
  Step 1 verifies they are present before measuring).
- CP 5.3 closes only when Axis A (STEP 2, adjudicated PASS) **and** Axis B
  (this STEP) both land.

## §8 — Adjudication ask

1. **Accept the §0 corrections** — Llama model-draft arm (not Mistral);
   substrate-OFF→ON axis (not FP16→INT4); and correct `FUTURE_SCOPE/F`.
2. **Approve the §3 three-arm Step 1 discriminator** as the first STEP 2B
   measurement, with the §3 decision tree governing what follows.
3. **Fix the §5 declared tolerance** (proposed: ≥0.45 resolved / <0.20+H3
   architectural) — or amend.
4. Confirm the §6 conditional time estimate.

No code until this memo is adjudicated.
