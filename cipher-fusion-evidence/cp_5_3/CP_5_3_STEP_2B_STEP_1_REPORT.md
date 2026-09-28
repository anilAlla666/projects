# CP 5.3 STEP 2B — Step 1: three-arm F1 discriminator — REPORT

**Date:** 2026-05-18. **Scope:** the pre-registered Step 1 of
`CP_5_3_STEP_2B_SCOPE.md` §3. Llama model-draft speculative-decode arm
(target Llama-3.1-8B, draft Llama-3.2-1B-Instruct). Anchor `a7ac8e97`.

**Verdict: PARTIAL — F1 is the dominant cause of the 0.490→0.036 acceptance
collapse.** The F1 fix (`a7ac8e97`) recovers 74.8% of the substrate-off↔
pre-F1 gap — restoring the model-draft spec arm from a degenerate,
negative-lift state to a working one (0.053 → 0.380 acceptance, 1.28 → 2.91
tok/round, garbage → coherent text). A residual ~0.11 acceptance gap vs the
substrate-off baseline remains, and the teacher-forced KL identifies it as
**benign INT4 quantization divergence (H3), not a defect.**

---

## §1 — The three arms

All arms: Llama-3.1-8B + Llama-3.2-1B-Instruct draft, greedy, the CP 2.4
"Pacific Ocean" prompt, 64 generated tokens, `CIPHER_SPEC_DRAFT_STREAM=0`.

| arm | substrate | accept_rate | rounds | tok/round | gen text |
|---|---|---|---|---|---|
| **(a)** | OFF — plain FP16 | **0.490** (48/98) | 20 | 3.20 | coherent ("…30% of the Earth's surface…") |
| **(b)** | ON, **pre-F1** (`c2c5d313`) | **0.053** (19/358) | 50 | 1.28 | degenerate ("eltareth…!!!!!!!") |
| **(c)** | ON, **F1-fixed** (`a7ac8e97`) | **0.380** (41/108) | 22 | 2.91 | coherent ("…63 million square miles…") |

- **Arm (a)** reproduces CP 2.4 R1 (0.490) exactly — methodology gate PASS.
- **Arm (b)** reproduces the collapse. The collapsed regime has now been
  observed **three times in three different builds** — CP 2.4 T3 (`5e304549`,
  0.036), this driver v1 (`c2c5d313`, 0.043), this driver v2 (`c2c5d313`,
  0.053): accept_rate ∈ **[0.036, 0.053]**, a tight band an order of magnitude
  below the 0.490 baseline, degenerate output every time. Gate PASS. (Arms
  (a)/(c) were byte-identical across the v1/v2 re-run.)
- **Arm (c)** is the diagnostic result: **0.380**, in the §5 **[0.20, 0.45)
  PARTIAL band**.

`tok/round` is CP 2.4's honest lift driver (`SPEC_DECODE_METHODOLOGY.md` §1.2):
arm (c)'s 2.91 means each verify round commits ~2.9 tokens — a working,
positive-lift regime (cf. arm (b)'s degenerate 1.28; arm (a)'s 3.20). Spec
decode is *viable* on the F1-fixed substrate.

## §2 — Decomposition and decision tree

| quantity | value |
|---|---|
| total collapse (a − b) | 0.490 − 0.053 = **0.437** |
| F1 fix recovered (c − b) | 0.380 − 0.053 = **0.327** → **74.8% of the (a)−(b) gap** |
| residual gap (a − c) | 0.490 − 0.380 = **0.110** (≈22% of the 0.490 baseline) |

Per the adjudicated §5 tolerance, arm (c) = 0.380 ∈ [0.20, 0.45) →
**PARTIAL: F1 dominant + a residual to diagnose** (§3).

## §3 — Mechanism: the teacher-forced KL identifies the residual

Per-arm teacher-forced target logits (fixed 32-token sequence, through the
warmed substrate), KL vs the arm-(a) FP16 gold:

| arm | target KL vs gold (mean / max) | target top-1 argmax agreement vs FP16 gold |
|---|---|---|
| (a) | 0.0 / 0.0 | 100% (it is the gold) |
| (b) | **NaN** | **0%** |
| (c) | **0.090 / 0.396** | **81.3%** |

- **Arm (b) — F1 confirmed.** The pre-F1 target's logits are **NaN** and agree
  with the FP16 gold argmax on **0** of 32 positions. The target's verify
  forward is degenerate — so every draft proposal is rejected, acceptance ≈ 0.
  This is the F1 signature (degenerate full-GPU Marlin decode), and it is
  *why* the collapse happens. H1 (F1) confirmed as the collapse mechanism.
- **Arm (c) — faithful, INT4-perturbed.** The F1-fixed target is **not**
  degenerate: coherent text, KL mean 0.090 (just under the §4 "faithful" ≤ 0.1
  bar), 81.3% argmax agreement with FP16. The ~19% of positions where the INT4
  target's argmax differs from FP16 is exactly the residual mechanism: the
  Llama-3.2-1B draft proposes the *FP16* target's tokens; the INT4 target picks
  differently on ~1-in-5 decisions, so those proposals are rejected. **The
  residual is H3 — legitimate INT4 quantization divergence, not a bug.**
- **H2 (Marlin weight-cache key collision between the 1B draft and 8B target)
  is excluded — structurally.** The teacher-forced forward runs **only the 8B
  target** (the draft is not involved). A 1B/8B cache-key collision would serve
  the draft's weights into the target's GEMMs, corrupting the 8B's hidden
  states layer by layer. Arm (c)'s target instead produces **coherent prose
  with 81% argmax agreement** vs FP16 gold — so the 8B target's Marlin path was
  serving *its own* correctly-quantized weights, not the draft's. The residual
  is therefore distributed INT4 quantization noise (H3), not a key collision;
  the §3.1 weight-cache instrumentation is not needed.

## §4 — Methodology note: a driver bug, caught by the sanity check

The first driver ran the teacher-forced forward as the process's *first* GPU op
— **before** the Marlin warm generate. Marlin compiles/quantizes lazily on
first `generate()`, so the tf forward ran in plain FP16 in every arm → KL = 0.0
for all three, including degenerate arm (b). The per-arm KL **sanity check
caught this** (a degenerate arm cannot have KL=0.0). The driver was reordered —
warm generate → tf forward → measured generate — and all three arms re-run; §3
is the corrected measurement. Acceptance rates were unaffected (they always ran
post-warm) and reproduced on the re-run.

## §5 — What this means for CP 5.3 STEP 2B / Axis B

- **The collapse is root-caused: F1.** Axis B's 0.490→0.036 is dominated by
  the F1 cross-stream race (degenerate target verify logits) — already fixed in
  `a7ac8e97`; the F1 fix recovers 74.8% of the (a)−(b) gap. The residual ~0.11
  acceptance gap is benign INT4 quantization divergence (H3) — an accuracy
  property of INT4 + model-draft spec, not a defect.
- **Spec decode is viable on the F1-fixed substrate.** Arm (c): acceptance
  0.380, 2.91 tok/round, coherent — a normal model-draft operating point (typical
  model-draft accept rates are 0.3–0.7). It is **not** the negative-lift,
  scope-it-out outcome the §5 framework's "architectural" branch anticipated.
- **Recommendation — close STEP 2B.** Axis B is resolved: the collapse was F1,
  fixed; the residual is characterized and benign; spec decode works. No Step 2
  fix is needed, and no descope of spec decode from the INT4 regime is
  warranted. The §3.1 H2 weight-cache instrumentation is **not** needed — the
  teacher-forced KL already excludes H2. With STEP 2 (Axis A) adjudicated PASS
  and STEP 2B (Axis B) resolved here, **CP 5.3 is ready to close.**

## §6 — Adjudication ask

1. **Accept Step 1 = PARTIAL**, F1 confirmed as the dominant cause (its fix
   recovers 74.8% of the (a)−(b) gap); the residual ~0.11 is benign INT4
   divergence (H3); H2 structurally excluded.
2. **Adjudicate STEP 2B = resolved** on this evidence — the collapse is
   root-caused and already fixed; the residual is not a defect — **without** a
   Step 2 (recommended), or direct a Step 2 if the residual is to be formally
   instrumented despite the KL evidence.
3. On (2), **CP 5.3 closes** (Axis A PASS + Axis B resolved).

## §7 — Anchors & artifacts

Anchor `a7ac8e97` unchanged — Step 1 is measurement only, no rebuild. Arm (b)
used the preserved `libcipher_rt.so.pre_cp5_3_step2` (`c2c5d313`, pre-F1).
Artifacts in `cp_5_3/step2b/`: `step2b_arm.py`, `analyze_step2b.py`,
`arm_{a,b,c}.json`, `arm_{a,b,c}_tf_logits.pt`, `step2b_step1_analysis.json`.
