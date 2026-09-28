# CP 5.6 Priority 2 — Teacher-Forced Correctness Gate + TPW Re-measurement

**Date:** 2026-05-18 **Type:** GPU measurement (H100 80GB, Mistral-7B-v0.1).
**Substrate:** `libcipher_rt.so` md5 **`a7ac8e97`** (CP 5.6 P1 fix), with
`libcipher_v2` `86618c30` / kmod `e2f50452` / `cipher_kv_bridge` `8d6ffe3f` —
all unchanged, nothing rebuilt this STEP.

**Verdict: C — the 3.617× tok/W headline does NOT survive re-measurement on
the F1-fixed substrate. It collapses to ~1.54×.** The headline was real
*timing* of the F1-degenerate looping decode that CP 2.4 measured; with the
loop removed the throughput half of the lift evaporates. CP 5.6 P2 is
complete; the 3.617× / 1.795× CP 2.4 composed headline must be **retracted
and corrected**.

---

## 1. What P2 set out to do

`F1_TPW_RECORDS_REVIEW.md` ruled the anchored **3.617× tok/W** headline
(CP 2.4 composed gate, all-on/vanilla) **SUSPECT**: it engaged the
F1-degenerate Marlin full-GPU path, the all-on arm carried the
`accept_rate=1.000` loop signature, and the harness recorded only timing —
zero output-correctness. CP 5.6 P1 shipped the cross-context ordering fix
(`a7ac8e97`). **P2 re-measures the headline on the fixed substrate, behind a
correctness gate**, to decide whether it can be re-asserted.

## 2. Methodology

Two decoupled measurements per (prompt, arm), n=5 prompts (the CP 2.4 varied
set), 3 arms (vanilla / marlin / all-on — env toggles identical to
`cp_2_4/run_composed.sh`), Mistral-7B-v0.1 FP16, B=1 greedy, 128-token budget.

- **Free-running** greedy decode — production mode, power-windowed — yields
  tok/s, watts, tok/W, decoded text, n-gram `accept_rate`.
- **Teacher-forced gate** — an incremental KV-cache decode whose input at
  every step is the *gold* token, never the model's own output. Cascade-free,
  and (explicit `cache_position` + `attention_mask`) numerically identical to
  the production forward — so it is FP-tie-noise-free. Per-step argmax-vs-gold
  agreement over the 128 gold positions; a pair passes at **≥99%**.
- **Gold** = clean-FP16 Mistral free-running greedy (no Marlin/spec/VOLT).
- The naive gate (substrate free-run vs a free-running gold at 99% identity)
  is mechanically broken — FP-tie non-determinism + the autoregressive cascade
  fail a correct substrate. Teacher-forcing removes both confounds.

**Power is arm-level** (1 Hz `nvidia-smi` is too sparse to slice per prompt;
~12-24 samples/arm gives a stable mean — VOLT-lock keeps it near constant) —
matching CP 2.4 `analyze_composed.py`. **Pre-flight:** `teacher_forced()`
reproduced `m.generate()` exactly on an 8-token toy case before any full run.

## 3. Sanity — PASS

Vanilla teacher-forced agreement vs gold: **1.0000 on all 5 prompts**
(128/128, 128/128, 128/128, 71/71, 128/128). The incremental KV-cache
teacher-forced loop reproduces the gold run exactly — the gate, the gold
capture and the harness are valid. Substrate failures below are therefore
real, not artefacts.

## 4. Results

Power (arm-level mean): vanilla **211.9 W**, marlin **180.5 W**,
all-on **92.5 W** (DVFS clock-lock @ 1000 MHz).

| prompt | arm | tok/s | tok/W | TF agree | gate | accept_rate |
|---|---|---|---|---|---|---|
| 0 | vanilla | 48.27 | 0.2278 | 1.0000 | PASS | — |
| 0 | marlin | 38.95 | 0.2157 | 0.9062 | FAIL | — |
| 0 | all-on | 33.55 | 0.3629 | 0.9062 | FAIL | 0.766 |
| 1 | vanilla | 48.24 | 0.2277 | 1.0000 | PASS | — |
| 1 | marlin | 38.87 | 0.2153 | 0.8984 | FAIL | — |
| 1 | all-on | 33.27 | 0.3599 | 0.8984 | FAIL | 0.575 |
| 2 | vanilla | 48.58 | 0.2293 | 1.0000 | PASS | — |
| 2 | marlin | 38.86 | 0.2152 | 1.0000 | PASS | — |
| 2 | all-on | 23.75 | 0.2569 | 1.0000 | PASS | 0.188 |
| 3 | vanilla | 48.41 | 0.2285 | 1.0000 | PASS | — |
| 3 | marlin | 38.79 | 0.2149 | 0.9296 | FAIL | — |
| 3 | all-on | 45.19 | 0.4888 | 0.9296 | FAIL | 0.500 |
| 4 | vanilla | 48.47 | 0.2288 | 1.0000 | PASS | — |
| 4 | marlin | 38.88 | 0.2154 | 0.9297 | FAIL | — |
| 4 | all-on | 26.73 | 0.2892 | 0.9297 | FAIL | 0.255 |

Per arm: vanilla TPW **0.2284** (5/5 pass); marlin **0.2153** (1/5 pass);
all-on all-prompt TPW **0.3515** (1/5 pass), tok/s mean **32.50**.

## 5. Finding — the headline collapses to ~1.54×, and why

**5.1 F1 is fixed.** In CP 2.4 the all-on arm reported `accept_rate=1.000` on
every prompt — CP 2.4 §3.2's own "loop-replay signature… base-model output
collapses into exact repetition." On `a7ac8e97` the all-on `accept_rate` is
**0.188–0.766** — normal n-gram acceptance. The degenerate loop is **gone**;
free-run output is coherent (distinct-token ratios near the clean-FP16 gold).
**CP 5.6 P1 worked.**

**5.2 The headline collapses.** Re-measured composed all-on/vanilla:

| metric | CP 2.4 headline | CP 5.6 P2 (a7ac8e97) |
|---|---|---|
| tok/s lift (all-on/vanilla) | **1.795×** | **0.672×** (all-on is *slower* than vanilla) |
| tok/W lift (all-on/vanilla) | **3.617×** | **1.540×** |

**5.3 The collapse decomposes cleanly.** 1.540 ≈ **0.672** (tok/s) × **2.29**
(power, 211.9 W → 92.5 W).
- **The DVFS power benefit is real and survives** — ~2.3× power reduction
  from the 1000 MHz clock-lock, output-correctness-independent. This is
  exactly what `F1_TPW_RECORDS_REVIEW.md` predicted ("not invalidated: the
  DVFS power component").
- **The 1.795× tok/s "speedup" does not survive.** It was n-gram speculative
  decode hitting 100% acceptance *because the decode was looping* — a 3-gram
  draft trivially predicts an exact repetition. Remove the loop and n-gram
  acceptance falls to 0.19–0.77; the all-on arm is then **slower than
  vanilla** (spec-decode overhead without the artificial acceptance). The
  3.617× headline was real timing of degenerate output.

**5.4 Corroborating — the 99% correctness gate.** marlin and all-on pass the
teacher-forced gate on only **1/5** prompts; the other four sit at
**89.8–93.0%** (all-on TF == marlin TF exactly — teacher-forcing bypasses
spec-decode and isolates the Marlin GEMM numerics; DVFS does not change
numerics). This is **honest Marlin INT4 quantization drift, not a substrate
bug**:
- prompt 2 (code) passes at **100%** on marlin and all-on — a bug cannot
  produce a perfect prompt; code is low-entropy/peaked so argmax is robust to
  small perturbation, while prose argmax flips at high-entropy positions —
  the textbook signature of quantization noise;
- the teacher-forced mismatches are plausible alternative tokens (`"the"`/`"a"`,
  `"want"`/`"know"`, `"largest"`/`"second"`, digit-for-digit) — not garbage;
- F1 P1's single-GEMM verify on `a7ac8e97` was rel_err 0.140 / cos 0.9905 —
  direction-correct INT4 noise.

So the substrate is **not broken** — it produces coherent, honest INT4 output.
But that INT4 output diverges from FP16 on ~7-10% of tokens, below a strict
99%-top-1-equivalence bar. The 3.617× cannot be re-asserted as efficiency on
output verified equivalent to FP16 — separately from, and in addition to, the
throughput collapse in §5.2.

## 6. Verdict — C

`cp56_p2_result.json`: **VERDICT C.** The 3.617× / 1.795× CP 2.4 composed
headline cannot be re-asserted. The re-measured composed lift on F1-fixed
honest decode is **1.54× tok/W** (0.67× tok/s × 2.3× power); the durable
component is the **~2.3× DVFS power benefit**, not throughput.

*A/B/C labels:* A (re-assert), B (correct the number), C (retract). The
defining "prior prompt" for these labels is not in this session's context —
the adjudicator should confirm the mapping; the numbers and decoded text
above are the load-bearing evidence.

## 7. Implications (for adjudication — not acted on here)

- **CP 2.4 is a closed CP whose central headline this finding retracts.** Its
  md5-anchored report (`CP_2_4_REPORT.md`, c1a2d310) states 3.617× tok/W /
  1.795× tok/s as the composed investor number. Phase 2 was closed 15/23 with
  CP 2.4 among them. This re-measurement supersedes that headline.
- The honest CIPHER composed claim on Mistral-7B B=1, F1-fixed: **~1.5× tok/W,
  carried by DVFS power reduction; no throughput lift (all-on is slower than
  vanilla at B=1)**. Consistent with the standing record that Marlin is a
  B≥8 win and a B=1 regression ([[cipher-t45-substrate-marlin]]) and that the
  VOLT envelope is memory-bandwidth-bound-decode-only ([[cipher-t43-envelope]]).
- The 99% teacher-forced gate is a strict FP16-equivalence bar; this Marlin
  INT4 sits at ~90-93%. Whether to ship on a looser INT4-quality gate, or to
  improve the quantizer, is a separate scoping question.

## 8. CP 5.6 status

CP 5.6 P1 (F1 fix) and P2 (gated re-measurement) are both **complete**. CP 5.6
delivered its scope: F1 is fixed, and the headline has been honestly
re-measured. The finding is negative for the headline. **Recommend CP 5.6
close on this evidence, with the 3.617× headline retracted/corrected to
~1.54×.** Per campaign discipline — **awaiting adjudication; no further CP
started.**

## 9. Artefacts

New (under `cipher-fusion-evidence/cp_5_6/`): `CP_5_6_P2_PLAN.md`,
`CP_5_6_P2_REPORT.md`, and `p2/` — `tf_gate_driver.py` (also at
`/home/ubuntu/cipher_rt_phase4/tf_gate_driver.py`), `run_cp56_p2.sh`,
`analyze_cp56_p2.py`, `test_analyze_cp56_p2.py`, `gold.json`,
`{vanilla,marlin,allon}.json` + `.watts.csv` + `.log`, `cp56_p2_result.json`,
`run_cp56_p2.runlog`, `analyze.log`, `preflight.log`. No anchors rotated; no
binaries rebuilt.
