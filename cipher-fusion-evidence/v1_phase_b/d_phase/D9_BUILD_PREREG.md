# D.9 BUILD — PRE-REGISTERED quality gate + measurement protocol (LOCKED before measurement)

**Date:** 2026-05-29. **This document is committed BEFORE any gate measurement.** Choosing ε or the
eval sets after seeing results is the retracted-numbers trap; this locks them. Adopted decision
(Anil): the Goal-3 lossless bar is the **industry bounded-quality standard** (per-channel/per-token
FP8 E4M3), replacing the strict token-identity gate that §1b showed forces FP8 coverage to ~0%.

## The gate (LOCKED)
**REACHED iff BOTH:** (1) full vLLM prefill forward MFU vs the 989 bf16 reference, at sustained
700 W, **≥ 85%**; AND (2) quality **within the pre-registered bar** below.

## Quality bar (LOCKED, from the lossless-FP8 literature)
| metric | threshold | role | basis |
|---|---|---|---|
| **WikiText-2 perplexity delta** vs fp16 | **≤ 0.3%** | **HARD GATE** | "essentially lossless" per-channel/per-token E4M3 (arXiv 2411.02355) |
| **MMLU accuracy delta** vs fp16 | **≤ 0.5%** (absolute) | **HARD GATE** | <1% degradation standard (arXiv 2503.09975) |
| **Output KL** (mean per-token, p_fp16‖q_fp8) | **≤ 0.01 nats** | **REPORTED DIAGNOSTIC (non-binding)** | derived: PPL≤0.3% ⟹ ΔCE≈ln(1.003)≈0.003 nats; 0.01 = ~3× margin |

**Metric-binding rule (LOCKED, per advisor):** **PPL and MMLU are the binding hard gates; KL is a
reported diagnostic only.** A run that passes PPL+MMLU is NOT failed by KL alone (KL=0.012 with
PPL+MMLU within-bar still passes); a low KL does NOT excuse a PPL/MMLU miss. Rationale: output-KL is
E over p_fp16's own distribution, while PPL-CE is E over the data — they coincide only for a
well-calibrated model, so KL is informative, not authoritative.

## Eval sets (LOCKED, fixed)
- **PPL:** WikiText-2-raw-v1 **test** split, concatenated + tokenized, **first 50 non-overlapping
  windows of 2048 tokens = 102,400 tokens**, mean per-window NLL → perplexity. (`datasets` lib;
  internet confirmed reachable.) Same windows for fp16 ref and fp8.
- **MMLU:** **0-shot, 500 questions** = first 100 each of {abstract_algebra, anatomy, astronomy,
  college_computer_science, high_school_mathematics}. Answer = argmax letter-logprob over A/B/C/D.
  Same questions for fp16 ref and fp8.
- **Output KL:** held-out = the WikiText-2 PPL windows (per-token KL of fp8 vs fp16 next-token dists).

## Implementation (LOCKED) + the reachability-vs-CIPHER-delivery split (stated)
- **Single FP8 instance for BOTH MFU and quality** (advisor: mixing impls voids the gate) =
  **vLLM-native FP8** (`quantization="fp8"`), the production serving path. The exact vLLM fp8 scheme
  (per-channel weight / per-token-dynamic activation vs per-tensor) is **reported at runtime**; if it
  defaults to per-tensor weight (worse quality) it is configured to per-channel or the limitation is
  named.
- **This measures REACHABILITY** (can FP8 prefill deliver ≥85%-at-lossless on this H100, on the path a
  customer runs). It does **NOT** validate CIPHER's `ba873f44` substrate-line FP8 delivery — that is
  blocked in-container by the `cipher_kv_bridge` Python 3.10/3.12 mismatch (§1b finding) and is
  carried, gated on the plugin fix. The report states this split; "vLLM FP8 reaches 85%" is NOT
  "CIPHER delivers 85%".

## Protocol order (LOCKED, per advisor — collapses the build)
1. Measure **all-fp8 quality FIRST** (PPL + MMLU + KL) vs fp16, same engine.
2. **If within bar** → coverage = 100%, the per-layer bounded-quality engagement is **moot**; measure
   that same all-fp8's prefill MFU (+ watts + clock) → REACHED if ≥85%.
3. **If all-fp8 exceeds the bar** → build the contingent per-layer engagement (engage FP8 only on
   within-bar layers); report MFU at the within-bar coverage.

## Confounds controlled (carried from §1/§1b — they recur)
- Prefix-cache OFF + distinct prompts per timed call (else MFU is a KV-cache-hit artifact).
- Sustained clock + aggregate watts logged during repeated prefills (not a burst).
- **Dual-denominator honesty:** ≥85% is vs the 989 bf16 anchor (FP8 via a bf16 yardstick — the
  campaign's pre-set denominator); vs the FP8-native 1979 peak the same work is ~45% and the 700 W
  wall stands. One line in the report; not a reopening.
- **Headline rule (LOCKED):** never "85% MFU" unqualified — always **"85% MFU at <0.3% PPL"**. The
  qualification IS the honesty.

## Outcomes (pre-registered)
- **REACHED** — MFU ≥85% AND PPL≤0.3% AND MMLU≤0.5%. Headline "85% MFU at <0.3% PPL (WikiText-2 / MMLU
  500q)". Full regression before any anchor rotation.
- **QUALITY-LIMITED** — MFU≥85% but PPL/MMLU exceed bar → back off coverage to stay within bar; report
  the resulting (lower) MFU-at-bar.
- **MFU-LIMITED** — within bar but MFU<85% → report the measured MFU + the binding constraint.

Anchors UNCHANGED (cipher_rt_phase4 `01d4effb`/`d7-rh1-close`, kmod `0.7.0`, evidence `fd5a22f`)
until the gate passes (Mem #16). Net-new default-OFF (Mem #24 intercept-level).
