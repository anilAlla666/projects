---
name: cipher-t43-envelope
description: "VOLT envelope characterization (2026-05-14 night, 7 conditions). T4.3.2's +57% reproduces on TinyLlama-1.1B B=1 (+54.97%). Envelope on Mistral-7B-class is -15% to -2% — VOLT is workload-mechanism-bounded, not workload-batch-bounded."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d28504c-5084-4f22-8df4-2a4f96d437ce
---

T4.3.x envelope campaign (2026-05-14 night) ran 7 matched-pair
conditions to characterize VOLT's lift across (model, batch, calibration,
binary, pod state). Outcome falsifies the "+57% on B=1 decode"
generalization and commits a narrower, mechanism-grounded envelope.

**The headline finding:**
T4.3.2's +57.28% measurement was real and reproduces on its own
workload (C6 +54.97%, CIs overlap). It does NOT generalize beyond
TinyLlama-1.1B B=1.

**The 7-condition envelope:**

| Cond | Model | B | Δ tok/W | 95% CI | mechanism |
|---|---|---|---|---|---|
| C6 | TinyLlama-1.1B | 1 | **+54.97%** | +54.60..+55.33 | bandwidth-bound: clock cut free |
| C2 | Mistral-7B | 8 | -1.33% | -4.03..+1.37 | mixed |
| C3 | Mistral-7B | 32 | -1.44% | -2.16..-0.73 | compute-bound: VOLT picks nominal |
| C7 | Mistral-7B | 1 @ MHz=1600 | -2.03% | -2.36..-1.70 | best clock for Mistral B=1: still neutral |
| C1 | Mistral-7B | 1 @ MHz=1000 | -13.78% | -14.79..-12.78 | over-aggressive cut: tok/s drops 41% |
| C4 | Mistral-7B | 1 @ MHz=1000 (post-reset) | -14.71% | -16.09..-13.32 | identical to C1 — pod-state irrelevant |
| C5 | Mistral-7B | 1 @ MHz=1000 (pre-T4.6.1 binary) | -15.45% | -16.74..-14.16 | identical to C1 — binary irrelevant |
| T4.3.2 anchor | TinyLlama-1.1B | 1 | +57.28% | +56.65..+57.90 | (prior measurement) |

**Falsification logic — what does NOT explain the C1 vs T4.3.2 gap:**
- ❌ Methodology only: C6 reproduces T4.3.2 on its workload
- ❌ Pod-state drift: C4 = C1 (within 1pp)
- ❌ Binary regression: C5 = C1 (within 2pp)
- ❌ Mistral B=1 sweet spot exists at some clock: C7 best-case is still neutral

**What DOES explain it:** model size × clock-target interact to
determine memory-bandwidth-vs-compute ratio. `batch_to_mhz` calibration
table is keyed only on batch and was tuned for TinyLlama-class
(1.1B params, 2.2GB weights). On Mistral-7B (14GB), B=1 → 1000MHz
mapping over-cuts; compute is no longer free-riding behind memory wait.

**Mechanism story (load-bearing):**
- TinyLlama-1.1B at 1980MHz: ~150GB/s bandwidth demand, far below
  H100's 3 TB/s HBM3 ceiling. Bandwidth-bound. Cut clock to 1005MHz
  → tok/s preserved (67→67), watts drop 36%, tok/W +57%.
- Mistral-7B at 1980MHz: ~500GB/s demand, kernels larger per memory
  wait. Cut to 1000MHz → compute can't hide → tok/s drops 41%, watts
  drop only 31%. Net -14%.
- Mistral-7B at 1600MHz (best calibration): tok/s drops 11%, watts
  drop 9%. Balanced → -2%. No clock gives positive lift on Mistral B=1.

**THE HONEST MOAT CLAIM (replaces "+57% on B=1 decode"):**

"VOLT delivers +55% tok/W on **memory-bandwidth-bound decode**
(small-model, low-batch, calibrated clock target). On compute-bound
workloads — 7B+ production regime — VOLT is neutral or negative
depending on calibration."

**Marvel-pitch implication:**
B200-parity-on-H100 framing assumed VOLT contributed meaningful lift
on production-realistic workloads. Data does not support that for 7B+.
Two paths:
1. **Narrow target:** "TinyLlama-class B200-parity" — defensible, smaller.
2. **Recalibrate:** extend batch_to_mhz to (model_size, batch) axis,
   re-measure C7-equivalent points across (TinyLlama, Mistral-7B,
   Llama-70B-quant) × clock sweep. Map lift surface. Determines
   whether envelope widens or moat is small-model only.

**What this does NOT change:**
- T4.3.2's measurement is real and reproduces (C6 +54.97%)
- VOLT mechanism (NVML clock-lock + kmod ioctl fall-through) works
  correctly across all conditions
- Discipline pattern (matched pairs, decode-window-only power
  averaging) is correct; gap was workload coverage, not method

**Methodology audit finding (the dominant variable):**
T4.3.2 anchor used TinyLlama-1.1B, prompt="Hello, world.",
MAX_NEW=32 per generate, tight loop inside 60s window via
wl01_decode_b1.py. Tonight's volt_b1_check.py used Mistral-7B,
single generate(max_new_tokens=400), 10Hz Python sampling.
**Model swap was the primary cause of the C1 vs T4.3.2 gap.**
Methodology second-order at most.

**Discipline:** measurement-only sub-phase. ABI nrs 1-10 untouched.
No kmod or libcipher_rt source changes. Fallback md5s 55ab8c0c /
86618c30 unchanged. Taint 12288.

**Artifacts:** `cipher-phase4-evidence/t4_3_envelope/{C1..C7}/` with
per-pair watts.csv + run.log + progress.txt + summary.json.
envelope_driver.py + run_condition.sh + analyze_condition.py durable
for future recalibration sweeps.

Linked: [[cipher-t431-volt-shipped]] (T4.3.1 original measurement),
[[cipher-t432-kmod-volt-ioctl]] (kmod ioctl path that works correctly
across all conditions), [[cipher-lift-framing]] (was the strategic
framing for the moat claim being characterized).

Full doc: `PHASE_4_T4_3_ENVELOPE.md`.
