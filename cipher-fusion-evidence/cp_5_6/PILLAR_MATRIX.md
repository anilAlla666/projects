# Phase A — Baseline Three-Pillar Matrix (WL01–WL24)

**Status: IN PROGRESS.** Substrate `a7ac8e97`. Methodology: `PHASE_A_PLAN.md`.
Per workload — PASS 1 (per-tenant MFU + TPW, 3 arms, teacher-forced ≥99%
gate), PASS 2 (4-tenant aggregate), PASS 3 (regime classification).

MFU = analytical FLOPs (2·N_params·tokens) ÷ H100 SXM5 dense FP16 peak
989.4 TFLOP/s ÷ wall. Power = arm-level mean (1 Hz nvidia-smi). "TF pass" =
prompts ≥99% teacher-forced top-1 agreement vs clean-FP16 gold.

Progress: WL01 ✓ (PASS 1) · WL02–WL24 pending.

---

### WL01 — LLM decode B=1 (TinyLlama-1.1B)

**Regime (PASS 3):** small-model (1.10B) single-stream autoregressive decode,
B=1, 128-token greedy. Memory-bandwidth-bound — compute units largely idle
(MFU ~0.02%). Actuators engaged: Marlin INT4 GEMM (handles the decode
matmuls); DVFS clock-lock @1000 MHz (all-on, 148 W → 85 W); n-gram spec
decode (all-on, accept_rate 0.20–0.44 — no loop). KV ops: none beyond stock.

**PASS 1** — per-tenant, 3 arms, n=5 prompts:

| arm | tok/s | W | tok/W | MFU% | TF pass |
|---|---|---|---|---|---|
| vanilla | 71.9 | 147.9 | 0.486 | 0.0199 | 5/5 |
| marlin | 74.1 | 146.7 | 0.505 | 0.0204 | 0/5 |
| all-on | 87.0 | 85.0 | 1.024 | 0.0239 | 0/5 |

- **Pillar 1 (MFU):** all-on/vanilla **1.20×** (0.0199% → 0.0239%). MFU is
  ~0.02% in absolute terms — expected for B=1 decode on a 1B model
  (memory-bound); the headroom is the multi-tenant pillar's to reclaim.
- **Pillar 2 (TPW):** all-on/vanilla **2.11×** all-prompt (0.486 → 1.024
  tok/W), carried by the DVFS power cut (148 W → 85 W) plus a real ~1.2×
  tok/s gain (Marlin INT4 helps at this small size, unlike Mistral-7B).
  **Gated TPW: n/a — substrate arms pass 0/5 the ≥99% teacher-forced gate**
  (agreements 0.84–0.91 — honest Marlin INT4 drift, same signature as
  CP 5.6 P2 on Mistral-7B; not a loop — accept_rate 0.20–0.44).
- **Pillar 3 (multi-tenant):** 4-tenant aggregate tok/W = **0.998** (2.04×
  the 1-tenant vanilla baseline). Decomposes to **concurrency 2.04× ×
  substrate 1.00×** — the substrate adds zero tok/W; 4-tenant vanilla is
  0.994. All-on is 2.3× slower in absolute throughput (86.7 vs 197.9 tok/s)
  for the same efficiency — DVFS trades power for speed 1:1. Uses only
  32/132 SMs (4×8-SM green partitions). Full detail: `WL01_MULTITENANT_REPORT.md`.
- **Observed gap:** the ≥99% TF gate is not met — Marlin INT4 top-1 vs FP16
  ~85–91% (honest INT4 drift; logit-KL ≈ 0.085, ~4/5 pass at KL≤0.1).
  Neither single- nor multi-tenant reaches ≥3.6× tok/W substrate-attributable.
  The whole 3.6× is open Phase B engineering — concurrency gives 2.0× free,
  the substrate currently contributes 0×.
