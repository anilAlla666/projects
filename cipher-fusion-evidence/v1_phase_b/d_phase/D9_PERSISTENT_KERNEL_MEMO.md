# D.9 — Deliver Goal-3 ≥85% MFU in v1 (compute-bound prefill) — design memo (STOP for approval)

**Date:** 2026-05-29. design-memo → **approve** → build → measure → close. **Collapses the
post-seed Phase-6 depth-win program into v1:** build the four named Phase-6 levers (FP8, L2-budget,
TMA/cluster attention, partition-aware Marlin) + the within-cap clock/tactic/fusion levers on the
deployed substrate and **measure to ≥85% sustained MFU on the compute-bound regime.** Supersedes
`d49dbc8` (which scoped the within-700W efficiency levers only). **No fake, no decode concession,
no Mem-#11 carve-out** (the §5 gate keeps KL=0 by conditional engagement, below).

## §0 Provenance / anchors + the codebase-split premise

| item | value | source |
|---|---|---|
| cipher_rt_phase4 | `ed130e7` / `01d4effb` / tag `d7-rh1-close` | UNCHANGED |
| cipher_kmod | canonical `02fc2d1`; **loaded `0.7.0`** | FP8/clock/tactic/fusion/attention = **rt-side**; **L2-budget = net-new kmod NR** (next free **NR 33**; 32=fairness) |
| cipher_kv_bridge.so | `5a3db034` | UNCHANGED |
| evidence | HEAD `d49dbc8` (post-R_I1) | — |
| power cap (HW, verified) | **700 W hard-locked** (Max=Default=700, `-pl` down-only) | `nvidia-smi -q -d POWER` |

**Codebase-split premise (verified on disk, state openly):** the 33-op MFU op-chain + the FP8
substitution stack live in **`cipher-may13-evidence`** (HEAD `fc8a9ae`): `src/cipher_fp8_fused_quant.cu`,
`include/cipher_fp8_compute.h`, `STAGE13_FP8_REPORT.md` (FP8 M=2..64 path, correctness PASS, 4096²
proxy measured). The **deployed `cipher_rt_phase4` has the actuators + `cipher_workload_detect.cpp`
but never ran the may13 MFU op-chain / FP8 path** → **the current MFU on the deployed substrate is
UNMEASURED.** We do **not** anchor on op31-prod's 75% (different codebase) or the retired 745
(Mem #6). §1 establishes the honest engineer-from number on **this** substrate first.

## §1 BASELINE RE-MEASUREMENT (the honest current number, FIRST — no estimate)

On the **current deployed** substrate, bf16:
- **(a) 4096² GEMM, all-on** (the op31-prod proxy re-run on THIS substrate) — confirms whether 75%
  reproduces or the deployed runtime sits at ~67%.
- **(b) a REAL prefill forward** (Mistral-7B / Llama-3-8B, compute-bound batch×seq), CP-3.3 MFU
  telemetry. This is the engineer-FROM number for the gate.
- **Decode is OUT OF SCOPE** (bandwidth-bound, sub-1% MFU → Goal-2/MBU, tok/W). Stated, not conflated.

**MFU denominator (define up front):** reported against the **bf16 989 TFLOPS reference peak**
(matching the 67% = 660/989 anchor) so the gate is apples-to-apples with the baseline. FP8 completes
the same model GEMMs on 2×-denser tensor cores (H100 FP8 TC **1979** vs bf16 **989**), delivering
more bf16-equivalent useful work/second/watt → it lifts **MFU-vs-989** past the bf16 power wall.
We report MFU vs **both** the 989 reference and the engaged-dtype peak; the **≥85% gate is vs the
989 reference**.

## §2 Regime + why headroom exists

Compute-bound **prefill / large-batch / training-adjacent** GEMM. Roofline ceiling = P_peak (989
bf16 / 1979 FP8), not I·BW. If §1 confirms ~67%, that is **power/efficiency-bound at 700 W, NOT
roofline-bound** — the cap forces sustained clock below max, and FP8's 2× density + the efficiency
levers recover useful-FLOPs/J within the cap. **Headroom is real; the question is coverage + compose
(§4/§5).**

## §3 The levers — mechanism @700W · built-vs-net-new · Mem-#24 · est. pts · confidence

- **L1 FP8 prefill GEMM — the load-bearing lever.** Intercept the prefill cublasGemm/cuBLASLt path;
  weight B → FP8 E4M3 (per-tensor absmax, one-time, static weights); activation A inline-quant;
  `cublasLtMatmul` FP8 types; **fp16 in/out**. **Built** in may13 (`cipher_fp8_fused_quant.cu` —
  single-launch grid-sync absmax→scale(=absmax/448)→quant; correctness PASS). **Net-new on the
  deployed substrate:** port the fused-quant kernel + the cuBLASLt fp16-in/FP8-compute path into
  `cipher_rt_phase4` and wire it to the prefill GEMM intercept. **Scope to PREFILL** (large GEMMs
  amortize the quant launch over the matmul; the may13 decode failure was 50 µs × 44,800 tiny calls
  + torch-2.7 redundancy — neither applies to prefill). Mem #24: intercept-level ✓. **FP8 doubles
  compute density → it is the lever that breaks the bf16 700-W power wall** (85%-of-989 = 841 TFLOPS
  is only 42% of FP8 peak → not power-blocked). **est. +12–20 pts, GATED BY COVERAGE (§5).**
  Confidence: med on potential, **low-med on coverage surviving KL=0**.
- **L2 L2-budget enforcement — net-new kmod NR 33.** `cudaAccessPolicyWindow` + a cross-tenant
  budget cap (the kmod field `l2_residency_kb` exists but has no enforcement NR) so hot weight tiles
  stay L2-resident → fewer HBM bytes → less power → higher clock at the cap. Mem #24: kmod ledger +
  rt-side window ✓. **est. +2–5 pts.** Confidence: low-med.
- **L3 TMA + thread-block-cluster attention — net-new kernels.** FA3-class warp-specialized
  (TMA/WGMMA ping-pong) hits 75–85% on H100. Substitute at the **W.5 attention intercept** — already
  GOT-patches 6 patterns (`cipher_rt_attn_6pattern.c`: P1 FA2 `run_mha_fwd`, P2 FA3 Hopper, P5
  FlashMLA, P6 PagedAttn-v2; P3/P4 FlashInfer JIT = residue). Mem #24: intercept-level kernel
  substitution ✓. **est. +2–6 pts on the attention-heavy prefill fraction.** Confidence: low-med.
- **L4 partition-aware Marlin — the weak leg (R-W3.1).** Marlin is INT4 and **structurally
  full-GPU**; partition-routing may deadlock; ORACLE must gate it on a full-GPU-tenant context. It
  is also primarily a decode-INT4 actuator, **not** the prefill-bf16/FP8 GEMM path — so it likely
  **does not compose** with the prefill regime here. **State honestly: if it doesn't compose, name
  it and report the stack WITHOUT it.** Mem #24: intercept-level ✓ but the partition-routing is the
  risk. **est. +0–3 pts.** Confidence: **low (expected non-compose for prefill).**
- **L5 VOLT clock-lock — BUILT (`cipher_rt_volt.c`).** Pin the max clock that holds ≤700 W → kill
  DVFS hunting-loss. Net-new: calibrate the fixed clock for the compute-bound class. **est. +2–5.**
  Confidence: med.
- **L6 tactic-pin + RMSNorm/SiLU fusion + PDL.** Most FLOP/J cuBLASLt algo per shape (intercept
  built; **algo-pin table net-new**) + fuse memory-bound epilogues into the GEMM + **PDL** overlap
  (both **net-new**, not on disk). Mem #24: intercept + NVRTC ✓. **est. +3–8.** Confidence: low-med.

## §4 Stacked math → 85%

67% = 0.943 TFLOPS/W; **85% = 841 TFLOPS = 1.121 TFLOPS/W ⇒ +~19% useful-FLOPs/J at 700 W.** **FP8
is the load-bearing lever** — its 2× density is the only thing that can clear the bf16 power wall
within 700 W; L2/L3/L5/L6 compose the remainder and harden the bf16 path; L4 is optional/weak. The
85% target therefore **hinges on FP8 COVERAGE** — the §5-measured fraction of prefill GEMMs that
pass output-identity. At high coverage the stack reaches ≥85% on prefill; **if coverage is too low
(FP8 E4M3 rounding flips output tokens), the +20 pt does not land → named PARTIAL cause (§7).**
**This is a build TARGET measured in §6, NOT asserted.**

## §5 KL=0 GATE (Mem #11 — NO amendment; conditional engagement)

**Every lever, including FP8, preserves OUTPUT: per-tenant KL=0 on the output distribution.** FP8 is
a **CONDITIONAL substitution under the standard contract** — engage **only** on the prefill
GEMM-class where the **full-forward output is token-identical** to the bf16 reference (argmax/sampled
token unchanged); **decline to bf16 passthrough** everywhere it isn't — the same decline pattern as
phi-2 declining Marlin on shape gates, or fp16 declining the dtype gate. **Coverage = the measured
fraction of prefill GEMMs FP8-able while the final output stays token-identical** (determined by
ablation vs the bf16 reference); reported, not assumed. **KL=0 holds because FP8 fires only where it
is output-preserving — NOT by tolerating loss.** Any divergence on an *engaged* path = **HARD STOP.**
All non-FP8 levers (L2, TMA passthrough-equivalent, clock-lock, tactic-pin, fusion) are
**bit-equivalent → KL=0 unconditionally.** *(Honest tension to name: strict token-identity may force
FP8 to decline broadly → low coverage → the +20 pt underdelivers; that is a PARTIAL outcome, not a
reason to relax the gate.)*

## §6 Measurement gate

Compute-bound **prefill (real forward)** + 4096² GEMM, this H100, bf16:
- **Re-baseline (§1)** → **+L5** → **+L6** → **+L1 (FP8)** → **+L2** → **+L3** → **+L4**, **each
  delta MEASURED** (no estimate survives unmeasured). **Report FP8 output-identity coverage.**
- **Power-cap sweep (200→700 W, down-only)** to confirm power-bound + locate the knee.
- **Gate: ≥85% sustained MFU (vs 989 bf16 ref) on the compute-bound prefill workload at ≤700 W,
  with KL=0 on output across all engaged paths.**

## §7 Pre-registered outcomes (anti-spin)

- **PASS** — ≥85% sustained at ≤700 W, KL=0 on output, per-lever contributions + FP8 coverage stated.
- **PARTIAL** — measured X (67<X<85); name the binding lever — **likely low FP8 output-identity
  coverage**, partition-Marlin non-compose, or TMA-attention — and whether it needs >700 W / more build.
- **CEILING** — hard wall <85% at 700 W with all levers on → name the cause (silicon FLOP/J at 700 W,
  or **FP8 coverage capped by output-identity**); 85% then needs a >700 W board / Blackwell. NOT
  faked, NOT forced — this is the build that makes the honest attempt: **we make it, we don't fake it.**

## §8 Close gate + discipline

- **Mem #24:** intercept-level only (cuBLAS/cuBLASLt + attention GOT-patch + VOLT + L2 kmod-NR + the
  GEMM dispatch path). A forward-replacing megakernel is **named OUT OF SCOPE / HARD STOP**.
- **Mem #11:** KL=0 on output for **all** levers; FP8 conditional-engage per §5 — **no carve-out**.
- **Mem #6:** build only on the §1 re-measured baseline + measured per-lever gains; **745/75% stays
  retired, never re-asserted.**
- **Mem #16:** anchors UNCHANGED until an approved build; net-new (FP8 port, L2 kmod NR 33, TMA
  kernels, algo-pin, fusion/PDL) → **full regression before any rotation**: every-prior-model KL=0
  OFF byte-identical + 30-min N=128 soak INCOHERENT=0 + the per-lever ablation. **Default-OFF**
  (worst case == the §1 baseline). The L2 kmod NR is additive (reserved NRs untouched).

**STOP — awaiting approval.** On approval, ordered build: §1 re-baseline → L5 → L6 → L1 (FP8 port,
the load-bearing lever) → L2 (kmod NR 33) → L3 → L4 (or named non-compose), each KL=0-gated and
ablation-measured, power-cap sweep, STOP at the per-outcome verdict.
