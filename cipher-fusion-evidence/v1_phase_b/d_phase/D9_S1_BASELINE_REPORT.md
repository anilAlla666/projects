# D.9 §1 — BASELINE RE-MEASUREMENT (deployed substrate, compute-bound prefill) — REPORT

**Date:** 2026-05-29. **Measure-only. Anchors UNCHANGED, no rotation, no substitution engaged
(KL N/A).** Establishes the honest current compute-bound MFU on the deployed `cipher_rt_phase4`
substrate — the engineer-FROM number for the D.9 lever strategy. All MFU vs the **989 TFLOPS
H100 bf16/fp16 dense reference peak** (the 67% anchor's denominator; fp16 throughout, no dtype
switch). HW: power cap **700 W hard-locked** (Max=Default=700, `-pl` down-only); sm.max 1980 MHz.

## §0 Anchors + two infrastructure findings (both load-bearing for V.1)

Anchors verified unchanged: rt `ed130e7`/`01d4effb`/`d7-rh1-close`; kmod loaded `0.7.0`; bridge
`5a3db034`; evidence `55c2126`.

1. **The deployed container image FORCES CIPHER on** — it bakes `CUDA_INJECTION64_PATH=/usr/lib/
   cipher/libcipher_rt.so` **plus engage env** `CIPHER_MARLIN=on, CIPHER_SENSE=on, CIPHER_KOOPMAN=1,
   CIPHER_KVDEDUP=1, CIPHER_AUDIT=1`. **`-e VAR=""` does NOT override it** (must `unset` at runtime).
   So a naive "baseline" run is actuators-ON, not observe-only. **Corrected:** I ran both — see §5
   (the existing actuators do NOT move compute-bound prefill MFU; Marlin's M≤64 gate excludes the
   M≥2048 prefill GEMMs, confirmed by ops-ON ≈ clean to <0.3 pt).
2. **CIPHER's CUPTI subscription blocks `torch.profiler`** (CUPTI is single-subscriber) → the
   component breakdown is only obtainable with CIPHER injection OFF. *Implication for V.1: MFU
   instrumentation cannot use torch.profiler while CIPHER's CUPTI telemetry is live; use CP-3.3
   telemetry (CIPHER's own) or profile CIPHER-off.*

The §1 numbers below are the **genuinely-clean** run (`unset` injection; 0 CIPHER log lines, `INJ=[]`).

## §1 (A) 4096² GEMM proxy — the GEMM ceiling on THIS substrate

Sustained fp16 4096³ GEMM, 60 s, CIPHER off: **643.6 TFLOPS = 65.1% MFU** (vs 989). Sustained clock
**~1300 MHz** (of 1980 max) at ~696 W — i.e. the 700 W cap throttles the clock to ~66% of max,
which IS the ~65% MFU. **The op31-prod 75% (different codebase) does NOT reproduce on the deployed
substrate; the proxy sits at ~65%, consistent with the retired-745's honest 67% sustained anchor.**
This is power-bound, not roofline-bound.

## §2 (B) Component-resolved prefill forward (Mistral-7B, fp16, CIPHER off)

`P_linear` = 7.241 B; 32 layers, 32 heads, head_dim 128. MFU vs 989. Component = CUDA-kernel
self-time split (categorization **verified** against the top-kernel dump: GEMM = `nvjet_sm90_hsh_*`,
attn = `cudnn_…_sdpa_sm90_flash_fprop_wgmma`, other = elementwise / `CatArrayBatchedCopy` (KV-cat) /
reduce (RMSNorm) / copy/cast).

| B | S | tokens | wall ms | MFU% (vs989) | clock MHz | GEMM% | attn% | epilogue% |
|---|---|---|---|---|---|---|---|---|
| 1 | 2048 | 2048 | 64 | 49.3 | 1800 | 62.8 | 4.7 | 32.5 |
| 1 | 4096 | 4096 | 151 | 42.3 | 1725 | 51.2 | 19.5 | 29.3 |
| 4 | 2048 | 8192 | 248 | 50.4 | 1680 | 64.3 | 4.1 | 31.6 |
| 8 | 2048 | 16384 | 491 | 50.5 | 1530 | 64.4 | 3.9 | 31.7 |
| 8 | 4096 | 32768 | 1193 | 43.2 | 1530 | 51.3 | 20.4 | 28.3 |
| **16** | **2048** | **32768** | **978** | **50.9** | 1425 | 64.7 | 3.9 | 31.4 |

**Key decomposition (best fat-shape B=16×S=2048, 50.9% MFU):** GEMM kernels carry 64.7% of time
(631 ms) and do all 474.6 TFLOP of linear work → **~752 TFLOPS = 76% MFU *during the GEMM kernels
themselves*** (higher than the proxy's 65% because the forward isn't continuously power-saturated, so
clock holds ~1425 MHz). **The forward's 50.9% is GEMM diluted by ~35% non-GEMM time** (≈31% epilogue
at ~0 useful FLOPs + ~4% attention). At S=4096 attention balloons to ~20% (O(S²)) and MFU falls to ~43%.

## §3 (C) FP8 delivered-throughput probe (torch._scaled_mm E4M3 vs fp16) — sizes L1

CIPHER-independent (`torch._scaled_mm` is not a CIPHER intercept path → valid regardless of §0).
Per prefill GEMM shape (M = B·S tokens; N,K = model dims):

| shape | fp16 TFLOPS | fp8 TFLOPS | fp8/fp16 | fp8 %vs989 |
|---|---|---|---|---|
| M2048 ffn_down (4096×14336) | 736 | 1365 | **1.85×** | 138% |
| M4096 ffn_down | 653 | 1225 | 1.88× | 124% |
| M8192 ffn_up (14336×4096) | 636 | 1226 | **1.93×** | 124% |
| M16384 ffn_down | 665 | 1310 | 1.97× | 132% |
| M32768 attn_proj (4096×4096) | 629 | 1237 | 1.97× | 125% |
| **all 15 shapes** | 622–736 | 1083–1365 | **1.71–1.97× (median 1.85×)** | 110–138% |

**FP8 delivers a real 1.71–1.97× on our prefill shapes — decisively beating the "<25% thin-GEMM
inference" concern.** Our prefill GEMMs (M≥2048) are fat enough to capture near-2× FP8. FP8 sustains
**110–138% of the 989 bf16 reference** (≈ 55–69% of the 1979 FP8 peak — NOT power-blocked the way
bf16 is). **FP8 is confirmed load-bearing.**

## §4 Plain statements (the §1 deliverable)

**(i) Deployed forward MFU at best fat-shape: 50.9% — but this is HF-EAGER (headline caveat).** The
product is vLLM, which fuses far more and uses CUDA graphs → the ~31% epilogue here is largely an
eager-mode artifact, so **50.9% is probably an artifactually-LOW baseline for the actual product.**
The GEMM proxy (65%, raw cuBLAS) is stack-independent and solid; the *forward* 50.9% is HF-eager-
specific. **Re-measure prefill MFU on the vLLM forward before declaring the engineer-from baseline or
sizing L6** — vLLM prefill MFU is likely materially higher, which would shrink L6's headroom and make
FP8 even more singularly load-bearing. (42–51% across the ladder, vs 989.)
**(ii) GEMM-proxy MFU 65.1%; proxy-vs-forward gap = ~14 pts** = non-GEMM dilution (epilogue ~31% +
attention ~4–20%). The proxy itself (65%) is the bf16 700 W power-cap ceiling (clock-throttled to
~66% of max). *(The retired 745/75% does not reproduce; 65% is the honest GEMM ceiling here.)*
**(iii) FP8 delivered ratio on our shapes: median 1.85× (1.71–1.97×)** — fat-GEMM regime, real.
**(iv) Where the gap to ~82–85% lives, and which levers are load-bearing + how big each must be:**

- **Epilogue (~31% of forward time, ≈0 useful FLOPs) is the single biggest forward-MFU drag** at the
  common S=2048. Lever = **L6 fusion** (RMSNorm/SiLU into the GEMM epilogue + eliminate the separate
  `CatArrayBatchedCopy`/elementwise/cast kernels). **Caveat (honest):** much of this 31% is **HF-eager
  artifact** — vLLM (the product) fuses far more, so the *production* epilogue fraction is likely
  lower and L6's headroom on vLLM is smaller than 31%. Re-measure on the vLLM forward before sizing L6.
- **GEMM (~63% of time) is bf16-power-capped.** Lever = **L1 FP8**, **load-bearing**. The robust,
  defensible result is the **RATIO: FP8 delivers ~1.85× (1.71–1.97×) over fp16** on prefill GEMMs
  (fp16 and fp8 measured the same way → the ratio is sound). **The absolute MFU landing is NOT yet
  evidenced and is burst-inflated** — two confounds: (a) Part C ran 20 iters, **not** a 60 s
  clock-sampled sustained loop, and FP8 tensor cores at full tilt also draw max power → also throttle
  toward the ~65% / ~1300 MHz sustained regime at 700 W; (b) the in-forward 76% GEMM MFU is
  **transient clock** (the forward isn't power-saturated; raising duty cycle via FP8+fusion pushes
  clock back down). A naive stack gives 73→93% — overshooting the 65% proxy ceiling, which is itself
  the tell that the stacking model ignores re-throttling. **So: FP8 ~1.85× ratio is real; the absolute
  path to ≥85% is a TARGET to validate, not evidenced here.** The build's §1 must re-measure FP8
  **sustained (60 s, clock-sampled)** and the FP8 coverage under the KL=0 conditional-engage gate
  before any 85% projection.
- **Attention is a LONG-CONTEXT-ONLY lever:** ~4% of time at S=2048 (negligible) but ~20% at S=4096.
  Lever = **L3 TMA/FA3** kernels at the W.5 attention intercept — **load-bearing only for long-prefill**;
  the current cuDNN flash SDPA is already reasonable at short S. Size L3 against the target context.
- **L4 partition-Marlin is IRRELEVANT to prefill (measured):** Marlin's M≤64 gate excludes the M≥2048
  prefill GEMMs — ops-ON ≈ clean to <0.3 pt confirms it never engaged. Marlin is a decode-INT4 actuator;
  it contributes 0 to compute-bound prefill MFU. Drop it from the prefill lever stack.
- **L5 VOLT clock-lock / L2-budget / tactic-pin:** secondary — the in-forward GEMM already holds ~76%
  at ~1425 MHz; clock-lock recovers little, L2/tactic trim the epilogue/power. Size after L1+L6.

**Lever priority (sized from the data): L1 FP8 (singularly load-bearing — the robust ~1.85× ratio) ≫
L6 fusion (epilogue 31% on HF-eager, but largely vLLM-fused already → re-measure on vLLM, headroom
likely small) > L3 TMA-attention (long-context only) > L5/L2/tactic (secondary) ≫ L4 Marlin (irrelevant
to prefill, measured).**

## §5 Net for Anil (honest sizing)

- **GEMM ceiling = 65% MFU, real and power-bound** (700 W throttles clock to ~66% of max) — solid,
  stack-independent (raw cuBLAS). The retired 745/75% does not reproduce.
- **FP8 gives a robust ~1.85× *ratio* on prefill GEMMs — the singularly load-bearing lever.**
- **The absolute path to ≥85% is NOT yet evidenced:** the forward baseline (50.9%) is HF-eager (vLLM
  likely higher), and the FP8/forward MFU numbers are *burst* (not sustained under the 700 W cap). So
  **≥85% remains a TARGET to validate** — PARTIAL (FP8 coverage / re-throttle) and CEILING (700 W
  FLOP/J wall) both stay live per the memo's §7.
- **The build's §1 must first nail: (1) FP8 sustained (60 s, clock-sampled) MFU at 700 W, (2) the
  baseline + epilogue headroom on the vLLM forward (not HF-eager), (3) FP8 coverage under KL=0
  conditional-engage.** Those three decide whether 85% is reachable or a named ceiling. **STOP here —
  no build, no further runs tonight; this report sizes the levers, it does not commit the landing.**

## §6 Discipline + ops-ON vs clean

Measure-only; anchors UNCHANGED; no rotation; no substitution engaged (KL N/A). **ops-ON vs clean:**
GEMM-proxy 65.1% (ON) vs 65.1% (clean); forward per-shape within ≤0.3 pt — **the deployed image's
existing actuator stack (Marlin/Sense/Koopman/KVDedup/Audit) does NOT move compute-bound prefill MFU**
(they target decode/INT4/KV, not large-M prefill GEMM). **No build recommendation beyond this sizing.**

Artifacts: `d9_s1_baseline.py`, `d9_s1_result.json` (clean), `d9_s1_result_opsON.json` (actuators-on,
for the comparison), `d9_s1_clean.log`, `d9_s1_partB2.log` (top-kernel verification).
