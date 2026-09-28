# D.9 BUILD — FP8 prefill reachability/quality gate (panel-verified) — REPORT

**Date:** 2026-05-29. **Reachability + quality of FP8 prefill (vLLM-executed), at the LOCKED
bounded-quality bar (D9_BUILD_PREREG.md `3f15517`: PPL≤0.3% HARD, MMLU≤0.5% HARD, KL≤0.01 diag).**
This is the *reachability* gate; **transparent CIPHER-delivery is a separate, BLOCKED finding —
see `D9_CIPHER_DELIVERY_REPORT.md` (vLLM GEMMs bypass CIPHER's intercept, nvjet).** Measure-only,
anchors UNCHANGED. Headline rule honored: always "X% MFU at Y% PPL", never bare.

## Verdict: FRONTIER — ≥85% MFU and ≤0.3% PPL are NOT cleanly co-satisfiable at the locked bar
(5-agent adversarial panel `wv4ti7h6v`: 2× FRONTIER-NOT-COSATISFIABLE, 1× QUALITY-LIMITED,
1× MFU-LIMITED — all agree no single config DEMONSTRATES both gates.)

## Measured (CIPHER-off, vs 989 bf16 ref; fp16 ref PPL 5.3525, MMLU 45.4%)
| axis | result |
|---|---|
| MFU, full-coverage FP8 (vLLM, per-tensor online) | **91.7–98.3% vs 989** (46–49% vs FP8-native 1979 — 700W wall); fp16 baseline ~64% |
| Quality, all-fp8 PER-CHANNEL (HF, true KL) | PPL **+0.37%** (FAIL 0.3%), MMLU −0.4pp (PASS), true KL 0.0036 (within 0.01) |
| Quality, all-fp8 per-tensor (vLLM) | PPL +0.44% (FAIL) |
| Per-layer back-off (per-channel) | only **drop down_proj** (SwiGLU, 27% of GEMM-time) clears: 73% coverage, PPL **+0.25%** PASS, MMLU +0.2pp |
| GEMM scheme bridge (1st-party) | per-channel rowwise FP8 GEMM = 0.886× per-tensor (~11% slower; both ~128–144% of 989) |

## The frontier (why not co-satisfiable at 0.3%)
- **Full coverage:** MFU 92–98% (PASS) but PPL +0.37% (FAIL the *locked* 0.3%).
- **Quality-passing (down_proj fp16, 73% coverage):** PPL +0.25% (PASS) but MFU **~80%** (per-channel,
  harmonic-blend MODEL) — FAILS 85%. The down_proj (PPL bottleneck) is 27% of GEMM-time; removing it
  from FP8 is exactly what drops MFU below 85%. Feasible-coverage regions are disjoint at the 0.3% bar.

## Panel-flagged caveats (recorded, honest)
1. **The within-bar ~80% MFU is a PLAUSIBILITY BAND (~77–86%), NOT gate-grade.** It is doubly-modeled
   (Amdahl harmonic blend × a single-shape per-tensor→per-channel GEMM ratio 0.886) and violates the
   prereg single-engine rule (quality = HF/per-channel, MFU = vLLM/per-tensor). The 85.9% figure
   borrowed the per-tensor rate for a per-channel config (scheme mismatch); the correct per-channel
   rate gives ~80%. Conclusion "<85% at the bar" is robust (even the most-optimistic inputs cap at
   85.87% via the wrong, quality-FAILING scheme); the point estimate is soft. **Settling measurement
   (env-blocked):** direct vLLM per-channel + down_proj-fp16 mixed-config prefill MFU at 700W —
   blocked by the llm-compressor↔vLLM `_C.abi3.so` ABI conflict (build per-channel weights offline,
   load into clean vLLM).
2. **"down_proj must stay fp16" is OVER-STRONG — it is "under PER-TOKEN activation scaling."** The
   over-bar margin is tiny (all-fp8 per-channel +0.37%, only 0.07pp over; down_proj ≈ +0.12pp). A
   **finer down_proj treatment NOT tested** — block/group (DeepSeek-V3-style 128-wide) activation
   scaling, or SmoothQuant migration of the silu(gate)·up channel-correlated outliers into the
   weights — could plausibly keep down_proj in FP8 within 0.3% → ~100% coverage → per-channel MFU
   88–98% → **flip to REACHED**. Dropping the whole down_proj (27%) is the coarsest fix, not minimal.
   **This is the single untested path most likely to change the reachability verdict.**

## Honest framing
- Against the **broad** lossless literature (<0.5%/<1% PPL, arXiv 2503.09975): full-coverage FP8 at
  +0.37% PPL IS "essentially lossless" (MMLU passes, true KL 0.0036) — so "92–98% MFU at <0.5% PPL"
  is a defensible reachability claim. But it **misses the SELF-LOCKED 0.3% bar** (locked precisely to
  forbid post-hoc relaxation) — **the bar is not relaxed.** At the locked 0.3%, it is FRONTIER.
- Dual denominator: 92–98% is vs the 989 bf16 yardstick (FP8 doing >bf16-peak bf16-equiv work);
  vs the FP8-native 1979 peak it is 46–49% (700W power wall). The "85%" is a bf16-reference metric.

**Reachability verdict: FRONTIER at the locked 0.3% bar (panel-confirmed); a finer down_proj scheme
is the untested path to a clean REACHED. NOTE: this is vLLM-executed FP8 — transparent CIPHER
delivery of it is BLOCKED (intercept bypass, separate report).** Anchors UNCHANGED, no rotation.
