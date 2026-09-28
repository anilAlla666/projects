# D.9 §1b — VALIDATION GATE (85%-reachable vs named-ceiling) — REPORT

**Date:** 2026-05-29. **Measure-only. Anchors UNCHANGED, no rotation, no production build.** KL=0
verified on every FP8-engaged probe by output-preservation (FP8 declines to fp16 where not
token-identical — that decline IS the finding). All MFU vs the **989 TFLOPS bf16/fp16 reference**
(the Goal-3 anchor's denominator; 700 W cap hard-locked). Sustained numbers only — no burst stacking.

**VERDICT: PARTIAL — lands at ~64% MFU; 85% NOT delivered under the zero-regression contract. The
binding constraint is the strict token-identity CONTRACT, not the hardware.** (Adversarially
verified by a 5-agent panel: 3 PARTIAL / 1 CEILING; completeness-critic synthesis = **PARTIAL@64%**.)

## §0 Provenance
Anchors: cipher_rt_phase4 `ed130e7`/`01d4effb`/`d7-rh1-close`; kmod loaded `0.7.0`; evidence
post-`4ee7236`. All runs CIPHER-OFF (injection `unset`; vLLM plugin disabled via `VLLM_PLUGINS=""`).

## §1 The three measured numbers (each kills an optimistic assumption)

**M1 — FP8 SUSTAINED (60 s clock-sampled, kills the burst-1.85× assumption):** at 700 W both dtypes
throttle to ~1300 MHz, so the per-clock tensor-core density ratio is preserved → **sustained
FP8/fp16 = 1.83× (1.82–1.84×) ≈ the burst 1.85×.** fp16 GEMM 65–66% MFU; **FP8 GEMM 118–122% of the
989 ref**. The ratio is real and clock-independent. (Against the FP8-native 1979 peak, FP8 GEMM is
~60% — still power-capped, the same 700 W wall as bf16.) `d9_s1b_fp8_sustained.json`.

**M2 — vLLM PREFILL baseline (kills the "50.9% HF-eager" baseline):** vLLM prefill MFU **63.2–65.3%**
across B=8/16 × S=2048/4096 (cudagraph ≈ eager-no-graph). This is **at the bf16 GEMM-proxy ceiling
(65%)** — vLLM already fuses the epilogue + graphs, so the HF-eager 50.9% (31% epilogue drag) was
**artifactually low**, and **L6 fusion has ~zero headroom on vLLM.** ⇒ the 64%→85% gap is entirely
the bf16 GEMM power wall, and **FP8 is the singular lever** (attention-TMA helps long-context only;
Marlin irrelevant to prefill). `d9_s1b_vllm_prefill.json`. *(First attempt gave impossible
800–1744% MFU — a prefix-cache hit artifact; fixed with `enable_prefix_caching=False` + distinct
prompts per call.)*

**M3 — FP8 OUTPUT-IDENTITY COVERAGE (kills the FP8 lever under zero-regression):** FP8 E4M3 (weight
+ activation quant), full-forward token-identity vs fp16 over 50 prompts (greedy + sampled):
| config | linears FP8 | greedy id | sampled id | result |
|---|---|---|---|---|
| ALL | 224 | 58% | 50% | DECLINE |
| FFN-only (per-tensor) | 96 | 66% | 64% | DECLINE |
| ATTN-only | 128 | 80% | 66% | DECLINE |
| FFN+ATTN layers≥4 | 196 | 62% | 56% | DECLINE |
| **FFN-only — PRODUCTION granularity** (per-channel W + per-token A, rowwise, **first-party**) | 96 | **68%** | 52% | **DECLINE** |
**No layer-class keeps the output token-identical → FP8-SAFE coverage = 0%** under the strict
contract. **The granularity confound is empirically CLOSED:** production-grade per-channel/per-token
scaling moved FFN only **66%→68% greedy (+2 pp)** — finer scaling fixes dynamic range, not E4M3's
~3-bit mantissa floor or greedy near-ties. `d9_s1b_fp8_coverage.json`, `d9_s1b_ffn_finegrain_verify.json`.

## §2 The arithmetic (sustained, no burst stacking)

The vLLM prefill is GEMM-dominated (~64%, epilogue fused). FP8 on a safe-coverage fraction **C** of
GEMM-time compresses it by the sustained ratio 1.83× → projected MFU-vs-989:
**`MFU(C) ≈ 64% / (1 − 0.454·C)`** (0.454 = 1 − 1/1.83).
- **85% requires C ≈ 0.54** (54% of GEMM-time FP8-safe).
- **Measured C = 0%** (M3) → `MFU = 64%`. Finegrain confirms C stays ~0% at production granularity.
- (At hypothetical C=1, MFU-vs-989 = 117% — FP8 beating the bf16 yardstick; see §4.)

**The measured coverage (0%) is ~54 points below the crossover. 85% is not delivered.**

## §3 Verdict — PARTIAL @ ~64% (panel-verified)

**PARTIAL.** Projected/achieved forward MFU stays at the **~64% vLLM baseline** because FP8 — the
singular lever — cannot engage under the zero-regression (token-identity) contract (coverage 0%,
robust to production-grade scaling). 85% is **not** reached.

- **It is PARTIAL, not CEILING (vs the 989 anchor):** the *hardware* clears 85% — FP8 sustains
  118–122% of 989; at hypothetical full coverage MFU-vs-989 ≈ 117%. So the wall is the
  **contract/coverage**, not a FLOP/J or numerics hardware ceiling. (The lone CEILING vote is valid
  only against the 1979 FP8-native denominator, which the Goal-3 anchor rejects — the critic flagged
  it as mislabeling a contract wall as hardware.)
- **The binding constraint is the strict zero-regression token-identity CONTRACT** — FP8 E4M3 flips
  Mistral-7B tokens on 20–50% of prompts even for a single layer-class, so output-preservation
  forces it to decline everywhere → the lever delivers ~0.

## §4 Denominator honesty (stated explicitly — it is the PARTIAL/CEILING fork)

- **vs 989 bf16 (the Goal-3 anchor, prior 67%=660/989):** PARTIAL@64% — coverage-limited; the
  hardware clears 85% but the contract blocks the lever. ≥85%-vs-989 via FP8 would be FP8 silicon
  beating a bf16 yardstick (throughput-equivalent-to-85%-bf16-MFU), legitimate only as the campaign's
  pre-set anchor.
- **vs 1979 FP8-native peak (honest per-dtype MFU):** 700 W caps fp16 at ~65% of 989 and FP8 at ~60%
  of 1979 → **85% true MFU is a hardware CEILING** regardless of coverage. The "85%" target is only
  reachable as a bf16-reference metric, which zero-regression blocks.

## §5 The decision for Anil + the one residual measurement

**Under the current strict KL=0 token-identity contract, 85% is NOT reachable on this H100: FP8 is
the only lever, and the contract forces its coverage to ~0%.** The fork:
1. **Hold strict KL=0 (zero-regression):** FP8 stays disabled → **PARTIAL@64%**; 85% is a named
   contract-induced wall (FP8 inherently lossy; bf16 alone is power-capped at 65%).
2. **The ONE untested path (recommended next measurement, NOT finer scaling — that's done):**
   re-score coverage under the contract's *literal* gate — **single forward-pass per-GEMM
   token-identity with a per-LAYER (not per-class) safe-set search** — instead of the strictly-harder
   24-token autoregressive sequence-identity gate used here (which compounds benign greedy near-tie
   flips into full DECLINEs). Outcomes: still-0% → locks PARTIAL@64% and hardens the wall as
   contract-induced; a harvestable subset → PARTIAL>64%; ≥~54% → REACHABLE-vs-989 (unlikely given the
   FFN-class 68% decline — FFN is 81% of GEMM-time and the least-passable class).
3. **Principal's gate choice is the real lever:** a bounded-quality gate (Δppl ≤ ε) — which you
   explicitly rejected — would make FP8 coverage high → REACHABLE-vs-989. 85%-reachability literally
   hinges on strict-token-identity vs bounded-quality. That is your call, not a measurement.

## §6 Discipline + carried findings
Measure-only; anchors UNCHANGED; no rotation; KL=0 preserved (FP8 declined to fp16 wherever not
token-identical). Two infra findings carried for V.1: the deployed image **forces CIPHER on**
(`unset` to disable, not `-e`); the deployed **vLLM-CIPHER KV plugin is broken in-container**
(`cipher_kv_bridge` Python 3.10/3.12 mismatch → vLLM engine init fails unless `VLLM_PLUGINS=""`).
Verdict adversarially verified (5-agent panel, `w8rlju3a9`: 3 PARTIAL/1 CEILING, critic PARTIAL@64%).

Artifacts: `d9_s1b_fp8_sustained.json`, `d9_s1b_vllm_prefill.json`, `d9_s1b_fp8_coverage.json`,
`d9_s1b_ffn_finegrain_verify.json`, the three probe scripts, panel output `w8rlju3a9`.
