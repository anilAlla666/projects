# CIPHER TRAINING-MFU LIFT — DRIVER-LEVEL FP8 TRAINING SUBSTITUTION (2026-06-10, panel-revised 06-11)

## VERDICT: **NO LIFT from the FP8 actuator AS-BUILT.** It is **net-negative for training-step MFU** (−25.0 % default-clock full-window / −23.7 % steady-state / −25.5 % iso-clock) **AND fails the convergence gate.** Post-panel probes show **both walls are measured implementation artifacts of the actuator, not FP8-fundamental walls**: (1) the slowdown is dominated by a per-step weight-requant churn from pointer-only cache keying (measured, reconstructs ~116 of the ~150 ms/step delta), and (2) the convergence FAIL is caused by **mathematically wrong backward GEMMs** (engine ignores transa/lda; measured rel-err √2 vs ground truth), **not** by FP8 precision. Surfaced to Anil; no successor built.

Co-measured on ONE config (real LoRA-finetune Mistral-7B, B=2×seq=2048): the FP8 actuator engages
(counter-proven), but slows the training step and corrupts the backward pass. The honest co-measured numbers
are the deliverable; the target was re-evidenced (literature frontier ~40–54 % BF16 is optimized full-pretrain —
a different regime than this finetune; imported, not measured here).

**Frozen anchor `libcipher_rt.so` md5 `2edba0d2136f8ede4713d90a8f7cd55f` — entry == exit, byte-identical, NEVER
rebuilt. No successor artifact was produced** (Phase 1 tax-fix turned out unnecessary at realistic batch; Phase 2
FP8 turned out net-negative as-built — nothing to ship). Default clocks except the explicit iso-clock pair.
MFU = analytic FLOPs ÷ 989.5 TFLOP/s (H100 SXM FP16/BF16 dense peak); training step = **4·N** (LoRA frozen base:
fwd 2N + input-grad 2N; no base weight-grad; adapter <1 %) — a **finetune**, not a pretrain proxy (full pretrain
would be 6N). 4N is applied to total params incl. embedding (+~1.8 % FLOP overcount) and excludes attention
matmuls (−6–7 %) — net conservative, and common to both arms of every comparison.

---

## PHASE 0 — BASELINE + TAX + GEMM-SHARE + REACHABILITY

**P0a — the real training-step baseline we never had:** real LoRA-finetune Mistral-7B, **B=2×2048, sdpa, AdamW**.
**MFU = 24.6 %** (`baseline_B2.json`, 50 steps; 1965 MHz sw-power-capped, 575 W, 55.6 GB); stability measured
separately over 100 steps: loss 0.055→0.0018, 0 NaN (`conv_baseline.json`, MFU 0.2468 — replicates). Far above the
prior toy B=1×512 (prior lane reported 10.7 %; re-measured in-dir at 11.6 %, `sweep_van_B1s512.json`) because
M=B·seq=4096 makes the linear GEMMs efficient. B=4 OOMs (real `torch.OutOfMemoryError`, `baseline_B4.err`); full
finetune + Adam does not fit 80 GB — so B=2×2048 LoRA is the largest fitting realistic config, stated as a finetune.

**P0b — the launch-density tax is a SMALL-BATCH ARTIFACT (key finding):**

| Config | M=B·seq | vanilla MFU | obs-only MFU | tax | exposed overhead |
|---|---|---|---|---|---|
| B=1×512 (Lane-1's toy) | 512 | 0.116 | 0.044 | **+162.7 %** | 210 ms/step |
| B=1×2048 | 2048 | 0.232 | 0.180 | +28.9 % | 75 ms/step |
| **B=2×2048 (realistic)** | 4096 | 0.246 | 0.239 | **+3.1 %** | 15 ms/step |

Substrate confirmed firing in all three (23,634 / 20,331 / 20,349 `[cipher_v2]` lines; FP8 actuator confirmed OFF in
all tax arms — "actuator DISABLED" in-log, counters 0). **Lane-1's "+150 % tax gates training" was measured at toy
B=1×512; at realistic training batch the tax is 3.1 %.** Mechanism, panel-corrected: the **absolute exposed
overhead itself collapses 210→75→15 ms/step** (a 14× drop — the constant-cost-amortization story is REFUTED by
these medians; a constant 210 ms would predict +43 % at B=2×2048, not the measured +3.1 %). The likely mechanism —
host-side per-launch work hiding under longer GPU-bound kernels (corroborated: 465 ms/step kernel time vs 487 ms
wall at B=2×2048) — is a **hypothesis, not isolated by measurement**. Do not extrapolate tax at other configs from
a constant-cost model. ⇒ **Phase 1 (CUPTI-gate tax-fix rebuild) NOT needed at realistic batch**; FP8 was engaged on
the frozen anchor directly. (Separability of CUPTI plane from the cuBLAS-GOT/FP8 plane remains proven in source;
a gate stays the scoped next build only if launch-bound small-batch training viability is ever needed.)

**P0c — training GEMM-share is LOW (45.5 %), not high:** GEMM 45.5 % / ATTENTION 4.1 % / **POINTWISE 49.1 %** /
OTHER 1.2 % (vanilla arm, `train_gemmshare.json`; name-based bucketing). Pointwise = `direct_copy`/`float16_copy`
(fp32-LoRA↔fp16-base casts) + `add/mul` (residuals, AdamW foreach) — HF-eager + fp32-adapter overhead, unfused.
~7 % of the GEMM bucket is fp32 adapter FFMA outside FP8's gate ⇒ FP8-addressable share ≈ 39 %. FP8 substitution
is Amdahl-limited here.

**P0d — reachability: GEMM-share-limited.** FP8 at a literature 1.3–1.5× training-GEMM speedup (NeMo-class
recipes; imported, not measured) projects e2e MFU 0.246→0.275–0.290 gross / 0.267–0.281 net-of-3.1 %-tax —
**+2–3.5 pts**, far short of the **literature frontier ~40–54 %** (Megatron ~47 %, MosaicML >50 %, SemiAnalysis
~54 % — optimized **full-pretrain** stacks with fused optimizer/norms/FSDP and a 6N FLOP model; not directly
comparable to a 4N HF-eager LoRA finetune). The 24.6 % baseline is below frontier because of the unfused
HF-eager/LoRA-fp32 harness, not GEMM inefficiency (GEMM-bucket-time MFU ≈ 0.246/0.455 ≈ 0.54). Pre-registered
BEFORE Phase 2 as a modest, non-frontier-clearing lift; proceeded to measure the real number.

## PHASE 1 — NOT BUILT (tax-fix unnecessary at realistic batch)

P0b showed the 3.1 % realistic-batch tax is not the gating problem Lane-1's toy measurement implied. Building a
successor to fix a 3 % tax would solve a non-problem. Per no-auto-degrade, **no successor was built; the frozen
anchor is used directly.**

## PHASE 2 — FP8 ENGAGED + MFU (the measurement)

Engaged on the frozen anchor, B=2×2048, `CIPHER_FP8=1` via `CUDA_INJECTION64_PATH`, `CIPHER_VOLT=off`,
`VLLM_PLUGINS=""`. **Engagement proven by counters** (exact across all runs: total=447·S, handled=288·S−66,
skipped=159·S+66 for S steps): `fp8_B2_default.json` handled=16,638 / total=25,926; warm20 17,214 / 26,820.

**The engaged set (panel-corrected from the err logs — the report previously mis-imported the vLLM fused-qkv
"6144" shape):** the six ENGAGED (out,in) shapes are q/o 4096×4096, k/v 1024×4096 + 4096×1024 (grad), gate/up
14336×4096, down 4096×14336, lm_head-grad 4096×32000. Steady-state handled = **all 222 backward input-grad GEMMs
+ forward q/o (64) + forward layer-0 k/v (2)**. Forward gate/up/down, k/v layers 1–31, and lm_head — **the entire
forward MLP — were NEVER substituted** (159 calls/step perpetually skipped). Cause (source-verified):
`observe_weight` keys the weight cache on pointer only and **resets the slot whenever the same pointer arrives
with swapped (K,N)** — which the fwd/bwd alternation does every step for every non-square weight ⇒ **159 weights
re-prequantized EVERY step** (each: 2×cudaMalloc + absmax/quant kernels + host-blocking cudaStreamSynchronize +
2×cudaFree). FLOP coverage of handled calls: **57.5 % of eligible GEMM FLOPs total; forward-only 15.2 %.**
(The earlier "9,288 weight prequant events" = the skipped counter = this per-step churn (159×58 + 66 one-time),
**not** a one-time startup cost. The earlier "prequant moved fully into warmup (0 in-window prequant)" claim is
**RETRACTED** — warm20's err shows no prequant lines only because that logging is gated on `CIPHER_FP8_VERBOSE`,
and warm20's skipped counter (159·S+66) proves the churn continued every step.)

| Regime | run | vanilla MFU | FP8 MFU | step-time Δ | MFU Δ | clock |
|---|---|---|---|---|---|---|
| default-clock, full window | `fp8_B2_default` | 0.246 | **0.185** | +33.4 % | **−25.0 %** | van 1965 (sw-cap) / fp8 1980, 463 W |
| default-clock, steady-state (longer warmup) | `fp8_B2_warm20` | 0.246 | **0.188** | +31.0 % | **−23.7 %** | fp8 1980, 470 W |
| iso-clock (both pinned, achieved median 1830; van dips 1800 under power-cap — conservative) | `isoT_*` | 0.243 | **0.181** | +34.2 % | **−25.5 %** | 1830 / 1830 |

**FP8 as-built is net-NEGATIVE in all three measurements.** The full-window vs steady-state pair shows startup
explains only ~2 pts of the +33 % (the prior report cited only the more favorable warm20 number as the headline —
fixed; the unreported full-window run is WORSE for FP8, so NO-LIFT strengthens). Default clock is FP8's best case
(110 W power relief, clocks 1965→1980) and it still loses; iso-clock loses more. Note: these Δs are vanilla vs
FP8-engaged, i.e. **inclusive of the 3.1 % hosting tax** (not "net of" it); actuator-only vs obs-only host =
−21.3 %. Warmup/window per run: baseline/sweeps 8 warm + 50 (40 sweep) measured; warm20 = 20 warm + 40 measured.

**MECHANISM (measured post-panel, `probeB_*.json` — torch.profiler under the substrate returns 0 events (CUPTI
client conflict, `profile_fp8.err`), so attribution is bottom-up per-call timing, counter-labeled):**
- The **steady handled path is NOT the problem**: a never-thrashing square weight (q/o) costs +0.02 ms/pair —
  per-call activation-quant piled on the step was the previously asserted mechanism and is **REFUTED as the
  dominant cost** (handled fwd q/o is actually −9 % faster than fp16).
- The **dominant cost is the requant churn** on the 159 thrashing weight-pairs/step: +0.1–1.2 ms/pair (noisy
  per-shape; malloc/sync-bound, not size-bound). Per-call deltas × per-step counts **reconstruct ~116 ms of the
  measured +150–176 ms/step**; + ~15 ms obs-plane tax ≈ the observed delta. An implementation artifact
  (pointer-only cache keying), not an FP8 cost.
- **FP8-Lt GEMM quality per shape is MIXED** (first-heuristic algo, no autotune): handled fwd gate_up +82 %
  *slower* than fp16 nvjet, lm_head −47 % *faster*, q/o −9 % faster. So even churn-free, net sign at these shapes
  is not established.

## PHASE 3 — TRAINING CONVERGENCE — **FAIL, root cause = WRONG MATH (measured), not FP8 precision**

100-step loss curves, same `manual_seed(0)`, same fixed batch, same lr=1e-5/clip=1.0, FP8 differs only by caller env:

| Step | baseline | FP8 |
|---|---|---|
| 1 | 0.055 | 0.065 |
| 25 | 0.007 | 0.065 |
| 50 | 0.004 | 0.066 |
| 75 | 0.002 | 0.072 |
| 100 | **0.0018** | **0.3335** |

Baseline memorizes smoothly. FP8 **never learns** (never beats its step-1 loss), then loss rises monotonically at
an accelerating rate to 5× start by step 100 (0 NaN). **Quality verdict: FAIL** for this actuator.

**Root cause MEASURED post-panel (`probeA_*.json`):** the shim records transa/transb/lda but the engine **never
reads them** — it hardcodes TRANSA=T/TRANSB=N with lda=k (the forward-linear layout). Backward input-grad GEMMs
use a different layout, and **every FP8-handled backward GEMM returns rel-err 1.414 (=√2) vs fp32 ground truth**
— i.e. output uncorrelated with the true value: **garbage gradients**, while handled forward calls show 0.0375
(normal FP8 quant noise). Since steady-state handled = ALL 222 bwd input-grads, every base-path gradient was
corrupt all along; lr=1e-5 + clip kept it bounded, hence stall-then-drift, and the FP8-arm MFU above was measured
on a numerically wrong backward pass. **The prior attribution ("per-tensor E4M3 too coarse for gradients, ~FP22
accumulation; blockwise FP8 required") is RETRACTED** — it was literature-imported, and the measured failure is a
layout bug, so this run provides **no evidence about FP8 numerics recipes at all**. Caveat kept from the panel:
the proxy is a fixed-batch memorization test (dynamic range 0.065→0.002, near-zero-loss regime), not a real-data
convergence test — sufficient to fail the actuator as-built, not to qualify any recipe.

## PHASE 4 — CO-MEASURED VERDICT

**Training-MFU lift NOT delivered by the FP8 actuator as-built.** On one config (real LoRA-finetune Mistral-7B
B=2×2048): baseline **24.6 %** → FP8-engaged **18.5 % default-clock full-window / 18.8 % steady-state / 18.1 %
iso-clock** (−25.0 / −23.7 / −25.5 %, inclusive of the 3.1 % hosting tax; actuator-only −21.3 %),
`fp8_calls_handled=16,638` (57.5 % of eligible GEMM FLOPs; fwd-only 15.2 %), **convergence FAIL** (never learns;
final loss 0.3335 vs 0.0018), frozen anchor unchanged, no successor. **Target re-evidenced to a literature 40–54 %
full-pretrain frontier (different regime, not directly comparable); FP8 as-built moves AWAY from the baseline,
not toward any target.**

**Two walls, both MEASURED as actuator implementation artifacts (panel + probes):**
1. **MFU wall** = per-step weight-requant churn from pointer-only cache keying (reconstructs ~116 of ~150 ms/step)
   + mixed per-shape FP8-Lt GEMM quality. NOT per-call act-quant (refuted), NOT FP8-fundamental.
2. **Quality wall** = wrong-math backward GEMMs from ignored transa/lda (rel-err √2, measured). NOT FP8 precision —
   no recipe conclusion (blockwise-vs-per-tensor) can be drawn from this run.

**Options for Anil (no silent scope change):**
1. **Fix the actuator (substrate-internal successor, scoped):** key the weight cache on (pointer, K, N) and
   **refuse non-(T,N) layouts** (bwd falls back to fp16 — correct by construction) + FP8-Lt algo autotune. Removes
   both walls' causes. The probes do NOT establish the fixed actuator wins: fwd-only FLOP coverage is 15.2–39 %
   (Amdahl ceiling +2–3.5 pts) and per-shape FP8 GEMM quality is mixed (gate_up fwd +82 % slower). Cheap to build;
   honest expectation: roughly neutral, decided by re-measurement.
2. **The real training-MFU ceiling here is the 49 % pointwise** (fp32-LoRA casts, HF-eager unfused elementwise,
   AdamW) — the lever is a fused training stack, **out of GEMM-intercept substrate scope**.
3. **Hosting viability is the good news** (independent of FP8): at this config the substrate hosting tax is
   **3.1 %** (not the toy 150 %) — the substrate can host realistic-batch training; the FP8 actuator as-built is
   the wrong tool, not the hosting.
4. Convergence qualification of ANY FP8 recipe (per-tensor delayed-scaling E4M3/E5M2, blockwise, fwd-only) remains
   **unmeasured here** — this run cannot rank recipes; a fixed-layout successor + real-data eval would be needed.

**Fleet note:** single-GPU result. Fleet-wide training adds the NCCL comms surface (all-reduce/all-gather, the
W.7 substep) — outside this GEMM-intercept build; not claimed.

## Integrity
- Frozen anchor md5 entry == exit `2edba0d2136f8ede4713d90a8f7cd55f` (re-verified 06-11 post-probes); never
  rebuilt; no successor; no framework/training-loop monkeypatch; engagement only via caller env
  (`CUDA_INJECTION64_PATH` + `CIPHER_FP8`); `VLLM_PLUGINS=""`; harness never sets FP8 env (panel-verified).
- Quality measured before any headline MFU; both clock regimes; iso pair conservative-direction (van power-cap
  dips); clocks reset to defaults after the iso pair (verified 1980 == default app clocks).
- Every MEASURED number → a `train_mfu/` JSON (baseline_B2, baseline_B4.err, sweep_*, tax_obsonly_B2,
  train_gemmshare, p0d_reachability, fp8_B2_default, fp8_B2_warm20, isoT_*, conv_*, convergence_verdict,
  probeA_*, probeB_*, fp8_arm_profile, train_mfu_verdict). Imported references are labeled as such (prior-lane
  10.7 %, frontier 40–54 %, NeMo 1.3–1.5×). Engaged runs exit with the known at-exit segfault AFTER results are
  written (exit=139; JSONs+counters intact, same behavior across all engaged runs).
- Known weaknesses kept visible: convergence proxy is fixed-batch memorization; tax-collapse mechanism is
  hypothesis-level; per-shape probe timings are single-run noisy (aggregate used); FP8-arm MFU measured on a
  wrong backward pass (inherent to as-built actuator).

## Verification panel (2026-06-11) — run AFTER the first report draft; all material fixes applied above
4-agent adversarial panel (lenses: numbers-recompute / methodology-confounds / overclaim-honesty /
engagement-mechanism), 102 tool-use verification steps. **Every headline number reproduced exactly from the raw
JSONs** (baseline 24.6 %, tax 162.7→28.9→3.1 %, FP8 −24/−25 %, loss table element-exact, counters exact-modeled
447·S/288·S−66/159·S+66, anchor md5). **12 material findings — all applied**, the big four:
1. Headline default-clock row silently used the favorable warm20 run while taking power/counters from the other
   run → table now shows both runs, full-window first (M1).
2. "Prequant moved to warmup / one-time 9,288 prequant events" wrong → it is a per-step 159-weight requant churn;
   claim retracted, churn disclosed as first-class (M2, M10).
3. Engaged-shape list partly imported from vLLM memory ("qkv 6144" doesn't exist in training); forward MLP never
   substituted; FLOP coverage 57.5 %/15.2 % added (M9). Tax constant-cost-amortization mechanism refuted by own
   medians; rewritten as measured-collapse + hypothesis (M3, M8). "Net of tax" → "inclusive of tax" (M7).
4. Both wall MECHANISMS were inferred/imported, not measured (M5, M6, M11, M12) → **two decisive probes run**:
   Probe A (per-call correctness): bwd rel-err √2 ⇒ wrong math, convergence attribution rewritten, blockwise
   claim retracted. Probe B2 (per-call timing, profiler-free): churn-dominated slowdown, act-quant mechanism
   refuted, mixed FP8-GEMM quality disclosed. Options for Anil rewritten accordingly.
12 of 24 minor findings edited in (run provenance per table row, frontier/NeMo labeled as literature, iso-clock
van dips disclosed, B=4-OOM artifact cited, warmup table, 4N overcount note, addressable-share ≈39 % footnote,
"diverges"→"never learns then accelerating rise", prior 10.7 % labeled, eager-OOM claim replaced, Option
qualifications, every-number scoping); the rest are logging/process improvements for a successor build (e.g.
record pin_mhz in JSON, md5-at-entry capture in-harness, counter-snapshot windows).
