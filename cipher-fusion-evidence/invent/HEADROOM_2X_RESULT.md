# HEADROOM-TO-2X SESSION — 2:4 REVERSED to LIVE via torchao (panel catch), DVFS honest split, detection coverage on FP8 (2026-06-11)

Continues PER_MODEL_BUNDLE_RESULT.md (the earlier SAME-DAY bundle session, ~16:40–17:45; "yesterday"
in drafts was wrong — panel fix). Protocol: H100, vLLM 0.20.2, cudagraph ON unless stated,
substrate NOT loaded (`VLLM_PLUGINS=` empty; s24 JSONs record `plugins_env`; chat-harness runs
evidenced by script exports + 0 "[cipher" lines in all retained stderr), `VLLM_DEEP_GEMM_WARMUP=skip`,
best-of-3, T=0. Power sampled at ~8 Hz effective (blocking nvidia-smi + 0.1 s sleep; ~8–17
samples/rep; cross-run tok/W spread ≤0.8% bounds the noise — panel-measured). Anchor `2edba0d2`
(cipher_rt_phase4/libcipher_rt.so) entry==exit. 5-agent adversarial panel: 6 material findings, all
applied; one REVERSED the 2:4 verdict (below).

## Integrity disclosures (panel-audited)
- **Prereg taint (P1 only):** PREREG_HEADROOM.md was written at 17:58:50 — 26 s AFTER cell A's
  result (incl. the `quant:None` observable P1 keys on) was on disk. P1 is therefore
  POST-HOC-TAINTED, not pre-registered (it is also over-determined: A/B, C/D, and the cell-F
  tombstone all agree). P2–P5 artifacts all postdate the prereg save (B 17:58:57; C–F 17:59–18:02;
  cov 18:07–08; dvfs 18:10–11). Scoreboard: **3 confirmed (P2/P4/P5), 1 tainted-but-confirmed
  (P1), 1 MISS (P3)**.
- **Contamination incident:** first cell batch ran without the env protocol → vLLM auto-loaded the
  cipher_v2 KV plugin; those runs died at init and were overwritten by the fixed re-run. The
  contamination failure mode is fatal init death, not perturbed numbers.
- **Provenance:** earlier-session dense@700W (4699/9.02) and EAGLE-only (6737/10.16) NREQ=32 JSONs
  were not retained (md-table-only). Re-measured this session: dense 4697.3 chat / 4710.0 raw;
  EAGLE-only 6742.5 (eagle_only_sat.log); bundle 8345.2; FP8 6634.8 (raw — cross-harness, delta
  +0.36% consistent with the measured +0.2–0.4% raw-vs-chat offset). bundle@300W re-run clean:
  6052.3/20.45 vs original 6061/20.41 (tpw_cap300_rerun.*). Still single-run stderr-less:
  tpw_cap400, spec_k7/k10. eagle_test2.py was edited between the tpw sweep and the dvfs runs;
  bridge measurement spec_k5 vs tpw_cap700 = −0.04% shows the edit is immaterial.

## 1. THE 2:4 STAGE — compressed-tensors path REMOVED upstream, but torchao online-quant path is
## LIVE: FP8+2:4 = 1.20x over FP8-dense, MEASURED (panel-driven reversal)

Saturated raw-prompt harness (s24_harness.py / s24g_harness.py), NREQ=32, 256 tok forced, same
prompts all cells, cudagraph ON except noted:

| cell | config | tok/s | tok/W | notes |
|---|---|---|---|---|
| A | Sparse-Llama-3.1-8B-2of4, bf16 | 4711.4 | 9.42 | dense kernels (quant None) |
| B | dense Llama-3.1-8B-Instruct, bf16 | 4710.0 | 9.18 | baseline |
| C | sparse + fp8 on-the-fly | 6600.5 | 14.40 | fp8 blind to zeros |
| D | dense + fp8 on-the-fly | 6634.8 | 14.05 | FP8-dense reference |
| E | sparse + EAGLE-3 (Instruct head) | 5897.5 | 9.92 | 1.25x/A; see P3 MISS |
| F | sparse via compressed-tensors sparsity_config | INIT FAIL | — | tombstone (below) |
| G1 | sparse via torchao SemiSparseWeightConfig bf16 | INIT FAIL | — | cuSPARSELt not capture-safe; eager variant crashes flash-attn (misaligned address) |
| **G2** | **sparse via torchao Float8DynActFloat8Weight PerRow + SPARSE_CUTLASS** | **7983.6** | **18.87** | **WORKS under cudagraph; mean power 423 W** |
| H | G2 + EAGLE-3 (Instruct head) | 13564.4† / 6413.3 | 28.8† / 12.14 | †degenerate-prompt run (s24_h.json): output "1.1.1…" is trivially draftable → inflated, DISCARDED as headline. Honest number = varied-prompt re-run 2026-06-12 (s24v_hv.json): **0.80x of G2-varied 8005.9 — EAGLE is NEGATIVE on this stack/workload**; see S24_CONTINUATION_RESULT.md |

- **P1 (tainted, over-determined): A/B = 1.0003** — a plain bf16 load runs DENSE kernels; the
  checkpoint is genuinely 2:4 (12 tensors incl. q/k/o across layers 0/15/31: 100% of 4-groups
  along the input/K dim ≤2 nonzeros, wrong-axis control 55–69%, zeros 0.5000 —
  verify_24_pattern.py/.json archived; panel independently reproduced).
- **P2 CONFIRMED: C/D = 0.995** — on-the-fly fp8 quantizes zeros like any value.
- **Cell F: compressed-tensors Sparse24 is a TOMBSTONE** — vLLM's scheme selection accepts the
  sparsity_config and routes to `CompressedTensors24`, whose every method raises
  `NotImplementedError("Sparse24 models are no longer supported by vLLM")` (0.20.2). The draft
  verdict "2:4 DEAD on this stack, engine work required" was a UNIVERSAL NEGATIVE the panel
  refuted by locating the in-tree torchao route — which we then measured:
- **Cell G2 (THE REVERSAL): vLLM's torchao integration online-quantizes the plain bf16 2:4
  checkpoint to FP8-PerRow + SPARSE_CUTLASS packing (cutlass 2:4 fp8 kernels, capture-safe).
  7983.6 tok/s / 18.87 tok/W = 1.203x / 1.343x over FP8-dense (D), = 1.695x / 2.06x over dense
  bf16 (B) — no EAGLE, no power cap.** Mean power DROPS to 423 W (vs D 472 W): sparse tensor-core
  GEMMs are faster AND cooler. Mechanics that were required (disclosed): config.json with
  quant_method "torchao" + serialized quant_type; a 3-line monkeypatch forcing
  `is_checkpoint_torchao_serialized=False` (vLLM marks any "torchao" quant_method as
  pre-serialized; ours is plain bf16 needing ONLINE quantize_); a `__main__` guard (torchao's
  presence makes vLLM's import chain initialize CUDA in the parent → forces spawn). G2 quality is
  NOT validated: output on this prompt is degenerate base-model repetition (same as A/C/E) and
  PPL was not measured; the neuralmagic recovery claims + FP8-PerRow are plausible-but-untested
  here.
- **bf16 2:4 (G1) stays WALLED**: cuSPARSELt descriptor init is rejected during cudagraph capture
  ("operation not supported"); the eager variant crashes in vllm-flash-attn (misaligned address).
  The live route is specifically FP8+2:4 via cutlass.
- **P3 MISS (predicted ≤1.0x): E/A = 1.25x.** The Instruct-trained EAGLE-3 head partially
  transfers to the pruned base. TWO confounds (panel): (a) degenerate repetitive output is
  trivially draftable — acceptance plausibly inflated (rate not extracted); (b) the 1.25x (raw
  harness) vs 1.43x (chat harness) comparison crosses harnesses. Do not generalize.

**Net consequence:** the bundle's "+2:4" headroom line is REAL on this stack after all — but ONLY
through the torchao FP8+2:4 route, and quality validation (PPL/benchmarks on the FP8-quantized
sparse model) is the gating next step before it joins the bundle.

## 2. DVFS HONEST SPLIT — 2.25–2.26x tok/W = 1.47x (cap, any model) x 1.54x (bundle), P4 CONFIRMED

Dense bf16 measured at the same caps as the earlier-session bundle tpw sweep (identical chat
harness config; eagle_test2.py edit bridged at −0.04%; power limit verified reset to 700 W after):

| cap | dense tok/s | dense tok/W | bundle tok/s | bundle tok/W | bundle/dense tok/W @cap |
|---|---|---|---|---|---|
| 700W | 4699 (4697.3 retained) | 9.02 (9.07) | 8353 | 14.24 | 1.58x |
| 400W | 4480.8 | 11.25 | 7149.6 | 18.17 | 1.62x |
| 300W | 3956.4 | 13.24 | 6061 (6052.3 retained) | 20.41 (20.45) | 1.54x |

- **Attribution of bundle@300W vs dense@700W (2.26x on md-only denominators; 2.25x on retained
  artifacts): 1.468x from the power cap (dense gets this too) x 1.542x from the bundle at matched
  cap.** Bundle multiplier roughly cap-invariant (1.54–1.62x) — FP8+EAGLE and DVFS compose cleanly.
- Throughput cost of 300W: dense −15.8%, bundle −27.4%. **Deployment frame: bundle@300W serves
  1.29x MORE than dense@700W at ~2.25x fewer joules/token.**
- **Goal-status (panel fix — the draft overclaimed):** "2x tok/W" is met ONLY in this cross-budget
  deployment frame. At MATCHED power the bundle-attributable multiplier is **1.54–1.62x — below
  2x at every cap tested**. If the founding goal requires matched-power attribution, it is NOT yet
  met by FP8+EAGLE alone. (G2's 2:4 stage adds 1.34x tok/W over FP8-dense uncapped — the matched-
  power 2x path now plausibly = FP8 x 2:4 x DVFS-at-cap, unmeasured as a triple.)

## 3. DETECTION COVERAGE ON THE FP8 BUNDLE — P5 CONFIRMED (substantively): the cuBLAS seam is
## BLIND to the FP8 target's decoder linears

Count-only LD_PRELOAD shim (rvcount.c; dlopen("libcublas.so.13") passthrough — dlsym(RTLD_NEXT)
segfaults because torch dlopens cuBLAS RTLD_LOCAL), counting GemmEx / StridedBatched / Lt /
Batched + (m,n,k) histogram for GemmEx. Chat harness, NREQ=32. Counts are FLOORS from the last
periodic dump (every 2048 GemmEx; final destructor dump lost to engine SIGKILL); panel verified
sum(shapes)==counter exactly in both dumps (no drops; 239/256 slots used — raise the cap if reused).

| run | tok/s (unshimmed ref) | GemmEx | Lt/Strided/Batched | seam sees |
|---|---|---|---|---|
| dense bf16 control | 4697.3 (4699 → free) | ≥26,624 | 0 | 128 decoder linears/forward, all 32 layers, every capture size. lm_head crossed ONCE (init profile, n=1024) — under cudagraph serving the dense seam does NOT see lm_head |
| FP8+EAGLE bundle | 8345.2 (8355 → free) | ≥4,096 | 0 | ONLY the bf16 EAGLE-3 drafter (k=8192 concat-qkv, k=12288 feature-fc, one decoder layer, 32000-entry draft-vocab head, partly EAGER: 813 draft-head calls at n=32) + the target's bf16 lm_head at verify sizes (n=192=32x6, 160+ calls) |

Attribution (panel fix — now explicit): the ambiguous-dim shapes ([4096,·,4096], [28672,·,4096],
[4096,·,14336]) are the DRAFTER's because their counts equal the drafter-unique k=8192 qkv count
EXACTLY (618 each, one-layer cadence vs the dense control's 32-layer multiplicity) and target qkv
[6144,·,4096] is ENTIRELY ABSENT. Prereg P5's literal criterion ("no m=28672 k=4096") is
technically violated by these drafter calls — the substantive prediction (target FP8 linears
invisible) is what's confirmed. The cutlass attribution is corroborated by source: vLLM 0.20.2
sm90 FP8 kernel selection = CutlassFP8ScaledMMLinearKernel → vllm._C (Marlin gated cc≥89-only-if-
forced, FlashInfer cc≥100), and cuBLAS classic API has no FP8 types (Lt-only). Caveat (panel): the
Lt/Strided/Batched hooks never fired in ANY run — no liveness control; a torch._scaled_mm
microbench under the shim is the cheap next-session control.

**Coverage arithmetic:** target lm_head = 1.051 GF of 15.01 GF target weight-FLOPs/token = 7.0%
covered; **~93% of TARGET weight-FLOPs uncovered (~89% bundle-wide if the covered bf16 drafter is
included per spec round)**. Since EAGLE verification is exact, drafter SDC corrupts throughput,
not output — output-integrity coverage ≈ lm_head only.

**Consequence (first-class product finding):** the detection/reliability runtime — the substrate's
driver-level role on co-designed bundles — does NOT transfer to FP8 models as built. Fork-1
detector + the −3.9% coop hook protect bf16/fp16 cuBLAS-dispatched models only. FP8 bundles (and
the G2 torchao route, also non-cuBLAS) need a cutlass-epilogue or vLLM-coop quantized-linear seam.
A3 compose-cost on the bundle is MOOT until that seam exists. Six-goals impact: see the dated
ADDENDUM appended to PER_MODEL_BUNDLE_RESULT.md (detection / reliability-SDC / goodput rows revert
to NEEDS-NEW-SEAM for FP8 bundles; TPW row → 1.54–1.62x matched-cap).

## 4. REMAINING HEADROOM — updated after the reversal

- **2:4 sparsity: LIVE via torchao FP8+2:4 (G2: 1.20x over FP8-dense), gated on quality
  validation.** bf16 2:4 walled (capture + eager crashes). compressed-tensors path dead upstream.
- **Deeper speculation: DEAD.** spec_k sweep: k=5 → 8349 (≈default 8355), k=7 −4.8%, k=10 −34.6%.
- **LIVE: EAGLE head co-trained on the FP8(+2:4) base** (cluster days, the per-model pipeline),
  **phase-aware DVFS**, **sparse-attention/long-context** (untested; short-context loads here).
- **2x throughput at saturation:** FP8+EAGLE = 1.78x (chat workload; later found DOUBLE-BOS-
  depressed — corrected single-BOS ≈2.14x, see S24_CONTINUATION_RESULT.md §2); FP8+2:4 = 1.70x (no EAGLE);
  the triple FP8+2:4+EAGLE measured 2026-06-12 = 6413.3 = 0.80x of FP8+2:4 alone — EAGLE NEGATIVE
  on raw-continuation traffic at saturation (acceptance 0.83 tok/step, break-even ≈1.2; see
  S24_CONTINUATION_RESULT.md). 2x tok/W at matched power: plausibly FP8 x 2:4 x DVFS, unmeasured
  as a triple.

## VERDICT
The panel reversed this session's biggest negative: **2:4 is NOT dead on this stack — vLLM's
torchao online-quant route serves the pruned checkpoint as FP8-PerRow + cutlass 2:4 at 1.20x
throughput / 1.34x tok/W over FP8-dense (1.70x/2.06x over dense bf16), under cudagraph,
quality-unvalidated.** DVFS honest split: 2.25–2.26x tok/W = 1.47x cap (any model) x 1.54x bundle;
at matched power the bundle is 1.54–1.62x — "2x tok/W" holds only in the deployment frame.
Detection coverage finding stands and now also covers G2: every efficient serving path for this
bundle (FP8 cutlass, torchao 2:4) bypasses the cuBLAS seam — **the reliability runtime must move
to a cutlass/vLLM-coop seam to compose with the very bundles that deliver the efficiency goals.**
Prereg integrity: P1 tainted (disclosed), P3 missed (kept, two confounds), P2/P4/P5 confirmed.
Next gates: (1) quality validation of the torchao FP8+2:4 model (PPL/benchmarks), (2) EAGLE head
co-trained on the FP8(+2:4) base, (3) the new detection seam. Anil's call.
