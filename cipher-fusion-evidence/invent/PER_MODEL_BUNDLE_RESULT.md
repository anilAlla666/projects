# PER-MODEL BUNDLE — the per-model co-design thesis, MEASURED (2026-06-11)

Built the staged per-model bundle on Llama-3.1-8B-Instruct (the model with public co-designed artifacts; Mistral-7B
would need its own EAGLE head + 2:4 produced = the same pipeline). Real vLLM, H100, cudagraph ON, SATURATED load
(NREQ=32 chat, 256 tok), best-of-3, power sampled. Anchor 2edba0d2 entry==exit; substrate NOT loaded (efficiency
stages are vLLM serving the co-designed bundle; the substrate's driver-level role = detection/reliability/hosting).

## THE BUNDLE (measured, quality-preserved, at saturated load vs dense bf16)
| stage | tok/s | x | tok/W | x | MFU | coherent |
|---|---|---|---|---|---|---|
| dense bf16 | 4699 | 1.00 | 9.02 | 1.00 | 7.6% | yes |
| +EAGLE-3 (trained drafter) | 6737 | 1.43 | 10.16 | 1.13 | 11.0% | yes |
| +FP8 | 6611 | 1.41 | 13.90 | 1.54 | 10.8% | yes |
| **FP8+EAGLE-3 (BUNDLE)** | **8355** | **1.78** | **14.16** | **1.57** | **13.6%** | **yes (exact)** |

Low-batch decode (batch-2 chat): dense 316.6 -> +EAGLE 505.6 = **1.60x**, output identical (EAGLE is exact greedy).

## THE KEY FINDINGS
1. **EAGLE-3 (a TRAINED per-model drafter) HOLDS UNDER SATURATED LOAD: 1.43x @NREQ=32** — overturns the earlier
   stress-test "spec dies under load." That was ngram (low acceptance, model-agnostic); EAGLE's acceptance is high
   enough that verify FLOPs pay off even when compute-bound. THIS is the per-model thesis: a trained drafter
   delivers the MFU/idle-compute lift the Jacobi probe proved you need a drafter for — and it works at saturation.
2. **The bundle composes: FP8 (memory/MBU/TPW) x EAGLE (idle-compute/MFU) = 1.78x throughput / 1.57x tok/W, quality
   preserved.** MFU lifts 7.6% -> 13.6% (1.78x) at saturated load — the founding MFU goal, delivered via co-design.
3. Only TWO stages so far. Headroom not yet added: 2:4 sparsity (+~1.2x prefill, checkpoint downloaded), EAGLE head
   co-trained on the FP8 model (higher acceptance), phase-aware DVFS (+tok/W), sparse-attention/LSA (long-context).
   Trajectory clears ~2x and goes beyond at long context.

## THE SIX GOALS — all addressed
| goal | how | status |
|---|---|---|
| **MFU** | FP8 (fewer bytes) x EAGLE (more tokens/weight-stream via trained drafter, uses idle tensor cores) | **1.78x MEASURED** (7.6->13.6%, saturated) |
| **MBU** | FP8 = half the model bytes/token | **1.41x MEASURED** (the FP8 stage) |
| **TPW** | FP8 + (composable) phase-aware DVFS | **1.57x MEASURED**; +DVFS adds more |
| **Failure detection** | SDC GEMM-interception checker (coop hook) | BUILT, -3.9% cudagraph cost (cipher-ra-coop); compose-cost on bundle = next |
| **Reliability** | per-GPU SDC+cohort+NVML verdict | BUILT (cipher-rc-pergpu); driver-level |
| **Goodput** | garbage-elimination (R.A+R.B) + EAGLE is EXACT (no quality loss = useful goodput) | BUILT; EAGLE preserves output exactly |

## HONEST CAVEATS
- Model = Llama-3.1-8B (public artifacts exist). Mistral-7B / any model = run the same pipeline: train an EAGLE
  head (~days cluster) + prune+recover 2:4 + calibrate FP8. That offline per-model work is the cost; it amortizes
  over serving volume (worth it for the top ~5-20 models).
- The efficiency stages are vLLM serving the co-designed checkpoint+head. The DRIVER-LEVEL substrate role is the
  detection/reliability/orchestration + transparent hosting -- NOT the source of the 1.78x (that's the bundle).
  Honest split: co-design delivers the efficiency; the substrate delivers safety + cross-tenant + zero-touch hosting.
- detection/reliability/goodput are BUILT+measured separately; their COMPOSED cost stacked on FP8+EAGLE+cudagraph
  is the next measurement (the coop hook vs spec-decode cudagraph interaction).
- EAGLE head co-trained on FP8 base would compose cleaner than the bf16 head on FP8 (still coherent here).

## VERDICT
The per-model co-design thesis is VALIDATED on our hardware: **1.78x throughput / 1.57x tok/W / 1.78x MFU at
saturated load, quality preserved**, from FP8 + a trained EAGLE-3 drafter — with clear headroom to ~2x+. This is
the first measured result that delivers the founding MFU goal, and it confirms the strategy: CIPHER ships per-model
co-designed bundles (the substrate hosts them + adds reliability), NOT a universal frozen-model trick.

---
## ADDENDUM (2026-06-11, headroom session — see HEADROOM_2X_RESULT.md)
The six-goals table above is AMENDED by measurement:
- **Failure detection / Reliability (SDC component) / Goodput (garbage-elimination)**: these rows
  were satisfied by cuBLAS-seam components. Measured on THIS bundle: the seam sees only the bf16
  drafter + lm_head (~7% of target weight-FLOPs; all 32 FP8 decoder layers invisible — cutlass
  kernels). For the FP8 bundle these rows revert from BUILT to NEEDS-NEW-SEAM (cutlass-epilogue /
  vLLM-coop at the quantized-linear level). "All 6 goals addressed" holds for bf16/fp16
  cuBLAS-dispatched models only.
- **TPW**: updates to 1.54–1.62x bundle-attributable at matched power cap; 2.26x only in the
  cross-budget deployment frame (bundle@300W vs dense@700W), of which 1.47x is a power-cap any
  model gets (dense@300W measured 13.24 tok/W).
- **Headroom list**: "+~1.2x prefill from 2:4" — compressed-tensors Sparse24 serving is REMOVED in
  vLLM 0.20.2 (tombstone); torchao online-quant route tested in the headroom session (see
  HEADROOM_2X_RESULT.md §1 for the outcome). Deeper spec-k is also dead (k=5 optimal; k=10 −35%).

---
## ADDENDUM 2 (2026-06-12, continuation session — see S24_CONTINUATION_RESULT.md)
**DOUBLE-BOS BUG (panel catch): every chat-harness number above is a LOWER BOUND.** The chat
harness rendered apply_chat_template(tokenize=False) and passed strings to llm.generate(), whose
text path adds a second BOS — the EAGLE drafter never saw its training encoding. Corrected
single-BOS measurement (comparable chat set, token-ids path, best-of-3): acceptance 2.37 tok/step,
**FP8+EAGLE = 10062.5 tok/s = 1.52x over FP8-only = ~2.14x dense bf16 — the headline 1.78x was
double-BOS-depressed; the founding 2x throughput goal at saturation is MET on chat traffic.**
(No power sampling on that pair — tok/W rows unchanged.)
**Scope amendment to "drafter HOLDS under saturation":** it holds ON-DISTRIBUTION. On
raw-continuation prompts the SAME matched drafter collapses to 0.52 accepted/step and EAGLE goes
NEGATIVE (−26% at NREQ=32); break-even ≈1.2 accepted/step; saturation cost model speedup ≈
(1+acc)/2.2 (3 measured points). The bundle's spec stage requires acceptance-aware gating on
measured counters, not workload labels. Also: cell H triple (FP8+2:4+EAGLE) measured NEGATIVE vs
FP8+2:4 on continuation traffic (0.80x); the 2:4 artifact failed the locked quality gate (+25%
wikitext PPL vs dense base, MMLU −5 pt McNemar p=0.0095) while the torchao SERVING ROUTE itself
is quality-free (+0.9% PPL, MMLU noise) — see S24_CONTINUATION_RESULT.md §3.
