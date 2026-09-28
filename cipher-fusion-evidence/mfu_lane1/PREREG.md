# PREREG — MFU DELIVERY LANE 1 (2026-06-10)

Written BEFORE any measurement. Anchor 2edba0d2136f8ede4713d90a8f7cd55f entry-verified. Clocks DEFAULT
(unlocked at entry, 345 MHz idle GpuIdle), matching MFU_AUDIT convention. MFU = analytic FLOPs ÷ 989.5 TFLOP/s
bf16 dense peak; prefill 2·N·T, train 4·N/token, N=7,241,732,096.

## Lane definitions locked before measurement

- 1a "engaged-observation-only" = unmodified anchor via CUDA_INJECTION64_PATH **+ CIPHER_VOLT=off**.
  SOURCE FINDING (pre-measurement, cipher_rt_volt.c:344-363): with CIPHER_VOLT UNSET the substrate defaults
  VOLT to **ARMED** and the classifier can self-engage a clock lock (this is what fired in yesterday's
  contaminated run). `CIPHER_VOLT=off` ⇒ "staying OFF (classifier override denied)". Therefore the honest
  zero-actuator lane requires the explicit disarm env; this is a DISARM (not an arming) and is disclosed.
  Zero-actuator proof per engaged run = ALL of: substrate-line count > 0 (loaded); "VOLT: CIPHER_VOLT=off —
  staying OFF" present; "FP8/MARLIN: actuator DISABLED" present (envs unset); zero "ENGAGED"/substitution
  lines; MATMUL totals actuators=0/handled=0 if flushed.
- Interleaving: process-level V,E,V,E,V,E (≥3 reps per arm per workload); per-rep values + medians reported.

## 1a pre-registered expectations

| Workload | Expected hosting tax (vanilla→engaged-observe-only) |
|---|---|
| torch-HF prefill 8×2048 (≥5 fwd/rep) | **~1–3 %** (per-call intercept cost amortizes over large GEMMs; the ~1.9× decode figure should NOT transfer — that was per-launch overhead on thousands of tiny launches) |
| eager vLLM prefill 8×2048×10 | **~1–5 %** (same amortization; more non-GEMM launches than torch path ⇒ possibly the high end). Injection expected to reach the V1 EngineCore worker via env inheritance; if it cannot, WALL-WITH-MECHANISM naming the process boundary |
| LoRA train step B=1 s512 ×40 | **~1–5 %** (launch-denser than prefill: bwd + optimizer small kernels ⇒ possibly above prefill tax) |

Surprise rule: tax >8 % on any compute-bound phase, or a negative tax beyond noise, is a surprise with mechanism.
VERDICT criterion (per workload): W3 "gates Goal-3 phases" if tax > ~3 % (eats the FP8 margin); else not gating.

## 1b pre-registered expectations

- Source side: substrate intercepts cublasGemmEx (+Lt variants) — exact table from source with file:line.
- Runtime: vLLM dense-fp16 prefill GEMM entry **UNKNOWN — the deciding measurement**. Conflicting priors,
  both pre-registered: D.9 probe recorded vLLM cublasGemmEx=0 vs torch=3375 (per the delivery plan), yet
  R.A 2026-06-08 intercepted vLLM 0.20.2 eager **decode** linears via cublasGemmEx successfully (629→602 tok/s
  detector numbers exist). Hypotheses to discriminate: (a) prefill large-M routes via cublasLtMatmul/nvjet
  while decode uses GemmEx; (b) the D.9 probe ran a different vLLM mode (cudagraph capture hides per-call
  entries); (c) version/path differences. Expected torch census: GemmEx-family dominant (V0/audit precedent).
- Expected route recommendation shape: torch-side first target (engagement proven) unless vLLM coverage turns
  out to be one already-intercepted symbol.

## 1c pre-registered expectations

- Expected recipe: **channelwise (per-output-channel) weight scales + dynamic per-token activation scales,
  E4M3, fp32 accumulation**, all linear projections quantized, lm_head excluded-or-included per record; the
  +0.37 % PPL eval = wikitext-2-style PPL (dataset/seq from record). Elements not in the record will be filled
  from vLLM's own FP8 w8a8 implementation and labeled ASSUMPTION-FROM-VLLM-SOURCE (file:line), never blurred.

## Honesty
No retracted numbers (3.617×, 14×, universal 85 % MFU, 2.96×, 7.43×, 0.127 %, unscoped +57.3 %). Walls =
WALL-WITH-MECHANISM. Every number to a JSON in mfu_lane1/.
