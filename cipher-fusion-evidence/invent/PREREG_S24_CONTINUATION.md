# PRE-REGISTRATION — s24 continuation session (2026-06-12, written BEFORE any run of this session)

Continues HEADROOM_2X_RESULT.md (2026-06-11). The 18:56 s24_hv run (cell H on varied prompts) was
killed during cudagraph capture when the session ended — no result was produced (s24_hv.log has no
S24G line; verified at session entry). This prereg is saved before ANY measurement run of today's
session, to avoid a repeat of the P1 post-hoc taint.

Protocol: H100, vLLM 0.20.2, cudagraph ON unless stated, NREQ=32 varied-prompt raw harness
(seeds identical to s24_g2v), `VLLM_PLUGINS=` empty (substrate NOT loaded),
`VLLM_DEEP_GEMM_WARMUP=skip`, best-of-3, T=0, power sampled as before (blocking nvidia-smi +
0.1 s sleep). Anchor 2edba0d2 verified at entry; will verify at exit.

## Planned cells (throughput, varied prompts, same seeds/order as s24_g2v)
- **Hv** = sparse24_tao_f8 via torchao (G2 route) + EAGLE-3 Instruct head, k=5 — the interrupted run.
- **Bv** = dense Llama-3.1-8B-Instruct bf16 (baseline on this prompt set).
- **Dv** = dense Instruct + fp8 on-the-fly.
- **Iv** = dense Instruct + fp8 + EAGLE-3 (the BUNDLE, on this harness/prompt set — first
  in-harness triple-vs-bundle comparison; all prior bundle numbers were chat-harness).
- G2v already exists from yesterday: 8005.9 tok/s / 18.62 tok/W, coherent output.

## Planned quality arms (WikiText-2-raw test PPL, teacher-forced prompt_logprobs, identical
## disjoint ~1024-token chunks across all arms, ≥30k scored tokens)
- **Qa** = neuralmagic/Sparse-Llama-3.1-8B-2of4 bf16 (dense kernels — as-stored quality).
- **Qb** = sparse24_tao_f8 via torchao FP8+2:4 G2 route (as-SERVED quality — THE gate).
- **Qc** = meta-llama/Llama-3.1-8B base bf16 (pruning-cost reference; 2of4 was pruned from base).
- **Qd** = meta-llama/Llama-3.1-8B base + fp8 on-the-fly (FP8-cost-on-dense context).

## Predictions

**P6 (Hv / G2v, EAGLE multiplier on coherent text):** the degenerate-prompt H gave 13564.4 =
1.70x over G2-degenerate — inflated (repetition is trivially draftable). On varied coherent
prompts with the MISMATCHED Instruct head on the pruned base, predict Hv/G2v ∈ **1.05–1.45x**
(point guess ~1.25x → ~10,000 tok/s). Below 1.05 = head transfer fails on real text; above 1.45
would exceed the matched-head chat-harness multiplier (1.43x), implausible.

**P6b (Iv / Dv, matched-head EAGLE multiplier this harness):** predict **1.20–1.50x**. Triple vs
bundle: triple wins iff (G2v/Dv) × (Hv/G2v) > (Iv/Dv). With G2v/D≈1.21 and a mismatched-head
penalty on Hv, predict triple and bundle land **within ±10% of each other** — the co-trained-head
pipeline step is what would break the tie. Either ordering reported as-is.

**P7 (quality gate):**
- **P7a (THE gate): Qb/Qa PPL ratio ∈ 1.00–1.03.** FP8-PerRow dynamic act+weight quant of an
  already-2:4 checkpoint should cost ~nil PPL; SPARSE_CUTLASS packing is exact w.r.t. the pruned
  weights (zeros are zeros).
- **P7b: Qa/Qc ∈ 1.02–1.12** (the pruning+recovery cost; NM claims ~99% downstream recovery).
- **P7c: Qd/Qc ∈ 1.00–1.02** (FP8 on dense ~free).
- **DECISION RULE (locked):** the 2:4 stage "joins the bundle" iff Qb/Qa ≤ 1.05 AND Qb/Qc ≤ 1.15.
  Else it stays a throughput-only curiosity pending a recovery-trained FP8-native artifact.

No post-hoc re-scoping of P6/P7; misses get reported as misses. Single-session, no panel taint:
this file is saved before the first vLLM process of 2026-06-12 starts.
