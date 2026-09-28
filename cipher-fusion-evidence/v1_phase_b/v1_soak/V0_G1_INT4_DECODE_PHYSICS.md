# V0 G-O1 increment-1: INT4 batch-1 decode physics — the latency premise, MEASURED

**2026-06-02. PROBE-FIRST (read-only; NO CIPHER source change; clean torch/transformers/vLLM, no `.so`;
deployed `/usr/lib/cipher` 1f305ce6 / staging 2edba0d2 UNCHANGED).** Harnesses: `g1_int4_vllm_physics.py`
(+ blocked `g1_int4_graphdecode.py`). Results: `g1_int4_vllm_physics_result.json` (int4),
`g1_vllm_physics_none.json` (fp16).

## Question (Anil authorized increment-1, not the engine)

Today's fp16 result ([[V0_G1_8B_ASSERT_ROOTCAUSE]]) measured manual static-KV graph-decode at KL=0 on the real
8B and **projected** int4 (pager regime) at **~205ms / 128-tok burst** (≈1.6ms/tok, assuming int4 is ~4×
bandwidth-floored). That projection was never measured. The engine's distinct-model-density latency advantage
leans on int4 making decode dramatically faster. **Increment-1 measures it.** Advisor framing: separate
**physics** (is int4 batch-1 decode bandwidth-floored ~4×?) from **integration** (CIPHER's Marlin across a
captured decode); do the cheap physics kill-shot first, on any correct bandwidth-floored Marlin int4 model,
**no CIPHER `.so` needed**.

## Method

Per-token decode latency isolated by **differencing two greedy single-seq bursts**:
`per_tok = (t128 − t16)/(128−16)` — cancels prefill + fixed startup. Same engine (vLLM 0.20.2,
`enforce_eager=False` → CUDA-graph decode), same method, Llama-3.1-8B, batch-1, H100 @700W.
- int4 = `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16` (GPTQ, **desc_act=true**, g128 sym) via
  `gptq_marlin`. 5.7GB safetensors.
- fp16 = `/home/ubuntu/models/Llama-3.1-8B`. ~16GB.
- Weight-bandwidth floor = safetensors bytes / 2.5TB/s (the divisor today's fp16 floor used).

**Instrument validated:** fp16 landed **on its floor** (6.25 vs 6.42ms) → vLLM is not adding meaningful
per-step overhead and the int4 number is NOT inflated by the scheduler. CUDA-graph capture of the int4 Marlin
decode succeeds inside vLLM → **int4 Marlin kernels are graph-capture-legal** (retires that feasibility
question at the kernel level).

## Result

| | per-tok | 128-tok burst | weight floor | at floor? |
|---|---|---|---|---|
| vLLM fp16 | **6.25ms** | 802ms | 6.42ms | **YES — bandwidth-floored** |
| vLLM int4 (gptq-Marlin) | **4.59ms** | **590ms** | 2.29ms | **NO — ~2× above floor** |

**Same-engine int4-vs-fp16 latency win = 6.25/4.59 = 1.36×.** NOT ~4×. Measured int4 128-tok = **590ms**, vs
the projected ~205ms (**optimistic ~2.9×**).

## Why the 4× projection was wrong — two independent layers

1. **Byte dilution (4× → 2.8×, kernel-INDEPENDENT, pure arithmetic):** the 5.7GB int4 checkpoint *includes
   the unquantized fp16 lm_head (128k×4096 ≈ 1GB), fp16 embeddings, and fp16 scales/zeros* — none touched by
   int4. True weight-byte ratio = 16/5.7 = **2.8×**, never 4×. The theoretical ceiling was always 2.8×.
2. **Fixed overhead (2.8× → 1.36×, partly kernel-DEPENDENT):** fp16 batch-1 decode hides its ~2ms of fixed
   per-token cost (attention, KV read, Marlin dequant, kernel launches, the unquantized layers) *under* its
   6.4ms weight read. int4's weight read is only 2.3ms, so that same ~2ms is no longer hidden — it roughly
   doubles int4's time above its floor. **int4 batch-1 decode is overhead-bound, not bandwidth-bound.**

## Honest bound (range, not a single number)

- **Measured:** ~1.4× (this checkpoint: desc_act=true gptq_marlin, a known-suboptimal W4A16 path → a floor of
  the achievable, not a ceiling).
- **Theoretical byte ceiling:** 2.8×.
- **Realistic best-case** (desc_act=false / vLLM Machete on Hopper / compressed-tensors W4A16): ~1.5–2×.
- **Never** the 4× / sub-205ms / sub-200ms the projection and the engine's latency story assumed.

The qualitative finding holds at **every** point in this range.

## Decision-relevant read

**int4-fast-decode is NOT where the engine earns its keep.** The 2.8× byte ceiling minus batch-1 overhead
leaves a modest single-stream latency win (~1.4–2×), and 128-tok stays sub-second either way (int4 590ms vs
fp16 802ms). If the multi-model engine is justified, it is on the **consolidation / agents-per-GPU density**
axis the prior arc established ([[cipher-go1-dispatch-router]], [[cipher-go1-fullbudget-100agent]],
[[cipher-pager-coresidence-build]]) — **not** on int4 making decode dramatically faster. Latency premise
**corrected**, not the engine outright killed; this removes one of the engine's claimed levers.

## Caveats / not-done

- Manual-harness apples-to-apples (today's `torch.cuda.graph` static-KV recipe on int4) was **BLOCKED**:
  the gptq checkpoint needs `optimum`+`gptqmodel`/`auto_gptq` to load in transformers; **gptqmodel failed to
  build** on this cu13/torch-2.11 stack, and the compressed-tensors candidates checked had no loadable quant
  config. vLLM (which DOES graph-capture the int4 Marlin decode) carries the physics + capture-legality
  finding; the fidelity number (graph-int4 == eager-int4 128/128) in MY harness remains a CIPHER-integration
  nicety, not a physics question.
- One time-boxed clean-kernel sharpening (a confirmed-loading desc_act=false / compressed-tensors W4A16 → vLLM
  Machete) would tighten ~1.5–2× to a measured number. Skipped to avoid re-entering the checkpoint-hunting
  loop; the decision does not hinge on 1.4 vs 2×.
- Op: vLLM `os._exit` orphans the `VLLM::EngineCore` (~49GB held) — reaped by exact PID (not blanket kill).
