# V0 GATE-1 increment g1.2: Marlin int4 into the engine -- CAPTURE-SAFE (non-act-order), density NOT delivered

**2026-06-03. MIXED, honest verdict. CORRECTNESS: int4 Marlin (non-act-order compressed-tensors W4A16) composes
CAPTURE-SAFE in the engine (KL=0). CAPTURE-SAFETY is CHECKPOINT-DEPENDENT (act-order = engine-incompatible). DENSITY:
NOT available via compressed-tensors on this stack (EARNED negative -- the forward materializes ~10GB fp16, retained,
even with run_compressed=True).** NO `.so` change; g1.2 added ONLY probes (inc-1+inc-2 modules BYTE-IDENTICAL,
`cipher_inc4.py` unchanged since g1.1); anchor 1f305ce6 (May-27) UNCHANGED. `CIPHER_RT_DISABLE_AUTO_INIT=1`. Probes:
`pager_g12_int4_capture_probe.py`, `pager_g12_density.py`.

## Correctness -- int4 Marlin composes CAPTURE-SAFE (the real win)

int4 = `RedHatAI/Qwen2.5-7B-Instruct-quantized.w4a16` (compressed-tensors W4A16 pack-quantized, NON-act-order),
loads in transformers via `compressed_tensors 0.15`. Same-precision oracle (int4-graph vs int4-EAGER, NOT
fp16-bit-exact, per the inc-1 framing):
- **STEP 1 (torch.cuda.graph, NO pager):** the int4 model captures into the engine's static-KV decode graph; graph
  vs eager = **64/64 KL=0**. The int4 kernel does NOT fire capture-illegal ops (unlike Koopman cusolver).
- **STEP 2 (pager region):** resident **64/64 KL=0** + pagecycle **64/64 KL=0** (survives a PHYSICAL evict/restore --
  VA stable, bytes restored). int4 weights route into the pager region (5.26GB) and replay correctly across paging.

So Marlin int4 (non-act-order) is **capture-safe in the engine path**, the first int4 actuator composition.

## Capture-safety is CHECKPOINT-DEPENDENT -- act-order is ENGINE-INCOMPATIBLE (decisively isolated)

The distinct `nm-testing/Meta-Llama-3-8B-Instruct-W4A16-ACTORDER-...` checkpoint **FAILS capture, even ALONE**:
`RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture`. Act-order (desc_act) applies a
**g_idx permutation with a host-side (CPU) op** that is capture-illegal -- the SAME family as Koopman's cusolver and
the inc-1 desc_act=true GPTQ issue. **Verdict: non-act-order int4 = capture-safe; act-order int4 = engine-incompatible
(do NOT force).** A real, cleanly-isolated boundary.

## Density -- NOT delivered via compressed-tensors on this stack (EARNED negative, not assumed)

The pager region holds the int4 packed weights (**5.26GB**, ~2.9× smaller than fp16's 15.4GB) -- but that footprint
is MISLEADING. Decisive HBM measurement (`device_map=cuda`, Qwen2.5-7B int4):
- load **+7.9GB** (packed int4), forward **+9.7GB more** (=18GB), **post-`empty_cache` = 17.3GB** -> the forward
  **materializes ~10GB of fp16 working weights and RETAINS them** (empty_cache does not free). The runtime is
  effectively fp16 (decompress-once-keep), not fused-Marlin.
- **Forced `run_compressed: true` in config.json** (so the checkpoint config can't override): **SAME 17.3GB** -> the
  decompression is not a config default, it's the stack's behavior (transformers 5.8.1 + compressed_tensors 0.15
  W4A16 on Hopper does not run truly fused).
- Co-residence: **4 captured int4 models exhaust HBM** (~18GB each) ≈ the inc-3 fp16 baseline (~5). **NO density gain.**

The 5.26GB pager region + ~10GB retained fp16 = ~17GB per captured int4 model ≈ fp16. The density axis -- the whole
point of g1.2 -- does NOT deliver via compressed-tensors here.

**OPEN PATH (next increment, NOT this one):** bnb NF4 keeps int4 packed with in-kernel fused dequant (proven in the
pager at inc-1, [[cipher-pager-int4-build]]) -> true ~4× density. It is a DIFFERENT quantizer with its OWN
capture-safety question (bnb dequant has host-side ops) -- a fresh probe arc, deliberately NOT chained here.

## Gate status (Mem #11) + honest framing

1. Correctness FIRST: per-model int4-graph-vs-int4-eager **KL=0** (STEP 1 + pager resident/pagecycle), non-act-order.
   (Engine teacher-forced both-regimes run not needed: int4 runs ~fp16 here, behaves as inc-4's fp16 which passed;
   the int4-specific capture-safety is proven at model+pager level.)
2. Capture-safe verdict: **non-act-order = capture-safe; act-order = engine-incompatible** (g_idx CPU op).
3. Density: **NOT delivered via compressed-tensors on this stack** (earned; ~4 co-reside ≈ fp16 ~5). bnb is the open path.
4. NON-REGRESSION: probes only; inc-1..inc-4 + g1.1 unaffected; OFF byte-identical; anchor UNCHANGED; GPU->0.

**FRAMING:** g1.1 (DVFS) composed AND delivered its energy axis (1.53× tok/W). g1.2's CORRECTNESS composes
(capture-safe int4, checkpoint-dependent) but its DENSITY axis does NOT deliver via compressed-tensors on this stack
-- gated on a fused-int4 path (bnb NF4, next increment). Honest: capture-safety is the earned g1.2 result; density is
an earned negative with a named open path.

## STOP for g1.3 (Koopman probe-and-report) decision

g1.2 closed: int4 Marlin capture-safe (non-act-order) + act-order incompatible + density-not-via-compressed-tensors
(bnb open). Remaining gate-1: g1.3 Koopman (cusolver, almost-certainly engine-incompatible -- a probe-and-report).
Density via bnb is a separate future increment. Anil's call.
