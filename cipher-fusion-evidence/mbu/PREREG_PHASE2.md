# PRE-REGISTRATION — MBU PHASE 2: is the ~2× low-batch headroom CAPTURABLE by a better 4-bit kernel? (2026-06-11)

Phase 1 found: fp16 decode BW-bound (MBU ~70%); 4-bit `gptq_marlin` gives only 1.35× of a 3.6× byte cut because it
is compute/overhead-bound at batch 1 (MBU 27%, bus 3.7× idle). I claimed a GEMV-specialized kernel could reach
~2.7–3.6× over fp16. **Phase 2 tests that claim directly** by benchmarking the 4-bit kernels vLLM already ships —
NO custom kernel written. This is the decisive moat test: if a better kernel exists, the headroom is real and the
substrate (machete-class) could host it; if every 4-bit kernel is ~27% MBU at batch 1, the 27% is a fundamental
low-batch-4-bit property and **MBU closes for real**.

## Setup
Same rig: real vLLM 0.20.2, Mistral-7B, cudagraph ON, batch 1 (+ 8 for context), decode-only, short ctx, T=0,
default clocks. Same roofline MBU + dmon. Arms (all ~4 GB weight budget, same arch — isolates the KERNEL):
- **gptq_marlin** (GPTQ checkpoint) — phase-1 baseline, 222 tok/s / 27% MBU @b1.
- **awq** (AWQ checkpoint, plain AWQ **GEMV-specialized** kernel; force via `quantization="awq"` /
  `VLLM_USE_TRITON_AWQ` if needed). This is the kernel explicitly tuned for batch-1 decode.
- **awq_marlin** (AWQ checkpoint through marlin) — controls for checkpoint vs kernel (AWQ weights, marlin kernel).
- (optional) **gptq** non-marlin / exllama if vLLM will route to it.

## Predictions (falsifiable, BEFORE measurement)
1. **If marlin is specifically batch-1-overhead-bound** (my phase-1 hypothesis), the AWQ GEMV kernel beats it at
   batch 1: awq tok/s > gptq_marlin's 222, MBU > 27%. A >1.3× kernel-over-kernel gain would EVIDENCE the headroom
   is real and capturable.
2. **If 27% MBU is a fundamental low-batch-4-bit-dequant limit** (the skeptical alternative), ALL kernels land
   ~27% MBU / ~200–250 tok/s at batch 1 — the bus stays idle regardless of kernel, the dequant arithmetic itself
   is the floor, and my "~2× capturable headroom" was wrong ⇒ **MBU closes like MFU after all.**
3. At batch 8+, marlin (GEMM-tuned) should catch up / overtake the GEMV kernels (GEMV kernels are batch-1 specialized).
4. Whatever the fastest 4-bit kernel is, it is still **vLLM-native** — so even a positive result only relocates the
   moat question to "can CIPHER's machete beat the best vLLM kernel", which THIS test does not answer (machete not
   run); it only answers whether the headroom EXISTS.

## Decision rule
- Best 4-bit kernel @b1 ≥ ~1.3× the marlin baseline (≥ ~290 tok/s) ⇒ headroom REAL and capturable ⇒ MBU stays an
  OPEN substrate-kernel opportunity (phase-1 verdict holds, strengthened).
- All 4-bit kernels @b1 within ~±15% of marlin's 222 (≤ ~255 tok/s) ⇒ 27% MBU is the low-batch 4-bit floor ⇒
  headroom NOT capturable by kernel choice ⇒ **revise phase-1: MBU closes, the ~2× claim was optimistic.**

## Disclosed risks / outs
vLLM may silently route AWQ→awq_marlin (defeating the GEMV comparison) — verify `quant_method` in-JSON per run;
the AWQ and GPTQ checkpoints differ slightly (different calibration) but both 4-bit same arch ⇒ ~equal byte budget,
fair kernel isolation; quality differs across checkpoints (not measured); cudagraph capture must include the batch.
Anchor 2edba0d2 entry==exit; substrate not loaded (`anchor_loaded` probe); scratch in `mbu/`.
