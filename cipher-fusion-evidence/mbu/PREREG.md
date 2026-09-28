# PRE-REGISTRATION — MBU: is reduced weight-bytes-streamed a real decode lever, and is it substrate-deliverable? (2026-06-11)

Written BEFORE measurement. Resolves the 6-axis mandate's least-explored axis (MBU). Question in two parts:
(A) PHYSICS — on memory-bound decode, does reducing weight bytes streamed (fp16 → 4-bit) convert to throughput,
and by how much vs the byte-reduction ratio? (B) MOAT — is any lift a SUBSTRATE deliverable, or just a
model-format choice the customer makes (the trap FP8 and bf16-adapters fell into)?

## Key prior-art constraint (do not re-derive wrong)
The batch-sweep ([[cipher-mfu-batch-sweep-settles-g3]]) established eager decode is **launch-bound** (flat
~111–118 ms/step), NOT bandwidth-bound. A bytes-reduction lever only helps when decode is actually BW-bound. So
the lever MUST be tested in the **cudagraph** regime (launch overhead removed) — measuring it eager would falsely
conclude "4-bit doesn't help" for the wrong reason. MBU = achieved HBM read BW ÷ H100 HBM3 peak (~3.35 TB/s).

## Setup
Real vLLM 0.20.2, Mistral-7B-v0.1, **cudagraph ON**, decode-only steady state, batch ∈ {1,2,4,8,16}, short ctx,
T=0 greedy. Two model arms:
- **fp16** (`mistralai/Mistral-7B-v0.1`): ~13.5 GB weights/forward.
- **4-bit GPTQ** (`TheBloke/Mistral-7B-v0.1-GPTQ`, gptq_marlin kernel): ~3.9 GB weights/forward (~3.5× fewer).
Metrics per (arm, batch): decode tok/s (best-of-2), achieved HBM BW via roofline (weight+KV+act bytes ÷
step-time) cross-checked against `nvidia-smi dmon -s u` mem% sampled during decode, MBU = achieved ÷ peak.

## Predictions (falsifiable, BEFORE measurement)
1. **fp16 cudagraph decode at batch=1 is weight-BW-bound** ⇒ MBU high (>50%); step time ≈ 13.5GB / achieved-BW.
   If instead MBU is low (<30%) even with cudagraph, decode is bound by something else (KV, kernel count) and the
   bytes lever is capped — record that.
2. **4-bit gives a sub-proportional lift at batch=1**: throughput ratio < 3.5× (the byte ratio), because
   (a) KV + activations are still fp16 (unreduced bytes), (b) GPTQ dequant adds compute, (c) non-weight overhead.
   Point estimate 1.5–2.5×. If it's ≈1× (no lift), decode wasn't weight-BW-bound after all.
3. **The 4-bit advantage SHRINKS with batch** — at high batch, weight reads amortize over the batch and decode
   becomes compute-bound, so 4-bit (which trades BW for dequant compute) loses its edge and may go NEGATIVE.
   So 4-bit/NF4 is a LOW-BATCH (memory-bound) lever only — consistent with "batching is a memory-bound-only lever."
4. **MOAT (the decisive one):** the lift, if real, comes from loading a pre-quantized 4-bit model — a customer
   model-format choice that vLLM hosts natively (gptq_marlin). The substrate's machete/NF4 actuators can *host*
   4-bit GEMMs but do not *uniquely provide* the lever. PREDICTION: MBU is **perf-only and NOT a substrate moat**,
   same structural verdict as FP8 — unless the substrate's on-the-fly NF4 (compress an fp16 model the customer
   ships, no model swap) is shown to deliver the same lift, which would make it substrate-reachable.

## Decision rule
- If 4-bit gives a real low-batch decode lift (>1.3× at batch≤2) ⇒ MBU is a REAL lever (physics confirmed); then
  the moat question decides whether it's a CIPHER deliverable or a vLLM/model-format feature.
- If the lift is ≈1× even in cudagraph ⇒ decode is not weight-BW-bound in this regime; MBU is CLOSED like MFU
  (no substrate lever, document the binding constraint — KV/kernel-count/compute).
- Substrate-deliverable ONLY if on-the-fly NF4 (no customer model change) delivers the lift; else perf-only/
  model-format, report as such.

## Disclosed risks / outs
GPTQ dequant overhead may dominate at low batch (kernel not BW-optimal) → measure the kernel, don't assume;
gptq_marlin requires the marlin kernel path (verify it engages, not a slow fallback); dmon mem% is a coarse BW
proxy (cross-check with roofline bytes/time); KV-cache bytes grow with context (use short ctx to keep weights
dominant); cudagraph capture sizes must include the tested batches; both arms at default clocks 1980.

## Discipline
Frozen anchor md5 2edba0d2 entry==exit (this is a vLLM/model measurement; substrate not loaded — `anchor_loaded`
probe in every JSON). Scratch only in `mbu/`. No anchor rebuild. Clocks default, sampled per run.
