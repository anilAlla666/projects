# V0 G-O3 single-GPU MFU: FP8 substitution on the torch path -- SUBSTITUTION REAL, both headline gates BOUND

**2026-06-03. EARNED-NEGATIVE on the headline (report-the-binding-term, like Koopman/g1.3). SUBSTITUTION-REAL: YES --
CIPHER's FP8 actuator substitutes an FP8 E4M3 GEMM for the bf16 GEMM at the cublasGemmEx boundary on a compute-bound
prefill (handled>0, all 225 Mistral linears, output sane). BOTH headline gates FAIL at the driver boundary: (1) MFU
never reaches 85% at any operating point (1.05x bf16 at equal clock; bf16 GEMMs are ALREADY 95-100% MFU so the forward
is non-GEMM-bound), (2) quality +0.72% PPL > 0.37% amended bound (per-tensor all-layers).** NO `.so` change -- probes
only, used the FP8 actuator already in the deployed Jun-01 `libcipher_rt.so`; anchor 1f305ce6 UNCHANGED. Substrate-line:
cublasGemmEx library-symbol-intercept (LD_PRELOAD) -- NOT vLLM-prefill (nvjet-blocked, v1.5), NOT nvjet+monkeypatch
(rejected). Probes: `go3_fp8_step1_substitution.py`, `go3_fp8_step2_mfu.py`, `go3_fp8_step2b_gemm_decomp.py`,
`go3_fp8_step3_quality.py`.

## STEP 1 -- SUBSTITUTION REALITY (the gate's first requirement)

Mistral-7B-v0.1 bf16, 2048-token prefill (compute-bound, cublas n=2048 >> the actuator's n>64 large-batch gate),
`LD_PRELOAD=$SO CIPHER_FP8=1`. The actuator (`cipher_rt_fp8_actuator.c`) engages large-M GEMMs (n>64, the complement
of Marlin's decode n<=64), observes each weight `FP8_STABILITY=2` times, one-time prequant, then `cipher_rt_fp8_engine_
matmul` (NVRTC absmax->E4M3 quant + per-tensor scalar fused cuBLASLt FP8 GEMM -> bf16). After a 1-forward warmup:
- fwd#2-5: **handled=900** (225 linears x 4 substituting forwards), **weights_quantized=225** (ALL q,k,v,o,gate,up,down,
  lm_head), **max_n=2048**, logits **finite**, `argmax_last=302` matches the pre-substitution bf16 forward.
- The prior "handled=0 intercept-only" was just the `FP8_STABILITY=2` warmup, NOT a missing substitution path; under
  LD_PRELOAD the cuBLASLt+nvrtc+driver symbols resolve (no `CUDA_INJECTION64_PATH` needed). **SUBSTITUTION-REAL = YES.**

## STEP 2 -- equal-clock MFU (the prior's DVFS trap, isolated)

The prior's 1.32x was DVFS clock-headroom; equal-clock was ~parity. Here the SM clock is LOCKED IN-PROCESS at 1200 MHz
(a bash-set lock is lost when the GPU idles between processes; verified flat min=max=1200 in every run -- and 1200 is
power-stable at ~344W << 700W limit, so neither dtype throttles -> identical clock, no FP8-power advantage):

| run @1200MHz (flat) | tok/s | TFLOP/s | MFU (vs bf16-peak@1200=600) | power | ratio |
|---|---|---|---|---|---|
| bf16 baseline | 22,361 | 342 | 57.0% | 344W | 1.00x |
| FP8 per-call | 23,410 | 358 | 59.7% | 261W | **1.05x** |
| FP8 + share-act | 24,726 | 378 | 63.1% | 272W | 1.11x (but see quality -- BROKEN) |

Lead metric = the FP8/bf16 throughput **RATIO 1.05x** (clock-independent). Absolute MFU is clock-dependent: at boost
(~1980 MHz) bf16=41%, FP8=40-43% -- **85% is missed at every operating point.** FP8 also cuts power ~24% (261 vs 344W
@ equal clock) -- a real energy result, but NOT the MFU axis.

### Why 85% is unreachable here -- TWO distinct reasons, both measured

**Reason A (the strong one): the bf16 GEMMs are ALREADY near-peak; the forward is non-GEMM-bound.** At 1200 MHz the
real prefill GEMMs hit (achieved/600 peak): q/o_proj **94.7%**, gate/up **98.6%**, down **~100%**, lm_head **99.0%**.
The GEMMs already saturate the tensor cores at n=2048. The forward's 57% (@1200) / 41% (@boost) MFU is therefore the
**~40-45% non-GEMM fraction** (attention, norms, launch gaps), NOT GEMM inefficiency. FP8 only speeds the
already-saturated GEMM portion. (This also preempts "use a bigger batch": GEMMs already saturate, and the quant/GEMM
cost ratio is ~batch-invariant -- both scale with n.) GEMM is ~54-67% of forward time (this run's 2b-sum ~54%; prior
~67.5%).

**Reason B (the FP8 tax): per-call activation re-quant collapses the FP8 win.** Raw per-tensor FP8 GEMM = **1.80x bf16**
(`torch._scaled_mm`, pre-quantized inputs -- the genuine tensor-core ceiling, measured across all four prefill shapes
1.70-1.82x). The actuator delivers **1.05x end-to-end** (measured). The gap = the per-call re-quant (bf16->fp8 every
GEMM, the driver boundary can't keep activations in fp8 across ops without fusing) + dispatch overhead + Amdahl on the
GEMM fraction. (An independent naive-quant decomposition `go3_fp8_step2b` lands 1.13x GEMM-level tax-included -- it
CONFIRMS the tax is large but OVERSTATES it for the actuator: its ~6-op torch quant has launch overhead the actuator's
fused NVRTC kernel avoids; do not read its per-shape numbers as the actuator's profile.)

**Ceiling math (approximate):** a zero-tax FP8 (raw 1.80x on a ~55-67% GEMM fraction) would ceiling at ~1.32-1.43x
end-to-end (~75-81% MFU@1200). Measured 1.05x (~60% MFU). So the per-call re-quant eats most of the FP8 headroom, AND
even a perfect zero-tax FP8 would fall short of 85% single-GPU -- the residual is the FP8-invariant non-GEMM fraction
that needs attention/norm **fusion** (native/vLLM-fused/CUTLASS/multi-GPU = the v1.5 / multi-GPU number), NOT
expressible at the cublasGemmEx boundary.

## STEP 3 -- quality (the load-bearing gate), real wikitext-2

PPL on wikitext-2-raw test (2048-tok contiguous chunk), FP8-substitution vs bf16:
- bf16 PPL **4.1319**; FP8 per-call PPL **4.1616 = +0.72%**, KL(bf16||fp8)=0.0045, max|Δlogit|=4.7. **FAILS the 0.3%
  bound AND the 0.37% amended bound** (corroborates the prior's per-tensor-all-layers +0.567%; single 2048-tok chunk,
  prior agrees).
- FP8 + share-act PPL **32.26 = +680%**, KL=2.06 -- **NON-VIABLE as measured (broken output, not a quant tradeoff)**;
  its 1.11x MFU is on broken output, not a usable result.
- The per-tensor scalar scheme on ALL layers (incl lm_head, m=vocab_size) is the quality break (the FP8-arc lm_head
  decline; scope locked all-layers per instruction -- declining lm_head is the flagged v-next lever, not done here).
  The **per-channel** scheme is the +0.37% (passing) scheme but has NO fused cuBLASLt path on cu13/H100
  ([[cublaslt-cu13-fp8-pertensor-only]]) -> needs custom CUTLASS = v1.5.

## Gate status (Mem #11) + honest framing

1. **SUBSTITUTION-REAL FIRST: PASS** -- handled>0, FP8 E4M3 GEMM replaces bf16 at the cublasGemmEx boundary, all 225
   layers, output sane. Library-symbol-intercept, substrate-line clean.
2. **MFU: FAIL the 85% gate** -- 1.05x bf16 at equal clock; 41% (boost) to 60% (@1200) MFU, never 85%. Reasons measured:
   bf16 GEMMs already 95-100% (non-GEMM-bound) + per-call re-quant tax (raw 1.80x -> 1.05x). bf16 baseline cited.
3. **QUALITY: FAIL the 0.37% bound** -- +0.72% PPL (per-tensor all-layers); share-act non-viable; per-channel passes but
   no fused path (v1.5).
4. **NON-REGRESSION:** NO `.so` change (existing FP8 actuator); the decode/density engine (inc-1..inc-4 + g1.1 + g1.2x)
   is STRUCTURALLY untouched -- the FP8 gate is n>64 (large-batch/prefill), decode is n<=64 (Marlin/graph path); CIPHER_FP8
   default-OFF (OFF == byte-identical, sampler/dispatch read-only); anchor 1f305ce6 UNCHANGED; GPU->0; clocks reset.

**NO RETRACTED NUMBERS (Mem #6):** the FP8 88-98%-of-bf16-MFU prefill figure was vLLM-EXEC reachability, NOT CIPHER --
this report cites ONLY what CIPHER's substitution measured (1.05x, +0.72% PPL).

**FRAMING:** G-O3 is the OPPOSITE regime to the decode/density engine -- compute-bound, single-tenant FP8. The
substitution mechanism is REAL and composes at the cublasGemmEx boundary; both headline gates (85% MFU, 0.37% PPL) are
structurally bound there -- the GEMMs already saturate (non-GEMM fusion needed for higher MFU) and the per-tensor
all-layer scheme breaks quality (fused per-channel CUTLASS needed). Earned negative with the mechanism measured and
every binding term named. The 85% MFU headline remains a FUSION/multi-GPU number (v1.5), not a cublasGemmEx-boundary
single-GPU number.

## STOP -- for V.0 = Anil's call

Multi-GPU/NCCL MFU is 8+-GPU-gated and DEFERRED (not attempted). Related: [[cipher-d9-fp8-actuator]],
[[cublaslt-cu13-fp8-pertensor-only]], [[cipher-gate1-g13-koopman]], [[cipher-gate1-g11-dvfs-into-engine]].
