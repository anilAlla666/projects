# PRE-REGISTRATION — R.A vLLM-COOP CAPTURE HOOK: cudagraph-frame detector marginal (2026-06-11)

Written BEFORE any build or measurement (entry artifacts: `integrity_entry_md5.txt`, recorded 05:44:50Z).
Resolves the WI-4 memo's load-bearing UNMEASURED claim: "capture hook → cudagraph-frame marginal ~6% est.
UNMEASURED". Fork-1 ships eager (+51.2% eager tax, +3.1% detector @N45). Path-1 walled EXTERNAL post-capture
graph mutation (vLLM rejects in init `profile_cudagraph_memory`). THIS test: CAPTURE-TIME insertion — the shim
emits detector ops inline behind each intercepted `cublasGemmEx` DURING vLLM's own `torch.cuda.graph` capture,
so the graph is vLLM-authored and self-consistent; periodicity via a python-level patch of
`vllm.compilation.cuda_graph.CUDAGraphWrapper.__call__` (dual entries per batch descriptor: vanilla + checked,
selected per step). The wrapper patch models a one-line vLLM source hook = "vLLM-coop", NOT substrate-legal
external injection (stated up front; the substrate line is unchanged — this measures the value of cooperation).

## Fixed config (matches fork-1 ra_realvllm for comparability)
Mistral-7B-v0.1 fp16, vLLM 0.20.2 v1, B=8, OUT=64, max_model_len=2048, gpu_mem_util=0.85, T=0 greedy,
ignore_eos, H100 default app clocks (1980). Detector predicate identical to fork-1: linear GEMMs
k∈{4096,14336} ∧ m≠32000; independent same-args-same-algo recompute into preallocated scratch; axpy(−1)+nrm2
(fp16 result, device pointer mode in-graph) ⇒ T=0 residual exactly 0 when clean. Injection: tiny captured CUDA
kernel XORs bit-14 of C[0] of one target linear GEMM ordinal (persistent, fires every replay) — same fault
class as fork-1.

## Arms
- A0 baseline: cudagraph (FULL_AND_PIECEWISE), no shim — tok/s; tie to fork-1's 1289; also with
  cudagraph_capture_sizes=[8] (the restriction used in all shimmed arms).
- A1 in-frame always-on: checks captured behind all ~128 decode linears, every replay — tok/s.
- A2 periodic dual-graph: vanilla+checked entries, checked replayed every Nth decode step, residual readback
  (D2H + check) on checked steps only — tok/s @N=45 and N=8.
- V validation: clean run (RV_INJECT=-1) ⇒ 0 FP across all checked slots; inject run ⇒ detection at first
  checked step, latency ≤N; counters + max residuals recorded as JSONs.

## Predictions (falsifiable, BEFORE measurement)
1. MECHANISM: capture-time insertion SURVIVES vLLM init incl. cudagraph memory profiling (the Path-1 wall was
   post-hoc mutation, not extra captured work). If this crashes like Path-1, the wall generalizes and the arc's
   "vLLM-coop hook" next-step dies — that is a publishable falsification.
2. A1 in-frame always-on costs roughly the decode GEMM share: predicted −40…−70% tok/s vs A0 (NOT shippable;
   measured to bound the always-on design).
3. A2 @N=45: total marginal vs A0 ≤3% (point estimate ~1.5% = (T_checked−T_vanilla)/45 + readback/45).
   A2 @N=8: ~6–12%. The "~6% est" from the prior arc is closest to A2@N8, not @N45.
4. Detection: persistent flip caught at the FIRST checked step (latency ≤N−1 steps), residual >0 exactly at the
   injected ordinal's slot; clean runs 0 FP (T=0 bit-identical in-graph recompute, same kernel+algo recorded).
5. Dual-graph memory: second FULL graph per captured size adds graph-pool memory; with capture_sizes=[8]
   expected O(100MB)-scale, not blocking at 0.85 util.

## Decision rule (pre-registered)
If A2@N45 total ≤3% vs A0 AND catch/0-FP validate ⇒ the WI-4 memo's "~6% est UNMEASURED" is REPLACED by a
measured number and the R.A product line becomes "detector rides cudagraph at measured X% with a one-line vLLM
coop hook; eager tax eliminated" (still vLLM-coop, not substrate-external). If the mechanism walls (pred-1
false) ⇒ record WALL-WITH-MECHANISM; eager host remains the only validated deployment.

## Disclosed risks / outs (named in advance)
cuBLAS workspace aliasing between real and recompute GEMM in-graph (same handle/stream — expected legal,
stream-ordered); piecewise pieces capturing during prefill (emit gated OFF outside the FULL decode capture);
legacy-stream D2H sync stalls on checked steps (counted in A2 cost, every Nth step only); beta≠0 linears would
break recompute-compare (fork-1 measured beta=0 on all checked linears; predicate unchanged); capture-size
restriction changes baseline (controlled: A0 measured both ways).
