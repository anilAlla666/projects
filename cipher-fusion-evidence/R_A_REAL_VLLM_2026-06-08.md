# R.A REAL-WORKLOAD VALIDATION — fork-1 detector inside REAL vLLM (eager, cuBLAS interception)

**Date:** 2026-06-08/09. **Model:** mistralai/Mistral-7B-v0.1 fp16 (cached). **Stack:** vLLM 0.20.2 / torch 2.11,
real KV cache, continuous batching, **real FlashAttention**, real token generation. **Host:** `enforce_eager` (the
only host where a cuBLAS interceptor sees every steady-state decode GEMM; under cudagraph the GEMMs replay via graph
launch and the interceptor never fires — so this run carries the eager tax and does **not** claim <3% live).
**Detector:** a SCRATCH `LD_PRELOAD` cuBLAS-interception shim (`ra_realvllm/detector_shim.c`), NOT the frozen anchor,
NOT a vLLM source edit. It intercepts `cublasGemmEx`; every N steps it re-runs each linear GEMM with an independent
`cublasGemmEx` (same args+algo) into a scratch buffer and compares `‖D′−C‖` via `cublasAxpyEx`+`cublasNrm2Ex` (T=0:
clean recompute is bit-identical). **Discipline:** anchor `2edba0d2…` frozen (md5 entry+exit); no edit to `ra_e2e/`,
`step_b_e2e/`, `ra_keystone/`; all new work in `ra_realvllm/`; no vLLM source edit, no monkeypatch; clock 1980 →
`-rgc`; vLLM children reaped. **Evidence:** `ra_realvllm/{disc_shim.c, detector_shim.c, vllm_decode.py, *_result.json,
tokps_result.json}`.

---

## BOTTOM LINE
**The fork-1 periodic-recompute SDC detector is VALIDATED end-to-end inside real vLLM** (real attention, KV cache,
continuous batching): it fires on real decode steps via cuBLAS interception, **catches an injected persistent SDC in a
real linear-GEMM output at the first check step (≤N)**, with **0 false positives** (T=0 bit-identical recompute,
`max_clean_residual=0` over 5,376 clean GEMM-checks across clean+bench runs). **Cost decomposition:** detector-marginal
**+3.1% @N=45 / +12.0% @N=8** measured on the eager base, on top of a **+51.2% eager tax** (cudagraph is 2.05× eager).
The detector itself is cheap on the eager base; the eager tax dominates and is an artifact of the *host*, not the
detector. **The unbuilt step toward a <3% live number is the cudagraph-preserving Path-1 in-graph node injection**
(Track-B
primitive) — it would let the detector ride inside the cudagraph and remove the eager tax.

---

## STEP-0 PRE-REGISTRATION (paper, before building the shim)
The detector recomputes all linear GEMMs on check steps (1/N of steps). Expected detector-marginal ≈
(linear-GEMM recompute work)/N over the trace. From ra-e2e the recompute ≈ 1× the linear-GEMM time of a step, and
linears dominate eager decode, so pre-reg **marginal ≈ c/N** with c≈O(1) of the linear-GEMM step fraction (plus the
per-GEMM compare). Pre-registered: **a few % at N=45, scaling ∝ 1/N**; the eager tax (separate) re-confirms Track-B's
~47%. (NB the as-built compare does a blocking `nrm2` host-sync per checked GEMM, so measured marginal sits above the
pure-recompute floor — named as the optimization to batch/async the compare.)

## (1) DETECTOR FIRES ON REAL DECODE STEPS via cuBLAS interception
Discovery (`disc_shim`, `discovery`): vLLM's fp16 linear path is **`cublasGemmEx`** (not cublasLt). Per decode forward
= **128 linear GEMMs** — fused **qkv (m=6144)**, o (m=4096,k=4096), fused **gate_up (m=28672)**, down (m=4096,k=14336)
× 32 layers — plus **lm_head (m=32000)**. Decode GEMMs carry n=token-count (≈22 in test). **Attention is FlashAttention
(a fused CUDA kernel, NOT cuBLAS) → it never enters the interceptor.** The detector checked **2,176 GEMMs across 17
check steps** (clean N=8 run) and 3,200 across 25 (bench) — firing confirmed on real decode steps.

## (2) INJECTED PERSISTENT SDC CAUGHT WITHIN ≤N
Persistent fault: bit-14 flip of one output element of **layer-0 down_proj on every step** (`detector_shim.c`
injection mode, stream-synced so the async GEMM does not overwrite the flip — a real, propagating corruption).
Result (`inject_result.json`, N=8): **caught at the first check step** (`detections` rises to 1 at the first check),
**17/17 check steps detected** the persistent fault, `max_detect_residual=323.75`, residual ≥ 2.0 on every catch.
**Measured:** caught at the first check step (injection persistent from step 0, which is a check step). The general
**≤ N−1 worst-case latency** (a fault onset at an arbitrary step is caught at the next check step) follows by
construction, not from a fault-injected-at-step-k sweep.

## (3) ZERO FALSE POSITIVES on clean runs (T=0)
Clean run (`clean_result.json`, N=8, no injection): **`detections=0`**, **`max_clean_residual=0`** over **2,176**
GEMM-checks; the bench clean run adds 3,200 more checks, also **0 detections / residual 0**. The independent recompute
is **bit-identical** to the original cuBLAS output (same algo, fp32-accum) ⇒ T=0 separates clean (exactly 0) from
corrupt (≥2.0) with no threshold tuning and no false alarm. **Liveness proof (not a dead comparator):** the injection
run's single `inject_result.json` carries `max_clean_residual=0` AND `max_detect_residual=323.75` from the *same run,
same binary, same comparator* — the comparator is provably live on the injected GEMMs while clean GEMMs read exactly 0.
(Measurement bug found & fixed mid-run: `cublasNrm2Ex` on fp16 input requires `resultType=CUDA_R_16F` — with the wrong
type it silently returned 0; both clean and injection were re-run after the fix.) **Sensitivity floor:** the residual
norm is read back in fp16 (the only valid nrm2 resultType for fp16 input), so the comparator separates 0 from the
tested SDC (residuals 2–324) with huge margin, but a corruption below ~fp16-ULP-of-the-norm would sit under the readout
floor — irrelevant for bit-level SDC, noted so T=0 is not read as infinite dynamic range.

## (4) THROUGHPUT — three numbers + the eager tax (`tokps_result.json`, Mistral-7B, batch 8, 64 decode tok)
| Configuration | decode tok/s | vs eager baseline |
|---|---|---|
| **Eager baseline** (no detector) | **629.2** | — |
| **Eager + detector, N=45** | **609.5** | **detector-marginal +3.1%** |
| Eager + detector, N=8 | 553.9 | detector-marginal +12.0% |
| **Cudagraph baseline** (no detector) | **1289.0** | **eager tax +51.2%** (cudagraph 2.05×) |

- **Detector marginal** (eager+detector vs eager): **+3.1% @N=45**, **+12.0% @N=8** — scales ∝ 1/N as pre-registered;
  the N=8 figure carries the per-GEMM `nrm2` host-sync (batchable).
- **Eager-vs-cudagraph tax re-confirmed: +51.2%** (cudagraph 1289 vs eager 629 tok/s) — consistent with Track-B's
  ~47% (Qwen2-7B). Total cost of the detector *as hosted here* vs the cudagraph baseline you'd actually deploy =
  **52.7% @N=45** (= 1 − 609.5/1289). **Note the denominators differ:** the eager tax (51.2%) and total (52.7%) are on
  the cudagraph base; the detector-marginal (3.1%) is on the eager base — they compose *multiplicatively*
  (609.5/1289 = (629.2/1289)×(609.5/629.2)), not additively (51.2+3.1≠52.7). The takeaway holds: the eager tax
  dominates and is a host artifact; the detector itself is small **on the eager base** (see the cudagraph-frame caveat
  in the closing section — its cudagraph-frame cost is ~6% and unmeasured).

## GEMM COVERAGE WITH ATTENTION PRESENT — the honest denominator
The interceptor **checks** the **linear projections** (qkv/o/gate_up/down) — the bulk of decode compute. lm_head is
*seen but not checked* (the shim excludes m==VOCAB, `detector_shim.c:81-82`), so it is uncovered alongside attention.
**Real attention runs as FlashAttention (not cuBLAS), so it is OUTSIDE the detector's coverage.** At the tested short
context (~24–88 tokens) the checked linear GEMMs are **>99%** of decode FLOPs, so the recompute covers essentially all
decode compute; at long context attention FLOPs grow (≈10–25% at 4–8K context), so the *uncovered* fraction (attention +
lm_head) grows with context length.
The detector's GEMM-output coverage holds with attention present (the projections are still cuBLAS), but **attention
QKᵀ/PV corruption is not covered by this shim** — it needs the separate attention-ABFT (Gap-3), which was analytically
established but not wired into this cuBLAS interceptor.

## OUT OF SCOPE (named, not silently dropped)
- **Deterministic "mercurial-core" faults:** the recompute runs on the *same* GPU and reproduces the wrong answer ⇒
  **BLIND**. Catching them needs cross-device DMR (FORK-2: ≥14% cost, SM-localized only).
- **Softmax-internal finite** corruption (algebraically invisible to the checksum, Gap-3).
- **NCCL / collective / multi-GPU / cross-rank** corruption — single-GPU scope only.

## FORK-1 INTEGRITY (md5, entry vs exit)
| File | entry | exit | match |
|---|---|---|---|
| `libcipher_rt.so` (anchor) | `2edba0d2136f8ede4713d90a8f7cd55f` | `2edba0d2136f8ede4713d90a8f7cd55f` | ✅ |
| `ra_e2e/common.py` | `8276b0814020a8d9167a0edb2da58e85` | `8276b0814020a8d9167a0edb2da58e85` | ✅ |
| `ra_e2e/run_e2e.py` | `2de53982eaf1aef14234edd8a5f6819a` | `2de53982eaf1aef14234edd8a5f6819a` | ✅ |
| `ra_e2e/calib.json` | `ade20e25357d108423210b4c2fea3ac9` | `ade20e25357d108423210b4c2fea3ac9` | ✅ |
| `ra_e2e/trace.json` | `7d55b9120d05a74249db3dcc7618bb4d` | `7d55b9120d05a74249db3dcc7618bb4d` | ✅ |
| `ra_e2e/run_e2e_summary.json` | `e5ac91b2d54407ab85f1d3ee82251d57` | `e5ac91b2d54407ab85f1d3ee82251d57` | ✅ |
| `ra_e2e/throughput.json` | `8131a2bd8f7ed7fd0e3821fc879cb98a` | `8131a2bd8f7ed7fd0e3821fc879cb98a` | ✅ |
| `ra_e2e/detect.json` | `cefa60651cc8cb56885923164c070a4c` | `cefa60651cc8cb56885923164c070a4c` | ✅ |

## GUARDRAILS (exit)
Anchor unchanged; no vLLM source edit; scratch shim only (cipher vLLM plugins excluded via `VLLM_PLUGINS`=lora-only;
`VLLM_USE_DEEP_GEMM=0`); clock reset `-rgc` (idle 375 / max 1980); no compute procs left; vLLM EngineCore reaped.

## WHAT THIS CHANGES vs the synthetic ra-e2e harness
ra-e2e excluded attention (identity q→o) and was not a real server. This run adds **real FlashAttention, real KV cache,
continuous batching, real tokens** — and the detector still fires, catches the injected persistent SDC, and holds 0 FP.
The remaining gap to a shippable <3%-live number is **not** detection (validated) but **hosting**: the eager interceptor
costs +51% (eager tax); the cudagraph-preserving **Path-1 in-graph node injection** (binary-confirmed seam, demonstrated
as a toy-graph primitive in Track B, not yet built for a real vLLM graph + GEMM-output-param wiring) is the named step
that would carry the detector inside the cudagraph and remove the eager tax. **Caveat (denominator):** the +3.1%
detector-marginal is measured against the *eager* baseline (629 tok/s); on a cudagraph host (1289 tok/s, 2×) the same
absolute recompute cost would be a **larger fraction (~6%), and is UNMEASURED** — a cudagraph-frame detector marginal is
exactly what Path-1 would let us measure. So Path-1 removes the 51% eager tax but the detector's own cost in the
cudagraph frame (~6%, unmeasured) may itself exceed 3% at N=45 and would need a larger N or a cheaper compare to hit <3%.
