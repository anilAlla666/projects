# D.10 step-0 origin probe — ABANDON the cublasLt-interception track (it's unnecessary)

**Date:** 2026-05-29. Read-only diagnostic; **no fix code**; anchor unchanged `9fe23143`.
**The approved "D.10 next-level interception" fix is NOT NEEDED.** The step-0 origin probe
proved CIPHER already intercepts the big GEMMs, and the apparent "blind actuators" were a
**workload-dtype policy decline (fp16), not an interception gap.**

## Proof 1 — CIPHER already intercepts the big GEMMs (at cublasGemmEx)

LD_PRELOAD interposer backtrace of torch's actual big-GEMM call (in-container, B=8 Mistral):
```
cublasLtHSHMatmul  ← libcublas.so.13 (internal) ← libcublas cublasGemmEx
  ← libcipher_rt.so cipher_rt_matmul_dispatch ← cipher_rt_cublasGemmEx_impl   ← CIPHER IS HERE
  ← libtorch_cuda structured_mm_out_cuda ← torch mm
```
Torch calls the **public `cublasGemmEx`** (which CIPHER GOT-patches → `cipher_rt_cublasGemmEx_impl`
→ `cipher_rt_matmul_dispatch`). The *real* `cublasGemmEx` then **internally** dispatches to
`cublasLtHSHMatmul` — BENEATH CIPHER's passthrough. So `HSH_shim_calls=0` / `umbrella=0` was a
red herring: the big GEMMs reach CIPHER at cublasGemmEx; the typed cublasLt variant is libcublas-
internal, below the interception point. **There is no interception gap.**

## Proof 2 — `marlin_engage=0` on fp16 is correct policy, not blindness

`maybe_handle_marlin` declines at the K.2 classifier gate (`cipher_rt_marlin_actuator.c:140-146`)
when `workload_class != UNKNOWN && !p->marlin_engage`. The classifier sets
`p.marlin_engage = int4_weights_detected || bf16_weights_detected`
(`cipher_workload_detect.cpp:855-915`). The fp16 test → `int4=0, bf16=0` → `marlin_engage=0`
→ Marlin **correctly declines by design** (its INT4 lever is for int4/bf16 weights, not fp16).

## Proof 3 — on bf16, Marlin engages + substitutes through the EXISTING interception

B=8 **bf16** Mistral, d10 staging, **no LT_ROUTE**, `CIPHER_MARLIN=1`:
`int4=0 bf16=1 marlin_engage=1 marlin_bf16_obs=17997 marlin_bf16_sub=17996`
+ "MARLIN: GPU quant kernels compiled in 0.11 s". **Marlin observed AND substituted ~18000
GEMMs** via cublasGemmEx — no cublasLt interception, no D.10 routing, no LD_PRELOAD fix. The
W.2 correctness gate did not disable (no `actuator_disabled_correctness` banner).

## Consequences (cascading corrections)

- **D.10 (LT-ROUTE) is UNNECESSARY** — built (`ed130e7`, default-OFF, correct-by-construction)
  but solves a non-problem. Keep it dormant (default-OFF == vanilla); do NOT pursue the
  next-level interception fix. No anchor rotation.
- **D.7 "Marlin non-engage PARTIAL" resolves** — it used fp16 models → policy decline. With
  bf16/int4 (the real serving dtype), Marlin engages; D.7's per-family routing is now testable.
- **The "actuators blind in-container" framing was wrong** — Marlin/Koopman are reachable
  (interception works); they engage on the dtype/class the classifier policy enables.
- **Real serving dtype matters:** vLLM serves **bf16 / INT4**, not fp16. So in the actual
  product config the actuators engage. The fp16 test mis-set the dtype.

## Recommendation (STOP for Anil)

**Abandon the cublasLt-interception track.** Instead:
1. **Re-run the D.7 R-H1 inference-KL gate with bf16 (or AWQ INT4) models** — Marlin now engages
   (proven), so the ≥3–5-family per-`model_uuid` routing + KL=0 is finally exercisable. This is
   the real path to closing D.7-positive.
2. **Re-baseline the actuator-goal map at bf16/int4** — Goals 2/3 (Marlin) and 4 (Koopman,
   verify its `koopman_engage` policy) engage on the right dtype; they are NOT interception-blocked.
3. Keep D.10 committed but dormant (default-OFF); leave the anchor at `9fe23143`.

Net: a substrate-line "interception" build was approved on a misdiagnosis; the step-0 probe
(its required first step) caught it before any fix code. CIPHER's matmul interception works;
the lever is workload dtype + classifier policy, not cublasLt interception.
