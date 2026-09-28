# vLLM DECODE actuation gap — DIAGNOSIS (handled=0 / 11658 passthrough)

**Date:** 2026-05-31. **Type:** READ-ONLY diagnosis (no build, no commit, anchors unchanged). Resolves
WHY Marlin/Koopman substitute 0 on stock-vLLM decode, BEFORE any fix. **Scope:** vLLM DECODE only
(prefill→v1.5, 85%-magnitude→multi-GPU — adjudicated, not re-opened). Every claim file:line + counter
cited; "engages" ≠ "substitutes"; "deferred" ≠ "impossible".

**Headline:** the `handled=0/11658` is **NOT** the "D10 LT-route" cause the engagement map hypothesized,
and it is **NOT a single cause**. The 11658 are `cublasGemmEx` that **reached the dispatch** (Cause 1
absent for them); they were declined because **the only enabled actuator in that test was Koopman, which
is fp16-substitution-only and observe-only on bf16 — and vLLM decode is bf16; Marlin (which CAN substitute
bf16) was env-OFF** (Cause 2). Underneath that: **Koopman has a genuine rank-wall** on general decode
(Cause 3 — a research wall) and **Marlin engages at B=1 but regresses** (Cause 3 — a regime wall, its win
is B≥8). The honest verdict: the GEMM-substitution lift's home is **batched decode (B≥8) via the POOL
path**, not single-stream B=1 vLLM decode.

---

## 1. THE DECODE GEMM SHAPE PROFILE (the 11658)

vLLM TinyLlama-1.1B V1 decode, 128-token burst (`WEEK_14_FOLLOWUP_OPTION_2_STEP_0_VLLM_WORKER_HOOK.md:30`):
- **M (batch) = 1** — single token per decode step (`:30`; `WEEK_12_STEP_5_D14_BACKFILL.md` "TinyLlama B=1 decode").
- **dtype = bfloat16** (CUDA_R_16BF), NOT fp16 — confirmed by the Koopman fp16-only gate filtering all
  11658 (`WEEK_14_FOLLOWUP_OPTION_2_STEP_0_5_KOOPMAN_REACHABILITY.md:117`; and logically: Koopman's
  fp16-only early-exit + `koopman_total=0` ⟹ the dtype is not fp16).
- **Three K/N shapes** (`WEEK_14_STEP_3_C_LMHEAD_VALIDATE.md:103-112`): FFN gate/up (K=2048, N=5632), FFN
  down (K=5632, N=2048), LM-head (K=2048, N=32000). **All three PASS the Marlin shape gate** — N≥1024,
  K≥1024, N%64==0 (5632/64=88, 2048/64=32, 32000/64=500), K%128==0 (2048/128=16, 5632/128=44)
  (`cipher_rt_marlin_actuator.c:265-280`). (Attention q/o K=N=2048 also pass; k/v N=256 GQA would be
  shape-declined — minor.)
- They reached the matmul dispatch: `cipher_rt_cublas_shim_calls 0→11658` (`WEEK_14_FOLLOWUP:43`) ==
  `MATMUL: exit totals calls=11658 handled=0 passthrough=11658 (actuators=1)` (`:47`).

## 2. WHICH CAUSE(S) — for the 11658, counter-cited

### Cause 1 (dispatch surface) — **ABSENT for the 11658**; a SEPARATE population is UNTESTED
- `cublasGemmEx` **routes** to the actuator dispatch: `cipher_rt_cublas_shim.c:231`
  `cipher_rt_matmul_dispatch(&call, g_real_gemmEx)`. The 11658 (`cipher_rt_cublas_shim_calls` =
  `g_shim_calls`, the cublasGemmEx counter, `:159,337-340`) == the dispatch's `calls=11658` → **all 11658
  reached the dispatch.** The shape log (§1: FFN+LM-head decode shapes) confirms the 11658 ARE the real
  decode GEMMs, not a side population. **⇒ Cause 1 (intercepted-but-unrouted) does NOT explain the 11658.**
- The "D10 LT-route" hypothesis applies to a **different surface**: `cublasLtMatmul` IS intercepted but
  **count+passthrough only**, no dispatch (`cipher_rt_cublas_shim.c:259-311`; separate counter
  `g_lt_shim_calls`, `:292`). Whether vLLM ALSO issues `cublasLtMatmul` decode GEMMs is **UNTESTED**
  (`g_lt_shim_calls` was not captured in WEEK_14 — NOT FOUND). If it does, those are unrouted (a real v1.x
  gap; the wiring pattern exists at `cipher_rt_cublaslt_variants.c:138-145`) — but it is **not** the
  documented 11658.

### Cause 2 (gate / which actuator) — **THE documented blocker**
- The dispatch has **no pre-actuator gate**; it iterates registered actuators
  (`cipher_rt_matmul_dispatch.c:88-126`). In WEEK_14, **`actuators=1` = Koopman** (CIPHER_KOOPMAN=1;
  **CIPHER_MARLIN was OFF** — `cipher_inject.c:62-66`, Marlin disabled at init `cipher_rt_marlin_actuator.c:337-356`).
- **Koopman declined all 11658 at its fp16-only dtype gate**, before its engine counter:
  `cipher_rt_koopman_engine.cpp:81` `#define CUDA_R_16F 2`; `:92,95` `g_skip_dtype` "pre-counter early-exit
  returns PASSTHROUGH; surfaces FP16-vs-BF16". vLLM decode is bf16 → all 11658 hit the early-exit →
  `koopman_calls_total=0` (`WEEK_14_FOLLOWUP_…_0_5:52-75`). **And even bf16 GEMMs that pass the gates are
  observe-only for Koopman** — `cipher_rt_koopman_engine.cpp:101-107` "bf16 path [is observe-only], per
  Anil empirical-first." **⇒ Koopman cannot substitute on bf16 decode by design (fp16-substitution-only).**
- **Marlin was never tested** (env-OFF). But Marlin HAS a bf16 substitution path
  (`cipher_rt_marlin_actuator.c:182-190` `if (call->Atype == CUDA_R_16BF)`), and for a bf16 decode workload
  the K.2 classifier sets **marlin_engage = (int4 OR bf16) = TRUE** (A3_SINGLE_TENANT_STREAM,
  `src/cipher_workload_detect.cpp:888`), so the classifier gate (`cipher_rt_marlin_actuator.c:154-156`)
  would **pass**; the shapes pass (§1). **⇒ Marlin WOULD engage+substitute on vLLM bf16 decode if enabled.**
  (Quantify: 11658/11658 declined by Koopman-fp16-gate; 0 tested for Marlin.)

### Cause 3 (actuator physics) — **the deeper wall behind Cause 2**
- **Koopman rank-wall (research wall):** even fixing the dtype gate, Koopman's prod-rank substitution
  fails on general decode — residual_ratio > 0.5 on WikiText at r=64 → sub=0
  (`W3_KOOPMAN_BF16_AUTOCAL_CLOSE_REPORT.md:17-20,79-82,128-132`). A genuine mathematical constraint; no
  dtype/routing fix makes Koopman substitute on general decode.
- **Marlin regime-wall:** Marlin **engages** at M=1 (churn test bf16 B=1 `sub>0`,
  `gate_churn.json`/`MARLIN_GC_ON_FREE_BUILD_REPORT.md`) but **regresses** vs vanilla at B=1 — it is
  batch-gated by design for its win (`cipher_rt_marlin_actuator.c:8-10,71` MARLIN_MAX_M_GATE; the B≥8
  designed regime, [[cipher-t45-substrate-marlin]]). At single-stream decode B=1 the INT4 dequant
  overhead dominates → engages-but-doesn't-help.

## 3. NAMED FIX PER CAUSE

| Cause | Fix | Est / verdict |
|---|---|---|
| **1 — LT-route** | NOT the 11658's blocker. SEPARATE: capture `g_lt_shim_calls` on a vLLM decode run; if vLLM issues cuBLASLt decode GEMMs, wire `cublasLtMatmul`→`cipher_rt_matmul_try_actuators` (pattern at `cipher_rt_cublaslt_variants.c:138-145`). | small build (~2-3 ED) **but does not explain the documented gap**; do only if `g_lt_shim_calls>0` on decode. |
| **2a — Koopman fp16-only gate** | add CUDA_R_16BF to Koopman's substitution path. | small — **but only EXPOSES the rank-wall (3); does not make Koopman substitute.** Not worth it alone. |
| **2b — Marlin env-OFF** | enable `CIPHER_MARLIN=on` + run a vLLM bf16-decode measurement. | **trivial (env, not a build)** — Marlin will engage+substitute on the bf16 decode shapes. **But see 3-regime.** |
| **3 — Koopman rank-wall** | none — physics/research (residual_ratio>0.5 WikiText r=64). | **WALL.** G4-Koopman on general vLLM decode is **v1.5/research**, not a build. |
| **3 — Marlin B=1 regime** | none for single-stream; Marlin's win is **B≥8 → cross-tenant POOL-batched decode**. | **REGIME wall** for B=1; the lift lives at batched decode (B≥8) via POOL. |

## 4. THE HONEST VERDICT — build vs wall

**The `handled=0/11658` is Cause 2** (the only-enabled actuator, Koopman, is fp16-substitution-only /
bf16-observe-only and vLLM decode is bf16; Marlin was OFF) — **not** a routing wall (Cause 1 absent for
the 11658) and **not yet a measured physics wall** (Marlin untested because OFF). Per actuator, on
stock-vLLM **single-stream B=1** decode:

- **Marlin — BUILD-able to ENGAGE, but a REGIME wall for *useful lift* at B=1.** Enabling Marlin
  (trivial, not even a build) makes it substitute on the bf16 decode shapes — but at **B=1 it regresses**
  (its win is B≥8). **So Marlin's GEMM-substitution lift on decode is delivered at BATCHED decode (B≥8)
  via the cross-tenant POOL path** (the G1-density lever, where the churn test already proved Marlin bf16
  engages) — **NOT** single-stream vLLM B=1 decode. **Named next:** enable Marlin + measure on
  POOL-batched (B≥8) vLLM decode, not single-stream.
- **Koopman — WALL (research).** bf16-observe-only by design **and** a rank-wall on fp16 general decode
  (residual_ratio>0.5). No dtype/routing fix makes it substitute on general decode. **G4-Koopman on vLLM
  decode is v1.5/research** (consistent with Goal 4 historically the weakest). Distinct from Marlin's
  regime wall: Koopman's is a *mathematical* wall (rank), Marlin's is a *batch-regime* wall.

**Founder distinction (precise):**
- G3 GEMM-lift / G4 Koopman on **single-stream B=1 vLLM decode** is **not a routing build** — the
  `handled=0` there is *correct/expected*, not a bug: B=1 is outside Marlin's regime and Koopman rank-walls.
- The GEMM-substitution lift's **real home is batched decode (B≥8) via cross-tenant POOL** — where
  Marlin's regime is met and Marlin bf16 engages (churn-proven). **v1's G3/G4 GEMM-substitution delivery
  is on the POOL-batched decode path, not single-stream vLLM B=1 decode.** This also matches G1: the
  density/batching lever IS where the weight-GEMM lift composes.
- **This corrects `V0_ENGAGEMENT_MAP_THREE_SUBSTRATE.md`'s "named next: D10 LT-route"** — the 11658 are
  routed `cublasGemmEx`; the real story is actuator-enablement (Marlin) + regime (B≥8 via POOL) + the
  Koopman rank-wall, NOT LT-routing. (LT-route remains a separate, untested, v1.x architectural item for
  any cuBLASLt decode population — capture `g_lt_shim_calls` to decide.)

**Net:** Marlin = **engages with an env-flip; useful only at B≥8 (POOL)** → a measurement, not a wall, on
the batched path. Koopman = **genuine research wall** on decode. Neither is fixed by LT-routing. Anil
decides: (a) measure Marlin on POOL-batched vLLM decode (the real G3 path), and/or (b) accept G4-Koopman
on vLLM decode as v1.5/research. No build, no commit, anchors unchanged.
