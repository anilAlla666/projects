# cuLaunchKernelEx substrate intercept + launch-signature auto-detect — SCOPE + DAY ESTIMATE

**Date:** 2026-05-31. **Type:** READ-ONLY scope (cite file:line; no build, no commit, anchors unchanged).
Goal: de-couple attention (and nvjet-GEMM) from per-framework symbol-keying by recognizing kernel TYPE
from the launch SIGNATURE at the CUDA driver-dispatch boundary, routing to an actuator. **Binding
constraint:** the intercept sits on the path EVERY kernel flows through (highest blast radius) → it MUST
be additive, default-OFF, and **byte-identical when recognition is OFF** — guaranteeable, not just
testable.

**Headline:** the **intercept is bounded** (the `cuLaunchKernel` shim already exists, observe+passthrough;
byte-identical-OFF is guaranteeable by construction). The **launch-signature classifier is the multi-week
wildcard** (a *truly framework-agnostic* attention signature from grid/block/shmem is research; the only
reliable per-type signal at launch is `cuFuncGetName` — a NAME, which is *still app-specific*). And
recognize→substitute **inherits the W.5 attention-math walls** (substitution kernel + KL gate unbuilt =
v1.5). ⇒ this build delivers **agnostic RECOGNITION plumbing + an nvjet hook**, **not** attention
substitution. **Recommend DEFER to v1.5, bundled with the W.5 attention-math substitution.**

---

## SCOPE (what exists vs net-new)

### 1. THE INTERCEPT — bounded; byte-identical-OFF GUARANTEEABLE by construction
- **Mechanism exists + reusable:** `cipher_rt_got_register(symname, trampoline, ...)`
  (`cipher_rt_got_patch.h:29`) + `cipher_rt_got_patch_apply` (`:36`) — the same GOT-patcher used for
  `cublasGemmEx` (`cipher_rt_cublas_shim.c:327`) and the attn 6-pattern.
- **`cuLaunchKernel` is ALREADY intercepted in the baseline** (observe + passthrough): the may13 layer
  hooks it via `cuGetProcAddress` (`src/may13_intercept/cipher_intercept_cudart.cpp:21` "Layer 1: Exported
  symbols (cuLaunchKernel, cudaLaunchKernel, etc.)"; `:244` the `cuLaunchKernel_fn` typedef), and
  `cipher_intercept_stubs.cpp:13` — *"normal cuLaunchKernel behavior — substrate's own counters fire
  normally"* (counter + passthrough). So the **intercept itself is mostly built**; the net-new work is a
  recognize-substitute hook on top.
- **Byte-identical-OFF guarantee:** add the recognize→substitute as a branch gated by a **default-false
  atomic flag** — OFF ⇒ branch-not-taken ⇒ the existing observe+passthrough baseline ⇒ **byte-identical by
  construction** (output, KL, and every actuator counter). The attn 6-pattern's naked tail-jmp
  (`W5_FLASHATTN_CLOSE_REPORT.md:32-36,49-50`) and the existing cuLaunchKernel passthrough are the proof
  this passthrough-by-construction approach holds. **Est: 2-4 ED** (reuse GOT + existing shim + the
  default-OFF gate).

### 2. THE LAUNCH-SIGNATURE CLASSIFIER — the wildcard (days if app-specific name, WEEKS if truly agnostic)
- **Signals available at launch:** grid/block/shmem (from `cuLaunchKernel`/`cuLaunchKernelEx` args);
  thread-count streaks (`cipher_rt_sm_packer.c:30-31` `g_small_launches`/`g_longest_streak`, a *crude*
  <4096-thread "small kernel" signature); and **`cuFuncGetName`** (resolved, per-fn-pointer name cache —
  `src/cipher_workload_detect.cpp:171,322-324`, `cipher_cupti.c:176`).
- **The honest tension (the range driver):**
  - **`cuFuncGetName` (the kernel NAME) → fast (~days) but STILL APP-SPECIFIC.** FA3's name
    `run_mha_fwd` ≠ SGLang's/TGI's attention kernel name. Recognizing by name is the 6-pattern problem
    moved one layer down — **not more substrate.**
  - **Grid/block/shmem/arg fingerprint → framework-agnostic but RESEARCH.** FA2/FA3/FlashMLA/PagedAttn
    grid/block **vary by seq_len/head_dim/batch/version** and **overlap** with other kernels; there is no
    documented stable agnostic fingerprint. The arg `void**` is opaque without per-kernel param-struct
    knowledge (e.g. `Flash_fwd_params`) — which is itself app-specific.
  - **⇒ Named uncertainty driver:** "is a robust *framework-agnostic* attention signature achievable from
    grid/block/shmem, or must recognition fall back to `cuFuncGetName` (app-specific)?" Unknown until
    prototyped. **Est: 2-6 WEEKS (research), high uncertainty.** No prior agnostic-signature work exists
    (the only fingerprints are GEMM-shape (TC probe / observe_gemm) and thread-count (sm_packer) — neither
    identifies "attention").
- **Fail-safe (the correctness constraint):** the classifier MUST be **high-precision / conservative** —
  unrecognized OR low-confidence ⇒ **passthrough, never substitute**. A false-positive (substitute the
  wrong kernel) is a correctness regression. Default = passthrough; substitution only on a confidence
  threshold. The fail-safe DEFAULT is easy; the high-precision *agnostic* recognition is the research.

### 3. RECOGNIZE → SUBSTITUTE — delivers the HOOK, NOT attention substitution
- Routing a recognized launch to an actuator reuses the dispatch (~3-5 ED). **But it inherits the W.5
  walls:** the attention **substitution kernel** (FA3-shape-matched/MLA/PagedAttn) + the **per-pattern KL
  acceptance gate** are UNBUILT (`W5_FLASHATTN_CLOSE_REPORT.md:52-53,96-98`, v1.x —
  `V0_W5_ATTENTION_SUBSTITUTION_DIAGNOSIS.md`). **So this build DELIVERS:** framework-agnostic
  *recognition* + routing (de-couples attention from per-framework symbols) + an nvjet-GEMM substitution
  *hook*. **It does NOT deliver:** attention-math substitution (kernel + KL oracle = v1.5). "De-couples
  recognition" ≠ "delivers substitution."

### 4. nvjet FALLBACK — catchable at the launch boundary, gated on the same classifier
- nvjet kernels launch via the driver ⇒ surface as `cuLaunchKernel` launches and ARE observed:
  `cipher_cupti.c:188` *"CUPTI catches their downstream launches here."* CUPTI can only observe (profiling)
  — but the **GOT-patch trampoline CAN substitute** the launch. So an nvjet prefill GEMM that bypasses
  `cublasGemmEx` (`D9_CIPHER_DELIVERY_REPORT.md:33-37`) is reachable at `cuLaunchKernel` for substitution
  **IF** recognized as a GEMM-of-shape-X — which is the **same signature-classifier research** (recognize
  GEMM from grid/block) plus a substitute kernel (Marlin exists). **nvjet-fallback = feasible-in-principle,
  gated on §2.**

---

## DAY-RANGE (broken down; uncertainty named)

| Component | Est | Confidence |
|---|---|---|
| Intercept + byte-identical-OFF gate (reuse GOT + existing cuLaunchKernel shim) | **2-4 ED** | HIGH (mechanism exists) |
| `cuLaunchKernelEx` ABI-faithful trampoline (complex `CUlaunchConfig`; segfault-class risk) | **2-3 ED** | MED (W.5 segfault precedent shows ABI bugs are real but gate-caught) |
| **Launch-signature classifier (agnostic recognition)** | **2-6 WEEKS** | **LOW — the wildcard; research** |
| Recognize→substitute routing (hook only; substitution kernel is v1.5) | **3-5 ED** | HIGH (routing) / N/A (kernel deferred) |
| **FULL REGRESSION MATRIX (universal-path; every actuator re-verified)** | **1-2 WEEKS** | mandatory, real ED |
| **TOTAL** | **~6-10 WEEKS** | **dominated by the classifier (2-6 wk) + regression (1-2 wk)** |

### THE FULL REGRESSION MATRIX (the binding constraint makes this real ED, not a footnote)
Because the intercept is universal, byte-identical-OFF must be re-proven **with the intercept installed +
recognition OFF**, across the ENTIRE prior surface — not just the standard set:
- **Output/KL:** every-prior-model KL=0 (4-family bf16 / the 9-cell K.1+W.1+W.2+W.3+W.5 close gate,
  `W5_FLASHATTN_CLOSE_REPORT.md:86`); OFF byte-identical vs vanilla (HARD STOP).
- **Every prior ACTUATOR fires identically (the universal-intercept-specific matrix):**
  - Marlin: `calls_handled` / `bf16_substituted` unchanged (W5 cites "Marlin bf16 20k+ subs" — must match).
  - VOLT: clock-lock unchanged ("VOLT 5/5 engage", `W5:87`).
  - KV-dedup: hit-rate / GiB-saved unchanged (N=2/N=4 measured).
  - **FA3 attention: intercept-count unchanged (5280/cell, `W5:79-82`)** — the launch intercept must not
    perturb the attn 6-pattern GOT-patches.
  - Machete/Koopman observe counts unchanged.
- **W7-11 microbenches** (`test_step3_{b0,b1,c}` + the resolver/ring suite as available) + **N=128 30-min
  soak** (no crash, no leak, coherence) — both with the intercept installed + OFF.
- **Per-launch overhead bound:** the universal path adds work to EVERY kernel; the OFF path must be a
  not-taken branch (~1 ns), and the soak must show no aggregate throughput regression beyond noise.

---

## WHAT IT DELIVERS vs NOT

- **DELIVERS (now, if built):** (1) framework-agnostic kernel-type RECOGNITION at the driver-dispatch
  boundary (de-couples attention from `_vllm_fa3_C` hardcoding → works on SGLang/TGI *if* the agnostic
  signature holds); (2) an nvjet-GEMM substitution HOOK (reach the GEMMs that bypass cuBLAS). **Both are
  PLUMBING.**
- **DOES NOT DELIVER:** attention-MATH substitution (the actual lift) — that needs the W.5 substitution
  kernel + KL acceptance gate (v1.5 quality research). Recognition without a substitution kernel
  substitutes nothing. **And if recognition must use `cuFuncGetName`, it is NOT even agnostic** — it's the
  6-pattern re-expressed, no substrate gain.

## BUILD vs DEFER — **DEFER to v1.5 (recommended)**

- This is a **substrate GENERALIZATION**, not a v1 deliverable. The v1 attention-HBM lever (KV-dedup
  footprint) is **already shipped** (`V0_ENGAGEMENT_MAP_THREE_SUBSTRATE.md` §1.3); V.0's gates engage via
  the EXISTING substrate (VOLT, KV-dedup, the vLLM-FA3 6-pattern intercept, the cuBLAS matmul) — **the
  agnostic recognition does not change any V.0 vLLM measurement** (vLLM is the validated workload; FA3
  already engages). **⇒ NOT a keystone before V.0.**
- Building recognition **without** the substitution kernel delivers nothing usable now; and the
  substitution kernel is itself v1.5 (W.5 walls). **They compose** — recognize-agnostically + substitute —
  so they should be built **together in v1.5**, with the nvjet-GEMM-substitute as the third member.
- **v1 ships honest:** "**substrate at GEMM (cuBLAS) / clock (VOLT) / KV-primitive (cuMemMap) / classifier
  (driver shapes); vLLM-validated attention adapter.**" The `cuLaunchKernelEx` agnostic-recognition
  substrate + attention-math substitution + nvjet fallback = the **v1.5 substrate-generalization bundle**.

## REGRESSION-RISK ASSESSMENT — guaranteeable vs testable

- **Byte-identical-OFF (output/KL/actuator-counts): GUARANTEEABLE BY CONSTRUCTION.** The recognize-
  substitute is a default-false-gated branch; OFF = the existing observe+passthrough baseline (which prior
  builds already include). No actuator's output or count changes when the new branch is not taken. This is
  the safe design (far better than testable-after).
- **Residual risks (TESTABLE, not eliminated by construction) — why the full regression is mandatory:**
  1. **Universal-path per-launch overhead** — even OFF, the gate-check runs on every kernel (the hottest
     path: millions of launches). Mitigated to a ~1 ns not-taken branch, but must be soak-measured (the
     existing shim already pays a counter cost; the new add must be negligible). **Testable.**
  2. **Trampoline ABI correctness** — `cuLaunchKernel`/`cuLaunchKernelEx` have complex signatures
     (`CUlaunchConfig`, grid/block triples, opaque `void**` params + extra); a faulty trampoline =
     segfault (the W.5 precedent: trampoline-shadowed-symbol → recursion → segfault, caught by the close
     gate as a HARD STOP, `W5:60-73`). **Testable; gate-caught, not construction-guaranteed.**
- **Net:** output-equivalence-OFF is guaranteeable by construction; the universal-blast-radius perf + the
  ABI correctness are testable — which is precisely **why the full regression matrix above is real ED and
  the binding constraint is honored by a default-OFF/tail-jmp design + an exhaustive every-actuator
  re-verification**, not by assertion.

**Anil decides:** build now as a keystone (≈6-10 wk, classifier-dominated, delivers plumbing not
substitution) — or **DEFER to the v1.5 substrate-generalization bundle** (recommended: not a V.0 keystone;
the v1 attention lever is already shipped via KV-dedup; recognition + substitution-kernel + nvjet compose
and should be built together). Read-only — no build, no commit, anchors unchanged.
