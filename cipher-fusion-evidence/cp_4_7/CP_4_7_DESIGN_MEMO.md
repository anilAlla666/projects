# CP 4.7 — kernel-fusion throughput lever — DESIGN MEMO

**Date:** 2026-05-16 (closed 2026-05-17). **Status:** **CP 4.7 CLOSED at memo —
§6 adjudicated to path 7.2 (close at memo).** This memo is the canonical
CP-4.7 artifact; no build was performed. One cheap characterization run *was*
authorized and performed during memo writing (the launch-count spot-check,
§1.3); it grounds the bound in measured reality. Per Phase-4+ engineering-marvel
discipline (CP 4.4 / CP 4.6.5+6) the memo opens with mechanism characterization
and bound calculation; here the bound was decisive enough to close the CP
without a build. Closure note: §9.

**Headline finding.** Two independent lines of evidence — a measured launch
topology on the production build, and a direct may13 fusion measurement —
agree: the kernel-fusion lever is a **characterized null in the agentic B=1
decode regime**. Predicted lift **+0.1 % to +0.9 %**, inside n=5 measurement
noise. The strongest single argument is **hook reachability** (§2): the cuBLAS
GEMM dispatch path the CP scope specifies sees **8 % of the per-token launch
surface**; the fusion targets are the other 92 %. §6 puts the paths to the
user; the memo recommends **7.2 (close at memo)**.

**Scope.** Migrate the canonical kernel-fusion actuator into `cipher_rt_phase4`
as a new `cipher_rt_*` module, wire it into the cuBLAS GEMM dispatch path,
validate throughput lift on top of the CP 2.4 composed stack (Marlin INT4 +
DVFS + spec decode, 3.617× tok/W, 1.795× tok/s; Marlin-arm absolute
**52.5 tok/s = 19.05 ms/token**).

**Anchors held:** kmod 0.4.8 (`E427CAFA4E94D548233DC7A` / `e2f50452`),
libcipher_v2 `86618c30`, libcipher_rt `c2c5d313`. A build would link
`cipher_rt_fusion.o` → new libcipher_rt anchor; `c2c5d313` preserved as
rollback, new anchor recorded only on a successful close.

---

## 1. Launch-overhead characterization

### 1.1 Mechanism

A kernel launch carries CPU-side enqueue overhead (~3–5 µs nominal on H100).
B=1 decode issues many launches per token; fusing adjacent kernels (RMSNorm,
activation, residual) eliminates launch *sites*. The lever earns **only to the
extent that per-launch overhead sits unhidden on the critical path** — if the
CPU enqueues ahead of the GPU (deep launch pipeline), the overhead is hidden
and removing launch sites buys nothing.

### 1.2 Bit-exact

Fusion changes *kernel boundaries*, not arithmetic — a fused RMSNorm+scale
computes the same values as the 6-kernel decomposition. Bit-exactness is a
hard correctness gate (§5), not an assumption.

### 1.3 Measured launch topology — c2c5d313, this pod

`cp47_launch_probe.py`, Mistral-7B fp16 B=1 decode under the production
substrate (`CUDA_INJECTION64_PATH=libcipher_rt.so`, c2c5d313), 256 measured
decode steps + prefill + 8 warmup = 265 forward passes. Substrate CUPTI +
matmul-substrate counters at teardown:

| Counter | Total | **Per forward pass** |
|---|---|---|
| All kernel launches (`DIAG-T4.2.4d total_launches`) | 738,989 | **2,789** |
| cuBLAS GEMM dispatches (`MATMUL calls`) | 59,625 | **225** (7.0 / layer × 32 ✓) |
| Attention SDPA (`cipher-attn tramp_calls`) | 8,480 | **32** (1 / layer ✓) |
| **Elementwise / reduce / other** (remainder) | — | **≈ 2,532** |

**fp16 is the right probe for launch *count*.** The launch topology — and
specifically the elementwise/reduce kernels that are the fusion targets — is
dtype-invariant; Marlin INT4 only swaps the GEMM kernel itself (one
cublasGemmEx-equivalent either way). The probe's own wallclock (32.8 tok/s,
30.5 ms/token) is fp16-eager per-step Python overhead and is **not** the bound
denominator — that is CP 2.4's measured Marlin INT4 **19.05 ms/token** (§3).

**The fusion surface.** Of 2,789 launches/token, **225 (8 %)** are GEMMs and
**32 (1 %)** are attention; the remaining **≈2,532 (91 %)** are the
elementwise/reduce kernels — RMSNorm (the may13 probe found PyTorch eager
decomposes one RMSNorm into 5–6 kernels), RoPE, SiLU, residual adds, KV-cache
index/copy, dtype casts. **These 91 % are the fusion targets.**

---

## 2. Fusion-eligibility — the hook-reachability bound (lead argument)

The CP scope specifies wiring into `cipher_rt_matmul_dispatch.c` — the cuBLAS
GEMM dispatch path. **That hook fires only on `cublasGemmEx` calls.** It
therefore sees **225 of 2,789 launches/token = 8 % of the surface.** The 2,532
fusion-target launches (91 %) are ATen elementwise/reduce kernels — they never
pass through the cuBLAS shim and are **structurally invisible** to the
specified hook.

Within the 8 % it *can* see, GEMM fusion is epilogue/prologue fusion:

| Fusion site (directive's list) | Reachable from the GEMM hook? |
|---|---|
| RMSNorm + Q/K/V projection | **No** — RMSNorm is a *prologue*; cuBLAS/cublasLt has no prologue-normalization fusion, and the RMSNorm kernels have already been enqueued before `cublasGemmEx` is called. |
| GEMM + bias add | **N/A** — Mistral-7B and Llama-3.1-8B q/k/v/o/gate/up/down projections are **bias-free**. The site does not exist for these models. |
| GEMM + activation (SiLU) | **Partially** — only the `gate_proj` SiLU epilogue (~1 site/layer = 32/token). And the production GEMM actuator is **Marlin** (a custom INT4 kernel, not cublasLt) — reaching it needs a *Marlin epilogue variant*, new kernel work. |
| Attention score + softmax | **No** — attention runs through the SDPA/cuDNN path (CP 4.6.1: cuDNN is the H100 hot path), already a fused kernel; not a cuBLAS call. |

**Net reachable fusion surface from the specified hook: ≈32 launches/token —
1.1 % of the 2,789.** The bias-free architecture of the CP-relevant models
deletes the largest item on the directive's list outright.

The may13 driver-fusion discovery (`CIPHER_DRIVER_FUSION_DISCOVERY.md`) reached
the same wall from the other side: PyTorch eager-mode RMSNorm/RoPE/SiLU are
**not monolithic kernels** — they decompose into 5–6 lambda-named kernels with
version-unstable mangled names; driver-level fusion of them needs sliding-window
sequence-detection (its "Path A": 3–5 days, version-fragile), which is **not**
a cuBLAS-dispatch-hook actuator at all.

This is the same bound *shape* as CP 4.4's L2 budget: an architectural
constraint that pre-empts the build — there, 31.25 MiB ≪ weight set; here, the
hook reaches 8 % of the launches and 1.1 % after the bias-free models prune the
list.

---

## 3. Predicted lift bounds

### 3a. Empirical ceiling — may13 fusion measurement (direct evidence)

The canonical fusion actuator's *own* may13 measurement (`step8_fusion_only.json`
vs `step8_baseline.json`; the internal `mode` tag reads `full` but this is a
harness labelling bug — every non-baseline step8 file carries it; the filename
is the arm, and the value sits correctly between baseline and the fp8+fusion
`step8_full.json`):

| Arm | B=1 Mistral-7B tps | vs baseline | B=8 tps | vs baseline |
|---|---|---|---|---|
| baseline | 8.628 | — | 48.237 | — |
| **fusion-only** | **8.704** | **+0.88 %** | **48.127** | **−0.23 %** |

This is the **most aggressive real fusion** — full Python-level replacement of
the whole RMSNorm/RoPE/SiLU decompositions with single fused kernels (far more
than a cuBLAS-hook actuator can reach). It landed at **+0.88 % on B=1** (inside
noise) and **−0.23 % on B=8** (null/negative). may13's *absolute* tps (8.7) is
a heavy-config baseline and does not transfer to the c2c5d313 Marlin regime —
but the *relative* fusion lift does, and it is the empirical ceiling.

### 3b. First-principles cross-check — launch-overhead arithmetic

`lift ≈ (launches eliminated × effective-unhidden-cost-per-launch) / wallclock`.

Most of the 3–5 µs nominal launch overhead is **pipelined-hidden** at B=1
eager. Solving the may13 result for the effective unhidden cost: fusion-only
bought +0.88 % of may13's ~115 ms/token ≈ **1.0 ms saved/token**. The number
of launches may13 fusion eliminated (`ΔN_may13`) is not recorded, but is
bounded — between ~500 (RMSNorm collapse alone, 6→1 across 64 norm calls) and
~1,800 (RMSNorm + RoPE + SiLU + residual all collapsed). So the **effective
unhidden cost is 0.56–2.0 µs/launch** (`1.0 ms / ΔN_may13`) — sub-µs to low-µs;
the pipeline hides most of the 3–5 µs nominal.

Applying that range forward to the c2c5d313 Marlin regime (19.05 ms/token):

| Scenario | launches eliminated | predicted lift |
|---|---|---|
| Hook-reachable (GEMM epilogue, §2) | ≈32 | **+0.09 % to +0.34 %** |
| Hypothetical full elementwise fusion (all 2,532 → ~700) | ≈1,800 | ≈+5 % to +19 % *theoretical* — **not reachable** from the specified hook; may13's real full-fusion implementation of it measured only +0.88 % |

The hook-reachable range (+0.09–0.34 %) and the may13 empirical ceiling
(+0.88 %) bracket the honest prediction — every value is inside n=5 noise.

### 3c. Pre-registered prediction band

| Regime | predicted point | **pre-registered band** | interpretation |
|---|---|---|---|
| Mistral-7B Marlin INT4, B=1 decode | +0.1 % | **1.000 – 1.009×** | measured null |
| Llama-3.1-8B fp16, B=1 decode | +0.1 % | **1.000 – 1.009×** | measured null |

Both bands sit inside the CP 2.4 rig's n=5 matched-pair noise (~±1 % tok/s).
The Llama-3.1-8B launch topology is **assumed equivalent** to the measured
Mistral topology — same 32-layer / 4096-hidden geometry, identical eager-mode
kernel decomposition; cheaply confirmable if a build is selected, but it cannot
change the §2 hook-reachability bound, which is architecture-independent.

**Pre-registered regime verdict:** the lever does **not** earn measurable lift
in any CP-4.7-relevant regime — not B=1 decode (may13: +0.88 %), not B=8
batched (may13: −0.23 %). Kernel fusion's genuine domain is eliminating
*large-activation memory round-trips* (prefill / training with big tensors),
not B=1 decode launch overhead — and the lever that *does* address per-launch
overhead wholesale is **CUDA-graph capture** (a different mechanism: capture
the whole launch sequence, replay with ~zero per-launch CPU cost), noted as a
re-scope candidate in §6.3.

---

## 4. Measurement methodology (only if §6 selects a build)

- **Launch count** — substrate CUPTI `DIAG-T4.2.4d total_launches` + matmul
  `MATMUL calls`, pre-fusion vs post-fusion; verify fused paths measurably
  reduce launch count (mechanism gate).
- **Throughput** — Mistral-7B Marlin INT4 B=1 decode, CP 2.4 varied-prompt set,
  n=5 matched pairs (fusion on/off interleaved), VOLT@1000 MHz.
- **Composed gate run** — `vanilla` / `marlin` / `marlin+fusion` / `all-on`
  (marlin + fusion + DVFS + spec); tok/s and tok/W.
- **Correctness** — bit-exact + cosine-similarity vs unfused baseline (hard gate).

---

## 5. Pass criteria (pre-specified)

1. **Bit-exact output** vs unfused baseline — **hard gate.**
2. **Mechanism validation** — fused paths measurably reduce per-token launch
   count.
3. **Throughput within the pre-registered band** (§3c) — i.e. **confirmed
   null**; a measured lift materially outside the band fails engineering-marvel
   validation (the bound would be wrong — investigate before claiming).
4. **Composition** — composes multiplicatively with the CP 2.4 + CP 4.6.5+6
   stack (trivially true for a ~1.00× factor).

---

## 6. Open decision — adjudication needed

The §2–§3 bound is decisive. The directive specified a two-path memo; a third
(re-scope) is added for completeness, per CP 4.4 precedent. **The user picks
one before item 2.**

### 6.1 — Path 7.1: build (items 2–8)
Only justified if the bound cleared the noise band. **It does not.** Building
would migrate the actuator, wire the GEMM-epilogue hook, and measure the
predicted ~+0.1 % null — ~1 GPU-week to confirm a bound already known from a
direct may13 measurement. Defensible *only* if the campaign wants a
measured-null artifact on file for CP 4.7 specifically.

### 6.2 — Path 7.2: close CP 4.7 at memo *(recommended)*
The bound is characterized-null and is backed by a *direct prior measurement*
(may13 fusion-only +0.88 % B=1) plus a measured hook-reachability ceiling
(8 % → 1.1 % of the launch surface). No build-time measurement adds
decision-relevant information. Phase 4.7 closes on this memo; ~1 GPU-week saved;
calendar pulls in toward Phase 4 close (only 4.8 integration-soak would remain).
Same pattern as CP 4.4.

### 6.3 — Path 7.3: re-scope
The launches *are* a real cost (2,789/token) — but the lever that removes
per-launch overhead wholesale is **CUDA-graph capture**, not kernel fusion, and
it is reachable without the cuBLAS-hook limitation (it captures the entire
stream). A re-scoped "CP 4.7 — CUDA-graph capture actuator" would need its own
mechanism+bound memo. Noted as an option, not recommended here — it is a
different CP.

**Recommendation: 6.2.** The marvel is the bound, not the lever. The directive
itself instructed: *"if the bound says characterized-null, close at memo and
don't waste the GPU week."* The bound says exactly that.

CP 4.7 is the **second consecutive instance** of bound-first discipline
working as designed: CP 4.4's spec ran into a hardware budget (31.25 MiB L2 ≪
weight set), CP 4.7's runs into an architectural one (the specified hook sees
8 % of the launch surface, 1.1 % after bias-free pruning). In both, the spec
as written assumed a constraint the spec did not model, and the memo surfaced
it before the build week — not an isolated null, a pattern.

### Secondary (only if 7.1/7.3 proceed)
- **D1 — Marlin epilogue.** The reachable SiLU-epilogue fusion needs a Marlin
  INT4 kernel epilogue variant — new kernel work, and Marlin's split-K
  coordination may already touch these paths (user-flagged interaction).
- **D2 — Llama-3.1-8B fp16 arm** — include as a second measured null arm.

---

## 7. Build STEP — 8 items (items 2–8 gated on §6)

1. **Design memo** — this document. Hold for adjudication.
2. Audit the canonical fusion actuator (`cipher-may13-evidence`: `rope_fused_kernel.py`,
   `fused_resid_rmsnorm.py`, `run_mistral_fused.py`; discovery doc).
3. Migrate to `cipher_rt_phase4` as `cipher_rt_fusion.{c,h}`; register in the
   `cipher_inject.c` init chain; hook the GEMM-epilogue path in
   `cipher_rt_matmul_dispatch.c`; Makefile per CP 2.5 structure.
4. Correctness — bit-exact + cosine-similarity vs unfused. Hard gate.
5. Launch-count measurement — CUPTI pre/post; verify the mechanism.
6. Throughput — Mistral-7B Marlin INT4 B=1, n=5 matched pairs, VOLT@1000 MHz;
   four-arm composed gate run.
7. Composition verification — update the composed scorecard.
8. Report — `CP_4_7_REPORT.md`, CP 2.4 / 2.5 / 4.4 / 4.6.5+6 register.

**Per the directive and campaign discipline (design-memo→approve→build):
HELD now, after item 1, before item 2.**

---

## 8. Calendar & anchors

| Path | Cost |
|---|---|
| 7.1 build | ~1 week |
| 7.2 close at memo | ~0 (this memo is the close) |
| 7.3 re-scope | fresh memo, calendar resets |

**Anchors:** kmod 0.4.8 (`E427CAFA4E94D548233DC7A` / `e2f50452`), libcipher_v2
`86618c30`, libcipher_rt `c2c5d313` (rollback point). No rebuild under 7.2.

**Concurrent housekeeping (not CP-blocking):** libcipher_v2 anchor decision
(`cc0479b8` vs `86618c30` diff) — 1–2 h, any session.

---

---

## 9. Closure note — CP 4.7 STEP closed at memo (§6 path 7.2)

CP 4.7 STEP closed at memo. The kernel-fusion lever does not earn measurable
lift at agentic B=1 decode regime — hook-reachable surface is ~1.1 % of total
launches under the specified cuBLAS GEMM dispatch hook, and the may13 canonical
fusion (far more aggressive Python-level fusion) measured +0.88 % B=1 /
−0.23 % B=8 — already in noise. Migration to `cipher_rt_phase4` deferred
indefinitely. Phase 4.7 closes via documented characterization rather than
measured null.

This is the second consecutive bound-first discipline closure (CP 4.4 +
CP 4.7) — the discipline is now established pattern. The marvel is the bound
that prevented the build, not the lever that did not earn.

**Items 2–8: not executed (path 7.2).** Anchors unchanged — no rebuild, no new
module linked: kmod `e2f50452`, libcipher_v2 `86618c30`, libcipher_rt
`c2c5d313`. The kernel-fusion actuator is **not** in `cipher_rt_phase4`. A
CUDA-graph-capture re-scope (§6.3) remains available as a separate future CP if
the per-launch overhead is to be addressed by the mechanism that actually
reaches it.

**CP 4.7 CLOSED 2026-05-17.**
