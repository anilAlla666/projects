# Fault-Injection STEP B (spike) — fused-epilogue ABFT checksum overhead — INTERIM VERDICT

**Date:** 2026-06-05. READ-ONLY spike, **NO .so change** — anchor
`/home/ubuntu/cipher_rt_phase4/libcipher_rt.so` md5 `2edba0d2136f8ede4713d90a8f7cd55f`
unchanged. All numbers from this run (H100 80GB HBM3, SM clock locked 1200 MHz,
CUDA 12.8 / sm_90). Harnesses + kernel in `step_b_spike/`
(`kernel_gemv.cu`, `harness_m1.py`, `harness_prefill.py`, `bgrad_probe.cu`,
`m1_result.json`, `prefill_result.json`).

## Question (carried from STEP A)

STEP A settled: ABFT row-checksum **coverage = YES** (fp32-accum, 100% of harmful
SDC caught at intercepted linears) but **overhead = NO for verify-as-a-separate-op**
(decode +77/127%; even captured into one CUDA graph ≥10.6%, floored by the 9.54 µs/op
H100 min-kernel-duration × kernel count). STEP A's only sub-3% candidate was a **fused
GEMM-epilogue** checksum (row-sum free byproduct + reference matvec `A·(ΣW)` = 1/N of
GEMM FLOPs ⇒ 0.003–0.098% **analytical, not measured**). Step B asks: **does the fused
path actually hit <3% in practice?**

## What the ABFT check is

For `C[M,N] = A[M,K] · Wᵀ` (W row-major `[N,K]`, the intercepted linear):
- **output row-sum** `s[m] = Σ_n C[m,n]` — a byproduct of the GEMM (fp32 accumulator).
- **reference** `r[m] = Σ_k A[m,k]·wref[k]`, where `wref[k] = Σ_n W[n,k]` (precomputed
  once per static weight). Algebraically `s[m] == r[m]`; flag if `|s-r| > T`.

`r` uses an **independent W-reduction** (`wref`) so a W-load/compute SDC corrupts `C`
(→`s`) but not `r` ⇒ detected. **Coverage caveat:** A is the shared operand of both `s`
and `r`, so this row-checksum detects W-side / product-compute SDC (STEP A's fault
model), **not** input-side SDC on A; HBM storage SDC is ECC-covered on H100.

---

## Finding 1 — cuBLASLt cannot host the checksum (the substitution gap is unavoidable)

`bgrad_probe` (cu12.8, `libcublasLt.so.12.8.4.1`), decisive on header semantics **and**
empirics: cuBLASLt's `BGRADA/BGRADB` epilogues reduce over the **contraction dim K**
(producing input-operand sums `Σ_k A`, `Σ_k B`), **not** the output row-sum `Σ_n C[m,n]`
the checksum needs. `AlgoGetHeuristic` returns INVALID_VALUE (BGRADA) / NOT_SUPPORTED
(BGRADB) for all tested shapes, while the plain DEFAULT-epilogue matmul succeeds.
**⇒ cuBLASLt exposes no output-over-N reduction epilogue. The zero-substitution-gap
cuBLAS path does not exist; the detector requires a custom kernel — exactly the
"substitution gap returns" risk STEP A flagged.**

## Finding 2 — DECODE (M=1): fused checksum is cheap = MEASURED <3% ✅

Custom fp16-read / fp32-accumulate GEMV (`kernel_gemv.cu`) on the **real Mistral-7B
down_proj** weight (`[4096,14336]`), one launch, three modes
(0 = GEMV, 1 = +row-sum, 2 = +row-sum +distributed reference). The reference is computed
**distributed across the whole grid** (each thread a grid-strided K-slice → per-block
partials → out-of-band final reduce), so it piggybacks on the GEMV's W-bandwidth wall
with no serial tail and no extra launch.

| quantity | value | meaning |
|---|---|---|
| GEMV correctness | rel-err **0.0** vs cuBLAS | custom kernel exact |
| coverage join (STEP A down_proj harmful) | **40/40 caught**, margin **13.0M×** | ABFT detects all harmful flips |
| **fused checksum-delta (mode2 − mode0)** | **+1.1%** (3 runs: +0.89 / +1.01 / +1.34%) | **the headline — PASS <3%** |
| row-sum-only delta (mode1 − mode0) | +1.0–1.5% | output reduction nearly free |
| reference contribution (mode2 − mode1) | **≈ 0%** (mode2 ≈ mode1) | reference fully hidden under W read |
| substitution gap (mode0 − cuBLAS) | +8.7–9.1% @1200 (+4.5% @ boost) | custom GEMV = 92% of cuBLAS bw — **orthogonal kernel-quality cost** |
| fused-actuator total vs cuBLAS | **+10.1%** | = substitution gap + checksum-delta |

**Interpretation (honest, two framings):**
- **The fused checksum itself is cheap: +1.1%.** STEP A's fusion thesis is *vindicated
  for the memory-bound decode regime* — the reference `A·wref` is absorbed under the
  117 MB W read (mode-2 ≈ mode-1). The earlier spike numbers (+6.28% side-stream
  overlap, +38% serial-block-0) were **implementation artifacts**: an extra kernel
  launch (the 9.54 µs STEP-A floor) and a one-block serial reference tail. True
  in-launch fusion removes both, as predicted.
- **But a deployable decode actuator costs +10% vs cuBLAS today,** because Finding 1
  forces replacing cuBLAS with a custom GEMV that trails cuBLAS bandwidth by ~8%.
  Reaching a <3% *deployable* decode number is now a **GEMV-tuning** problem (close the
  92%→100% substitution gap), not a checksum problem.

## Finding 3 — PREFILL (M=2048): fusion is the gated tensor-core build (NOT measured)

Prefill GEMMs are **tensor-core-bound** — measured cuBLAS throughput **down 760 / v 433
/ gate 723 TFLOP/s** (note: `prefill_result.json` `tflops` field is mislabelled ×1000).
The checksum work (`s`=Σ_n C row-reduction, `r`=A·wref) is CUDA-core / reduction work.

- **Separate/overlapped paths empirically FAIL** (valid cuBLAS-baseline measurements):
  reference re-reads the **58 MB A** (memory-bound) ⇒ overlap **+6.2% (gate) … +48%
  (v_proj)**; naive separate-ops **+40–65%**. These are correctly above 3%.
- **A CUDA-core fused GEMM is not a valid prefill proxy** — it runs ~10× slower than
  cuBLAS tensor cores, so its checksum-delta would be artificially tiny and would not
  transfer. (Not measured, deliberately.)
- **Only <3% candidate = fuse into a real tensor-core GEMM** (CUTLASS): `s` via an
  epilogue row-reduction (CUTLASS EVT can host this — the lighter half); `r = A·wref`
  piggybacked on resident A-tiles in the mainloop (the custom, harder half).
  Analytical floor: `r` = M·K MACs = 1/N of GEMM = **0.007–0.024%**; `s` ≈ 1/(2K) =
  **0.003–0.012%** ⇒ **<0.1% if perfectly fused** (consistent with STEP A's 0.003–0.098%).
  **This number is analytical-not-measured** — measuring it **is** the Step-B
  production build.

---

## Verdict

STEP A's re-scope is **REFINED, not overturned**:

1. **Fused-epilogue checksum is cheap *where the operand is resident in-launch*** —
   decode **measured +1.1%** (<3%), reference fully hidden. The fusion thesis is real.
2. **The recurring tax STEP A named is the substitution gap, not the checksum.**
   cuBLAS can't host the reduction (Finding 1), so the actuator pays for replacing
   cuBLAS: +8–9% on decode (custom GEMV at 92% of cuBLAS), and for prefill it forces a
   **custom tensor-core CUTLASS kernel**.
3. **The prefill make-or-break is still open and is the gated build:** does a CUTLASS
   tensor-core GEMM with EVT epilogue row-sum + mainloop A·wref piggyback hit <3% in
   practice? Analytically yes (<0.1%); empirically unmeasured.

**STOP at the build. .so untouched.** Step-B production = (a) a CUTLASS tensor-core GEMM
with fused dual-checksum epilogue/mainloop (prefill), and/or (b) closing the decode GEMV
substitution gap. Both are kernel-engineering commitments = **Anil's call**, now backed
by the decisive decode measurement and the prefill analytical/empirical bounds.
Relates to engine fault work [[cipher-go1-increment4-both-regimes]] (the teacher-forced
"100/100 FAULT=0" substrate this detector would protect).
