# Fault-Injection STEP B — END-TO-END: SDC detection cost in real serving

**Date:** 2026-06-05  **Mode:** READ-ONLY measurement, **NO `.so` touched, no production wiring changed.**
**Anchor:** `/home/ubuntu/cipher_rt_phase4/libcipher_rt.so` md5 `2edba0d2136f8ede4713d90a8f7cd55f` — **UNCHANGED entry+exit.**
**HW:** H100 80GB HBM3. Clock locked **1980 MHz** for this throughput run; ran **1830 MHz under sustained load** (power-cap, reported, reset to default at exit). nvcc 12.8 / sm_90a; peak read BW **3.02 TB/s @1830**.
**Detector insertion:** the per-GEMM check was **simulated faithfully by issuing the exact extra kernels** (`v=A@u`, `Cg=C@g` via the Step-B `libtuned_gemv.so`; periodic full recompute via cuBLAS) on the real Mistral-7B GEMM shapes inside a real, **saturated** serving step, with correct fp16-read/fp32-accum byte traffic. No vLLM source / no monkeypatch; vLLM's own engine-core would not start under the injected runtime, so the realistic-saturated loop is a CUDA-graphed weight-streaming decode step (eager HF decode is launch-bound, not saturated — see §2).
**Files (`cipher-fusion-evidence/step_b_e2e/`):** `f_of_M.py/.json` (the spine), `synth_decode_graph.py`+`synth_decode_B*.json` (saturated step), `overlap_phase.py`+`overlap_M*.json`, `coverage_spotcheck.py/.json`, `synthesis_e2e.py/.json`, `probe_baseline.py`, `dmon_*.txt`.

---

## 1. What this run adds (and carries forward as settled)

Every prior Step B number was a % of **one isolated GEMM on an idle GPU**. Settled and carried forward, not relitigated:
- **Decode** fused checksum +1.1% on the isolated decode GEMV; coverage 200/200. (isolated)
- **Prefill** GEMM-output checksum (fused-ABFT ≡ Freivalds) FAIL <3% of one isolated prefill GEMM (≥6% from A+C re-reads). (isolated)
- **cuBLASLt cannot host the checksum** (Finding 1, BGRAD reduces over K not output-N) ⇒ any *fused* check needs a custom GEMM ⇒ a **substitution gap**. This is load-bearing below.

This run answers the **open** E2E questions: (a) does decode +1.1% survive *continuous batching*; (b) is prefill detection, amortized over a real serving run where prefill is a minority of GPU time, under 3% of **end-to-end** throughput; (c) does spare HBM bandwidth exist under **saturated** serving so the check overlaps.

---

## 2. Saturated serving baseline (saturation evidenced, not asserted)

| phase | regime | step time | tok/s | SM% | **MEM%** |
|---|---|---|---|---|---|
| prefill (M=2048 tile) | compute-bound | 43.16 ms | 47,454 | 100 | **48%** |
| decode (B=128, CUDA graph) | memory-bound | 6.70 ms | 19,098 | ~100 | **87%** |
| decode (B=128, eager HF) | **launch-bound** | 18.93 ms | 3,380 | 100 | 52% |

Eager HF decode is launch-bound (4.3× the 4.4 ms weight-read floor) — measuring the detector there would **understate** it (extra kernels fill idle launch gaps). The CUDA-graphed decode removes launch gaps → saturated (MEM% 87%, memory-bound). Per-token cost: prefill **21.1 µs/tok**, decode **52.4 µs/tok** (B=128).

---

## 3. THE SPINE — f(M): detector per-GEMM cost vs row-count (a decode GEMM at batch B is a GEMM at M=B)

Floor = (extra HBM bytes)/BW ÷ GEMM time (realization-independent; numerator & denominator at the same 1830 clock ⇒ **ratios are clean** despite the power-cap). `v-only` = Cg as a free epilogue byproduct; `v+Cg` = Cg as a separate read.

| M (=batch B in decode) | v-only floor | v+Cg floor | v+Cg *measured separate* |
|---|---|---|---|
| 1 | 0.01% | 0.02% | 94% (launch-bound) |
| 64 | 0.63% | 1.33% | ~50% (launch-bound) |
| **128** | **1.20%** | **2.52%** | ~50% (launch-bound) |
| **256** | **2.16%** | **4.54%** | — |
| 512 | 3.11% | 6.56% | — |
| 2048 (prefill) | **4.02%** | **8.47%** | +10.2% (in-loop, bandwidth-bound) |

**Headline:** "+1.1% decode" is a **B=1** number. The check's A+C re-read grows ∝B, so the *check floor* crosses 3% at **B≈256 (v+Cg) / 512 (v-only)**. But the floor is **unreachable by any deployable architecture at decode batch sizes** (next section).

---

## 4. The substitution gap is a first-class E2E throughput cost — for BOTH phases

The f(M) floor assumes the check is **fused** into the GEMM (no extra launch). The two deployable architectures:
- **Separate kernels** (cuBLAS unchanged, no gap): launch-bound at decode M — `meas%` 94% at M=128; **in-loop +42–54%** measured (480 tiny per-linear gemvs, 1 warp/SM). A 1 MB A-read cannot amortize a ~5 µs launch. **≫3%.**
- **Fused** (reaches the floor): requires replacing cuBLAS, which **cannot host the checksum** (Finding 1) ⇒ substitution gap. Measured deployable: **decode +10.1% vs cuBLAS** (+8.7% gap + 1.1% check, spike); **prefill +20–89%** (CUTLASS 54–84% of cuBLAS).

So **no per-step config ships <3% E2E**: decode +10% (gap **closeable** — memory-bound GEMV, 92→100% BW) or +42% (separate); prefill +20–89% (gap **fatal** — compute-bound tensor-core GEMM) or launch-bound separate. Counting the gap symmetrically (it drops that % of tok/s on *every* GEMM in a throughput run) is the correction this arc's check-vs-deployable discipline demands.

---

## 5. Overlap under saturation (per-phase) — does idle-GPU spare bandwidth survive?

Idle-GPU showed 41–80% of the check hides. Under **saturation** the deployable fused check is **bandwidth-bound**, so its overlap potential = spare HBM = MEM% headroom:

| phase | MEM% | spare HBM | fused-check overlap |
|---|---|---|---|
| prefill | 48% | yes (~52%) | hides (consistent with idle-GPU) |
| **decode** | **87%** | **no** | **fully exposed** |

(My separate-kernel overlap measured 46% hidden in decode — but that is *launch-bound* kernels hiding in SM-idle slots, a mechanism the deployable *fused* check doesn't have; MEM% is the right evidence.) **Decode dominates serving wall-time and has no HBM headroom ⇒ overlap does not rescue the E2E number.**

---

## 6. The gap-free deployable path: periodic recompute over cuBLAS (config 3) — for BOTH phases

Recompute reuses the **existing cuBLAS GEMM** and compares — no custom kernel, **no substitution gap**. Overhead amortizes as check_cost/N; detection latency = N steps.

| N | prefill checksum/N (×p-share E2E) | full recompute/N (×phase-share E2E) | latency |
|---|---|---|---|
| 1 | 8.47% | 100% | 1 step |
| 4 | **2.12%** | 25% | 4 steps |
| 16 | 0.53% | 6.2% | 16 steps |
| 34 | 0.25% | **~2.9% (unified, both phases)** | 34 steps |
| 64 | 0.13% | 1.6% | 64 steps |

- **Prefill** (gap-free): periodic checksum **<3% E2E at N≥4** (latency 4 prefill steps); bit-exact full recompute <3% at N≈16–21 (latency 16–21).
- **Decode** (gap-free): per-step is gated (§4), so decode routes here too — unified full-step recompute every **N≈34** = **+2.9% E2E**, latency 34 steps, covers **both** phases (overhead is 100%/N independent of mix).
- **Coverage model:** "every N" catches **persistent** SDC within N steps (the spec's worst-case-corrupted-steps metric assumes persistence). A **transient** single-step fault in an unchecked step is caught w.p. **~1/N** — checksum coverage of the Step-A class is 100% *on the checked steps*, not across skipped steps.

---

## 7. Serving-mix amortization (config 2, prefill checksum every step), for reference

E2E = prefill per-GEMM (v-only 4.0% / v+Cg 8.5%) × prefill-wall-share (B=128 cost model):

| P / D | prefill share | cfg2 v-only | cfg2 v+Cg |
|---|---|---|---|
| 2048 / 128 | 86.6% | 3.48% | 7.33% |
| 512 / 128 | 61.7% | 2.48% | 5.23% |
| 512 / 512 | 28.7% | 1.15% | 2.43% |
| 128 / 512 | 9.1% | 0.37% | 0.77% |

Serving-mix amortization **does** pull the prefill *check floor* under 3% for decode-heavy workloads — but only the floor; the deployable cost still carries the §4 gap (fatal for prefill's compute-bound GEMM). So mix-amortization alone does not make per-step prefill shippable; periodic recompute does.

---

## 8. Coverage spot-check (full 200/200 settled prior; this confirms in-context)

Real Mistral layer-0 `gate_proj`, real activations, fp32 accum: clean **0 false positives** (T = max clean residual); one Step-A-class bit-14 flip (δ=203) → caught, **margin 23,254×**. Detector still flags inside the loop.

---

## 9. PRE-REGISTERED VERDICT → **PARTIAL**

**Driver: the substitution gap (blocks every per-step config) + recompute detection-latency (the gap-free deployable path).**

- **No per-step detection config ships <3% of end-to-end throughput deployable.** Both phases are gated on the substitution gap: decode +10.1% (gap *closeable* — memory-bound GEMV) ; prefill +20–89% (gap *fatal* — compute-bound GEMM). The f(M) *check floor* (decode <3% to B≈256; prefill amortized) is real but unreachable, because reaching it requires replacing cuBLAS, which cannot host the checksum (Finding 1). Separate-kernel checks avoid the gap but are launch-bound in-loop (+42–54%).
- **The gap-free <3% path is periodic recompute over cuBLAS, covering both phases**, at a stated detection latency: prefill checksum at **N≥4** (latency 4 steps) or unified bit-exact full recompute at **N≈34** (latency 34 steps, +2.9% E2E). This is the spec's "per-run not per-GEMM" cost model — and it is what the isolated DO-NOT-PROCEED missed.
- **Overlap does not rescue it:** the dominant decode phase is HBM-saturated (MEM% 87%, no spare); idle-GPU spare bandwidth is a prefill-only, non-dominant phenomenon.
- **Coverage is not the limiter** (100% settled + in-loop spot-check), with two caveats: output-side single-element fault model only (attention-internal QKᵀ/PV blind spot stands), and periodic recompute catches transient single-step SDC only w.p. ~1/N.

**Consequence / what ships:** per-step fused checksum is cheaper-in-principle but gated on the substitution gap for both regimes. **Decode is the closer-to-shippable regime** (gap closeable; if closed, per-step decode checksum is <3% E2E to B≈256). **Prefill — and decode until its gap is closed — route to periodic recompute over cuBLAS**, a gap-free <3% E2E detector at a detection latency the product must accept (≈4 prefill steps for Step-A-class checksum; ≈34 steps for bit-exact all-SDC). Protects the teacher-forced 100/100-FAULT=0 substrate. **Anil's call:** ship periodic-recompute (both phases, latency N) now / invest in closing the decode GEMV substitution gap for per-step decode / accept prefill at periodic-recompute latency.

---

## 10. Guardrails
- Anchor `2edba0d2…` UNCHANGED entry+exit; no `.so` touched, no tags moved, no production wiring changed.
- Clock locked 1980 for the run (ran 1830 under power-cap; f(M) ratios clock-clean — same clock num/denom); **reset to default at exit** (recorded below).
- No processes left. Every number from this run.

**Exit state:** `nvidia-smi -rgc` → SM idling 345 MHz, max/applications **1980 MHz (unlocked default)**. Anchor md5 `2edba0d2136f8ede4713d90a8f7cd55f` UNCHANGED. 0 compute processes resident.
