# Fault-Injection STEP B — FREIVALDS prefill SDC detection (a different mechanism)

**Date:** 2026-06-05  **Mode:** READ-ONLY diagnostic, **NO `.so` touched.**
**Anchor:** `/home/ubuntu/cipher_rt_phase4/libcipher_rt.so` md5 `2edba0d2136f8ede4713d90a8f7cd55f` — verified **UNCHANGED at entry and exit.**
**HW:** H100 80GB HBM3, SM clock **locked 1200 MHz** for ratios (reset to default at exit, see Guardrails). nvcc 12.8, arch sm_90a, CUTLASS 4.1.0 (tilelang-vendored). Every number is from this run.
**Files (in `cipher-fusion-evidence/step_b_freivalds/`):** `step0_floor.py/.json`, `harness_overhead.py/.json`, `tuned_gemv.cu`/`libtuned_gemv.so`, `harness_tuned.py/.json`, `overlap_test.py/.json`, `coverage_freivalds.py/.json`, `synthesis.py/.json`, baseline `libgemm_base.so` (the Step-B-prefill CUTLASS sm90 WGMMA GEMM).

---

## 1. Why Freivalds, and what it had to beat

Step B fused-ABFT FAILED `<3%` on every real Mistral prefill GEMM (gate/up +4.1%, q/o +7.4%, down +14.4%, k/v +16.8%; machinery-subtracted). Root cause was **structural**: the row-checksum reference `r[m]=Σ_k A[m,k]·wref[k]` is algebraically identical to the output row-sum `Σ_n C[m,n]`, so any such check must independently traverse all of `A` (M·K work) on a compute-bound GEMM. This step tests a check whose reference is **not** the output — **Freivalds**: verify `C[M,N]=A[M,K]·W[N,K]ᵀ` with a random `g[N]`:

```
u = Wᵀ·g  [K]      (K·N MACs, reads all of W)
v = A·u   [M]      (M·K MACs, reads all of A)   — the independent reference
Cg = C·g  [M]      (M·N MACs, reads all of C)   — projection of the output
flag if |v − Cg| > T
```

Exact arithmetic: `v ≡ Cg`; the clean residual is fp rounding only (fp32 accum, Step A discipline).
**Decode is done and ships** (fused checksum +1.1%, 200/200 coverage). This step is **prefill only.**

---

## 2. STEP 0 — analytical cost floor (written before any measurement)

The spec's **FLOP floor** = added MACs as a fraction of the GEMM's M·K·N MACs (dominant of `1/M` for u, `1/N` for v, `1/K` for Cg):

| shape | M | K | N | 1/M (u) | 1/N (v) | 1/K (Cg) | **FLOP floor** |
|---|---|---|---|---|---|---|---|
| k/v_proj | 2048 | 4096 | 1024 | 0.049% | 0.098% | 0.024% | **0.098%** |
| q/o_proj | 2048 | 4096 | 4096 | 0.049% | 0.024% | 0.024% | **0.049%** |
| down_proj | 2048 | 14336 | 4096 | 0.049% | 0.024% | 0.007% | **0.049%** |
| gate/up_proj | 2048 | 4096 | 14336 | 0.049% | 0.007% | 0.024% | **0.049%** |

The FLOP floor is `0.05–0.10%` — two orders under 3%, so the spec said "the step doesn't stop here." **But that floor is not the realizable cost.** Each check is a matrix-vector product with arithmetic intensity ≈ 1 MAC/element — **memory-bound**. Its time is `bytes/BW`, not `MACs/FLOPS`, i.e. `I* × (FLOP fraction)` where `I* = GEMM_FLOPS/BW` is the roofline ridge (≈ 295–330 on H100). The **bandwidth-aware floor** (the real prediction, measured peak read BW = **2.97 TB/s @1200 MHz**):

| shape | u=Wᵀg read | v=Au read | Cg=Cg read | **layout-(i) BW floor** (3 gemvs, vs CUTLASS base) |
|---|---|---|---|---|
| k/v_proj | 3.18% | 6.37% | 1.59% | **11.1%** |
| down_proj | 7.82% | 3.91% | 1.12% | **12.9%** |
| gate/up_proj | 6.50% | 0.93% | 3.25% | **10.7%** |

**Pre-registered prediction from Step 0:** the FLOP floor will *not* be realized; the checks pay exposed HBM bandwidth (10.7–12.9%), the same trap that sank fused-ABFT. The only sub-3% sub-term is the irreducible `v=A@u` for the widest-N shape (gate, 0.93%). **The measurements below confirm this and close every escape hatch.**

---

## 3. Method

Baseline GEMM = the real CUTLASS sm90 WGMMA kernel (`libgemm_base.so`, 54–84% of cuBLAS) **and** cuBLAS (`torch.nn.functional.linear`). Check matvecs timed as **fp16-read / fp32-accum** gemvs (deployable byte traffic; upcasting inputs would 2× the bytes). A tuned coalesced warp-per-row gemv (`tuned_gemv.cu`, rel-err vs fp32 ref ≤ 6e-7) gives the tuned number; the **hard byte-floor = bytes/BW_peak** is the unattackable lower bound. CUDA-event timing, 50–80 warmup, median over 200–300 iters, clock locked 1200 MHz (held under load, 80→435 W flat-clock). Overhead % = (GEMM + check) vs bare GEMM.

---

## 4. Measured overhead

### 4a. Primitives & layout band (vs cuBLAS base, the deployable denominator)

| shape | GEMM cuBLAS / CUTLASS | **v=A@u** (irreducible) | u=Wᵀg (per-step W-read) | Cg=C@g | **layout-(i) all-separate (UB)** |
|---|---|---|---|---|---|
| k/v_proj | 47.2 / 89.1 µs | 11.96% fl / 25.98% tn | 5.98% fl | 2.99% fl / 25.0% tn | **139.8%** |
| q/o_proj | 134.2 / 181.2 µs | 4.20% / 9.11% | 8.41% | 4.20% / 9.11% | **55.2%** |
| down_proj | 409.6 / 490.1 µs | 4.82% / 8.63% | 9.64% | 1.38% / 2.98% | **29.6%** |
| gate/up_proj | 417.2 / 599.9 µs | **1.35% / 2.94%** | 9.46% | 4.73% / 8.46% | **28.8%** |

`fl` = hard byte-floor (bytes/2.97 TB/s, no kernel beats it); `tn` = tuned warp-per-row gemv (occupancy-limited at 46–56% of peak — so the floor is the right lower bound). **The FLOP floor (0.05–0.10%) is confirmed un-realized; real cost is 28.8–139.8% (layout i).**

### 4b. The independent A-traversal is irreducible — and it is the same wall as fused-ABFT

`v=A@u` reads all of A (M·K) just like fused-ABFT's `r=A@wref`. **Freivalds with `g=𝟙`, fully fused, IS fused-ABFT** (`wref = Σ_n W = Wᵀ·𝟙 = u|_{g=𝟙}`). So the mainloop-fused realization is already measured: **gate +4.1%, FAIL** (cited, not rebuilt — layout (ii) correctly skipped on this equivalence). Freivalds-as-separate-GEMV only moves that A-read out of the WGMMA mainloop (FFMA contention) into a clean HBM gemv (bandwidth) — a different mechanism, **same wall**: only gate's v-term is `<3%` (1.35% floor / 2.94% tuned).

### 4c. Concurrent overlap — measured, and a closed red herring

Can u,v hide under the compute-bound GEMM (separate stream)? **Yes, partially** (event-based, idle GPU): hidden fraction of u+v = k/v **+80%**, q/o **+56%**, down **+41%**, gate **+71%** — the WGMMA GEMM saturates tensor-core *throughput* but not HBM or SM occupancy, leaving spare bandwidth. **Not load-bearing for the verdict:** (1) that spare bandwidth belongs to the next layer's GEMM in a saturated serving pipeline, so the serial number is deployable; (2) even granting full idle-GPU overlap, down still sits at **~+11.5%**; (3) gate — the only shape with any sub-3% term — passes on the *serial* number without needing overlap.

### 4d. No complete deployable config is <3% on ANY shape (gate is not an exception)

Fixed `g` amortizes the W-read (u) to free. The two **unavoidable** re-reads remain — A (for v) and C (for Cg) — and they **trade off** (gate: cheap A / expensive C; down: expensive A / cheap C):

| shape | **Path A** — cuBLAS GEMM (no subst. gap) + separate v,Cg = v+Cg | **Path B** — custom GEMM (Cg epilogue) + v = check + substitution gap |
|---|---|---|
| k/v_proj | 14.9% fl / 51.0% tn — **FAIL** | chk 6.3–13.8% + gap **+89%** — FAIL |
| q/o_proj | 8.4% / 18.2% — **FAIL** | chk 3.1–6.7% + gap **+35%** — FAIL |
| down_proj | 6.2% / 11.6% — **FAIL** | chk 4.0–7.2% + gap **+20%** — FAIL |
| gate/up_proj | 6.1% / 11.4% — **FAIL** | chk 0.9–2.0% + gap **+44%** — FAIL |

Path A's v+Cg **sums to ≥6% on every shape** (fixed-g removes only the W-read, not the A & C reads). Path B's check term can be ~2% (gate) but hosting the Cg row-reduction epilogue requires a custom GEMM (cuBLASLt cannot host it — Step B Finding 1, BGRAD reduces over K not N), whose substitution gap (CUTLASS 54–84% of cuBLAS) is **+20…+89%** — it fails on the GEMM replacement alone. **Gate fails unconditionally.**

---

## 5. Coverage (replay of Step A's harmful single-bit-flip set, 200 flips)

Step A injected a single bit flip at output element `C[m, chan]` (`δ = abs_delta`, or NaN/inf). Freivalds signal at that row = `δ·g[chan]` (vs the row-sum's `δ`). `T_freivalds` = per-shape max clean `|v−Cg|` over rows, fp32 accum, **same g used for T and signal** (0 clean false positives by construction; `T = 3.2e-3 … 1.8e-2` per shape).

| s (independent g) | compounded coverage |
|---|---|
| **1** | **200/200 = 100.00%** |
| 2 | 200/200 = 100.00% |
| 4 | 200/200 = 100.00% |

41/200 are NaN/inf (caught by any check). 159 finite, all caught at s=1 with margins **63× – 26M×**. Worst-case single-`g` miss probability (smallest δ=2.76/4.64, g~N(0,1)) = **0.23%** ⇒ per-step finite-class coverage ≥ 99.8%, **s=2 ⇒ >99.999%**. **Coverage is NOT the limiter; it compounds to ~1.** Caveats: T was synthetic-calibrated (random A,W), cross-checked against Step A's real row-sum T scale — margins stay >40×, conclusion robust. Fault model = output-side single-element C corruption (Step A's blind spot stands: no attention-internal QKᵀ / PV·V coverage).

---

## 6. PRE-REGISTERED VERDICT → **DO-NOT-PROCEED** (prefill, as a universal `<3%` detector)

Overhead exceeds 3% on **every** real prefill shape, in **every** complete config (Path A 6.1–14.9% floor; Path B +20–89% substitution gap), **gate included**. Coverage is high (100% @ s=1, compounds to ~1) — **not** the limiting factor.

**Driving factor — the A·u traffic cost (exposed HBM bandwidth), not the coverage/T separation.** Verifying a product requires an independent traversal of an input (`v=A@u` reads all of A, ∝ `I*/N` = 4.8% down … 12% k/v vs cuBLAS); plus, for cuBLAS-GEMM configs, the output projection `Cg=C@g` reads all of C (∝ `I*/K`). These two reads trade off and sum to ≥6% on every shape. On a compute-bound GEMM these touches are exposed bandwidth, not hidden compute, so the 0.05–0.10% FLOP floor is unrealizable. This is the **same wall as fused-ABFT, reached by a different mechanism** (separate-GEMV HBM bandwidth vs mainloop WGMMA/FFMA contention) — and the `g=𝟙` equivalence proves they are the same check.

- The spec's **random-g** Freivalds is additionally DOA (per-step W-read for u = 6–10% alone).
- **Gate** has a sub-3% *check term* (like decode's +1.1%) but — unlike decode (gap +8.7%, memory-bound GEMV, plausibly closeable) — gate's gap is +44% with no closing path: **the decode-parallel that does not close.**
- This is **not** the spec's PARTIAL branch (which was for "<3% holds but coverage needs large s") — here overhead, not coverage, fails.

**Consequence:** GEMM-output verification is exhausted for prefill. **Decode ships (+1.1%, the shippable detector path);** prefill reliability routes to periodic/sampled full recompute or statistical parameter monitoring. Protects the teacher-forced 100/100-FAULT=0 substrate. **Anil's call:** ship decode detector / accept prefill gap (recompute-based) / stop.

---

## 7. Guardrails

- Anchor `2edba0d2136f8ede4713d90a8f7cd55f` **UNCHANGED** at entry and exit; no `.so` touched, no tags moved.
- GPU clock **reset to default** at exit (`nvidia-smi -rgc`); post-reset state recorded below.
- No processes left; weights untouched. Every number above is from this run.

**Exit state (recorded at completion):** clock reset via `nvidia-smi -rgc` → SM idling **345 MHz**, max 1980, **applications clock 1980 MHz (unlocked default)** — no longer pinned at 1200. Anchor md5 `2edba0d2136f8ede4713d90a8f7cd55f` UNCHANGED. No compute processes resident.
