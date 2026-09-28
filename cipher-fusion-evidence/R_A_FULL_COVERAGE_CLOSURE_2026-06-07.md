# R.A FULL-COVERAGE CLOSURE — ALL SDC GAPS, ONE RUN

**Date:** 2026-06-08 (report name per spec).
**Discipline:** READ-ONLY. No production `.so` change (anchor `libcipher_rt.so` md5
`2edba0d2136f8ede4713d90a8f7cd55f` confirmed UNCHANGED at entry and exit). No application-layer edit,
no vLLM source edit, no monkeypatch. Measurement path uninjected (LD_PRELOAD / CUDA_INJECTION64_PATH
empty) except Gap 6's minimal CUDA-graph mechanism probe (no production `.so`, no real injection).
fp32 accumulation throughout. Clock locked 1980 (ran 1830 under load); reset `-rgc` at exit.
**Method:** real Mistral-7B-v0.1 shapes, saturated CUDA-graph serving harness (vLLM per Gap 6); Step-A
harmful fault class (fp16 top-exponent bit-14 flip → NaN/inf or finite |δ|≥2.76), plus Gap-3
attention-internal and Gap-5 deterministic-SM models. Every number from this run unless cited to
ra-e2e (the prior real-trace E2E anchor, +2.96% @N=45).
**Evidence:** `cipher-fusion-evidence/full_coverage/{gap1..gap6,wave2}_*/` + `_STATUS_AND_PREREG.md`
(pre-registered verdicts written BEFORE measurement).
**Verification:** 7-agent adversarial panel (read-only) — Gaps 1–5 + Wave-2 sound; Gap 6 had one
material finding (frozen-graph leg was asserted, not measured) — **now re-measured** (replay tracks a
new input but omits a post-capture kernel) — plus minor wording fixes (err-900 name, enforce_eager cost
labeled an estimate, Gap-1 residual byte figures), all applied below.

---

## EXECUTIVE VERDICT

**Full SDC coverage is NOT achievable as one <3% always-on detector.** The deployable closure is a
**unified PERIODIC (every N steps) recompute+ABFT detector** that catches **PERSISTENT** faults across
**linear-GEMM + non-GEMM + attention** (0 FP; propagation coverage measured). Full coverage requires
recompute **from the layer input** so non-GEMM corruption propagates to the GEMM-output compare — which
costs **r ≈ 2.6** (synthetic upper bound), well above the linear-only shipped detector's r=1.49. That
lands at **3.57% E2E @N=45 / 2.51% @N=64** (synthetic): the deployable point is **N in the 45–64 band
(latency ≤ N−1 = 44–63 steps)**, full coverage costing *more* than the +2.96% linear-only shipped
detector. (The real-trace E2E is lower — the real base includes attention the synthetic forward omits —
but was not re-measured this run.) Everything that requires **per-step** checking (transient single-step
coverage) **walls** on one
unified mechanism — the **substitution gap**: the cheap check exists only as a fused GEMM epilogue,
cuBLASLt cannot host a row-sum-over-N epilogue (Finding 1), and a custom GEMM that can is slower than
cuBLAS by more than the check saves. Two fault classes are intrinsically outside recompute
(softmax-internal finite corruption; deterministic compute faults) and need either acceptance or 2×
spatial redundancy. Detector hosting inside cudagraphed vLLM is structurally blocked (Gap 6).

Of the six gaps: **0 closed cleanly to "<3% per-step everywhere"; 3 CLOSED-WITH-BOUND; 3
WALL-WITH-MECHANISM.** Every wall is named with its driving mechanism — which, per the mandate's
honesty clause, is the deliverable, not a failure.

---

## PER-GAP RESULTS (Step-0 floor → measurement → pre-registered verdict)

### GAP 4 — non-GEMM output storage faults. Verdict: **decode CLOSED-WITH-BOUND / prefill WALL-WITH-MECHANISM**
- **Not subsumed:** a storage flip in a non-GEMM *output* (RMSNorm h, SiLU act, RoPE q/k) is read
  identically by the forward GEMM and by the GEMM-output recompute (same cached corrupted input) → both
  agree → blind. Catching it requires regenerating the op *from its input* = a distinct fault site / its
  own matrix row.
- **Coverage:** 40/40 injected non-GEMM flips caught, T=0, 0 clean FP.
- **Step-0 floor (Rule 4):** separate-pass activation re-read @2.97 TB/s → **decode 2.56%**, **prefill 5.59%**.
- **Measured (unfused):** +71.4% decode / +54.1% prefill. **Kernel-count wall CONFIRMED by profiling:**
  +2305 kernels, *identical for M=128 and M=512* (M-independent ⇒ count, not work), slope linear
  0.21/0.29 ms per checked layer, per-kernel ~2.9/4.0 µs (the H100 graph min-duration; graph removes
  launch gaps, not per-kernel duration).
- **Verdict:** decode bandwidth floor 2.56% < 3% (closes if fused to one streaming pass); prefill floor
  5.59% > 3% (walls even fully fused — MLP gate/up/act `[M,14336]` re-read ~2.5 GB dominates; only
  inline epilogue fusion overlaps it = the substitution gap). **Flip hit:** prefill floor >3% ⇒ WALL.

### GAP 5 — deterministic SM-localized compute faults. Verdict: **CLOSED-WITH-BOUND**
- **Mechanism proven:** green-context split (`cuGreenCtxCreate` + `cuDevSmResourceSplitByCount`) of 132
  SMs into **disjoint 64+64** sets (S_A ∩ S_B = ∅, verified; 4 stranded). A deterministic SM-localized
  fault (FAULTY_SM=64 ∈ S_A, ∉ S_B) is **caught — guaranteed** by recompute on the disjoint partition.
  This is the *only* catch for deterministic-compute faults, which recompute on the same hardware
  reproduces (and is therefore blind to).
- **Honesty crux (advisor):** same-SM recompute also "caught," but **by accident** — block→SM mapping is
  not stable (recurrence 0.0156 ≈ 1/64, roving); the principled catch is the disjoint partition.
- **Bounds:** (a) SM-localized only — shared L2/HBM-path faults recur on both partitions, blind; (b) the
  fault is **simulated** by an smid-conditioned flip — this demonstrates the spatial-redundancy
  *mechanism* for the class, not that real compute faults are SM-localized; (c) cost ≈ **2× SMs** (dual
  redundant execution) — a redundancy *mode*, not a <3% always-on detector.

### GAP 3 — attention QKᵀ / PV internal faults (the named Step-A blind spot). Verdict: **CLOSED-WITH-BOUND**
- **Coverage:** S-check (raw QKᵀ row-sum vs Q·colsum(K)) catches **score flips 200/200 = 100%**; O-check
  (O row-sum vs P·rowsum(V)) catches **PV-output flips 200/200 = 100%**. Margins ~10⁴–10⁶×; NaN/inf
  caught by an explicit guard.
- **0 clean FP over 100 runs** both regimes, with T = 8× the clean-roundoff floor. (Attention's checksum
  reference uses a *different* fp evaluation order than the output, so clean roundoff is nonzero — unlike
  the bit-exact GEMM recompute where T=0 — requiring a safety-factored threshold, not T=0.)
- **Bound / residual (pre-registered BLIND):** softmax-internal finite corruption (P probabilities;
  online-softmax `l_i`/`m_i` bookkeeping) scales O and the checksum reference identically → algebraically
  invisible. ~81–88% caught *incidentally* by fp-evaluation-order divergence (not guaranteed), ~12–19%
  genuinely blind. FlashAttn streaming verified: appended rowsum-V column survives online rescaling
  (PRESERVED); an `l_i` fault is BLIND (O wrong by 7e-2 while the check column matches).
- **Step-0 floor (Rule 4):** combined check FLOP 0.488% (analytical); realizable needs FA-epilogue
  fusion (rowsum-V as a checksum column); decode reductions must be incremental, not from-scratch.

### GAP 2 — per-step GEMM checksum to cuBLAS parity. Verdict: **WALL-WITH-MECHANISM**
- Row-sum identity `chk = Y.sum(1) == ref = x·(W.sum0)`, `s` precomputed once.
- **Step-0 floor (Rule 4):** FLOP 0.028% (checksum ~free arithmetically); bandwidth (separate-reduction
  re-read of X+Y) **decode 2.72% (<3%) / prefill 5.89% (>3%)**.
- **Measured (separate reduction over plain cuBLAS):** decode +76.8% / prefill +58.3% — the kernel-count
  wall (+2016 / +1824 kernels, ~3–5 µs each), nowhere near the bandwidth floor.
- **Wall mechanism:** to reach the bandwidth floor the row-sum must fuse into the GEMM epilogue, but
  **cuBLASLt cannot host row-sum-over-N (Finding 1: BGRAD reduces over K, not the output dim)** ⇒ a
  custom CUTLASS GEMM that can is slower than cuBLAS by more than the check saves (substitution gap:
  prior +4.1% gate/up, +10.1% decode GEMV). **No per-GEMM checksum is <3% in both phases over cuBLAS.**
- Coverage of *harmful* GEMM-output faults = Step-A-settled (200/200, same checksum the shipped detector
  uses); this run's quick re-confirm 21/30 = harmful caught, 9 benign below the fp16-output roundoff
  threshold (threshold-muddied, not a coverage result).

### GAP 1 — phase-matched per-step transient coverage (THE HEART). Verdict: **WALL-WITH-MECHANISM (hypothesis FALSIFIED)**
- Mandate hypothesis: match the check to the idle resource (compute-heavy check in memory-bound decode;
  bandwidth-heavy check in compute-bound prefill). **Falsified by direct measurement.**
- **Measured concurrent-stream overlap:** prefill **+58.7%** (vs serial +68.4%) — the check did *not*
  hide under 50% spare HBM; decode overlap **+183%**, *worse* than serial +163% — it collides for the
  scarce bandwidth. (Eager decode is launch-inflated; the clean serial figure is Gap 2's graph +76.8%.)
- **Step-0 roofline (Rule 4) does the work:** the check's bandwidth is only 5.6% of peak / 11.2% of
  spare in prefill, 2.4% of peak in decode (spare from assumed MEM 50%/79% util — *conservative*:
  granting generous spare and still failing strengthens the wall) — **bandwidth is not the limiter.** The limiter is
  **kernel-count** (~1800 tiny reduction kernels), which a second stream cannot co-schedule into the
  tensor-core-bound GEMMs. Phase-matching to the idle *resource* is therefore moot.
- **Wall:** the only fix — fuse the check into few/epilogue kernels so spare HBM becomes usable — is the
  substitution gap (Gap 2). Both phases wall on one mechanism (kernel-count → fusion → substitution gap).
- **Residual (stated, unbuilt):** a custom fused-to-~32-kernels bandwidth-bound check *might* overlap
  prefill's spare HBM (check traffic ~2.68 GB ≈ 0.9 ms at peak / ~1.8 ms at 50% spare, inside the ~15 ms
  compute) *if* SMs co-schedule it — needs a custom kernel + multi-stream graph capture
  (substitution-gap-adjacent). Not closed.

### GAP 6 — detector hosting in vLLM. Verdict: **WALL-WITH-MECHANISM (cudagraphed) / co-run via enforce_eager (caveated)**
- **Mechanism measured** (minimal CUDA-graph probe, no vLLM run): (1) **frozen graph (measured):** after
  capture, replay re-ran the captured op with a *new* input (ybuf=14) but did **not** re-run a
  post-capture kernel (zbuf stayed 300, not 700) ⇒ the captured op-set is frozen — an injected detector
  kernel launched after capture is never replayed; (2a) host-sync `.item()` during capture → capture
  invalidated (`cudaErrorStreamCaptureInvalidated`); (2b) raw `cudaHostAlloc` during capture → **err 900
  / `cudaErrorStreamCaptureUnsupported`** ("operation not permitted when stream is capturing"), which
  invalidates the capture — the exact injected-library failure ("vLLM died under injected cipher";
  matches the prior Koopman `cudaHostAlloc` err 900).
- **vLLM 0.20.2 source root-cause (cited):** decode cudagraph is **ON by default** (O2 →
  `FULL_AND_PIECEWISE`, `config/vllm.py:336,228`); capture at `cuda_graph.py:308-312` uses a dedicated
  graph pool on a single stream; replay is frozen (`cuda_graph.py:355`, no kernel insertion); external
  alloc/stream during capture trips stream-capture invalidation; **`enforce_eager`**
  (`config/vllm.py:895-901` → `cudagraph_mode=NONE`; capture skipped `gpu_model_runner.py:6050-6056`) is
  the only co-run path. Its throughput cost (**~10–30% decode** — the agent's estimate, *not* measured
  this run since vLLM was not run) is the price of disabling cudagraph.
- **Root cause:** an injected in-context recompute detector cannot be hosted in cudagraphed vLLM decode
  (can't add to a frozen graph; injecting during capture invalidates it; a separate process lacks
  in-context activations). The eager co-run path sacrifices vLLM's main decode optimization.

---

## WAVE-2 UNIFIED FULL-COVERAGE SCORECARD

Per-step coverage walls (Gap 1), so the deployable unified detector is **PERIODIC**: every N steps,
recompute the step **from its layer input** (fresh non-GEMM + cuBLAS GEMM) and compare GEMM outputs,
plus the attention ABFT checksums (Gap 3). Two claims — that this catches non-GEMM faults via
propagation, and its cost — were **measured** this run (advisor close-out; `wave2_cov_cost.py`), not
asserted.

**(A) Propagation coverage — MEASURED.** Clean recompute is bit-identical ⇒ **T = 0 ⇒ 0 FP by
construction**. Injecting Step-A flips into the *original* step's stored values, the recompute-from-input
GEMM-output compare catches: **non-GEMM h 12/20, non-GEMM act 17/20** (the class Gap 4 proved the
from-cached-h detector is *blind* to), GEMM **q 14/20**. The misses are *benign by the T=0 logic*:
not-caught ⟺ no checked GEMM output changed ⟺ the corruption was absorbed in fp16 rounding before any
checked output ⟺ the result is uncorrupted. So **every harmful non-GEMM fault that propagates detectably
is caught** (with attn-ABFT covering RoPE'd Q/K → attention). Propagation coverage is confirmed.

**(B) Real cost — MEASURED, and it raises N.** Recompute **from input** (full coverage) costs
**r = 2.59 decode / 2.62 prefill** — far above ra-e2e's r=1.49 for recompute *from cached h* (which is
non-GEMM-blind). Wall-weighted E2E `(r−1)/N`:

| N | Wall-weighted E2E | Latency (≤ N−1) | Verdict |
|---|---|---|---|
| 45 | **3.57%** | ≤44 steps | **over 3%** |
| 64 | **2.51%** | ≤63 steps | under 3% |

These synthetic numbers are an **upper bound** (the synthetic forward is GEMM-heavy — no real attention —
so the recompute is a larger fraction of base than on the real trace; the real-trace E2E is lower but was
not re-measured this run). **Bottom line: full coverage costs more than the +2.96% linear-only shipped
detector** — the deployable point sits in the **N = 45–64 band (latency 44–63 steps)**, and the confident
"<3% at N=45" of the linear-only detector does *not* carry over to full coverage; reaching <3% needs N
toward 64. **The per-step (transient) unified detector does NOT fit <3% at any N** — it walls on the
substitution gap.

---

## COVERAGE MATRIX (fault class × caught? / latency / cost / mechanism)

| Fault class | Caught? | Latency | Cost | Mechanism / bound |
|---|---|---|---|---|
| Linear-GEMM output, **persistent** | ✅ | ≤44 (N=45) | **+2.96%** | periodic recompute-from-cached-h (shipped R.A, linear-only) |
| Non-GEMM output (RMSNorm/RoPE/SiLU/residual), **persistent** | ✅ harmful (12–17/20; misses benign, T=0) | ≤44–63 | **3.57% @N=45 / 2.51% @N=64** | propagation to GEMM compare in recompute-**from-input** (Wave-2) |
| Attention QKᵀ score, **persistent** | ✅ 100% | ≤ N−1 | +0.01% | S-check (Gap 3) |
| Attention PV output, **persistent** | ✅ 100% | ≤ N−1 | +0.01% | O-check (Gap 3) |
| Linear-GEMM output, **transient single-step** | ⚠️ ~1/N | — | <3% **WALLS** | per-step needs fused epilogue = substitution gap (Gap 1/2) |
| Non-GEMM output, **transient single-step** | ⚠️ decode only | — | prefill **WALLS** | decode floor 2.56% (fused); prefill 5.59% bandwidth (Gap 4) |
| Attention **softmax-internal finite** (P, l_i/m_i) | ❌ ~12–19% blind | — | — | scales O & ref identically — algebraic blind (Gap 3) |
| Deterministic **SM-localized** compute fault | ✅ (simulated) | 1 step | **~2× SMs** | disjoint-SM green-ctx redundancy (Gap 5) |
| Deterministic **non-SM-localized** compute fault | ❌ | — | — | recompute reproduces it; spatial redundancy doesn't isolate it |
| Detector hosting in **cudagraphed vLLM** | ❌ | — | — | frozen graph + capture invalidation (Gap 6); eager co-run −10–30% |

✅ caught in-band · ⚠️ partial/walled on cost · ❌ uncovered residual.

---

## HONEST RESIDUAL — what fraction of the SDC space is now covered

**Covered (deployable, persistent-within-N, 0 FP):** all *harmful persistent* SDC in linear-GEMM outputs,
non-GEMM outputs (caught via propagation, measured), and attention QKᵀ/PV — i.e. the dominant
storage/transient-into-state fault surface of the forward pass. This is a strict superset of the shipped
R.A detector (which covered only persistent linear-GEMM and is *blind* to non-GEMM outputs). **The added
coverage is not free:** catching non-GEMM faults requires recompute-from-input (r≈2.6 vs the linear-only
r=1.49), which lifts E2E to **3.57% @N=45 / 2.51% @N=64** (synthetic upper bound) — so the unified
full-coverage detector deploys at **N in the 45–64 band (latency 44–63 steps)** to hold <3%, costing
*more* than the linear-only +2.96%, not ~zero marginal. Attention ABFT itself is the only ~free add
(+0.01%).

**Residual (NOT covered by any <3% always-on detector), each with its named wall:**
1. **Transient single-step SDC** (linear or non-GEMM-prefill) — per-step checking walls on the
   **substitution gap** (kernel-count → fusion → custom GEMM slower than cuBLAS). Coverage stays ~1/N.
2. **Softmax-internal finite corruption** — algebraically blind to the attention checksum (~12–19%);
   only NaN/inf and score/PV-output flips are caught.
3. **Deterministic non-SM-localized compute faults** — recompute reproduces them; disjoint-SM redundancy
   (Gap 5) catches only the SM-localized subset, at 2× cost, and only for a *simulated* fault model.
4. **In-context detection inside cudagraphed vLLM** — structurally blocked; only an `enforce_eager`
   co-run hosts it, at −10–30% decode throughput.

**Bottom line:** "catch ALL SDC" resolves to *"catch all harmful PERSISTENT SDC across GEMM + non-GEMM +
attention within N (N in the 45–64 band, latency 44–63 steps, <3% only toward N=64 because full coverage
needs recompute-from-input at r≈2.6); transient single-step, softmax-internal finite, and deterministic
non-SM-localized faults remain walled with named mechanisms."* The single recurring wall is the
**substitution gap** — every per-step path that would close ALL classes requires a fused GEMM epilogue
cuBLASLt cannot host and a custom kernel slower than cuBLAS.

---

## GUARDRAILS (exit state)
- Production `.so` md5 `2edba0d2136f8ede4713d90a8f7cd55f` — **UNCHANGED** (verified at exit).
- No production `.so` injected on any measurement path (Gap 6 used a minimal CUDA-graph probe only).
- Clock reset `nvidia-smi -rgc`; no compute procs left; no tags moved.
