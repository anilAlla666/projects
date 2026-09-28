# R.A KEYSTONE + PRODUCT INTEGRATION — the checksum-fused GEMM keystone, vLLM hosting, and R.A as a shippable feature

**Date:** 2026-06-08.
**Discipline:** READ-ONLY. No production `.so` change (anchor `libcipher_rt.so` md5
`2edba0d2136f8ede4713d90a8f7cd55f` confirmed UNCHANGED at entry and exit, re-verified after the vLLM runs).
No application-layer / cuBLAS-source / vLLM-source edit, no monkeypatch. Track-A/coverage measurement path
uninjected (a plain torch matmul shows no cipher banner); Track-B's vLLM runs used an interposition shim
(LD_PRELOAD, not the production `.so`) and explicitly EXCLUDED the cipher vLLM plugins for a clean tax.
**Clock:** locked 1980 via `nvidia-smi -lgc 1980`; under load the card ran 1440–1830 MHz (power/thermal, not
a flaky lock — applications-clock=1980, throttle=GpuIdle at idle); reset `-rgc` at exit. The headline GEMM ratios are
from the **interleaved (drift-canceling)** run (`interleaved_1980.json`), robust to the clock variation (cuBLAS and
custom timed per-rep back-to-back); a non-interleaved `base_vs_cublas_1980.json` exists and also FAILs all four shapes
(superseded by interleaved). fp32 accumulation throughout.
**Method:** real Mistral-7B linear shapes (k/v K4096·N1024, q/o K4096·N4096, down K14336·N4096, gate/up
K4096·N14336, M=2048 prefill); the existing CUTLASS sm90 WGMMA fused-checksum kernel (`step_b_prefill_build/
libkernel_dedup.so`, KernelTmaWarpSpecializedCooperative, dedup row-sum reference); Step-A harmful fault class
(fp16 bit-13/14 flip). Every number from THIS run unless cited to ra-e2e (the +2.96% real-trace E2E anchor) or
to the step-b-spike decode result (+1.1% / +10.1%).
**Evidence:** `cipher-fusion-evidence/ra_keystone/{trackA,trackB}/` (trackC = synthesis of A+B + the ra-e2e /
full-coverage-closure anchors, no fresh measurement) + `_STATUS_AND_PREREG.md` (pre-registered
verdicts written before measurement, advisor-reviewed).

---

## EXECUTIVE VERDICT

The mandate's hypothesis — that all open transient walls reduce to ONE keystone (a checksum-fused GEMM at cuBLAS
parity) and that building it would close per-step SDC checking <3% — is **falsified, with mechanism.** And the
Gap-6 hosting wall is **downgraded** from "structurally blocked" to "feasible-but-fiddly."

- **TRACK A (keystone) = WALL-WITH-MECHANISM, over-determined.** Fresh vs cuBLAS, the fused-checksum WGMMA GEMM is
  **+27% to +72%** on every real Mistral shape — failing the <3% per-step budget by 9–24×. The wall has TWO
  independent terms, each fatal alone: (1) the **WGMMA-vs-cuBLAS parity gap** (+10–45%), and (2) the **wave-quantized
  independent-reference epilogue** (+3–19%; <3% only beyond N≈22–28K, larger than any real GEMM). The single
  highest-value tuning lever — pingpong warp-specialization to hide the reference — is **implausible** from the
  CUTLASS source (it overlaps the wrong phase, `r` is mainloop-resident, no spare warpgroup, stream-K forbidden).
  Per-step transient checking stays walled; the substitution gap is real.
- **TRACK B (hosting) = CLOSED-WITH-BOUND.** A device check-node CAN be injected at the `cudaGraphInstantiateWithFlags`
  interposition seam — demonstrated on a **real `torch.cuda.graph` (the same capture API vLLM uses, not a real vLLM
  decode graph)**: it executes in replay and the host reads its flag without invalidating capture, **no source edit**
  (Path 1). This downgrades the prior "an injected detector cannot be hosted in cudagraphed vLLM" wall. The bound: it
  was a 7-node toy graph, the node is a constant-writer (a real checksum also reads the GEMM output), and production
  wiring (allocator-compatible insertion + `cudaGraphKernelNodeGetParams` output-param recovery) is unrun. The no-source
  fallback, `enforce_eager`, costs a **measured ~47%** decode throughput (cudagraph 1.9×) — far heavier than the prior
  10–30% estimate.
- **TRACK C (product).** Two axes the mandate's framing fused are in fact **orthogonal**: *hostable* (Track B) and
  *cheap-enough-per-step* (Track A). Path-1 hosts a per-step in-graph check — exactly what Track A cost-walled. The
  detector that actually **ships** is the **periodic recompute** — linear-GEMM-only **+2.96% @N=45** (ra-e2e); full
  linear+non-GEMM+attention unified **+2.51% @N=64 / +3.57% @N=45** (full-coverage-closure, NOT ra-e2e) — a different
  execution model (out-of-graph periodic pass, keeping cudagraph). R.A ships as a **persistent-within-N SDC detector**,
  NOT a per-step transient one.

---

## TRACK A — the keystone: checksum-fused WGMMA GEMM vs cuBLAS

### Step 0 — paper ceiling (Rule 4, derived before measurement)
For C = X·W (X=[M,K], W=[K,N]): the row-sum `s[m]=Σ_n C[m,n]` is a **free** fp32-accum epilogue byproduct
(`s/GEMM = 1/(2K)` = 0.003–0.012% FLOP). The detection cost is the **independent reference** `r[m]=Σ_k X[m,k]·wref[k]`
(`wref=Σ_n W[k,n]`, precomputed once), `r/GEMM = 1/N` FLOP (0.007–0.098%). FLOP is a red herring (<0.1%); the realized
cost is **wave-count-limited** — `r` is a CUDA-core serial-FMA reduction chain (length K) that cannot overlap the
tensor-core WGMMA mainloop, so its exposed latency is amortized only when enough waves exist (`waves = N/2112` at
M=2048). Crossing <3% needs ~10–13.6 waves (**N≈22–28K**), larger than any real Mistral GEMM (max N=14336 ≈ 6.8 waves).
Long-K worsens it sub-linearly (`r`'s chain is length K). **The parity↔epilogue requirements fight**: closing parity
shrinks the base denominator (fixed `r` becomes a larger %) and removes the mainloop slack that hides the reference.
Pre-registered analytical verdict: **no real Mistral prefill shape admits a <3%-vs-cuBLAS fused-checksum GEMM.**

### Measurement — fresh @1980, interleaved (drift-canceling), median of 200 reps
`ra_keystone/trackA/{trackA_vs_cublas.py, trackA_interleaved.py, interleaved_1980.json}`. C bit-exact vs cuBLAS
(rel-err 0) all shapes.

| shape | N | waves | **parity gap** (custom vs cuBLAS) | **checksum** (fused vs custom) | **TOTAL vs cuBLAS** | <3%? |
|---|---|---|---|---|---|---|
| k/v_proj | 1024 | 0.48 | +45.0% | +18.8% | **+72.3%** | FAIL |
| q/o_proj | 4096 | 1.94 | +20.2% | +6.9% | **+28.5%** | FAIL |
| down_proj | 4096 (K=14336) | 1.94 | +10.3% | +15.0% | **+26.9%** | FAIL |
| gate/up_proj | 14336 | 6.79 | +28.4% | +3.3% | **+32.6%** | FAIL |

The WALL is **over-determined**: (1) the parity gap alone (+10–45%) fails every shape; (2) even at *hypothetical
perfect cuBLAS parity*, the checksum epilogue alone (gate/up +3.3% best → k/v +18.8%) still fails every shape — though
the best-shape margin is thin (gate/up +3.28%, just over the 3% bar in this interleaved run; the non-interleaved base
run puts it at +10.4%, so leg (2) is more robust there). The checksum is wave-count-limited as the paper ceiling
predicts at the extremes (gate/up's 6.8 waves → +3.3% lowest; k/v's 0.48 waves → +18.8% highest); the two equal-wave
middle shapes (q/o, down both 1.94 waves) differ +6.9% vs +15.0% — the long-K (K=14336) serial-chain term, as Step 0 notes.

### The falsifier — pingpong-hides-`r`: IMPLAUSIBLE (analytical, from CUTLASS 4.1.0 headers)
The one lever that could hide the exposed reference reduction is warp-specialization. From
`sm90_gemm_tma_warpspecialized_pingpong.hpp`: (1) pingpong overlaps **MMA(WG_a) ↔ EPILOGUE(WG_b)** across symmetric
warpgroups on different tiles, but `r` is **mainloop-resident** (reads A as the K-loop streams it; full-row A ≈ 3.7 MB
≫ 228 KB smem, can't move to the epilogue) → it overlaps the wrong phase; (2) **no spare warpgroup** in tensor-core-
bound prefill (cooperative already uses both WGs for WGMMA + the `r`-reduction); (3) the cost is serial-chain(K) +
wave-count, both **schedule-independent**; (4) pingpong **`static_assert`-forbids stream-K** (line 105) — the one
low-wave lever, exactly where the worst FAILs are. A no-build, Rule-4 resolution: pingpong cannot rescue the keystone.
Reinforced by the data — since the parity gap dominates, even a perfect checksum→0 leaves the parity gap failing.
**No kernel was built to chase parity**: it is orthogonal, known-hard, and winning it *worsens* the epilogue %.

### Coverage — CONFIRMED
`ra_keystone/trackA/{coverage_inject.py, coverage_result.json}`. The ABFT identity `s = rowsum_N(C)` vs the kernel's
independent reference `r` catches Step-A bit-13/14 flips **200/200 = 100%** on all four shapes, **0 clean false
positives**. The operating threshold is **T = max(8×clean-residual, 1e-3) = 1e-3** (`coverage_inject.py:37`; the 1e-3
floor governed all four shapes since 8×clean-residual ≤ 2.7e-4) — ~30× above the worst fp32 clean residual (3.3e-5) and
well below the smallest injected fault, so both 200/200 and 0 FP hold with margin. (This validates the detection
*identity* using the kernel's emitted `r`; the kernel exposes no internal compare-flag — the compare is `s` vs `r`.)

### Decode (M=1) nuance — preserved, NOT collapsed into the prefill WALL
Decode is memory-bound, different physics: at M=1 the GEMV reads all of W and `r` piggybacks free under the weight
read → fused checksum **+1.1% (<3% PASS)** (cited from `step-b-spike`, a memory number, not this prefill run). It
carries a **separate +10.1% deployable substitution gap** vs cuBLAS (cuBLASLt cannot host the row-sum-over-N epilogue
[Finding 1] → a custom GEMV is mandatory, and it runs at ~92% of cuBLAS bandwidth). Decode remains a sub-3%
*checksum* candidate gated on closing that GEMV bandwidth gap; prefill fails on every real shape.

**Track A verdict: WALL-WITH-MECHANISM.** Per-step transient checking is not <3%-deployable on any real Mistral
prefill GEMM. Dominant mechanism: the exposed, non-overlappable CUDA-core reference matvec (wave-count-limited),
compounded by the WGMMA-vs-cuBLAS parity gap; the two fight. The substitution gap stands.

---

## TRACK B — Gap-6 hosting: can the detector co-run inside production vLLM?

vLLM 0.20.2 / torch 2.11 builds its decode graph by **stream capture** through `torch.cuda.graph`
(`cuda_graph.py:308-312`), `keep_graph=False` → `cudaStreamEndCapture` + `cudaGraphInstantiateWithFlags` +
`cudaGraphDestroy` inside one `capture_end()` C++ call (no Python seam). But `libtorch_cuda.so` imports
`cudaGraphInstantiateWithFlags` as an **UNDEF** symbol from the separate `libcudart.so.13` — so it is interposable
at the runtime-symbol boundary (`CUDA_INJECTION64_PATH` / `LD_PRELOAD`), no vLLM source edit.

### Path 1 — capture-time device-node injection: DEMONSTRATED on a real torch.cuda.graph (downgrades the capture-invalidation wall)
`ra_keystone/trackB/{gap6_inject_shim.c, gap6_path1_demo.py, gap6_path1_result.json}`. A pure-C shim (dlsym
RTLD_NEXT; no cudart link, no headers, no nvcc kernel — version-safe across nvcc-12.8/torch-cu13) interposes
`cudaGraphInstantiateWithFlags`, receives the populated `cudaGraph_t` **before** instantiation, and adds a device node
(a memset writing a **constant** sentinel to a shim-owned buffer — it reads no graph tensor). On a **synthetic 7-node
graph built with the same `torch.cuda.graph` capture API vLLM uses** (NOT a real vLLM decode graph; the captured graph
reported 0 edges so all nodes are leaves — both a root-node and a leaf-dependency injection were run, identical result):
- **hook fired** (intercepted the exact runtime symbol torch calls);
- the **injected device node EXECUTES IN REPLAY** (flag 0 → 0xC1);
- the **host reads the flag post-replay** via the shim's own `cudaMalloc`'d buffer — **no host sync inside capture**;
- **capture is NOT invalidated** — re-replay works and the graph produces correct output with new input.

This downgrades the prior Gap-6 verdict ("an injected in-context detector cannot be hosted in cudagraphed vLLM")
to **feasible-but-fiddly**: a device check-node can be hosted at the EndCapture→Instantiate seam without touching
vLLM. **Bound:** (i) this was a toy graph, not a real vLLM decode graph; (ii) the injected node writes a constant — a
real checksum kernel additionally **reads** the last GEMM's output, which needs the unrun `cudaGraphKernelNodeGetParams`
output-param recovery (kernel-param-layout reverse-engineering), and the leaf-ordering to run such a check *after* the
GEMMs exists in the shim but its value depends on that recovery; (iii) adding a raw out-of-pool node perturbs torch's
*next fresh allocation* once (`cudaErrorInvalidValue`, self-recovering; a control run with injection disabled is clean),
so production needs allocator-compatible insertion. The demonstrated claim is the injection PRIMITIVE (node executes in
replay, capture stays valid), not an end-to-end in-vLLM checksum.

### Path 2 — `enforce_eager` co-run: tax MEASURED ~47% (prior 10–30% estimate REFUTED)
`ra_keystone/trackB/{gap6_path2_eager_tax.py, gap6_path2_tax.json}`. Qwen2-7B fp16, vLLM 0.20.2, cipher plugins
**excluded** (`VLLM_PLUGINS`=lora-only) + `VLLM_USE_DEEP_GEMM=0` (clean/uninjected decode tok/s):

| batch | cudagraph (graph) | enforce_eager | cudagraph speedup | **enforce_eager tax** |
|---|---|---|---|---|
| 1 | 170.4 tok/s | 90.1 tok/s | 1.89× | **47.1%** |
| 32 | 5209 tok/s | 2734 tok/s | 1.91× | **47.5%** |

Disabling cudagraph nearly **halves** decode throughput. So `enforce_eager` is an *expensive* hosting path, not a
cheap fallback — which is decision-relevant for Track C.

**Track B verdict: CLOSED-WITH-BOUND.** Capture-time node injection hosts a detector node without a source edit
(the per-step path); `enforce_eager` hosts any eager detector at a measured ~47% tax. (Found incidentally: vLLM
general-plugins `cipher_vllm_kv`/`cipher_vllm_kvdedup` auto-load the cipher runtime in vLLM — not in plain torch.)

---

## TRACK C — R.A as a shipping feature

**The orthogonal-axes correction (load-bearing):** Track B's node-injection most directly hosts a *per-step in-graph*
ABFT check — which Track A just cost-walled (+27–72%). The detector that ships is the **periodic recompute** (every N
steps re-run the step and compare), a *different* execution model: a separate out-of-graph pass, NOT a single node in
the main decode graph. "Hostable" (Track B) and "cheap-enough-per-step" (Track A) are **orthogonal**. So the product is
the **persistent-within-N** detector, hosted by periodic out-of-graph recompute that keeps cudagraph: **linear-GEMM-only
+2.96% E2E @N=45** (ra-e2e), or the **full linear+non-GEMM+attention unified +2.51% @N=64 / +3.57% @N=45**
(full-coverage-closure — the unified detector fits <3% only at N≈64), with `enforce_eager` (~47%, measured here) as the
expensive no-source fallback.

### Coverage matrix — the product datasheet (fault class × shippable claim)

| Fault class | Caught? | Latency | Cost | Per-step or periodic | Hosting path |
|---|---|---|---|---|---|
| Linear-GEMM output, **persistent** | ✅ | ≤ N−1 steps | in ~3% (N≈45–64) | periodic | out-of-graph recompute (keeps cudagraph) |
| Non-GEMM output, **persistent** | ✅ | ≤ N−1 | in ~3% (regen-from-input) | periodic | out-of-graph recompute |
| Attention QKᵀ / PV, **persistent** | ✅ 100% | ≤ N−1 | +0.01% | periodic | out-of-graph recompute |
| Linear-GEMM output, **transient single-step** | ⚠️ ~1/N | — | **WALLS** (+27–72% fused GEMM) | per-step (needed) | Path-1 node-injection *hostable* but cost-walled |
| Non-GEMM output, **transient single-step** | ⚠️ decode only | — | prefill WALLS | per-step | hostable, cost-walled |
| Attention **softmax-internal finite** | ❌ ~12–19% blind | — | — | — | algebraic blind spot |
| Deterministic **SM-localized** | ✅ (simulated) | 1 step | ~2× SMs | redundancy mode | green-ctx (opt-in) |
| Deterministic **non-SM-localized** | ❌ | — | — | — | recompute reproduces it |
| Detector hosting in **cudagraphed vLLM** | ✅ node-injection (toy-graph demo) / ⚠️ eager | — | Path-1 fiddly / eager ~47% | per-step host (still cost-walled) vs periodic out-of-graph | seam-interposed (Path-1) or enforce_eager |

The per-class "in ~3%" costs above are individually honest, but the **combined unified detector (all persistent classes
together) is +3.57% @N=45 / +2.51% @N=64** (full-coverage-closure) — it fits <3% only at **N≈64**; only the linear-only
configuration is <3% across the full N=45–64 band (ra-e2e +2.96%@45 / +2.15%@64). Per-step transient (the row that
would need the keystone) is **hostable but cost-walled** — the precise, honest intersection of Tracks A and B. Decode
transient checksum is the lone sub-3% per-step candidate (+1.1%, gated on the GEMV bandwidth gap).

### The feature, in one paragraph (for a neocloud customer)
**R.A is a silent-data-corruption detector for LLM inference that rides inside vLLM at <3% throughput cost.** It
catches *persistent* SDC — stuck/repeating bit-level corruption in the linear-GEMM outputs, the non-GEMM outputs
(RMSNorm/RoPE/SiLU/residual), and the attention QKᵀ/PV products — by periodically recomputing a step and comparing,
out-of-graph so vLLM's CUDA-graph decode speed is preserved. **Linear-GEMM-only persistent coverage is +2.96% E2E at
N=45; the full linear+non-GEMM+attention unified coverage fits <3% at N≈64 (+2.51%; it is +3.57% at N=45, over budget).**
It detects within N steps (sub-second), with **zero false positives** — T=0 by construction for the GEMM/non-GEMM
recompute paths; the attention QKᵀ/PV check uses T=8×roundoff with 0 FP measured over 100 runs — and a strict superset
of the prior linear-only coverage. **Residuals, named honestly:** *transient single-step* faults are caught only ~1/N (a per-step in-graph
check is hostable via CUDA-graph node injection but is cost-walled by the cuBLAS parity + reference-epilogue
substitution gap — best real shape +27% vs cuBLAS); *softmax-internal finite* corruption (~12–19%) is algebraically
blind; *deterministic non-SM-localized compute* faults are reproduced by recompute (the SM-localized subset is caught
by an opt-in 2× green-context redundancy mode). The no-source-edit eager hosting path costs ~47% decode throughput, so
the shipping config uses periodic out-of-graph recompute to keep cudagraph.

---

## GUARDRAILS (exit state)
- Production `.so` md5 `2edba0d2136f8ede4713d90a8f7cd55f` — **UNCHANGED** (verified at entry and after the vLLM runs).
- Track-A/coverage path uninjected (plain torch matmul shows no cipher banner); Track-B used an LD_PRELOAD shim
  (not the production `.so`) with the cipher vLLM plugins explicitly excluded for a clean tax.
- Clock reset `nvidia-smi -rgc`; vLLM EngineCore children reaped; no compute procs left; no `.so` touched; no tags moved.
