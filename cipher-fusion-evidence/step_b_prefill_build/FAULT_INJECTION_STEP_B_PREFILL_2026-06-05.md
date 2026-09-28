# Fault-Injection STEP B — PREFILL fused-epilogue ABFT checksum: the gated CUTLASS make-or-break

Date: 2026-06-05. READ-ONLY spike. NO .so change (production `libcipher_rt.so` md5
`2edba0d2136f8ede4713d90a8f7cd55f` UNCHANGED). H100 80GB, SM **locked 1200 MHz**
(verified held throughout under load: power 81W→463W, temp 37→50°C, clock flat 1200),
CUDA 12.8 / nvcc 12.8, CUTLASS 4.1.0 (tilelang vendored), arch sm_90a.

Files (this dir): `kernel_dedup.cu` (+`libkernel_dedup.so`), `harness_dedup.py`,
`sweep_n.py`, `stability.py`, `machinery.py`, results `*_result.json`.
Predecessor (redundant SS build): `kernel_r.cu`/`gemm_base.cu`. Carries
[[cipher-fault-injection-step-b-spike]] (decode +1.1% PASS; prefill = this build).

## The question
Decode (M=1) fused checksum measured +1.1% (<3% ✅) because M=1 is memory-bound and the
tiny `A·wref` reference piggybacks free under the 117MB weight read. Prefill (M=2048) is
tensor-core-bound; the reference matvec `r[m]=Σ_k A[m,k]·wref[k]` (independent of the
GEMM's C-accumulation, required for SDC detection) is CUDA-core work. Does it stay <3%
fused into a real CUTLASS WGMMA mainloop?

Mechanism: `s[m]=Σ_n C[m,n]` (output row-sum, free fp32-accum epilogue byproduct) is
checked against `r[m]=Σ_k A[m,k]·wref[k]`, `wref[k]=Σ_n W[n,k]` (precomputed once on
static W). In exact arithmetic `r≡s`; a fault in the product compute makes `s≠r`. `r`
must touch A directly (independence) → irreducible M·K MAC reference matvec.

## Build
Real CUTLASS sm90 `KernelTmaWarpSpecializedCooperative` GEMM (TileShape 128×256×64,
4 stages, MMA_64x256x16_F32F16F16_SS), builder-generated collective **subclassed**:
only `mma()` overridden to inject `r` from smem-resident A-tiles; load/epilogue/scheduler
inherited unchanged. Modes on one binary: 0=GEMM, 1=correct scalar r (validation,
C rel-err 0, r rel-err 6e-7), 2=efficient-vec r in EVERY CTA, 3=efficient-vec r gated to
ONE CTA per row-tile.

## The decisive correction (advisor catch): r was computed REDUNDANTLY
`r[m]` depends only on row m of A — **independent of N** — yet the first build computed
it in every N-tile CTA, recomputing it `N/BLK_N` times (16× down, 56× gate). A correct
impl computes `r` **once per row-tile** (the col-tile-0 CTA), exactly the "compute once"
logic already correct for the output row-sum `s`. mode2 (redundant, +14–17%) was an
UPPER bound, not a floor. mode3 deduplicates via a global per-work-tile atomic
(`seq % R == 0`, R=N/BLK_N), broadcast to the 256 consumer threads by NamedBarrier.
**Gate verified: fires exactly 16× = M/BLK_M = row_tiles, every shape, every launch** —
total r-traffic = M·K (the irreducible amount), spread evenly across tiles.

## Result — dedup is real and large, but wave-count-limited; real shapes still FAIL 3%
Deduplication cuts overhead by roughly the wave count, and the residual amortizes as more
waves give the persistent scheduler slack to hide the 16 r-laden tiles. Stable (6-rep,
interleaved, ±0.1–0.6pp) and **machinery-subtracted** (the atomic+threadfence+NamedBarrier
runs on every tile in mode3 but costs only +0.5–1.3%; a real `n_coord==0` gate pays 0):

| shape (Mistral-7B prefill, M=2048) | K | N | ~waves | redundant (mode2) | **TRUE dedup r-floor** | <3%? |
|---|---|---|---|---|---|---|
| k/v_proj | 4096 | 1024 | 0.48 | +13–14% | **+16.8%** | FAIL |
| q/o_proj | 4096 | 4096 | 1.94 | +14% | **+7.4%** | FAIL |
| down_proj | 14336 | 4096 | 1.94 | +16% | **+14.4%** | FAIL |
| **gate/up_proj** | 4096 | 14336 | 6.79 | +13% | **+4.1%** | FAIL (closest) |
| synth N=20480 | 4096 | 20480 | 9.70 | +16% | +2.2% | pass |
| synth N=28672 | 4096 | 28672 | 13.58 | +12% | +1.1% | PASS |

N-sweep (`sweep_n.py`) confirms a monotonic amortization curve: overhead falls from ~13%
(<1 wave) toward ~1% (13.6 waves), **crossing 3% only around N≈18–20K (~9–10 waves)** —
larger than any GEMM in the model. Two dependences: (1) primary = wave count ∝ N;
(2) secondary = K — at equal N=4096/1.94-waves, long-K down_proj (+14.4%) is ~2× worse
than short-K q/o_proj (+7.4%), because r's per-thread serial FMA-reduction chain
(length K) cannot hide in the WGMMA-pipelined mainloop when K is long.

These floors are OPTIMISTIC lower bounds: (a) baseline CUTLASS is 54–84% of cuBLAS
(`base_result.json`); a fixed r-cost is a smaller % of a slower base, so a tuned/peak
baseline makes the overhead LARGER; (b) mode3's read uses scrambled k-pairing (faithful
traffic, wrong value) — a correct impl adds swizzle-correct addressing ABOVE this floor.
Both push the real number up. The FAIL verdict therefore holds firmly.

## Verdict
PREFILL fused ABFT checksum **does NOT reach <3% on any real Mistral-7B prefill GEMM.**
The deduplication catch was correct and material (gate/up_proj 13%→4.1%), and it reframes
the cost as wave-limited amortization rather than a flat tax — but the crossover sits at
synthetic N≥~20K. The best real shape (gate/up_proj, N=14336) lands at **+4.1%**, within
striking distance but over the bar; small-N and long-K shapes are far worse (+7–17%).

Reconciles the arc: decode (M=1, memory-bound) +1.1% PASS stands; prefill
(tensor-core-bound) fails because the independent A-reference is exposed CUDA-core work
whose only escape — amortizing 16 tiles across many waves — needs more waves than real
GEMMs provide. Earlier "flat +15%" was the redundancy artifact (wrong mechanism);
a one-off "+2.24% gate PASS" was a low-side measurement outlier — the stable value is +4.1%.

Not deployable as a universal prefill detector under 3%. Decode detector remains the
shippable path (gated on closing the custom-GEMV substitution gap, Finding 2 of the spike).
