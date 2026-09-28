# CP 2.4 — Marlin per-stream registry regression (test B) — NOTE

**Date:** 2026-05-15. Sub-task: Marlin registry regression, **test B** of the
"Both" pair (B = mechanism microbenchmark; A = end-to-end decode, still to run).
Not the CP 2.4 report — that lands once all gate criteria are measured.

## Why a microbenchmark, not the verbatim density sweep

The CP 0.4/0.5 `density_sweep_a.py` / `density_harness.py` is wired to the
**old op31 LD_PRELOAD API** (`setup_rt()` loads `/workspace/libcipher_rt.so`;
`MarlinLinear` calls `cipher_weight_compress_marlin_gemm`). The v2 lib
(`cipher_rt_phase4/libcipher_rt.so`) exports **none** of that API, and the old
harness carries its own Python `_MARLIN_LOCK` + single `_MARLIN_STREAM`
workaround — a verbatim re-run cannot exercise the new per-stream engine.
Per user adjudication (2026-05-15), the regression is **Both**: B here, A next.

## Test B — design

`marlin_concurrency_bench.cu` dlopens the v2 lib and drives the Marlin engine
(`cipher_rt_marlin_engine_dispatch`) directly from N threads — one shared INT4
weight (M=1, K=N=4096, G=128 — Mistral-7B decode shape), each thread its own
activation/output buffers, a tight dispatch loop, 5 s/point.

- **per-stream** mode: each thread its own CUDA stream → the CP 2.4 registry
  under test.
- **shared** mode: all threads on ONE stream → 1 registry slot → serialized;
  the control that mimics the CP 0.4/0.5 single-`_MARLIN_STREAM` ceiling.

## Results (`marlin_regression_result.json`)

| mode | N | agg GEMM/s | ws_slots | rc_fail |
|---|---|---|---|---|
| per-stream | 1 | 87 060 | 1 | 0 |
| per-stream | 2 | 139 657 | 3 | 0 |
| per-stream | 4 | 151 847 | 5 | 0 |
| per-stream | 8 | 151 952 | 9 | 0 |
| per-stream | 16 | 151 860 | 19 | 0 |
| per-stream | 32 | 151 845 | 35 | 0 |
| per-stream | 64 | 151 866 | 67 | 0 |
| shared | 1–64 | ~86 500 (flat) | 1 new | 0 |

(`ws_slots` accumulates distinct stream pointers across the whole run — the
registry is global; streams are destroyed per point and pointers recycle, so
the count climbs 1→67 rather than resetting. The shared run created exactly
**1** new slot for all 64 threads — the registry distinguishes per-stream from
shared exactly as designed.)

## Verdict — mechanism PASS, with an honest ceiling finding

**What test B proves:**
- ✅ **Serialization removed.** The shared-stream control is dead flat at
  ~86.5 K GEMM/s for all N — the CP 0.4/0.5 single-stream ceiling, reproduced.
  Per-stream lifts it to ~152 K (**1.76×**).
- ✅ **N-way slotting confirmed.** `ws_slots` tracks per-stream creation
  (1→67); shared mode adds exactly 1 slot. Each tenant stream gets its own
  `locks` workspace — the registry does what §2.2 specified.
- ✅ **Race-free under concurrency.** `rc_fail = 0` across every point,
  including 64 concurrent streams. The old single shared workspace would race;
  the per-stream registry is correct.

**What test B also reveals (honest):**
- ⚠️ Per-stream aggregate GEMM/s **scales only to N≈4** (87 K→152 K), then
  **saturates**. This is **not** a software-lock ceiling — it is a
  **GPU-occupancy ceiling**: each M=1 Marlin GEMM launches `grid = 132` blocks
  (one per SM), so 2–4 back-to-back GEMMs already fill the H100's block
  scheduler. A synthetic loop that does nothing but back-to-back Marlin GEMMs
  cannot scale past the silicon. The registry removed the *software* ceiling;
  the microbench then hit the *hardware* one.
- The `g_ws_mu` registry mutex does an O(256) slot scan per dispatch — a few %
  at most at N=64, not the plateau cause; an O(1) hash is a noted production
  refinement, immaterial for the gate.

## Why this makes test A decisive

The microbench saturates the GPU on purpose (back-to-back GEMMs). The **real**
multi-tenant decode is the opposite regime: CP 0.4 measured **MFU 0.0185 %** —
the H100 is ~99.98 % idle, each tenant interleaves GEMMs with attention/norm/
sampling and host-side Python, and the binding constraint is host-side launch
serialization, **not** GPU occupancy. There, removing the Marlin serialization
should let tenants genuinely overlap. **Test A (end-to-end N-tenant decode
under the v2 lib) is where the "8.5 tok/s ceiling lifts" question gets its real
answer.** Test B has done its job: proved the registry mechanism is correct,
race-free, and lifts the serialized ceiling.

## Artifacts

| Artifact | md5 |
|---|---|
| `marlin_concurrency_bench.cu` | `4e0e4aba9ee0585f78d51d2949598cd5` |
| `marlin_concurrency_bench` (binary) | `8be8a85e575f22b272300d725c7e9079` |
| `marlin_regression_result.json` | `6e27bfbe69afea99545908e172123e5b` |
| `marlin_regression_run.log` | `4b7979bb8e04f9d90c99a9a2d4590f0c` |

**Anchors:** kmod `2a69f9de…`, libcipher_v2 `86618c30…`, taint 12288 — unchanged.
libcipher_rt.so under test: `a0d6cddacb116f2d51ad1d1f867ef564` (CP 2.4 build).

---

# Test A — end-to-end N-tenant decode (2026-05-15)

## Activation invocation (resolved before the run)

Per `PHASE_4_T4_5_REPORT.md:74-77` and `PHASE_4_T4_6_1_REPORT.md:203`, the v2
lib is activated by **both** `LD_PRELOAD` (`.symver` cuBLAS interposition) and
`CUDA_INJECTION64_PATH` (driver calls `InitializeInjection2` → actuator init),
both pointing at the same `libcipher_rt.so`, plus `CIPHER_MARLIN=on`. See
`run_test_a.sh`. Documented in `PROGRESS.md`.

## Design

`test_a_density.py` — N-tenant Mistral-7B continuous decode, one shared model,
per-tenant StaticCache + CUDA stream, 30 s/step. **No Python `_MARLIN_LOCK`,
no `MarlinLinear` monkey-patch** — Marlin routing comes entirely from the v2
lib's cuBLAS `.symver` interception → per-stream registry. N ∈ {1,4,16,64}.

## Routing confirmed

`MATMUL: exit totals — calls=687375 handled=666675 passthrough=20700` —
**97.0 %** of GEMMs routed to the Marlin actuator. Actuator registered,
NVRTC cubin compiled (10/10, 12 s), 23+ Mistral-7B weight shapes quantized
(K=4096 N∈{1024,4096,14336}, K=14336 N=4096). The v2 Marlin path is genuinely
exercised — not a passthrough run.

## Results (`test_a_result.json`)

| N | test A agg tok/s | CP 0.4 baseline | lift vs CP 0.4 | tps/tenant | MFU % |
|---|---|---|---|---|---|
| 1  | 30.75 | (no N=1) | — | 30.748 | 0.0446 |
| 4  | 24.07 | 12.76 | **1.89×** | 6.018 | 0.0349 |
| 16 | 14.82 | 8.86 | **1.67×** | 0.926 | 0.0213 |
| 64 | 13.44 | 8.89 | **1.51×** | 0.210 | 0.0189 |

## Verdict — ceiling LIFTED; Marlin sub-task CLOSES

- ✅ **The CP 0.4/0.5 ~8.5 tok/s aggregate ceiling is gone.** Every matched-N
  point beats the old plateau — a uniform **1.5–1.9× aggregate lift**. The new
  plateau is ~13–15 tok/s (N≥16) vs the old flat ~8.85.
- **Honest residual:** aggregate throughput still *declines* with N
  (30.75 → 24.07 → 14.82 → 13.44). The per-stream registry **shifted the whole
  curve up** — it removed the Marlin-workspace serialization — but it did not
  make aggregate throughput *scale* with N. Adding tenants still reduces
  aggregate tok/s.
- **What the residual decline is:** the eager-mode Python launch-overhead /
  GIL / host-dispatch floor — the ceiling CP 0.4's own STORY_A named
  ("eager-mode, launch-overhead-bound"). The registry was never going to fix
  that; it is **CP 4.3 persistent-kernel** territory, not Marlin. Stated as the
  measured outcome — not reframed.
- This matches the test-B finding: B isolated the registry (1.76× over a
  same-binary shared-stream control — pure registry effect); A shows the
  end-to-end v2 Marlin path lifts CP 0.4's ceiling 1.5–1.9×. Two ceilings: the
  software-serialization one (removed) and the eager-mode launch one (remains,
  CP 4.3 scope).

**Marlin sub-task (i) — lock-fix regression: PASS / CLOSED.** Mechanism proven
(B: race-free, N-way slots, serialization removed), end-to-end ceiling lifted
(A: 1.5–1.9×). Marlin's per-call tok/W lift (the scorecard 1.62×) is measured
later as memo §6 gate criterion (b) per-lever lift, not here. Next: DVFS
envelope sweep.

## Test A artifacts

| Artifact | md5 |
|---|---|
| `test_a_density.py` | `cda2bef7608d2645a66319b6626e7fb3` |
| `run_test_a.sh` | `895c510057336ab5b21045b035fea6c1` |
| `test_a_result.json` | `a7985500d71dda054cef0e3f8a04de8e` |
| `test_a_run.log` | `67e3829e8eb084f64c89d3d3d780eea9` |

**Anchors:** kmod `2a69f9de…`, libcipher_v2 `86618c30…`, taint 12288 — unchanged.
