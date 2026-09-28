# R.A FORK-2 — TRANSIENT-COVERAGE ATTACK (three mechanisms, end-to-end on the ra-e2e saturated trace)

**Date:** 2026-06-08 (report name per spec; run executed 2026-06-08→09).
**Mandate:** beat periodic-within-N transient coverage at <3% by a mechanism that does NOT use a per-step in-GEMM
checksum (that path is exhausted — Track A WALL, +27–72% vs cuBLAS). Additive, fork-1-preserving, read-only.
**Discipline:** no production `.so` change (anchor `2edba0d2136f8ede4713d90a8f7cd55f`); no edit to `ra_e2e/`,
`step_b_e2e/`, `ra_keystone/`, or any Track-B/hosting file — all new work in `ra_fork2/`; each mechanism is a
MEASUREMENT. fp32/fp16 as ra-e2e. Clock locked 1980 (ran ~1830 under load), `-rgc` at exit. Single-GPU serial,
exclusivity-guarded; vLLM children N/A (no vLLM run this fork). Measurement path uninjected (no `.so`).
**End-to-end scope:** the ra-e2e saturated CUDA-graph serving model — real Mistral-7B linear shapes, the real
6301-step interleaved trace (3370 decode / 2931 prefill, phase wall 39.0% / 61.0%), per-phase recompute cost model
(decode r=1.488, prefill r=1.326), saturation SM=100% both / MEM 71% decode / 52% prefill. Per-step GPU deltas
measured on the real decode (B=128) / prefill (M=512) GEMM shapes; Mech-1 catch simulated over the actual trace.
**This run does NOT host inside live vLLM.** Track B established the only no-source in-vLLM host is `enforce_eager`
(~47% decode tax) and the in-graph node path is a toy-graph primitive (not closed). No transient coverage is claimed
*inside vLLM* here — these are end-to-end measurements in the ra-e2e harness model. See the in-vLLM gap statement.
**Evidence:** `ra_fork2/` — `granularity_probe.py`+`granularity_result.json`, `dmr_sweep.py`+`dmr_sweep_result.json`,
`mech1_sim.py`+`mech1_result.json`, and `_plumbing_test.py` (green-context+cuBLAS API validation, no result file).

---

## EXECUTIVE VERDICT — all three WALL: Mech-1 by a conservation theorem, Mech-2/3 by hardware floors

**The serial-recompute conservation law (the spine for Mech-1 vs periodic).** Let R = Σt_recompute / Σt_base over the
trace = **1.389** (measured; `mech1_result.json`). For any **serial** recompute scheme (run on the main resource),
periodic-N overhead = R/N and periodic single-step transient catch = 1/N ⇒ **catch/cost = 1/R ≈ 0.72** (0.72 pp of
transient catch per 1% spent). Mechanism 1 (sampled per-step) is the same serial budget redistributed across steps, so
it obeys the same law — it **cannot beat periodic**. This conservation is exact only for *serial* mechanisms; the two
*spatial* mechanisms (Mech-2/3) are NOT bound by 1/R, and instead wall on **hardware floors**: green-context partition
granularity and global HBM bandwidth. So "all three WALL" is one conservation theorem (Mech-1) **plus** two hardware
floors (Mech-2/3) — not a single law, and the report does not pretend otherwise.

| Mechanism | Verdict | Driving mechanism |
|---|---|---|
| **1 — sampled rotating per-step subset** | **WALL** | catch = cost/R, *identical* to periodic-N at equal cost (measured ±0.21pp). Redistributing the same serial check-budget across steps conserves single-step catch. |
| **2 — redundant-SM DMR shadow** | **WALL** for <3% transient | green-context min partition = **8 SMs = 6.06%**; smallest shadow costs **14.2% decode / 38.0% prefill** (granularity + 12 stranded SMs + HBM contention) — can't enter <3%. NB: DMR is *spatially* concurrent, so its transient coverage/cost is **~1.0** (measured, S=8 decode 0.14/14.2%) — genuinely *better* per-cost than periodic's 0.72 — but the **≥14% absolute floor** (granularity/stranding/HBM) blocks <3% regardless. Its other upgrade — **deterministic SM-localized** coverage — is from Gap-5 (prior), not measured this fork. |
| **3 — async lagged full-recompute on a disjoint partition** | **WALL** | full transient coverage (latency ~1–2 steps) needs the shadow to match main's step rate ⇒ ~50/50 split (S=64), costing **48% decode / 61% prefill**. Disjoint-SM placement does NOT relieve it: at SM=100% there is no spare *compute*, and the recompute still reads weights from HBM ⇒ decode HBM contention measured up to **29.4%**. |

**Bottom line:** per-step transient detection at <3% materially better than periodic-N is **not achievable on this
hardware by these mechanisms**. Mech-1 *ties* periodic (serial conservation, can't beat it); Mech-2 and Mech-3 are
**floored above the <3% region** by green-context granularity (8-SM minimum + stranding) and HBM contention — Mech-2 is
actually more cost-efficient than periodic per unit transient coverage (~1.0 vs 0.72), so on hardware with finer SM
partitioning it could modestly beat periodic, but on *this* H100 the 8-SM floor puts its cheapest config at ≥14%.
**Fork-1 (periodic-within-N) ships unchanged.** The single fold-in candidate is **Mech-2 DMR as an opt-in redundancy
mode for DETERMINISTIC faults** (a different fault class, ≥14% cost) — NOT a transient upgrade and NOT <3%.

---

## PER-MECHANISM (Step-0 paper floor → measured → pre-registered verdict)

### Mechanism 1 — sampled rotating per-step subset. Verdict: **WALL**
- **Step-0 (paper):** check fraction f of GEMMs every step ⇒ cost = f·R, single-step transient catch = f (the fault's
  GEMM is in the subset w.p. f). Periodic-N: cost = R/N, catch = 1/N. Eliminating: **both give catch = cost/R.**
  Identical at equal cost. (Note: real per-step fixed compare/launch overhead makes sampled *f* slightly *worse* than
  f·R, so the tie is a ceiling for sampling, not a floor.)
- **Measured** (`mech1_result.json`, 200k single-step transients over the real 6301-step trace, G=225 GEMMs/step):

  | cost budget | periodic-N catch | sampled-f catch | Δ (sampled−periodic) |
  |---|---|---|---|
  | 13.95% (N=10) | 10.13% | 10.19% (f=0.100) | +0.06pp |
  | 7.03% (N=20) | 5.05% | 4.87% (f=0.051) | −0.17pp |
  | 4.70% (N=30) | 3.41% | 3.57% (f=0.034) | +0.16pp |
  | **3.09% (N=45)** | **2.23%** | **2.21% (f=0.022)** | **−0.02pp** |
  | 2.17% (N=64) | 1.56% | 1.77% (f=0.016) | +0.21pp |

  Sampled tracks periodic within ±0.21pp (rounding from integer subset size k/225). **No improvement.** WALL.

### Mechanism 2 — redundant-SM DMR shadow on a disjoint green-context partition. Verdict: **WALL (<3% transient)**
- **Step-0 (paper):** at SM=100%, carving S shadow-SMs costs ≈S/132 main throughput; the shadow covers ≈S/132 of
  GEMMs/step ⇒ transient coverage ≈ cost (≈ periodic). The decisive limits: green-context partition **granularity**
  and **global HBM** (not isolated by SM partitioning).
- **Measured granularity** (`granularity_result.json`): on this H100 the green-context SM split is **multiples of 8,
  minimum 8 SMs** (even minCount=1 → 8-SM groups; 15 groups of 8 = 120, **12 stranded**). Smallest shadow = **8 SMs =
  6.06%** — already >3% before any contention.
- **Measured cost/coverage** (`dmr_sweep.py`, real weight-streamed Mistral forward, shadow doing real GEMMs
  concurrently on a disjoint partition):

  | shadow SMs | main SMs | phase | DMR cost vs full-132 | HBM-contention component | coverage (GEMMs/step) |
  |---|---|---|---|---|---|
  | 8 | 112 | decode | **14.2%** | +4.4% | 0.14 |
  | 8 | 112 | prefill | **38.0%** | +1.2% | 0.12 |
  | 16 | 104 | decode | 28.4% | +10.9% | 0.33 |
  | 32 | 88 | decode | 29.7% | +15.5% | 0.61 |
  | 64 | 56 | decode | 48.0% | +29.4% | 1.37 |
  | 64 | 56 | prefill | 61.2% | +5.9% | 1.51 |

  The **smallest possible shadow costs ≥14% (decode) / ≥38% (prefill)** — an order of magnitude over 3%. Decode HBM
  contention rises with shadow size (4.4%→29.4%), **confirming** that green-context SM partitioning does not isolate
  the global HBM the decode shadow streams weights from. DMR's transient **coverage/cost is ~1.0** (decode S=8:
  0.14/14.2% ≈ 0.98) — genuinely *better* per-cost than periodic's 1/R = 0.72, because spatial duplication runs
  concurrently and is **not** bound by the serial conservation law. (A real DMR adds only an output **compare**, which
  is an O(M·N) elementwise/bandwidth op — a small *additive* tax, NOT a full O(M·N·K) recompute, so it does not pull
  the ratio down to 0.72; the GEMM recompute is already inside the measured 14.2%. The honest ratio is ~1.0 minus a
  small compare tax, still clearly above periodic's 0.72.) **What walls Mech-2 is the absolute cost floor, not its
  efficiency:** the 8-SM granularity + 12 stranded SMs + decode HBM contention floor the cheapest shadow at ≥14%, so
  it cannot enter the <3% region on this hardware whatever its per-cost efficiency.
- **Verdict:** WALL for <3% transient (the ≥14% granularity/stranding/HBM floor). A separate, *different-fault-class*
  upgrade — duplication on disjoint SMs catching **deterministic SM-localized** faults that recompute-on-same-HW cannot
  — is established by **Gap-5 (prior run), NOT measured in this fork** (this fork's `dmr_sweep` measures throughput
  cost + coverage-ratio only, no fault injection/compare/detection); it would ship as an opt-in redundancy mode at
  ≥14% cost, not a transient improvement.

### Mechanism 3 — async lagged full-recompute on a disjoint partition. Verdict: **WALL**
- **Step-0 (paper):** to catch *every* transient at ~1–2 step latency the shadow must fully recompute every step ⇒ it
  must match main's step rate ⇒ ~as many SMs as main ⇒ ~50% main loss. The premise "spare COMPUTE" is falsified by
  SM=100%; and a disjoint partition still reads weights from HBM, so decode (the HBM-bound phase) still collides.
- **Measured** (same sweep, S=64 = the smallest split that lets the shadow keep main's rate): coverage **1.37 decode /
  1.51 prefill** (shadow ≥ main rate ⇒ full per-step recompute achievable; this is the same GEMM-only no-compare
  coverage as Mech-2 — a real DMR adds the small O(M·N) compare tax to both equally) at cost **48.0% decode / 61.2%
  prefill**; decode HBM contention **29.4%** (vs prefill 5.9% — decode is HBM-bound, prefill compute-bound, exactly as
  the step_b_e2e §5 same-stream-overlap finding predicted, and SM disjointness does not fix it). Far over 3%.
- **Verdict:** WALL. Lagged full-recompute = doubling GEMM work on a saturated single GPU; SM placement cannot create
  spare compute, and decode HBM contention persists.

---

## TRANSIENT COVERAGE / COST FRONTIER (all three vs the periodic-N baseline)

Single-step transient catch (%) achievable at a given throughput cost (%), saturated trace:

| Cost budget | Periodic-N (fork-1) | Mech-1 sampled | Mech-2 DMR | Mech-3 lagged-full |
|---|---|---|---|---|
| **3%** | **2.2%** | 2.2% (≡ periodic) | **unreachable** (floor ≥14%) | unreachable (floor ≥48%) |
| 14% (decode floor of M2) | 10.1% | 10.1% | ~14% (coverage/cost ~1.0 > periodic) − small compare tax | partial (= M2) |
| 48% (decode floor of M3 full) | 34.6%* | 34.6%* | — | **100%** transient, latency ~1–2 steps |

\*periodic/sampled catch = cost/R, linear up to cost=R≈139% (where catch=1); at 48% cost = 0.48/1.389 = 34.6%.
Coverage figures for Mech-2/3 are GEMM-only (no-compare); a real DMR adds a small O(M·N) compare tax to both.
**Reading:** in the **<3% region only periodic-N and Mech-1 exist, and they coincide** (catch = 0.72×cost ≤ 2.2%).
Mech-2 and Mech-3 cannot enter the <3% region at all (green-context granularity floors them at 14%/48%). Where they
*do* live (≥14%), Mech-2 is actually *more cost-efficient* than periodic per unit transient coverage (~1.0 vs 0.72,
spatial concurrency) — but floored at ≥14%, so it is irrelevant to the <3% question; its distinct value is
deterministic-fault coverage (a different class, Gap-5). Mech-3 buys *full* transient catch at ~1–2 step latency for
~48–61%. **No mechanism beats periodic-N for transient
within 3%.**

**Credibility notes.** (1) The mandate cited "r~2.6"; that is the full-coverage recompute-from-input factor. This run
uses ra-e2e's measured GEMM-recompute **R=1.389**; the conservation result (Mech-1 = periodic) is **R-invariant** — it
holds for any R because catch/cost = 1/R for both. (2) **What WOULD beat periodic**, and why it's out of reach: a
**risk-biased (non-random) subset** that checks GEMMs with a higher SDC prior first — explicitly out of scope (the
mandate specifies a *random* rotating subset; a useful prior would require per-GEMM fault-rate information we do not
have), or **genuinely idle resources** (none at SM=100%). The wall is real precisely because the only escapes are
information we lack or hardware we don't have. (3) Mech-2's prefill green-context partition shows a super-linear
throughput drop (112-SM partition = 62.8% of full-132, vs 84.8% SM-proportional) — green-ctx overhead and/or M=512
wave-quantization on 112 SMs; this only *raises* measured cost, hardening the wall.

---

## IN-vLLM GAP STATEMENT
These are end-to-end measurements in the ra-e2e saturated-harness *model* (the same real Mistral-7B shapes, real
trace, saturation, and per-phase cost model the shipped fork-1 detector was validated on), **not** measurements
inside a live vLLM serving process. Track B established that the only no-source-edit way to host any of this inside
real cudagraphed vLLM is `enforce_eager` (measured ~47% decode tax), and that capture-time in-graph node injection is
a demonstrated *primitive* on a toy `torch.cuda.graph` (not an end-to-end in-vLLM checksum). No transient coverage is
claimed *inside vLLM* by this fork. The shipped detector's host remains periodic out-of-graph recompute (fork-1).

---

## FORK-1 INTEGRITY (md5, entry vs exit)
| File | Entry md5 | Exit md5 | Match |
|---|---|---|---|
| `cipher_rt_phase4/libcipher_rt.so` (anchor) | `2edba0d2136f8ede4713d90a8f7cd55f` | `2edba0d2136f8ede4713d90a8f7cd55f` | ✅ |
| `ra_e2e/common.py` | `8276b0814020a8d9167a0edb2da58e85` | `8276b0814020a8d9167a0edb2da58e85` | ✅ |
| `ra_e2e/run_e2e.py` | `2de53982eaf1aef14234edd8a5f6819a` | `2de53982eaf1aef14234edd8a5f6819a` | ✅ |
| `ra_e2e/calib.json` | `ade20e25357d108423210b4c2fea3ac9` | `ade20e25357d108423210b4c2fea3ac9` | ✅ |
| `ra_e2e/trace.json` | `7d55b9120d05a74249db3dcc7618bb4d` | `7d55b9120d05a74249db3dcc7618bb4d` | ✅ |
| `ra_e2e/run_e2e_summary.json` | `e5ac91b2d54407ab85f1d3ee82251d57` | `e5ac91b2d54407ab85f1d3ee82251d57` | ✅ |
| `ra_e2e/throughput.json` | `8131a2bd8f7ed7fd0e3821fc879cb98a` | `8131a2bd8f7ed7fd0e3821fc879cb98a` | ✅ |
| `ra_e2e/detect.json` | `cefa60651cc8cb56885923164c070a4c` | `cefa60651cc8cb56885923164c070a4c` | ✅ |

Fork-1 uncorrupted: anchor and all ra-e2e shipping files byte-identical entry→exit. (Exit md5 re-confirmed in the
Guardrails run below.)

---

## GUARDRAILS (exit state)
- Anchor `2edba0d2136f8ede4713d90a8f7cd55f` UNCHANGED (entry + exit). No `.so`/source/vLLM/monkeypatch touched.
- All new work in `ra_fork2/`; no fork-1 file edited (md5 table above). Measurement path uninjected.
- Clock reset `nvidia-smi -rgc`; no compute procs left; no tags moved. (post-reset state recorded in Guardrails run.)
