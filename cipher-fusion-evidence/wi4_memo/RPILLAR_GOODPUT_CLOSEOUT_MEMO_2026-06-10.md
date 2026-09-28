# CIPHER R-PILLAR GOODPUT CLOSE-OUT MEMO

**Date:** 2026-06-10 (loop ran 2026-06-09; WI-4 completed on resume; memo panel-verified, fixes applied — see panel
note at end). **Scope:** the reliability pillar — online SDC detection (R.A), quarantine (R.B), per-GPU verdict +
fleet schema (R.C), and the three goodput definitions (WI-1/2/3). **Question answered:** under which definition of
*goodput* does CIPHER's R-pillar pay for itself, and what is the honest deployment posture? Every number below
traces to a panel-verified report (provenance table at the end).

---

## 1. Why three definitions

The R.A+R.B capstone established that "does CIPHER improve goodput?" is ill-posed until *goodput* is defined. Under
the serving-correctness definition the answer is a measured NO — and treating that as the win condition was the
original framing error. The close-out therefore measured each definition separately, pre-registered before each
measurement:

| Goodput definition | Verdict | Headline |
|---|---|---|
| **Correctness (serving under fault)** — useful tokens delivered | **NOT a win — by mechanism** | Detector detects, doesn't correct: WITH and WITHOUT deliver the **same 17 useful tokens**; CIPHER's value is **garbage served 239 → 25 (N=45) / 4 (N=8) / 0 (batch withhold)** at +4.0% / +12.6% overhead. A **safety** trade, priced in throughput. *(Bound: batch=1 greedy, one fault site, launch-bound host — the trade structure generalizes; the absolute tok/s and overhead percentages are config-specific. Persistent-fault property: between-check transients still ship silently, catch ~1/N.)* |
| **TRAINING (progress-time / total-time)** — badput reduction | **Conditional WIN** | Persistent SDC caught within N (latency 4 @N=8 / 25 @N=45, 0 FP / 3,064 checks). Silent-SDC regime **+3.7 → +26.8pp** effective-training-time (**computed @N=8 detector, +5.59% overhead; p=1 upper bound; +9.5pp @p=0.5**); **−4.8pp NET LOSS @N=8** for fast-discovered faults (N=45 runs at +0.99% overhead with latency 25 — its pp deltas were not computed in WI-1). Win condition = *silent discovery* AND *covered op* AND *fault rate above breakeven*. *(Bound: measured on a single-GPU 7B LoRA step loop — real fp16 GEMM timing, but distributed-job badput (NCCL, all-reduce, sharded checkpoint) out of scope; breakeven fault-rate is step-time-invariant, the GPU-seconds model is single-node.)* |
| **SERVING (SLO-attainment rate)** — DistServe-style | **Unmeasured for the mux — WALL** | Single-tenant baseline is latency-comfortable (strict-SLO attainment **≥0.98 across λ=1–64** — 0.988 at λ=32/64, 0.995 sustained; TTFT p90 ≤76 ms, TPOT ≈14 ms; **no SLO knee reached, no sustained ceiling established** — transient runs; eager host, conservative on latency; one GPU, one model). The mux lives in the compiled substrate and needs a vLLM-cooperating host; its validated **3.06–3.30×** stays strictly in the **tok/W-density** lane, never relabeled SLO-goodput. |

## 2. The reliability product that DID validate

- **Per-GPU verdict (R.C):** R.A SDC ⊕ W.6 cohort ⊕ NVML fuse into one rc.v1 verdict on real vLLM — injected
  persistent SDC → DEGRADED with mid-run onset (detected step 64 / confirmed 72), **0 FP over 4,352+ clean checks**
  (a benign SwPowerCap throttle did not escalate), descriptor three-signal-consistent (`VLLM::EngineCore`).
  The signal is **online correctness, orthogonal to hardware counters** — by mechanism, not a measured side-by-side
  (DCGM/dcgmi absent on the pod; NVML-only telemetry) — and NOT "caught a hardware SDC DCGM missed" (software flip ⇒
  NVML-clean is expected). Co-residence does not smear the per-process signal (clean co-tenant 0/514 detections
  during the 46.6 s fault window, n_resident=2 in 92/92 samples).
- **Fleet consumability (WI-3, post-panel v2):** rc.v1 aggregates by `gpu.uuid` end-to-end — negatively-tested
  schema (4/4 malformed classes rejected), totals-level idempotence, 1000-GPU **simulated** replay (one real event
  replayed; onset-step varied only — signal diversity is synthetic and thin). Aggregation LOGIC validated; real
  multi-GPU scale + a deployed collector service remain **unbuilt**.
- **Quarantine (R.B):** persistent SDC quarantined within ≤N at the dispatch boundary, **0 false quarantine**;
  batch withhold-whole is substrate-legal (0 garbage served *for persistent/on-check faults*); **streaming/
  already-emitted = WALL** (needs vLLM).

## 3. Fleet economics (the deployment decision rule)

Detector overhead is paid on **every** GPU continuously; badput-avoided accrues only on GPUs that fault. So the
fleet is NET-positive **iff the fleet fault rate (faults per GPU-step) exceeds WI-1's breakeven** — **1 per 17,820
GPU-steps (N=8) / 1 per 98,121 (N=45)** in the silent-1000 regime (additive per GPU-step under homogeneity;
cross-check 138.35/(0.0559×0.1389)=17,818 ✓). WI-3's depicted 2%-degraded/10k-step fleet sits at 1 per 500,000 —
**NET-NEGATIVE (−5.11pp @N=8 / −0.80pp @N=45; gross 0.77 GPU-h avoided vs 21.6 / 3.8 GPU-h overhead)** — and WI-3's
pre-registered "fleet WITH > WITHOUT" expectation was **falsified as stated** (its formula omitted healthy-GPU
overhead; disclosed as a PREREG deviation). **Calibration against the only empirical anchor in the arc:** the N=45
breakeven (1/98,121 GPU-steps) is ≈1 fault per 3.8 GPU-hours at the measured 0.1389 s step (≈38–114 GPU-h at WI-1's
stated 10–30× real-7B step times), while WI-1's Meta Llama-3 anchor is ~1 interruption per ~50,700 GPU-hours
*all-cause* (SDC a rarer subset) — orders of magnitude below breakeven. **No source measures a realistic fleet-wide
SDC rate above breakeven.** The honest pitch is therefore the decision rule, not a deployment blanket: **fleet-wide
always-on does not pay at anchored rates; deployment is targeted — suspect GPUs/jobs, post-incident triage, or
environments whose believed silent-SDC rate exceeds the breakeven.** (WI-1's per-fault badput figures are
modeled-accounting — measured step time × modeled discovery delay — and the +pp examples are p=1 upper bounds.)

## 4. Standing walls (none re-litigable without new mechanism)

1. **Eager hosting tax +51.2%** (629.2 eager vs 1289.0 cudagraph tok/s): Path-1 in-graph injection is
   WALL-WITH-MECHANISM — vLLM rejects ANY external mutation of its decode cudagraph (even a trivial memset node
   crashes init memory-profiling). Detector marginal itself: **+3.1% @N=45 on the eager base**.
2. **Transients between checks are MISSED entirely** (R_AB measured: 237/256 tokens corrupted, GEMM clean at next
   check); catch ~1/N. Fork-2: no non-checksum mechanism beats periodic at <3% (sampling = conservation; DMR =
   ≥14.2% hardware floor; lagged recompute = doubling).
3. **Coverage = fp16 linear GEMMs only** (attention/FlashAttn, lm_head, non-GEMM uncovered ⇒ p<1 in fault space);
   **mercurial-core blind** (same-GPU recompute reproduces deterministic faults).
4. **Per-tenant causation** — v1 localizes observation, not cause (v2 needs a causation signal).
5. **Streaming quarantine** and **mux-as-SLO-goodput** both need vLLM/substrate cooperation (out per substrate line).

## 5. Posture

**Where the R-pillar pays (each measurement-backed, each conditional):** (1) **training under a silent-SDC threat
model whose fault rate exceeds the breakeven** — silence is necessary but NOT sufficient (WI-3's all-silent depicted
fleet still loses); at anchored rates this means targeted deployment, not fleet-wide always-on (§3); (2)
**safety-critical serving** — zero-garbage batch withhold or ~90% garbage-served cut streaming @N=45 for +4.0%,
*for persistent/on-check faults only* (between-check transients still ship silently); (3) **fleet reliability
triage** — which GPU, since when, what was running; a correctness signal hardware counters do not carry.
**Where it does not pay:** serving throughput/goodput (it strictly reduces it), SLO-goodput claims (unmeasured —
walled), rare fast-discovered faults, **transient-fault regimes (mostly missed between checks)**, per-tenant
attribution, fleet-wide always-on at anchored fault rates.

**What would change the answer:** a vLLM-cooperating capture hook — removes the 51.2% eager tax, but the detector's
**cudagraph-frame marginal is ~6% (estimate, UNMEASURED; may exceed 3% at N=45 — needs larger N or a cheaper compare
to get under 3% live)**; a causation signal (per-tenant v2); a deployed collector + real multi-GPU run; a
steady-state SLO soak to find WI-2's missing knee; opt-in DMR (≥14%) for the deterministic-fault mode; and any
measured fleet SDC-rate evidence to test the §3 anchor.

## Provenance
| Claim | Source (all in cipher-fusion-evidence/) |
|---|---|
| 17 useful tokens; 239→25/4/0; +4.0%/+12.6%; transient between-check miss; batch=1 bound | `R_AB_GOODPUT_RECOVERY_2026-06-09.md` (+ rc_ab/goodput.json, quarantine_demo.json) |
| Latency 4/25; 0 FP 3,064; +0.99%/+5.59%; +3.7/+26.8/+9.5/−4.8pp (@N=8); breakeven 17,820/98,121; 138.35 modeled; LoRA bound; Meta anchor | `wi1_trainingbadput/WI1_TRAINING_BADPUT_2026-06-09.md` (+ wi1_accounting.json) |
| Attainment ≥0.98 (λ=1–16) / 0.988 (λ=32/64) / 0.995 sustained; TTFT p90 ≤76 ms; TPOT ≈14 ms; no knee; mux WALL; 3.06–3.30× tok/W lane | `wi2_servinggoodput/WI2_SERVING_GOODPUT_2026-06-09.md` (+ wi2_lam*.json, wi2_sustained.json; floor corrected 2026-06-10) |
| Fleet agg validated; −5.11/−0.80pp; 0.77 GPU-h gross vs 21.6/3.8 overhead; 1/500k; PREREG falsification | `wi3_fleetagg/WI3_FLEET_AGG_2026-06-09.md` (+ fleet_view.json md5 c451b691) |
| DEGRADED onset 64/72; 0 FP 4,352+; 0/514 co-tenant; orthogonality-by-mechanism; DCGM absent | `R_C_PERGPU_FLEET_2026-06-09.md` (+ rc_pergpu/inject_verdict.json) |
| 629.2/1289.0 tok/s; +51.2% tax; +3.1% eager marginal @N=45; ~6% cudagraph-frame estimate | `R_A_REAL_VLLM_2026-06-08.md` (+ ra_realvllm/tokps_result.json) |
| Path-1 memset-node crash (mutation itself is the wall); cudagraph marginal unmeasured | `R_A_PATH1_BUILD_2026-06-09.md` |
| Fork-2: sampling conservation; DMR ≥14.2% floor; lagged doubling | `R_A_FORK2_TRANSIENT_2026-06-08.md` |

**Integrity notes:** R_AB's overhead figures are the kmod-decoupled re-measurement (an inherited `/dev/cipher`
heartbeat was panel-caught, disabled, all runs re-run; tokens unchanged); WI-1's detector numbers post-date a
panel-disclosed NaN-FP bug fix (two run-batches discarded). Both episodes are recorded in their source reports.

**Memo panel (2026-06-10, 3-dimension adversarial: NUMBERS-TRACE / HONESTY / COMPLETENESS):** ~45 load-bearing
values traced exact to sources, no untraceable numbers. Materials found+fixed in this version: WI-2 attainment floor
0.988→0.98 (contradicted by wi2_lam*.json; WI-2 prose corrected at source with a dated note); §5 capture-hook
claim restated to the ~6%-unmeasured cudagraph-frame estimate; "deploy at N=45" recommendation replaced by the
rate>breakeven decision rule + Meta-anchor calibration (fleet-wide always-on NOT evidenced); WI-1 single-GPU-LoRA,
R_AB batch=1-greedy, and persistent-only bounds restored at every pitch site; transient regimes added to
"does not pay". Minors fixed: N=8 config labels on pp figures; DCGM-absent caveat; eager-host bound;
kmod/NaN episodes recorded; WI-3 thin-diversity clause; wi2_sustained.json provenance.

**Discipline:** memo only (no GPU run); anchor `2edba0d2…` md5 verified at memo time; WI-4 wrote only in `wi4_memo/`
(plus the disclosed WI-2 prose correction).
