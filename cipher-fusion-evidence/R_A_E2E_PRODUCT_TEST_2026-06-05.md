# R.A END-TO-END PRODUCT TEST — one detector, real serving workload

**Run date:** 2026-06-07 (file named per spec `R_A_E2E_PRODUCT_TEST_2026-06-05.md`).
**Mode:** READ-ONLY. No production `.so` touched, no production wiring changed.
**Anchor:** `/home/ubuntu/cipher_rt_phase4/libcipher_rt.so` md5 `2edba0d2136f8ede4713d90a8f7cd55f` — confirmed at entry; re-confirm at exit (§ Guardrails).
**Measurement path is NOT injected:** `LD_PRELOAD=[]`, `CUDA_INJECTION64_PATH=[]` (confirmed). Standalone PyTorch harness, real Mistral-7B-v0.1 weights, fp16 GEMMs. torch 2.11.0+cu130, H100 80GB HBM3.
**HW clock:** locked to 1980 MHz for the run; achieved clock under sustained load reported (prior runs ran ~1830 under power-cap). Reset `nvidia-smi -rgc` at exit; post-reset state reported.

This run tests the periodic-recompute detector — the gap-free `<3%` path the prior STEP B END-TO-END run (`step_b_e2e/`) identified — as **ONE configured detector** against a **realistic interleaved serving trace**, end to end. Prior runs measured detector cost in pieces (isolated GEMM, then a single saturated step) and returned PARTIAL. This run answers: does it hold serving throughput over a real trace, does it catch corruption in flight, how fast, and does it ever false-alarm.

---

## The detector under test (single config, no menu)

**Mechanism:** periodic **full recompute over cuBLAS** — the gap-free path (no custom kernel, no substitution gap; see `step_b_e2e` §4–6 for why every *per-step fused* config is blocked by the cuBLASLt substitution gap). Every **N** serving steps, re-run **all linear GEMMs of the current step** (32 layers × {q,k,v,o,gate,up,down} + lm_head) with a **second independent cuBLAS call** and compare against the cached output of that same step. The GEMM is fp32-accumulate / fp16-output (cuBLAS default, unchanged); the output **comparison is fp16 elementwise** `max|recompute − cache|`. Threshold **T** from the clean residual distribution (target 0 clean false positives by construction).
  - **[as-built note — comparison dtype]** The pre-registration above (written before building) said "fp32 abs-diff"; during implementation the *output comparison* was refined to **fp16** (the GEMM itself still accumulates in fp32 internally). Rationale: clean recompute is bit-identical → residual is exactly 0, and a Step-A flip gives a large fp16-representable diff, so fp16 separates clean(0) from corrupt **without the fp32-upcast traffic tax**. The measured **r and N below reflect the fp16 comparison**; an fp32-upcast comparison would raise r and push the shipping N higher (a more conservative, more expensive variant).

- **N is chosen empirically** (not assumed): the smallest N whose *measured* trace-weighted throughput overhead is `<3%` — i.e. the best (tightest) detection latency that still meets the throughput bar. The prior estimate was N≈34 (+2.9%); this run reports the measured N **even if it lands above 34** (that would be a finding — latency worse than the prior estimate — not a number to force back down).
- **What recompute catches (recurrence assumption, stated up front):** recompute flags a fault **iff the fault does not recur identically in the recompute**. Injecting a corrupted *stored* output (the Step-A fault model: fp16 top-exponent bit flip, |δ|≥2.76) → the independent recompute produces the clean value → caught. A **deterministic compute-unit fault** that yields the *same* wrong output on every call (including the recompute) is **invisible** to recompute-based detection. This is a named blind spot alongside the attention-internal QKᵀ/PV blind spot (recompute covers linear-GEMM outputs only).
- **Rotating variant (noted, not the primary):** recomputing *one layer per step*, rotating through all 32, has the same total cost and the same coverage as full-recompute-every-32, but smooths the per-N latency spike. Primary config tested = full recompute every N steps.

## The workload (real, interleaved — not a microbench)

- **Model/shapes:** real Mistral-7B-v0.1 (H=4096, I=14336, KV=1024, V=32000, L=32), fp16, real weight byte-traffic (~14.3 GB streamed per step). Attention-internal QKᵀ/PV excluded from the cost loop (the stated detector blind spot); excluding it *shrinks the step denominator ⇒ overstates the detector overhead %* ⇒ conservative for the throughput headline.
- **Serving loop:** custom **saturated CUDA-graph serving loop** (the prompt-permitted fallback). vLLM is **not** used as the host, for a principled reason, not a startup failure: the full-recompute detector needs the **in-context activations + weights** to re-run and compare, so a separate concurrent process/stream *fundamentally cannot host it* (separate CUDA context/memory); and in-engine insertion would be a forbidden production-wiring change that *also* breaks vLLM's CUDA-graph capture (the conditional every-N recompute branch cannot live inside a captured graph). The custom loop reproduces the real saturated decode regime (graph-captured, launch-gap-free) that eager HF cannot reach, with the detector inserted exactly where an engine's eager scheduler would place it (between captured replays).
- **Trace:** a generated request trace — mixed prompt lengths and decode lengths, arrivals over time — driven through a continuous-batching scheduler (chunked prefill; decode batch held near capacity = saturation). At least several thousand decode steps so prefill and decode interleave as in real serving. Observed prefill/decode wall-time split reported.
- **Saturation evidenced, not asserted:** SM% and MEM% sampled during prefill and decode.

---

## PRE-REGISTERED VERDICT CRITERIA (written before measuring)

The four-part scorecard decides the verdict. In **every** branch the **persistent-fault claim (strong)** is separated from the **transient-fault claim (probabilistic)**.

- **SHIPPABLE if:** trace throughput delta `<3%` **AND** persistent corruption caught within a bounded, stated latency (≤ N steps) **AND** 0 clean false positives. Transient single-step coverage stated as a known limit (~1/N per pass), not hidden.
- **SHIPPABLE-WITH-CAVEAT if:** throughput delta `<3%` and persistent-catch hold and 0 FP, **but** transient single-step coverage (~1/N at the chosen N) is too low to support an *all-SDC* claim. Then the product claim that ships is **"catches persistent SDC within N steps"** (a real product) with transient single-step explicitly probabilistic. State the N/coverage frontier.
- **NOT SHIPPABLE if:** throughput delta exceeds 3% at every N that gives useful detection latency, **OR** clean false positives are nonzero and unfixable. Name the driver.

**Expected (pre-registered prediction, to be confirmed or refuted):** throughput delta `<3%` at the empirically chosen N; 0 clean FP (T from a bit-identical or near-bit-identical recompute); persistent caught within N (latency uniform ~0..N−1); transient caught only when the faulted step is itself a check step → **caught-fraction ~1/N, detection latency a point mass at 0** (a transient fault on a non-check step is never caught, because every later check step recomputes its *own* step's GEMMs from the same propagated input). Predicted landing: **SHIPPABLE-WITH-CAVEAT** — persistent-SDC detector with bounded latency N at `<3%`; transient single-step is the caveat.

---

## RESULTS

All numbers measured this run (`ra_e2e/run_e2e.py` → `calib.json`, `trace.json`, `throughput.json`, `detect.json`, `run_e2e_summary.json`, console `full_run.log`). Real Mistral-7B-v0.1 fp16, H100, clock locked 1980 (achieved 1830/1590 under sustained load — power cap, reported per phase).

### 0. Calibration (measured, not assumed)

| step | base time | base tok/s | recompute time | **r** (recompute/base) | SM% | **MEM%** | clk (load) | power |
|---|---|---|---|---|---|---|---|---|
| decode (B=128) | 8.423 ms | 15,197 | 12.532 ms | **1.488** | 100 | **71%** | 1830 MHz | 619 W |
| prefill (M=512 chunk) | 15.143 ms | 33,811 | 20.079 ms | **1.326** | 100 | **52%** | 1590 MHz | 668 W |

- **Saturation is evidenced, not asserted:** both regimes **SM%=100** (the operative saturation here — back-to-back, launch-gap-free graph replays). Decode MEM%=71, prefill MEM%=52. Clock under sustained load: **decode 1830 MHz at 619 W** (below the 700 W cap → this drop is clock-lock/thermal slack, *not* power-capping); **prefill 1590 MHz at 668 W** (near the cap → power-capped). The run was *locked* to 1980. Reconciliation with the prior `step_b_e2e` (decode MEM 87%, prefill 48%): prefill matches (52≈48); decode is lower here (71 vs 87) because this forward adds `clamp_`+fp-`rms` compute and an identity-attention pass-through, raising the compute share and lowering memory-boundedness. **This does not affect the headline** — the detector cost is counted serially (§1), so the result is overlap/spare-BW-independent regardless of the exact MEM%.
- **Threshold T = 0.0**, measured: max clean recompute residual over **80** independently-generated clean check-steps (mixed B=128 / M=512) = **exactly 0** — two cuBLAS calls of the same shape are bit-identical, so the clean residual is 0. Trigger rule = residual > 0, so **any nonzero diff is caught**; the relevant separation is clean 0 vs the **smallest observed fault residual = 48.5** (not "infinite margin" — that figure is just an artifact of dividing by T=0). 0 FP is by construction (input-independent determinism), not by sampling — see §3.
- **r > 1.0 is the live finding.** The recompute is an *independent forward* (the recompute GEMMs ≈ base forward) **plus** the fp16 elementwise comparison. The comparison adds 33–49% over the base forward (r = 1.33–1.49). The prior STEP B END-TO-END estimate "+2.9% at N≈34" implicitly assumed the comparison was free (r=1.0). **This run measures it.**
- **Measurement design — cost path vs correctness path (disclosed):** the **timed** detector (`forward_recompute`, CUDA-graph-captured) measures the *cost* of recompute+compare and, for speed, compares against shared per-*shape* scratch buffers (H/KV/I/V) whose contents are irrelevant to the timing — so the graphed path is a faithful **cost** model but not a deployable per-layer comparator as literally written (a deployed detector keys the cache per (layer,name)). The **correctness** results (FP, transient, persistent catch, T) are produced by a separate **eager, value-exact** path (`eager_forward_cache` vs `eager_recompute_residual`) using a proper per-(layer,name) cache and real fault injection. Both paths share the determinism property (clean residual exactly 0, verified on each); the latency-in-*steps* and catch-fraction results are independent of eager-vs-graph execution.

### N choice (empirical: smallest N with measured trace-weighted overhead < 3%)

Trace-weighted overhead(N) = Σ_check t_recompute / Σ_all t_base (check every N steps):

| N | 1 | 4 | 8 | 16 | 32 | 34 | **45** | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| overhead | 138.9% | 34.8% | 17.5% | 8.73% | 4.30% | 4.17% | **2.98%** | 2.97% | 2.15% |

**Chosen shipping default: N = 45** (smallest N with measured overhead < 3% → best/tightest detection latency under the throughput bar). **N is a soft number in a 34–47 band, not a precise spec** — what is robust is the *verdict*, that the gap-free path ships at <3% with bounded latency:
- *Lower bound ~34* — the prior N≈34 is recovered as the *recompute-GEMM floor* (r=1.0, fused/free comparison → crosses 3% at N≈34); a fused comparison kernel gets here.
- *Measured 45* — the configured detector uses standard torch comparison ops (r=1.33–1.49) → N=45 (the conservative *measured* number, exactly as pre-registered: "report N even if above 34 — that's a finding").
- *Upper ~47* — the timed comparison reads 4 per-*shape* scratch buffers (~13 MB, L2-resident → cache-read side nearly free), so N=45 is a *mild optimistic bound* on a deployed per-(layer,name) comparator (~225 tensors ≈ 360 MB read **cold** from HBM each check ≈ +0.14 ms on the ~4 ms compare → r→~1.51, **N→~47**, ≈ +2 steps). Immaterial to the verdict.
- *Margin option* — N=64 → +2.15% (latency 63) buys comfortable headroom below 3%.

### 1. Throughput (headline) — real wall-clock over the whole trace

**Trace:** 1500 requests (mixed prompts 64–2048, mixed decodes 16–768, Poisson arrivals), continuous-batching scheduler (chunked prefill M=512, decode batch held near B=128 = saturation) → **6301 steps (3370 decode / 2931 prefill)**. Observed split: **prefill 61.0% / decode 39.0% of wall** — a realistic, prefill-leaning interleaved mix. 140 detector check-steps at N=45.

| | wall | tok/s (total) | decode tok/s | clk (load) | power |
|---|---|---|---|---|---|
| **WITHOUT detector** | 72.51 s | 20,914 | 5,041 | 1830 MHz | 596 W |
| **WITH detector (N=45)** | 74.66 s | 20,313 | 4,896 | 1830 MHz | 598 W |
| **Δ** | **+2.15 s wall (+2.96%)** | −601 (−2.88%) | −145 (−2.88%) | — | — |

**Headline: end-to-end wall-clock overhead = +2.96%** (the deployable cost convention: extra-wall / base-wall). The equivalent *throughput drop* is −2.88% (tok/s_on/tok/s_off; the two differ because Δ% is taken against different denominators — wall-overhead is the larger, more conservative figure). **Both are under the 3% bar.** Margin is thin (2.96% vs 3.0%), attributable entirely to the comparison cost (r); for comfortable margin, N=64 → +2.15% (latency 63), or a fused comparison → ~+2.9% at N≈34. Two properties make the *delta* robust even though the absolute tok/s are idealized (see note):
- **Mix-independent:** overhead ≈ r/N per check regardless of the 61/39 split (the detector fires every N steps on whichever step type is current); the measured +2.96% matches the analytic trace-weighted 2.98% to 0.02 pp.
- **Overlap-independent:** periodic recompute is counted *serially* (the ON pass replays `g_base` then `g_recompute` on the same default stream → the extra time is fully counted), so the number does **not** depend on the spare-HBM/overlap question that defeated the per-step fused-checksum path (`step_b_e2e §5`). The dominant-decode-has-no-spare-BW problem is irrelevant to this product.
- **[note — absolute numbers are idealized, the delta is not]** The loop replays one fixed captured graph per type (B=128 / M=512) with a fixed input buffer — no KV-cache growth, no varying decode batch, no scheduler stalls, no padding cost. So the *absolute* tok/s and "saturated decode regime" are best-case fixed-graph replay, **not** a full continuous-batching engine. The **headline delta** is structurally insulated from this (mix- and overlap-independent, analytic-corroborated), which is why the custom loop is a sound vehicle for the *overhead %* even though it idealizes throughput level.

### 2. Detection latency: steps-from-injection-to-catch

Fault injected = Step-A harmful mechanism (fp16 **top-exponent bit-14 flip** into a stored GEMM output). Note on magnitude: a bit-14 flip flips the *right* bit but does not by itself pin |δ|≥2.76 — the resulting δ is workload-dependent (a flip on a small output can land as low as 2.0). This does not weaken detection: with **T=0 any nonzero δ is caught**, including sub-2.76 flips. In this run all injected flips landed on large GEMM outputs, so all were in the harmful δ≥2.76 class (smallest observed 48.5).

- **Persistent (stuck) fault** — latency is a real spread, `latency = N−1−(inj_step mod N)`, uniform on **0..44**. Over 30 injections (inj_step drawn across `[0,3N)`): **min 4 / mean 25.2 / max 44 steps** (bound N−1 = 44; min 4 is a finite-sample low, not the floor). The latency is the **schedule offset** (deterministic arithmetic from inj_step) and the **catch is verified** for all 30 (a corrupted step's residual > T). At the trace's ~11.5 ms mean step time that bound is **≤ ~0.5 s** (mean ~0.29 s). **100% caught** (all 30 had residual > 0; residuals **48.5–27,680** ≫ 0).
- **Transient (single-step) fault** — latency is a **point mass at 0**, not a distribution: a transient flip at step *t* is caught **iff *t* is itself a check step** (recompute-clean vs cached-corrupt, latency 0); if *t* is a non-check step the corruption propagates but every later check step recomputes *its own* step's GEMMs from the same propagated input → residual 0 → never caught. Verification that the *mechanism* fires: **60/60** injections caught when the faulted step is a check step (residuals **147.9–23,984**, all finite, no inf/nan). The caught-fraction over a random check schedule is the analytic **1/N = 2.2%** (not an empirically-traced fraction; it is the combinatorial P(faulted step is a check step)).

### 3. False positives over the clean trace (must be 0)

- **0 false positives.** T=0 from a bit-identical recompute; **80 independently-generated clean check-steps (both M sizes) → max residual = 0.0**, trigger count = 0. The full trace's 140 clean check-steps are **not** individually re-run through the FP test; the 0-FP-over-the-full-trace claim is licensed by **input-independent determinism** (every clean check is a same-shape cuBLAS recompute = bit-identical = residual exactly 0 ≤ T=0), corroborated by the 80-sample max of 0.0 — not by representative sampling. The separation is clean 0 vs the smallest observed fault residual 48.5 (the "infinite margin" in `detect.json` is just the T=0 division artifact). No diagnosis needed (criterion met). (FP/catch are from the eager value-exact path; the timed graph path's determinism was separately confirmed to also give clean residual 0 — see §0 "Measurement design".)

### 4. Persistent vs transient coverage (the product's real coverage claim)

| fault class | model | caught? | latency | coverage |
|---|---|---|---|---|
| **Persistent / stuck** | corrupts every step | **Yes, 100%** | ≤ N−1 = 44 steps (mean ~22–25) | **complete within N** |
| **Transient / single-step** | corrupts one step once | only if that step is a check step | 0 (point mass) | **~1/N = 2.2%** per pass |

- **N/coverage frontier:** there is **no N that gives both <3% throughput and useful transient coverage** — N=1 catches 100% of transients but costs +139%; N=45 costs <3% but catches 2.2% of transients. Catching all transients requires per-step checking (N=1) or a *fused per-step checksum* — and the fused per-step checksum is exactly the path `step_b_e2e` found gap-blocked (the substitution gap, the PARTIAL driver). **The gap-free recompute path trades transient coverage for deployability.**
- **Named blind spots (so the persistent claim does not overread):** (1) recompute catches a fault **only if it does not recur identically in the recompute** — a *deterministic compute-unit fault* that yields the same wrong output on every call (including the recompute) is **invisible**. The injected/validated class is the storage/transient output-bit-flip (the Step-A model), for which recompute is valid. (2) Attention-internal QKᵀ/PV is uncovered (recompute checks linear-GEMM outputs only) — the standing Step-A blind spot.
- **Rotating variant (responsive to the spec's "one layer's GEMMs / sampled layer set"):** recomputing one layer per step, rotating through all 32, has the same total cost and coverage as full-recompute-every-32 but smooths the per-N latency spike. Not the primary config; primary = full recompute every N.

---

## VERDICT → **SHIPPABLE-WITH-CAVEAT** (as pre-registered)

Throughput delta **+2.96% < 3%** ✓ · persistent corruption caught within a **bounded N−1 = 44-step** latency ✓ · **0 clean false positives** ✓. The discriminator is the *product claim*, not the data — so the two claims are separated:

- **Persistent-SDC claim (strong) → SHIPPABLE.** The configured periodic-recompute detector is a **gap-free, bounded-latency (≤44 steps = N−1, ~0.3–0.5 s) SDC detector — for linear-GEMM-output, recurrence-breaking (storage/transient) corruption — at +2.96% end-to-end wall overhead over a realistic interleaved serving trace, with 0 false positives.** It needs no custom kernel (no substitution gap) and no overlap headroom. It protects the teacher-forced 100/100-FAULT=0 substrate. (The fault-class qualifier is load-bearing — see the blind spots in §4; it is **not** an all-fault detector.)
- **Transient single-step claim (probabilistic) → CAVEAT, stated not hidden.** Per-pass coverage is ~1/N = **2.2%** at N=45 (latency 0 when caught). The product that ships is *"catches persistent SDC within N steps,"* **not** *"catches all SDC."* No acceptable-throughput N raises transient coverage materially; catching every transient needs the gap-blocked per-step fused checksum.

**Driver of the caveat:** the periodic structure itself (a transient on a non-check step is structurally invisible to same-step recompute), backed by the standing substitution-gap finding that blocks the only per-step alternative. **Refinement vs the prior PARTIAL:** the gap-free path is confirmed shippable end-to-end, at **N=45 / +2.96%** (the prior +2.9%@34 was the recompute-GEMM floor; this run measured the +33–49% comparison cost, moving the shipping N from 34 to 45 — or back toward 34 with a fused comparison). **Anil's call** remains: ship periodic-recompute as a persistent-SDC detector now (N=45, accept ≤44-step latency = N−1 and ~2.2% transient coverage) / invest in a fused comparison kernel to tighten N toward 34 / accept the per-step fused-checksum substitution gap to gain transient coverage.

---

## Guardrails (exit state)

- **Anchor `/home/ubuntu/cipher_rt_phase4/libcipher_rt.so` md5 `2edba0d2136f8ede4713d90a8f7cd55f` — UNCHANGED entry + exit.** No `.so` touched, no production wiring changed, no tags moved.
- **Measurement path confirmed NOT injected:** `LD_PRELOAD=[]`, `CUDA_INJECTION64_PATH=[]` (checked at entry). Standalone PyTorch harness.
- **Clock:** locked 1980 MHz for the run; achieved **1830 MHz (decode) / 1590 MHz (prefill)** under sustained load (700 W power cap). **Reset `nvidia-smi -rgc` at exit → SM idling 375 MHz, max/applications 1980 MHz (unlocked default).**
- **0 compute processes resident at exit.** Every number from this run.
