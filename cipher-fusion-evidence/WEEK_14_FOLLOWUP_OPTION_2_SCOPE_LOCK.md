# W14 FOLLOWUP — Option-2 Real-Workload Koopman Measurement (Scope-Lock)

**Date:** 2026-05-25
**Amendment:** Step 0 prepended per Anil 2026-05-25; §10 Q1 Option (a) confirmed
**Pre-state anchor:** `week-13-14-complete` (verified on disk at amendment time)
- `cipher_rt_phase4` `25970f3` (tags `week-13-14-complete`, `week-14-complete`, `week-14-step-3-c-lmhead-validate`, `week-14-step-3-remember-validate`)
- `libcipher_rt.so` md5 `097cf8d907a7e866a3e3640eb0993003`
- `cipher_kmod` `8c643fc` (0.6.5, tags `week-13-14-complete`, `week-9-complete`, `week-9-step-5-n128-soak`)
- `cipher_kv_bridge.so` at W12 Step 3 anchor
- `cipher-fusion-evidence` `d5924192` (this file lands as next commit)

**Authority:**
- W13-14 close §5 (`WEEK_13_14_COMPLETE.md:81-89`) names this campaign as the queued next step
- Anil adjudication 2026-05-24: *"see once we close this we will work on option 2"*
- Anil amendment 2026-05-25: Step 0 prerequisite + §10 Q1 Option (a) confirmed
- Memory pointer: `w14-step-3-followup-mistral-tok-s`, `vllm-v1-worker-subprocess`

**Type:** post-phase measurement campaign (not a numbered week; sub-campaign before W15-17 CP 5.5)

---

## 1. Why this campaign

W14 Step 3 S3.C proved the substrate fires Koopman exactly once on a single synthetic LM-head matmul under a harness-raised β threshold. The substrate ships with `CIPHER_KOOPMAN_OOD_THRESHOLD=0.05` which keeps Koopman from firing on real production inputs (per W14 Step 2 G §5.3 OOD sweep). At v1, Goal 4 substrate is **correct but the actuator never fires on real customer workloads.** The v1 product claim ("O(1) Koopman compute substitution") is not yet measured on a real workload.

This campaign settles the question one way or the other and pre-commits to **both outcomes shipping at quality.** It is not a pass/fail campaign.

---

## 2. The strong prior — read this first

W14 Step 2 E closeout (cipher-fusion-evidence `e4682df`):

> "alpha-matched calibration cannot recover top-1 gate; beta threshold sweep shows no knee with positive utility (ALL measurement inputs residual > 0.5 vs calibration V_x manifold; Koopman never fires when correct, output garbage when forced)"

This is the architectural-ceiling finding from rank-64 vs real LLM activations + prompt-trajectory diversity + softmax sensitivity composition. The W14 Step 2 E sweep was on **calibration-distribution synthetic inputs**, not vLLM real decode tokens. Whether the picture differs on real decode tokens is exactly what this campaign measures.

**The prior says expected outcome is Branch B (architectural ceiling reproduces).** The scope-lock is structured to short-circuit out at the calibration gate if Branch B is confirmed early, rather than spending the A/B measurement budget chasing a measurement we already know the shape of.

---

## 3. Three locked decisions

1. **Guarded pipeline, not parallel deliverables.** Calibration is Step 1 and acts as a HARD GATE. If no positive-utility knee exists on TinyLlama real-decode warmup tokens, Steps 2-3 are skipped and the campaign exits directly to the reframe close-out. Honest framing is not deferred to the end — it is the default branch unless the prior is contradicted.

2. **TinyLlama N=4 is the primary headline.** Mistral-7B N=4 is conditional on both (a) TinyLlama showing a positive-utility knee, AND (b) the carry-forward Mistral E.7 env-block being resolved at the start of Step 3. If E.7 is not resolved, Step 3 is documented as deferred and the campaign closes on TinyLlama alone with explicit "Mistral pending E.7 resolution" framing.

3. **Both branches ship at quality.** Pre-committed framing language:
   - **Branch A (positive utility found AND positive net tok/s at Step 2):** "O(1) compute substitution delivers X% tok/s lift at Y% top-1 on TinyLlama-1.1B vLLM N=4 decode" + Mistral statement if Step 3 ran
   - **Branch B (architectural ceiling reproduces on real workload at calibration gate):** "substrate-deliverable subset, fire path validated end-to-end, production firing deferred to v2 pending [specific architectural finding from this campaign]" — no scope-down language, no apology language
   - **Branch C (Step 1 finds knee BUT Step 2 tok/s lift ≤ 0):** "substrate fires correctly at calibrated threshold (top-1 ≥ 0.90, substitution rate [X]%) but engine-path overhead dominates the saving at real N=4 decode; production net-positive deployment deferred to v2 pending substrate-overhead reduction. Substrate-deliverable contract intact." Step 1 calibration finding remains a positive engineering result; net-tok/s deferral is the v2 surface area, not a Goal 4 failure.

---

## 4. Four-step sequence (Step 0 + three measurement steps)

### Step 0 — vLLM V1 worker-subprocess GOT patch hook (1-2 ED) — HARD PREREQUISITE

**Why this step exists (not in original scope-lock):** vLLM 0.20.2 V1 architecture spawns an EngineCore subprocess for the worker. The worker subprocess inherits parent's `LD_PRELOAD=libcipher_rt.so` but **never runs `InitializeInjection2()`** — so `cublasGemmEx` and SDPA-dispatch GOT patches are not installed in the worker. All vLLM decode GEMMs run unintercepted in the worker subprocess. A previous option-2 attempt (session terminated by SSH disconnect mid-debug) ran five sweep phases against the current substrate and recorded 0 handled calls at every β. Reading was initially "Koopman doesn't fire" — actual cause was substrate never saw the GEMMs. Without Step 0, Step 1 calibration measures false-negative zeros and Branch B exits on a null measurement substrate. See memory `vllm-v1-worker-subprocess`.

**Scope:**
- Identify the correct vLLM 0.20.2 worker-init hook site. Candidates: `vllm.worker.Worker.init_device`, existing `cipher_vllm_plugin` worker-time entry, `VLLM_WORKER_MULTIPROC_METHOD`-aware init point. Required ordering: AFTER `libcublas` is mapped into the worker subprocess, BEFORE the worker's first cuBLAS call (same ordering pattern that works in parent).
- Extend `cipher_vllm_plugin` to call `InitializeInjection2()` from the worker hook. May require ensuring `InitializeInjection2` is exported from `libcipher_rt.so` for the plugin to `dlsym` it.
- No substrate (cipher_rt_phase4) code change expected; plugin-only edit. If substrate symbol-export change is needed, surface as Step 0 sub-step before proceeding.

**Tag:** `option-2-step-0-vllm-worker-init-hook`

**Step doc:** `WEEK_14_FOLLOWUP_OPTION_2_STEP_0_VLLM_WORKER_HOOK.md` — must capture the Goal 5 framing residue (§7 honest residue #6).

**HARD VERIFICATION GATE — must pass before Step 1 starts:**
- **Gate (1):** `/proc/<worker_pid>/maps` shows `libcipher_rt.so` loaded in the EngineCore worker subprocess. **NECESSARY-NOT-SUFFICIENT** — LD_PRELOAD inherits across spawn, so this gate also passed in the failing prior attempt (`/tmp/option2_sweep_test.json` 2026-05-24 16:23). Library presence ≠ GOT patches installed.
- **Gate (2):** **Counter-delta discriminator** (advisor #2). Snapshot substrate counters in the worker process BEFORE and AFTER a short decode burst (TinyLlama, ~128 tokens, LD_PRELOAD-only no-CUDA_INJECTION64_PATH). At minimum one of `cipher_rt_cublas_total` / `cipher_rt_koopman_calls_total` must strictly increase in the worker. Stayed-at-0 was the prior attempt's symptom; strict increase confirms GOT patches are installed and intercepting in the worker process specifically (not just the parent). The prior `option2_uniproc.json` (uniproc workaround, parent process) showed `total=284800 / skipped=281600` after a decode burst — that is the order-of-magnitude signal expected at Step 0 pass under multiproc.
- **Gate (3):** With `CIPHER_KOOPMAN=1 CIPHER_KOOPMAN_OOD_THRESHOLD=0.99` (always-fire-approx), `cipher_rt_koopman_calls_handled` strictly increases in worker — confirms the Koopman path is reachable (not just classifier/skipped path). At β=0.99 we should see handled ≥ 1; at β=0.05 we expect handled stays 0 (architectural ceiling, Branch B prior).
- If Gate (2) fails (counter still 0 after fix), the diagnosis is more complex than "InitializeInjection2 never ran" and Step 0 does not fix the actual failure — STOP and surface; do not silently re-attempt. Fallback to non-vLLM measurement (custom PyTorch loop) reframes the entire campaign and requires re-adjudication.

**HARD BUDGET:** If Step 0 cannot land cleanly within 2 ED, surface to user. Do not extend Step 0 silently.

### Step 1 — Calibration gate on TinyLlama warmup (1.5-2.5 ED) — HARD GATE

**Workload:** TinyLlama-1.1B vLLM decode (N=1 is sufficient for calibration; warmup tokens only)

**Method:**
- Use existing `bench_llm.py` (md5 `558865fd`) infra at week-6-bench-harness anchor
- Instrument substrate to record per-GEMM `residual_ratio` over a 5000-token warmup run
- Sweep β over `{0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9}` (W14 Step 2 G default 0.05 → S3.C harness 0.7 → α/β extension upper bound)
- For each β, compute: (i) fire rate (% GEMMs handled), (ii) top-1 vs vanilla on fired GEMMs, (iii) KL distribution on fired GEMMs
- Surface residual_ratio histogram for visual knee inspection
- **Substrate-overhead microbench (per §10 Q1 Option (a) — Anil 2026-05-25 confirmed):** 1k synthetic GEMM workload at `CIPHER_KOOPMAN=1 β=0.99` (engine on, fire_rate=0; classifier + residual_ratio compute + dispatch overhead path only) vs `CIPHER_KOOPMAN=0` (engine bypassed). Per-GEMM overhead delta = break-even floor; any positive substitution at Step 2 must clear this delta to deliver net-positive tok/s. Output: a single break-even substitution-rate number, recorded alongside the residual_ratio histogram.

**HARD GATE — exit conditions:**
- **GATE A (Branch A path):** ∃ β where `fire_rate ≥ 0.01 AND top-1_on_fires ≥ 0.90` → proceed to Step 2 at the highest such β
- **GATE B (Branch B path):** ∀ β in sweep, `NOT(fire_rate ≥ 0.01 AND top-1_on_fires ≥ 0.90)` → SKIP Step 2, SKIP Step 3, proceed to Step 4 (close-out Branch B). This is the strict negation of GATE A — no adjudication step embedded.

**Tag:** `option-2-calibration-gate`

**~LOC:** ~150 LOC harness extension + sweep runner, no substrate code change

### Step 2 — TinyLlama N=4 vLLM A/B at calibrated β (1-2 ED, CONDITIONAL on GATE A)

**Workload:** TinyLlama-1.1B vLLM N=4 decode (matches W6 Phase A cross-tenant baseline)

**Method:**
- Vanilla baseline run: `CIPHER_KOOPMAN=0` (engine disabled at env-gate per W13-14 close §3 line 54; NOT β=0.05 — at β=0.05 the classifier + residual_ratio compute still runs per GEMM and contaminates the A/B by exactly the overhead we are trying to measure)
- Koopman ON run: `CIPHER_KOOPMAN=1` β = Step 1 calibrated value
- The A/B isolates the *engine path including overhead*, not just the fire path
- Metrics on both: tok/s aggregate, per-tenant tok/s, top-1 (Koopman vs vanilla per fired GEMM), KL distribution, substitution rate

**Reporting:**
- Headline: tok/s delta with confidence interval
- Validation: substitution rate × top-1 confirms Step 1 calibration holds at N=4
- Honest framing: even if positive, this is one workload class — not a generalization claim

**Tag:** `option-2-tinyllama-ab`

**~LOC:** ~80 LOC A/B runner using bench_llm.py, no substrate code change

### Step 3 — Mistral-7B N=4 A/B + E.7 unblock (1 ED, CONDITIONAL on Step 2 + E.7)

**E.7 prerequisite:** carry-forward from W7-9 → W10-12 → W13-14. Step 3 begins with explicit E.7 status check. If still blocked, document and close on TinyLlama only.

**Workload:** Mistral-7B-Instruct vLLM N=4 decode (matches Track 2 SC6 + W14 Step 2 G7 baseline workload)

**Method:** identical to Step 2 with Mistral-7B-shaped GEMMs

**Tag (if run):** `option-2-mistral-ab`
**Tag (if deferred):** `option-2-mistral-deferred-e7`

**~LOC:** ~30 LOC delta to Step 2 runner (workload swap)

### Step 4 — Close-out with branch-specific framing (1 ED, ALWAYS)

**Deliverable:** `WEEK_14_FOLLOWUP_OPTION_2_COMPLETE.md` with branch-A or branch-B framing per §3 decision 3.

**Substrate state at close:** unchanged from `week-13-14-complete` — this campaign is measurement-only; no source touched.

**Tag:** `option-2-complete` aliased to whichever sub-tag was the terminal step

**Bookkeeping:** memory `w14-step-3-followup-mistral-tok-s` upgraded from "queued" to "CLOSED branch [A|B]" with terminal anchors; W15-17 CP 5.5 entry-state confirmed unchanged

---

## 5. Total estimate (revised post-Step-0 + §10 Q1 (a))

| Branch | Step 0 | Step 1 | Step 2 | Step 3 | Step 4 | Total |
|---|---|---|---|---|---|---|
| **Branch B (calibration-gate exit)** | 1-2 ED | 1.5-2.5 ED | — | — | 0.5 ED | **3-5 ED** |
| **Branch C (knee found, Step 2 net ≤ 0)** | 1-2 ED | 1.5-2.5 ED | 1-2 ED | — (skipped — net-tok/s already disproven on smaller workload) | 0.5 ED | **4-7 ED** |
| **Branch A, Mistral deferred (E.7)** | 1-2 ED | 1.5-2.5 ED | 1-2 ED | 0.5 ED (E.7 check only) | 0.5 ED | **4.5-7.5 ED** |
| **Branch A, full run** | 1-2 ED | 1.5-2.5 ED | 1-2 ED | 1 ED | 0.5 ED | **5-8 ED** |

Range **3-8 ED** depending on branch. Step 0 + §10 Q1 (a) overhead microbench add ~2-3 ED vs the original 2-6 ED range. Trade is interpretability and substrate-actually-running confidence — the previous attempt's null measurement is the reason the trade is worth it.

Consistent with advisor sanity-check that the intellectual work was done in W14 Step 2 E; this is execution + measurement + honest writeup. Step 0 is the one piece of new engineering and it is plumbing, not Koopman work.

---

## 6. Five risks

| ID | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R-O2.1 | Branch B confirms — campaign exits at calibration gate | **HIGH** (per W14 Step 2 E prior) | Pre-committed framing (§3 decision 3); reframe is the deliverable not a fallback |
| R-O2.2 | Mistral E.7 env-block still active at Step 3 | **MEDIUM** (carry-forward 3 phases) | Step 3 begins with E.7 check; defer-and-document path locked in |
| R-O2.3 | β calibration on TinyLlama doesn't transfer to Mistral architecture | **MEDIUM** | Step 3 re-calibrates β on Mistral warmup before A/B (~10 min runtime) |
| R-O2.4 | vLLM TinyLlama doesn't enumerate REGISTER_STREAMS (D14 finding #3) blocking substrate engagement | **LOW** | Koopman engine is matmul-dispatch substrate (T4.5.1), not stream-routing; D14 finding affects D3/D5 path not Koopman fire path |
| R-O2.5 | Branch A "positive utility" finding doesn't survive longer soak (overhead dominates) | **MEDIUM-LOW** | Step 2 includes minimum 5-min run, not single-batch microbench; tok/s aggregate over ≥10k tokens; §10 Q1 (a) microbench gives advance break-even floor |
| R-O2.6 | Step 0 worker-init hook does not land cleanly within 2 ED budget (e.g. vLLM V1 worker spawn semantics force a fork-without-libcublas-yet ordering, or InitializeInjection2 symbol isn't externally callable) | **MEDIUM** | HARD BUDGET at 2 ED then surface; fallback path (non-vLLM PyTorch loop) requires re-adjudication, NOT silent substitution |
| R-O2.7 | Step 0 lands but the verification gate's β=0.99 force-fire produces high handled count but garbage outputs (top-1 collapse on real decode at any β) | **LOW** | This is Branch B by another name — surface as Branch B at Step 0 close with calibration skipped, since the gate itself proves the architectural ceiling at β=0.99 |

---

## 7. Honest residue (declared up-front)

1. **The prior says Branch B is expected.** This campaign is structured to confirm the prior efficiently, not to extract a positive result that may not exist. If Branch A appears, it's a surprise relative to W14 Step 2 E; Step 2 + Step 3 then validate it.
2. **TinyLlama-1.1B is not "real customer workload" — it's a small-cost first pass.** Even Branch A on TinyLlama alone is honest about the workload class.
3. **N=4 mirrors W6 Phase A baseline + Track 2 SC6** — known good measurement context, not novel.
4. **No substrate code change across this campaign.** All anchors stay at `week-13-14-complete`. Substrate is fixed; we are measuring its real-workload behavior.
5. **Mistral E.7 carry-forward** — if it's still blocked at Step 3, that is itself a finding (4-phase carry-forward is a queue priority signal for W15-17 CP 5.5 entry checklist).
6. **Goal 5 framing residue surfaced by Step 0** (Anil 2026-05-25): Memory #1's framing "LD_PRELOAD-only deployment, zero customer code changes" is **incomplete as stated** for vLLM V1 deployments. The cipher_vllm_plugin worker-init hook is a deployment-side requirement (a small plugin install, not application/model code changes). The honest post-Step-0 reframe is: **"LD_PRELOAD + drop-in cipher_vllm_plugin, zero application code changes."** Step 0 step doc must capture this; pitch-language reconciliation deferred to post-Branch resolution.
7. **Original `bench_llm.py` smoke result** that motivated week6-bench-harness ("CIPHER neutral vs vanilla") was likely captured with the same vLLM V1 worker-subprocess null-substrate condition. Step 0 verification gate retroactively informs how to interpret that historical neutral result; out-of-scope for option-2 but flagged for W15-17 entry checklist.

---

## 8. Pre-commit framing (per §3 decision 3)

**Branch A outcome statement template:**
> v1.2.3 Goal 4 (O(1) compute substitution): on TinyLlama-1.1B vLLM N=4 decode, Koopman substrate at β=[X] delivers [Y]% tok/s lift at [Z]% top-1 vs vanilla cuBLAS over [N] tokens. [Mistral statement: shipped / pending E.7 / not measured]. This is one workload class; generalization to wider model families is W15-17 CP 5.5 scope.

**Branch B outcome statement template:**
> v1.2.3 Goal 4 (narrow-domain Koopman substrate): substrate ships correct and validated end-to-end (W13-14 LM-head harness PASS). On real TinyLlama-1.1B vLLM N=4 decode the architectural ceiling identified at W14 Step 2 E reproduces — Koopman either does not fire on real-decode residuals at any β with positive utility, or fires only at top-1 degraded below the 0.90 contract. Production firing is deferred to v2 pending [specific finding]. The substrate-deliverable contract from W14 Step 2 G CLOSURE remains intact.

**Branch C outcome statement template:**
> v1.2.3 Goal 4 (narrow-domain Koopman substrate with calibrated real-workload fire path): substrate fires correctly on real TinyLlama-1.1B vLLM N=4 decode at calibrated β=[X] (substitution rate [Y]%, top-1 [Z] ≥ 0.90 per fired GEMM). Engine-path overhead (per W13-14 substrate-overhead instrumentation [or W14 followup overhead microbench if §10 Q1 enabled]) dominates the saving at this workload; net tok/s delta is [-W]% at N=4 over [N] tokens. Production net-positive deployment is deferred to v2 pending engine-overhead reduction work, scoped at W15-17 CP 5.5 or later. Substrate fire-path correctness on real workloads is a new engineering result not present at W13-14 close — the substrate-deliverable contract advances; the productization claim does not yet.

---

## 9. No HARD STOP triggered

Scope-lock ready. Next implementation prompt (on user approval) is Step 1 calibration gate on TinyLlama warmup.

**v1.2.3 §7 W15-17 entry-state:** unchanged regardless of branch. Substrate at `week-13-14-complete`. CP 5.5 hybrid heterogeneous-model benchmark scope intact.

---

## 10. Adjudication record

**Q1 (CLOSED 2026-05-25, Option (a)):** Step 1 includes a substrate-overhead microbench. Anil amendment: *"On §10 Q1 — Option (a) confirmed. Add substrate-overhead microbench to Step 1. The +0.5 ED buys Branch C interpretability."* Microbench specification folded into §4 Step 1 method.

**Amendment record:** Step 0 added per Anil 2026-05-25 as hard prerequisite — vLLM V1 worker-subprocess GOT patch hook. Estimates revised in §5. Honest residue #6 (Goal 5 framing) and #7 (bench_llm.py historical neutral result reinterpretation) added.

No open questions at scope-lock landing. On commit, Step 0 begins.

---

**Related memory:**
- [[w13-14-complete]] — predecessor; substrate baseline
- [[w14-step-3-followup-mistral-tok-s]] — queued-campaign pointer (this is its implementation)
- [[w14-step-2-koopman-tier]] + [[h2-cgs2-fix]] — substrate fixes ceiling came from
- [[cipher-t43-envelope]] — prior pattern: substrate validates on test workload, doesn't generalize to 7B+ regime
- [[cipher-lift-framing]] — aggregate-lift framing precedent
- [[week6-bench-harness]] — `bench_llm.py` infra used in Steps 1-3
- [[scope-drift-audit]] — D10 LM head 7.43× already de-gated; this campaign honors that de-gating
