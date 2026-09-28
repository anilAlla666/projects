# CIPHER Measurement Methodology — Industry Alignment (pre-Step-4+5)

**Date:** 2026-05-18. **Type:** paperwork, no GPU. **Purpose:** align CIPHER's
measurement methodology with industry serving/power-benchmark practice before
the merged Step 4+5 multi-tenant continuous-batching build, so methodology
gaps are caught upfront — the CP 5.6 P2 retraction happened because they were
not.

## Provenance note (read first)

The four external standards — **TokenPowerBench (AAAI 2026)**, **MLPerf Power
(2025)**, **ML.Energy**, the **"1/W Law" paper** — were **verified 2026-05-18**
via web search against the primary sources (`CITATION_VERIFICATION.md`). All
four are real publications at the claimed venues (TokenPowerBench AAAI 2026 /
arXiv 2512.03024; MLPerf Power arXiv 2410.12032, MLPerf Inference v5.x 2025;
ML.Energy arXiv 2505.06371; 1/W Law arXiv 2603.17280). None was removed; three
minor characterization corrections (TokenPowerBench power granularity;
ML.Energy dataset naming; the 1/W law's figure and thesis) are applied in
Section 2 below.

What does **not** depend on those citations, and is asserted with confidence:
Sections 1 (CIPHER's actual methodology — audited from this campaign's
artefacts), 3, and 4 (the adoption plan and Step-4+5 spec) rest on standard
LLM-serving-benchmark practice — Poisson arrivals, length distributions,
sustained-load windows, TTFT/ITL latency, vLLM/SGLang-style continuous-batch
evaluation — which is stable, cutoff-independent engineering practice.

---

## SECTION 1 — Current CIPHER methodology audit

### 1.1 TPW pillar (Session 2 Steps 1–3, Mistral verification)

- **Workload:** free-running greedy decode (`do_sample=False`), fixed
  128-token budget, **5 fixed synthetic prompts** (factual / narrative / code
  / reasoning / conversational).
- **Power:** NVML GPU **board** power (`nvidia-smi --query-gpu=power.draw`),
  sampled ~0.5 s (~2 Hz). GPU only — excludes CPU, DRAM, PSU loss, node.
- **Power aggregation:** arm-level **mean** over the decode window (sentinel
  `DECODE_START/END`); `tok/W = aggregate tok/s ÷ arm-level mean power`.
- **Correctness gate:** **teacher-forced** (cascade-free) — gold prefix fed at
  every position; per-step logit-KL(gold‖substrate) ≤ 0.1 (and/or top-1
  agreement ≥ 99 %). KL threshold calibrated to honest INT4 drift (~0.07–0.10
  observed for Marlin INT4). This is the methodology that replaced the broken
  free-running gate (CP 5.6 P2, and again the Step-4 N=16 mid-session fix).
- **Substrate-attributable** = batched-executor tok/W ÷ naive-N-concurrent
  tok/W (N independent processes, no fusion).
- **Duration:** ~12 s burst per arm (5 prompts × ~2.4 s).

### 1.2 MFU pillar (`pillar_driver.py`)

- **Achieved FLOPs:** analytical `2 · N_params · tokens` per forward pass
  (`6·N·tokens` would apply to training). `N_params` from the loaded model.
- **Denominator:** H100 SXM5 dense FP16/BF16 tensor peak **989.4 TFLOP/s**
  (no sparsity).
- **MFU** = achieved FLOPs/s ÷ peak.
- **Known omissions (honest):** the `2·N·tokens` term counts only the
  weight-matmul FLOPs. It **omits attention FLOPs** (Q·Kᵀ and softmax·V),
  which are sequence-length-dependent — negligible for B=1 short decode
  (< few %), but materially understated for long-context / prefill regimes.
  It also assumes dense FP16; an INT4 (Marlin) arm does fewer effective FLOPs
  per byte, so MFU across arms is not strictly like-for-like.

### 1.3 Multi-tenant pillar (Phase B prototype)

- **Architecture:** N tenant **client** processes + one batched-decode
  **executor** (Form A — executor holds the model, clients are request
  streams over a Unix socket).
- **Scheduling:** static barrier (Steps 1–3) or continuous-batch with
  **eviction** (Step 4) — admission not yet built.
- **Same-model only** — all tenants run the identical model (that is what
  fuses). Heterogeneous-model multi-tenancy untested.
- **Substrate-attributable** vs **naive-N-concurrent** (N independent
  processes, each its own weight copy, no fusion) on the same H100.
- **Prompts:** identical 5 prompts replicated across tenants (Steps 1–3); per-
  tenant `gen_len` for the Step-4 heterogeneous test.

---

## SECTION 2 — Industry-standard comparison (per the brief; see provenance note)

### TokenPowerBench (AAAI 2026)
| axis | CIPHER | status |
|---|---|---|
| phase-aware prefill/decode separation | decode-only, no separation | **DIVERGE** |
| vendor telemetry, no external meters | NVML telemetry | ALIGN |
| multi-granularity power (GPU/node/system) | GPU board only | **DIVERGE** |
| service-unit normalization (J/token) | tok/W = inverse of J/token | ALIGN (equivalent) |

### MLPerf Power (2025)
| axis | CIPHER | status |
|---|---|---|
| wall-plug power, certified meter | NVML GPU board only — misses CPU/DRAM/PSU loss | **DIVERGE** |
| sustained-load measurement | ~12 s burst | **DIVERGE** |
| per-scenario QPS targets | no QPS enforced | **DIVERGE** |
| accuracy gate | teacher-forced KL ≤ 0.1 — *stricter* than a 99.9%-FP32-reference gate for quantized models | ALIGN (stricter) |

### ML.Energy benchmark
| axis | CIPHER | status |
|---|---|---|
| real traffic patterns (realistic service datasets) | 5 fixed synthetic prompts | **DIVERGE** |
| verbosity-aware output-length distribution | fixed 128-token output | **DIVERGE** |
| multiple batch sizes | tenant-level B=1 only (batched at the executor) | PARTIAL |

### "1/W Law" paper (arXiv 2603.17280)
| axis | CIPHER | status |
|---|---|---|
| context-length sensitivity — tok/W ≈ halves per context-window doubling (~12× across 4K→64K) | single, short context length | **DIVERGE** |
| routing topology as the dominant energy lever | not modelled | **DIVERGE** |
| (corroboration) tok/W set by concurrent-sequences-in-flight | CIPHER's Phase B batch-scan measured the same relationship directly | **ALIGN** |

### Multi-tenant serving practice (vLLM / SGLang / TGI / LMSYS)
| axis | CIPHER | status |
|---|---|---|
| Poisson arrival at rate λ | synchronized burst / static barrier | **DIVERGE** |
| ShareGPT-style sequence-length distribution | fixed length | **DIVERGE** |
| TTFT + ITL latency tracking | none (throughput/power only) | **DIVERGE** |
| QoS-threshold throughput | not measured | **DIVERGE** |
| **baseline = a real batching server (vLLM)** | baseline = **naive N-concurrent** (no batching at all) | **DIVERGE — and flagged below** |

**The most consequential divergence — the baseline.** CIPHER's
substrate-attributable number uses **naive-N-concurrent** as the denominator.
That measures "does batching help versus no batching" — and the answer is a
large yes (3.69× at N=8 TinyLlama, 3.26× at N=4 Mistral). But naive-concurrent
is a **weak baseline**: it is *no batching at all*, and it degrades sharply on
big models (naive 4-concurrent Mistral-7B: 85.9 tok/s at 325 W — the Mistral
3.26× is partly that poor denominator). The honest comparison is **not** "CIPHER
tok/W vs vLLM tok/W" — vLLM is a throughput/latency framework and does not
claim tok/W as a primary metric; a metric race against it is
apples-to-oranges. The right axis is **architectural**: vLLM fuses requests
*within* one serving process; CIPHER fuses *across* separate tenant processes,
which vLLM structurally cannot. Section 3's 3-arm benchmark measures that
distinction directly. Until then, the substrate-attributable number is honest
only as "batching vs no-batching" and must be labelled that way — never as
"CIPHER beats the field."

---

## SECTION 3 — Adoption plan for the Step 4+5 build

### CRITICAL — adopt before Step 4+5 builds
- **Poisson arrival** at rate λ (open-loop) — replaces the synchronized
  barrier; required for any honest multi-tenant latency claim.
- **Sequence-length distribution** — per-request `gen_len` drawn from a
  distribution, not fixed. Step-4+5 uses **uniform integer [32, 256]** as the
  reproducible proxy (it spans the heterogeneity range that exposed the 57.7 %
  drain); a ShareGPT-style **bimodal** distribution is the more realistic
  follow-up (FUTURE).
- **Per-tenant QoS latency** — TTFT (time-to-first-token), ITL (inter-token
  latency), per-tenant p50/p95/p99, measured **from request submission**.
- **Substrate-attributable baseline** — explicitly: naive N-concurrent
  processes, *no CIPHER fusion*, under the **identical** arrival pattern and
  identical prompt/length sequences as the CIPHER arm.
- **Correctness gate** — per-tenant teacher-forced KL ≤ 0.1, unchanged. Solid.

### IMPORTANT — adopt as part of the Step 4+5 build
- **Sustained-load measurement** — ≥ 60 s steady-state after a warm-up
  boundary, not a 12 s burst. (Requires enough offered load to *stay* loaded
  for the full window — see the Section-4 fix.)
- **Workload mix** — measure **same-model** multi-tenant (CIPHER fuses) **and
  different-model** multi-tenant (CIPHER cannot fuse — falls back to naive).
  Report both, so scope is honest.
- **Multiple operating points** — B=1 decode (current), B≥8 batched decode,
  and a prefill-heavy regime.

### BEFORE EXTERNAL / PEAK XV CLAIMS — the 3-arm cross-process benchmark

Not required for CP 5.6 **internal** closure — Arm 1 vs Arm 3 (naive vs
CIPHER) suffices there. Required before any external / Peak XV claim. It does
**not** race tok/W against vLLM — vLLM is a throughput/latency framework and
makes no primary tok/W claim; that race is apples-to-oranges. It measures the
**architectural distinction**: vLLM fuses requests *within* one serving
process; CIPHER fuses *across* separate tenant processes, which vLLM cannot.

Scenario — N separate Python processes, each loading its own model, B=1
decode; identical hardware, prompts, arrival pattern across arms:
- **Arm 1 — naive:** each process runs independently, no fusion. (CP 5.6's
  current substrate-attributable baseline.)
- **Arm 2 — vLLM intra-process:** each process runs vLLM internally — vLLM
  continuous batching engages *within* a process, never *across* processes.
  Note: with one request stream per tenant process, vLLM has nothing to batch
  intra-process, so Arm 2 is expected ≈ Arm 1 for this B=1-per-process
  topology — and that is precisely the point: it shows, by measurement, where
  vLLM structurally cannot help.
- **Arm 3 — CIPHER substrate:** processes independent from the tenant's view;
  CIPHER's substrate fuses decode steps across them transparently.

Metrics: aggregate tok/s, aggregate tok/W, per-tenant TTFT + ITL, per-tenant
teacher-forced correctness gate. The number that justifies the substrate-layer
integration cost over a vLLM-only alternative is **Arm 3 vs Arm 2** — CIPHER's
lift on the workload where the architectural distinction is the whole point
(N separate customer processes that vLLM cannot fuse across).

### FUTURE — post-CP-5.6, for full industry-benchmark parity
- Wall-plug power (Yokogawa-class certified meter).
- Phase-aware prefill/decode power separation.
- Multi-granularity power (CPU / DRAM / GPU separately).
- Real ShareGPT prompt replay (vs synthetic prompts).

---

## SECTION 4 — Recommended methodology specification for Step 4+5

### Brief inconsistency, fixed first

The brief's STREAM scenario says "16 requests, Poisson over a 30 s window"
then "measure 60 s steady-state after a 30 s warm-up." **16 requests cannot
sustain a 30 s + 60 s = 90 s window** — at ~3–6 s of decode each, 16 requests
drain in well under a minute. Two clean fixes; Step 4+5 uses **both**, for the
two things they each measure correctly:

- **Open-loop Poisson stream (latency)** — a *continuous* arrival process for
  the full 90 s, λ tuned so the steady-state in-flight count ≈ 16. With mean
  service ≈ 5 s, λ ≈ 16 / 5 ≈ **3.2 req/s** → ≈ 290 requests over 90 s. This
  measures TTFT/ITL latency under realistic load.
- **Closed-loop N-in-flight (throughput/power)** — exactly N tenants, each
  resubmitting a new request the instant its previous one completes, holding N
  in flight for the whole window. This measures steady-state aggregate tok/W
  cleanly (no arrival-rate confound).

### STREAM scenario (sustained, primary)

- **Load:** open-loop Poisson, λ ≈ 3.2 req/s, steady-state in-flight ≈ 16.
- **Per-request `gen_len`:** uniform integer in **[32, 256]**.
- **Prompts:** the 5-prompt set, cycled (FUTURE: ShareGPT replay).
- **Window:** 30 s warm-up, then **60 s steady-state** measurement.
- **Measure:** aggregate tok/W; aggregate throughput tok/s; per-tenant TTFT
  p50/p95/p99; per-tenant ITL p50/p95/p99.
- **Correctness:** per-request teacher-forced KL ≤ 0.1.
- **Baseline:** identical Poisson stream + identical prompts/`gen_len`
  sequence on **naive concurrent processes** (no fusion). Report
  substrate-attributable = CIPHER ÷ naive.
- **Target:** aggregate tok/W within 10 % of the Session-2 Step-2 static
  barrier N=8 result (3.342 tok/W); substrate-attributable ≥ 3× at the
  steady-state in-flight count; per-tenant p99 latency ≤ 1.5× single-tenant
  baseline.

### BURST scenario (heterogeneous, drain test)

- **Load:** 8 tenant requests arriving simultaneously.
- **Per-request `gen_len`:** heterogeneous, mixed {32, 64, 128, 256} (2 each).
- **Correctness:** per-tenant teacher-forced KL ≤ 0.1.
- **Measure:** aggregate tok/W; per-tenant completion latency; **drain cost
  vs an equal-length baseline** (the Step-4 metric — must now be < 15 %, the
  whole point of adding admission).
- **Baseline:** identical burst + `gen_len` set on naive concurrent processes.

### Critical methodology guards (apply to both scenarios)

1. **Both arms, identical inputs.** CIPHER and naive measured on the *same*
   arrival pattern and the *same* prompt/`gen_len` sequences (seed-fixed).
2. **Honest aggregate tok/W** = (total **useful** tokens) ÷ (mean full-GPU
   power) ÷ (wall time). Useful = tokens that pass the correctness gate;
   **exclude** padding tokens and any failed-correctness tokens.
3. **Latency from submission.** TTFT/ITL measured from request *submission*,
   not from batch fusion — the queue/barrier wait is part of CIPHER's real
   tradeoff and must be charged to it.
4. **Variance.** Multi-tenant has more run-to-run variance than single-tenant
   — run **each arm 3×**, report **mean and range**.
5. **Power window discipline.** Power averaged over the 60 s steady-state
   window only (STREAM) / the burst decode window (BURST); warm-up excluded.
6. **Anchor.** `a7ac8e97` unchanged unless the executor moves into
   `libcipher_rt`; rotate + preserve `.pre_*` backup per discipline if so.

### Honesty framing for whatever Step 4+5 produces

The Step-4+5 number will be **multi-tenant aggregate efficiency from
cross-tenant batching, vs naive concurrency** — a different claim from the
retracted CP 2.4 single-tenant composed 3.617×, and from CP 5.6 P2's honest
single-tenant 1.54×. It must be reported as such. It is **not** characterised
against vLLM until the 3-arm cross-process benchmark (§3) is run. CP 5.6 may
close on the naive comparison (Arm 1 vs Arm 3) for *internal* engineering
sign-off; any *external* statement needs the 3-arm benchmark first — framed as
the architectural distinction (cross-process fusion), not a tok/W metric race.

---

## Recommendation

Adopt all **CRITICAL** items into the Step-4+5 harness before it runs (they
are harness-design decisions, cheap to bake in now, expensive to retrofit).
Adopt the **IMPORTANT** items within the Step-4+5 build. CP 5.6 closes on the
naive baseline (Arm 1 vs Arm 3). The **3-arm cross-process benchmark** is a
hard gate before any external / Peak XV claim — and is framed as the
architectural distinction (cross-process fusion vLLM cannot do), not a
tok/W-vs-tok/W metric race. Defer **FUTURE** items. Adjudicate which adoptions
apply, then greenlight Step 4+5 with this methodology baked in.
