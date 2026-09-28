# V.0 weighted-mix harness — SCOPE + DESIGN MEMO

**Date:** 2026-05-30. **Type:** READ-ONLY scope discovery + design memo. **No build, no commit,
anchors unchanged.** STOP for Anil's approval (design → approve → build, same as the Marlin GC step).
Every claim cited file:line, not memory. V.0 = punch-list item 2 (the NET-NEW scaffold every CP-5.5
goal-gate runs on). Predecessor: Marlin GC-on-free **staging-closed** (`marlin-gc-on-free-staging`, .so
`944f5706`) — it surfaced the V.0 evict-wiring requirement, which this memo scopes **conditionally** (§3).

**Headline:** V.0 builds the 100-worker weighted-mix orchestrator on `bench_llm.py` + a per-gate
**engagement map on stock vLLM**. The dominant, cited risk is **not** harness mechanics — it is that
**vLLM's prefill GEMMs bypass CIPHER's public-cuBLAS intercept via libcublas-internal `nvjet`**
(`D9_CIPHER_DELIVERY_REPORT.md:24-42`), so the *GEMM-substitution* levers (Marlin/FP8/Koopman) see 0 calls
on vLLM-prefill while the *non-GEMM* levers (VOLT clock, POOL batching, KV-dedup plugin, CUPTI MFU
measurement) engage. The memo's spine is mapping which of the 5 goals' providers engage on stock vLLM —
and surfacing the **product-scope question** that decision raises to Anil (§3, §4).

---

## PART 1 — SCOPE (cited)

### 1.1 The V.0/CP-5.5 spec — two mix framings + the 5 goal-gates
- **Workload mix — two documented decompositions (V.0 should be mix-PARAMETERIZED to run either):**
  - `CIPHER_REENGINEERING_PLAN.md:1623`: *"5 prefill + 80 decode + 15 burst (per the v1.2.2 mix)"* +
    *"≥5 different model families coexisting: Mistral-7B + Qwen-7B + Llama-3-8B + 2 SLMs"*.
  - `V1_CP55_SOAK_MEMO.md:39`: *"30 agent-inference + 30 continuous-batched-serving + 20 single-tenant
    streaming + 10 batch-inference + 10 RAG"* (audit `:521`); mixed dtypes bf16/fp16/AWQ-INT4; models
    Llama-3-8B/Mistral-7B/TinyLlama/Qwen-2.5. On 1 H100 these are a **representative weighted soak**
    (time-share / right-size to 80 GB, **logged not silently capped** — `V1_CP55_SOAK_MEMO.md:42-46`).
  - **(Correction:** my earlier readiness-audit line "5 prefill + 80 decode + 15 burst per
    V1_CP55_SOAK_MEMO.md:39" conflated the two — `:39` is the 30/30/20/10/10 class mix; the prefill/
    decode/burst framing is `PLAN.md:1623`. Both are real; V.0 takes the mix as a parameter.)
- **The 5 goal-gates** (`V1_CP55_SOAK_MEMO.md:26-30`), each with its **provider** (`:24` column) — this
  provider is what determines engagement under nvjet (§1.4):
  | Goal | Interim gate | Provider (`:24`) |
  |---|---|---|
  | 1 density | ≥10 agents/H100, **KL=0** | W.4 POOL batching + vLLM KV (CP 5.1) |
  | 2 tok/W | ≥1.5× weighted vs vanilla (post-v1 ≥2×) | **VOLT** auto-activate (clock) |
  | 3 MFU | ≥70% weighted-mean (post-v1 85%, **GEMM-lift DE-PRIORITIZED**) | CUPTI MFU + actuators |
  | 4 Koopman | handled>0 on long-context prefill, KL-preserved | W.3 Koopman (actuator) |
  | 5 auto-profile | profile + caps <30s of `vllm serve`, no env | K.1 classifier + cipher-platform |

### 1.2 What exists to build on — `bench_llm.py` (single-tenant)
`week6/bench_v2/bench_llm.py` (906 LOC) is a **single-instance** MLPerf-v5.1 LLM-server bench:
- `StreamingRunner` drives **vLLM `AsyncLLMEngine`** (`:474-485,501`) — in-process async serving, per-token
  async streaming traces.
- `TelemetrySampler` 100 Hz NVML + trapezoidal energy → tok/W, tok/J (`:127-237,186-213`); `HardwareLock`
  clock-lock/verify (`:58-121`); `WorkloadGenerator` agentic/sharegpt (`:303-353`); `aggregate_iteration`
  phase-split prefill/decode **MFU** vs 989 TF + p99 tails (`:539-611`); `bootstrap_ci_of_delta` CIPHER
  vs vanilla CI (`:643`).
- **It is single-tenant / sequential-batch** (`:812-857` one runner, sequential iterations). **No
  `bench_llm_multi.py` exists** (find: absent — the multi-tenant orchestrators in `cp_5_4/step1_6/` are
  SM-partition research, not weighted-mix LLM bench). **GAP to V.0:** (a) 1→N concurrent right-sized
  instances; (b) the weighted mix (prefill/decode/burst or the 5 classes); (c) ≥5 families coexisting;
  (d) cross-worker **weighted-aggregate** gate computation; (e) the density/Koopman/auto-profile gates
  (absent today); (f) the evict-wiring (§3).

### 1.3 The `vllm serve` carry — NO real serve harness; and the nvjet BYPASS
- **No real `vllm serve` / `openai.api_server` e2e harness exists** anywhere in-tree — all harnesses use
  offline `AsyncLLMEngine` or `LLM()` (workflow scope-sweep; `D9_CIPHER_DELIVERY_REPORT.md` confirms the
  container-vLLM path). (My earlier "`D9_BUILD_REPORT.md:13-18` NOT-FOUND" cite was wrong — those lines
  are FP8 measurement; the real carry is the absent serve harness + the bypass below.)
- **⛔ BINDING FINDING (`D9_CIPHER_DELIVERY_REPORT.md:24-42`):** with CIPHER active under container vLLM,
  a real 8×2048 vLLM **prefill** intercepted **ZERO GEMMs** — *"MATMUL (cublasGemmEx): calls=0 …
  classifier obs=0 … vLLM never calls the public cuBLAS symbols CIPHER patches"*; *"vLLM/torch on cu13
  dispatch prefill GEMMs through libcublas-internal `nvjet`, not the public cublasGemmEx/cublasLtMatmul"*
  (the D.10-abandoned intercept depth). **CIPHER's CUPTI path observes but cannot substitute.** ⇒ **the
  GEMM-substitution actuators (Marlin, FP8, Koopman) do not engage on vLLM-prefill.**
  - **NOT universal:** the Marlin churn test (this campaign) proved **torch HF *decode* GEMMs DO hit
    public `cublasGemmEx`** (Marlin engaged, `sub>0`). So engagement is path+shape-specific: torch
    decode-path → public cuBLAS (engages); vLLM big-prefill → nvjet (bypass). The W.4 POOL executor
    (`formb6_executor.py:56,126`) issues its batched decode via **torch `AutoModelForCausalLM` forward**
    (public-cuBLAS-engageable) — **but runs VANILLA today** (`formb6_executor.py:11` *"Executor is VANILLA
    (no CUDA_INJECTION64_PATH → zero shim tax)"*).

### 1.4 The runtime V.0 drives — CUDA_INJECTION64_PATH
- **`CUDA_INJECTION64_PATH` is the confirmed working CIPHER-injection method** for vLLM/torch processes
  (`v1_phase_b/k1_close_gate/run_close_gate.sh:8,18`, `run_multitenant_w6sc.sh:16,23-26`). LD_PRELOAD is
  **not "dead" but scope-modified** — Phase A restores an LD_PRELOAD-only Goal-5 path via a worker-init
  GOT hook (`V1_PHASE_A_SCOPE_LOCK.md:21-27`); for V.0's measurement, CUDA_INJECTION64_PATH per serve/
  worker process is the method.
- **Which .so:** to exercise the staging substrate (incl. the Marlin GC fix), V.0 injects the **staging
  `944f5706`** (or the repackaged `.deb` once the infra blocker clears — `V1_CP55_READINESS_FINDINGS.md`
  §1/§4: deployed `.deb` lags at `1f305ce6`; repackage to the validated anchor is the sudo/infra step).

---

## PART 2 — THE MULTI-GPU BOUNDARY (what V.0 can validate now)

- **Single-GPU measurable now** (interim gates, all serving/inference-class — `V1_CP55_SOAK_MEMO.md:35`;
  `V1_CP55_READINESS_FINDINGS.md:39-44`): density ≥10 agents KL=0; weighted-mix tok/W ≥1.5×; weighted-mean
  MFU ≥70%; Koopman handled>0; auto-profile <30s. Plus KV-dedup (W5), per-agent p99 (`PLAN.md:100`),
  cross-tenant tok/W 3.06-5.98× (`PLAN.md:102`), May-13 7-problem repro, crypto-billing (single-GPU-
  checkable).
- **Requires 8+ GPUs (v2-deferred):** the **85% MFU multi-GPU composition** (`PLAN.md:101,1019` — post-
  CP-5.5 depth-win) + **NCCL 3-ops** (NCCL_P2P/OVERLAP/STRAGGLER cross-rank, `PLAN.md:1716-1724`).
- **⇒ V.0 is the single-GPU foundation, NOT the multi-GPU capstone.** It builds the harness + measures the
  single-GPU interim gates; the 85%-MFU + NCCL-overlap slice waits for the cluster (per Anil's decision).

---

## PART 3 — DESIGN MEMO (design only; build is a separate approved step)

### 3.0 The spine: a per-gate engagement map on stock vLLM (the V.0 pre-flight)
Given nvjet (§1.3), V.0's **first deliverable is the SOAK §1 readiness pre-flight, sharpened**: for each
goal, does its provider ENGAGE on stock vLLM, or bypass? My cited projection (V.0 confirms empirically):

| Goal | Provider | Engages on stock vLLM? (projected) |
|---|---|---|
| 1 density | POOL batching (CIPHER's own torch executor) + KV-dedup plugin | **LIKELY** — batching is CIPHER's component (not GEMM-substitution); KV-dedup plugin loads under container vLLM (`D9_CIPHER_DELIVERY_REPORT.md:20-22`). Confirm POOL admits on stock serve. |
| 2 tok/W | VOLT (clock, NVML/kmod) | **LIKELY** — clock control is independent of the GEMM intercept. |
| 3 MFU | CUPTI measurement + GEMM actuators | **SPLIT** — CUPTI *measures* vLLM MFU (works); the GEMM-substitution *lift* (Marlin/FP8) **bypasses on vLLM-prefill (nvjet)**. The 70% interim is vLLM-native + non-GEMM levers; 85% GEMM-lift is already DE-PRIORITIZED (`V1_CP55_SOAK_MEMO.md:28`). |
| 4 Koopman | Koopman actuator | **AT RISK** — actuator on the GEMM/kernel intercept → bypasses on vLLM-prefill (the genuine nvjet casualty). May engage on a torch-decode path. |
| 5 auto-profile | K.1 classifier (intercept-fed) | **AT RISK** — classifier `obs=0` on the vLLM-prefill bypass test (`D9_CIPHER_DELIVERY_REPORT.md:28`); confirm whether the CUPTI-observation path profiles without the cuBLAS intercept. |

**This map IS V.0's core contribution.** Goals 1/2 (VOLT + POOL/KV, non-GEMM) likely survive; Goal 4 +
the actuator-half of Goal 3 are the bypass casualties. V.0's pre-flight resolves each empirically on a
stock serve and **surfaces non-engagement as a finding** (not hand-forced — `V1_CP55_SOAK_MEMO.md:33,55`).

### 3.1 The 100-worker weighted-mix orchestrator (on `bench_llm.py`)
A CIPHER-owned **controller** (test harness, not a framework patch) that:
1. **Spawns the weighted mix** of right-sized model instances across the ≥5 families, mix-parameterized to
   run either the `PLAN.md:1623` (5/80/15) or `SOAK:39` (30/30/20/10/10) decomposition; logs every
   right-sizing/time-share decision (`V1_CP55_SOAK_MEMO.md:46` honesty).
2. **Reuses `bench_llm.py`'s primitives** — `StreamingRunner`/`AsyncLLMEngine` driver, `TelemetrySampler`
   (tok/W), `aggregate_iteration` (MFU/p99), `bootstrap_ci_of_delta` (vs a vanilla `vllm serve` baseline on
   the identical mix) — lifting them from single-instance to a **cross-worker weighted aggregator** that
   computes the per-class and **weighted-mean** gate (the weighting matches the mix proportions so tok/W &
   MFU are the *weighted* gate, `V1_CP55_SOAK_MEMO.md:44`).
3. **Measures each gate** per §3.0: density (≥10 concurrent agents, KL=0 vs solo refs — the
   Marlin-churn-test KL methodology), weighted tok/W (VOLT-on vs vanilla), weighted-mean MFU (CUPTI),
   Koopman handled-count, auto-profile latency (<30s of serve start).
4. **Drives stock `vllm serve` per the zero-intervention contract** (`V1_CP55_SOAK_MEMO.md:19,83`, Mem
   #19) — CUDA_INJECTION64_PATH per worker; no LD_PRELOAD gymnastics, no per-workload hand-tuning.
   *(Open: bench_llm uses in-process `AsyncLLMEngine`; the spec says `vllm serve`. V.0 should drive the
   real serve path (HTTP or AsyncLLMEngine-as-stock) so the auto-activation attestation is genuine — a net-
   new serve driver, the absent-harness carry §1.3.)*

### 3.2 THE EVICT-WIRING — load-bearing CONDITIONALLY (the honest scoping)
The Marlin GC `evict_weight()`/`reclaim_retired()` (`cipher_rt_marlin.h:36-37`) bound the fp16-surrogate
memory under churn — **but only where Marlin ENGAGES**, i.e. a **torch-decode path on public cuBLAS** (the
churn test). Therefore:
- **IF V.0's cross-tenant path engages Marlin** — i.e. CIPHER is injected into the POOL/torch-decode
  executor (`formb6_executor.py`, today VANILLA) so Marlin substitutes on the batched decode — **then
  evict-wiring IS load-bearing**: under the 100-agent time-share, model-free must call evict+reclaim or the
  surrogate leaks (the UNBOUND-churn OOM). The user's premise holds, **via the POOL path**.
- **IF V.0's path is stock-vLLM-prefill** — Marlin bypasses (nvjet) → no kits → **evict is moot there**.
- **So the wiring is conditional on the topology**, which is itself **a product-scope question for Anil**
  (§4): is v1's demonstrable cross-tenant value the **non-GEMM** levers (VOLT tok/W + POOL/KV density) on
  stock vLLM, or do we inject CIPHER into the torch-decode executor to stack Marlin (where evict matters)?

**The wiring mechanism (when load-bearing), CIPHER-side only:**
- **No CIPHER-side residence layer tracks model load/free today** (`MARLIN_GC_ON_FREE_BUILD_REPORT.md:144-147`;
  `cipher_rt_coresidence.c` is cohort-discovery only; evict has no external caller,
  `cipher_rt_marlin_engine.cpp:1628`). So V.0 adds the model-free hook in the **CIPHER-owned controller**:
  on time-share model-free, it calls evict + reclaim. Since the controller owns the model object (it loads
  it via HF/torch in the POOL executor), it enumerates weight `data_ptr`s and calls `evict_weight(ptr)` —
  exactly as the churn harness `evict(m)` does — **no vLLM source patch**.
- **Recommended small substrate addition (a separate backfill step, flagged not built):** an
  `evict_model(model_id)` engine API using CIPHER's existing `bind_model` registry (`g_wptr_model`), so the
  controller evicts by `model_id` without enumerating ptrs. This is what makes evict-on-free clean **if**
  the model is owned by vLLM (ptrs not directly accessible) — avoiding any reach into vLLM internals. If
  evict ever required touching vLLM's `model_runner.model` to get ptrs, that is a **Mem #24 flag** (not a
  relaxation Anil can grant inside V.0) — the `evict_model(model_id)` path sidesteps it.
- **SYNC-BEFORE-RECLAIM contract (satisfied by construction):** `reclaim_retired()` must be called at a
  quiescent point — it waits `borrow==0` + `cudaDeviceSynchronize` (`cipher_rt_marlin.h:28-37`,
  `cipher_rt_marlin_engine.cpp:1545-1577`). The controller frees a model **only when it has stopped serving
  it** (drain → quiesce → evict → reclaim before loading the next time-share model) → no concurrent
  dispatch on that model's kits → `borrow==0` naturally → reclaim safe. V.0 states this ordering explicitly.

### 3.3 SUBSTRATE-LINE CHECK (Mem #24) — CONFIRMED, with one flag
V.0 is a **harness** (test orchestrator) + a CIPHER-owned controller; it drives the runtime via
`CUDA_INJECTION64_PATH` and calls CIPHER's public engine API. It does **NOT** modify vLLM source, add
framework hooks, or patch the scheduler. The evict-wiring is at the CIPHER-owned controller / a CIPHER
serve shim, calling `evict_weight`/`evict_model`. **FLAG:** if (and only if) the engaged topology requires
reading vLLM-internal model params to get weight ptrs, that couples to vLLM internals (Mem #24 grey) — the
`evict_model(model_id)` substrate addition (§3.2) is the clean avoidance; surfaced, not assumed-relaxed.

### 3.4 MARVEL STANDARD (Mem #21) — coverage, negative-case, eng-debt, confidence
- **Full coverage:** all 5 goal-gates + the full weighted mix (not a 2-tenant smoke) + the per-gate
  engagement map. The 100-on-1-H100 right-sizing is logged, not silently capped (`:46`).
- **Negative-case / close gate the V.0 harness must assert:** per-agent **KL=0** on density (Mem #11 HARD
  STOP); fairness across agents; **no-leak under churn** (the evict path, when engaged — flat GPU mem across
  the soak, as the Marlin soak gated); no-crash; and — the V.0-specific one — **each gate's engagement is
  measured, non-engagement surfaced as a finding** (a goal that can't auto-activate is reported PARTIAL,
  not faked, `V1_CP55_SOAK_MEMO.md:33`).
- **Eng-debt forecast:** (1) the real `vllm serve` (HTTP) driver is net-new (absent-harness carry); (2)
  `evict_model(model_id)` substrate addition if the POOL-inject topology is chosen; (3) the nvjet
  bypass caps GEMM-substitution gates on vLLM — a deeper (nvjet) intercept is D.10-abandoned, out of v1
  scope; (4) the 100-different-model substrate caps (CP54 64→128 G1, WA 16→100 G2, KV-dedup keying G3,
  Marlin tenant-scope G4, VA sizing G5) are landed per the readiness audit but their **compose** at the
  full mix is itself what V.0 first exercises (`WEEK_6_ARCHITECTURE_GAP_AUDIT.md:16-34`).
- **Confidence calibration:** HIGH that Goals 1/2 (VOLT + POOL/KV, non-GEMM) are single-GPU-deliverable;
  MEDIUM that Goal 5 auto-profile fires on stock serve (the audit's KEYSTONE-GAP, `V1_CP55_SOAK_MEMO.md:30`);
  LOW that Goal 4 Koopman + GEMM-lift Goal 3 engage on stock-vLLM-prefill (nvjet) — these are the projected
  bypass casualties V.0 must confirm/surface.

### 3.5 ENGAGEMENT GATE (Mem #18/#25)
V.0's "customer measurement" IS the 100-agent **stock-workload** run: the weighted mix on stock `vllm
serve` + cipher-platform auto-activation, no intervention. It engages (claims) Goals **1 (density via
POOL/KV) + 2 (tok/W via VOLT)** on the family set {Mistral-7B, Qwen-7B/2.5, Llama-3-8B, 2 SLMs}; it
*measures* Goal 3 MFU (CUPTI) and *probes* Goals 4/5 engagement — reporting honestly which engage on stock
vLLM given nvjet.

---

## PART 4 — V.0 ↔ V.1 RELATIONSHIP (cited)

They are **sequential builds, not the same build** (`V1_CP55_SOAK_MEMO.md:50-70`, the 7-item plan):
- **V.0 = the harness** (SOAK §3 **item 2**, *"Soak harness — build the weighted mixed-workload driver
  over stock vllm serve; right-size to 80 GB"*) + the §3.0 per-gate engagement pre-flight (item 1). V.0
  closes the **"weighted-mix harness net-new" blocker** (`V1_CP55_READINESS_FINDINGS.md` §4.4) and the
  absent-`vllm-serve` carry. **V.0 delivers a runnable, self-tested harness + the engagement map.**
- **V.1 (CP-5.5) = the run on the harness** (SOAK §3 **items 1,3-7** + §4): executes the 5-goal gate
  (density/tok-W/MFU/Koopman/auto-profile), writes `V1_CP55_SOAK_REPORT.md`, 5/5 = v1 ships. V.1 **also**
  needs the **infra unblock** (repackage `.deb` to the validated anchor + reload kmod + restart docker +
  CDI — the sudo/Anil blockers, `V1_CP55_READINESS_FINDINGS.md` §4.1-4.3), separate from V.0.
- **So: V.0 closes the HARNESS (+ engagement map); V.1 closes the GOALS.**

---

## STOP — Anil's call

1. **The design** — the 100-worker weighted-mix orchestrator on `bench_llm.py` + the per-gate engagement
   map pre-flight (§3.0-3.1).
2. **The PRODUCT-SCOPE question (the real decision):** given the cited **nvjet bypass** of CIPHER's
   GEMM-substitution on stock vLLM, is v1's demonstrable cross-tenant value the **non-GEMM levers** (VOLT
   tok/W + POOL/KV density) on stock `vllm serve` — **or** do we inject CIPHER into the torch-decode POOL
   executor to stack Marlin (where evict-wiring becomes load-bearing)? This decides §3.2 and whether the
   `evict_model(model_id)` substrate addition is in scope.
3. **The evict-wiring scope** — load-bearing iff the engaged path is Marlin-engaging (the POOL/torch
   topology, #2); moot on stock-vLLM-prefill.

**No build, no commit, anchors unchanged.** Build is a separate approved step. Commit (when approved) as
Anil, no co-author.
