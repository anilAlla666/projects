# W14 Followup — Option-2 Step 0.5: Koopman Engine Reachability Probe (CLOSED)

**Date:** 2026-05-25
**Pre-state anchor:** `week-13-14-complete` (verified on disk at probe time)
- `cipher_rt_phase4` `25970f3` (`libcipher_rt.so` md5 `097cf8d907a7e866a3e3640eb0993003`)
- `cipher_kmod` `8c643fc` (0.6.5)
- `cipher_kv_bridge.so` at W12 Step 3 anchor (unused; `CIPHER_KV_ALLOC=0` per Step 0 hygiene)
- `cipher_vllm_plugin/cipher_vllm_kv.py` Step 0 anchor `e77a3a58a5f075985c838110f9193585`

**Tag:** `option-2-step-0-5-koopman-reachability-probe`
**Substrate-tree status:** UNCHANGED. Plugin-only edit: `cipher_vllm_kv.py` md5 `e77a3a58` → `305003d545cfeb6d2cc8418b0d5e4a97` (+8 LOC across the existing counter-dump function; adds 3 already-exported C symbols to the dump tuple).
**Pre-condition:** Step 0 PASS (`WEEK_14_FOLLOWUP_OPTION_2_STEP_0_VLLM_WORKER_HOOK.md`) — worker-subprocess GOT patches confirmed installed.

**Verdict:** **OUTCOME (a)** — classifier never routes to Koopman. Step 1 β-sweep is **NOT meaningful** on the substrate as-is.

---

## 1. Why this step exists

Step 0 closed with `cipher_rt_cublas_shim_calls: 0 → 11658` in the worker but `cipher_rt_koopman_calls_total: 0` even at `CIPHER_KOOPMAN=1 CIPHER_KOOPMAN_OOD_THRESHOLD=0.99` (always-fire-approx). Step 0 §2 Gate 3 documented the failure as "informational" but `WEEK_14_FOLLOWUP_OPTION_2_STEP_0_VLLM_WORKER_HOOK.md` §7 entry checklist #4 explicitly flagged: *"Step 1 calibration formal sweep must surface why — classifier filter at dispatcher level, or residual_ratio > 0.99 on real inputs."*

Anil 2026-05-25 inserted this Step 0.5 as a HARD GATE before Step 1: localize the drop layer first, then adjudicate whether the β-sweep planned for Step 1 is meaningful on substrate as-is. Without localization, Step 1 would sweep a knob inside a path that is not being reached, and the campaign would burn 1.5–2.5 ED measuring zero across seven β values.

---

## 2. Method

**Substrate audit (no code change required):** All 5 counters named in the Step 0.5 scope already exist as exported C symbols in `libcipher_rt.so`:

| Scope-spec counter | Exported symbol | File:line |
|---|---|---|
| `cipher_rt_cublas_shim_calls` (Step 0: 11658) | `cipher_rt_cublas_shim_calls` | `cipher_rt_cublas_shim.c:222` |
| matmul-dispatch entry count | `cipher_rt_matmul_calls_total` | `cipher_rt_matmul_dispatch.c:118` |
| classifier decision histogram (PASSTHROUGH / KOOPMAN / other) | `cipher_rt_matmul_calls_handled` + `cipher_rt_matmul_calls_passthrough` | `cipher_rt_matmul_dispatch.c:120-122` |
| `cipher_rt_koopman_calls_total` (Step 0: 0) | `cipher_rt_koopman_calls_total` | `cipher_rt_koopman_engine.cpp:237` |
| `cipher_rt_koopman_calls_skipped` | `cipher_rt_koopman_calls_skipped` | `cipher_rt_koopman_engine.cpp:247` |

The classifier-decision histogram collapses to 2 buckets at the matmul-dispatch level (handled / passthrough) plus 1 bucket inside the Koopman actuator (skipped, after engine entry). Per-actuator histogram is trivial here because `actuators=1` (Koopman only) — see Step 0 atexit log "`MATMUL: exit totals — calls=11658 handled=0 passthrough=11658 (actuators=1)`".

**Plugin extension:** `_install_counter_dump_for_verification()` in `cipher_vllm_kv.py` was extended to include the 3 matmul-dispatch counters in addition to the 4 already captured. This is plugin-side only — no substrate code touched. New plugin md5 `305003d5`; snapshot at `plugin_snapshots/cipher_vllm_kv.py.w14_followup_step_0_5`.

**Probe harness:** `option2/step_0_5/verify_step0_5.py`. Identical structure to `verify_step0.py` (vLLM V1 AsyncLLMEngine, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, TinyLlama-1.1B-Chat-v1.0, 128-token greedy decode), with extended gate evaluation that classifies into outcomes (a)/(b)/(c) per scope.

**Env:** `LD_PRELOAD=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so`, `CUDA_INJECTION64_PATH` UNSET (Goal 5 LD_PRELOAD-only mode), `CIPHER_KOOPMAN=1`, `CIPHER_KOOPMAN_OOD_THRESHOLD=0.99` (always-fire-approx), `CIPHER_KV_ALLOC=0` (Step 0 hygiene).

---

## 3. Result

Worker pid `2035928`, 4 snapshots (install / sigusr1×2 / atexit) over 11.13 s, decode burst 1.37 s, output `" and kind queen. She had a daughter named Lily, who was a beautiful and kind-hea"` (byte-identical to Step 0).

| Counter | Δ over decode | absolute |
|---|---|---|
| `cipher_rt_cublas_shim_calls` | **+11658** | 11658 |
| `cipher_rt_matmul_calls_total` | **+11658** | 11658 |
| `cipher_rt_matmul_calls_handled` | 0 | 0 |
| `cipher_rt_matmul_calls_passthrough` | **+11658** | 11658 |
| `cipher_rt_koopman_calls_total` | **0** | 0 |
| `cipher_rt_koopman_calls_handled` | 0 | 0 |
| `cipher_rt_koopman_calls_skipped` | 0 | 0 |

Worker-side stderr corroboration (worker pid `2036370` in the second verification run; identical-outcome cross-check captured at `probe_run_2_full_stderr.log`):

| Line | Source | Evidence |
|---|---|---|
| L52 | worker init (after EngineCore spawn, before `(EngineCore pid=...) [cipher-vllm-kv] InitializeInjection2`) | `[cipher_v2] MATMUL: actuator 'koopman' registered at priority 0 (slot 0/1)` |
| L53 | worker init | `[CIPHER KOOPMAN] engine registered with matmul-dispatch substrate (verbose=0)` |
| L51 | worker init | `[cipher_v2] MARLIN: actuator DISABLED (CIPHER_MARLIN not set)` (Koopman is the ONLY matmul actuator in the worker) |
| L118 | worker atexit | `[cipher_v2] MATMUL: exit totals — calls=11658 handled=0 passthrough=11658 (actuators=1)` |
| L147 | parent atexit | `[cipher_v2] MATMUL: exit totals — calls=0 handled=0 passthrough=0 (actuators=1)` (parent loaded libcipher_rt but ran no decode GEMMs — confirms the 11658 above is the worker's count, not the parent's) |

The Koopman actuator IS registered in the WORKER process (L52-53), Marlin is explicitly DISABLED (L51), and the worker's atexit `actuators=1` (L118) is therefore Koopman by elimination. Registration in `cipher_rt_koopman_init` (`cipher_rt_koopman_engine.cpp:200-225`) sets `g_enabled.store(1)` AFTER successful `cipher_rt_matmul_register_actuator`, so a successful registration log message (L53) implies g_enabled=1 in the worker. The substrate is correctly wired end-to-end through matmul-dispatch entry; the call drops INSIDE the Koopman actuator's `maybe_handle_koopman` before its g_calls_total increment.

**Full result:** `option2/step_0_5/probe_result.json`. Worker JSON dump: `option2/step_0_5/worker_2035928.json`.

---

## 4. Mapping to scope outcomes

Scope §1 outcomes restated against measured data:

- **(a) Classifier never routes to Koopman** — REQUIRES `matmul_total > 0 AND koopman_total = 0`. **MEASURED: m_total=11658, k_total=0.** ✓
- **(b) Classifier routes; Koopman skips at residual gate** — REQUIRES `koopman_total > 0 AND koopman_handled = 0`. Measured: k_total=0. ✗
- **(c) Some other layer drops the call** — REQUIRES `m_total = 0 OR m_total ≠ shim`. Measured: m_total = shim = 11658. ✗

**Confirmed: OUTCOME (a).**

---

## 5. Localization — which early-exit inside maybe_handle_koopman

The Koopman actuator's `maybe_handle_koopman` (`cipher_rt_koopman_engine.cpp:95-113`) has four PASSTHROUGH-returning early-exits BEFORE the `g_calls_total.fetch_add` at line 113:

```c
98:    if (!g_enabled.load(std::memory_order_relaxed))
99:        return CIPHER_RT_MATMUL_PASSTHROUGH;
100:
101:    /* FP16-only — the .cu kernel is FP16-in/FP16-out */
102:    if (call->Atype != CUDA_R_16F ||
103:        call->Btype != CUDA_R_16F ||
104:        call->Ctype != CUDA_R_16F)
105:        return CIPHER_RT_MATMUL_PASSTHROUGH;
106:
107:    /* Defensive: bail on degenerate dimensions */
108:    if (call->m <= 0 || call->n <= 0 || call->k <= 0)
109:        return CIPHER_RT_MATMUL_PASSTHROUGH;
110:    if (!call->B || !call->C)
111:        return CIPHER_RT_MATMUL_PASSTHROUGH;
112:
113:    g_calls_total.fetch_add(1, std::memory_order_relaxed);
```

Four candidates, ranked by evidence:

1. **`g_enabled` check (line 98-99)** — ELIMINATED. The init function at `cipher_rt_koopman_engine.cpp:200-225` only proceeds to `g_enabled.store(1)` AFTER successful registration; the worker stderr shows both registration ("`MATMUL: actuator 'koopman' registered`") and engine-up ("`[CIPHER KOOPMAN] engine registered with matmul-dispatch substrate`"), so g_enabled=1.

2. **FP16-only dtype check (line 102-105)** — **HIGH CONFIDENCE.** The vLLM EngineCore log line shows `dtype=torch.bfloat16` for TinyLlama-1.1B-Chat-v1.0 (vLLM 0.20.2 default for this model). The Koopman .cu kernel is FP16-in/FP16-out (per the inline comment at line 101); BF16 has CUDA enum `CUDA_R_16BF=14`, FP16 is `CUDA_R_16F=2` — the equality check fails on every BF16 GEMM. With 11658/11658 GEMMs at bf16, every call hits this early-exit.

3. **Degenerate-dim check (line 108-109)** — UNLIKELY. TinyLlama's hidden dim (2048), head dim (64), num heads (32), and LM-head shape (2048→32000) all yield m/n/k ≥ 1; vLLM would crash on degenerate-dim GEMMs upstream.

4. **Null-ptr check (line 110-111)** — UNLIKELY. A null tensor pointer in a vLLM decode GEMM would page-fault the model long before this check.

**Strongest inference: the FP16-only dtype gate filters 11658/11658 GEMMs because TinyLlama runs in bf16 under vLLM V1 default.** Direct proof would require a transient `fprintf(stderr, "Atype=%d", call->Atype)` at the actuator entry, or a per-early-exit counter pair (one each for dtype-skip / dim-skip / null-skip). Both are sub-step candidates — see §8.

---

## 6. Step 1 meaningfulness adjudication — HARD GATE

**Scope-lock §4 Step 1 method (line 85):** *"Sweep β over {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9}"*

β controls `g_ood_threshold` (line 90), which gates the OOD residual check at lines 133-141. That check sits **AFTER** `g_calls_total.fetch_add(1)` (line 113). With koopman_total=0, the OOD path is unreachable; no value of β changes any observable.

**Verdict: Step 1 β-sweep is NOT meaningful on substrate as-is.** Running it would produce 7 zero rows, identical to Step 0's Gate 3 result, with no new information.

**Implication for scope-lock §4 Step 1 GATE A / GATE B (line 91-92):** GATE B (`∀ β, NOT(fire_rate ≥ 0.01 AND top-1 ≥ 0.90)`) trivially holds when fire_rate=0 by construction. Scope-lock's GATE B → Branch B path therefore fires on a null measurement, exactly as the prior null-attempt (`/tmp/option2_sweep_test.json` 2026-05-24) did — except this time the null is attributable to a known substrate constraint (FP16-only) rather than to the W14 Step 2 E architectural ceiling (residual_ratio > 0.5 OOD).

This is a **different mechanism reaching the same outcome** as the Step 2 E prior. The campaign's Branch B framing in scope-lock §3 decision 3 (line 50) and §8 (line 188-189) needs an addendum to distinguish "Koopman would not fire because no real input is in-distribution to the calibrated manifold" (Step 2 E's finding) from "Koopman cannot fire because the production model's dtype is outside the actuator's input contract" (Step 0.5's finding). Both are honest residues; the latter is the binding constraint on substrate as-is.

---

## 7. Three options for Step 1 reframe (surface to user adjudication)

The campaign cannot continue to Step 1 as scope-locked without one of these:

**Option α — Confirm-and-close-as-Branch-B-FP16 (LOW EFFORT, ~0.5 ED).** Add 3 per-early-exit counters (`cipher_rt_koopman_skip_dtype`, `_skip_dim`, `_skip_null`) inside `maybe_handle_koopman` (1 LOC each, before the `return`). Re-run the probe. If `skip_dtype == 11658`, the BF16 hypothesis is direct evidence. Campaign closes at Branch B with the sharper finding folded in: *"Koopman actuator is FP16-only by .cu kernel contract; vLLM V1 production inference defaults to bfloat16; substrate ships correct but never fires on real-decode by dtype mismatch, not by OOD residual."* No Step 2 / Step 3 run.

**Option β — Port .cu kernel to BF16 (HIGH EFFORT, several ED, substrate-substantive).** Add a BF16 variant of the Koopman .cu kernel (or a runtime BF16↔FP16 conversion wrapper). After the port, re-run Step 0.5 to confirm koopman_total > 0, then run scope-locked Step 1 calibration sweep. This is the only path to a **Branch A** outcome — the campaign's scope-lock §3 decision 3 (a) pre-committed framing. Substantive risk: conversion overhead may dominate Koopman savings (analogous to Branch C in original scope §3 decision 3 (c)).

**Option γ — Workaround: force vLLM to FP16 (LOW EFFORT, ~0.3 ED to re-run, but model-quality risk).** Re-run the probe with `dtype=float16` (vLLM AsyncEngineArgs `dtype="float16"`). If koopman_total > 0 under FP16, Step 1 β-sweep can proceed against an FP16 vLLM deployment, but this introduces a deployment-side constraint that conflicts with bf16-trained-model best practice. TinyLlama-1.1B's published behavior is at bf16; FP16 inference quality is a separate engineering question. This option produces a campaign outcome that's honest about a vLLM-side workaround being a precondition.

**Recommendation:** Option α first (cheap, gives sharp evidence for the campaign closeout), then user adjudicates whether to invest in Option β or pivot to a different actuator surface area. Option γ is a measurement curiosity, not a deployment path.

---

## 8. Honest residue

1. **FP16-hypothesis is high-confidence but not directly proven.** vLLM log shows `dtype=torch.bfloat16` for the engine, and the Koopman dtype gate is the only check before `g_calls_total++` that bf16 model inputs trigger. But per-early-exit counters do not exist — adding them is a 1-LOC substrate change per exit; surfaced as Option α in §7. The current evidence is observational + inferential, not direct counter evidence.

2. **Campaign Branch B mechanism is sharper than scope-lock anticipated.** Scope-lock §3 decision 3 (line 50) framed Branch B as "architectural ceiling identified at W14 Step 2 E reproduces". Step 0.5's finding is upstream of that ceiling: even if Step 2 E's residual-ratio-OOD ceiling were eliminated, the actuator wouldn't fire on bf16 inputs because of the dtype gate. The Branch B template (scope-lock §8 line 188) needs an addendum if the campaign exits Branch B after Option α; the bracketed `[specific architectural finding from this campaign]` slot in the template is the place for "FP16-only actuator vs bf16 production dtype" to land.

3. **Mistral-7B Step 3 inheritance is the same.** Mistral-7B-Instruct under vLLM also defaults to bf16; same dtype-gate failure mode will reproduce. Step 3's E.7 environment block is therefore not the only carry-forward — even if E.7 were resolved tomorrow, Step 3 would null-measure for the same reason as TinyLlama at Step 0.5.

4. **The "classifier" in the user's scope §1 (a) is the matmul-dispatch + actuator-early-exit composite.** This substrate has no separate classifier between matmul-dispatch and Koopman engine entry; matmul-dispatch iterates registered actuators and each actuator self-classifies via early-exits. The user's mental model maps cleanly: m_total ≠ 0 AND k_total = 0 IS "classifier never routes". The localization is exact even though the substrate's naming doesn't match the spec's wording verbatim.

5. **Mode-conflict with `cipher_rt_classify_substrate`.** The worker stderr shows `CLASSIFY: actuator 'may13_default' registered at priority 0 (slot 0/1)` (a separate classifier substrate at `cipher_rt_classify_substrate.cpp`, wrapping `cipher::classify_launch` on the kernel-launch path). This is NOT in the matmul/Koopman path — it's a sibling subsystem for kernel-launch classification. Mentioned here to prevent future confusion if the term "classifier" is searched in the codebase.

---

## 9. Surprises and process notes

### 9.1 vLLM log explicitly shows the BF16 hypothesis evidence on first run

The vLLM `core.py` engine-config dump on the very first probe run printed `dtype=torch.bfloat16` in plain text. No second run needed to surface the strong-prior for the dtype-gate hypothesis. The Step 0 run also printed this in its stderr — but no one was looking for the dtype line at Step 0 because the Step 0 gate didn't decompose the Koopman path's early-exits. Lesson for future verification doc text: name the model dtype alongside the env vars in the harness banner.

### 9.2 Probe re-uses Step 0 plugin counter-dump infrastructure cleanly

The Step 0 dump was designed for 4 counters; extending to 7 was an 8-LOC edit (adding 3 names to two tuples). The pattern generalizes — any future C symbol with signature `unsigned long fn(void)` can be added by name. Reference: `cipher_vllm_kv.py:740-758` after Step 0.5 edit.

### 9.3 The matmul-dispatch atexit log already had the answer

`[cipher_v2] MATMUL: exit totals — calls=11658 handled=0 passthrough=11658 (actuators=1)` was emitted during Step 0 verification and recorded in Step 0 step doc §2 Gate 2 corroboration block. Reading this together with `[CIPHER KOOPMAN] engine registered` and the `actuators=1` value should have triggered the outcome (a) inference at Step 0 close — the only registered actuator returning PASSTHROUGH on 100% of calls means that actuator's early-exits filtered them. Step 0 doc §2 Gate 3 closing paragraph (line 57) named "(a) doesn't reach the engine entry for any shape class (classifier filter at the MATMUL dispatcher level)" as one of two upstream-evidence hypotheses but did not pin which one. Step 0.5 is what pinned it.

### 9.4 No KV-bridge VA pool exhaustion this run

Step 0 §4.4 noted a first-pass crash from KV-bridge VA pool exhaustion, resolved by `CIPHER_KV_ALLOC=0`. Step 0.5 used the same env from the outset; no crash, no incident.

### 9.5 Advisor-flagged verification: worker-side Koopman registration

After the first probe run, advisor flagged that the §5 elimination of the `g_enabled` early-exit relied on inferring "worker has actuators=1 = Koopman" from a stderr line that lacked an explicit `(EngineCore pid=...)` prefix. A second probe run was conducted with full stderr captured to `probe_run_2_full_stderr.log`. The capture shows the WORKER process emits BOTH the substrate registration line (L52: `MATMUL: actuator 'koopman' registered`) AND the Koopman engine init line (L53: `[CIPHER KOOPMAN] engine registered`) in its own init sequence (between EngineCore spawn and the `(EngineCore pid=...) [cipher-vllm-kv] InitializeInjection2` line). The substrate lines lack the EngineCore prefix because libcipher_rt writes via `fprintf(stderr, ...)` at LD_PRELOAD load time, which precedes vLLM's stderr-capture-and-prefix wrapper engaging on the worker subprocess. The parent's atexit line shows `calls=0` (it loaded libcipher_rt but ran no decode GEMMs), confirming the `calls=11658 actuators=1` atexit line attributes to the worker process. The g_enabled elimination in §5 holds. Outcome (a) classification holds across both runs (identical counter deltas, byte-identical decode output).

---

## 10. Substrate state at Step 0.5 close

| Tree | Commit / md5 | Notes |
|---|---|---|
| `cipher_rt_phase4` | `25970f3` / libcipher_rt.so `097cf8d9` | UNCHANGED (`week-13-14-complete`) |
| `cipher_kmod` | `8c643fc` | UNCHANGED |
| `cipher_kv_bridge.so` | W12 Step 3 anchor | UNCHANGED (unused; `CIPHER_KV_ALLOC=0`) |
| `cipher_vllm_plugin/cipher_vllm_kv.py` | md5 `e77a3a58` → `305003d5` | +8 LOC in `_install_counter_dump_for_verification`; snapshot at `plugin_snapshots/cipher_vllm_kv.py.w14_followup_step_0_5` |
| `cipher-fusion-evidence` | this commit lands `option-2-step-0-5-koopman-reachability-probe` tag | this doc + `option2/step_0_5/` artifact tree |

---

## 11. Step 0.5 ED actual vs budgeted

**Budget:** 0.5 ED (scope spec).

**Actual:** ~0.3 ED in one session (substrate counter audit → plugin-dump extension → probe re-run → outcome adjudication → step doc). Within budget. Counter audit was faster than budgeted because all 5 spec counters already exist as C symbols.

---

## 12. HARD GATE evaluation (per scope spec)

> *"HARD GATE before Step 1: localized which layer drops the call AND adjudicated whether Step 1 calibration sweep is meaningful on the substrate as-is."*

| Gate clause | Status |
|---|---|
| Localized which layer drops the call | **PASS** — drop is inside `maybe_handle_koopman` (cipher_rt_koopman_engine.cpp:98-111), between matmul-dispatch handoff and the `g_calls_total` increment; high-confidence inference points to the FP16-only dtype gate (line 102-105) given vLLM TinyLlama runs in bf16 |
| Adjudicated Step 1 meaningfulness | **PASS** — Step 1 β-sweep is NOT meaningful on substrate as-is; surfaced 3 reframe options (§7) for user adjudication |

**HARD GATE: PASS.** Step 1 as scope-locked is **blocked pending user adjudication of §7 options.**

---

## 13. Related memory

- [[w14-followup-option-2-step-0]] — predecessor; surfaced the `koopman_calls_total=0` finding that Step 0.5 investigates
- [[vllm-v1-worker-subprocess]] — Step 0's underlying fix; remains the precondition for any future Koopman probe under vLLM V1
- [[w13-14-complete]] — substrate anchor (unchanged by Step 0.5)
- [[w14-step-2-koopman-tier]] — substrate-side closure of W14 Step 2; FP16-only kernel ships per .cu contract at that close
- [[cipher-t43-envelope]] — precedent for "substrate validates on test workload, doesn't generalize to production regime"; Step 0.5 is a structurally similar finding (substrate validates on FP16 synthetic; doesn't reach bf16 production)
- [[cipher-lift-framing]] — precedent for honest single-workload-class framing; if campaign exits via Option α, framing language inherits from here

---

**HARD GATE PASSED. Step 1 BLOCKED on user adjudication of §7 reframe options (α / β / γ).**

**Recommended next prompt (on user approval of Option α):** add 3 per-early-exit counters to `maybe_handle_koopman` (1 LOC each), rebuild libcipher_rt.so (substrate change — rotates anchor from `25970f3`), re-run probe, confirm `skip_dtype` accounts for the 11658, then close campaign on Branch B with FP16/BF16 finding folded into the Branch B template (scope-lock §8 line 188-189). Eng-day ~0.5.
