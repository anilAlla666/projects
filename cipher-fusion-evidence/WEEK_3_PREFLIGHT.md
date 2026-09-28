# Week 3 Pre-flight — Wave 5 §5.5 W3 Verification + SDPA Correctness Inventory

**HEADLINE STATUS: WEEK 3 ENTRY READY.**

Wave 5 §5.5 W3 has 5 VERIFIED + 2 SYNTHESIS-HYPOTHESIS + 0 CONTRADICTION (within W3 scope; 2 Wave-5 contradictions attributed to Week 4). SDPA bit-identical test is **Case A** — Track 2 SC6 exists, runs on this pod, last PASSed 2026-05-19 on both TinyLlama-1.1B and Mistral-7B.

| section | finding | impact |
| --- | --- | --- |
| Part 1 | §7 Week 3 scoped: classifier → oracle → SUBSTITUTE-table → Marlin/cuBLAS/attn/PASS_THROUGH | Implementation deliverable: `cipher_rt_dispatch.cpp` |
| Part 1 | Wave 5 narrows W3 to GEMM-only (attn defers to v1.5) | Implementation scope clarified |
| Part 2 | 5 Wave-5 W3 named targets VERIFIED on disk | No CONTRADICTION; clean entry |
| Part 2 | 2 SYNTHESIS-HYPOTHESIS (observer `maybe_handle` vs Step 4's `observe`; `CIPHER_DISPATCH_LIVE` env not yet defined) | Build-time additions in Week 3 |
| Part 3 | Track 2 SC6 (bit-identical via `torch.equal`) located + models on disk | Case A correctness gate |
| Part 4 | Recommend Track 2 SC6 as Week 3 load-bearing gate | Re-run as Step-0 pre-edit check |
| Part 5 | WEEK 3 ENTRY READY | Scope-lock draft can proceed |

**Date:** 2026-05-20
**Read-only diagnostic.** No source modifications.

---

## Part 1 — v1.2.2 §7 Week 3 baseline scope (verbatim)

### §7 Week 3 — Dispatch routing goes live (CIPHER_REENGINEERING_PLAN.md:1295-1320)

**Goal:** "Classifier output drives actuator selection. Per-regime routing per §4.3." (L1297)

**Changes (5 named items, L1299-1304):**

| # | item | landing |
| ---:| --- | --- |
| 1 | Dispatch table from §4.5 implemented | **`cipher_rt_dispatch.cpp`** (new file) |
| 2 | For each kernel: CLASSIFY → ORACLE → SUBSTITUTE-table → route to {Marlin, cuBLAS shim, attn dispatch, PASS_THROUGH} | hot-path wiring (CUPTI extension or new file) |
| 3 | Marlin lane GATED on (`kernel_class == LARGE_GEMM`) AND (tenant has full-GPU primary ctx, not partitioned) | Marlin actuator gate extension |
| 4 | VOLT engagement: trigger 1200 MHz lock on detect-decode-band | volt extension |
| 5 | SENSE phase transitions trigger DSM PROPOSE for tool-idle detection | cipher_cp54_sched kmod |

**Behavioral tests (L1306-1312):**

- Regime 1 (WL03 prefill): MFU ≥ 60% baseline → 85% with Marlin engaged
- Regime 2 (WL01 decode): tok/W +14% with DVFS engaged
- Regime 3 (WL05 multi-tenant): isolation + density
- Regime 5 (heterogeneous): 1 prefill + 4 decode concurrent
- **W1 regression PASS**
- **Track 2 SC6 PASS** ← canonical correctness gate

**Rollback (L1314):** `CIPHER_DISPATCH_LIVE=0` → all-PASS_THROUGH; anchor `libcipher_rt.so.week3_pre` preserved.

**Risk register (L1316-1319):**
- R-W3.1 HIGH: mis-routing (decode kernel → Marlin)
- R-W3.2 HIGH: DVFS lock vs concurrent prefill tenant
- R-W3.3 MEDIUM: tool-call DSM thrashing

### Wave 5 §5.5 W3 scope-correction (CIPHER_LOGIC_AUDIT_WAVE_5_FUSION_PLAN.md:716-777)

Wave 5 explicitly narrows §7 Week 3:

> **Scope correction vs Section 7.** The original Section 7 calls for all substrates to go live in Week 3. **Scope this to GEMM only:** the attn lane has no substitute actuator today and the LP-2 fix is brand new (landed Week 2). Defer the attn dispatch routing to v1.5 when the first attn substitute actuator (Op-3 SUBSTITUTE for FAVOR+ / FlashSwiftKey) is ready. (L721-726)

**Effect on Step 6 of Week 2:** the LP-2 refactor we shipped at Step 1 is what makes the attn defer SAFE — when attn substitutes land in v1.5, the trampolines correctly honor route(). No further change needed for v1.

---

## Part 2 — Wave 5 §5.5 W3 verification sweep

7 named targets in Wave 5 §5.5 W3 (lines 716-777). Each verified on disk:

| # | Wave 5 line | Target | Status | Notes |
| ---:| --- | --- | --- | --- |
| W3-1 | L729 | `cipher_rt_classify_observer.c::maybe_handle` reads `dec` from oracle, publishes TLS `substitute_hint = recipe_id` | **SYNTHESIS-HYPOTHESIS** | Step 4 shipped `observe()` (not `maybe_handle()`). Naming drift; the per-call function we built is the correct landing site for the W3-prescribed behavior. No CONTRADICTION — the function exists, just named differently. |
| W3-2 | L732 | `cipher_rt_marlin_actuator.c::maybe_handle_marlin` (Ca.9) read TLS hint, short-circuit STABILITY_THRESHOLD | VERIFIED | Function at `cipher_rt_marlin_actuator.c:56`; registered as actuator at L173. |
| W3-3 | L770 | Marlin actuator L106 — "explicit B<8 filter" | VERIFIED (with line drift) | The shape gate is at L100-114: `marlin_M > MARLIN_MAX_M_GATE` reject, plus `marlin_N < 1024 || marlin_K < 1024` reject. Wave 5's "B<8" is shorthand for the MARLIN_MAX_M_GATE check. |
| W3-4 | L736 | `cipher_rt_volt.c` L54-63 LUT (1000/1600/1980 MHz table) | VERIFIED (with 2-line drift) | `batch_to_mhz()` at `cipher_rt_volt.c:54-65` with cases 1/8/32/64 → 1000/1600/1980/1980 MHz. |
| W3-5 | L739 | `cipher_cp54_sched.c::COMPACT_MIGRATE` ioctl nr 20 | VERIFIED | Comment `nr 20 — COMPACT_MIGRATE` at `cipher_cp54_sched.c:794`. (Already noted in `WAVE_5_S5_5_VERIFICATION.md` Week 3 row.) |
| W3-6 | L775 | `cipher_cp54_mig_ratelimit_ms = 10000` | VERIFIED | Defined at `cipher_cp54_sched.c:97` (`static uint cipher_cp54_mig_ratelimit_ms = 10000;`). |
| W3-7 | L763 | `CIPHER_DISPATCH_LIVE=0` rollback env | **SYNTHESIS-HYPOTHESIS** | Env var not yet defined anywhere in source (grep clean). Week 3 must add it alongside the dispatch table. |

### W3-1 detail — observer naming drift

Wave 5 prescribes: "`cipher_rt_classify_observer.c`: `maybe_handle` now reads `dec` from the oracle..."

Step 4 built:
```c
void cipher_rt_classify_observer_observe(
    const struct cipher_rt_classify_call *call,
    const struct cipher_rt_classify_out  *out,
    int                                   result);
```

The function exists; it's named `observe`, not `maybe_handle`. Wave 5's `maybe_handle` is the *actuator* pattern (matmul/attn substrates use that name); observers don't have to use it. The Week 3 brief should clearly state: "extend `cipher_rt_classify_observer_observe()` to additionally read oracle decision + publish TLS substitute_hint when PERMIT + op_class==GEMM + registry hit."

No file-renaming or function-renaming needed. Naming drift documented; resolution = use Step 4's existing function name.

### W3-7 detail — `CIPHER_DISPATCH_LIVE` env not yet wired

Searched: `grep -rn 'CIPHER_DISPATCH_LIVE' /home/ubuntu/cipher_rt_phase4/ /home/ubuntu/cipher_kmod/` → zero matches.

Wave 5 prescribes it as the Week 3 rollback hook. Week 3 build:
- Read env var at `cipher_rt_dispatch.cpp` init (or substrate init).
- If unset / "0" → all CLASSIFY-routed kernels return PASSTHROUGH (no actuator dispatch, Week-2 behavior).
- If "1" → live dispatch.

Boolean rollback toggle; trivial to wire. SYNTHESIS-HYPOTHESIS, ~10 LOC.

### Wave 5 W3 invariant naming drift

L745: "SDPA on **Llama-3-8B** bit-identical (Track 2 SC6 PASS)"

**SC6 actually uses TinyLlama-1.1B + Mistral-7B-v0.1, not Llama-3-8B.** Llama-3-8B is not in `/home/ubuntu/models/`. The named test (Track 2 SC6) exists; the cited model is wrong. The invariant should be restated: "SDPA on Mistral-7B + TinyLlama bit-identical via Track 2 SC6". Naming drift; not a blocker.

### Cross-reference with prior verification (WAVE_5_S5_5_VERIFICATION.md)

The Week-1-era verification sweep already covered Wave 5 W3-5 (L739) and W3-6 (L775). Both were VERIFIED. The two CONTRADICTIONs in `WAVE_5_S5_5_VERIFICATION.md` (cipher_audit.cpp absence; cipher_partition_slot/cipher_slots naming) are attributed to **Week 4**, not Week 3. They do not block Week 3 entry.

---

## Part 3 — SDPA bit-identical test inventory

### 3.1 Track 2 SC6 — located + recently passed

**Path:** `/home/ubuntu/cipher-fusion-evidence/phase_c/sc6_run.py`
**Companion files:** `sc6_models.py`, `sc6_consumer.py`, `sc6_producer.py`, `sc6_aggregate.py`, `sc6_independent_tenant.py`
**Invocation:** `python3 sc6_run.py [model_name] [shared|independent]`

**What it tests (bit-identical claim):**

```python
# sc6_run.py:137
bit[when].append(bool(torch.equal(yp, yc)))
# Verifies: producer's forward-pass output yp is byte-equal to
# each consumer's output yc, for both:
#   pre  — before producer is SIGKILL'd
#   post — after producer dies (consumer holds the arena alone)
```

This is **exact-bit comparison** (`torch.equal()` returns True iff every element matches bit-for-bit), not numerical tolerance. The test exercises the attention path implicitly because the model's forward pass includes SDPA.

**Last successful run:** 2026-05-19 12:04 (result JSONs on disk at `phase_c/sc6_Mistral-7B_consumer_*_result.json`). Mistral-7B: 7/7 PASS at N=4; TinyLlama: 7/7 PASS at N=4.

**Models on disk** (`/home/ubuntu/models/`):
- `TinyLlama-1.1B` ✓ (2.2 GiB)
- `Mistral-7B-v0.1` ✓ (14.6 GiB)
- `Llama-3-8B` ✗ (Wave 5's name; absent)

**Expected runtime:** ~5-10 minutes per (model, phase) tuple based on past runs. For the Week 3 gate, the `shared` phase is sufficient (bit-identical assertions live in shared-phase consumers).

**CIPHER injection mode:** SC6 currently runs without `CUDA_INJECTION64_PATH`/`LD_PRELOAD` — it tests the kmod arena weight-sharing primitives, not the substrate. For Week 3 we need to additionally run it WITH `CUDA_INJECTION64_PATH=libcipher_rt.so` to verify the dispatch routing doesn't break bit-identical attention output.

### 3.2 Other SDPA-related tests on this pod

| file | category | what |
| --- | --- | --- |
| `cipher-fusion-evidence/cp_2_5/cp25_sdpa_smoke.py` | **SMOKE** | Checks `[cipher-attn] tramp_calls > 0` after 20 SDPA calls. Substrate-exercise test; not correctness. |
| `cipher_rt_phase4/test_spec_generate.py` | (unknown — non-SDPA) | spec_decode-related |
| `cipher_rt_phase4/test_cipher_spec_decode.py` | (unknown — non-SDPA) | spec_decode-related |
| `cipher-may13-evidence/tests/test_*.py` (~15 files) | various | L3/observer/sense/comply/receipt smokes; none target SDPA bit-identity |
| `cipher-may13-evidence/tests/bench_continuous_batching.py` | PERFORMANCE | benchmark, not correctness |
| Step 1 + Step 6 inline Mistral SDPA loop | SMOKE | rc + finite output; what I built ad-hoc during Week 2 |

### 3.3 Mistral SDPA smoke (Steps 1 + 6 of Week 2) — informational

Already proven to run. Latency: ~10s for 150-iteration loop. Confirms substrate doesn't crash; **not** bit-identical.

### 3.4 Inventory verdict

**Case A — BIT-IDENTICAL test exists and runs on this pod.** Track 2 SC6 is the canonical fit.

The only nuance is that SC6 was designed to test weight-sharing (Track 2 SC2-SC5 primitives), not classify-routing per se. For Week 3's purpose, the SDPA bit-identical assertion is **incidental but load-bearing**: if classify-routing perturbs attention output between producer and consumer (e.g., a routing decision is non-deterministic across processes), SC6's `torch.equal()` check fails immediately. That's exactly the catch we want.

---

## Part 4 — Correctness gate recommendation

**Recommendation: Track 2 SC6 (shared phase) as Week 3 load-bearing gate**, instrumented in TWO modes for thoroughness:

| mode | invocation | gate |
| --- | --- | --- |
| baseline | `python3 sc6_run.py Mistral-7B shared` | Reproduces 2026-05-19 PASS; confirms classify-routing doesn't change semantics relative to vanilla torch |
| under CIPHER | `CUDA_INJECTION64_PATH=$(realpath libcipher_rt.so) python3 sc6_run.py Mistral-7B shared` | The new test: classify substrate is active during the consumer forward pass; bit-identity must still hold |

### Per-step usage pattern

- **Week 3 Step 0** (this pre-flight): note the gate.
- **Week 3 Step 1** (`cipher_rt_dispatch.cpp` scaffolding): SC6 should PASS unchanged (dispatch not yet live; substrate just observes).
- **Week 3 Step N** (dispatch goes live with `CIPHER_DISPATCH_LIVE=1`): SC6 **must PASS** — this is the regression gate per Wave 5 I-W3.3.
- **Week 3 Step closeout**: SC6 PASS recorded for both Mistral-7B and TinyLlama.

### TinyLlama as the cheap pre-check

TinyLlama-1.1B SC6 runs in ~2-3 minutes vs Mistral-7B at ~5-10 min. Recommend Step-level brief structure: TinyLlama SC6 PASS gates the step; Mistral-7B SC6 PASS gates the week.

### Smoke-only gates (insufficient on their own)

Step 1/Step 6 inline Mistral SDPA loop and `cp25_sdpa_smoke.py` continue to serve as quick smoke checks (rc + finite output). They catch crashes but NOT correctness perturbation. Use them per-edit; rely on SC6 per-step.

### Honest weakness

SC6 tests **one canonical prompt** (`"The history of computing spans..."`, `sc6_models.py:18-20`). It proves the classify-routing doesn't perturb that specific forward pass. It does not prove every prompt is bit-identical. For Week 3's purpose this is sufficient — any classify-routing bug that's prompt-conditional would still likely surface at the canonical prompt (the failure modes are typically "this kernel substituted vs not" which is geometry-driven, not prompt-driven). Phase 5 / CP 5.5 expands prompt coverage.

---

## Part 5 — Week 3 entry readiness

### Verdict: **WEEK 3 ENTRY READY**

- **Part 1**: §7 Week 3 scope captured verbatim. Wave 5 narrows it cleanly to GEMM-only (LP-2 already shipped Step 1).
- **Part 2**: 5 VERIFIED + 2 SYNTHESIS-HYPOTHESIS + 0 CONTRADICTION. The 2 hypotheses are both naming/missing-env class, resolvable inline in Week 3 implementation. No structural surprise.
- **Part 3**: Track 2 SC6 located, runnable, models present, last PASS 2026-05-19. Case A.
- **Part 4**: Correctness gate recommended (Track 2 SC6 in baseline + CIPHER-injected modes).

### Recommended Week 3 brief structure

The scope-lock brief should land 4 implementation steps:

| step | deliverable | gate |
| --- | --- | --- |
| 1 | `cipher_rt_dispatch.cpp` scaffolding (table + lookup, no actuation yet) | TinyLlama SC6 PASS |
| 2 | Extend `cipher_rt_classify_observer_observe()` to publish TLS `substitute_hint` (W3-1 naming-drift resolution; oracle decode + registry lookup gated on PERMIT + GEMM + hit) | TinyLlama SC6 PASS |
| 3 | Marlin actuator: consume TLS hint to short-circuit STABILITY_THRESHOLD (W3-2). VOLT extension for decode-band lock (W3-4) | TinyLlama SC6 PASS + Mistral SDPA smoke |
| 4 | Activate `CIPHER_DISPATCH_LIVE=1` env-gated routing (W3-7); SENSE → DSM PROPOSE wiring (W3-5) | **Mistral-7B SC6 PASS (load-bearing); CP 5.4 15/15; W1 regression** |
| 5 | Closeout: `week-3-complete` tag on all 3 trees | — |

### Surfaced for Week 3 scope-lock brief

- **`CIPHER_DISPATCH_LIVE=0` default** — first half of Week 3 ships the substrate but routes nothing (rollback-safe by default). Step 4 flips the default to 1 only after the bit-identical gate confirms.
- **`libcipher_rt.so.week3_pre` anchor preservation** — recommend snapshotting at week-2-complete (= current state) before Step 1 starts; rollback target if any step PARTIAL.
- **Sites 1+2 from Week 2 pre-flight**: still dropped. Week 3 dispatch table lives in `cipher_rt_dispatch.cpp` and consumes the classifier hint that CUPTI populates per-launch via TLS. cuBLAS GemmEx and SDPA shims consult the TLS, not their own classify_route call.

### Out-of-scope confirms

- Attn dispatch routing: defers to v1.5 per Wave 5 L724-726.
- Sites 1+2 cuBLAS/SDPA CLASSIFY direct wiring: still N/A (geometry mismatch persists).
- Wave 5 Week 4 contradictions (cipher_audit.cpp / slot array naming): Week 4 work; doesn't block Week 3.

---

## Telemetry on disk

- `/home/ubuntu/cipher-fusion-evidence/CIPHER_REENGINEERING_PLAN.md:1295-1320` — §7 Week 3
- `/home/ubuntu/cipher-fusion-evidence/CIPHER_LOGIC_AUDIT_WAVE_5_FUSION_PLAN.md:716-777` — Wave 5 §5.5 W3
- `/home/ubuntu/cipher-fusion-evidence/WAVE_5_S5_5_VERIFICATION.md` — prior Week-1-era verification sweep
- `/home/ubuntu/cipher-fusion-evidence/phase_c/sc6_run.py` — Track 2 SC6 driver
- `/home/ubuntu/cipher-fusion-evidence/phase_c/sc6_models.py` — model registry (TinyLlama + Mistral-7B)
- `/home/ubuntu/cipher-fusion-evidence/phase_c/sc6_Mistral-7B_consumer_*_result.json` — last successful run 2026-05-19
- `/home/ubuntu/models/TinyLlama-1.1B/`, `/home/ubuntu/models/Mistral-7B-v0.1/` — model weights present
- `/home/ubuntu/cipher_rt_phase4/cipher_rt_marlin_actuator.c:56,106,173` — Marlin actuator + gate
- `/home/ubuntu/cipher_rt_phase4/cipher_rt_volt.c:54-65` — VOLT batch_to_mhz LUT
- `/home/ubuntu/cipher_kmod/cipher_cp54_sched.c:97,794` — COMPACT_MIGRATE + ratelimit
- `/home/ubuntu/cipher_rt_phase4/cipher_rt_classify_observer.c` — Step 4 observer

---

## Discipline notes

- Read-only diagnostic. No source-tree changes. Tree state at `week-2-complete` on all three trees.
- 0 CONTRADICTION; entry is clean.
- 2 SYNTHESIS-HYPOTHESIS surfaced for inline resolution in Week 3 implementation (both are naming/env-var class, ~30 LOC combined).
- Track 2 SC6 is the load-bearing correctness gate. Recommend it as per-step (TinyLlama) + per-week (Mistral-7B) check.
- Wave 5's "Llama-3-8B" reference in I-W3.3 is a model-name drift; SC6 uses Mistral-7B which is more representative anyway.

---

## Awaiting

Week 3 scope-lock brief, drafted with:
- 4 implementation steps + 1 closeout (per Part 5 recommended structure)
- `CIPHER_DISPATCH_LIVE=0` default; flip-to-1 only at Step 4
- `libcipher_rt.so.week3_pre` anchor at week-2-complete
- Track 2 SC6 as the load-bearing gate (TinyLlama per-step, Mistral-7B per-week)
- 2 SYNTHESIS-HYPOTHESIS items resolved in the brief (`observe()` vs `maybe_handle` naming + `CIPHER_DISPATCH_LIVE` env definition)
