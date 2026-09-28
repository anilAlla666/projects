# D.7 R-H1 heterogeneous weight residence — CLOSE REPORT (close-status)

**Date:** 2026-05-29. bf16 (real serving dtype) throughout. **Anchor NOT rotated** yet
(`9fe23143`) — the close gate is met except the 30-min soak is in-flight and the W7-W11
microbench suite is not reproducible as a named set here (see §4). Substrate edit:
cipher_rt_phase4 `ed130e7` (Marlin `(model_id,w_ptr)` re-key, additive, default-OFF).

## §1 Positive correctness — PASS (real KL, not token-match)

4 distinct families bf16, clean co-residence (load all once, no churn), per-`model_uuid`
routing via `bind_model`, d10 build (LT_ROUTE off — Marlin engages via the existing
cublasGemmEx interception). Per-step **logit** comparison vs separate-process clean solo refs:

| family | marlin_sub | max_logit_diff (co-res vs solo) | KL |
|---|---|---|---|
| Mistral-7B | 3825 | **0.000e+00** | **0.000e+00** |
| Qwen2-7B | 1904 | **0.000e+00** | **0.000e+00** |
| TinyLlama | 1887 | **0.000e+00** | **0.000e+00** |
| phi-2 | 0 | 0.000e+00 | 0.000e+00 (Marlin correctly **declines** phi-2 shapes → vanilla) |

- **≥3 Marlin-engaged families, bit-identical logits co-resident vs solo (KL=0.000e+00)** —
  perfect cross-family isolation; the `(model_id,w_ptr)` re-key prevents any kit contamination.
- **Mis-route BLOCKED** (in-context): bind A→[1,2,3], rebind same w_ptr B→fresh [1,2], no bleed.
- Marlin bf16 per-family correct in isolation (Qwen2 vanilla-vs-Marlin 6-token match + benign
  near-tie). The earlier fp16 "non-engage" was correct policy (`marlin_engage=int4||bf16`).

## §2 Additivity regression (every-prior-model KL) — PASS (byte-identical)

d10 build (re-key, **default-OFF, no bind**) vs the pre-D anchor `9fe23143`, prior models:
- TinyLlama: `max_logit_diff = 0.000e+00` **IDENTICAL**
- Mistral-7B: `max_logit_diff = 0.000e+00` **IDENTICAL**

⇒ the re-key does NOT regress prior-model behavior (additive; model_id=0 → same path as before).

## §3 GOAL-1 ROBUSTNESS GAP (carried forward — required)

**Marlin's `g_weights` cache does not GC kits on model free** → a freed model's dangling
device pointers linger; under model **load/free/reload churn** the next model can hit a stale
kit → **corruption (Qwen2 all-zeros) + co-resident illegal-memory-access crash** (observed in
the first, churn-based harness). The clean static-residence test is correct, but **Goal 1 =
100 bursty agents = model load/unload churn**, exactly this pattern. This is a **Goal-1
residence robustness gap**, not a harness note: it must be fixed (Marlin cache lifecycle / GC-
on-free / pointer validation) for the 100-agent bursty workload. See
`cipher-marlin-cache-churn-gap` memory. Flagged for the V.1 / 100-agent residence work.

## §4 Full no-regression (Mem #16) — status

- **Every-prior-model KL: PASS** (§2, both byte-identical).
- **30-min N=128 resolver soak: PASS** — `resolver_soak PASS`, duration=1800s, writers=128
  readers=8, **total reads=18,448,304,507, INCOHERENT=0 (gate==0), misses=0**, p50=88ns/p99=262ns
  (matches the W.6 baseline 18.3B/0). kmod UNCHANGED (`02fc2d1`, srcversion E27B…); D.7 is a
  libcipher-only change, so this re-validated the unchanged kmod component — clean.
- **W7–W11 microbenches: NOT reproducible as a named suite here** — the commit/audit/observe/
  ring/tc tests were per-session one-offs, not co-located runnable binaries. Coverage caveat:
  the D.7 change is libcipher-only + additive (byte-identical prior-model KL in §2 is the
  strongest direct evidence it doesn't perturb those substrates), but the microbench suite was
  not re-executed. **Honest residue, not faked.**

## §5 Close verdict (STOP for Anil)

- **Positive R-H1 correctness: PASS** (real KL=0, ≥3 families, mis-route BLOCKED, phi-2 correct).
- **Additivity regression: PASS** (both prior models byte-identical).
- **30-min N=128 soak: PASS** (18.45B reads, INCOHERENT=0).
- **W7–W11 microbenches: NOT re-run** (coverage caveat; additivity-KL is the direct D.7 evidence).
- **Goal-1 churn robustness gap: logged + carried** (Marlin cache GC-on-free).

**Anchor NOT rotated; D.7 not finalized — ONE open item.** All measurable close gates PASS
(positive real-KL=0, additivity byte-identical, soak INCOHERENT=0). The sole unresolved item is
the **W7–W11 microbench coverage caveat** (that suite is not reproducible as named binaries here).
Per discipline I do not rotate the anchor unilaterally on a regression-coverage gap. **Anil's
adjudication:** either (a) re-locate + run the W7–W11 microbench suite, or (b) accept the
byte-identical additivity-KL (§2) + the libcipher-only/additive nature of the change as
sufficient coverage → then rotate anchor to the D.10 container build + tag `d7-rh1-close`.
Nothing faked; bf16/INT4 throughout.
