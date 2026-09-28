# Geometry-recognized substitution — FILE MANIFEST + CAPTURE RESULTS (Step 1)

**Date:** 2026-05-31. **Type:** build-in-progress; this reports the FILE MANIFEST + the Step-1 capture
(per "report manifest + capture first, then build"). No commit yet; staging `.so` `6df65682` (= marlin-gc
`944f5706` + the default-OFF capture instrumentation). Anchors UNCHANGED. STOP at the close gate for tag.

**Headline:** the Step-1 capture **empirically PROVES the geometry recognizer fires on real vLLM FA,
name-free** (2 FA kernels → ATTENTION, 1 GEMM → GEMM, by grid/block/shmem alone). But the capture ALSO
clarified the WIRE step (Step 3): the actuators' **engage gates are already framework-agnostic** (cuBLAS
dtype/shape, clock, plugin — not names), so the name-keying the WIRE replaces is the **attention 6-pattern
INTERCEPT** (`_vllm_fa3_C`), and recognized-attention→KV-dedup is a category mismatch. So the WIRE delivers
**geometry-keyed attention INTERCEPTION (replacing the symbol-keyed 6-pattern), observe/passthrough** — the
substrate-integrity win — with attention *substitution* staying v1.5. **Surfacing this for Anil before
building Step 3** (the capture reshaped its scope).

---

## FILE MANIFEST

### Step 1 — CAPTURE instrumentation (BUILT; additive; default-OFF)
| File | Action | additive vs modifies-shipped |
|---|---|---|
| `cipher_rt_geom_capture.h` / `.c` | **CREATE** | additive (new TU) — env-gated per-fn geometry+verdict dump |
| `cipher_cupti.c` | **AMEND** | modifies-shipped, but **additive behavior**: 1 gated call after `classify_route` (`:200`) + 1 include; default-OFF ⇒ byte-identical |
| `Makefile` | **AMEND** | additive (add `cipher_rt_geom_capture.o` to OBJS) |
| `geom_capture_run.py` / `geom_capture_torch.py` / `geom_capture_analyze.py` | **CREATE** (harness) | evidence; not shipped |

### Step 2 — VALIDATE/TUNE (capture shows NO tuning needed; see results)
| File | Action |
|---|---|
| `include/may13/cipher_classify.hpp` | **UNCHANGED** — the existing ATTENTION/GEMM rules already fire correctly on real vLLM geometry (2/2 FA → ATTENTION via the FMHA rule `:108`; GEMM → GEMM). No threshold tuning required. (Minor: vLLM RMSNorm binned ELEMENTWISE not REDUCTION — not blocking; only matters if RMSNorm-routing were in scope.) |

### Step 3 — WIRE (scope CLARIFIED by the capture; pending Anil — see below)
| File | Action (provisional) |
|---|---|
| `cipher_rt_attn_6pattern.c` and/or a new `cipher_rt_geom_attn_route` | **AMEND/CREATE** | make attention INTERCEPTION geometry-triggered (replace the `_vllm_fa3_C` symbol GOT-patch), default-OFF, observe/passthrough. (NOT launch-level substitution — operand gap; substitution kernel is v1.5.) |
| `cipher_rt_classify_substrate.cpp` | (only if a substitution-routing actuator is added) | default-OFF |

---

## CAPTURE RESULTS — the three answers (REAL vLLM 0.20.2 FLASH_ATTN, TinyLlama prefill+decode)

Capture: `CUDA_INJECTION64_PATH`=staging `.so`, `CIPHER_GEOM_CAPTURE=1`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`,
`VLLM_USE_DEEP_GEMM=0` (the host vLLM needed these — DeepGEMM/FP8 init otherwise crashes). 50 unique
kernels captured with geometry + the live geometry-classifier verdict (`out.op_class` from
`classify_route`). Names absent (vLLM V1 runs kernels in a worker subprocess; the name table dumped from
the main process — a known worker-subprocess split; the geometry + verdict ARE the worker's, captured
per-process).

**Q1 — does the ATTENTION band fire on real vLLM FA geometry? YES (validated).**
- 2 kernels: `grid=(132,1,1) block=(256,1,1) shmem=98304` (96 KB) and `…shmem=109568` (107 KB) → classified
  **ATTENTION**. These are the FA kernels (FLASH_ATTN backend confirmed active; no other kernel in a
  TinyLlama forward has ≥96 KB shmem — the 42 elementwise have shmem=0). Matched the **FMHA rule**
  (`cipher_classify.hpp:108`: `shmem≥49152 && bx≤256 && by==1 && grid_aspect≥4`), NOT the FA2/FA3 rule
  (`:102`, `bx==128`) — vLLM FA uses `bx=256`, and the broader rule correctly caught it. **No tuning
  needed.**

**Q2 — is there a name-dependence geometry removes? YES, demonstrably.**
- The 50 verdicts were produced from grid/block/shmem ALONE (the capture carried no names) — yet FA→
  ATTENTION and GEMM→GEMM were correct. So the geometry path classifies **without any name**, and would
  recognize a *renamed* or *non-vLLM* FA kernel of the same geometry that a name-`strstr` ("flash"/"fa3")
  would miss. **The geometry recognizer is genuinely framework-agnostic — proven on real kernels.**
- **CAVEAT the capture surfaced (reshapes Step 3):** CIPHER's actuator ENGAGE decisions are *already*
  name-independent — Marlin engages by cuBLAS dtype/shape, VOLT by clock, KV-dedup by the vLLM KV-alloc
  plugin. The **only symbol-keyed thing is the attention 6-pattern INTERCEPT** (`_vllm_fa3_C` hardcode) —
  which is observe/passthrough, not an engage gate. So "geometry-gate the engage decision" is largely a
  no-op (engagement is already agnostic); the real WIRE target is **replacing the 6-pattern's symbol-keyed
  attention interception with geometry-keyed interception**.

**Q3 — is nvjet (cuBLAS-bypass prefill GEMM) recognizable? PARTIAL — needs a prefill-specific capture.**
- This run captured 1 GEMM-verdict kernel (`block=512 shmem=93184`) — GEMMs ARE geometry-recognizable. But
  no isolated **nvjet** prefill GEMM was labeled here (TinyLlama decode via FLASH_ATTN; names absent). The
  scoped nvjet follow-on (is the nvjet launch geometry-distinct AND its operands extractable from the
  opaque launch args?) needs a prefill capture + a param-layout probe — **gated, not assumed**, and only
  relevant to the prefill→v1.5 path anyway.

---

## WHAT THE CAPTURE PROVED vs WHAT THE WIRE CAN DELIVER

- **PROVED:** geometry-based recognition of attention + GEMM **works on real vLLM kernels, name-free** (the
  framework-agnostic recognizer is real, validated — the prior "research/defer" was a phantom; the verdict's
  capture caveat is now closed). The recognizer + capture + classifier are BUILT.
- **THE WIRE (Step 3), scope clarified by the capture:** it delivers **geometry-keyed attention
  INTERCEPTION**, replacing the `_vllm_fa3_C` symbol GOT-patch — so attention detection/observation is
  framework-agnostic (the substrate-integrity win the audit demanded). It is **observe/passthrough**:
  - attention SUBSTITUTION (the lossy/optimized kernel + KL gate) stays **v1.5** (W.5 wall);
  - recognized-attention → KV-dedup is a **category mismatch** (KV-dedup is plugin-triggered at KV-alloc,
    not by an attention launch) — so it is NOT a wiring target;
  - launch-level GEMM→Marlin is **out** (operand gap; Marlin already engages at cuBLAS, agnostically).
- So the honest Step-3 deliverable is: **the attention intercept becomes geometry-keyed (works on
  SGLang/TGI/renamed kernels), de-coupled from `_vllm_fa3_C`** — composing with the v1.5 attention
  substitution. That IS the substrate-integrity keystone; it is NOT new substitution.

## STEP 3 — BUILT: geometry-keyed attention interception (CONFIRMED scope)

**Built** (staging `.so` **`7cdc68ef`** = marlin-gc `944f5706` + default-OFF geom capture + geom-attn
detection). A framework-agnostic attention DETECTION counter keyed on `classify_launch==ATTENTION`
(geometry, no symbol), env `CIPHER_GEOM_ATTN` default-OFF, **observe-only** (a counter; never
substitutes — substitution is v1.5). Replaces the `_vllm_fa3_C` 6-pattern *as the detection mechanism*;
the 6-pattern stays as the v1.5 in-path substitution hook.

**Files:** `cipher_rt_geom_capture.{c,h}` AMEND (additive: the geom-attn counter + `CIPHER_GEOM_ATTN`
gate + accessors `cipher_rt_geom_attn_intercepts()`/`cipher_rt_geom_launches()` + an atexit count log);
`cipher_cupti.c` (the gated call from Step 1, no new change); `cipher_rt_attn_6pattern.c` UNCHANGED;
`Makefile` (TU added in Step 1). No W7-12 TU, no kmod.

### Validation (close gate)
- **AGNOSTIC PROOF — PROVEN** (`geom_attn_torch.json`): on **torch-HF SDPA** (a DIFFERENT attention lib),
  single-process, **geom_attn_intercepts = 10,656** (geometry detector fires) while the symbol-keyed
  6-pattern **= 0** (`p1_fa2=p2_fa3=p5_mla=p6_paged=0` — the `_vllm_fa3_C` lib never loads in a torch
  process). Detection works where the symbol hardcode is blind ⇒ **framework-agnostic, name-free.**
- **Selective (no false-positive flood):** geom_attn / total launches = **6.1%** (10,656 / 175,884) —
  fires on the attention launches, not the GEMMs/norms.
- **vLLM RECOGNITION — PROVEN** (Step-1 capture): geometry → ATTENTION on the 2 real vLLM FA kernels
  (`block=256, shmem 96-107 KB`), via CUPTI, in the V1 worker.
- **vLLM per-call PARITY — partially blocked (infra, not defect):** the V1 **worker subprocess** tears
  down before the atexit counter flushes, and the symbol 6-pattern GOT-patch **needs the WEEK_14
  worker-init hook to arm** in the worker (so `attn_p2=0` there — not because there's no FA). The
  **CUPTI-geometry detection works in the worker without that hook** (Step-1 capture proves it) — which
  *reinforces* agnosticism: geometry detection is more robust than the symbol GOT-patch. Exact per-call
  count in the worker is the documented worker-teardown obstacle.
- **byte-identical-OFF (Mem #11 HARD STOP) — PASS 4/4** (`off_byte_identical_geom`: vanilla vs both-geom-
  envs-unset, max_abs_diff 0.0). Installing the instrumentation changes nothing when OFF.
- **30-min SOAK on the universal path (GEOM_ATTN=ON) — PASS** (`geom_attn_soak.json`): 1100 decode gens /
  1800 s, **geom_attn counter +3,907,200** (heavily exercised, not silently off) + 64.3 M launches counted,
  **nan=0, incoherent=0, memory FLAT (0.0 MiB checkpoint spread), no crash.** Overhead GEOM_ATTN ON vs OFF
  = **−0.02%** (`geom_ovh_{on,off}.json`: 942.66 vs 942.85 ms — the ~1 ns/launch claim verified, within
  noise). The binding non-regression on the every-launch path is PROVEN by measurement, not asserted.
- **Regression — observe-only ⇒ output-neutral by construction:** the geom-attn path is a gated counter
  that touches NO actuator and NEVER substitutes; with the gate OFF it early-outs (byte-identical 4/4),
  and even ON it only increments a counter — so every prior actuator's count + output are structurally
  unchanged (no per-actuator re-run needed; the marlin-gc full regression + soak already cover the
  substrate). Fail-safe holds: geometry-UNKNOWN/non-attention → not counted (6.1% selectivity confirms).

### Verdict
**Geometry-keyed attention DETECTION is built + validated: framework-agnostic (torch-HF 10,656 vs
symbol 0), selective, vLLM-recognizing, byte-identical-OFF 4/4.** This removes the last vLLM symbol-
coupling the substrate-integrity audit flagged *as the detection mechanism* — attention detection now
works on SGLang/TGI/renamed FA by geometry. Attention *substitution* remains v1.5 (W.5). The one
unmeasured item (exact vLLM per-call parity) is the documented V1-worker-teardown obstacle, with
recognition itself proven by the capture.

## STOP — for Anil (the capture reshaped Step 3)

Step 1 (capture) is DONE and POSITIVE: **agnostic geometry recognition is validated on real vLLM FA.**
Before building Step 3, confirm its scope given the capture's findings:
1. **WIRE = geometry-keyed attention INTERCEPTION** (replace `_vllm_fa3_C` GOT-patch with a geometry trigger;
   observe/passthrough; default-OFF) — the framework-agnostic-detection keystone. **Recommended.**
2. Attention *substitution*, recognized→KV-dedup, and launch→Marlin are **NOT** Step-3 (v1.5 / category-
   mismatch / operand-gap, respectively) — per the capture.
3. The nvjet follow-on is gated on a prefill capture (prefill→v1.5).

Or: if "geometry-keyed engage-gating" was the intent, the capture shows engagement is already agnostic
(no-op refactor) — so the keystone really is #1 (the attention intercept). Read-only on the analysis; the
capture instrumentation is built (default-OFF, byte-identical). No further build until Anil confirms the
Step-3 scope. Commit as Anil, no co-author, when authorized.
