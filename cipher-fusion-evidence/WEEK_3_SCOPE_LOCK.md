# Week 3 Scope-Lock

**HEADLINE STATUS: SCOPE LOCKED.**

Pre-flight (`WEEK_3_PREFLIGHT.md` md5 `053010728a5e007ab9347bfd621f8773`) findings re-verified on disk; no drift since pre-flight ran. Wave 5 §5.5 W3: 5 VERIFIED + 2 SYNTHESIS-HYPOTHESIS + 0 CONTRADICTION (within W3 scope). 5 steps decomposed with named deliverables, goal traceability, SC6 cadence, and the `CIPHER_DISPATCH_LIVE` env progression (0→0→0→1→1). Total budget ~14-20h.

**Date:** 2026-05-20
**Phase:** CIPHER Re-engineering Plan v1.2.2 §7 Week 3 entry
**Pre-conditions verified:**
  - `cipher_rt_phase4 @ f9c32322` (tag `week-2-complete`)
  - `cipher_kmod @ 0ce4b8e2` (tag `week-2-complete`)
  - `cipher-may13-evidence @ fc8a9ae6` (tag `week-2-complete`)
  - `WEEK_3_PREFLIGHT.md` md5 verified
  - `cipher_rt_dispatch.{cpp,h}` not yet on disk (correct — Step 1 creates)

---

## Part 1 — v1.2.2 §7 Week 3 + Wave 5 §5.5 W3 reconciliation

### v1.2.2 §7 Week 3 (CIPHER_REENGINEERING_PLAN.md:1295-1320)

"**Dispatch routing goes live.**" 5 named changes:

1. New file `cipher_rt_dispatch.cpp` implements the §4.5 dispatch table.
2. Per-kernel flow: CLASSIFY → ORACLE → SUBSTITUTE-table → {Marlin, cuBLAS shim, attn dispatch, PASS_THROUGH}.
3. Marlin lane gated on `kernel_class == LARGE_GEMM` AND tenant has full-GPU primary ctx.
4. VOLT engagement triggers 1200 MHz lock on detect-decode-band.
5. SENSE phase transitions trigger DSM PROPOSE for tool-idle detection.

### Wave 5 §5.5 W3 scope correction (CIPHER_LOGIC_AUDIT_WAVE_5_FUSION_PLAN.md:716-777)

> "Scope this to GEMM only: the attn lane has no substitute actuator today and the LP-2 fix is brand new (landed Week 2). Defer the attn dispatch routing to v1.5 when the first attn substitute actuator (Op-3 SUBSTITUTE for FAVOR+ / FlashSwiftKey) is ready."

**Effect**: Week 3 routing target set is `{Marlin, PASS_THROUGH}` for GEMM-classified kernels. cuBLAS shim and attn dispatch entries in the dispatch table remain wired but route to PASS_THROUGH-equivalent paths in v1. Attn lane goes live v1.5; the LP-2 refactor at Step 1 of Week 2 made this defer safe.

### Synthesis-hypothesis fold-in plan

| # | Wave 5 line | finding | resolution step |
| ---:| --- | --- | --- |
| W3-1 | L729 | "`cipher_rt_classify_observer.c::maybe_handle`" vs Step 4's `observe()` | **Step 2** extends `cipher_rt_classify_observer_observe()` (the actual Step 4 name) with hint-publish behavior. Wave 5 `maybe_handle` interpreted as the function name placeholder; Step 4's `observe` is the canonical handle. |
| W3-7 | L763 | `CIPHER_DISPATCH_LIVE=0` env var not yet defined | **Step 1** adds the env-read at `cipher_rt_dispatch.cpp` init. Default value 0 (Week-2-state PASS_THROUGH). Step 4 flips the default to 1. |

All other Wave 5 W3 named targets VERIFIED on disk (Part 5 re-confirms post-scope-lock).

---

## Part 2 — 5-step decomposition

### Step 1 — `cipher_rt_dispatch.cpp` scaffolding + `CIPHER_DISPATCH_LIVE` env

**Goal mapping:** Goal 3 (MFU foundation), Goal 5 (LD_PRELOAD-only — substrate-as-DSO compatible).
**Estimate:** 2-3h
**SC6 cadence:** TinyLlama baseline + CIPHER-injected (~5 min)
**CIPHER_DISPATCH_LIVE:** **0** (default)

**Named deliverables:**

- New file `cipher_rt_phase4/cipher_rt_dispatch.h` (~80 LOC):
  ```c
  struct cipher_rt_dispatch_decision {
      enum { DEST_MARLIN, DEST_CUBLAS_SHIM, DEST_ATTN, DEST_PASSTHROUGH } dest;
      uint32_t recipe_id;
      uint8_t  reason;
  };

  /* Per-kernel decision from CLASSIFY+ORACLE+registry lookup. v1: GEMM
   * lane only; non-GEMM short-circuits to DEST_PASSTHROUGH. */
  void cipher_rt_dispatch_decide(
      const struct cipher_rt_classify_call *call,
      const struct cipher_rt_classify_out  *out,
      struct cipher_rt_dispatch_decision   *decision);

  /* CIPHER_DISPATCH_LIVE env: 0 (default) -> always DEST_PASSTHROUGH;
   * 1 -> classifier-driven routing. Read once at init via cipher_rt_dispatch_init(). */
  int cipher_rt_dispatch_init(void);
  int cipher_rt_dispatch_live(void);
  ```

- New file `cipher_rt_phase4/cipher_rt_dispatch.cpp` (~150 LOC): table + lookup + env-read. No actuation. Includes:
  - Static `dispatch_live_flag` atomic int read from `CIPHER_DISPATCH_LIVE` env at init.
  - `cipher_rt_dispatch_decide()` returns DEST_PASSTHROUGH if `dispatch_live_flag == 0`; otherwise consults the §4.5 table.
  - `dispatch_table[]`: static 7×N matrix (per OpClass × per ContextHint). v1 populates only the GEMM row.

- `Makefile`: `cipher_rt_dispatch.o` to OBJS; per-source rule mirroring matmul_dispatch.

**Verification gate:** library invariant hold — md5 changes (expected; new TU), CP 5.4 15/15 PASS, TinyLlama SC6 PASS bit-identical in **both** modes (baseline + CIPHER-injected). Substrate is quiescent at runtime (dispatch_table not yet consulted by anyone).

**Rollback target:** `git -C cipher_rt_phase4 reset --hard week-2-complete`. Anchor `libcipher_rt.so.week3_pre` snapshotted before Step 1 starts (recommended).

---

### Step 2 — Observer extension: TLS `substitute_hint`

**Goal mapping:** Goal 2 (100-tenant), Goal 3 (MFU), Goal 4 (Koopman O(1) — published hint is the O(1)-lookup substrate).
**Estimate:** 2-3h
**SC6 cadence:** TinyLlama (~5 min)
**CIPHER_DISPATCH_LIVE:** 0

**Named deliverables:**

- `cipher_rt_phase4/cipher_rt_classify_observer.c`: extend the existing `cipher_rt_classify_observer_observe()` (the Step-4 function name; Wave 5's `maybe_handle` resolves here):
  - After the existing 3 atomic counter increments, if `result == CIPHER_RT_CLASSIFY_HANDLED` AND `out->op_class == 0 (GEMM)`:
    1. Call may13's `cipher_oracle_decide(&g_oracle, &oq)` (decision gate).
    2. If `PERMIT`: call `cipher_registry_lookup(&g_registry, op_class, shape_hash, arch)`.
    3. If registry hit: publish to TLS `substitute_hint = recipe_id`.
  - Hint cleared at the start of each call (or set to MARLIN_SHAPE_UNKNOWN sentinel).

- New TLS storage in `cipher_rt_classify_observer.c`:
  ```c
  static __thread uint32_t t_substitute_hint = MARLIN_SHAPE_UNKNOWN;
  ```

- `cipher_rt_classify_observer.h`: 1-line accessor:
  ```c
  uint32_t cipher_rt_classify_get_substitute_hint(void);   /* TLS read */
  void     cipher_rt_classify_clear_substitute_hint(void); /* TLS clear */
  ```

**Verification gate:** library invariant (md5 changes; new symbols `cipher_rt_classify_get_substitute_hint`, `cipher_rt_classify_clear_substitute_hint`), CP 5.4 15/15 PASS, TinyLlama SC6 PASS. No actuator reads the hint yet — bit-identity must hold.

---

### Step 3 — Marlin actuator hint consumption + VOLT 1200 MHz lock

**Goal mapping:** Goal 1 (tok/W — VOLT lock improves energy consistency), Goal 3 (MFU — Marlin selection becomes hint-guided).
**Estimate:** 3-4h
**SC6 cadence:** TinyLlama (~5 min)
**CIPHER_DISPATCH_LIVE:** 0

**Named deliverables:**

- `cipher_rt_phase4/cipher_rt_marlin_actuator.c` (existing file; extend `maybe_handle_marlin` at L56):
  - At the top of the function (before the existing geometry checks at L100-114), read `cipher_rt_classify_get_substitute_hint()`.
  - If hint != `MARLIN_SHAPE_UNKNOWN` AND hint matches the current call's recipe key:
    - Short-circuit the STABILITY_THRESHOLD observation (the 3-call warmup at the existing geometry check).
    - Increments a new diagnostic counter `g_calls_hint_short_circuit`.
  - Read-only on the hint — Marlin still passes-through if `MARLIN_SHAPE_KNOWN` and gate passes (i.e., the hint is a fast-path indicator, not a routing decision).

- `cipher_rt_phase4/cipher_rt_volt.c`: extend `batch_to_mhz()` at L54-65 with a decode-band 1200 MHz case:
  ```c
  case 2: case 3: case 4:  /* decode batch 1<B<8 — DVFS decode-band */
      return 1200;
  ```
  Wave 5 L736 cites this as "1000 MHz lock" but the v1.2.2 §7 L1303 spec says "1200 MHz lock on detect-decode-band". v1.2.2 takes precedence; 1200 MHz aligns with the cipher-t431-volt-shipped memory's +60% tok/W lift at B=1 decode.

- VOLT engagement trigger: read `cipher_rt_classify_get_substitute_hint()` — if hint != MARLIN_SHAPE_KNOWN AND op_class indicates decode pattern, call `cipher_rt_volt_lock(1200)`.

**Verification gate:** library invariant, CP 5.4 PASS, TinyLlama SC6 PASS, Mistral SDPA smoke (validates VOLT path doesn't crash). Hint consumption is read-only; bit-identity must hold.

---

### Step 4 — `CIPHER_DISPATCH_LIVE=1` flip + SENSE→DSM PROPOSE wiring

**Goal mapping:** Goal 1, Goal 2, Goal 3, Goal 4 — the integration step.
**Estimate:** 4-6h
**SC6 cadence:** TinyLlama (~3 min) + **Mistral-7B (~10 min, load-bearing)**
**CIPHER_DISPATCH_LIVE:** **1** (the flip)

**Named deliverables:**

- `cipher_rt_phase4/cipher_rt_dispatch.cpp`: flip the default behavior:
  - `cipher_rt_dispatch_live()` returns 1 if `CIPHER_DISPATCH_LIVE` env unset OR set to non-"0" value. (Was 0-default Steps 1-3; now flips to 1-default.)
  - `CIPHER_DISPATCH_LIVE=0` becomes the explicit-rollback env value (matches v1.2.2 §7 L1314).

- `cipher_rt_phase4/cipher_cupti.c`: in the existing CUPTI callback (post-Step-6), after `cipher_rt_classify_observer_observe()`:
  - Call `cipher_rt_dispatch_decide(&call, &out, &decision)`.
  - Stash `decision.recipe_id` in the same TLS used in Step 2 (substitute_hint shadows the v1.2.2 §4.5 table output).

- `cipher_kmod/cipher_cp54_sched.c`: extend COMPACT_MIGRATE handler (the existing `nr 20`-handler near L794) with a SENSE-input path:
  - New ioctl `CIPHER_CP54_PROPOSE_SENSE_MIGRATE` (proposed nr=26; next-free) takes a `struct cipher_sense_signal { uint32_t tenant_id, sense_band; uint8_t reason; }` payload.
  - Pushes a migrate proposal to the existing rate-limited queue (`cipher_cp54_mig_ratelimit_ms = 10000`).
  - **Decision**: take nr=26 OR fold into existing nr 20 with an extension flag. The brief surfaces both; Step 4 picks one.

- `cipher_rt_phase4`: new userspace caller pushing SENSE signals (likely inside `cipher_rt_classify_observer_observe()` on phase transition; or a separate `cipher_rt_sense_proxy.c`).

**Verification gate:** **Mistral-7B SC6 PASS** (load-bearing). Both modes (baseline + CIPHER-injected) must show `torch.equal()` for all 4 consumers. If mismatch: STOP, surface as ADJUDICATION REQUIRED (may be numerical-equivalent-not-bit-identical, may be real routing bug; gate triggers full investigation).

**Rollback target if Step 4 fails Mistral SC6**: `git -C cipher_rt_phase4 reset --hard week-3-step-3-marlin-hint-consume` (Step 3 anchor); restore week3_pre `libcipher_rt.so` if needed.

---

### Step 5 — Closeout

**Goal mapping:** integration milestone.
**Estimate:** 1-2h
**SC6 cadence:** final Mistral-7B confirmation (~10 min).
**CIPHER_DISPATCH_LIVE:** 1 (Step 4's flip persists).

**Named deliverables:**

- `week-3-complete` tag on all 3 trees (cipher_rt_phase4, cipher_kmod, cipher-may13-evidence).
- `WEEK_3_CLOSEOUT.md` mirroring WEEK_2_STEP_7_CLOSEOUT.md structure (8 parts).
- `WAVE_5_S5_5_W3_VERIFICATION.md` only if any Wave 5 W3 finding differs from pre-flight (Part 2 W3-1/W3-7 inline resolutions documented; otherwise no new doc).
- Bench numbers captured for the v1.2.2 §7 Week 3 behavioral tests (WL01 decode tok/W, WL03 prefill MFU, WL05 multi-tenant) — recommended, not load-bearing for the step itself.

**Verification gate:** Mistral-7B SC6 PASS confirmed twice (Step 4 + Step 5); CP 5.4 15/15 PASS; W1 regression PASS.

---

### Total budget

| step | estimate | SC6 overhead | tagged | live flag |
| ---:| ---:| ---:| --- | ---:|
| 1 dispatch.cpp scaffolding | 2-3h | +5 min | `week-3-step-1-dispatch-scaffold` | 0 |
| 2 observer hint publish | 2-3h | +5 min | `week-3-step-2-classify-hint-publish` | 0 |
| 3 Marlin hint + VOLT | 3-4h | +5 min | `week-3-step-3-marlin-hint-consume` | 0 |
| 4 live flip + SENSE→DSM | 4-6h | +15 min | `week-3-step-4-dispatch-live` | **1** |
| 5 closeout | 1-2h | +10 min | `week-3-complete` (all 3 trees) | 1 |
| **total** | **12-18h** | **~40 min** | — | — |

Plan reserve: ~2h buffer for SC6 first-run setup, model-cache warmup, and ad-hoc smoke retries. Realistic total: **14-20h**.

---

## Part 3 — Goal traceability matrix

| step | Goal 1 (tok/W) | Goal 2 (100-tenant) | Goal 3 (MFU) | Goal 4 (Koopman O(1)) | Goal 5 (LD_PRELOAD-only) |
| ---:| --- | --- | --- | --- | --- |
| 1 — dispatch.cpp scaffold | foundation | foundation | foundation | — | preserved (DSO-compat) |
| 2 — observer hint publish | — | hint enables per-tenant routing | hint enables shape-aware actuator selection | published recipe_id = O(1) lookup substrate | preserved |
| 3 — Marlin hint + VOLT lock | **VOLT 1200 MHz lock = +14% tok/W per cipher-t43-envelope** | — | **Marlin shape short-circuit = warmup elimination** | hint consumption demonstrates O(1) | preserved |
| 4 — live flip + SENSE→DSM | full DVFS engaged | DSM PROPOSE enables cross-tenant SM migration | classifier-driven Marlin selection | full O(1) chain | preserved |
| 5 — closeout | integration confirmed | integration confirmed | integration confirmed | integration confirmed | preserved |

**Every step serves at least one goal.** No step needs removal.

---

## Part 4 — SC6 gating cadence

### Per-step SC6 invocations

```bash
# Steps 1-3 (Marlin still uses static priority; bit-identity must hold under
# both modes since classify substrate is observe-only at this point)
cd /home/ubuntu/cipher-fusion-evidence/phase_c

# Mode A — baseline (vanilla torch, no CIPHER)
python3 sc6_run.py TinyLlama shared
# Mode B — CIPHER-injected
CUDA_INJECTION64_PATH=$(realpath /home/ubuntu/cipher_rt_phase4/libcipher_rt.so) \
    python3 sc6_run.py TinyLlama shared

# Expected pass output (sc6_*_result.json):
#   "all4_fwd1_bit_identical": true
#   "all4_fwd2_bit_identical_after_producer_death": true
```

### Step 4 SC6 — load-bearing

```bash
# TinyLlama first (fast smoke; ~3 min)
python3 sc6_run.py TinyLlama shared
CUDA_INJECTION64_PATH=... python3 sc6_run.py TinyLlama shared

# Mistral-7B (load-bearing; ~10 min)
python3 sc6_run.py Mistral-7B shared
CUDA_INJECTION64_PATH=... python3 sc6_run.py Mistral-7B shared

# Both must produce all4_fwd1_bit_identical=true AND
# all4_fwd2_bit_identical_after_producer_death=true.
```

### Failure handling

| failure | when | response |
| --- | --- | --- |
| TinyLlama SC6 PASS in baseline, FAIL under CIPHER-injection (Steps 1-3) | substrate is supposed to be observe-only | **STOP** at the step's gate, surface as CONTRADICTION; the substrate should not change semantics yet |
| Mistral-7B SC6 PASS in baseline, FAIL under CIPHER-injection (Step 4 only) | dispatch routing flipped live; bit-identity must hold or Step 4 fails | **STOP** and surface as ADJUDICATION REQUIRED; two sub-failures possible: (a) real routing bug → rollback to Step 3; (b) numerical-equivalent-not-bit-identical (TF32 vs FP16 path swap) → may be acceptable with explicit downgrade to NUMERICAL-TOLERANCE gate, but only with user adjudication |
| SC6 PASS in CIPHER-injection but baseline FAILS | baseline broken (unlikely; same model, no env change) | re-run; if persistent, investigate model file / torch version |

---

## Part 5 — Pre-emptive on-disk verification (re-confirm pre-flight)

Verified on disk **post-scope-lock**, no drift since pre-flight `f47e54ba`:

| Wave 5 ref | target | path:line | status |
| --- | --- | --- | --- |
| L729 | `cipher_rt_classify_observer_observe()` (Wave 5 calls it `maybe_handle`) | `cipher_rt_classify_observer.c:40` | VERIFIED — Step 4 function name; Step 2 extends it |
| L732 | `maybe_handle_marlin` | `cipher_rt_marlin_actuator.c:56` (registered at `:173`) | VERIFIED |
| L770 | Marlin shape gate (Wave 5 "B<8") | `cipher_rt_marlin_actuator.c:100-114` (MARLIN_MAX_M_GATE + N/K<1024) | VERIFIED (semantic match; line numbers stable from pre-flight) |
| L736 | `batch_to_mhz` LUT | `cipher_rt_volt.c:55-63` | VERIFIED (1000/1600/1980 MHz LUT; Step 3 adds 1200 MHz decode-band) |
| L739 | COMPACT_MIGRATE nr 20 | `cipher_cp54_sched.c:794` | VERIFIED |
| L775 | `cipher_cp54_mig_ratelimit_ms = 10000` | `cipher_cp54_sched.c:97` | VERIFIED |
| L763 | `CIPHER_DISPATCH_LIVE` | not on disk anywhere | SYNTHESIS-HYPOTHESIS confirmed; Step 1 adds |
| n/a | `cipher_rt_dispatch.{cpp,h}` | does NOT exist | correct — Step 1 creates |

**Drift: NONE.** Pre-flight findings stable. SCOPE LOCKED.

---

## Part 6 — Open items deferred

Items NOT in Week 3 scope, surfaced for visibility:

| item | source | target | reason for defer |
| --- | --- | --- | --- |
| `cipher_audit.cpp` retirement target absent | `WAVE_5_S5_5_VERIFICATION.md` C3 | Week 4 entry-window | Wave 5 W4 work; W3 doesn't touch AUDIT |
| `cipher_partition_slot[32]` naming + count drift | `WAVE_5_S5_5_VERIFICATION.md` C4 | Week 4 (LP-8 retirement) | LP-8 retirement is W4 deliverable |
| Multi-tenant ADD push semantics for classify_stats | Week 2 Step 6 finding | CP 5.5 (Week 13-14) | Currently SET is sufficient for single-tenant |
| Partial-batch flush loss at process exit | Week 2 Step 6 G.3 honest note | CP 5.5 multi-tenant work | Acceptable for v1.2.2 single-tenant; ~256-launch ceiling on loss |
| nvcc build non-determinism | Week 2 Step 3 finding | ops hardening (post-Week-5) | Functional invariants are the binding check; reproducibility is convenience |
| Sites 1+2 cuBLAS/SDPA CLASSIFY direct wiring | Week 2 Step 6 pre-flight | v1.5 (post-cuBLAS-actuator) | Geometry-mismatch persists; CUPTI catches downstream launches |
| Attn dispatch routing | Wave 5 §5.5 W3 L724 | v1.5 (first attn substitute actuator) | LP-2 Step-1 defensive defer keeps trampolines correct |
| Llama-3-8B model fetch (Wave 5 invariant naming) | Wave 5 W3 L745 | not required for Week 3 | SC6 uses Mistral-7B; more representative anyway |

**No new items surfaced during scope-lock.** All deferrals are pre-existing from Week 2 closeout or pre-flight.

---

## Discipline notes

- Pure paperwork. No source modifications.
- `WEEK_3_PREFLIGHT.md` md5 verified before drafting (`05301072...`).
- 5-step decomposition aligns with WEEK_3_PREFLIGHT.md Part 5 recommendation.
- `CIPHER_DISPATCH_LIVE` env progression locked: 0 (Steps 1-3) → 1 (Step 4+).
- SC6 cadence locked: TinyLlama per-step + Mistral-7B per-step-4 + closeout.
- All 5 verified Wave 5 W3 targets re-checked on disk post-scope-lock; no drift.
- 0 CONTRADICTION; 2 SYNTHESIS-HYPOTHESIS resolved via Step 1 (env) and Step 2 (function name).

---

## Awaiting

Week 3 Step 1 brief, drafted with:
- `cipher_rt_dispatch.{cpp,h}` scaffolding (~230 LOC across both files)
- `CIPHER_DISPATCH_LIVE` env-read at init; default 0
- Makefile extension (1 OBJS + 1 rule)
- `libcipher_rt.so.week3_pre` anchor snapshot before edits
- TinyLlama SC6 gate (both modes)
- CP 5.4 regression hold
- Tag `week-3-step-1-dispatch-scaffold` on PASS
