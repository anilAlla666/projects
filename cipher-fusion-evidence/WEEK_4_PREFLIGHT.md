# Week 4 Pre-flight — Bundled Paperwork

**HEADLINE STATUS: WEEK 4 ENTRY READY (with 1 PARTIAL-READY deferred for adjudication).**

5 paperwork tasks bundled. v1.2.2 §7 + Wave 5 §5.5 W4 align cleanly. C3 cipher_audit.cpp resolves to **RESOLVED-RETIREMENT-ALREADY-DONE** (file never existed; retirement is structurally already in effect). LOOP + PIPELINE port closures are **near-leaf** (only system undefs + already-ported cipher_sense_get_type). Sub-4 measurement infra is buildable (~50 LOC). Real oracle_decide wiring is feasible via selective init (~20 LOC).

| section | finding | impact |
| --- | --- | --- |
| Part 1 | §7 W4 = 12 .cpp ports + Prometheus + LP-8 retirement | Implementation surface defined |
| Part 2 | Wave 5 W4 sweep: 3 VERIFIED + 1 SYNTHESIS-HYPOTHESIS + 2 known-CONTRADICTION (C3, C4) | 0 new contradiction |
| Part 3 | C3: cipher_audit.cpp doesn't exist; retirement structurally already done | **RESOLVED** — Week 4 documents the no-op |
| Part 4 | LOOP=COMPLEMENTARY, PIPELINE=INDEPENDENT vs Sub-4 | All three ship; no conflict |
| Part 5 | Sub-4 measurement infra: ~50 LOC `/proc/cipher/sense_transitions` | Step 6 of Week 4 (additive) |
| Part 6 | Real oracle_decide via selective init (skip CUDA paths) | **FEASIBLE** ~20 LOC; replaces hardcoded permit=1 |
| Part 7 | 6-step Week 4 decomp; ~25-35h budget | Within scope |
| Part 8 | Entry-ready; 1 deferred decision (oracle wiring) | Adjudicate in scope-lock |

**Date:** 2026-05-20. **Read-only diagnostic.** No source modifications.

---

## Part 1 — v1.2.2 §7 Week 4 baseline (verbatim from CIPHER_REENGINEERING_PLAN.md:1321-1356)

### Goal (L1323)

"Port the observability tier. Per-tenant billing/fairness/audit becomes customer-facing."

### Named .cpp ports (L1325-1338)

| # | source (cipher-may13-evidence/src/) | destination (cipher_rt_phase4/) | note |
| ---:| --- | --- | --- |
| 1 | `cipher_audit.cpp` | `cipher_rt_audit.{c,h}` (already exists; Wave 5 says **RETIRE**) | C3 — see Part 3 |
| 2 | `cipher_trace.cpp` | `cipher_rt_trace.cpp` | |
| 3 | `cipher_receipt.cpp` | `cipher_rt_receipt.cpp` | |
| 4 | `cipher_carbon.cpp` | `cipher_rt_carbon.cpp` | |
| 5+6 | `cipher_fairness.cpp` + `cipher_fairness_shm.cpp` | `cipher_rt_fairness.cpp` | two files merge |
| 7 | `cipher_guard.cpp` | `cipher_rt_guard.cpp` | |
| 8 | `cipher_comply.cpp` | `cipher_rt_comply.cpp` | |
| 9 | `cipher_loop.cpp` | `cipher_rt_loop.cpp` | Op 26 LOOP |
| 10 | `cipher_pipeline.cpp` | `cipher_rt_pipeline.cpp` | Op 27 PIPELINE |
| 11 | `cipher_determinism.cpp` | `cipher_rt_determinism.cpp` | |
| 12 | `cipher_continuity.cpp` | `cipher_rt_continuity.cpp` | |
| 13 | `cipher_pulse.cpp` | `cipher_rt_pulse.cpp` | |
| — | PREDICT, SHIELD, SUSTAIN, THERMOSTAT | (defer to v1.5) | partial ops |

12 source files → 11 rt_phase4 targets (FAIRNESS merge).

### Prometheus exporter additions (L1340-1344)

- `cipher_tenant_fairness_quota{tenant=X}`
- `cipher_tenant_carbon_grams_co2{tenant=X}`
- `cipher_tenant_receipt_hash{tenant=X}` (compact form)
- `cipher_tenant_session_band{tenant=X}` (HUMAN/AGENT/BATCH)

### Behavioral tests (L1346-1350)

- Per-tenant FAIRNESS quota debits visible at `/proc/cipher/fairness`
- AUDIT chain advances per launch (kmod nr 8 telemetry)
- TRACE bounded buffer rotates without drops at sustained load
- RECEIPT per-session HMAC verifiable

### Risk register (L1354-1356)

- R-W4.1 LOW: observer overhead < 1 µs/launch combined
- R-W4.2 LOW: Prometheus cardinality (1000 series at N=100 tenants)

---

## Part 2 — Wave 5 §5.5 W4 verification sweep

Wave 5 §5.5 W4 at lines 780-859 (CIPHER_LOGIC_AUDIT_WAVE_5_FUSION_PLAN.md). Adds the following beyond v1.2.2 §7:

| # | Wave 5 line | Target | Status | Notes |
| ---:| --- | --- | --- | --- |
| W4-1 | L786 | `cipher-may13-evidence/src/cipher_audit.cpp` (retirement target) | **CONTRADICTION (known)** | File DOES NOT EXIST. See Part 3 — resolves to no-op. |
| W4-2 | L789-799 | 11 `cipher_<op>.cpp` source files in evidence | **VERIFIED** | All 11 .cpp present at expected paths; line counts 200-1000 each (per `WAVE_5_S5_5_VERIFICATION.md` Week 4 row L789) |
| W4-3 | L803 | `cipher_10ops.h` 65536-entry SPMC ring | **VERIFIED** | `CIPHER_RING_SIZE 65536u` at `cipher_10ops.h:72`; ring buf at `:79`; header already ported in Week 2 |
| W4-4 | L804 | `cipher_10ops_impl.cpp::stage1_shadow` | **VERIFIED** (function at `:385`; spawned at `:955` per prior sweep) | Pre-existing concern: cipher_10ops_impl.cpp has 50 cross-TU undefs (Week 3 Step 4 pre-flight); porting it brings ~23 transitive sources. Wave 5 says it "must port" but doesn't measure the closure cost. See Part 7 for decision. |
| W4-5 | L810 | `cipher_dev_request_sm_partition` handler in `cipher_dev.c` | **VERIFIED** | at `cipher_dev.c:107`; already returns -ENOSYS per `:110` |
| W4-6 | L812 | slot array `cipher_partition_slot[32]` | **CONTRADICTION (known C4)** | Type at `cipher_partition_allocator.c:88`; array is `cipher_slots[CIPHER_PARTITION_SLOTS_MAX]` (33 slots, not 32). Naming + count drift per prior sweep. LP-8 retirement cleanly resolves this since the array is being removed entirely. |
| W4-7 | L814 | `cipher_state_updater.c::cipher_state_updater_fn` | **SYNTHESIS-HYPOTHESIS** | Wave 5 prescribes moving _tick() reclaim semantics into this function. Need to verify the function exists + has appropriate hook points. Not pre-checked yet; deferred to Week 4 LP-8 step. |
| W4-8 | L817-820 | `cipher_pid_stats::sm_partition_mask/count` fields preserved; written by CP 5.4 at `cipher_tenant_snapshot.c:207` | **VERIFIED** (prior sweep) | Fields exist; cipher_set_sm_partition_mask path already populated by CP 5.4 |

**Sweep result: 4 VERIFIED + 1 SYNTHESIS-HYPOTHESIS + 2 known CONTRADICTIONs.** Both contradictions are *self-resolving* — C3 (cipher_audit.cpp) because the file never existed, C4 (slot array) because LP-8 removes it.

---

## Part 3 — C3 cipher_audit.cpp adjudication — RESOLVED

### Evidence

- **Wave 5 says** (L786): "AUDIT: retire `cipher-may13-evidence/src/cipher_audit.cpp` in favor of existing Wave 2 `cipher_rt_audit.{c,h}` (already wired at priority 0 on both substrates)."
- **v1.2.2 says** (L1326): "`cipher_audit.cpp` (may13) — RETIRE in favor of existing `cipher_rt_audit.{c,h}`."
- **On disk** (`find cipher-may13-evidence -name 'cipher_audit*'`): **zero results.** The file does NOT exist.
- **cipher_rt_audit.{c,h}** at `cipher_rt_phase4/cipher_rt_audit.{c,h}` (6.7 KB + 2.9 KB): EXISTS, has been shipped since CP 2.5 / Wave 2, already wired as priority-0 observer on matmul + attn substrates.
- **may13 AUDIT logic**: lives inline in `cipher_10ops_impl.cpp:341-378` (function `audit_chain_update` at L346) with a commented-out call site at `:472-473`.

### Categorization

**RESOLVED-RETIREMENT-ALREADY-DONE.** Two-layer reasoning:

1. The file Wave 5 names as a retirement target does not exist as a standalone file.
2. The AUDIT logic it ostensibly contains (`audit_chain_update`) lives inside `cipher_10ops_impl.cpp` and is NOT ported into cipher_rt_phase4 today (the call site at `:472-473` is commented out anyway).
3. Therefore the retirement is **structurally already in effect** — the production AUDIT is `cipher_rt_audit.{c,h}` (already shipped), and the may13 inline AUDIT (`audit_chain_update`) has never reached cipher_rt_phase4.

### Week 4 action

Document the no-op resolution in the Week 4 closeout. No file to delete; no symbol to retire. The Wave 5/v1.2.2 retirement prescription is a documentation correction: the named .cpp doesn't exist; AUDIT was always going to be cipher_rt_audit, and that's what production uses.

If `cipher_10ops_impl.cpp` is ever ported in a later week (W4-4 says "must port" per Wave 5), the inline `audit_chain_update` block stays commented-out as it is in evidence today. Standalone AUDIT survives at cipher_rt_audit.

---

## Part 4 — LOOP / PIPELINE inspection

### Part 4.1 — Source presence

| file | exists? |
| --- | --- |
| `cipher-may13-evidence/include/cipher_loop.h` | YES |
| `cipher-may13-evidence/src/cipher_loop.cpp` | YES |
| `cipher-may13-evidence/include/cipher_pipeline.h` | YES |
| `cipher-may13-evidence/src/cipher_pipeline.cpp` | YES |

### Part 4.2 — LOOP public API (`cipher_loop.h`)

```c
/* Op 26 LOOP — Agentic runaway detection (Stage 1 monitor).
 * Default OFF: env var CIPHER_LOOP=on enables. Benefits from CIPHER_SENSE=on. */

int      cipher_loop_init(void);
void     cipher_loop_observe(const CipherRingEntry* ev);
int      cipher_loop_get_score(uint64_t fingerprint);    /* 0..3 runaway score */
void     cipher_loop_report(void);
unsigned cipher_loop_session_count(void);
unsigned cipher_loop_runaway_count(void);
```

**Detection (3 signals):**
- S1 Shape-cycle repetition (rolling 64-event window, period ≤8 sustained for 3 cycles)
- S2 Burn rate ≥ 4× baseline (decodes/sec)
- S3 Prefill drought (>500 events since last prefill AND S1 active)

Score = S1+S2+S3 (0..3); score ≥ 2 → `CIPHER_LOOP_RUNAWAY`.

**v1 action**: writes `CIPHER_BAND_DEMOTED` hint (mirrors SHIELD's `BAND_PROTECTED`). ARBITRATE logs only; real SM throttle is v2.

### Part 4.3 — PIPELINE public API (`cipher_pipeline.h`)

```c
/* Op 27 PIPELINE — Multi-agent session correlation (Stage 1 observer). */

int      cipher_pipeline_init(void);
void     cipher_pipeline_observe(const CipherRingEntry* ev);
unsigned cipher_pipeline_session_count(void);
unsigned cipher_pipeline_edge_count(void);
void     cipher_pipeline_report(void);
```

**Detection**: per-session bounded shape set (first 32 distinct `params_hash`); report-time pairwise Jaccard similarity → upstream/downstream edges (sim ≥ 0.5).

### Part 4.4 — Undef closures

`cipher_loop.cpp`: **8 undef, 2 non-system** (`__fprintf_chk`, `__stack_chk_fail` — both fortified-libc). **Self-contained.**

`cipher_pipeline.cpp`: **9 undef, 3 non-system** (`__fprintf_chk`, `__stack_chk_fail`, **`cipher_sense_get_type`**). The third is already exported by libcipher_rt.so (ported Week 2 Step 2). **Self-contained on current closure.**

### Part 4.5 — Categorization vs Sub-4 (invented transition wrapper)

| op | category | rationale |
| --- | --- | --- |
| LOOP | **COMPLEMENTARY** | Detects AGENT *runaway* (iteration patterns + burn rate + prefill drought). Different signal from Sub-4's session-class *transitions*. Sub-4 says "this tenant became BATCH"; LOOP says "this AGENT session is iterating dangerously." |
| PIPELINE | **INDEPENDENT** | Multi-agent session correlation + edge detection. Different goal entirely (pipeline-graph topology). |

### Part 4.6 — Recommendation

Port both alongside Sub-4. They produce different signals; no overlap. **Sub-4 stays production per Q3=a.** LOOP optionally augments Sub-4's proposal stream with `CIPHER_BAND_DEMOTED` hints when runaway detected (Week 4 could wire LOOP's runaway hint into the existing DSM PROPOSE queue; surface for scope-lock).

---

## Part 5 — Sub-4 measurement infrastructure scoping

### Part 5.1 — What exists today

- `/proc/cipher/dsm_proposals` (Week 3 Step 4 Option II-a): emitted proposals (producer-side telemetry, 256-slot ring, `tenant`/`source`/`target`/`reason`/`confidence`/`age`).
- `cipher_rt_sense_transition_proposals_queued()` + `_pushed()` diagnostic accessors.

### Part 5.2 — What's missing for tuning

- **Per-tenant transition log**: timestamp + hysteresis state at transition (which counter triggered? was N=8 hit cleanly, or did debounce flap?).
- **False-positive tracking**: was a proposal emitted when no real transition occurred? (would need workload-labeled ground truth)
- **True-positive tracking**: did the wrapper miss a transition? (requires external classifier as oracle)
- **Hysteresis effectiveness**: how often did N=8 prevent flap-triggered proposals?
- **Tool-idle K=100ms accuracy**: did K trigger when agent was actually idle, or when it just happened to gap?

### Part 5.3 — Minimal measurement surface (proposed for Week 4)

Add to `cipher_rt_sense_transition.c` (~30 LOC):

```c
struct sense_transition_log_entry {
    uint64_t timestamp_ns;
    uint32_t tenant_id;
    uint32_t prev_class;
    uint32_t new_class;
    uint32_t stability_count;   /* at moment of transition */
    uint32_t debounce_count;
    uint8_t  reason;            /* TRANSITION_DETECTED or AGENT_IDLE */
    uint8_t  pad[3];
};

static struct sense_transition_log_entry g_transition_log[512];
static atomic_uint g_transition_log_head;
```

Plus a `/proc/cipher/sense_transitions` proc node (~20 LOC kmod-side, mirrors `dsm_proposals` pattern but reads userspace-pushed log via a new ioctl, OR userspace-only emit via `cipher_rt_sense_transition.c`'s own `report()` to `/tmp/sense_transitions.json`).

**Userspace-only emit recommended** for v1 (avoids new ioctl + kmod state). Operator captures /tmp/sense_transitions.json snapshots between workload runs.

### Part 5.4 — Total scope

| component | LOC | est time |
| --- | ---:| ---:|
| Transition log storage in cipher_rt_sense_transition.c | 30 | 30 min |
| Log-emit at transition points (already-existing emit sites) | 10 | 15 min |
| `cipher_rt_sense_transition_report()` /tmp/ JSON dump | 20 | 30 min |
| Wire init/report from cipher_inject.c::cipher_v2_init_body | 5 | 10 min |
| **total** | **~65 LOC** | **~1.5h** |

Standalone Week 4 step (Step 6); doesn't depend on observability tier port.

### Part 5.5 — Tuning workflow (out of scope for Week 4)

1. Run representative workloads (interactive, batch, mixed) with `CIPHER_SENSE=1`
2. Capture `/tmp/sense_transitions.json` snapshots
3. Inspect proposals vs expected behavior
4. Adjust N/M/K thresholds in source; recompile; repeat
5. Multi-week iterative process; not in Week 4 scope

---

## Part 6 — Real `cipher::oracle_decide` deferred-init feasibility

### Part 6.1 — `cipher_init()` body

From `src/may13/cipher_runtime.cpp:24-82`:

```c
int cipher_init(int device_ordinal) {
    cipher_green_ctx_init(...);     // CUDA + cuda_runtime
    cipher_l2_persist_init(...);    // CUDA
    cipher_liquid_state_init(...);  // CUDA (it's a .cu file)
    cipher_telemetry_init(...);     // CUPTI + NVML (non-fatal failure)
    cipher_layer3_init();           // cipher_oracle + cipher_registry + ...
    g_cipher.initialized = true;
    ...
}
```

**Five subsystems**: 4 touch CUDA/NVML/CUPTI (incompatible with deferred-init); 1 (`cipher_layer3_init`) initializes the oracle/registry/recipe table.

### Part 6.2 — oracle_decide state requirements

From `src/may13/cipher_oracle.cpp`:
- L106 (`cipher_oracle_init`): sets `state->initialized = true`
- L270, L310: `if (!state->initialized) return;` — short-circuits on uninit
- L32-33: reads `state->topo_class_counts[]`, `state->total_decisions`
- L37, L46-47, L62-63: reads/writes `state->phase.detected_phase`, `state->phase.phase_entry_step`
- L41-42, L58-60: reads `state->topo_class_counts[5/6]` (training ops) and `[0/3/4]` (inference ops)
- L85: `state->liquid = liquid` — accepts NULL liquid_mgr (L197 then guards `state->liquid && state->liquid->initialized`)

### Part 6.3 — Selective init feasibility

`cipher_oracle_init(state, liquid, cfg)`:
- Takes a `CipherOracleState*` and a `CipherLiquidStateMgr*` (can be NULL)
- Allocates nothing externally; just initializes the state struct in-place
- Does NOT call CUDA, NVML, or kmod
- L106 sets `state->initialized = true`

So we CAN:
1. Declare a `static CipherOracleState g_rt_oracle;` in cipher_rt_phase4 native code
2. At libcipher_rt init time, call `cipher_oracle_init(&g_rt_oracle, NULL, NULL)` — uses defaults, no liquid mgr
3. In the observer's hint-publish block, call `cipher_oracle_decide(&g_rt_oracle, &query)` instead of hardcoded `permit=1`

**Selective-init feasibility: HIGH.** ~20 LOC of new code in cipher_rt_phase4 (state declaration + init call + observer extension).

### Part 6.4 — Three approaches

| approach | LOC | risk | recommendation |
| --- | ---:| --- | --- |
| (a) Call full `cipher_init()` at .so load | 1 | **HIGH** — touches CUDA at static-init; violates deferred-init contract; breaks loader smoke | NOT recommended |
| (b) Selective oracle-only init via `cipher_oracle_init(&local_state, NULL, NULL)` | ~20 | LOW — pure-function init; no external state | **RECOMMENDED** |
| (c) Keep hardcoded `permit=1` indefinitely | 0 | LOW (v1 limitation; Marlin's own gate is binding) | acceptable if Week 4 budget tight |

### Part 6.5 — Week 4 fit

Approach (b) is a clean fit for Week 4. Folds into Step 2 observer extension (replace Step-2's hardcoded permit with real `cipher_oracle_decide(&g_rt_oracle, ...)` call).

---

## Part 7 — Week 4 step decomposition

### Recommended 6-step structure (~25-35h)

| step | name | LOC | est time |
| ---:| --- | ---:| ---:|
| 1 | Selective oracle init + observer permit upgrade (replace hardcoded permit=1 with real oracle_decide) | ~20 | 1-2h |
| 2 | Observability tier ports — Tier A (LOOP, PIPELINE, PULSE, CONTINUITY) | ~400 | 4-6h |
| 3 | Observability tier ports — Tier B (TRACE, RECEIPT, CARBON, FAIRNESS+SHM, GUARD, COMPLY, DETERMINISM) | ~600 | 4-6h |
| 4 | LP-8 allocator retirement (C4 resolves) | ~80 | 2-3h |
| 5 | Prometheus exporter additions (4 new metrics + tenant labels) | ~150 | 3-4h |
| 6 | Sub-4 measurement infrastructure (`/tmp/sense_transitions.json` + tenant log + report) | ~65 | 1.5h |
| 7 | Week 4 closeout (week-4-complete tag, doc) | — | 1-2h |
| — | C3 cipher_audit retirement documentation (folded into closeout) | — | — |
| — | VOLT 1000 vs 1200 MHz adjudication (folded into Step 5 or closeout) | — | — |
| **total** | | **~1315 LOC** | **~17-25h** |

### Step grouping rationale

- **Tier A (Step 2)**: LOOP + PIPELINE + PULSE + CONTINUITY — these are session-level monitors; cleanest port set; LOOP+PIPELINE confirmed self-contained per Part 4. CONTINUITY + PULSE need closure inspection but likely similar pattern.
- **Tier B (Step 3)**: TRACE + RECEIPT + CARBON + FAIRNESS + GUARD + COMPLY + DETERMINISM — these are billing/audit/telemetry; may need shared infrastructure (bounded buffers, HMAC, /proc nodes).
- **Step 1 (oracle)** first so that the entire observability tier can call real `cipher_oracle_decide()` for hint-publish.
- **Step 4 (LP-8)** as standalone — touches kmod cipher_dev.c + cipher_partition_allocator.c; clean ABI work.
- **Step 5 (Prometheus)** at the end since it depends on Steps 1-4 producing data.
- **Step 6 (Sub-4 measurement)** small standalone; can run in parallel with any other step.

### Goal traceability

Every step serves Goals 1-5 directly or indirectly:
- Step 1: Goal 4 (real oracle is the Koopman O(1) gate)
- Step 2 (LOOP/PIPELINE): Goal 2 (100-tenant via runaway detection + pipeline graph)
- Step 3 (TRACE/RECEIPT/CARBON/FAIRNESS/GUARD/COMPLY/DETERMINISM): Goal 2 (customer-facing observability) + Goal 1 indirectly (carbon = energy)
- Step 4 (LP-8): Goal 5 (LD_PRELOAD-only — leaner build)
- Step 5 (Prometheus): Goal 2 (operator visibility at scale)
- Step 6 (Sub-4 measurement): tuning infrastructure for Week 3 Sub-4

---

## Part 8 — Recommendation + readiness

### Verdict: **WEEK 4 ENTRY READY**

- **Part 1**: §7 W4 scope captured verbatim; 12 ports + Prometheus + LP-8.
- **Part 2**: Wave 5 W4 sweep clean (4 VERIFIED + 1 SYNTHESIS-HYPOTHESIS + 2 self-resolving CONTRADICTIONs).
- **Part 3**: C3 RESOLVED — no-op retirement (file never existed).
- **Part 4**: LOOP + PIPELINE confirmed self-contained; COMPLEMENTARY + INDEPENDENT relative to Sub-4; all three ship in parallel.
- **Part 5**: Sub-4 measurement infra ~65 LOC standalone step.
- **Part 6**: Real oracle wiring feasible via selective init (~20 LOC).
- **Part 7**: 6-step decomp + closeout, ~17-25h vs 12-14-week-trajectory slack of 10.5-12.5 weeks remaining.

### Surfaced for Week 4 scope-lock brief

1. **Oracle wiring approach**: (a) full cipher_init / (b) selective oracle_init / (c) keep hardcoded. **Recommend (b)** — clean, low-risk, ~20 LOC.
2. **cipher_10ops_impl.cpp port (Wave 5 W4-4)**: would bring 50 cross-TU undefs and ~23 transitive sources (per Week 3 Step 4 pre-flight). **Surface for adjudication**: port or skip? The Stage-1 ring is "load-bearing fan-out" per Wave 5, but LOOP/PIPELINE confirmed they don't need `g_cipher_10ops` to function (they use CipherRingEntry as a type only, same pattern as cipher_sense.cpp). **Recommend skip** for v1; revisit at Phase 5 CP 5.5.
3. **VOLT 1000 vs 1200 MHz**: paperwork-only adjudication; fold into Week 4 closeout.
4. **Sub-4 measurement step**: fold into Week 4 or defer to Week 5? Currently planned as Step 6; could defer if budget tight.
5. **Prometheus exporter**: assumes cipher-exporter exists on this pod. **Need to verify** — if exporter is a separate process, Week 4 may need a Step 4.5 to extend it.

### Out-of-scope confirms

- PREDICT, SHIELD, SUSTAIN, THERMOSTAT defer to v1.5 per v1.2.2 L1338.
- Sites 1+2 cuBLAS/SDPA CLASSIFY direct wiring: still N/A.
- Week 5 CP 5.5 100-tenant benchmark: Week 5 scope (separate brief).

---

## Telemetry on disk

- `cipher-fusion-evidence/CIPHER_REENGINEERING_PLAN.md:1321-1356` — §7 Week 4 verbatim
- `cipher-fusion-evidence/CIPHER_LOGIC_AUDIT_WAVE_5_FUSION_PLAN.md:780-859` — Wave 5 §5.5 W4
- `cipher-fusion-evidence/WAVE_5_S5_5_VERIFICATION.md` — prior sweep (W4 rows + C3/C4 contradictions)
- `cipher_rt_phase4/cipher_rt_audit.{c,h}` — existing production AUDIT (~6.7 KB + 2.9 KB)
- `cipher-may13-evidence/include/cipher_loop.h` + `src/cipher_loop.cpp` — Op 26 source
- `cipher-may13-evidence/include/cipher_pipeline.h` + `src/cipher_pipeline.cpp` — Op 27 source
- `cipher_rt_phase4/src/may13/cipher_oracle.cpp` — `cipher_oracle_init` at `:106`; selective init feasibility
- `cipher_rt_phase4/cipher_rt_sense_transition.c` — Week 3 Sub-4 invented wrapper (measurement infra add site)

---

## Discipline notes

- Read-only diagnostic; no source-tree changes.
- W4-1 (C3 cipher_audit.cpp): self-resolving; documented for closeout.
- W4-6 (C4 slot array): self-resolving via LP-8 retirement (Step 4).
- W4-4 (cipher_10ops_impl.cpp port): structural decision surfaced for scope-lock; recommend skip (LOOP/PIPELINE don't need it).
- No new contradictions surfaced this pre-flight.
- 5 paperwork tasks delivered.

---

## Awaiting

Week 4 scope-lock brief, drafted with:
- 6 implementation steps + closeout per Part 7
- Oracle wiring approach (b) selective init
- cipher_10ops_impl.cpp port: SKIP (per recommendation)
- VOLT discrepancy adjudication folded into closeout
- Sub-4 measurement infrastructure as Step 6 (or deferred)
- Prometheus exporter scope: verify cipher-exporter availability first

Estimated Week 4 budget: ~17-25h, well within plan reserve.
