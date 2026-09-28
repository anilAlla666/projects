# CIPHER — Code-Reality Audit — 2026-05-16

**Type:** pure code-reading audit. No GPU, no builds, no code changes — read +
report only. **Method:** five read-only agents, each scoped to one section,
all classifications grounded in `file:line` evidence on disk; doc and memory
claims are used only where the code confirms them, and every contradiction is
called out. Per-section detail is preserved verbatim in
`cipher-fusion-evidence/audit_section_{1a,1b,2,3,4}.md`.

**Bottom line up front.**
- The canonical CIPHER op surface is **not** "15 stubs" — in the may13 build
  it is **19 WORKING + 14 PARTIAL, zero true stubs** (§1.3). But **zero of the
  33 ops are in the shipping Phase 4 runtime `c2c5d313`** (§1.5) — the 33-op
  tree is the legacy/canonical build; the fusion campaign's job is to migrate
  them into `cipher_rt_phase4`. Next-CP scoping must price the migration.
- Phase 4 is **IN-FLIGHT, not closed** — 4.7 and 4.8 have zero work product on
  disk despite a doc that says "Phase 4 closes" (§2.3).
- Anchors are sound; one un-promoted build (`cc0479b8`) to re-anchor (§3).
- Goal 4 (O(1) Koopman substitution) fires **zero** times; closing it is
  **architectural, multi-week** work, not a single TODO (§4).

---

## Section 1 — The 33-ops landscape

**Canonical source:** `PHASE_4_OP_WORKLOAD_MATRIX.md:10` — "the 33 ops = 12
core Stage 0/1/2 + 21 overlay (numbered 13–31 + STRAGGLER + NCCL_P2P)."

**Scoping finding (the brief's premise was wrong).** The brief states the 12
core ops live in `cipher_10ops_impl.cpp`. That file's own header (lines 5–6)
scopes itself to Stage 1 + Stage 2 only. The **6 Stage 0 ops live elsewhere**:
`cipher_dispatch.cpp` (hot path — CLASSIFY, SUBSTITUTE, oracle call site),
`cipher_oracle.cpp` (the safety oracle), `cipher_classify.hpp` (CLASSIFY
fingerprinter), `cipher_intercept*.cpp` (RING_WRITE, SAMPLE, SPECULATE-check
call sites). The audit was expanded accordingly.

### 1.1 — The 12 core ops

| Op# | Name | Stage | Class | File:Line evidence | Stress firing | Notes |
|----|------|-------|-------|--------------------|---------------|-------|
| 1 | CLASSIFY | 0 | **WORKING** | `cipher_classify.hpp:94-224`; called `cipher_dispatch.cpp:406-411,523-530` | indirect FIRES (`DETERMINISM dispatch_count=36450`) | 7-class geometry fingerprinter + 512-slot lock-free cache; runs every launch. |
| 2 | ORACLE | 0 | **WORKING** | `cipher_oracle.cpp:293-377`; called `cipher_dispatch.cpp:496,558` | UNKNOWN (no per-op counter) | Real 5-gate safety logic (phase, min-confidence, structural lookup, EMA demotion, N≤4). |
| 3 | SUBSTITUTE | 0 | **PARTIAL** | `cipher_dispatch.cpp:200-395` | UNKNOWN (no `[L3.2] SUBSTITUTE` markers in logs) | GEMM relaunch+cache path real; block-sub init fails on `/workspace` path bug (`per_op_validation.log:17-18`); recipe types 2/4 are passthrough `return true` stubs (`:379-390`). |
| 4 | COMMIT | 0 | **WORKING** (name unresolved) | behavior at `cipher_dispatch.cpp:587-589` (`return CIPHER_SUBSTITUTED`) | UNKNOWN | No symbol named `COMMIT` exists (`grep -rn COMMIT` empty). The code's own Stage-0 print (`cipher_10ops_impl.cpp:971-972`) lists position 4 as **ORCHESTRATE** — the likely canonical-name mapping. The behaviour — final dispatch of the chosen kernel — is implemented as the dispatch-return path. Classified WORKING-by-behaviour; the canonical-name↔code-name reconciliation (COMMIT↔ORCHESTRATE, ORACLE↔SPECULATE_CHECK, SAMPLE↔GENERATE) is **unresolved** and is itself an audit finding. |
| 5 | SAMPLE | 0 | **PARTIAL** | `cipher_intercept_cudart.cpp:2353-2425` | UNKNOWN | Real norm-ring; but `maybe_collect_sample` hard-stops after 500 launches (`:2409-2411`) — warmup burst, not continuous; `CIPHER_SAMPLE_DIM=1` scalar only. |
| 6 | RING_WRITE | 0 | **WORKING** | `cipher_10ops.h:83-100`; call sites `cipher_intercept.cpp:180-181,374-375` | UNKNOWN (no counter dumped) | Real SPMC disruptor-ring; fires once per launch when 10ops initialized. |
| 7 | REMEMBER | 1 | **WORKING** | `cipher_10ops_impl.cpp:428-458` | **NEVER FIRES** (`per_op_validation.log:20` — Stage 1 thread not spawned) | Real CfC LNN forward pass; only runs if ≥1 observer env var set. |
| 8 | VALIDATE | 1 | **PARTIAL** | `cipher_10ops_impl.cpp:460-475` | **NEVER FIRES** | Welford stats real, but observational-only: never increments `validate_failures`; 3-σ detector `rs_ok` defined and **never called** → "anomalies detected" structurally always 0. |
| 9 | AUDIT | 1 | **PARTIAL** | `cipher_10ops_impl.cpp:341-378`, live path `472-474` | **NEVER FIRES** | HMAC-SHA256 chain function real, but the call is **commented out** in the live path (`:472-473` "skip HMAC on detached thread"). Only the counter runs. |
| 10 | SPECULATE | 1 | **WORKING** | write `cipher_10ops_impl.cpp:477-505`; check `cipher_intercept.cpp:127-149` | **NEVER FIRES** (write side; Stage 1 off) | Complete; caveat — Stage-0 check compares against still-unclassified `op_class=0xFF` (`cipher_intercept.cpp:129`), a likely correctness weakness. |
| 11 | ADAPT | 2 | **PARTIAL** | `cipher_10ops_impl.cpp:597-765`, KEN `99-285` | **NEVER FIRES** (Stage 2 thread not spawned) | EDMD/Koopman solve real, but feeds `h_before == h_after` degenerate pair (`:693-701`); EDMD input is synthetic near-identity (`:663-669`). |
| 12 | ARBITRATE | 2 | **PARTIAL** | `cipher_10ops_impl.cpp:767-814` | **NEVER FIRES** | Writes a SHM demand integer; the actual Green Context SM rebalance is unimplemented — `:789,799,803` all "Production: call cuDevSmResourceSplit/cuGreenCtx*" TODO comments. |

Core tally: **WORKING 6 · PARTIAL 6** (COMMIT counted WORKING-by-behaviour).
All six Stage 1/2 ops never executed in the stress run
(`per_op_validation.log:20` — "Stage 1/2 threads skipped — no observers
enabled").

### 1.2 — The 21 overlay ops

| Op# | Name | File | Class | File:Line | Stress firing | Notes |
|-----|------|------|-------|-----------|---------------|-------|
| 13 | SENSE | cipher_sense.cpp | **WORKING** | :171-273 | UNKNOWN | Session-classification state machine. |
| 14 | SHIELD | cipher_shield.cpp | **PARTIAL** | :90-146 | UNKNOWN | ITL/jitter real; Protections 2/3 detection-only, flags never consumed (`:7-10,162-164`). |
| 15 | SUSTAIN | cipher_sustain.cpp | **PARTIAL** | :90-105,177-183 | UNKNOWN | KV-pressure slope real; `sustain_compress` flag has no consumer (`:5-6,122`). |
| 16 | GUARD | cipher_guard.cpp | **WORKING** | :99-130 | FIRES (`leak_count=0`) | Cross-session leak detector. |
| 17 | PREDICT | cipher_predict.cpp | **WORKING** | :160-245 | FIRES (`ptr_seen=153`) | Hot-pointer promotion into persist engine. |
| 18 | RECEIPT | cipher_receipt.cpp | **WORKING** | :123-225 | SILENT (no session attributed) | Real HMAC-SHA256 proof-of-compute. |
| 19 | CONTINUITY | cipher_continuity.cpp | **PARTIAL** | :130-175 | FIRES (`manifest_count_total=0`) | Region tracking only; no manifest ever written, no checkpoint worker (`:3-5`). |
| 20 | THERMOSTAT | cipher_thermostat.cpp | **WORKING** | :166-236 | UNKNOWN | Detector + real consumer (VOLT reads its flag at `cipher_volt.cpp:321`); init banner "not consumed" is **stale**. |
| 21 | DETERMINISM | cipher_determinism.cpp | **WORKING** | :50-63 | FIRES (`dispatch_count=36450`) | Complete. |
| 22 | PULSE | cipher_pulse.cpp | **PARTIAL** | :177-358 | UNKNOWN | Signals 1+3 real; Signal 2 deferred, severity ceiling hard-capped at 2 / CRITICAL unreachable (`:7-8,261-263`). |
| 23 | CARBON | cipher_carbon.cpp | **WORKING** | :84-169 | FIRES | Per-session carbon estimator. |
| 24 | FAIRNESS | cipher_fairness.cpp | **WORKING** | :88-117 | UNKNOWN | Per-tenant quota table complete; `cipher_fairness_shm.cpp` is a separate cross-process variant. |
| 25 | TOPOLOGY | cipher_topology.cpp | **WORKING** | :26-55 | FIRES | Static peer-adjacency; no-op observe is by design. |
| 26 | LOOP | cipher_loop.cpp | **WORKING** | :148-229 | FIRES (`runaway_count=7`) | 3-signal runaway detector with sticky latch. |
| 27 | PIPELINE | cipher_pipeline.cpp | **WORKING** | :117-239 | FIRES (`edge_count=10`) | Multi-agent session correlation. |
| 28 | TRACE | cipher_trace.cpp | **WORKING** | :50-116 | FIRES (`written=8192`) | Bounded JSONL exporter; drop-on-full by design. |
| 29 | COMPLY | cipher_comply.cpp | **WORKING** | :46-82 | SILENT (fires-as-designed) | Aggregation-only verdict op. |
| 30 | VOLT | cipher_volt.cpp | **PARTIAL** | :307-366 | UNKNOWN | Full DVFS actuator present; runs classifier-only / DEGRADED on this pod (NVML clock-set NOT_SUPPORTED). |
| 31 | HIBERNATE | cipher_hibernate.cpp | **PARTIAL** | :139-170 | UNKNOWN | Idle detector real; power-limit actuator gated off when NVML probe fails (documented pod state). |
| 32 | STRAGGLER | cipher_straggler.cpp | **PARTIAL** | :178-261 | UNKNOWN | Local slowdown detection real; cross-rank attribution NOT IMPLEMENTED (`:8-10,126-129`). |
| 33 | NCCL_P2P | cipher_nccl.cpp | **PARTIAL** | :34-291 | UNKNOWN | Algo-selection policy real; eBPF P2P-routing path simulated / CPU-stub (`:21-29`). |

Overlay tally: **WORKING 13 · PARTIAL 8 · STUB 0 · MISSING 0**.

### 1.3 — Honest-gap correction: there are no "15 stubs"

The brief's stated target — "the concrete list of the 15 stubs" — does not
survive contact with the code. The 33-op accounting (in the may13 source
tree — see the §1.5 production-runtime caveat) is:

| Class | Count | Ops |
|---|---|---|
| **WORKING** | 19 | CLASSIFY, ORACLE, SUBSTITUTE-name…—see note; RING_WRITE, REMEMBER, SPECULATE, COMMIT (by behaviour); SENSE, GUARD, PREDICT, RECEIPT, THERMOSTAT, DETERMINISM, CARBON, FAIRNESS, TOPOLOGY, LOOP, PIPELINE, TRACE, COMPLY |
| **PARTIAL** | 14 | SUBSTITUTE, SAMPLE, VALIDATE, AUDIT, ADAPT, ARBITRATE; SHIELD, SUSTAIN, CONTINUITY, PULSE, VOLT, HIBERNATE, STRAGGLER, NCCL_P2P |
| **STUB** | 0 | — every op file contains real, runnable logic |
| **MISSING** | 0 | — COMMIT has no symbol by that name but the behaviour exists (counted WORKING; name reconciliation unresolved) |

(WORKING list = the 5 core + COMMIT + 13 overlay = 19; CLASSIFY/ORACLE/
RING_WRITE/REMEMBER/SPECULATE are the 5 core WORKING ops.)

So the "15" is roughly the right *count* of not-fully-WORKING ops
(14 PARTIAL) but **"stub" is the wrong word**. Not one op is an empty/no-op
stub. The 14 PARTIAL ops all have substantive logic; what makes them PARTIAL
is one of three recurring patterns:

1. **Detection-only where an actuator is intended** — the op computes a
   signal/flag that nothing downstream consumes (SHIELD Protections 2/3,
   SUSTAIN `sustain_compress`, VALIDATE `rs_ok` never called).
2. **Actuator gated off** — the actuator code exists but is environment-
   disabled (VOLT/HIBERNATE NVML NOT_SUPPORTED on this pod) or commented out
   (AUDIT HMAC call) or replaced by a TODO (ARBITRATE SM rebalance).
3. **Half the op explicitly deferred** — PULSE Signal 2, STRAGGLER cross-rank
   attribution, NCCL_P2P eBPF path, CONTINUITY manifest writer.

**This is the single most important finding for next-CP planning:** the work
remaining on the op surface is not "write 15 stubs from scratch" — it is
"wire 14 already-built detectors to their actuators / consumers." That is a
materially smaller and lower-risk body of work than the memory framing implied.

### 1.4 — Stress-suite firing

The only valid firing record is `cipher-may13-evidence/stress2/
per_op_validation.json`. It exercises a **non-canonical** 21-op set; its
intersection with the canonical overlay-21 is exactly 11 ops. Result:

- **FIRES (8):** GUARD, PREDICT, DETERMINISM, CARBON, TOPOLOGY, LOOP,
  PIPELINE, TRACE.
- **SILENT, by gating not bug (2):** RECEIPT, COMPLY (no session attributed
  at the 50-token workload).
- **NEVER FIRES (6 core Stage 1/2):** REMEMBER, VALIDATE, AUDIT, SPECULATE-
  write, ADAPT, ARBITRATE — the stress run set no observer env var, so the
  Stage 1/2 threads were never spawned.
- **UNKNOWN (the rest):** never exercised by that run — SENSE, SHIELD,
  SUSTAIN, THERMOSTAT, PULSE, FAIRNESS, VOLT, HIBERNATE, STRAGGLER, NCCL_P2P,
  and all 6 Stage 0 ops (no per-op Stage 0 counters exist).

Doc conflict flagged: `OPS_VALIDATION_REPORT.md`'s "13/21 FIRE" headline
refers to its non-canonical 21-op set, **not** the canonical overlay-21 — the
two must not be conflated.

### 1.5 — Production-runtime caveat (load-bearing)

**The 33 ops audited in §1.1–§1.4 are NOT in the shipping Phase 4 runtime.**

Verified by `nm -D` on the production `cipher_rt_phase4/libcipher_rt.so`
(`c2c5d313`, the CP 2.5 anchor):

```
$ nm -D libcipher_rt.so | grep -cE 'cipher_(sense|shield|...|nccl)_'   → 0
$ nm -D libcipher_rt.so | grep -E '10ops|cipher_dispatch|cipher_classify|cipher_oracle' → (nothing)
$ nm -D cipher-may13-evidence/libcipher_rt.so.preroadmap | grep -c overlay-ops → 30
```

**Zero** of the 33 ops' symbols are in the production runtime. The 33-op
source tree (`cipher-may13-evidence/src/`, the `cipher_*.cpp` family) is the
**legacy / canonical-design build**; the shipping Phase 4 runtime
(`cipher_rt_phase4/`, the `cipher_rt_*.o` family in the Makefile `OBJS`) is a
**separate, smaller codebase**. The Phase 4 runtime re-implements only a
subset as `cipher_rt_*` modules: ARBITRATE (`cipher_rt_partition_router` /
`_arbitrate` / `_sm_packer` / `_green_ctx`), VOLT (`cipher_rt_volt`),
SUBSTITUTE (`cipher_rt_matmul_dispatch` / `_cublas_shim` / `_marlin_*`),
AUDIT (`cipher_rt_audit`), plus the new attention substrate and GOT patcher.
The other ~29 canonical ops have **no presence in `c2c5d313`**.

Consequences for this audit:
- §1.1–§1.4 accurately characterize the **canonical op surface as it exists
  in the may13 build** — which is the correct object for "the 33 ops," but is
  not what ships today.
- The §1.4 stress-firing evidence (`stress2/per_op_validation.*`) is firing
  data for the **may13 build**, not for `c2c5d313`.
- This is itself the largest §1 honest-gap: the fusion campaign's per-CP work
  is precisely the migration + per-tenant rebuild of these ops into the
  Phase 4 runtime. Section 2's "4.4 L2 persist — not present in
  `cipher_rt_phase4/`" is one instance of this general fact.

---

## Section 2 — Phase 4 progress audit

### 2.1 — T4.6.x KV/attention series

| Item | Status | Code evidence | Test evidence | Honest gap |
|---|---|---|---|---|
| **T4.6.1** attention substrate | **SHIPPED** | `cipher_rt_attn_dispatch.cpp` (3 SDPA trampolines, compiled into `libcipher_rt.so`) | `PHASE_4_T4_6_1_REPORT.md:125-142` 3-indicator PASS | Substrate only — REDIRECTED treated as PASSTHROUGH (`:157-162`); flash/efficient trampolines runtime-untested; no 25-test regression. Doc honest. |
| **T4.6.2** KV page allocator | **SHIPPED (component) — no standalone report** | `cipher_rt_kv_alloc.{c,h}` (680 LOC) | `vmm_bench.log` PASS; `gate_d/alloc_unit_test.log` ALL PASS | No `PHASE_4_T4_6_2_*.md` exists; status inferred from code + benches. |
| **T4.6.3** L1 in-process dedup | **SHIPPED** | xxhash64 dedup path in `cipher_rt_kv_alloc.c` | `t4_6_3_dedup_report.md:130-138` 15/15 PASS | Honest: single-process, synthesized pages, not merged into live slab path. |
| **T4.6.4** L3 cuIpc/kmod dedup | **SHIPPED** | `cipher_kmod/cipher_kvdedup.c` (477 LOC, in `.ko`) | `t4_6_4_report.md:14-21` 5/5 PASS; re-verified `gate_d/kvdedup_xproc.log` | Report's kmod md5 `b263ad30` is stale (current `e2f50452` — anchor rotation, not regression). **Phase-closure over-claim — see §2.3.** |
| **T4.6.5** real-trace validation gate | **NOT-STARTED** | none | none | The project's own words: `t4_6_4_report.md:78-79` "T4.6.5 … is the first **Phase 5** deliverable." |
| **T4.6.6** 50–100 tenant integration | **NOT-STARTED** | none | none | Named only as future scope; no work product on disk. |

### 2.2 — Phase 4 sub-phases 4.3–4.8

| Item | Status | Code evidence | Honest gap |
|---|---|---|---|
| **4.3 DVFS/VOLT** | **SHIPPED** | `cipher_rt_volt.{c,h}`, kmod `CIPHER_SET_CLOCK_MHZ` | Doc-vs-doc contradiction: `PHASE_4_T4_3_ENVELOPE.md:45-61` reports Mistral-7B B=1 VOLT at −13.78%/−15.45%; `CP_2_4_REPORT.md:58-61` **retracts** that envelope as a harness artifact and reports +62.45% tok/W. The +62% is null-validated and credible; a reader of the envelope doc alone would conclude the opposite. |
| **4.4 L2 persist** | **IN-FLIGHT (PARTIAL)** | `cipher-may13-evidence/src/cipher_l2_persist.cu` (real impl) — **not present in `cipher_rt_phase4/`** | `phase_audit/phase4/audit.md:40-50` verdict PARTIAL: mechanism fires in the may13 snapshot only; canonical 1.3–1.6× tok/W gate never measured; not in the Phase 4 runtime tree. |
| **4.5 Weight/Marlin** | **SHIPPED (regime-limited)** | `cipher_rt_marlin_engine.cpp` + actuator + cublas shim, all in `libcipher_rt.so` | Marlin regresses at B=1 (`CP_2_4_REPORT.md:168-172`); TMA memory substitution NOT DONE (`phase_audit/phase4/audit.md:72-94`); partition-aware Marlin deferred to Phase 5. Shipped for its designed B≥8 regime. |
| **4.6 KV** | **IN-FLIGHT** | T4.6.1–4 shipped | T4.6.5 + T4.6.6 not started; the multi-tenant moat on real KV is unearned until T4.6.5. |
| **4.7 Fusion + agentic + compliance** | **NOT-STARTED** | none in `cipher_rt_phase4/` (only legacy may13 `cipher_fusion.cpp`) | No P4.7 cluster work product exists. |
| **4.8 Integration + 24h soak** | **NOT-STARTED** | harness `cipher_measurement/soak_24h.sh` only — unrun | G4 24-hour soak gate (`PHASE_4_ARCHITECTURE.md:22`) never executed. |

### 2.3 — Biggest doc-vs-code gap

`t4_6_4_report.md:11,75-77` declares **"Phase 4 closes"** on the strength of
the KV-dedup five-indicator gate alone. This is contradicted by the project's
own binding spec: `PHASE_4_ARCHITECTURE.md:134-135` defines **P4.7** (Fusion +
agentic + compliance) and **P4.8** (integration + G3 density + **G4 24-hour
soak** + evidence tarball) as required Phase 4 sub-phases, and `:151` makes
the final P4.8 ship gate the four G1–G4 criteria measured simultaneously on
real silicon. **No P4.7 or P4.8 work product exists on disk.** The same
T4.6.4 report further calls T4.6.5 "the first Phase 5 deliverable" — i.e. it
simultaneously closes Phase 4 and pushes its own next step into Phase 5,
skipping P4.7 and P4.8. **Honest status: Phase 4 is IN-FLIGHT** — 4.3 and 4.5
shipped (with regime caveats), 4.6 partially shipped (T4.6.1–4 done,
T4.6.5–6 not), 4.4 partial, 4.7 + 4.8 not started.

---

## Section 3 — Anchor + version drift audit

Every md5 below is a literal `md5sum` result; every srcversion a literal
`modinfo -F srcversion`.

### 3.1 — libcipher_v2: 86618c30 vs cc0479b8

| md5 | path | size / mtime | identity |
|---|---|---|---|
| `86618c30…` | `libcipher_v2/libcipher_v2.so.v0.2.0` | 16496 B / 05-13 07:31 | **Canonical anchor**, Phase-2 v0.2.0 |
| `cc0479b8…` | `libcipher_v2/libcipher_v2.so` | 16968 B / 05-13 10:05 | Un-anchored Phase 3 Task 5 CUPTI build |

**Why it was rebuilt — DETERMINED, not a mystery.** `cc0479b8` is the Phase 3
Task 5 CUPTI integration: it adds source `cipher_cupti.c`, a new
`DT_NEEDED libcupti.so.12`, and undefined imports `cuptiSubscribe` /
`cuptiEnableCallback` / `clock_gettime`. The exported ABI
(`InitializeInjection`/`InitializeInjection2`) is **unchanged** between the
two. It matches `PHASE_3_NOTES.md` §"Layer C — libcipher_v2 CUPTI integration
(Task 5)". The anchor was simply never re-cut after the Phase-2 milestone.
**Recommendation: re-anchor `cc0479b8`** as a fresh version
(e.g. `libcipher_v2.so.v0.3.0` with recorded md5); it is a functional superset
of `86618c30`, not a stray. The CP 2.5 `.deb` correctly still bundles
`86618c30` — keep it documented. (Per the adjudicated CP 2.5 flag this is
**Phase 2 housekeeping, not blocking** — recorded here for the ledger.)

### 3.2 — libcipher_rt rollback chain

Chain `a0d6cdda → 55e2e323 → caae0bb8 → (b1a3424c diag) → c63b6abc →
5e304549 → 2f845393 → c2c5d313` is **intact**. 7 of 8 anchors exist as files
on disk (verified by `md5sum`). The one absent — `b1a3424c` — is documented
in `MARLIN_HANG_INSTRUMENTATION.md:139` as a throw-away instrumented
diagnostic, explicitly "not a campaign anchor," deliberately never preserved.
Current canonical `c2c5d313` is confirmed by sidecar
`cipher_rt_phase4/libcipher_rt.so.cp2_5.md5`. **Recommendation:** document
`b1a3424c` as intentionally file-less in the anchor ledger so a future audit
does not re-flag it. No chain repair needed.

### 3.3 — kmod anchors

| md5 | path | srcversion | finding |
|---|---|---|---|
| `e2f50452…` | `cipher_kmod/cipher_kmod.ko` | `E427CAFA4E94D548233DC7A` | Reference build, kmod 0.4.8 — VERIFIED |
| `6654d9e5…` | `/lib/modules/…/updates/dkms/cipher_kmod.ko` | `E427CAFA4E94D548233DC7A` | DKMS rebuild — **same srcversion**; md5/size differ only by DKMS strip. Not drift. |
| `2a69f9de…` | `cp_2_4/kmod_0.4.7_pre_devnode.ko` | `3088289AA293398399B1903` | 0.4.7 pre-devnode rollback — VERIFIED; distinct srcversion correct (predates the 0.4.8 devnode change). |

No kmod drift. All three anchors verified; identity relationships exactly as
expected.

---

## Section 4 — Goal 4 (O(1) Koopman substitution) audit

**Goal 4 = an O(1) Koopman-operator-predicted substitute that replaces
expensive kernel/attention compute on the hot path.**

### 4.1 — The hot-path fallback

There is no single TODO — there are **four disconnected substitution
substrates**, each gated to passthrough:

1. **THE hot-path gate** — `cipher_dispatch.cpp:570-574`: every launch hits
   `cipher_registry_lookup`; it returns no entry; comment "L3.5 EDMD pipeline
   — wired in Week 4-5"; `return CIPHER_PASS_THROUGH`. The registry is seeded
   only with **32 hardcoded Llama-3-70B-scale shapes** (`cipher_recipes.cpp:346`)
   — no stress workload (Llama-3.2-1B, Mistral-7B, TinyLlama) emits those, so
   `apply_recipe()` is never reached.
2. **Attention Koopman** — `cipher_attn_koopman.cpp:495-499`: reaches the
   `@V_cache` step and reverts — "fused_path_not_yet_wired".
3. **Attention REDIRECT** — `cipher_rt_attn_dispatch.cpp:158-164`: "substitution
   path not implemented in T4.6.1 — passing through."
4. **Flow-substitute** — `cipher_flow_substitute.cpp:490-492`: ships as an
   explicit STUB. Its real-substitution branch gates on
   `x_ptr/w_ptr/y_ptr/hidden` — declared at `:66-70`, **never assigned** — so
   the branch is structurally dead.

### 4.2 — Exists vs missing

| Component | Status |
|---|---|
| Koopman/EDMD solver (`K = ΨY·ΨX†`, predict) | **EXISTS** — `cipher_edmd.cpp`, `cipher_edmd_live.cpp` |
| KEN — Koopman Eigenfunction Network | **EXISTS** — `cipher_10ops_impl.cpp:99-225` |
| Runtime Koopman module (`cipher_koopman_runtime.cpp`) | **EXISTS but DEAD CODE** — zero callers anywhere |
| Hot-path dispatcher (CLASSIFY→ORACLE→registry→SUBSTITUTE) | **EXISTS, wired** — `cipher_dispatch.cpp:402-590` |
| O(1) substitute kernels (block-`O(Kr)`, cached-`O(1)`) | **EXISTS** — `cipher_block_sub_kernel.cu`, `cipher_attn_koopman_kernel.cu` |
| Dynamic registry population from derived surrogates | **MISSING** — no `cipher_registry_add/insert`; registry is 32 static shapes |
| Wiring of `cipher_koopman_runtime` to the hot path | **MISSING** — no caller |
| Flow-substitute X/W/Y arg extraction from CUDA struct args | **MISSING** |
| Attention fused-kernel substitution wiring | **MISSING** |

**Net:** the math and the substitute kernels exist; the **connective tissue**
does not. The substitution branch is bypassed at the registry gate before any
Koopman predict runs.

### 4.3 — Substitution-firing finding: confirmed zero

Searched all `stress/` + `stress2/` logs for substitution-success prints
(`[CIPHER L3.2] SUBSTITUTE`, `[O(1)-block]`, `[CIPHER L1.3] Registry HIT`,
`FLOW SUBST … real_substituted`) — **0 matches**.
`per_op_validation.json` reports `KOOPMAN` op `SILENT (saw_qk=0)`;
`per_op_validation.log:108` — `Substitutions: 0 (0.0%)`.
The ADAPT op *does* solve Koopman 84,611× (`per_op_v2.log`, fit_error=0.0002)
but that only nudges a shadow LNN — zero kernel substitutions. The D3
"100% substitution rate" is the unrelated FP8/Marlin system, not Koopman.

### 4.4 — Path to close: architectural (multi-week)

**Not** "matrix solve done, lookup is a TODO" (which would be ~1 week). The
solver *is* done — but Goal 4 is blocked by **four mutually-disconnected
substitution substrates**, no single one of which closes the goal. The
cleanest evidence: `cipher_koopman_runtime.cpp` — the module literally named
"Runtime Koopman Derivation" — is fully built with **zero callers**. Closing
Goal 4 requires: (i) pick one substrate; (ii) build the surrogate→registry
population pipeline behind it (the registry is hardcoded to 32 static shapes);
(iii) wire hot-path argument extraction (CUDA struct-arg reverse-engineering —
fragile, multi-session per CLAUDE.md); (iv) retire the three competing stubs.
**Estimate: multi-week, architectural.**

---

## Synthesis — ground truth for next-CP planning

1. **The op surface is healthier than "15 stubs" — but it is not the
   shipping runtime.** In the may13 canonical build: 19/33 WORKING,
   14 PARTIAL, 0 true stubs — the PARTIAL work is "wire an existing detector
   to an actuator," not "build a stub." **However (§1.5):** zero of the 33
   ops are in the production Phase 4 runtime (`c2c5d313`). The real next-CP
   work is therefore **not** "wire 14 detectors in place" — it is **migrate
   each canonical op from the may13 tree into the `cipher_rt_phase4` build,
   per-tenant, then wire it.** That is a materially larger, multi-CP arc —
   which is exactly what the fusion campaign's per-CP structure is for. Any
   next-CP estimate must price in the migration, not just the wiring.
2. **Phase 4 is not closed.** 4.7 (Fusion+agentic) and 4.8 (integration +
   24h soak + G4 gate) have zero work product. Any "Phase 4 closed" claim
   should be corrected; the honest state is 4.3/4.5 shipped, 4.6 partial,
   4.4 partial, 4.7/4.8 not started.
3. **Anchors are sound.** One action: re-anchor `cc0479b8` (libcipher_v2 +CUPTI)
   as a fresh version; one bookkeeping note: mark `b1a3424c` intentionally
   file-less. No kmod drift.
4. **Goal 4 is the biggest single gap.** Zero substitutions fire; closing it
   is architectural (multi-week), and the first decision is which of the four
   substrates to keep. It should be scoped as its own multi-STEP arc, not
   folded into an op-wiring CP.

**Audit constraint honored:** no code changed during this audit — read +
report only. Anchors preserved: kmod `e2f50452`, libcipher_v2 `86618c30`,
libcipher_rt `c2c5d313`.

*Per-section detail with full evidence:
`cipher-fusion-evidence/audit_section_{1a,1b,2,3,4}.md`.*
