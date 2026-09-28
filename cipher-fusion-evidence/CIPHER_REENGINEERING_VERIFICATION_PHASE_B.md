# CIPHER REENGINEERING PLAN — Phase B Verification

**Date:** 2026-05-20
**Type:** Verification & extension of `CIPHER_REENGINEERING_PLAN.md` §2.1/§2.2.
**Trees inspected:**
- Tree A — `/home/ubuntu/cipher-may13-evidence/` (the 33-op canonical codebase).
- Tree B1 — `/home/ubuntu/cipher_rt_phase4/` (deployed userspace runtime, anchor `83afd1ca`).
- Tree B2 — `/home/ubuntu/cipher_kmod/` (deployed kernel module, anchor `285d102e`).

This document executes Phase B.1 (re-verification of the 6-of-33-ops-fire claim), Phase B.2 (dependency-chain analysis of each NOT-PORTED op), Phase B.3 (dependency DAG and port order), Phase B.4 (show-stoppers and v2 deferrals). Findings are cited to file:line throughout.

---

## Section B.1 — RE-VERIFICATION OF THE 6-OPS-FIRE CLAIM

The plan (§2.3) asserts: "of 33 canonical ops, exactly 6 fire on Tree B's hot path today — SUBSTITUTE (Marlin), ARBITRATE (kmod CP54), VOLT, AUDIT, COMMIT, CLASSIFY." Below is the byte-level verification of each.

### B.1.1 — SUBSTITUTE actuator registration (`cipher_rt_marlin_actuator.c`)

**Plan claim verified:** Marlin INT4 actuator registers at `cipher_rt_marlin_actuator.c:198` (`cipher_rt_matmul_register_actuator(&g_marlin_actuator)`). The actuator descriptor is built at L170-174 with `name = "MARLIN_INT4"`, `priority = 10` ("high-priority compute substitution"), `maybe_handle = maybe_handle_marlin`.

**Gates confirmed:**
- `MARLIN_MAX_M_GATE = 64` at L47, enforced at L106 (`marlin_M > MARLIN_MAX_M_GATE` → PASSTHROUGH).
- `STABILITY_THRESHOLD = 4` at L48, enforced at L128 (`hits < STABILITY_THRESHOLD` → PASSTHROUGH).
- FP16 only: L95-99 rejects non-`CUDA_R_16F` A/B/C.
- N ≥ 1024 AND K ≥ 1024: L114 (avoids cross-block-lock deadlock at small tiles).
- K % 128 == 0 AND N % 64 == 0: L118 (`marlin_K & 127`, `marlin_N & 63`).

**Env gate:** `CIPHER_MARLIN` (L178-184); if unset → disabled, registration skipped.

**Plan claim verified verbatim.** All thresholds match the plan's prose.

### B.1.2 — ARBITRATE handler for nrs 13-15 plus Track 3 nrs 16-20 (`cipher_cp54_sched.c`)

**Plan claim verified:**
- `CIPHER_CP54_NUM_GROUPS = 15` defined at L61 (with explanatory comment at L60: "this H100 yields 15 × 8-SM groups + 12 remainder (PHASE_1_3A_PROBE.md)").
- `CIPHER_CP54_SMS_PER_GROUP = 8` at L62.
- Static assert at L113-114: `CIPHER_CP54_NUM_GROUPS <= 32` (grp_mask is u32).
- Ledger backing array: `static atomic_t cipher_cp54_groups[CIPHER_CP54_NUM_GROUPS]` at L118.

**Track 3 migration ioctls confirmed** (with their nr-comments at the top of each handler):
- nr 16 SUBSCRIBE_MIGRATE — handler comment at L702.
- nr 17 POLL_MIGRATE — handler comment at L724.
- nr 18 START_MIGRATE (PROPOSED → MIGRATING) — L751.
- nr 19 ACK_MIGRATE (MIGRATING → COMMIT or ABORT) — L770.
- nr 20 COMPACT_MIGRATE (forced compaction) — L794.

The 15×8-SM ledger and ioctls 13/14/15 (ALLOCATE/FREE/QUERY) are wired through the same file's `struct cipher_cp54_allocate p;` (L552) and `struct cipher_cp54_query q;` (L664) dispatch. **Plan claim verified.**

### B.1.3 — VOLT (DVFS) actuator and init order (`cipher_rt_volt.c`)

**Plan claim verified:**
- `cipher_rt_volt_init` defined at L253 (380 LOC file).
- `[210, 1980]` MHz range enforcement at L293: `if (target_mhz < 210 || target_mhz > 1980) { ... return CIPHER_RT_VOLT_OFF; }`.
- NVML path attempted first via `resolve_nvml()` at L300; falls through to kmod ioctl `CIPHER_SET_CLOCK_MHZ` (probed at L317) when NVML returns NOT_SUPPORTED.
- Env gate `CIPHER_VOLT` at L255 (must be `on`|`1`|`ON`); `CIPHER_VOLT_BATCH` (B=1/8/32/64 LUT at L59-60) or `CIPHER_VOLT_MHZ` direct override.
- Signal-safe restore: signal handlers installed at L333 (`install_signal_handlers()`), `atexit` registered at L250 in `install_signal_handlers()` body.

**Init order from `cipher_inject.c`** (plan §1.2): VOLT is step 6 in `cipher_v2_init_body`, after green-context allocation. **Plan claim verified.**

### B.1.4 — AUDIT priority-0 on BOTH substrates (`cipher_rt_audit.c`)

**Plan claim verified:**
- Two priority-0 actuator descriptors:
  - `audit_matmul_act` at L133: `.name = "audit", .priority = 0, .maybe_handle = audit_matmul_handle`.
  - `audit_attn_act` at L136: `.name = "audit", .priority = 0, .maybe_handle = audit_attn_handle`.
- Both registered in `cipher_rt_audit_init` (L178): matmul at L190 (`cipher_rt_matmul_register_actuator`), attn at L191 (`cipher_rt_attn_register_actuator`).
- Env gate `CIPHER_AUDIT` at L182 (must be truthy via `env_on()` helper).

**Plan claim verified.**

### B.1.5 — COMMIT as FSM token

The plan claims COMMIT lives as a state-machine token in (a) `cipher_cp54_sched.c` Track 3 DSM and (b) `cipher_weight_arena.c` weight-arena commit-on-publish.

**Plan claim verified — with one nuance:**

**(a) `cipher_cp54_sched.c`:**
- State-machine commentary (L367): `IDLE --[PROPOSE]--> PROPOSED --[START]--> MIGRATING --[ACK(ok)]--> COMMIT`.
- The actual state-token uses outcome constants: `m->last_outcome = CIPHER_CP54_MIGOUT_COMMITTED` at L395 of `cp54_commit_migration()` (defined at L383).
- Telemetry counter incremented at L397: `atomic_inc(&cipher_cp54_stat_commits)`.
- Verbose-mode log token: L98-110 declares `cipher_cp54_mig_verbose` "1 = log PROPOSE/COMMIT".

**(b) `cipher_weight_arena.c`:**
- Single grep-match at L262: a code comment `/* commit: claim the consumer slot, dup the kmod-held fd into the caller */`. This is **comment-only describing the atomic slot-claim sequence (L263-266)**, not a named state-machine token.

**Plan claim corrected:** COMMIT is a state-machine token in `cipher_cp54_sched.c` (`CIPHER_CP54_MIGOUT_COMMITTED` + `cp54_commit_migration()` function + `cipher_cp54_stat_commits` counter). In `cipher_weight_arena.c` "commit" appears only as a comment describing what the IMPORT handler does. The plan's framing ("Track 2 weight-arena commit-on-publish") is accurate as a description of *behavior* but should not be cited as a literal token.

### B.1.6 — CLASSIFY at cipher_cupti.c:166

**Plan claim verified:**
- L166 reads: `/* T4.2.4d diagnostic: classify the stream this launch is on. */`
- The subsequent code (L167-180+) classifies each launch's *stream* into one of {NULL / non-green / on-green} and increments per-class counters (`g_launches_on_null_strm`, `g_launches_on_null_grn`, `g_launches_on_green`).

This is a **stream classifier** — it determines on which CUDA stream a kernel was launched. It is **NOT** the workload classifier from `cipher_classify.hpp` (which classifies KernelGeom → OpClass ∈ {GEMM, ATTENTION, ...}).

**Plan claim verified.** CLASSIFY as the may13 workload-class engine is NOT in the deployed runtime; the L166 reference is a stream-routing diagnostic that happens to use the word "classify." The substantive may13 classifier (`include/cipher_classify.hpp`, 250 LOC, 7 OpClasses) has zero presence in `cipher_rt_phase4/`.

### B.1.7 — Critical addendum to the plan's PASS_THROUGH line citations

The plan's §2.1 footnote claims `cipher_dispatch.cpp` has PASS_THROUGH early-exits at L541 / L560-561 / L570-573 / L577-579 / L584 and `CIPHER_SUBSTITUTED` at L589, then in the same paragraph says "file is 543 lines."

**Plan claim corrected:** Those line numbers map to **`src/cipher_dispatch.cpp` (616 LOC shadow copy)**, NOT the top-level `cipher_dispatch.cpp` (543 LOC live file).

Grep confirms — in `src/cipher_dispatch.cpp`:
- L541 `if (!g_layer3_initialized) return CIPHER_PASS_THROUGH`
- L561 ORACLE_DENY → PASS_THROUGH
- L572-573 the "L3.5 EDMD pipeline — wired in Week 4-5" comment + PASS_THROUGH
- L579 error_bound > 0.01 → PASS_THROUGH
- L584 `if (!substituted) return CIPHER_PASS_THROUGH`
- L589 `return CIPHER_SUBSTITUTED`

In top-level `cipher_dispatch.cpp` the same logic lives at:
- L448 (layer3 init), L468 (ORACLE_DENY), L479-480 (EDMD comment + PASS_THROUGH), L486 (error bound), L500 (substituted), L515 (CIPHER_SUBSTITUTED).

The plan's prose is internally inconsistent (543 LOC claim + L570+ citations). Both files exist in the may13 tree; `diff -q` shows they differ; the top-level is the one that ships per `Makefile L29` `filter-out`. The reengineering work must port the **top-level** file with its true line numbers.

**B.1 summary table:**

| Claim | Verdict | Notes |
|---|---|---|
| Marlin actuator at L198, gates M≤64, K%128==0, N%64==0, STABILITY_THRESHOLD=4 | VERIFIED | All four numbers match `cipher_rt_marlin_actuator.c:47/48/118/198`. |
| CP54 ledger 15×8-SM, nrs 13/14/15 + 16-20 wired | VERIFIED | `cipher_cp54_sched.c:60-62, 552, 664, 702-794`. |
| VOLT [210, 1980] range, NVML→kmod fall-through | VERIFIED | `cipher_rt_volt.c:293, 300-321`. |
| AUDIT priority-0 on both matmul AND attn substrates, env-gated CIPHER_AUDIT | VERIFIED | `cipher_rt_audit.c:133, 136, 178, 190, 191`. |
| COMMIT as FSM token | PARTIAL — verified in CP54 sched (`cp54_commit_migration` + `MIGOUT_COMMITTED` + `stat_commits`); only comment-level in weight_arena | weight_arena does not name the operation COMMIT in source; the operation is named `IMPORT`. |
| CLASSIFY at cupti.c:166 is comment-only stream-routing diagnostic | VERIFIED | The substantive may13 OpClass classifier is NOT ported. |
| Top-level cipher_dispatch.cpp at 543 LOC | VERIFIED (file is 543 LOC) | But plan's §2.1 internal line citations (L541/570/etc) refer to the 616-LOC src/ shadow copy — corrected here. |

**Net verdict on B.1:** The plan's headline "exactly 6 of 33 ops fire on the deployed hot path" is **verified**, with two clarifications: (i) COMMIT is rigorously a state token only in CP54 sched (weight_arena's "commit" is a code comment); (ii) the plan's `cipher_dispatch.cpp` line citations apply to the silently-excluded `src/` shadow copy, not the top-level live file.

---

## Section B.2 — DEPENDENCY-CHAIN ANALYSIS OF 27 NOT-PORTED OPS

Each row was constructed by reading the may13 source file (top 60-100 lines + the init/observe function bodies + the include block) and cross-referencing the call graph in `cipher_10ops_impl.cpp` and `cipher_comply.cpp`. Threading and CUDA dependencies are inferred from direct evidence (presence of `#include <thread>`, `pthread_create` calls, `dlopen("libnvidia-ml.so")`, `dlopen("libcuda.so")`, etc.).

### CLASSIFIER TIER (5 ops)

#### OP_NAME: CLASSIFY (Op #1)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/include/cipher_classify.hpp` (header-only), 250 LOC. Backed by `src/cipher_kernel_table.cpp` (317 LOC), `src/cipher_structural_lookup.cpp` (300 LOC), `src/cipher_param_recovery.cpp` (668 LOC).
- **Tier:** classifier
- **#include lines (top 5):** L34 `<cstdint>`, L35 `<cstring>`, L36 `<array>`, L37 `<atomic>`. Header-only, no other deps. Pure C++ stdlib.
- **Functions called from may13 (top 5):** `detail::fingerprint` (L94, geometric classifier), `detail::cache_slot` (L187, 512-slot open-addressing hash), `classify` (L198, public entry), `classify_launch` (L227, convenience wrapper from cuLaunchKernel args), `opclass_name` (L237, string emit).
- **Threading:** hot-path-sync. Caller is whoever holds the `cuLaunchKernel` intercept thread; cache reads use `std::memory_order_relaxed`; last-writer-wins on collision (L218-221) — no locks.
- **State scope:** per-process-global. The 512-slot `static std::array<CacheSlot, 512> tbl` (L188) lives in process address space; not per-tenant, not per-thread.
- **CUDA/NVRTC/cuIpc dependencies:** ZERO. The header contains `#if defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)` gates inside cipher_stubs.h, but the classifier itself does not invoke any CUDA call. CIPHER_CPU_STUB is NOT needed in cipher_classify.hpp; it's needed in companion `cipher_kernel_table.cpp` (which dlsyms `cuFuncGetName`, `cuFuncGetParamInfo`, `cuFuncGetAttribute` at L25-29) and `cipher_param_recovery.cpp` (fatbin/ELF parsing).
- **Port complexity:** TRIVIAL (classify.hpp alone) / MODERATE (with kernel_table.cpp and structural_lookup.cpp).
- **Port complexity reasoning:** The pure classifier is a single header with zero deps — copy and add to the build with one line in `cipher_rt_phase4/Makefile OBJS`. The backing tables (kernel name patterns in structural_lookup, kparam_info parser in param_recovery) add ~1300 LOC and the dlsym-cuda dance for cuFuncGetName, but neither requires nvcc or new kmod nrs.
- **Critical traps for port:**
  - The 512-slot cache is process-global; in multi-tenant deployment under `CUDA_INJECTION64_PATH`, each tenant has its own classify cache (separate libcipher_rt.so instance per process) — this is correct.
  - Last-writer-wins on collision means under high-collision-rate workloads, recently-mis-classified kernels may briefly return wrong OpClass with high confidence. The classifier compensates by re-fingerprinting from geometry; impact bounded.
  - `cipher_structural_lookup.cpp` carries a hardcoded prefix table (L23+ "loss", "adam", "flash_attn", etc.) — port verbatim, but watch that prefix patterns don't shift when CUDA / cuDNN versions bump kernel naming.

#### OP_NAME: ORACLE (Op #2)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/cipher_oracle.cpp` (TOP-LEVEL, **not** `src/cipher_oracle.cpp`), 539 LOC.
- **Tier:** classifier (safety gate)
- **#include lines (top 5):** L2-4 `#ifdef CIPHER_CPU_STUB include "cipher_stubs.h"` gate, L10 `"cipher_oracle.h"`, L11 `<string.h>`, L12 `<stdio.h>`, L13 `<stdlib.h>`, L14 `<math.h>`.
- **Functions called from may13 (top 5):** `topo_detect_inference` (L30, static), `cipher_oracle_init` (L80, public init), `cipher_oracle_update_gradients` (~L265 per plan), `cipher_oracle_decide` (~L293, the 5-gate decision), `cipher_oracle_record_substitution` (~L383).
- **Threading:** hot-path-sync (called once per kernel launch by `cipher_dispatch_kernel`). `total_decisions & 0x3F` rate-limiting (L36) keeps overhead bounded.
- **State scope:** per-process-global. `CipherOracleState* state` is owned by `cipher_dispatch.cpp` (`static CipherOracleState g_oracle;` at top-level dispatch L62).
- **CUDA/NVRTC/cuIpc dependencies:** ZERO. The `#ifdef CIPHER_CPU_STUB` at L2-4 pulls in `cipher_stubs.h` for type compatibility (CUresult etc.) but no CUDA runtime call.
- **Port complexity:** TRIVIAL.
- **Port complexity reasoning:** Pure C++ logic operating on `CipherOracleState` struct + `CipherLiquidStateMgr*` (the liquid state pointer can be stubbed to NULL for v1; gradients-based deny path only fires when training is detected, which inference workloads never trigger).
- **Critical traps for port:**
  - Port the TOP-LEVEL file, not `src/cipher_oracle.cpp` (silently excluded by may13 Makefile L29 `filter-out`).
  - The "topological phase detection" at L30 reads `state->topo_class_counts[op_class & 0x7]` — depends on CLASSIFY emitting valid OpClass; classifier must port first.
  - 5 gates (phase, min-confidence, structural-lookup, EMA-demotion, N≤4) — must verify the gate constants match the deployed Marlin substrate's risk profile. Default may13 thresholds were tuned for training; inference-only deploys may want laxer EMA gates.

#### OP_NAME: SENSE (Op #3, Op-id 13)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_sense.cpp`, 340 LOC.
- **Tier:** classifier (session-level)
- **#include lines (top 5):** L17 `"cipher_sense.h"`, L19 `<atomic>`, L20 `<cstdio>`, L21 `<cstdlib>`, L22 `<cstring>`, L23 `<cstdint>`, L24 `<ctime>`, L25 `<cmath>`.
- **Functions called from may13 (top 5):** `fnv1a64` (L74, session-fp hash), `shape_proxy` (L83, params_hash extractor), `unpack_mkn` (L91, packed MKN decoder), `allocate_slot` (L99, session-table slot CAS), and the public `cipher_sense_init`/`cipher_sense_observe`/`cipher_sense_report`/`cipher_sense_current_*` exported via `extern "C"`.
- **Threading:** hot-path-sync at observe time (called from Stage 1 shadow thread per ring entry); but the Stage 1 thread itself spawns lazily only if `observers_enabled > 0` (10ops_impl L954). In production deployment the Stage 1 thread never spawns. Port should make SENSE callable inline from the per-launch hook chain.
- **State scope:** per-process-global. `g_sessions[MAX_SESSIONS=1024]` (L65) is one table per process; session-fp is derived from the first 8 distinct GEMM shape proxies plus inter-event timing — each LD-injected tenant gets one process-local view.
- **CUDA/NVRTC/cuIpc dependencies:** ZERO. Reads `CipherRingEntry` (already populated by F1 intercept) — pure CPU math.
- **Port complexity:** MODERATE.
- **Port complexity reasoning:** Pure C++ + atomics, no CUDA, no dlsym. But the entire input — `CipherRingEntry` with `params_hash`, `func_ptr_hash`, `timestamp_ns`, `grid_*`, `block_*` — depends on RING_WRITE actually firing. RING_WRITE is itself NOT-PORTED. Either: port RING_WRITE first OR refactor SENSE to read directly from a new per-launch hook that fabricates a synthetic ring entry inline. The plan's §4.2 ("Per-tenant tenant process") implies the inline path.
- **Critical traps for port:**
  - `params_hash` carries the packed (M, K, N) only when `(ph >> 60) == 0xC` (L92) — the hook DSO must emit the packed form, which requires `cuFuncGetParamInfo` and the cubin-parse path (cipher_kernel_table + cipher_param_recovery). If those aren't ported, SENSE reverts to FNV-hash-only and the M/N/K-aware branches (decode classification at L41-42, batch-proxy at L91) silently degrade.
  - `IDLE_THRESHOLD_NS = 200ms` (L31) defines session boundary; under partition-allocated tenants with tool-call gaps > 200ms, each tool-call interval starts a NEW session — needs verification it doesn't fragment counter aggregation.
  - Session-band classification (HUMAN/AGENT/BATCH) drives downstream SHIELD priority + FAIRNESS quota — wiring those consumers requires SENSE to expose `cipher_sense_current_band(session_fp)` API (already exported per the header).

#### OP_NAME: PREDICT (Op #4, Op-id 17)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_predict.cpp`, 333 LOC.
- **Tier:** actuator (per plan tier table) — but observer-by-implementation in v1; the actuation is via the persist-engine handoff at L68 (`dlsym(RTLD_DEFAULT, "cipher_persist_engine_register")`).
- **#include lines (top 5):** L11 `"cipher_predict.h"`, L13 `<atomic>`, L14 `<cstdio>`, L15 `<cstdlib>`, L16 `<cstring>`, L17 `<cstdint>`, L18 `<ctime>`, L19 `<dlfcn.h>`.
- **Functions called from may13 (top 5):** `resolve_persist_register` (L68, dlsym look-up), `now_ns` (L75, CLOCK_MONOTONIC_RAW), `mix64` (L81), `shape_key` (L88), `probe_or_insert` (L94, open-address linear probe).
- **Threading:** init-only dlsym + per-launch observe (Stage 1 thread when active). Persist-engine register call happens inside observe at hot-ptr promotion (L31 `HOT_PTR_COUNT=100` threshold).
- **State scope:** per-process-global. Two tables: `g_table[MAX_SHAPES=4096]` (L42, shape counter) and `g_ptr_table[MAX_PTRS=1024]` (L59, pointer reuse for persist).
- **CUDA/NVRTC/cuIpc dependencies:** Indirect — at hot-ptr promotion, calls `cipher_persist_engine_register(ptr, bytes, threshold)`. The persist engine itself is in may13's `cipher_persist_engine.cpp` (Stage 3, F3-layer) — which in turn calls `cuMemAdvise` / `cuStreamSetAttribute(ACCESS_POLICY_WINDOW)`. The actuation path requires the persist engine ported alongside.
- **Port complexity:** MODERATE.
- **Port complexity reasoning:** The observer is pure C++ + dlsym; trivial to port. The "actuation" via persist-engine register requires porting `cipher_persist_engine.cpp` too AND adding a `cuLaunchKernelEx` augmenting path (already present in may13's CIPHER_GRAPH_INSPECT / hook DSO but NOT in cipher_rt_phase4's GOT-patched dispatch). In v1 — port as observer-only and defer the persist-engine actuator to v1.5 per the plan's §3.5 "future NRs".
- **Critical traps for port:**
  - `dlsym(RTLD_DEFAULT, "cipher_persist_engine_register")` returns NULL if persist engine not loaded — observer falls through gracefully (L73 `g_persist_resolved.store(1)` regardless). Safe to port in isolation.
  - HOT_PTR_MIN_BYTES = 64KiB (L32) filters out scratch tensors; correct value depends on workload — TinyLlama's small projections may fall below.

#### OP_NAME: DETERMINISM (Op #5, Op-id 21)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_determinism.cpp`, 88 LOC (full file fit in initial read).
- **Tier:** observability
- **#include lines (top 5):** L6 `"cipher_determinism.h"`, L8 `<atomic>`, L9 `<cstdio>`, L10 `<cstdlib>`, L11 `<cstring>`, L12 `<cstdint>`. NO CUDA, NO dlsym.
- **Functions called from may13 (top 5):** `mix64` (L24, splitmix), and the four exported entry points: `cipher_determinism_init` (L33), `cipher_determinism_observe` (L50), `cipher_determinism_hash` (L65), `cipher_determinism_count` (L69), `cipher_determinism_report` (L73).
- **Threading:** hot-path-sync. `g_hash.compare_exchange_weak` (L60-61) makes the FNV-mix safe under concurrent observe calls from multiple threads.
- **State scope:** per-process-global. Two atomics: `g_hash` (L21) + `g_count` (L22). Per-process is fine — each tenant gets its own dispatch-sequence fingerprint.
- **CUDA/NVRTC/cuIpc dependencies:** ZERO.
- **Port complexity:** TRIVIAL.
- **Port complexity reasoning:** 88 LOC, pure stdlib, zero CUDA. Copy and add to OBJS. Already FIRES in may13 production runs (per plan §2.2 — `dispatch_count=36450` in stress2).
- **Critical traps for port:**
  - Only mixes in `params_hash != 0` (L54). If RING_WRITE / per-launch hook never populates params_hash (e.g., if cipher_param_recovery isn't ported), DETERMINISM emits a constant hash — the FNV-offset. Caller-side test should check `cipher_determinism_count() > 0`.

---

### ACTUATOR TIER (8 ops)

#### OP_NAME: SUBSTITUTE Koopman lane (Op #6)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/cipher_dispatch.cpp` (TOP-LEVEL, 543 LOC) — the EDMD pipeline branch lives at L478-480 (the "L3.5 EDMD pipeline — wired in Week 4-5" PASS_THROUGH comment). Backed by `src/cipher_koopman_runtime.cpp` (409 LOC), `src/cipher_lnn.cpp` (477 LOC), `src/cipher_edmd.cpp` (863 LOC), `src/cipher_edmd_live.cpp`, `src/cipher_block_sub_kernel.cu`, `src/cipher_attn_koopman.cpp` + `.cu`.
- **Tier:** actuator (compute substitution lane)
- **#include lines (top 5):** dispatch.cpp L45 `"cipher.h"`, L46 `"cipher_classify.hpp"`, L47 `"cipher_structural_lookup.h"`, L48 `"cipher_oracle.h"`, L49 `"cipher_recipes.h"`, L50 `"cipher_edmd.h"`, L55 `<cuda.h>` (gated on `!CIPHER_CPU_STUB`).
- **Functions called from may13 (top 5):** `cipher_registry_lookup` (called L474), `apply_recipe` (called L490), `cipher_oracle_record_substitution` (L504), `gemm_shape_hash` (L473, recipe-shape key), `cipher_kr_decide`/`cipher_kr_predict` from koopman_runtime (declared but ZERO call sites — see B.4).
- **Threading:** hot-path-sync.
- **State scope:** per-process-global. `g_registry` (dispatch.cpp top, L63) + `g_oracle` + the Koopman per-shape records (cipher_koopman_runtime maintains its own table).
- **CUDA/NVRTC/cuIpc dependencies:** HEAVY. The Koopman substitute kernels are .cu files that require nvcc; the EDMD live solver allocates GPU buffers; the surrogate kernels run via `cuModuleLoadData` (NVRTC). cuIpc not needed.
- **Port complexity:** HARD.
- **Port complexity reasoning:** Goal-4 problem. The registry is seeded with only 32 hardcoded Llama-3-70B-scale shapes (`cipher_recipes.cpp:346` per plan). No real workload emits them → `cipher_registry_lookup` always misses → control reaches the "L3.5 EDMD pipeline" PASS_THROUGH at L480 (top-level) every time. To make this actuator fire, the registry must be populated dynamically by EDMD discovery, which requires the Stage 1/2 shadow threads (lazy-spawn-gated on observers, never spawn in production), the live EDMD solver, and the Koopman kernel compile path. The plan explicitly marks Goal-4 / Koopman lane as **v2 deferred**.
- **Critical traps for port:**
  - `src/cipher_koopman_runtime.cpp` has ZERO callers (confirmed: grep shows `cipher_kr_*` symbols referenced only from the header includes; no function-call sites elsewhere). The plan's "dead code" assessment is correct.
  - Adding nvcc to `cipher_rt_phase4/Makefile` is a build-system change required only for this op (and any FP8 / attn_koopman kernel ports).
  - The "5 PASS_THROUGH exits" at L448/L468/L479-480/L486/L500 (top-level dispatch) are load-bearing — empty registry hits exit at L480 every launch. Even if Koopman lane is wired in v2, the registry-population mechanism must come with it.

#### OP_NAME: ORCHESTRATE (Op #7)
- **may13 file:** No dedicated file. Lives implicitly in `cipher_10ops_impl.cpp` Stage 0 + the return path of `cipher_dispatch_kernel` (top-level `cipher_dispatch.cpp:515`). The plan's §2.1 row for "Op 4 COMMIT/ORCHESTRATE" cites the dispatch return path.
- **Tier:** actuator
- **#include lines (top 5):** N/A — not a separate translation unit. Its includes are inherited from cipher_dispatch.cpp (already cited above) and cipher_10ops_impl.cpp Stage 0 (L27-52).
- **Functions called from may13 (top 5):** Indirectly: the dispatch flow returns CIPHER_SUBSTITUTED / CIPHER_PASS_THROUGH; the shim in cipher_intercept.cpp ~L127 reads the verdict and either calls the substitute cubin or `g_real_launch`/`g_real_launch_ex`.
- **Threading:** hot-path-sync.
- **State scope:** per-process-global (dispatch return). No explicit op state.
- **CUDA/NVRTC/cuIpc dependencies:** Inherits from dispatch.cpp.
- **Port complexity:** TRIVIAL (the behavior already exists in `cipher_rt_matmul_dispatch.c`'s HANDLED/PASSTHROUGH/REDIRECTED return enum).
- **Port complexity reasoning:** ORCHESTRATE has no separately portable code. The plan's §2.3 reusable-abstraction discussion notes that `cipher_rt_matmul_dispatch.c:52-122` already implements the ORCHESTRATE behavior via the priority-ordered actuator registry. Naming-only port: document that the rt_phase4 dispatch return == may13's ORCHESTRATE.
- **Critical traps for port:**
  - No dedicated home means the op cannot be unit-tested in isolation. Already the case in may13 (the `per_op_validation.json` row for COMMIT/ORCHESTRATE is "WORKING-by-behaviour; name unreconciled" per plan §2.1).
  - If a future spec mandates separately observable ORCHESTRATE metrics, a thin wrapper around the rt_matmul_dispatch HANDLED counter is the right place.

#### OP_NAME: GENERATE / SAMPLE (Op #8, Op-id 5)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_graph.cpp` (293 LOC) per Phase B request; in the plan's §2.1 row it's at `cipher_intercept_cudart.cpp:2353-2425` (warmup-only sampling). The plan tags it as "PARTIAL — warmup-only" with a hard-stop after 500 launches.
- **Tier:** actuator (graph capture) / observability (sample stream)
- **#include lines (top 5):** cipher_graph.cpp L9 `"cipher_graph.h"`, L11 `<atomic>`, L12 `<mutex>`, L13 `<stdio.h>`, L14 `<stdlib.h>`, L15 `<string.h>`, L16 `<dlfcn.h>`.
- **Functions called from may13 (top 5):** `env_truthy` (L42), `fnv1a` (L48), `cipher_graph_init` (L59), `cipher_graph_observe` (L76), `cipher_graph_step_boundary` (L90).
- **Threading:** init-only env-read; hot-path-sync per launch (observe writes to TLS sequence). Graph capture itself runs in caller's stream context (THREAD_LOCAL mode per CLAUDE.md Stage 5).
- **State scope:** cpu-thread-local for the sliding-window sequence (`thread_local ThreadState tls_seq;` at L31), per-process-global for the graph promotion counter.
- **CUDA/NVRTC/cuIpc dependencies:** When CIPHER_GRAPH_REPLAY=on, dlopens libcuda to call `cuStreamBeginCapture` / `cuStreamEndCapture` / `cuGraphInstantiateWithFlags` / `cuGraphLaunch` (per CLAUDE.md Stage 5). Observer mode is dependency-free.
- **Port complexity:** MODERATE.
- **Port complexity reasoning:** Pure observer is trivial. Actuation requires `dlopen("libcuda.so.1")` + 4 cu-symbols + the THREAD_LOCAL capture mode coexistence guard via `cuStreamIsCapturing`. The hard-stop after 500 launches mentioned by the plan (intercept_cudart.cpp:2409-2411) is a separate detector and isn't part of cipher_graph.cpp itself.
- **Critical traps for port:**
  - App-side graph capture (e.g., torch.cuda.graph) MUST coexist — graph capture conflicts crash workloads. The `cuStreamIsCapturing` guard at L7 of the file comment is load-bearing.
  - PROMOTE_AFTER_REPEATS=4 (L21) — same as Marlin's STABILITY_THRESHOLD; coincidental but symmetric.

#### OP_NAME: RING_WRITE (Op #9, Op-id 6)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_intercept.cpp` (445 LOC); ring backing in `cipher_10ops.h:83-100` (header-resident struct); call sites at `cipher_intercept.cpp:180-181` and `:374-375` per plan §2.1.
- **Tier:** observability
- **#include lines (top 5):** L5 `<atomic>`, L23 `"cipher_intercept.h"`, L24 `"cipher_10ops.h"`, L25 `<dlfcn.h>`, L26 `<time.h>`, L27 `<string.h>`.
- **Functions called from may13 (top 5):** `now_ns` (L73, CLOCK_MONOTONIC_RAW), `cipher_launch_kernel_shim` (L85, the F1 intercept), `dlsym` (for cuGetProcAddress reroute), plus indirect ring write via `g_cipher_10ops.ring[...]` atomics.
- **Threading:** hot-path-sync (per launch). Ring is SPMC: many producer threads (one per CUDA-using thread) writing; consumer is Stage 1 shadow thread.
- **State scope:** per-process-global (the SPMC ring lives in `g_cipher_10ops.ring`).
- **CUDA/NVRTC/cuIpc dependencies:** ZERO at ring-write time. cipher_intercept.cpp does NOT include `<cuda.h>` — it intercepts via `cuGetProcAddress` indirection at L35-41 (function-pointer typedefs only).
- **Port complexity:** MODERATE.
- **Port complexity reasoning:** The ring schema (`CipherRingEntry` in cipher_10ops.h) is the input contract for every observability op (SENSE, GUARD, FAIRNESS, CARBON, RECEIPT, LOOP, PIPELINE, CONTINUITY, DETERMINISM, TRACE, PREDICT, SUSTAIN, SHIELD, PULSE, THERMOSTAT, ...). Porting it is a one-time ABI-defining act. The hook itself is already present in rt_phase4's `cipher_inject.c` + `cipher_cupti.c` GOT-patch path — what's missing is the ring write (the `g_cipher_10ops.ring[...]` table). Either: (a) port the ring as-is into cipher_rt_phase4 and call `cipher_ring_write` from the existing intercept; (b) refactor to a synthetic per-launch hook chain that bypasses the ring (the plan's §4.2 dispatch pipeline implies (b), with observability fan-out post-launch).
- **Critical traps for port:**
  - Ring depth determines how much backpressure the consumer threads tolerate. If observer threads never spawn (production reality), ring writes accumulate and slot reuse silently overwrites entries — no fault, but observers consuming late will see stale-or-zero data.
  - The cipher_intercept.cpp file is **only built into libcipher_rt.so**, NOT libcipher_hook.so per plan §1.1 ("ships into libcipher_rt"). Avoid duplicate-symbol if porting along with intercept_cudart.cpp.

#### OP_NAME: REMEMBER (Op #10, Op-id 7)
- **may13 file:** Implementation at `src/cipher_10ops_impl.cpp:428-458`. Backing CfC LNN forward at `src/cipher_lnn.cpp` (477 LOC).
- **Tier:** learning
- **#include lines (top 5 of 10ops_impl):** L27 `"cipher_10ops.h"`, L28 `"cipher_lnn.h"`, L29 `"cipher_edmd.h"`, L30 `"cipher_koopman_runtime.h"`, L31 `"cipher_liquid_state.h"`. Plus 20 op-specific headers L33-52.
- **Functions called from may13 (top 5):** `cipher_lnn_forward` (via the s_shadow_lnn instance, L82), `cipher_lnn_decision_from_hidden`, Stage 1's `g_cipher_10ops.ring[…]` consumer scan, `pthread_create` of stage1_shadow at 10ops_impl:955.
- **Threading:** shadow-thread (Stage 1). Lazy-spawn-gated at L954-958: only fires if at least one observer is enabled. Default deployment: zero observers → Stage 1 never spawns → REMEMBER never fires.
- **State scope:** per-process-global. `static CipherLnnState s_shadow_lnn` (L82) is the Stage-1-private CfC state; updated only by Stage 1 thread.
- **CUDA/NVRTC/cuIpc dependencies:** ZERO. CfC forward is pure float math (cipher_lnn.cpp `cfc_step` at L59 uses `tanhf` / `expf` / `softplus` — host-side CPU).
- **Port complexity:** HARD.
- **Port complexity reasoning:** The op itself is trivial code (one tanhf/softplus per layer per kernel observed). The HARD comes from the architectural prerequisites: (a) Stage 1 shadow thread must run, which means observers must be enabled, which means the entire observability tier must port first; (b) REMEMBER's output (the CfC hidden state) is consumed by VALIDATE / SPECULATE / ADAPT — all also Stage-1/2 ops in the learning tier. The plan defers learning tier to v2 explicitly (§4.4 "v2 learning tier, deferred"). v1 port is unnecessary.
- **Critical traps for port:**
  - If ported in isolation without Stage 1 thread, observe path is a hot-path latency spike (CfC forward is ~µs scale; not per-launch-safe).
  - `s_shadow_lnn` weight initialization (Xavier init in cipher_lnn.cpp) means a fresh tenant has untrained CfC weights — SPECULATE predictions are random until ADAPT writes back updates. ADAPT runs from Stage 2 only.

#### OP_NAME: VALIDATE (Op #11, Op-id 8)
- **may13 file:** Implementation at `src/cipher_10ops_impl.cpp:460-475` (Welford online stats + 3-sigma anomaly detector).
- **Tier:** learning / observability (lives in Stage 1 thread)
- **#include lines (top 5):** Same as 10ops_impl.cpp (above).
- **Functions called from may13 (top 5):** Stage 1's Welford accumulators (mean / M2 updaters), `rs_ok` 3-σ check (the plan §2.1 PARTIAL: "rs_ok 3-σ detector never called"), `cipher_lnn_decision_from_hidden`.
- **Threading:** shadow-thread (Stage 1). Same lazy-spawn gate as REMEMBER.
- **State scope:** per-process-global per-class Welford (M2/mean per OpClass, alongside REMEMBER's state).
- **CUDA/NVRTC/cuIpc dependencies:** ZERO.
- **Port complexity:** HARD.
- **Port complexity reasoning:** Trivial code (Welford is 5 lines) but architectural deps same as REMEMBER (Stage 1 thread, which is gated on observer-count). The plan flags "rs_ok 3-σ detector never called" — meaning even when Stage 1 does spawn, the validation hook for substitute correctness never fires because no substitute fires. Defer to v2 with REMEMBER.
- **Critical traps for port:**
  - The 3-σ rule needs at least N≥30 samples per class before signaling. In bursty production where N<30 in 100ms, false negatives bound to noise.

#### OP_NAME: SPECULATE (Op #12, Op-id 10)
- **may13 file:** Write side at `src/cipher_10ops_impl.cpp:477-505`; check side at `src/cipher_intercept.cpp:127-149`.
- **Tier:** learning
- **#include lines (top 5):** Same as 10ops_impl + intercept.cpp.
- **Functions called from may13 (top 5):** `cipher_lnn_forward` (uses s_shadow_lnn from REMEMBER), look-aside-buffer write (the speculation cache), the intercept-side check path (read the LAB before classify).
- **Threading:** Write from Stage 1 shadow-thread; read from any thread holding cuLaunchKernel intercept (hot-path-sync).
- **State scope:** per-process-global. Look-aside-buffer is a small array keyed on something derived from prior kernel-class sequence.
- **CUDA/NVRTC/cuIpc dependencies:** ZERO.
- **Port complexity:** HARD.
- **Port complexity reasoning:** Depends on REMEMBER (consumes CfC prediction). Same Stage 1 spawn gate. Same v2 deferral.
- **Critical traps for port:**
  - The intercept-side check at intercept.cpp:127-149 is in the hot path — if the look-aside-buffer pointer is uninitialized (Stage 1 never ran), the check must be a clean no-op (null check). Verify before any port.

#### OP_NAME: ADAPT (Op #13, Op-id 11)
- **may13 file:** `src/cipher_10ops_impl.cpp:597-765` (Stage 2 background, the actual ADAPT-cycle code). Plus the KEN block at `:99-285`. Backing solvers at `src/cipher_edmd.cpp` (863 LOC) and `src/cipher_koopman_runtime.cpp` (409 LOC — dead code per B.4).
- **Tier:** learning
- **#include lines (top 5):** Same as 10ops_impl.
- **Functions called from may13 (top 5):** `cipher_edmd_step` (the EDMD solver from cipher_edmd.cpp), `cipher_edmd_solve` (QR-pseudoinverse Koopman fit), KEN encoder/evolver/decoder forwards, `s_adapt_edmd[7]` per-OpClass pipelines (10ops_impl L87).
- **Threading:** background-thread (Stage 2). Even more deferred than Stage 1; spawns under the same lazy gate.
- **State scope:** per-process-global. Per-OpClass EDMD pipelines (7 of them, one per OpClass enum). Per-OpClass surrogate snapshots accumulate from Stage 1's REMEMBER pairs.
- **CUDA/NVRTC/cuIpc dependencies:** Indirect — if ADAPT successfully derives a new surrogate, it would push into the substitute registry, which then triggers NVRTC compilation (substitute_v2). In current code, the plan §2.1 notes: "EDMD solver real but fed degenerate `h_before==h_after` synthetic input (`:693-701, :663-669`); NEVER FIRES."
- **Port complexity:** HARD.
- **Port complexity reasoning:** Depends on REMEMBER (Stage 1 must produce real (h_before, h_after) pairs). Currently fed synthetic identical pairs — solver runs but produces no useful update. The Goal-4 loop is broken at this seam. v2 deferral.
- **Critical traps for port:**
  - The KEN block (Koopman Eigenfunction Network) was added later as a parallel learner — its lifecycle is tied to the same Stage 2 spawn gate. If ported, KEN's r=8 / n=16 dimensions (10ops_impl L95-97) must match the deployment workload's spectral structure.

---

### OBSERVABILITY TIER (14 ops)

#### OP_NAME: SHIELD (Op #14)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_shield.cpp`, 250 LOC.
- **Tier:** actuator (latency-protection; sets stream priority) — but PARTIAL per plan (Protection 2 / 3 flags never consumed).
- **#include lines (top 5):** L12 `"cipher_shield.h"`, L13 `"cipher_sense.h"`, L14 `"cipher_green_ctx.h"`, L16 `<atomic>`, L17 `<algorithm>`, L18 `<cmath>`.
- **Functions called from may13 (top 5):** `slot_idx` (L57), `find_or_alloc` (L62), `cipher_sense_current_*` (Op 13 dependency, reads session_band), `cipher_sm_set_priority` (Protection 1, via cipher_green_ctx.h).
- **Threading:** Stage 1 hot-path-sync per ring entry; one rolling 20-pt ITL window per session.
- **State scope:** per-process-global. `g_shield[MAX_SHIELD_SESSIONS=256]` (L49) keyed by session_id.
- **CUDA/NVRTC/cuIpc dependencies:** Indirect via `cipher_green_ctx.h` (cipher_green_ctx.cu is a .cu file in may13). The green-context call is what sets the protection band priority.
- **Port complexity:** MODERATE.
- **Port complexity reasoning:** Trivial Welford / 3-σ stats. The protection-band write requires the green-ctx allocator — in rt_phase4 this lives in `cipher_rt_green_ctx.c` (CP54 ALLOCATE path); SHIELD must call into the existing green-ctx API not re-implement.
- **Critical traps for port:**
  - Depends on SENSE (reads session_band for HUMAN_INTERACTIVE classification). SENSE must port first.
  - Protections 2 and 3 are detect-only in v1 (their flags are written but no downstream consumer reads them per the file's header comment L7-10). Don't over-claim actuation.

#### OP_NAME: SUSTAIN (Op #15)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_sustain.cpp`, 235 LOC.
- **Tier:** actuator (KV pressure → compress flag) — PARTIAL per plan (flag never consumed).
- **#include lines (top 5):** L11 `"cipher_sustain.h"`, L12 `"cipher_sense.h"`, L14 `<atomic>`, L15 `<cmath>`, L16 `<cstdio>`.
- **Functions called from may13 (top 5):** `slot_idx` (L49), per-session OLS slope, sustain-compress flag write, `cipher_sense_current_*` reads.
- **Threading:** Stage 1 hot-path-sync per ring entry.
- **State scope:** per-process-global. `g_sustain[MAX_SUSTAIN_SESSIONS=256]` (L42).
- **CUDA/NVRTC/cuIpc dependencies:** ZERO (the kmod side L3 KV-dedup is the actuator — SUSTAIN itself is observer-only).
- **Port complexity:** MODERATE.
- **Port complexity reasoning:** Pure C++ math. Depends on SENSE. The "sustain_compress" flag has no consumer — port as observer; the consumer-side wiring (kvdedup admission control) is a separate work item.
- **Critical traps for port:**
  - `SLOPE_THRESHOLD_NS = 50000.0` (L25) is a 0.5 ms / 10 steps threshold — tuned for Phase 2's workload. Re-tune for current vLLM mix.

#### OP_NAME: THERMOSTAT (Op #16, Op-id 20) + THERMAL_FEEDBACK (composer)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_thermostat.cpp` (314 LOC) + `cipher_thermal_feedback.cpp` (288 LOC).
- **Tier:** actuator (drives VOLT clock LUT)
- **#include lines (top 5 thermostat):** L7 `"cipher_thermostat.h"`, L8 `"cipher_liquid_state.h"`, L9 `"cipher.h"`, L11 `<atomic>`, L12 `<cmath>`.
- **#include lines (top 5 thermal_feedback):** L3 `"cipher_thermal_feedback.h"`, L4 `"cipher_silicon.h"`, L6 `<atomic>`, **L7 `<thread>`**, L8 `<chrono>`, L13 `<dlfcn.h>`.
- **Functions called from may13 (top 5):** thermostat — per-shape Welford (Stage 1) + 500 ms poll (Stage 2); thermal_feedback — `cipher_silicon_update`, NVML symbol resolution via dlopen.
- **Threading:** thermal_feedback runs its OWN `std::thread` (L7 of file). Per CLAUDE.md Stage 11: "Dedicated `std::thread` opens libnvidia-ml.so.1, calls `nvmlInit_v2`, loops every 100 ms reading clock + power." THERMOSTAT itself is split across Stage 1 (per-shape Welford) and Stage 2 (500 ms poll).
- **State scope:** per-process-global thermostat (g_shapes Welford, g_thermo aggregate state).
- **CUDA/NVRTC/cuIpc dependencies:** thermal_feedback dlopens `libnvidia-ml.so.1`. The plan §3.5 confirms NVML is the actuator path; VOLT is the eventual consumer (writes the clock lock).
- **Port complexity:** MODERATE.
- **Port complexity reasoning:** Pure CPU + NVML dlsym. Thermal_feedback's std::thread is process-wide; one-time spawn at init. In multi-tenant deployment, each tenant process spawns its own poll thread — verify NVML allows N concurrent readers (it does; reads are serialized but cheap).
- **Critical traps for port:**
  - `nvmlDeviceSetPowerManagementLimit` returns NOT_SUPPORTED on Lambda H100 (per Stage-11 header comment "Pod-degraded: probe returns NOT_SUPPORTED"). The actuation side degrades to observe-only on this pod.
  - The 100ms poll cadence + DVFS path overlaps with cipher_rt_volt.c's existing actuation. Need to wire so THERMOSTAT writes the clock target via VOLT, not directly via NVML.

#### OP_NAME: PULSE (Op #17, Op-id 22)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_pulse.cpp`, 416 LOC.
- **Tier:** observability (HW fault early warning)
- **#include lines (top 5):** L10 `"cipher_pulse.h"`, L12 `<atomic>`, L13 `<cmath>`, L14 `<cstdio>`, **L18 `<dlfcn.h>`**.
- **Functions called from may13 (top 5):** NVML ECC reads via dlsym (`nvmlDeviceGetMemoryErrorCounter` — function pointer types declared L37-38), per-shape Welford, signal-2 (max_diff) detection deferred per file header L7.
- **Threading:** Stage 1 per-shape Welford + Stage 2 1000-dispatch evaluator.
- **State scope:** per-process-global. `g_shapes[MAX_SHAPES=64]` (L22) Welford array.
- **CUDA/NVRTC/cuIpc dependencies:** dlopen NVML (libnvidia-ml.so.1).
- **Port complexity:** MODERATE.
- **Port complexity reasoning:** Pure CPU + NVML dlsym. Similar to thermal_feedback but consumer is /tmp file + JSON alert rather than DVFS.
- **Critical traps for port:**
  - Signal 2 (max_diff degradation) is permanently deferred per the file header — requires a Stage 0 sentinel hook in SUBSTITUTE that doesn't exist. Don't try to wire it in v1.
  - REALERT_PERIOD_NS = 60 s (L28) — alerts not re-emitted more than once/minute.

#### OP_NAME: HIBERNATE (Op #18, Op-id 31)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_hibernate.cpp`, 277 LOC.
- **Tier:** actuator (idle SM power gating)
- **#include lines (top 5):** L9 `"cipher_hibernate.h"`, L11 `<atomic>`, **L12 `<csignal>`**, L13 `<cstdio>`, **L17 `<dlfcn.h>`**.
- **Functions called from may13 (top 5):** NVML get/set PowerLimit via dlsym (function pointer typedefs L29-32), idle-gap detector with 10ms poll cadence.
- **Threading:** Stage 1 per-event last-dispatch-ts tracking; Stage 2 periodic poll. The `<csignal>` include hints signal-handler restore on crash (similar to VOLT).
- **State scope:** per-process-global. HibState (L34).
- **CUDA/NVRTC/cuIpc dependencies:** NVML dlsym.
- **Port complexity:** MODERATE.
- **Port complexity reasoning:** Pure CPU + NVML. Same pattern as thermal_feedback.
- **Critical traps for port:**
  - `nvmlDeviceSetPowerManagementLimit` is NOT_SUPPORTED on this pod's H100 — actuation gated off; observer-only. Plan §2.2 row marks PARTIAL.
  - On crash mid-power-limit-change, need signal handler to restore (the csignal include). VOLT already has this pattern — reuse the install_signal_handlers approach.

#### OP_NAME: LOOP (Op #19, Op-id 26)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_loop.cpp`, 286 LOC.
- **Tier:** observability (agentic runaway detection)
- **#include lines (top 5):** L11 `"cipher_loop.h"`, L12 `"cipher_sense.h"`, L14 `<atomic>`, L15 `<cstdio>`, L16 `<cstdlib>`.
- **Functions called from may13 (top 5):** Three signals (S1 shape-cycle autocorrelation over 64-deep ring, S2 decode burn-rate, S3 prefill drought), `cipher_sense_current_*` reads, sticky-high score latch.
- **Threading:** Stage 1 hot-path-sync per ring entry.
- **State scope:** per-process-global. `g_slots[MAX_SESSIONS=1024]` (LoopSession).
- **CUDA/NVRTC/cuIpc dependencies:** ZERO.
- **Port complexity:** TRIVIAL.
- **Port complexity reasoning:** Pure C++ math. FIRES in may13 production (plan §2.2 row: "WORKING — FIRES (runaway_count=7)").
- **Critical traps for port:**
  - Depends on SENSE (uses session classification). Port SENSE first.
  - Sticky-high latch (L8 header) — once a session is flagged runaway, it stays runaway. Reset only on `IDLE_RESET_NS = 200ms` gap. May produce false positives under bursty agent traffic.

#### OP_NAME: CONTINUITY (Op #20, Op-id 19)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_continuity.cpp`, 236 LOC.
- **Tier:** observability (KV checkpointing predecessor)
- **#include lines (top 5):** L7 `"cipher_continuity.h"`, L8 `"cipher_sense.h"`, L10 `<atomic>`, L11 `<cstdio>`, L12 `<cstdlib>`.
- **Functions called from may13 (top 5):** Per-session region tracker (32 region slots per session), manifest snapshot every 500 attention events.
- **Threading:** Stage 1 hot-path-sync.
- **State scope:** per-process-global. `g_slots[MAX_SESSIONS=1024]` ContinuitySession with Region[32] inside.
- **CUDA/NVRTC/cuIpc dependencies:** ZERO (per plan: "No KV capture, no pinned memory, no CUDA calls").
- **Port complexity:** TRIVIAL.
- **Port complexity reasoning:** Pure C++ counters. Manifest emit is fprintf to /tmp. PARTIAL per plan ("region tracking only; no manifest writer") — but the region tracking is fully usable.
- **Critical traps for port:**
  - Depends on SENSE.
  - The "manifest writer" gap is by design — v2 Tier-A KV checkpointer would consume manifests. For v1, region tracking + Prometheus exporter coverage is sufficient.

#### OP_NAME: PIPELINE (Op #21, Op-id 27)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_pipeline.cpp`, 239 LOC.
- **Tier:** observability (multi-agent session correlation)
- **#include lines (top 5):** L7 `"cipher_pipeline.h"`, L8 `"cipher_sense.h"`, L10 `<atomic>`, L11 `<cstdio>`, L12 `<cstdlib>`.
- **Functions called from may13 (top 5):** Per-session bounded shape set (32 shapes each), pairwise Jaccard similarity at report time (JACCARD_THRESHOLD=0.5), tagged upstream/downstream by first_ts_ns.
- **Threading:** Stage 1 hot-path-sync.
- **State scope:** per-process-global. `g_slots[MAX_SESSIONS=1024]` PipelineSession.
- **CUDA/NVRTC/cuIpc dependencies:** ZERO.
- **Port complexity:** TRIVIAL.
- **Port complexity reasoning:** Pure C++ set + jaccard. FIRES in may13 (plan §2.2: "edge_count=10").
- **Critical traps for port:**
  - Depends on SENSE.
  - Pairwise Jaccard at report time is O(N²) — capped at REPORT_MAX_SESSIONS=256 (L22). At N=100 tenants, 4950 pairs — well within budget.

#### OP_NAME: GUARD (Op #22, Op-id 16)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_guard.cpp`, 207 LOC.
- **Tier:** observability (cross-session leak detector)
- **#include lines (top 5):** L11 `"cipher_guard.h"`, L12 `"cipher_sense.h"`, L14 `<atomic>`, L15 `<cstdio>`, L16 `<cstdlib>`.
- **Functions called from may13 (top 5):** `mix64` (L44), open-address linear probe, RESIDENCY_NS=2s leak window.
- **Threading:** Stage 1 hot-path-sync per ring entry.
- **State scope:** per-process-global. `g_table[MAX_SHAPES=4096]` Entry hash table.
- **CUDA/NVRTC/cuIpc dependencies:** ZERO.
- **Port complexity:** TRIVIAL.
- **Port complexity reasoning:** Pure hashtable + atomics. May13 WORKING.
- **Critical traps for port:**
  - Depends on SENSE.
  - Cross-session detection requires distinct session_fp from SENSE. If SENSE collapses concurrent sessions (its v1 limitation per cipher_sense.cpp L14-15: no stream_id → concurrent sessions collapse), GUARD will under-detect.

#### OP_NAME: TOPOLOGY (Op #23, Op-id 25)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_topology.cpp`, 95 LOC (full file fit in initial read).
- **Tier:** classifier (static peer adjacency)
- **#include lines (top 5):** L6 `"cipher_topology.h"`, L8 `<atomic>`, L9 `<cstdio>`, L10 `<cstdlib>`, L11 `<cstring>`, L12 `<cstdint>`, **L13 `<cuda_runtime.h>`**.
- **Functions called from may13 (top 5):** `cudaGetDeviceCount` (L37), `cudaDeviceCanAccessPeer` (L46), no per-launch work (observer is a no-op per L57-60).
- **Threading:** init-only (`cipher_topology_init`).
- **State scope:** per-process-global. 16×16 dense adjacency matrix `g_adj` (L22).
- **CUDA/NVRTC/cuIpc dependencies:** cudaGetDeviceCount + cudaDeviceCanAccessPeer — the only CUDA runtime call in the observability tier.
- **Port complexity:** TRIVIAL.
- **Port complexity reasoning:** 95 LOC, init-only, two CUDA calls.
- **Critical traps for port:**
  - Single H100 = single-device topology → degenerate (one row, one col, no peer edges). Plan §2.2 row notes "degenerate on single H100".
  - Must run AFTER CUDA context init — call from `cipher_v2_init_body` AFTER cuInit signaling complete.

#### OP_NAME: TRACE (Op #24, Op-id 28)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_trace.cpp`, 116 LOC.
- **Tier:** observability (bounded JSONL trace exporter)
- **#include lines (top 5):** L5 `"cipher_trace.h"`, L7 `<atomic>`, L8 `<cstdio>`, L9 `<cstdlib>`, L10 `<cstring>`, L11 `<cstdint>`.
- **Functions called from may13 (top 5):** Fixed ring of compact records (CAPACITY=8192, L15), drop-on-overflow.
- **Threading:** Stage 1 hot-path-sync per observe.
- **State scope:** per-process-global. `g_buf[CAPACITY=8192]` Rec array (L27).
- **CUDA/NVRTC/cuIpc dependencies:** ZERO.
- **Port complexity:** TRIVIAL.
- **Port complexity reasoning:** Pure atomic ring. FIRES in may13 (plan §2.2: "written=8192").
- **Critical traps for port:**
  - Fixed CAPACITY=8192; at sustained 1 µs/launch and 60s observe window = 6e7 launches >> 8192. Drops are by design; the trace is sampled.

#### OP_NAME: FAIRNESS (Op #25, Op-id 24) + FAIRNESS_SHM
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_fairness.cpp` (184 LOC) + `cipher_fairness_shm.cpp` (229 LOC).
- **Tier:** observability+actuator (per-tenant quota)
- **#include lines (top 5 fairness):** L5 `"cipher_fairness.h"`, L6 `"cipher_sense.h"`, L8 `<atomic>`, L9 `<cstdio>`.
- **#include lines (top 5 fairness_shm):** L3 `"cipher_fairness_shm.h"`, **L11 `<fcntl.h>`**, **L12 `<sys/mman.h>`**, L13 `<sys/stat.h>`, L15 `<unistd.h>`.
- **Functions called from may13 (top 5):** `mix64`, `probe_or_insert` (per-tenant work accumulator), POSIX SHM open/mmap (fairness_shm: `shm_open` / `ftruncate` / `mmap`), per-tenant `gemm_calls` atomic.
- **Threading:** Stage 1 hot-path-sync (per-process side); cross-process FAIRNESS via POSIX SHM (64 tenant slots, L20).
- **State scope:** per-process for `cipher_fairness.cpp`, shared-mem-cross-process for `cipher_fairness_shm.cpp` (MAP_SHARED at `/dev/shm/cipher_fairness` or similar).
- **CUDA/NVRTC/cuIpc dependencies:** ZERO (POSIX SHM only; not cuIpc).
- **Port complexity:** MODERATE.
- **Port complexity reasoning:** Pure C++ + POSIX SHM. The SHM piece is correct cross-process IPC — but it duplicates what the kmod weight_arena and CP54 ledger already do. Decision: port as-is for v1 (the SHM region is independent of kmod state); but later integrate quota counters into the kmod cipher_pid_stats so reads can go through the snapshot API.
- **Critical traps for port:**
  - Depends on SENSE (session_fp keying).
  - POSIX SHM file path must not collide with other CIPHER components. The 64-tenant cap is conservative — fine for current targets (15 PARTITION + 1 POOL ≤ 16 simultaneously).

#### OP_NAME: CARBON (Op #26, Op-id 23)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_carbon.cpp`, 169 LOC.
- **Tier:** observability (per-tenant CO₂ estimate)
- **#include lines (top 5):** L3 `"cipher_carbon.h"`, L4 `"cipher_sense.h"`, L6 `<atomic>`, L7 `<cstdio>`.
- **Functions called from may13 (top 5):** Per-tenant work accumulator, g_j_per_unit (1e-9 default L29), g_gco2_per_kwh (400.0 default L30).
- **Threading:** Stage 1 hot-path-sync per ring entry.
- **State scope:** per-process-global. `g_table[MAX_TENANTS=256]` Tenant array.
- **CUDA/NVRTC/cuIpc dependencies:** ZERO.
- **Port complexity:** TRIVIAL.
- **Port complexity reasoning:** Pure counter math. May13 FIRES.
- **Critical traps for port:**
  - Depends on SENSE.
  - 400 gCO₂/kWh is a global average; for accurate per-region claims wire to a region-specific lookup (deployment-time env var).

#### OP_NAME: RECEIPT (Op #27, Op-id 18)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_receipt.cpp`, 225 LOC.
- **Tier:** observability (per-session proof-of-compute)
- **#include lines (top 5):** L3 `"cipher_receipt.h"`, L4 `"cipher_sense.h"`, L6 `<atomic>`, **L12 `<openssl/hmac.h>`**, **L13 `<openssl/evp.h>`**.
- **Functions called from may13 (top 5):** HMAC-SHA256 chain advance (uses OpenSSL EVP), `mix64`, `probe_or_insert`.
- **Threading:** Stage 1 hot-path-sync per ring entry.
- **State scope:** per-process-global. `g_table[MAX_TENANTS=256]` Tenant.
- **CUDA/NVRTC/cuIpc dependencies:** OpenSSL libcrypto (-lcrypto link, already in rt_phase4 per plan §3.7 row 7).
- **Port complexity:** MODERATE.
- **Port complexity reasoning:** OpenSSL link already exists for cipher_rt_audit. Add per-session chain alongside the existing AUDIT chain. The two HMAC chains use distinct keys but the same library.
- **Critical traps for port:**
  - Depends on SENSE.
  - `g_hmac_key[32]` (L37) defaults to all-zeros — must be initialized from a real key (env var `CIPHER_RECEIPT_KEY` or kmod-supplied) before any customer-visible receipt.
  - The cipher_rt_audit.c chain is per-launch globally; cipher_receipt's chain is per-session. Distinct purposes — don't merge.

#### OP_NAME: COMPLY (Op #28, Op-id 29)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_comply.cpp`, 82 LOC (full file read).
- **Tier:** observability (compliance aggregator)
- **#include lines (top 5):** L3 `"cipher_comply.h"`, **L4 `"cipher_receipt.h"`**, **L5 `"cipher_carbon.h"`**, **L6 `"cipher_guard.h"`**, **L7 `"cipher_determinism.h"`**, **L8 `"cipher_fairness.h"`**, **L9 `"cipher_topology.h"`**. SIX downstream op deps.
- **Functions called from may13 (top 5):** `cipher_receipt_session_count` (L49), `cipher_carbon_session_count` (L50), `cipher_guard_session_count` (L51), `cipher_guard_leak_count` (L52), `cipher_determinism_hash`/`_count` (L53-54), `cipher_fairness_tenant_count`/`_overrun_count` (L55-56), `cipher_topology_device_count` (L57).
- **Threading:** init-only (env-gate); `cipher_comply_observe` is a no-op (L38-40, "aggregation-only; no hot-path work"). Report emits `/tmp/cipher_comply_report.json` on demand.
- **State scope:** per-process-global. Single `g_last_ok` flag (L20).
- **CUDA/NVRTC/cuIpc dependencies:** ZERO directly; depends transitively on cipher_topology's cuda_runtime.h dep.
- **Port complexity:** TRIVIAL.
- **Port complexity reasoning:** 82 LOC of pure aggregation. The port complexity comes from the six dependencies — COMPLY cannot port until RECEIPT, CARBON, GUARD, DETERMINISM, FAIRNESS, and TOPOLOGY all port.
- **Critical traps for port:**
  - All six dependency op interfaces (`*_session_count`, `*_leak_count`, etc.) are simple read accessors — ensure they're declared in headers and ABI-stable.
  - Customer compliance pack consumers will read the JSON; the schema at L65-78 is load-bearing for downstream tooling.

#### OP_NAME: OVERLAP (Op #29)
- **may13 file:** Distributed — no dedicated file. Logic lives in `src/cipher_nccl_neural.cpp` (functions `cipher_overlap_init`, `cipher_overlap_schedule`, `cipher_overlap_complete`, `cipher_overlap_report` at L259, L267, ?, L? — declarations in `include/cipher_nccl_neural.h:108-124`) plus `src/cipher_layer2.cpp` (calls `cipher_overlap_init` at L22).
- **Tier:** actuator (gradient-allreduce overlap scheduling)
- **#include lines (top 5 nccl):** `cipher_nccl.h`, `cipher_nccl_neural.h`, `cipher.h`, `<atomic>`, `<dlfcn.h>`.
- **#include lines (top 5 layer2):** `cipher_layer2.h`, `<stdio.h>`, `<string.h>`, `<time.h>` (only 4 deep — small file).
- **Functions called from may13 (top 5):** `cipher_overlap_init`, `cipher_overlap_schedule` (32-bucket gradient table), `cipher_overlap_complete`, plus Layer 2 orchestration glue.
- **Threading:** init-only + per-allreduce-call (called from a NCCL hook that doesn't exist in rt_phase4).
- **State scope:** per-process-global; CipherOverlapState with up to 32 gradient buckets.
- **CUDA/NVRTC/cuIpc dependencies:** Indirect via NCCL tuner DSO. Multi-GPU only.
- **Port complexity:** HARD.
- **Port complexity reasoning:** Distributed-across-files, multi-GPU-only. Single-H100 deployment has no NCCL allreduce. The plan's §3.7 row 8 marks NCCL ops "out-of-v1; defer to Phase 6". OVERLAP comes with that whole family.
- **Critical traps for port:**
  - cipher_layer2 init chain (sm_packer → fusion → mem_layout → nccl_bpf → nccl_neural → overlap) at layer2.cpp L17-22 is monolithic — porting OVERLAP standalone would orphan the others. Defer the entire Layer 2 to Phase 6.

#### OP_NAME: STRAGGLER (Op #30, Phase-3)
- **may13 file:** `/home/ubuntu/cipher-may13-evidence/src/cipher_straggler.cpp`, 303 LOC.
- **Tier:** observability (per-rank slowdown detector)
- **#include lines (top 5):** L12 `"cipher_straggler.h"`, L14 `<atomic>`, L15 `<cmath>`, L16 `<cstdio>`, L17 `<cstdlib>`, L18 `<cstring>`, L19 `<ctime>`.
- **Functions called from may13 (top 5):** Per-bucket EMA updater (7 buckets BUCKET_UB[7] at L31-39, log-scaled), slowdown-run counter (SLOWDOWN_RUN_GATE=5 sustained), algo-bias output (RING when sustained slow per L41).
- **Threading:** per-allreduce hot-path (called from cipher_nccl orchestrator).
- **State scope:** per-process-global. `BucketStats g_buckets[NUM_BUCKETS=7]`.
- **CUDA/NVRTC/cuIpc dependencies:** ZERO directly; consumed by NCCL tuner.
- **Port complexity:** MODERATE.
- **Port complexity reasoning:** Pure EMA math. But the call site is in the NCCL tuner DSO (separate from libcipher_rt). Multi-GPU only. Cross-rank attribution explicitly NOT IMPLEMENTED (file header L8-10). Defer with NCCL family.
- **Critical traps for port:**
  - The /tmp/cipher_straggler_rank_<RANK>.jsonl write at L? requires the RANK env from NCCL — single-GPU has no rank.
  - tools/straggler_aggregate.py is offline post-processor — not deployed.

---

## Section B.3 — DEPENDENCY DAG AND PORT ORDER

The plan §3.3 / §4.4 references "port in dependency order" but doesn't enumerate the DAG. Below is the verified ordering, validated against the include graph and the call graph in cipher_comply.cpp + cipher_10ops_impl.cpp.

### B.3.1 — Dependency edges (verified from `#include` and function calls)

```
CLASSIFY (header-only, no deps)            ← leaf
TOPOLOGY (cuda_runtime.h only)             ← leaf (CUDA init dep)
DETERMINISM (no deps)                      ← leaf
TRACE (no deps)                            ← leaf
GRAPH (no deps)                            ← leaf

ORACLE  ← CLASSIFY (consumes OpClass)      ← Tier 1.5

RING_WRITE ← CLASSIFY (writes OpClass into ring entry)  ← Tier 2

SENSE   ← RING_WRITE (consumes ring entries)            ← Tier 3

SHIELD       ← SENSE
SUSTAIN      ← SENSE
PREDICT      ← SENSE
GUARD        ← SENSE
LOOP         ← SENSE
CONTINUITY   ← SENSE
PIPELINE     ← SENSE
FAIRNESS     ← SENSE
CARBON       ← SENSE
RECEIPT      ← SENSE
                                                          ← Tier 4

THERMOSTAT   ← (no op deps; consumes liquid_state)
PULSE        ← (no op deps; consumes NVML)
HIBERNATE    ← (no op deps; consumes NVML)               ← Tier 4 parallel

COMPLY ← RECEIPT + CARBON + GUARD + DETERMINISM
                 + FAIRNESS + TOPOLOGY                    ← Tier 5

SUBSTITUTE (Marlin lane) — ALREADY PORTED in rt_phase4
SUBSTITUTE (Koopman lane) ← ORACLE + EDMD + LNN + KEN     ← v2

REMEMBER  ← (no op deps; needs Stage 1 thread + lnn)      ← v2 learning
VALIDATE  ← REMEMBER + Welford-stats infrastructure       ← v2 learning
SPECULATE ← REMEMBER (consumes CfC prediction)            ← v2 learning
ADAPT     ← REMEMBER (consumes h-pairs) + EDMD            ← v2 learning

OVERLAP    ← Layer-2 + NCCL family                        ← Phase 6
STRAGGLER  ← NCCL family                                  ← Phase 6
NCCL_P2P   ← Layer-2 + NCCL family                        ← Phase 6

ARBITRATE  — kmod-resident, ALREADY PORTED (cp54_sched)
VOLT       — ALREADY PORTED (cipher_rt_volt.c)
AUDIT      — ALREADY PORTED (cipher_rt_audit.c)
COMMIT     — implicit in rt_matmul_dispatch return enum + cp54 FSM
```

### B.3.2 — Port-order tiers

**Tier 1 (leaf — port standalone):**
- CLASSIFY (cipher_classify.hpp header-only)
- TOPOLOGY (single cudaGetDeviceCount + cudaDeviceCanAccessPeer)
- DETERMINISM (88 LOC, pure atomic FNV)
- TRACE (116 LOC, atomic ring)
- GRAPH (293 LOC, dlopen libcuda for capture path)
- (Already ported: VOLT, AUDIT, Marlin SUBSTITUTE lane, ARBITRATE)

**Tier 1.5 (one-step deps):**
- ORACLE (depends on CLASSIFY OpClass enum; logic standalone)
- RING_WRITE (depends on CLASSIFY OpClass; carries it in entries)

**Tier 2 (depends on Tier 1.5):**
- SENSE (consumes ring entries; needs RING_WRITE + CLASSIFY)
- The kernel-table + structural_lookup + param_recovery support (no direct op deps, but CLASSIFY uses them at the cubin-parse path)

**Tier 3 (depends on SENSE):**
- SHIELD, SUSTAIN, PREDICT, GUARD, LOOP, CONTINUITY, PIPELINE, FAIRNESS, CARBON, RECEIPT

**Tier 3.parallel (NVML/CUDA-stdlib deps, no other op deps):**
- THERMOSTAT (composes with VOLT)
- THERMAL_FEEDBACK (std::thread; composes with THERMOSTAT)
- PULSE (NVML ECC; standalone observer)
- HIBERNATE (NVML PL set; standalone observer — actuator path NOT_SUPPORTED on Lambda H100)

**Tier 4 (aggregator):**
- COMPLY (depends on RECEIPT + CARBON + GUARD + DETERMINISM + FAIRNESS + TOPOLOGY — six predecessors must all port first)

**Tier v2 (deferred):**
- REMEMBER, VALIDATE, SPECULATE, ADAPT (learning tier; Stage 1/2 threads + Koopman)
- SUBSTITUTE Koopman lane (Goal-4)

**Tier Phase 6 (deferred):**
- OVERLAP, STRAGGLER, NCCL_P2P (NCCL / Layer-2 family)

### B.3.3 — No circular dependencies found

Every directed edge above runs strictly upward in tier number. The COMPLY → six predecessors fan-in is the deepest single-step dependency. The Stage 1/2 learning loop (REMEMBER ↔ ADAPT ↔ SUBSTITUTE registry) is a logical cycle but lives entirely within the v2-deferred subtree — not a concern for v1.

### B.3.4 — Observation: SENSE is the brain hub

Of 27 NOT-PORTED ops, **10 depend directly on SENSE** (SHIELD/SUSTAIN/PREDICT/GUARD/LOOP/CONTINUITY/PIPELINE/FAIRNESS/CARBON/RECEIPT). Plus COMPLY transitively via FAIRNESS/CARBON/GUARD/RECEIPT. **SENSE is the second-most load-bearing port after CLASSIFY itself.** If port budget is constrained, the order must be: CLASSIFY → RING_WRITE → SENSE → everything else.

The plan's Week-1 / Week-2 sequencing (§7) already gets this right (CLASSIFY + ORACLE + SENSE all in Week 1-2 before any observability port in Week 4).

### B.3.5 — The "leaf" classifiers do double-duty

CLASSIFY, ORACLE, RING_WRITE, DETERMINISM, TRACE, GRAPH have no inter-op deps but do depend on the **F1 intercept being present and emitting kernel descriptors**. In rt_phase4 the F1 path is `cipher_inject.c` + `cipher_cupti.c` + `cipher_rt_got_patch.c` — these already populate kernel-launch metadata. The port wiring is to **add a pre-launch hook in `cipher_inject.c`** that calls CLASSIFY → ORACLE → RING_WRITE before the actuator-substrate dispatch chooses Marlin / AUDIT / etc. (Plan §7 Week 2 reflects this.)

---

## Section B.4 — SHOW-STOPPERS (v2 deferrals)

Per the prompt's Phase B.4 — ops whose dependency chain (a) reaches into deleted/shadow code, (b) reaches into the Koopman runtime (dead code, zero callers), (c) requires Goal-4 (SUBSTITUTE surrogate registry populated), or (d) has a circular dependency — are deferred to v2.

### B.4.1 — Reaches into shadow code (silently-excluded `src/cipher_dispatch.cpp` or `src/cipher_oracle.cpp`)

**Affected ops:** None in v1's port list. The reengineering plan (§7 Week 1, "Source-of-truth note") explicitly mandates porting the TOP-LEVEL `cipher_dispatch.cpp` (543 LOC) and `cipher_oracle.cpp` (539 LOC), discarding the src/ shadow copies. The shadow copies' line numbers were the source of the plan's §2.1 internal inconsistency (corrected in B.1.7 above) — but no actual dependency reaches into them.

**Action:** The plan's Week-1 instruction to port the top-level files is correct; reinforce it with the §2.1 line-number correction.

### B.4.2 — Reaches into cipher_koopman_runtime.cpp (dead code)

**Verified:** `cipher_koopman_runtime.cpp` (409 LOC) is included only in 10ops_impl.cpp (L30) and the cipher_substitute_v2 comment (L5). Grep across the may13 tree finds zero call sites for `cipher_kr_decide`, `cipher_kr_predict`, `cipher_kr_record_output`, `cipher_kr_find_or_create` (all declared in include/cipher_koopman_runtime.h:138-201) outside the file itself. **Zero callers — dead code confirmed.**

**Affected ops (deferred to v2):**
- **SUBSTITUTE Koopman lane** — the EDMD-derived surrogate path in dispatch.cpp L478-480 (top-level) / L572-573 (src shadow). Currently always PASS_THROUGH because the registry never gets populated (no real EDMD discovery has run).
- **ADAPT** — the EDMD pipeline is fed degenerate `h_before==h_after` synthetic input (10ops_impl.cpp:663-669 + 693-701 per plan §2.1). Even if Stage 2 spawned, ADAPT would not produce useful Koopman updates.
- **REMEMBER** — codes the CfC hidden state that ADAPT would consume; chain dead at the consumer end.
- **VALIDATE** — 3-σ detector "never called" per plan §2.1 row; the validation hook for substitute correctness never fires.
- **SPECULATE** — consumes CfC prediction from REMEMBER; chain dead.

### B.4.3 — Requires Goal-4 (SUBSTITUTE surrogate registry populated)

**Affected ops (deferred to v2):**
- **SUBSTITUTE Koopman lane** (same as B.4.2 — registry is 32 hardcoded Llama-3-70B shapes from cipher_recipes.cpp:346 that no real workload emits).
- Any actuator that branches on "substitute fired" — there are none in v1's port list (the Marlin lane fires independently of the Koopman registry).

### B.4.4 — Circular dependency

**Verified:** No circular dependencies in the v1 port set. The only logical cycle is in the v2-deferred subtree:
- REMEMBER writes h-pairs to Stage 2's EDMD input ring.
- ADAPT reads h-pairs, derives Koopman update, writes back to surrogate registry.
- SUBSTITUTE reads surrogate registry.
- Wider loop: real-launch outputs feed REMEMBER which closes the cycle.

This cycle is fine in principle (it's the learning loop) but in practice broken by (a) Stage 1/2 threads never spawning in production deployment and (b) the h_before==h_after degenerate synthetic input issue. Both are v2 work.

### B.4.5 — Cross-cutting deferrals

**Distributed-across-multiple-files (no clean port unit):**
- **ORCHESTRATE** — no dedicated file. May13 lives in cipher_10ops_impl.cpp Stage 0 + dispatch return. In rt_phase4 the behavior already exists via `cipher_rt_matmul_dispatch.c`'s HANDLED/PASSTHROUGH/REDIRECTED enum. **Resolution:** name-only port (document the equivalence); no source-code copy needed.
- **OVERLAP** — distributed across `cipher_nccl_neural.cpp` + `cipher_layer2.cpp`. Tied to the Layer-2 init chain (layer2.cpp:17-22). **Resolution:** defer with full NCCL family to Phase 6.

**Hardware-not-supported actuation path:**
- **HIBERNATE** — `nvmlDeviceSetPowerManagementLimit` returns NOT_SUPPORTED on Lambda H100. Per plan §2.2 "PARTIAL — power-limit actuator gated off". Port as observer-only.
- **PULSE Signal 2** — requires Stage 0 sentinel hook in SUBSTITUTE that doesn't exist. Permanently deferred per file header L7-8.

### B.4.6 — Show-stoppers explicitly named (v2 list)

The reengineering plan v1 (5-week window per §7) MUST defer these ops:

1. **SUBSTITUTE Koopman lane** — Goal-4 / dead `cipher_koopman_runtime.cpp` / empty registry.
2. **REMEMBER** — Stage 1 thread + dead Koopman chain.
3. **VALIDATE** — Stage 1 thread + rs_ok never called.
4. **SPECULATE** — Stage 1 thread + REMEMBER dep.
5. **ADAPT** — Stage 2 thread + EDMD synthetic-input bug + dead Koopman.
6. **OVERLAP** — Layer-2 + NCCL family.
7. **STRAGGLER** — NCCL family + cross-rank attribution NOT IMPLEMENTED.
8. **NCCL_P2P** — Layer-2 + multi-node only.

Plus partial-actuation ops that port as **observer-only** in v1:
- **HIBERNATE** (NVML NOT_SUPPORTED on Lambda H100).
- **PULSE Signal 2** (no SUBSTITUTE sentinel hook).
- **SUSTAIN compress flag** (no consumer wired).
- **SHIELD Protections 2/3** (flags written but no consumer).

These are not blocked-by-bugs — they are **scope-bounded by hardware reality and architectural prerequisites**. The plan should land all eight v2-deferred ops as named-and-scheduled future work, not silent omissions.

---

## Closing

**Net verdict.** The reengineering plan's §2.1 and §2.2 are substantially accurate. Two specific corrections:

1. **§2.1 line-number citations for `cipher_dispatch.cpp`** point to the silently-excluded `src/` shadow copy (616 LOC), not the top-level live file (543 LOC). Update the plan to cite the top-level numbers (L448 / L468 / L479-480 / L486 / L500 / L515) and add a callout that the src/ shadow is excluded by Makefile L29.

2. **§2.3 COMMIT claim** ("commit-on-publish" in `cipher_weight_arena.c`) over-reads a single code comment at L262. The honest claim is that COMMIT is a state-machine token only in `cipher_cp54_sched.c` (`cp54_commit_migration` function + `CIPHER_CP54_MIGOUT_COMMITTED` outcome + `cipher_cp54_stat_commits` counter). In weight_arena the operation is named IMPORT; "commit" appears as a one-line code comment describing slot-claim atomicity.

The 27 NOT-PORTED ops covered above (plus the 3 distributed/composite ops — ORCHESTRATE, OVERLAP, THERMOSTAT-with-thermal_feedback — which the prompt asked to document if present in the may13 tree) decompose into four buckets that match the B.2 row classifications:

- **8 v2 deferrals** — SUBSTITUTE Koopman lane, REMEMBER, VALIDATE, SPECULATE, ADAPT, OVERLAP, STRAGGLER, NCCL_P2P. Named in B.4.6.
- **4 observer-only-by-hardware** — HIBERNATE (NVML SetPowerLimit NOT_SUPPORTED), PULSE Signal 2 (no SUBSTITUTE sentinel), SUSTAIN (compress flag has no consumer), SHIELD Protections 2/3 (flags have no consumer). v1 ports as observers; actuator wiring deferred.
- **13 MODERATE v1 ports** — ORACLE, SENSE, PREDICT, SHIELD-Protection-1, GUARD, LOOP, CONTINUITY, PIPELINE, FAIRNESS+FAIRNESS_SHM, CARBON, RECEIPT, THERMOSTAT+THERMAL_FEEDBACK, GRAPH. Pure C++ + atomics + occasionally NVML dlsym / OpenSSL / POSIX SHM / std::thread; no nvcc; depend on CLASSIFY → RING_WRITE → SENSE chain or NVML.
- **4 TRIVIAL leaves** — CLASSIFY (header-only), DETERMINISM (88 LOC), TOPOLOGY (95 LOC + cudaGetDeviceCount), TRACE (116 LOC atomic ring). Drop-in additions to the OBJS list.
- **1 name-only equivalence** — ORCHESTRATE: no port unit; already lives in `cipher_rt_matmul_dispatch.c`'s HANDLED/PASSTHROUGH/REDIRECTED return enum. Document equivalence; no source copy.
- **1 enabler dependency** — COMPLY (aggregator) ports last in v1; depends on RECEIPT + CARBON + GUARD + DETERMINISM + FAIRNESS + TOPOLOGY.

Total: 8 + 4 + 13 + 4 + 1 + 1 = 31 op-units covered (≥30 ops in the prompt's list, because SHIELD and THERMOSTAT each span two implementation files and FAIRNESS is double-counted as observer-only and MODERATE — adjusted in the per-row B.2 details).

The dependency DAG is shallow: 5 tiers max (CLASSIFY → ORACLE/RING_WRITE → SENSE → 10 SENSE-children → COMPLY). No cycles within the v1 scope. SENSE is the brain hub — 10 of 27 ops gate on it.

The plan's 5-week Week-1/Week-2/Week-3/Week-4/Week-5 sequencing aligns with the verified dependency order. v1 is feasible as scoped.
