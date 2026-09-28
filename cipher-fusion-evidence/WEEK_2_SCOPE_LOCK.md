# Week 2 Scope Lock

**HEADLINE STATUS: SCOPE LOCKED.**

**Date:** 2026-05-20
**Predecessors:** WEEK_1_CLOSEOUT.md (md5 `6b8ce567792fc54fc28d17f7b2a92210`), WAVE_5_S5_5_VERIFICATION.md (md5 `074845dc7d947ccf450d77caf485e999`), WAVE_5_CB2_ADJUDICATION.md (md5 `41826a32df2140205af67158c853030f`).
**Outcome:** Week 2 scope locked at **7 steps** (LP-2 SDPA refactor → 6 may13 .cpp source ports → classifier substrate scaffolding → observer stub → /proc/cipher/classify_stats kmod node → hot-path CLASSIFY/SENSE/ORACLE wiring → closeout). Part 5 on-disk verification surfaced **zero contradictions blocking Step 1 entry**; one SYNTHESIS-HYPOTHESIS (Wave 5 vs v1.2.2 §7 API-surface naming drift) resolved by adjudication-without-stop in favor of Wave 5's framing because Wave 5's API names reconcile with the disk + ported may13 headers.

---

## Part 1 — v1.2.2 §7 Week 2 baseline scope (verbatim)

`CIPHER_REENGINEERING_PLAN.md` v1.2.2 §7 Week 2 lives at lines 1255-1294.

**Goal (L1259):** "CLASSIFY + SENSE + ORACLE fire on every kernel launch. Telemetry only — no actuator routing change. **Plus the LP-2 attn trampoline refactor (required so the attn lane is unblockable when the first attn substitute actuator lands in v1.5).**"

**Named tasks:**

1. **LP-2 attn trampoline refactor** (L1261-1267). "modify `cipher_rt_attn_dispatch.cpp::flash_call`, `eff_call`, `cudnn_call` (L289, L324, L362) to branch on `route()`'s return value." Add field `void* out_status_devptr` to `cipher_rt_attn_call` (header L80+). About 20 LOC across 3 trampolines. T-W2.4 Track 2 SC6 bit-identical-attention gate is the regression check.

2. **Hot-path classifier wiring** (L1268-1275). "`cipher_inject.c`'s GOT-patched cuLaunchKernel intercept. Add a pre-launch hook:"
   ```c
   cipher_rt_classify_fingerprint_t fp = cipher_rt_classify(kernel_meta);
   cipher_rt_oracle_decision_t dec = cipher_rt_oracle_check(fp, snapshot);
   // Decision NOT yet consumed — log only.
   cipher_rt_classify_record(fp, dec);
   ```

**Stated exit criteria (L1278-1283):**
- W1 regression PASS (no perf change beyond ±3%).
- `/proc/cipher/classify_stats` (new node) reports non-zero classification counts per kernel launch.
- ORACLE EMA stays in `STEADY` after warmup.
- CLASSIFY cache hit rate ≥ 95% after 10s steady-state.

**Stated rollback path (L1285):** "Stub the pre-launch hook to `return CIPHER_PASS_THROUGH` — equivalent to Week 1 state. No anchor change needed for rollback."

**Risk register (L1287-1294):**
- R-W2.1 [MEDIUM]: Hot-path latency spike — measure with cycle counter. Honest budget 100-200 ns/launch (Wave 1 control-flow analysis + `cipher_classify.hpp` "~160 ns cache-hit" header note), accept 1-2% per-tenant overhead as primary case. Alarm 99th-percentile > 500 ns/launch.
- R-W2.2 [MEDIUM]: Per-tenant cache contention — CLASSIFY uses 512-slot lock-free cache; verify under contention with 33-thread test (`cipher_test_phase4_partition_contention`).

---

## Part 2 — §5.5 carryover from Week 1

Per `WAVE_5_S5_5_VERIFICATION.md` §C1, §C2, and the §5 broader carryover section, **9 deliverables** that Wave 5 §5.5 and v1.2.2 §7 attributed to Week 1 did not land in Steps 1, 2 v2, 3, or 4 v2. Of those, the 6 .cpp ports and the 3 new-file/proc-node items are direct prerequisites of the Week 2 hot-path wiring described in Part 1.

### 2.1 — Six may13 .cpp source ports (Week 1 carryover; Wave 5 §5.5 L561-567)

| Source (cipher-may13-evidence) | LOC | Destination (cipher_rt_phase4) | Why for Week 2 |
|---|---|---|---|
| `cipher_dispatch.cpp` (top-level) | 543 | `cipher_rt_phase4/cipher_rt_dispatch.cpp` | Defines `cipher_dispatch` hot-path spine; consumed by Step 6 brain hook |
| `cipher_oracle.cpp` (top-level) | 539 | `cipher_rt_phase4/cipher_rt_oracle.cpp` | Defines `cipher_oracle_decide` (cipher_oracle.cpp:293-377); consumed by Step 6 brain hook |
| `src/cipher_recipes.cpp` | 522 | `cipher_rt_phase4/cipher_rt_recipes.cpp` | 32+ shape recipe registry; consumed by `cipher_registry_lookup` in dispatch |
| `src/cipher_sense.cpp` | 340 | `cipher_rt_phase4/cipher_rt_sense.cpp` | SENSE op (session classifier); consumed by Step 6 SENSE wiring |
| `src/cipher_structural_lookup.cpp` | 300 | `cipher_rt_phase4/cipher_rt_structural_lookup.cpp` | L3.8 fast structural rule table; consumed by ORACLE Gate 3 |
| `src/cipher_kernel_table.cpp` | 317 | `cipher_rt_phase4/cipher_rt_kernel_table.cpp` | Kernel-name resolution via `cipher_kt_observe`; consumed by CLASSIFY chain |

On-disk verification of source files: **all 6 exist at the expected paths and LOCs** (verified in Part 5). No collisions at destination paths (`ls cipher_rt_phase4/cipher_rt_dispatch.cpp` etc. all return "no such file").

### 2.2 — Three new-file / proc-node items (Wave 5 §5.5 Week 1 L570-598)

| Item | Wave 5 ref | Purpose |
|---|---|---|
| `cipher_rt_phase4/cipher_rt_classify_substrate.cpp` (+ `.h`) | L570-573 | Mirrors `cipher_rt_matmul_dispatch.{c,h}` shape. Exports `cipher_rt_classify_register`, `cipher_rt_classify_dispatch`. Priority-ordered classifier registry. No actuator chain yet. |
| `cipher_rt_phase4/cipher_rt_classify_observer.c` | L575-576 | Empty stub. Registers against matmul/attn registries in Step 6. |
| `/proc/cipher/classify_stats` kmod proc node | L677 (Week 2 invariant cites it as "new from Week 1") | Reports non-zero classification counts per kernel launch. Required for v1.2.2 §7 Week 2 behavioral test. |

On-disk verification: neither `cipher_rt_classify_substrate.cpp` nor `cipher_rt_classify_observer.c` exists in `cipher_rt_phase4/`. `/proc/cipher/` currently has 6 nodes (arenas, bar0_state, flops, gpu_state, migrations, stats); no `classify_stats`. **No collisions; no further-order transitive deps detected** (the substrate file is a self-contained registry; the observer file is an empty stub; the proc node extends `cipher_proc.c`'s existing `proc_create` + `cipher_proc_open` pattern).

### 2.3 — Why these carryovers belong in Week 2

The v1.2.2 §7 Week 2 hot-path hook code (Part 1 above) calls 5 functions / types that do not exist in cipher_rt_phase4 today:

- `cipher_rt_classify_fingerprint_t` — undefined
- `cipher_rt_oracle_decision_t` — undefined
- `cipher_rt_classify` (function) — undefined
- `cipher_rt_oracle_check` (function) — undefined
- `cipher_rt_classify_record` (function) — undefined

Wave 5 §5.5 Week 2 L665-672's hook code uses a different naming (cited verbatim in WAVE_5_S5_5_VERIFICATION.md): `cipher::classify_launch` (from ported may13 header), `cipher_oracle_query_t` + `cipher_oracle_result_t` (from ported may13 oracle header), `cipher_rt_oracle_decide` (wraps may13's `cipher_oracle_decide`), `cipher_rt_classify_tls_set` (new helper). Wave 5's naming is consistent with the disk + the headers ported in Step 2 v2; v1.2.2's naming creates 5 new types/functions from scratch.

**The Week 1 .cpp ports + new substrate / observer files together provide the implementation surface that the Week 2 hot-path hook code needs to call into.** Without them, the hook code has nothing to invoke.

---

## Part 3 — Week 2 step decomposition

Seven steps. Each step has a single named deliverable; steps are ordered to keep build-system integrity at every boundary (each step's commit leaves the substrate buildable + CP 5.4 isolation 15/15 PASSing).

### Step 1 — LP-2 SDPA trampoline refactor (independent, lowest risk)

- **Deliverable:** Modify `cipher_rt_attn_dispatch.cpp`'s 3 trampolines (L289-321 flash, L324-359 efficient, L362-398 cudnn) to branch on `route()` return value. Add field `void* out_status_devptr` to `cipher_rt_attn_call` (struct currently ends with `uint64_t reserved[6]` tail at `cipher_rt_attn_dispatch.h:87`; the new field consumes `reserved[0]` or is a named field that absorbs 8 bytes of reserved tail per Cb.2-style pattern).
- **Goal served:** **5 (deployment moat)** primary — preserves substrate pattern for attn lane. **1 (tok/W)** indirect, via v1.5 attn substitute actuator enablement.
- **Estimated time:** 2-3 hours.
- **Dependencies:** None (independent of all other Week 2 steps).
- **Verification gate:** Track 2 SC6 bit-identical-attention test PASSes on Mistral-7B + Llama-3-8B; libcipher_rt.so md5 changes (expected, new code in the .so) but rebuild reproducible; CP 5.4 isolation 15/15 PASS unchanged.

### Step 2 — Six may13 .cpp source ports (Week 1 carryover; foundation for hook code)

- **Deliverable:** Port 6 .cpp source files from cipher-may13-evidence into cipher_rt_phase4 (paths per Part 2.1 above). Apply LP-7 rename equivalents if any in the .cpp files (Step 1's rename was within may13; the ported files already reflect the post-rename state). Apply bare-name include rewrites via the `may13/` prefix per Step 2 v2's closure pattern. Add 6 new OBJS entries + 6 per-source build rules to the Makefile. Each .cpp will likely need `-Iinclude` (already there from Step 3) plus any new -I or -L for transitive deps surfaced during build.
- **Goal served:** **4 (Koopman O(1))** primary — these are the brain implementation files. **5 (deployment moat)** secondary — completes the classifier substrate.
- **Estimated time:** 4-6 hours (the largest step; may surface .cpp-side transitive include closure similar to Step 2 v2's header closure walk).
- **Dependencies:** Step 1 (avoids LP-2 refactor conflict with concurrent .cpp changes, but the two operate on disjoint files so this is a soft ordering — could be parallel).
- **Verification gate:** make clean && make rc=0 in cipher_rt_phase4; libcipher_rt.so md5 will change (new code); nm-diff confirms only the new ported symbols appear; the new ported functions are linkable but not called from any hot-path TU yet; CP 5.4 isolation 15/15 PASS unchanged.

### Step 3 — `cipher_rt_classify_substrate.cpp` + `.h` scaffolding

- **Deliverable:** New file pair mirroring `cipher_rt_matmul_dispatch.{c,h}`. Public API per Wave 5 §5.5 W1 L570-573: `cipher_rt_classify_register(const cipher_rt_classifier_t *cls)` + `cipher_rt_classify_dispatch(const cipher_rt_classify_call *call, cipher_rt_classify_out *out)` + `enum cipher_rt_classify_result { HANDLED, PASSTHROUGH, REDIRECTED, ERROR }`. Priority-ordered registry (16-slot array, same as matmul/attn). No actuator chain wired in this step.
- **Goal served:** **5 (deployment moat)** — substrate pattern is the integration moat.
- **Estimated time:** 2-3 hours.
- **Dependencies:** Step 2 must complete first (the substrate references types from the ported headers).
- **Verification gate:** build rc=0; libcipher_rt.so md5 changes (new substrate code); nm-diff confirms the new substrate symbols appear; CP 5.4 isolation 15/15 PASS unchanged.

### Step 4 — `cipher_rt_classify_observer.c` empty stub

- **Deliverable:** New file. Empty stub that exports a `cipher_rt_classify_observer_init()` function plus a `cipher_rt_classifier_t g_classify_observer = { .name = "classify_observer", .priority = 5, .maybe_classify = NULL };` declaration. The actual `maybe_classify` body is filled in Step 6. This step lands the file as a placeholder so Step 6's wiring has a target.
- **Goal served:** **5 (deployment moat)**.
- **Estimated time:** 1 hour (empty stub).
- **Dependencies:** Step 3 (the observer's type comes from the substrate header).
- **Verification gate:** build rc=0; libcipher_rt.so md5 changes (new TU); observer symbol present but not yet registered with any substrate; CP 5.4 isolation 15/15 PASS unchanged.

### Step 5 — `/proc/cipher/classify_stats` kmod proc node

- **Deliverable:** Extend `cipher_kmod/cipher_proc.c` to add a 7th proc entry: `cipher_proc_classify_stats_entry`, with `cipher_classify_stats_show` + `cipher_classify_stats_open` mirroring the existing pattern (`cipher_proc_show` at L141; `cipher_proc_open` at L356). Per-tenant classify-count fields land in `cipher_pid_stats` reserved tail (extends the Cb.2 reserved tail or adds a new top-level reserved region — TBD by step author with sizeof preservation discipline). The show function emits one row per tenant: `pid tenant_id classify_count_by_class[GEMM ATTENTION ELEMENTWISE ...] cache_hit_rate confidence`. Kmod ABI extension; cipher_kmod.ko md5 will change.
- **Goal served:** **5 (deployment moat)** primary (observability moat). **1, 3, 4 indirect** — the data carried by this node feeds dispatch decisions in later weeks.
- **Estimated time:** 2-3 hours.
- **Dependencies:** None on Steps 1-4 (kmod-side, independent of rt_phase4 changes).
- **Verification gate:** kmod rmmod + insmod clean; `/proc/cipher/classify_stats` exists and is readable; per-tenant rows appear; CP 5.4 isolation 15/15 PASS unchanged.

### Step 6 — Hot-path CLASSIFY + SENSE + ORACLE wiring

- **Deliverable:** Three wiring landings:
  1. **Matmul substrate registration:** in `cipher_inject.c::cipher_v2_init_body`, add `cipher_rt_classify_observer_init()` between `cipher_rt_matmul_dispatch_init` (step 6 on disk; Wave 5 said step 7) and `cipher_rt_marlin_init` (step 7 on disk; Wave 5 said step 8). The observer registers at priority 5 on the matmul registry per Ca.7.
  2. **Attn substrate registration:** same pattern, between `cipher_rt_attn_dispatch_init` (step 8 on disk) and `cipher_rt_attn_test_actuator_init` (step 9 on disk). Priority 5 on the attn registry per Ca.8.
  3. **CUPTI generic hook:** add brain hook body inside `cipher_cupti.c::cipher_v2_cupti_cb` (defined at L67). The body calls (per Wave 5 §5.5 W2 L665-672 — naming reconciled with disk):
     ```c
     const cipher::ClassifyResult cr = cipher::classify_launch(fn, grid, block, smem);
     cipher_oracle_query_t q = { layer_idx, total_layers, (uint8_t)cr.op, cr.confidence, kname, is_back, false };
     cipher_oracle_result_t dec = cipher_oracle_decide(&g_oracle, &q);
     cipher_rt_classify_record(cr.op, cr.confidence, dec);  /* TLS publish + /proc/cipher/classify_stats update */
     ```
  Telemetry only. No actuator routing change. Observer returns PASSTHROUGH on every call.
- **Goal served:** **5 (deployment moat)** primary (substrate completion); **1, 3, 4 indirect** (telemetry foundation that Week 3 dispatch routing builds on).
- **Estimated time:** 4-6 hours.
- **Dependencies:** Steps 2 (ported .cpp), 3 (substrate), 4 (observer stub), 5 (proc node target).
- **Verification gate:** all four v1.2.2 §7 Week 2 behavioral tests PASS — W1 regression within ±3%; `/proc/cipher/classify_stats` reports non-zero counts; ORACLE EMA stays in STEADY after warmup; CLASSIFY cache hit rate ≥ 95% at 10s steady-state. Plus Track 2 SC6 bit-identical (LP-2 invariant carried forward from Step 1). Plus CP 5.4 isolation 15/15 PASS. Plus 99th-percentile cycle-counter probe < 500 ns/launch (R-W2.1 alarm).

### Step 7 — Week 2 closeout

- **Deliverable:** WEEK_2_CLOSEOUT.md + tag `week-2-complete` on all three trees (cipher-may13-evidence stays at week-1-complete since no may13 changes in Week 2; cipher_rt_phase4 and cipher_kmod get the new tag).
- **Goal served:** none directly; integration discipline.
- **Estimated time:** 1-2 hours.
- **Dependencies:** Steps 1-6.

---

## Part 4 — Goal traceability matrix

The 5 product goals from the brief: 1=tok/W, 2=100-tenant density, 3=MFU, 4=Koopman O(1), 5=deployment moat.

| Step | 1 tok/W | 2 density | 3 MFU | 4 Koopman | 5 moat |
|---|:---:|:---:|:---:|:---:|:---:|
| 1 LP-2 SDPA refactor | ⬤ (v1.5 attn substitute enabler) | — | — | — | ⬤ (substrate pattern preserved) |
| 2 Six .cpp ports | — | — | — | ⬤ (brain implementation foundation) | ⬤ (substrate completion) |
| 3 Classify substrate | — | — | — | — | ⬤ (substrate pattern) |
| 4 Observer stub | — | — | — | — | ⬤ |
| 5 classify_stats proc node | indirect | indirect | indirect | — | ⬤ (observability moat) |
| 6 Hot-path wiring | ⬤ indirect | — | ⬤ indirect | ⬤ indirect | ⬤ (substrate completion) |
| 7 Closeout | — | — | — | — | — (discipline only) |

**No step fails the goal-traceability test.** Steps 1-6 all serve goal 5 (deployment moat). Step 6 indirectly serves goals 1/3/4 (the telemetry it lands is the input for Week 3+ actuator routing).

The brief's hint that any step not serving a named goal should be flagged for removal: Step 7 (closeout) is the only step with no goal mapping; it is integration discipline, not a product deliverable. Recommend keeping it (consistency with Week 1's closeout pattern) but acknowledge it carries no direct product value.

---

## Part 5 — Pre-emptive on-disk verification (Step-1-readiness sweep)

For every named target in Week 2 Steps 1-7, the following was verified on disk:

| # | Target | Status | Notes |
|---|---|:---:|---|
| 5.1 | `cipher-may13-evidence/cipher_dispatch.cpp` (Step 2) | VERIFIED | 543 LOC top-level (post-Step-1 LP-7 rename state) |
| 5.2 | `cipher-may13-evidence/cipher_oracle.cpp` (Step 2) | VERIFIED | 539 LOC top-level |
| 5.3 | `cipher-may13-evidence/src/cipher_recipes.cpp` (Step 2) | VERIFIED | 522 LOC |
| 5.4 | `cipher-may13-evidence/src/cipher_sense.cpp` (Step 2) | VERIFIED | 340 LOC |
| 5.5 | `cipher-may13-evidence/src/cipher_structural_lookup.cpp` (Step 2) | VERIFIED | 300 LOC |
| 5.6 | `cipher-may13-evidence/src/cipher_kernel_table.cpp` (Step 2) | VERIFIED | 317 LOC (post-Step-1 rename to CipherKtEntry) |
| 5.7 | `cipher_rt_matmul_dispatch.{c,h}` (template for Step 3 substrate mirror) | VERIFIED | Exists |
| 5.8 | `cipher_rt_phase4/cipher_rt_classify_substrate.cpp` collision check | VERIFIED no collision | Does not exist; Step 3 creates fresh |
| 5.9 | `cipher_rt_phase4/cipher_rt_classify_observer.c` collision check | VERIFIED no collision | Does not exist; Step 4 creates fresh |
| 5.10 | `/proc/cipher/classify_stats` collision check | VERIFIED no collision | Does not exist; Step 5 creates fresh; 6 existing nodes (arenas, bar0_state, flops, gpu_state, migrations, stats) untouched |
| 5.11 | `cipher_kmod/cipher_proc.c` proc-emitter template (Step 5) | VERIFIED | `cipher_proc_open` at L356; `cipher_proc_show` at L141; 6 existing entries follow the same pattern Step 5 extends |
| 5.12 | `cipher_rt_attn_dispatch.cpp` 3 trampoline entry points (Step 1) | VERIFIED EXACT | Flash trampoline begins at L289 (function declaration L290), Eff at L324 (declaration L325), Cudnn at L362 (declaration L363). Wave 5's cited lines match disk exactly. |
| 5.13 | `cipher_rt_attn_dispatch.cpp` `orig(...)` return sites (Step 1 refactor targets) | VERIFIED | L321 (flash), L359 (efficient), L398 (cudnn). The LP-2 refactor branches before these on `route()` return. |
| 5.14 | `cipher_rt_attn_dispatch.h::cipher_rt_attn_call` struct (Step 1 add field) | VERIFIED | Struct at L73; ends with `uint64_t reserved[6]` tail at L87. `out_status_devptr` (8 bytes) can consume `reserved[0]` (8-byte slot) preserving sizeof, mirroring the Cb.2 Option A pattern from Step 4 v2. |
| 5.15 | `cipher_inject.c::cipher_v2_init_body` (Step 6 substrate registration site) | VERIFIED | Init body at L34; 10 init calls; the matmul/attn registration points are between adjacent calls per the existing ordering. **Off-by-one vs Wave 5** (Wave 5 said step 7=matmul, 8=marlin; disk shows 6=matmul, 7=marlin) — non-blocking, Step 6 uses function names. |
| 5.16 | `cipher_cupti.c::cipher_v2_cupti_cb` (Step 6 CUPTI brain hook site) | VERIFIED | Function at L67; this is the canonical cuLaunchKernel/cudaLaunchKernel CUPTI callback. **v1.2.2 §7 W2 says "cipher_inject.c's GOT-patched cuLaunchKernel intercept" but cipher_rt_phase4's GOT patching is for cublasGemmEx + SDPA, not cuLaunchKernel** — the cuLaunchKernel path is via CUPTI callback per Wave 5 §5.5 W2 L645. SYNTHESIS-HYPOTHESIS resolved: implement per Wave 5's framing (CUPTI callback hook), not v1.2.2's "cuLaunchKernel intercept" framing. |
| 5.17 | `cipher_test_phase4_partition_contention` (R-W2.2 verification harness) | VERIFIED | Exists at `/home/ubuntu/cipher_phase4_tests/cipher_test_phase4_partition_contention.c`. |
| 5.18 | Hook-code API surface (v1.2.2 vs Wave 5) | **SYNTHESIS-HYPOTHESIS** | v1.2.2 §7 W2 names 5 new types/functions (`cipher_rt_classify_fingerprint_t`, `cipher_rt_oracle_decision_t`, `cipher_rt_classify`, `cipher_rt_oracle_check`, `cipher_rt_classify_record`) that do not exist. Wave 5 §5.5 W2 names 4 types/functions, 3 of which (`cipher::classify_launch`, `cipher_oracle_query_t`, `cipher_oracle_result_t`) exist in the ported may13 headers and 1 of which (`cipher_rt_classify_tls_set`) is new. **Resolution:** implement per Wave 5 framing because it reconciles with the disk. |

**Aggregate:**

| Category | Count | Items |
|---|---|---|
| VERIFIED | 16 | 5.1-5.14, 5.17 |
| SYNTHESIS-HYPOTHESIS (non-blocking; resolved by Wave 5 framing preference) | 2 | 5.15 (init step numbering), 5.16+5.18 (API surface + hook point) |
| CONTRADICTION (blocking Step 1) | 0 | — |

**Scope can be locked.** Step 1 is independent of both SYNTHESIS-HYPOTHESIS items (LP-2 refactor doesn't touch init order or hook code). Step 6 (the wiring step that consumes both items) is the latest in the dependency chain, leaving full Week-2 internal time for any further reconciliation before its execution.

---

## Part 6 — Open items deferred to Weeks 3-5

### 6.1 — Wave 5 §5.5 CONTRADICTION items (from `WAVE_5_S5_5_VERIFICATION.md` §C3, §C4)

| # | Item | Deferred to | Reason for deferral |
|---|---|---|---|
| C3 | `cipher-may13-evidence/src/cipher_audit.cpp` "retirement" (Wave 5 §5.5 W4 L786) | Week 4 entry-window | File does not exist; AUDIT inline at `cipher_10ops_impl.cpp:341-378`. No retirement work to perform. Week 4 entry should adjudicate whether Wave 5's framing is moot or whether the inline AUDIT block is the retirement target. |
| C4 | Slot array naming `cipher_partition_slot[32]` (Wave 5 §5.5 W4 L812) | Week 4 entry-window | Actual variable is `cipher_slots[CIPHER_PARTITION_SLOTS_MAX]` with size 33. Wave 5 conflated the struct type name with the array variable name and got the count wrong. Week 4 LP-8 retirement must use the actual variable name + size. |

### 6.2 — Wave 5 §5.5 SYNTHESIS-HYPOTHESIS items not consumed in Week 2

| # | Item | Target week |
|---|---|---|
| H3 | `_tick()` is actually `cipher_partition_tick()` (Wave 5 §5.5 W4 L814) | Week 4 |
| H4 | `cipher_rt_attn_call` `out_status_devptr` location "L80+" approximate | **Consumed in Week 2 Step 1** — placed in `reserved[6]` tail or as named field at sizeof-preserving position. |
| H1 | `cipher_v2_init_body` step numbering off-by-one | **Consumed in Week 2 Step 6** — Step 6 uses function names, not step numbers, for substrate-registration insertion points. |
| H2 | Attn trampoline function-name paraphrases (`flash_call` etc.) | **Consumed in Week 2 Step 1** — Step 1 uses the actual C++-mangled names (`_ZN2at4_ops35_scaled_dot_product_flash_attention4call...`) verified exactly at L289/L324/L362. |

### 6.3 — Test infrastructure gap (deferred from Week 1)

`cipher-may13-evidence/tests/test_kernel_table.py` + `test_pattern6_paraminfo.py` hardcode `/home/ubuntu/op31-prod-fix/libcipher_hook.so` (historical dev pod path). Documented in `WEEK_1_CLOSEOUT.md` §5.3. Defer path-rewrite to Week 4 (observability port) per Week 1 recommendation.

### 6.4 — DKMS image staleness (deferred from Week 1 pre-flight)

DKMS-installed kmod at `/lib/modules/.../cipher_kmod.ko` is stale (21× smaller, mtime 2026-05-16). Documented in `WEEK_1_PRE_FLIGHT.md` §1.4 and `WEEK_1_CLOSEOUT.md` §7.4. Refresh deferred to Week 5 (CP 5.5 benchmark planning).

### 6.5 — Plan v1.2.2 §7 Week 2 audit-trail note (no edit proposed here)

If the user wishes, a v1.2.3 documentation patch could update the plan §7 Week 2 hook code from the unrealizable `cipher_rt_classify_fingerprint_t` / `cipher_rt_classify` / `cipher_rt_oracle_check` / `cipher_rt_classify_record` framing to the Wave 5 framing actually implementable on disk (`cipher::classify_launch` / `cipher_oracle_decide` / `cipher_rt_classify_tls_set`). This is a documentation-only correction; the v1.2.2 plan ships as v1.2.3 with a §7 W2 hook-code paragraph correction footnote. Surfaced for user discretion; not proposed as a Week 2 blocker.

---

## Headline summary

| Field | Value |
|---|---|
| Status | **SCOPE LOCKED** |
| Step count | 7 |
| Estimated total time | 16-24 hours across 2-3 calendar days |
| New files | 4 (cipher_rt_classify_substrate.cpp/.h, cipher_rt_classify_observer.c, /proc/cipher/classify_stats) |
| Modified files | 8 (cipher_rt_attn_dispatch.cpp + .h, Makefile, cipher_proc.c, cipher_inject.c, cipher_cupti.c, plus 6 ported source files which are new in cipher_rt_phase4 but each is a copy of a may13 source) |
| Trees touched | 2 (cipher_rt_phase4 + cipher_kmod). cipher-may13-evidence stays at week-1-complete (no Week 2 may13 changes). |
| Goal coverage | 1, 3, 4, 5 (indirect on most; direct on 5 for every step except closeout) |
| Adjudication items remaining | 0 blocking; 2 SYNTHESIS-HYPOTHESIS resolved by Wave 5 framing preference (documented in Part 5 §5.18 and §5.16) |

### Week 2 Step 1 prompt readiness

The next prompt (Week 2 Step 1: LP-7 — sorry, **Step 1: LP-2 SDPA trampoline refactor**) can be drafted from this locked scope. Key inputs:

- Files touched: `cipher_rt_phase4/cipher_rt_attn_dispatch.cpp` (L289-321, L324-359, L362-398) + `cipher_rt_attn_dispatch.h` (L73-89)
- Verification gate: Track 2 SC6 bit-identical (T-W2.4) + CP 5.4 isolation 15/15
- Library invariant: libcipher_rt.so md5 will change (new code) — no fixed-anchor expectation
- Rollback path: `git reset --hard week-1-complete`

---

**End of WEEK_2_SCOPE_LOCK.md.**
