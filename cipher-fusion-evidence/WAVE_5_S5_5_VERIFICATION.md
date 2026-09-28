# Wave 5 §5.5 Weeks 2-5 verification sweep

**Date:** 2026-05-20
**Purpose:** Pre-emptively sweep Wave 5 §5.5 Weeks 2-5 entries for the same class of synthesis errors that produced the Cb.2 contradictions surfaced in Week 1 Step 4. Pure verification; no edits proposed.
**Categories used:**
- **VERIFIED** — Wave 5's named target exists on disk with the structure / location / name Wave 5 describes.
- **SYNTHESIS-HYPOTHESIS** — target exists but Wave 5's specifics drift in non-blocking ways (off-by-one line numbers, paraphrased names, minor count discrepancies). Execution can proceed with awareness.
- **CONTRADICTION** — Wave 5's named target does not exist on disk, or the disk structure cannot be reconciled with Wave 5's description without adjudication.

---

## Aggregate result table

| Week | Wave 5 ref | Target | Status | Notes |
|---|---|---|---|---|
| 2 | L640-644 | `cipher_v2_init_body` init step numbers 7-10 | **SYNTHESIS-HYPOTHESIS** | Off-by-one: disk shows steps 6-9. Functions exist and relative order correct. |
| 2 | L645 | `cipher_cupti.c::cipher_v2_cupti_cb` | VERIFIED | Defined at `cipher_cupti.c:67`. |
| 2 | L650-651 | `cipher_rt_attn_dispatch.cpp::flash_call/eff_call/cudnn_call` at L289/L324/L362 | **SYNTHESIS-HYPOTHESIS** | Functions exist with C++-mangled names; Wave 5 paraphrases (`flash_call` etc.) are short-form. Lines approximate. |
| 2 | L662-663 | `cipher_rt_attn_call` struct + add `void* out_status_devptr` field | VERIFIED (existing) + NEW WORK | Struct at `cipher_rt_attn_dispatch.h:73`. Field does not exist yet — Week 2 work to add. |
| 2 | L665-672 | Brain hook bodies (`classify_launch`, `cipher_oracle_query_t`, `cipher_rt_oracle_decide`, `cipher_rt_classify_tls_set`) | **MIXED** | `classify_launch` exists in ported header `include/may13/cipher_classify.hpp`; `cipher_rt_oracle_decide` / `cipher_rt_classify_tls_set` not yet created (Week 2 work to author). |
| 2 | L677 | `/proc/cipher/classify_stats` (claimed "new node from Week 1") | **CONTRADICTION** | Node does NOT exist on disk. Wave 5 said Week 1 would create it; Step 1-4 v2 did NOT create it (none of those steps touched cipher_proc.c). |
| 2 | L710 | `cipher_test_phase4_partition_contention.c` (33-thread harness) | VERIFIED | Exists at `/home/ubuntu/cipher_phase4_tests/cipher_test_phase4_partition_contention.c`. |
| 3 | L729 | `cipher_rt_classify_observer.c` (claimed Week 1 created empty stub) | **CONTRADICTION** | File does NOT exist on disk. Wave 5 Week 1 said it would be created; Step 1-4 v2 did NOT create it. |
| 3 | L732 | `cipher_rt_marlin_actuator.c::maybe_handle_marlin` (Ca.9) | VERIFIED | Function at `cipher_rt_marlin_actuator.c:56`, registered at L173. |
| 3 | L735-737 | VOLT `CIPHER_VOLT_BATCH=1` env + `cipher_volt.c:54-63` LUT | VERIFIED | `batch_to_mhz` at `cipher_volt.c:54-63` exactly; case 1→1000 MHz, 8→1600, 32/64→1980 MHz. Env var read at `cipher_volt.c:256`. |
| 3 | L739 | `cipher_cp54_sched.c::COMPACT_MIGRATE` ioctl nr 20 | VERIFIED | Comment `nr 20 — COMPACT_MIGRATE` at `cipher_cp54_sched.c:794`. |
| 3 | L775 | `cipher_cp54_mig_ratelimit_ms = 10000` | VERIFIED | Defined at `cipher_cp54_sched.c:97`. |
| 4 | L786 | `cipher-may13-evidence/src/cipher_audit.cpp` (target of retirement) | **CONTRADICTION** | File does NOT exist. AUDIT in may13 lives inline in `cipher_10ops_impl.cpp:341-378` (function `audit_chain_update`). Wave 5 named the wrong file to retire. |
| 4 | L789-799 | 11 `cipher_<op>.cpp` source files in `cipher-may13-evidence/src/` (TRACE, RECEIPT, CARBON, FAIRNESS, FAIRNESS_SHM, GUARD, COMPLY, LOOP, PIPELINE, DETERMINISM, CONTINUITY, PULSE) | VERIFIED | All 12 .cpp files verified to exist with sizes 3 KB to 16 KB. (FAIRNESS counted as 2 files — `.cpp` + `_shm.cpp`.) |
| 4 | L803 | `cipher_10ops.h` 65536-entry SPMC ring | VERIFIED | `CIPHER_RING_SIZE 65536u` at `cipher_10ops.h:72`; ring buf at `:79`. |
| 4 | L804 | `cipher_10ops_impl.cpp::stage1_shadow` | VERIFIED | Function at `cipher_10ops_impl.cpp:385`; spawned at `:955`. |
| 4 | L810 | `cipher_dev_request_sm_partition` handler in `cipher_dev.c` | VERIFIED | At `cipher_dev.c:107`; returns -ENOSYS per `:110`. |
| 4 | L812 | slot array `cipher_partition_slot[32]` | **CONTRADICTION** | Type `struct cipher_partition_slot` at `cipher_partition_allocator.c:88`; the ARRAY variable is `cipher_slots[CIPHER_PARTITION_SLOTS_MAX]` at `:91`. Size is 33 (H100 132 SMs / 4 = 33 slots), not 32. Wave 5 conflated the struct name with the array name AND got the count wrong. |
| 4 | L814-816 | `_tick()` function in cipher_partition_allocator.c | **SYNTHESIS-HYPOTHESIS** | Function actually named `cipher_partition_tick()` at `cipher_partition_allocator.c:429`. Wave 5's `_tick()` is a paraphrase. |
| 4 | L815 | `cipher_state_updater.c::cipher_state_updater_fn` | VERIFIED | Function at `cipher_state_updater.c:139`; spawned at `:194`. |
| 4 | L817-820 | `cipher_pid_stats::sm_partition_mask/count` + `cipher_set_sm_partition_mask` at `cipher_tenant_snapshot.c:207` | VERIFIED (precise line match) | Fields at `cipher_internal.h:144-145`; function at `cipher_tenant_snapshot.c:207` exactly. |
| 4 | L837 | `/proc/cipher/fairness` (new entry, Week 4 deliverable) | NEW WORK | Does not exist today; Wave 5 says Week 4 creates it. No verification needed at this point. |
| 5 | L874 | 100-tenant launcher | VERIFIED | `cipher_measurement/density_pack.sh` exists; N=2 → 100 sweep. |
| 5 | L879 | Track 2 weight HBM measurement | VERIFIED | `phase_c/sc6_*.py` infrastructure (sc6_run.py, sc6_models.py, sc6_consumer.py, sc6_independent_tenant.py, sc6_aggregate.py, sc6_producer.py) exists. |
| 5 | L880 | KV-prefix dedup hit rate harness | VERIFIED | `cp_4_6_5_6/t466_dedup_scale.c` and `t466_isolation.c` exist. |
| 5 | L881 | 24-hour soak G4 gate | VERIFIED | `cipher_measurement/soak_24h.sh` exists. |
| 5 | L875 | vLLM 100-instance baseline (2× fleet tok/W gate) | VERIFIED | vLLM env at `/home/ubuntu/vllm_env/`; baseline harness in `future_scope_a/phase3_*.py`. |

**Aggregate counts:**

| Category | Count |
|---|---|
| VERIFIED | 18 |
| SYNTHESIS-HYPOTHESIS | 4 |
| CONTRADICTION | 4 |
| NEW WORK (Week 2 / Week 4 to create) | 4 |

---

## CONTRADICTION details — surfaced for Week 2 entry-window adjudication

The four CONTRADICTION items are surfaced verbatim. **No mitigation proposed in this document.**

### C1 — `/proc/cipher/classify_stats` does not exist

**Wave 5 reference:** §5.5 Week 2 L677 says: "`/proc/cipher/classify_stats` (new node from Week 1) reports non-zero classification counts per kernel launch."

**Disk evidence:** `/proc/cipher/` contains: `arenas, bar0_state, flops, gpu_state, migrations, stats`. **No `classify_stats` node.** `grep -rn classify_stats /home/ubuntu/cipher_kmod/` returns zero hits.

**Why this surfaced now:** Wave 5's Week 1 scope (§5.5 L568-598) included two new files for `cipher_rt_phase4` (`cipher_rt_classify_substrate.cpp` and `cipher_rt_classify_observer.c`) and the new `/proc/cipher/classify_stats` node. None of these landed in Steps 1, 2 v2, 3, or 4 v2. Step 3 created `cipher_may13_harness.cpp` as the cross-tree compile harness; the Wave-5-named classify substrate and observer files were not within the brief's scope for any of the four Week 1 steps. The `/proc/cipher/classify_stats` node was never created.

**Implication for Week 2:** The Week 2 lossless invariant I-W2.2 (`/proc/cipher/classify_stats | awk '$2 > 0'`) cannot be tested without first creating the node. The node creation likely belongs in Week 2 alongside the brain-hook wiring, not in Week 1.

### C2 — `cipher_rt_classify_observer.c` does not exist

**Wave 5 reference:** §5.5 Week 1 L575-576 says: "**New file:** `cipher_rt_phase4/cipher_rt_classify_observer.c` — empty stub; will register against matmul/attn registries in Week 2." §5.5 Week 3 L729 then says: "`cipher_rt_classify_observer.c`: maybe_handle now reads `dec` from the oracle ...".

**Disk evidence:** `ls /home/ubuntu/cipher_rt_phase4/cipher_rt_classify_observer.c` — file does not exist. The cipher_rt_phase4 tree currently has the original 16-source build set plus 1 new TU (`src/cipher_may13_harness.cpp` from Step 3).

**Implication for Week 2:** Week 2's brain-hook wiring depends on this file existing (the registration call goes through it). It needs to be created at the entry of Week 2.

### C3 — `cipher-may13-evidence/src/cipher_audit.cpp` does not exist

**Wave 5 reference:** §5.5 Week 4 L786 says: "AUDIT: retire `cipher-may13-evidence/src/cipher_audit.cpp` in favor of existing Wave 2 `cipher_rt_audit.{c,h}` (already wired at priority 0 on both substrates)."

**Disk evidence:** `find /home/ubuntu/cipher-may13-evidence -name "cipher_audit*"` returns no matches. AUDIT logic in may13 lives inline inside `cipher_10ops_impl.cpp` at lines 341-378 (function `audit_chain_update`).

**Implication for Week 4:** There is no standalone `cipher_audit.cpp` to retire; the may13-side AUDIT lives inside the larger `cipher_10ops_impl.cpp`. The retirement work is therefore narrower than Wave 5's framing — the may13 AUDIT block (`cipher_10ops_impl.cpp:341-378`) does not get ported, and the existing production `cipher_rt_audit.{c,h}` continues to be the only AUDIT implementation. No file-deletion is needed.

### C4 — Slot array naming + count

**Wave 5 reference:** §5.5 Week 4 L812 says: "Remove the slot array `cipher_partition_slot[32]` and all writes to it."

**Disk evidence:** The TYPE is `struct cipher_partition_slot` (defined at `cipher_partition_allocator.c:88`). The ARRAY VARIABLE is `cipher_slots[CIPHER_PARTITION_SLOTS_MAX]` (declared at `:91`). The size constant is `CIPHER_PARTITION_SLOTS_MAX`, which evaluates to **33** (the file header comment: "H100 has 132 SMs. Partitioned into blocks of 4 SMs = 33 slots."). Wave 5's "32" probably comes from a separate constraint comment ("the slot count at 32 to keep (1U << i) inside [u32 mask]") which is a constraint on the mask representation, not the array size.

**Implication for Week 4:** The retirement target is the **variable `cipher_slots`** plus all write paths into it, not a non-existent variable named `cipher_partition_slot[32]`. The count is 33, not 32. Wave 5's naming is wrong but the intent is recoverable.

---

## SYNTHESIS-HYPOTHESIS details — execution can proceed with awareness

### H1 — `cipher_v2_init_body` step numbers off-by-one

Wave 5 cites steps 7-10 (matmul_dispatch / marlin / attn_dispatch / attn_test_actuator). Disk shows the same functions at positions **6-9** (the file actually has 10 init calls total, but step 1 is `cipher_rt_green_ctx_cp54_init`, not the `cipher_v2_tenant_register` Wave 5 may have assumed). Functions all exist and the relative ordering matches Wave 5's intent. Week 2 brain-hook registration can use the function names (not step numbers) without ambiguity.

### H2 — Attn trampoline function names are paraphrases

Wave 5 names `flash_call / eff_call / cudnn_call`. The actual functions in `cipher_rt_attn_dispatch.cpp` are C++-mangled SDPA call signatures (`_ZN2at4_ops35_scaled_dot_product_flash_attention4call...` etc.) at approximately L289/L324/L362. Wave 5's short names are paraphrases for readability. Week 2 LP-2 refactor must edit the mangled-name functions.

### H3 — `_tick()` is `cipher_partition_tick()`

Wave 5 cites `_tick()` (Week 4 L814). Disk has `cipher_partition_tick()` at `cipher_partition_allocator.c:429`. Same function, different naming. LP-8 retirement work is unaffected.

### H4 — `cipher_rt_attn_call` `out_status_devptr` location

Wave 5 says "add field at header L80+". Disk struct ends at `cipher_rt_attn_dispatch.h:102` (with `maybe_handle` function pointer). The field can land anywhere in the struct; "L80+" is approximate. Week 2 LP-2 refactor work adds the field; no current contradiction.

---

## NEW WORK items — Week 1 scope that did not land

Wave 5 §5.5 Week 1 listed the following items as Week 1 deliverables that **were not created** by Steps 1, 2 v2, 3, or 4 v2:

| Wave 5 ref | Item | Status |
|---|---|---|
| §5.5 W1 L570-573 | `cipher_rt_phase4/cipher_rt_classify_substrate.cpp` (new file; mirrors matmul/attn substrate shape) | Did not land |
| §5.5 W1 L575-576 | `cipher_rt_phase4/cipher_rt_classify_observer.c` (new empty stub) | Did not land |
| §5.5 W2 L677 (cited as "new from Week 1") | `/proc/cipher/classify_stats` proc node | Did not land |
| §5.5 W1 L597-598 Makefile additions | "Add 9 new objects to `OBJS` (7 ports + 1 new substrate + 1 new observer stub)" | Partial: Step 3 added 1 new object (`cipher_may13_harness.o`); the substrate and observer .o were not added; the 7 may13 source-side ports were not added |

Step 3's brief asked for a single cross-tree compile harness TU and Makefile integration with `-Iinclude`. That delivered the minimum viable proof of cross-tree linkability. Wave 5's broader Week 1 scope (9 new OBJS, 2 new substrate / observer files, 1 new proc node, 7 may13 source ports) was not within any of the Step 1-4 v2 prompts.

These items are not lost — they need to be folded into Week 2 entry-window work (most plausibly Week 2 Step 1 will create `cipher_rt_classify_substrate.cpp`, `cipher_rt_classify_observer.c`, and `/proc/cipher/classify_stats` together as the foundation for the brain-hook wiring). The 7 may13 source ports (the `.cpp` files corresponding to the 12 headers we ported in Step 2 v2) are Week 2-4 work in any reasonable framing.

---

## Recommendation surfaced for Week 2 entry-window

Before Week 2 Step 1 begins, the user should adjudicate the four CONTRADICTION items in this document:

1. **C1 (classify_stats node):** Is creation in Week 2 acceptable, or does Wave 5's Week 1 attribution require a Step 5 amendment that creates the node now?
2. **C2 (classify_observer.c):** Same question — Wave 5 Week 1 attributed; we did not create. Create in Week 2 entry, or amend Week 1?
3. **C3 (cipher_audit.cpp retirement):** No file to retire; the AUDIT block lives inline in `cipher_10ops_impl.cpp`. Wave 5's framing is moot; should Wave 5 §5.5 Week 4 be amended to drop the retirement step, or do we treat the AUDIT inline block as the retirement target?
4. **C4 (slot array naming):** The retirement variable is `cipher_slots` (not `cipher_partition_slot[32]`). Wave 5's text needs a correction for Week 4 to execute mechanically.

The four SYNTHESIS-HYPOTHESIS items can be accepted as Wave 5 paraphrases without re-adjudication; Week 2-5 execution should use the actual on-disk names (verified in the table above) when authoring sed patterns or function references.

---

**End of WAVE_5_S5_5_VERIFICATION.md.**
