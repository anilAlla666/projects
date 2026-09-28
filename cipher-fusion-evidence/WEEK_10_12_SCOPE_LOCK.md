# Weeks 10-12 scope-lock — RING_WRITE + G3 + G4 + G5 + 3 blog additions

**Date:** 2026-05-23
**Predecessor:** `WEEK_7_9_SCOPE_LOCK.md` + `WEEK_9_STEP_5_N128_SOAK.md` (v1.2.3 §7 W7-9 CLOSED at `week-9-complete`)
**Calendar:** Weeks 10, 11, 12 (3 weeks)
**Closes:** v1.2.3 §7 W10-12 row (line 1257) — RING_WRITE substrate + G3 KV-dedup model-keying + G4 Marlin tenant-scoped weight kit + G5 VA pool per-tenant sizing
**Output:** 3-step sequence with each step composing one blog-derived structural addition (CFL invariant + TC saturation probe + L2 persistence)

## 1. Pre-conditions verified

| Item | Expected | Observed | OK |
|------|----------|----------|----|
| `cipher_kmod` tag | `week-9-complete` | `week-9-complete` (8c643fc, MODULE_VERSION 0.6.5) | ✓ |
| `cipher_rt_phase4` tag | `week-9-complete` | `week-9-complete` (c93a141) | ✓ |
| v1.2.3 §7 W7-9 marked DONE | yes | yes (Step 5 ALL GATES PASS, 1h soak 10.5T reads/0 incoherent) | ✓ |
| `WEEK_9_STEP_5_N128_SOAK.md` present | yes | yes (cipher-fusion-evidence 9b875af) | ✓ |
| `WEEK_7_9_SCOPE_LOCK.md` present (pattern reference) | yes | yes | ✓ |
| `WEEK_6_ARCHITECTURE_GAP_AUDIT.md` (G3/G4/G5 source) | yes | yes | ✓ |

## 2. Three locked decisions (user 2026-05-23)

1. **RING_WRITE is ADDITIVE on top of existing telemetry**, not a replacement. The W7-9 Step 4 cipher_rt_commit seqlock snapshot path stays; Step 1 builds NEW infrastructure alongside (confirmed by §4.9 line 921-957 — "RING_WRITE and CUPTI run in parallel… the two paths never block each other").
2. **G3 + G4 land TOGETHER** in Step 2. Both are model-keyed actuator pathway changes consuming the W7-9 Step 1 G10 `model_uuid`. Shared design pattern (per-tenant snapshot lookup → hash/cache keyed on model_uuid).
3. **3 blog-derived structural additions DISTRIBUTED** across the steps, not appended as a fourth step:
   - **CFL stability invariant** → Step 1 RING_WRITE (producer/consumer rate property; belongs in RING_WRITE design)
   - **TC saturation probe** → Step 2 G3+G4 (classifier signal informs model-keyed actuator pathway selection)
   - **L2 persistence** → Step 3 G5 (per-tenant memory-tier policy; lives alongside per-model VA sizing)

## 3. Open questions — resolved at scope-lock

### 3.1 (B.1) RING_WRITE vs existing seqlock snapshots

**Answer: parallel channels (option a from prompt).** Per `CIPHER_REENGINEERING_PLAN.md` §4.9 line 925-948:

> RING_WRITE fires synchronously from the LD_PRELOAD-equivalent injection (the GOT-patched cuLaunchKernel intercept) at sub-microsecond cost, on the launch thread, with the kernel descriptor in hand. The two substrates feed different consumers:
> - CLASSIFY hot-path cache → RING_WRITE (pre-launch, on-thread)
> - ORACLE phase detection → RING_WRITE post-dispatch entry
> - AUDIT chain → RING_WRITE post-dispatch entry
> - REMEMBER (Koopman tier) → RING_WRITE post-dispatch entry

The W7-9 Step 4 cipher_rt_commit seqlock is per-tenant publish-on-COMMIT (slow-path consumers — `_report()` overlay ops draining once per launch via `cipher_rt_snapshot_acquire`). RING_WRITE is per-launch lock-free inline producer with downstream consumers. **Step 1 builds the producer + ring structure; no consumer change required.**

### 3.2 (B.4) RING_WRITE consumer side

Per §4.9 line 928-948 table: consumers are CLASSIFY (already-in-tree as classifier substrate `cipher_rt_classify_observer.c`), ORACLE (already-in-tree at `cipher_rt_oracle_bridge.o`), AUDIT (W7-9 Step 3 G6 kmod chain already consumes via NR 28 ioctl from `cipher_rt_commit_token_boundary`), REMEMBER (W13-14 Koopman tier).

**Step 1 ships the producer side only.** Existing CLASSIFY/ORACLE classifier signals already read the seqlock snapshot; they migrate to the ring lazily as their respective ops re-port. The AUDIT chain consumer is already wired via Step 3 G6. REMEMBER is W13-14 work. **No consumer-side ambiguity remains.**

### 3.3 (B.3) SDPA stream-resolution residue from W7-9

Per `WEEK_9_STEP_5_N128_SOAK.md` §9 item 1: SDPA dispatch wire still uses `tenant_id=0` because `cipher_rt_attn_call` descriptor doesn't carry a CUDA stream pointer.

**Folded into Step 2** (per prompt B.3 suggestion). Step 2 already edits the per-launch dispatch shim pattern for G3 KV-dedup model-keying, which means modifying the same `cipher_rt_matmul_call` + `cipher_rt_attn_call` descriptors. Extending `cipher_rt_attn_call` with a `stream` field is a 3-line ABI tweak + 5-line plumbing change in `cipher_rt_attn_dispatch.cpp` route() at line 135. Counted in Step 2's eng-day budget.

### 3.4 (B.5) Step ordering rigidity

CFL stability invariant in Step 1 RING_WRITE constrains how G3+G4 in Step 2 will report their skip/hit counters via the new ring. So Step 1 → Step 2 is **load-bearing sequential**. Step 3 G5 VA-pool sizing has no direct dependency on Step 1's CFL or Step 2's TC probe (G5 is plugin-side compute_va_gib at REGISTER_TENANT time, completely orthogonal), but the L2 persistence work piggybacks Step 3 because both touch per-tenant memory-tier policy.

**Sequential ordering is correct.** No parallel-step opportunity sufficient to reorder.

### 3.5 (B.2) N=128 regression smoke cadence

Each step ends with a 30-minute regression smoke against the W9 substrate baseline. Exact invocation:

```
LD_LIBRARY_PATH=/home/ubuntu/cipher_rt_phase4 \
  /tmp/step5_baseline/cipher_test_commit_n128 1800
```

Pass criteria (from W7-9 Step 5 closeout):
- Atomicity 128000/128000
- Writers-only fairness ratio min ≥ 0.85
- All-thread ratio reported (telemetry, not gated)
- Reader coherence 0 incoherent

## 4. Three-step sequence with file:line citations

### Step 1 — RING_WRITE producer + CFL invariant (W10)

**Calendar:** Week 10 (5 eng-days)
**Touches:** `cipher_rt_phase4` only. New files: `cipher_rt_ring.{c,h}` (per §4.9 line 953-957). Edits: `cipher_inject.c` (post-dispatch hook in cuLaunchKernel intercept), `Makefile` (OBJS). No kmod changes — RING_WRITE is pure-userspace lock-free producer.
**ABI impact:** none (no new kmod ioctl; AUDIT chain still uses NR 28 from W7-9 Step 3).
**Close tag:** `week-10-step-1-ring-write`

**Sub-elements:**

| Component | Source / spec | LOC |
|-----------|---------------|-----|
| `cipher_rt_ring.h` — per-tenant ring struct + producer inline | port from `cipher-may13-evidence/include/cipher_10ops.h:83-110` (`cipher_ring_write` inline + `CipherRing` struct + `ring_writes` counter) | ~80 |
| `cipher_rt_ring.c` — ring init/exit + per-tenant slab allocation | ~120 |
| `cipher_inject.c` post-dispatch hook | similar pattern to `cipher_intercept.cpp:180-181, 374-375` (may13 call sites for `cipher_ring_write`) | ~20 |
| Consumer-spawn predicate (env-gated per consumer per §4.9 L954) | ~30 |
| **CFL stability invariant** (blog addition #1) — runtime check on producer/consumer rate divergence; throttles producer when ring fill exceeds high-water mark | ~50 |

Total: ~300 LOC new + ~20 LOC edits.

**Verification gate at close:**
- `test_ring_write` smoke (new, ~120 LOC): N=128 producer threads × 4 consumer threads; verify ring entries are FIFO-ordered per tenant; CFL invariant trips when producer rate > 2× consumer rate (synthetic stress); ring overflow handled gracefully (drop counter advances, no producer block per spec line 227).
- N=128 30-min regression smoke (per §3.5 above).

### Step 2 — G3 KV-dedup model-keying + G4 Marlin tenant-scoped weight kit + TC saturation probe + SDPA stream resolution (W11)

**Calendar:** Week 11 (5 eng-days)
**Touches:**
- G3: `cipher_rt_phase4/cipher_rt_kv_alloc.h:112` (xxhash64 chained table — re-key from content-only to `(model_uuid, layer_idx, head_idx, dtype, content_hash)` per audit line 251)
- G4: `cipher_rt_phase4/cipher_rt_marlin_engine.cpp:960` (`cipher_rt_marlin_engine_observe_weight`) + line 1065 (`weights_count`) — per-tenant arena lookup keyed on `model_uuid` from per-tenant snapshot (audit line 254)
- TC saturation probe (blog addition #2): new observe op following `cipher_rt_classify_observer.c:53` pattern (`cipher_rt_classify_observer_observe`)
- SDPA stream resolution: `cipher_rt_attn_dispatch.cpp:135` `route()` — extend `cipher_rt_attn_call` with `stream` field; wire `cipher_v2_current_tenant_id_from_stream(call.stream)` analogous to cublas shim's W7-9 Step 5 hot path
**ABI impact:** none expected (`model_uuid` already plumbed via Step 1 G10's `cipher_tenant_snapshot.model_uuid` field, exposed in the W7-9 Step 4 snapshot at `cipher_rt_commit.h:50-51`).
**Close tag:** `week-11-step-2-g3-g4-tc-probe`

**Sub-elements:**

| Component | Source / spec | LOC |
|-----------|---------------|-----|
| G3 hash function re-keying (`cipher_rt_kv_alloc.h:49, 112`) | audit §3.1 G3 (line 251); R-W10.1 budget gate ≤ 1 µs/lookup | ~200 |
| G4 Marlin per-tenant arena (`cipher_rt_marlin_engine.cpp:960`) | audit §3.1 G4 (line 254); R-W10.2 cubin cache stays keyed on (K,N) only | ~200 |
| TC saturation probe (blog addition) — observation hook in classifier substrate | follows `cipher_rt_classify_observer_observe` pattern (line 53) | ~150 |
| SDPA stream-resolution carry from W9 residue | small fix; descriptor + dispatch + cublas-shim-style resolver call | ~30 |

Total: ~580 LOC new + ~30 LOC edits.

**Verification gate at close:**
- **Cryptographic gate — cross-model KV-dedup test:** Load Mistral-7B + Llama-3.1-8B via Step 1 G10 distinct uuids; verify content-identical KV blocks do NOT alias across models. Closes G3's silent cross-model attention corruption hazard per audit §2.7 + plan line 1737 R-G3.1. Pass criteria: teacher-forced KL ≤ 5.5e-5 across 6 cross-model pairs (plan line 1737 gate).
- Marlin per-tenant arena memory growth gate: at N=4 with 4 distinct models, total Marlin weight cache footprint ≤ 4 × single-model footprint (no shared-arena leak).
- N=128 30-min regression smoke.

### Step 3 — G5 VA pool path-a + L2 persistence policy (W12)

**Calendar:** Week 12 (4 eng-days)
**Touches:**
- G5: `cipher_vllm_kv.py:193-194` (current default `va_gib = int(os.environ.get("CIPHER_KV_VA_POOL_GIB", "80"))`) — replace with `compute_va_gib(hf_config)` per-model sizing (W6 G5 audit memory `g5-path-a-verified` confirmed feasibility)
- L2 persistence (blog addition #3): refine `cipher_rt_phase4/src/may13/cipher_l2_persist.cu:117` (`cipher_l2_persist_apply`) — currently called from `cipher_runtime.cpp:49,95,123` (init/reset/report). Per-tenant `cudaAccessPolicyWindow` policy via Step 1's view-slot tenant_id keying.
**ABI impact:** none (plugin-side change for G5; op-internal for L2 refinement).
**Close tag:** `week-12-step-3-g5-l2-persist` (== **`week-12-complete`**)

**Sub-elements:**

| Component | Source / spec | LOC |
|-----------|---------------|-----|
| G5 `compute_va_gib(hf_config)` replacing va_gib=80 | W6 G5 audit (cipher-fusion-evidence 35f9b6c) confirmed path-a feasibility; reads hf_config fields per plugin lines 119-121 (model_type, hidden_size, num_hidden_layers, intermediate_size, vocab_size, max_position_embeddings) + adds num_key_value_heads + head_dim for KV computation | ~150 (modify + new) |
| L2 persistence refinement (blog addition) — per-tenant cudaAccessPolicyWindow policy keyed on Step 5 view-slot tenant_id | ~200 |

Total: ~350 LOC new + edits.

**Verification gate at close:**
- **VA density gate:** 100 tenants × 5 model families fits within W6 path-a headline budget (437 GiB scenario; 152 GiB with Track 2 weight-sharing per memory `g5-path-a-verified`). Verify via plugin dry-run that aggregate va_gib across 100 tenants stays under 1 TiB host VA.
- **L2 hit-rate gate:** measure L2 hit rate before/after policy refinement on synthetic multi-tenant workload. Pass criteria: hit rate improves OR stays within noise on hot-path measurements.
- N=128 30-min regression smoke.

**This step closes v1.2.3 §7 W10-12.** Both repos tag `week-12-complete` analogous to `week-9-complete` on W7-9 close.

## 5. Risk register

Format mirrors `WEEK_7_9_SCOPE_LOCK.md` §8.

| ID | Description | Likelihood | Impact | Mitigation | Detection (regression smoke) |
|----|-------------|------------|--------|------------|------------------------------|
| **R-W10.1** | Re-keyed KV-dedup hash inflates per-page xxhash64 cost from O(1) to multi-key trie walk | HIGH | BLOCKER | Microbench new vs old at 8192 keys × 100 lookups/s. If > 1 µs/lookup, switch to hash-of-hashes keyed on model_uuid first (audit §7.3 R3). HARD STOP: if budget unmakeable, dedup ships single-model-only in v1; heterogeneous-model framing degrades CP 5.5 per-model-density story | Step 2 close — `test_kv_dedup_hash` microbench: 8192 keys, mean p99 cost ≤ 1 µs |
| **R-W10.2** | Marlin cubin JIT cache thrash at 100 unique `(model_uuid, K, N)` tuples | HIGH | MAJOR | Keep cubin cache keyed on `(K,N)` only (per audit §7.3 R4); `model_uuid` in weight cache (cheap to grow to ~100s entries) but NOT in JIT cache (expensive to invalidate). Test 5 distinct models × Mistral/Qwen/Llama shape sets | Step 2 close — unique cubin count ≤ 32 across 5-family shape sets |
| **R-W10.3** | G5 path-a `compute_va_gib` accuracy for non-standard hf_configs (Phi-2, untested fields) | MEDIUM | MAJOR | W6 audit memory `g5-path-a-verified` already validated path-a for Llama/Mistral/TinyLlama. Step 3 extends test matrix to 5 model families; fallback is conservative-upper-bound formula when fields missing | Step 3 close — VA-pool plugin dry-run across 5 model families; total ≤ 1 TiB |
| **R-W10.4** (NEW) | RING_WRITE producer overhead breaks the cublas/SDPA hot path | MEDIUM | MAJOR | §4.9 line 925 budgets RING_WRITE at sub-µs per launch. May13 baseline at `cipher_intercept.cpp:180-181` was already verified at this cost. Mitigation: env-gate producer at `CIPHER_RING_WRITE=1` (default OFF until Step 1 close); compose with W7-9 Step 5 hot path lookup to keep total ≤ 200 ns p99 | Step 1 close — `test_ring_write` overhead p99 ≤ 500 ns; N=128 regression atomicity holds |
| **R-W10.5** (NEW) | CFL invariant false-positive rate (producer self-throttles when consumer is healthy) | MEDIUM | MINOR | Synthetic stress test in `test_ring_write`: producer rate = consumer rate × N for N ∈ {0.5, 1, 2, 4}; invariant trips only at N ≥ 2 | Step 1 close — `test_ring_write` Case D: 0 false-trips at N=1 |
| **R-W10.6** (NEW) | TC saturation probe noise (classifier signal jitter) confuses model-keyed actuator pathway selection | LOW | MINOR | Probe lands as ADDITIVE observation only; no actuator pathway change in Step 2. v2 wires the probe into pathway selection once data shows real signal. v1 ships the probe + telemetry only | Step 2 close — TC probe emits to ring buffer; downstream consumer is W13-14 |
| **R-W10.7** (NEW) | L2 persistence refinement requires CUDA compute capability ≥ 8.0 | LOW | MINOR | H100 is sm_90 → cudaAccessPolicyWindow available. Document. Soft fallback: if cuda runtime version returns no policy support, refinement is no-op | Step 3 close — `cipher_l2_persist_apply` returns success on H100 |
| **R-W10.8** (NEW) | SDPA descriptor extension (B.3 fold) breaks T4.6.1 attention-substrate signature compatibility | LOW | MAJOR | Additive field append at end of `cipher_rt_attn_call`; existing actuators with sizeof() expectation stay valid (struct only grows). Test against test_actuator from T4.6.1 | Step 2 close — `cipher_rt_attn_test_actuator` still callable; no SIGSEGV on dispatch |

R-G3.1 (silent cross-model KV corruption) from plan line 1737 stays open until Step 2 cryptographic gate passes. R-G5.1 (VA pool exhaustion) from plan line 1738 closes at Step 3 close.

## 6. Eng-day budget + total estimate

| Item | Eng-days |
|------|----------|
| Step 1 RING_WRITE + CFL | 5 |
| Step 2 G3+G4+TC probe+SDPA carry | 5 |
| Step 3 G5+L2 persistence | 4 |
| Per-step regression smoke (3 × 0.5) | 1.5 |
| Step doc + commit + memory anchor (3 × 0.5) | 1.5 |
| **Total** | **17** |

Calendar: **3 weeks** (W10 + W11 + W12). Plan line 75 commits +2 weeks for W10-12 (G3+G4+G5 fold-in over v1.2.2's W9-10 RING_WRITE), so calendar is exactly on plan. Eng-day estimate 17 is within v1.2.3 §7 reasonable variance (no scope creep; plan line 75 budgets ~20-25 eng-days net add for v1.2.3 total — Step 5 already absorbed ~6 of those).

No HARD STOP triggered: estimate doesn't diverge > 30% from commitment.

## 7. Next prompt — Step 1 RING_WRITE producer + CFL invariant

```
TASK: W10-12 Step 1 — RING_WRITE producer substrate + CFL invariant.
     ~5 eng-days. cipher_rt_phase4 only (no kmod ABI change).
```

Specifics for the implementation prompt:
- Port `cipher_ring_write` inline + `CipherRing` struct from `cipher-may13-evidence/include/cipher_10ops.h:83-110` → `cipher_rt_phase4/cipher_rt_ring.{c,h}`
- Producer hook in `cipher_inject.c` post-dispatch (analog of may13 `cipher_intercept.cpp:180-181, 374-375`)
- CFL invariant: high-water mark + producer throttle when ring fill > 75%
- Consumer-spawn env predicates per §4.9 line 954
- `test_ring_write` gate: 5/5 cases (FIFO, overflow drop, CFL trip, no-false-trip, overhead p99)
- N=128 regression smoke at close (`cipher_test_commit_n128 1800`)
- Close tag: `week-10-step-1-ring-write`

## 8. v1.2.3 §7 W10-12 row (line 1257)

> **10-12** | **RING_WRITE lock-free inline telemetry substrate build + G3 KV-dedup model-aware keying + G4 Marlin tenant-scoped weight kit + G5 VA pool implementation per W6 audit decision** *(extended from W9-10 to W10-12 per B.1)* | v1.2.3

This scope-lock satisfies the row in full. Blog-derived additions (CFL invariant, TC saturation probe, L2 persistence) are integrated INTO the three steps per locked decision #3 — they don't extend the W10-12 row; they refine the substrate of the row's existing line items.

## 9. Honest residue / open items

1. **REMEMBER consumer (Koopman tier)** consumes RING_WRITE per §4.9 line 928-948 but lands in W13-14, not W10-12. Step 1 ships the producer; the consumer slot is reserved by env predicates.
2. **CLASSIFY/ORACLE migration to RING_WRITE-driven signals** is per-op, follows after W10-12 (W13-14 Koopman tier pulls them through). No work in this scope.
3. **G12 (Koopman registry model-keying)** explicitly DEFERRED to W13-14 per plan line 1242, even though G10 (model_uuid) is already plumbed. Reason: Koopman recipe registry port lands W13-14; model-keying piggybacks.
4. **Mistral E.7 environment block** (Step 4/5 residue) carries forward. v1.2.3 §7 §7 W10-12 doesn't unblock it; either deep_gemm install or document as v1 measurement environment limitation.
5. **L2 persistence per-tenant policy depends on Step 5's view-slot tenant_id resolution.** Default-stream tenants (tenant_id=0 fallback) get the global L2 policy; named-stream tenants get per-tenant policy. Documented as v1 boundary; v2 graceful degradation deferred.

## 10. Fingerprints expected at W10-12 close

```
cipher_rt_phase4   <new commit>     tag week-12-complete
libcipher_rt.so    <new md5>        (~ +1000 LOC across 3 steps)
cipher_vllm_kv.py  <new md5>        (G5 compute_va_gib added, Step 3)
cipher_kmod        unchanged        (no kmod ABI change for W10-12; stays at 0.6.5 / 8c643fc)

cipher-fusion-evidence:
  WEEK_10_STEP_1_RING_WRITE.md
  WEEK_11_STEP_2_G3_G4_TC_PROBE.md
  WEEK_12_STEP_3_G5_L2_PERSIST.md
```
