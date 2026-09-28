# W13 Step 1: G12 Koopman recipe registry model-keying

**Date:** 2026-05-23
**v1.2.3 §7 W13-14:** Step 1 of 3
**Cumulative closures:** G1, G2 (W6), G3 (W11), G4 (W11), G5 (W12), G6 (W7), G10 (W7), **G12 (this step)**
**Tag:** `week-13-step-1-g12-koopman-keying`

## 1. Pre-condition verification

| Component | Tag / Anchor | md5 |
|-----------|--------------|-----|
| `cipher_rt_phase4` (pre-step) | `week-12-step-6-d3-l2-wireup` (`adbe121`) | `libcipher_rt.so` `33412ffc97b714662e0824e1de8362bd` |
| `cipher_kmod` | `week-9-complete` (0.6.5) | `cipher_kmod.ko` `8c9fdd016897436ceff382c4e9178e07` |
| `cipher_vllm_plugin` | (untagged tree) | `cipher_vllm_kv.py` `2b6cedab89387c30becd49a27313ceb4` |
| `cipher-fusion-evidence` | `682c658` (audit 15/15 closed) | `WEEK_12_STEP_9_AUDIT_CLOSURE_BATCH.md` present |
| `cipher_kv_bridge.so` | unchanged | `5a3db034ea2b950bf426eb48140ec8fd` |

Baseline `libcipher_rt.so` preserved at `/tmp/step13_1_baseline/libcipher_rt.so`
(`33412ffc97b714662e0824e1de8362bd`).

## 2. Rationale

`WEEK_6_ARCHITECTURE_GAP_AUDIT.md` §3.1 G12 line 74 documented the
Koopman recipe registry as shape-only keyed. Two different model
families that compute the same `(M, N, K)` GEMM shape collide on the
registry entry: a recipe seeded for Mistral's LM head shape would also
fire for Llama's LM head shape with the same dimensions. This is the
last G-series substrate gap before the Koopman tier lands in v1.

Step 1 closes G12 by adding a pure helper that mixes `model_uuid` into
`shape_hash` using the same XOR pattern G3 uses at
`cipher_rt_kv_alloc.c:698-708`. Per W13-14 scope-lock + advisor catch
this step, the helper is additive — the existing
`cipher_registry_lookup` signature is unchanged; callers compute the
mixed key at call-site. MODEL_UNKNOWN (`lo == hi == 0`) returns
`shape_hash` unchanged, preserving the pre-G12 single-tenant compat
path.

## 3. Substrate change

### 3.1 `cipher_recipes.h` header surface

Added one public extern "C" declaration at L184-201 alongside
`cipher_registry_lookup`:

```c
uint32_t cipher_rt_recipe_model_key(uint32_t shape_hash,
                                    uint64_t model_uuid_lo,
                                    uint64_t model_uuid_hi);
```

### 3.2 `cipher_recipes.cpp` implementation

Added at L502-520 directly after `cipher_registry_lookup`:

```cpp
extern "C" uint32_t cipher_rt_recipe_model_key(uint32_t shape_hash,
                                               uint64_t model_uuid_lo,
                                               uint64_t model_uuid_hi)
{
    if (model_uuid_lo == 0ULL && model_uuid_hi == 0ULL) return shape_hash;
    uint64_t mix = (uint64_t)shape_hash
                   ^ (model_uuid_lo * 0x9E3779B97F4A7C15ULL)
                   ^ (model_uuid_hi * 0x517CC1B727220A95ULL);
    return (uint32_t)(mix ^ (mix >> 32));
}
```

### 3.3 Pattern alignment

Constants `0x9E3779B97F4A7C15ULL` and `0x517CC1B727220A95ULL` match
G3's `cipher_rt_kv_dedup_model_hash` at `cipher_rt_kv_alloc.c:705-707`
verbatim. The 32-bit return type matches `CipherRegistryEntry::shape_hash`
field type; the fold `mix ^ (mix >> 32)` redistributes XOR-mixed bits
across the lower 32 to preserve avalanche.

## 4. Scope decision: `cipher_kernel_table` UNTOUCHED

W13-14 scope-lock at L68 listed `cipher_kernel_table.cpp` as a
~100 LOC re-key target. Reading the file confirms it is keyed on
`CUfunction` fn_handle pointers (`cipher_kernel_table.cpp:241`), not on
shape_hash. Different model loads (different `cuModuleLoad` calls)
produce different fn_handle pointers, so cross-model aliasing is
**structurally impossible** in this table. G12 keying does not apply
here.

This step leaves `cipher_kernel_table.cpp` and its header unchanged.
Documented as a scope-lock estimate that did not match the actual
code structure. Step 2 may revisit if observability fields (e.g.,
per-entry `model_uuid_lo/hi` for JSON dump filtering) become useful;
that is purely additive observability, not a G12 substrate fix.

## 5. Build

Build clean (warning at `cipher_recipes.cpp:354` is pre-existing
zero-init style for the dead 32 registry entries — Step 2 work):

```
src/may13/cipher_recipes.cpp:354:31: warning: missing initializer for member ...
[8 lines pre-existing zero-init style]
g++ -shared -fPIC -o libcipher_rt.so [50 object files]
```

| File | Pre-Step 1 md5 | Post-Step 1 md5 |
|------|-----------------|------------------|
| `libcipher_rt.so` | `33412ffc97b714662e0824e1de8362bd` | **`1753eb84128a313ce4b2c5ce23947c6d`** |
| `cipher_kmod.ko` | `8c9fdd016897436ceff382c4e9178e07` | UNCHANGED |
| `cipher_vllm_kv.py` | `2b6cedab89387c30becd49a27313ceb4` | UNCHANGED |
| `cipher_kv_bridge.so` | `5a3db034ea2b950bf426eb48140ec8fd` | UNCHANGED |

Symbol export verified: `cipher_rt_recipe_model_key` at offset
`0x1afb0` in `libcipher_rt.so` dynamic symbol table.

## 6. Algebraic gate: `test_g12_recipe_keying`

Source: `/tmp/step13_1_baseline/test_g12_recipe_keying.c` (~140 LOC,
C11). Mirrors `test_g3_cross_model_keying.c` at
`/tmp/step2_w11_baseline/`. 4 phases × 6 shapes × 6 uuids:

```
  computed 6 shapes x 6 uuids = 36 keys
  within-shape collisions across all 6 shapes: 0 (target 0)
  determinism: same (shape, uuid) re-key -> same key: OK
  MODEL_UNKNOWN passthrough (shape -> shape): OK
  MODEL_UNKNOWN distinct from all real uuids per shape: OK

=== test_g12_recipe_keying: PASS (4/4 phases) ===
```

Test inputs:
- 6 model uuids: MODEL_UNKNOWN + 5 distinct synthetic uuids (mirrors
  test_g3 fixture set)
- 6 shape_hash values: LM-head 4096, attention QKV, GEMM tall, GEMM
  wide, smallest non-zero, largest

Gates:
- Phase 1 within-shape collisions: 0 / 90 expected (load-bearing G12
  anti-aliasing)
- Phase 2 determinism: same input -> same output across 6 shapes
- Phase 3 MODEL_UNKNOWN passthrough: pre-G12 single-tenant compat
  preserved
- Phase 4 MODEL_UNKNOWN distinct from real uuids per shape

## 7. W7-12 microbench regression

Run at Step 1 substrate (`libcipher_rt.so` md5 `1753eb84...`):

| Test | Cases | Result |
|------|-------|--------|
| `test_commit_atomicity` | 4/4 | PASS |
| `test_observe_publish` | 3/3 | PASS |
| `test_register_model` | 5/5 | PASS |
| `test_audit_chain` | 2/3 | PASS Case 1+2, Case 3 FAIL (pre-existing at W12 Step 6 baseline byte-identical .so; not Step 1 regression) |
| `test_resolver` | 3/3 | PASS |
| `test_ring_write` | 6/6 | PASS |
| `test_g3_cross_model_keying` | — | dlopen failed on libtorch_python (pre-existing at W12 Step 6 baseline; env issue not Step 1 regression) |
| `test_tc_probe` | 17/17 | PASS (false-positive 0.00%, false-negative 0.00%) |
| `test_sdpa_tenant_routing` | 4/4 | PASS |
| `test_g5_va_density` | 5 families | PASS |
| `test_l2_wireup` | 5/5 | PASS |

**Both non-PASS results are pre-existing at the W12 Step 6 D3 baseline**
(verified by reverting `libcipher_rt.so` to byte-identical `33412ffc...`
and re-running: same failures). Zero Step 1 regression.

## 8. N=128 30-min soak

Launched at Step 1 substrate (libcipher_rt.so md5 `1753eb84...`) with
N_TENANTS=128, N_READERS=4, duration=1800 s, dual-slot drain pattern.

```
=== SOAK RESULTS ===
Wall time             : 1801.43 s
Total publishes       : 20674896631 (11476911 /s aggregate)
Mean per tenant       : 161522630
Min/Max per tenant    : 146052644 / 175405290

-- Fairness (substrate property; GATED on writers-only) --
Writers-only ratio    : min=0.904 max=1.086 (gate: min >= 0.85)

-- All-thread ratio (telemetry; OS-scheduler noise baseline) --
All-thread (W+R) ratio: min=0.979 max=1.017 (NOT gated)
Per-tenant W+R total  : min=22364975701 max=23241440861 mean=22848237366

-- Coherence (GATED) --
Reader coherent reads : 2903899486274
Reader incoherent     : 0 (gate: == 0)

Gates: atomicity=1 writers_fairness=1 coherence=1
ALL GATES PASS
```

| Metric | Value | Gate | Result |
|--------|-------|------|--------|
| Atomicity | 128000/128000 | 128000/128000 | PASS |
| Writers-only fairness | min 0.904 max 1.086 | min >= 0.85 | PASS |
| Coherence | 0 incoherent of 2.904 trillion reads | == 0 | PASS |
| Aggregate publish rate | 11.48 M/s | within +/-10% vs W12 Step 6 D3 baseline 10.96 M/s | +4.7% PASS |

Performance trend vs prior soaks across W7-W12 substrate:

| Step | libcipher_rt.so | Aggregate M/s | Delta vs prior |
|------|-----------------|----------------|----------------|
| W9 Step 5 | 16b0bcc6 | 10.80 (1h soak) | baseline |
| W10 Step 1 | 66a33d78 | 11.30 | +4.6% |
| W11 Step 2 | af24c13a | 10.31 | -8.7% |
| W12 Step 3 | 9db95d4d | 10.66 | +3.4% |
| W12 Step 4 | e650b49f | 11.52 | +8.1% |
| W12 Step 5 | 3e22be9d | (same range) | env-gate |
| W12 Step 6 | 33412ffc | 10.96 | -4.86% |
| **W13 Step 1** | **1753eb84** | **11.48** | **+4.7%** |

All deltas across the W9-W13 substrate ladder are within +/-10% gate.

## 9. Honest residue — Step 2 file-location finding (HARD STOP surface)

**Per advisor catch this step:** W13-14 scope-lock at L84-85 names
`cipher_rt_phase4/src/may13/cipher_lnn.cpp` and
`cipher_rt_phase4/src/may13/cipher_koopman_runtime.cpp` as Step 2
targets. Both files **do not exist** in `cipher_rt_phase4`. They exist
in `cipher-may13-evidence/src/`:

- `/home/ubuntu/cipher-may13-evidence/src/cipher_lnn.cpp`
- `/home/ubuntu/cipher-may13-evidence/src/cipher_koopman_runtime.cpp`
- `/home/ubuntu/cipher-may13-evidence/include/cipher_lnn.h`
- `/home/ubuntu/cipher-may13-evidence/include/cipher_koopman_runtime.h`

Step 2 implementation must port these into cipher_rt_phase4 (~430 +
~200 = ~630 LOC of source ports) BEFORE the EDMD-pipeline-real-input
fix + recipe seeding + SUBSTITUTE-Koopman lane wire can begin. The
scope-lock LOC estimate (~610 LOC total) implicitly counted the port
weight but did not flag the file-location finding.

**This is a Step 2 pre-flight HARD STOP per task spec discipline.** The
advisor recommended surfacing this at Step 1 close so Anil has time to
adjudicate before Step 2 starts. Three options:

(i) **Port the may13 sources into cipher_rt_phase4** — adds 1-2 eng-days
to Step 2 (file-port + nvcc rule + header path adjustments) on top of
the 5-eng-day Step 2 estimate, total ~6-7 eng-days.

(ii) **Reframe Step 2 around the existing in-tree EDMD surface** —
`/home/ubuntu/cipher_rt_phase4/include/may13/cipher_edmd.h` exists in
cipher_rt_phase4. If the EDMD pipeline real-input fix and recipe
seeding can land against this header surface without porting the full
lnn + koopman_runtime sources, Step 2 scope narrows to just the EDMD
pipeline + recipe seeding (~3 eng-days). Loses the CfC LNN forward
that REMEMBER consumer (Step 3) needs to feed.

(iii) **Defer the full CfC LNN forward to v2** — Step 2 lands EDMD
pipeline + recipe seeding + SUBSTITUTE lane against the LM-head
narrow domain only. REMEMBER consumer (Step 3) wires the slot-3 ring
producer + drain into a stub LNN forward (or skip Step 3 LNN entirely
and validate Koopman-lane-fires at Step 2 close). LM-head 5.5e-5 KL
gate still lands. Drops the runtime-adaptation surface from v1
narrow-domain claim.

**NO AUTO SCOPE-DEGRADE applied.** Step 1 closes with this finding
surfaced for Anil adjudication.

## 10. Files

- `cipher_rt_phase4/include/may13/cipher_recipes.h` (1 new declaration, ~18 LOC)
- `cipher_rt_phase4/src/may13/cipher_recipes.cpp` (1 new function, ~19 LOC)
- `/tmp/step13_1_baseline/libcipher_rt.so` (preserved pre-step baseline)
- `/tmp/step13_1_baseline/test_g12_recipe_keying.c` (~140 LOC)
- `/tmp/step13_1_baseline/test_g12_recipe_keying` (compiled binary)
- `/tmp/step13_1_baseline/n128_soak.log` (30-min soak log)
- `cipher-fusion-evidence/WEEK_13_STEP_1_G12_KOOPMAN_KEYING.md` (this doc)

## 11. Anchors

| Tree | Anchor | Status |
|------|--------|--------|
| `cipher_rt_phase4` | `week-13-step-1-g12-koopman-keying` | **NEW** |
| `cipher_kmod` | `week-9-complete` (0.6.5) | UNCHANGED |
| `cipher_vllm_plugin/cipher_vllm_kv.py` | `2b6cedab89387c30becd49a27313ceb4` | UNCHANGED |
| `cipher-fusion-evidence` | new HEAD with this file | step-doc commit |

Tag chain ahead: `week-14-step-2-koopman-tier` → `week-14-step-3-remember-validate` (alias `week-14-complete`).
