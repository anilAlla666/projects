# Weeks 13-14 scope-lock — Koopman tier integration + G12 model-keyed registry

**Date:** 2026-05-23
**Predecessor:** `WEEK_10_12_SCOPE_LOCK.md` + `WEEK_12_STEP_3_G5_L2_PERSIST.md` (v1.2.3 §7 W10-12 row CLOSED at `week-12-complete`)
**Calendar:** Weeks 13, 14 (2 weeks)
**Closes:** v1.2.3 §7 W13-14 row (line 1258) — Koopman tier integration (EDMD pipeline real-input fix + recipe registry narrow-domain seeding + SUBSTITUTE-Koopman lane validation) + G12 Koopman registry model-keying
**Output:** 3-step sequence with G12 leading (substrate edit) then Koopman tier on top + REMEMBER consumer wire + validation gate

## 1. Pre-conditions verified

| Item | Expected | Observed | OK |
|------|----------|----------|----|
| `cipher_rt_phase4` tag | `week-12-complete` | `week-12-complete` (fec9cc3) | ✓ |
| `cipher_kmod` tag | `week-9-complete` (0.6.5) | `week-9-complete` (8c643fc, 0.6.5) | ✓ |
| `libcipher_rt.so` md5 | `9db95d4d…` | `9db95d4db890335f86fb108b43734ee2` | ✓ |
| `cipher_vllm_kv.py` md5 | `1009817c…` | `1009817ca1975fd38da96c3005f56759` | ✓ |
| v1.2.3 §7 W10-12 row CLOSED | yes | yes (W12 Step 3 closeout `494b8e9`) | ✓ |
| `WEEK_12_STEP_3_G5_L2_PERSIST.md` present | yes | yes | ✓ |
| RING_WRITE slot 3 `CIPHER_RT_RING_EVENT_REMEMBER` reserved | yes | yes (`cipher_rt_ring_write.h:53`) | ✓ |
| Koopman headers ported to cipher_rt_phase4 | yes | yes (`include/may13/cipher_edmd.h`, `cipher_recipes.h`, `cipher_lnn` structures) | ✓ |

## 2. Locked decisions (user 2026-05-23 — pre-Step-1)

1. **G12 model-keying lands first as Step 1** — standalone substrate edit per the W6 audit's ~300 LOC / 3 eng-days estimate. Same pattern as W7-9 G10 (ABI scaffolding before consumers): the registry is re-keyed BEFORE the Koopman tier wires consumers that depend on the model-keyed lookups. Reasoning: keeps the change small + atomic + testable in isolation.
2. **Narrow domain v1 scope** per plan §1 line 103: "EDMD pipeline + recipe registry narrow-domain seeding + SUBSTITUTE-Koopman lane validation." Wide-domain Koopman is v2 research. Step 2 implements the narrow domain.
3. **CP 5.5 full benchmark is W15-17, NOT W13-14.** Step 3 of W13-14 ships a validation gate (does the Koopman substitution actually fire on synthetic LM-head-class workload), not the full hybrid heterogeneous benchmark. Per scope-lock B.4.

## 3. Open questions — resolved at scope-lock

### 3.1 (B.1) G12 standalone vs composed

**Answer: standalone Step 1, composed in Step 2.** Per W6 audit line 279 ("Mostly mechanical: every cache lookup in `cipher_recipes.cpp` (522 LOC) and `cipher_kernel_table.cpp` (317 LOC) takes a `model_uuid` prefix"), G12 is a 300 LOC mechanical re-key. The Koopman tier sits on top in Step 2; the model-keyed registry IS the substrate the tier consumes.

This mirrors W7-9 (G10 ABI scaffolding alone in Step 1, then G3/G4/G12 consumers in subsequent steps).

### 3.2 (B.2) "Narrow domain v1" scope

**Answer:** per plan §1 line 103 and §4 (TBD specific section):
- **EDMD pipeline real-input fix**: snapshot-feeder corrects the may13 audit-discovered issue (`include/may13/cipher_edmd.h` exists but no real input wiring)
- **Recipe registry seeding**: per-layer rank-parameterized recovery; Tikhonov regularization α = 0.01·σ₁²; spectral radius ≤ 1
- **SUBSTITUTE-Koopman lane validation**: existing `SUBSTITUTE` actuator at `cipher_dispatch.cpp:200-395` has a Koopman lane that has NEVER FIRED per plan line 401; Step 2 wires the lane + seeds recipes that match a narrow set of shapes
- **Wide-domain Koopman remains v2 research** — explicit per plan line 103

Narrow domain target (per memory anchor for goal 4): the lm_head matmul is the canonical small-scope target. Verification reference "LM head 7.43× substitution speedup" was mentioned in the W13-14 task prompt but the specific provenance is not located in cipher-fusion-evidence or the cipher_rt_phase4 source tree (see honest residue §9). Step 2 + Step 3 measure the actual narrow-domain speedup against the v1.2.3 substrate; the "7.43×" framing is treated as informational, not a gate.

### 3.3 (B.3) REMEMBER consumer wiring

**Answer:** per plan §4.9 line 934, REMEMBER (Koopman tier) consumes RING_WRITE slot 3 to update the CfC (Closed-form Continuous-time) LNN hidden state per launch — "the CfC hidden-state update needs the per-launch class + decision." Step 3 wires the consumer thread that drains RING_WRITE slot 3 into the CfC LNN forward at `cipher_lnn.cpp:428-458` (per plan line 384 + audit citation). Producer side is already shipped (W10-12 Step 1 ring + classify_observer emitter).

### 3.4 (B.4) Coordination with W15-17 CP 5.5

**Answer: deferred.** Plan line 1259: CP 5.5 hybrid heterogeneous benchmark is W15-17. W13-14 ships substrate + ONE validation that Koopman actually fires (Step 3); the 24-hour soak + cryptographic-billing-chain + ≥5 model families benchmark is W15-17.

## 4. Three-step sequence with file:line citations

### Step 1 — G12 Koopman registry model-keying (W13)

**Calendar:** Week 13 (3 eng-days)
**Touches:** `cipher_rt_phase4/src/may13/cipher_recipes.cpp` (522 LOC) + `cipher_rt_phase4/src/may13/cipher_kernel_table.cpp` (324 LOC). Re-key registry lookups from shape-only to `(model_uuid, shape, …)`.
**ABI impact:** none (no kmod change; model_uuid plumbed via W7-9 Step 1 G10 snapshot; same weak-link pattern as G3/G4)
**Close tag:** `week-13-step-1-g12-koopman-keying`

**Sub-elements:**

| Component | Source / spec | LOC |
|-----------|---------------|-----|
| `cipher_recipes.cpp` registry-lookup re-key | W6 audit §3.1 G12 line 74 | ~150 |
| `cipher_kernel_table.cpp` table-lookup re-key | same | ~100 |
| RING_WRITE event reservation slot 3 `REMEMBER` (already reserved Step 1) | wired via consumer in Step 3 | — |
| `test_g12_recipe_keying.c` smoke (algebraic — two model uuids × same shape → different recipe entries) | analogous to test_g3_cross_model_keying.py | ~80 |

Total: ~330 LOC matching audit's 300 LOC estimate (10% slack).

**Verification gate at close:**
- `test_g12_recipe_keying` algebraic gate: model uuid A + shape S → recipe entry; model uuid B + same shape S → different entry. No alias.
- W7-12 microbench regression (test_commit_atomicity 4/4, test_audit_chain 5/5, test_observe_publish 3/3, test_resolver 3/3, test_ring_write 6/6, test_g3_cross_model_keying PASS, test_tc_probe PASS, test_g5_va_density PASS)
- N=128 30-min regression smoke (per W10-12 pattern; gate: within ±10% of W12 Step 3 baseline 10.66 M/s, atomicity 128000/128000, 0 incoherent)

### Step 2 — Koopman tier (EDMD + recipe seeding + SUBSTITUTE lane) (W13-W14)

**Calendar:** Weeks 13-14 (5 eng-days)
**Touches:**
- `cipher_rt_phase4/src/may13/cipher_recipes.cpp` — narrow-domain recipe seeding (32+ entries currently dead per plan line 401; replace with narrow-domain entries matching realistic LM head + small attention shapes)
- `cipher_rt_phase4/src/may13/cipher_lnn.cpp` — CfC LNN forward port from may13 (~430 LOC; currently audit-flagged as dead code per plan line 198)
- `cipher_rt_phase4/src/may13/cipher_koopman_runtime.cpp` — L1.1 Koopman derivation port (dead code per plan line 200)
- `cipher_rt_phase4/cipher_rt_dispatch.c` — SUBSTITUTE-Koopman lane wire (Koopman path that has NEVER FIRED per plan line 401)
- May need nvcc compilation rules in Makefile per plan line 583 ("Add nvcc rules to cipher_rt_phase4/Makefile — DEFERRED to v1.5 (Koopman ports)") — Step 2 unlocks this
**ABI impact:** none expected; pure userspace + may13 source ports
**Close tag:** `week-14-step-2-koopman-tier`

**Sub-elements:**

| Component | Source / spec | LOC |
|-----------|---------------|-----|
| EDMD pipeline real-input snapshot-feeder fix | plan §1 line 103 + `include/may13/cipher_edmd.h` | ~200 |
| Narrow-domain recipe registry seeding (replace dead 32 entries with LM-head-class targets) | plan line 401 | ~150 |
| SUBSTITUTE-Koopman lane dispatch wire (at `cipher_rt_dispatch.c` or equivalent — Koopman branch of SUBSTITUTE actuator) | plan line 380 SUBSTITUTE actuator + plan line 401 dead branch | ~150 |
| Per-tenant Koopman op routing (snapshot model_uuid → recipe lookup via G12) | composes with Step 1 G12 | ~50 |
| RING_WRITE emission on each Koopman-eligible launch (slot REMEMBER) | weak-linked cipher_rt_ring_write | ~30 |
| Optional: nvcc compilation rules for `.cu` kernels (currently deferred per plan line 583) | only if narrow-domain seeding needs `cipher_block_sub_kernel.cu` | ~30 |

Total: ~610 LOC. Step expanded from prompt's ~4-5 eng-days to ~5 eng-days; the may13 port surface is larger than first estimated.

**Verification gate at close:**
- `test_koopman_substitute_lane.c` smoke: register a narrow-domain recipe; issue a matmul matching it; verify the SUBSTITUTE actuator's Koopman lane fires (counter at `cipher_oracle.h:129 billing_gemm_substituted` advances) and the result is numerically close to the dense GEMM (KL ≤ 5.5e-5 vs reference, mirroring W11 Step 2 cryptographic gate framing)
- W7-12 microbench regression (all PASS)
- N=128 30-min regression smoke (gate same as Step 1)

### Step 3 — REMEMBER consumer + LM-head validation + N=128 soak close (W14)

**Calendar:** Week 14 (2 eng-days)
**Touches:**
- `cipher_rt_phase4/cipher_rt_remember_consumer.{c,h}` (new ~150 LOC) — drain RING_WRITE slot 3 into the CfC LNN forward at `cipher_lnn.cpp:428-458`
- `cipher_inject.c` — REMEMBER consumer thread spawn (env-gated `CIPHER_REMEMBER=1`)
**ABI impact:** none
**Close tag:** `week-14-step-3-remember-validate` (== **`week-14-complete`**)

**Sub-elements:**

| Component | Source / spec | LOC |
|-----------|---------------|-----|
| `cipher_rt_remember_consumer.{c,h}` — RING_WRITE slot 3 drain → CfC LNN update | plan §4.9 line 934 + `cipher_lnn.cpp:428-458` | ~150 |
| `cipher_inject.c` REMEMBER thread spawn predicate (env-gated) | plan §4.9 line 954 + W10 Step 1 pattern | ~20 |
| LM-head substitution validation harness (synthetic narrow-domain workload) | informational measurement of Koopman speedup at LM head class shape | ~150 |
| N=128 30-min regression soak close at `week-14-complete` | composed gate same as W10-12 | — |

Total: ~320 LOC.

**Verification gates at close:**
- LM-head validation: substitute fires on at least one Koopman-eligible matmul; numerical result within KL ≤ 5.5e-5 of dense baseline; speedup measurement INFORMATIONAL (not gated against the "7.43×" prior figure since provenance unverified — see §9 residue)
- N=128 30-min regression soak: atomicity 128000/128000, writers-only fairness ≥ 0.85, 0 incoherent reads, aggregate within ±10% of W12 Step 3 baseline 10.66 M/s
- W7-12 microbench regression all PASS
- CFL throttle telemetry surfaced (Step 1 residue closed): the soak now reports `cipher_rt_ring_total_throttled` per tenant; gate < 5% under steady-state load

This step closes v1.2.3 §7 W13-14. Both repos tag `week-14-complete`.

## 5. Risk register

Mapped from W6 audit + W10-12 carries.

| ID | Description | Likelihood | Impact | Mitigation | Detection (regression smoke) |
|----|-------------|------------|--------|------------|------------------------------|
| **R-W13.1** | G12 recipe-table re-key inflates lookup cost (parallels R-W10.1 G3 budget) | LOW | MAJOR | Pre-existing recipe lookup is std::unordered_map; adding model_uuid prefix is one more hash field. Profile: 8192-shape × 100-lookup microbench, p99 ≤ 1 µs (R-W10.1 threshold). | Step 1 close — `test_g12_recipe_keying` p99 lookup |
| **R-W13.2** | EDMD snapshot-feeder fix needs nvcc compilation surface (Makefile change per plan line 583) | MEDIUM | MAJOR | Step 2 adds nvcc rules if needed; documented in scope-lock. Falls back to host-only EDMD if .cu port too disruptive | Step 2 close — build PASS at clean rebuild |
| **R-W14.1** | Narrow-domain recipe seeding doesn't match any real workload (matches plan line 401 audit: registry is dead) | MEDIUM | BLOCKER for "Koopman fires" claim | Step 2 must validate AT LEAST ONE real workload's matmul matches a seeded recipe. Use LM head shapes from TinyLlama (validated path-a target in W6/W11/W12) as the seed set. | Step 2 close — `test_koopman_substitute_lane` confirms `billing_gemm_substituted` > 0 |
| **R-W14.2** | LM-head 7.43× prior measurement may not reproduce on v1.2.3 substrate | MEDIUM | MINOR (informational only — not a gate per §3.2) | Treat "7.43×" as historical context; measure actual speedup against v1.2.3 substrate as informational telemetry. Real benchmark is W15-17 CP 5.5. | Step 3 LM-head validation — reports actual speedup, gates only on correctness (KL ≤ 5.5e-5) |
| **R-W14.3** | REMEMBER drain rate may not match RING_WRITE producer (CFL throttle fires) | MEDIUM | MAJOR | Step 1 ring is 4096 entries × 64 B; producer ~100k-400k/s aggregate at v1 density; REMEMBER CfC update is ~µs per call. If drain rate < producer, CFL kicks in. Mitigation: batched REMEMBER drain (drain 256 entries per CfC forward) | Step 3 close — soak reports `cipher_rt_ring_total_throttled` < 5% steady-state |
| **R-W14.4** | SUBSTITUTE-Koopman lane composes poorly with W11 G3/G4 model-keyed registries (multiple model-keyed lookups per launch) | LOW | MINOR | All three (G3 KV-dedup, G4 Marlin, G12 Koopman) use the same `cipher_get_current_tenant_snapshot()` weak-linked accessor; the snapshot lookup is amortized | Step 3 soak — aggregate rate within ±10% of W12 Step 3 baseline |
| **R-W13.3** | Goal 4 was explicitly DEFERRED by may13 audit (plan line 401: "v1 deliberately defers fixing it"); the v1.2.3 RE-ADOPTION may surface unresolved engineering challenges that delayed it | HIGH | MAJOR | Plan §1 line 103 narrowed scope from wide-domain to narrow-domain explicitly. Step 2 SUBSTITUTE-Koopman lane validation IS the deferred work; if it can't ship, the v1 Koopman claim degrades to "registry seeded but substitution lane unproven; W15-17 measures actual fire rate." | Step 2 close — validation harness shows Koopman lane fires on AT LEAST ONE real-workload shape; if zero matches, scope-degrade documented |

R-G3.1 / R-G5.1 from W10-12 already closed. R-W7.5 (contention scaling) closed at W7-9.

## 6. Eng-day budget + total estimate

| Item | Eng-days |
|------|----------|
| Step 1 G12 registry re-key | 3 |
| Step 2 EDMD pipeline + recipe seeding + SUBSTITUTE-Koopman lane | 5 |
| Step 3 REMEMBER consumer + LM-head validation + soak close | 2 |
| Per-step regression smoke (3 × 0.5) | 1.5 |
| Step doc + commit + memory anchor (3 × 0.5) | 1.5 |
| **Total** | **13** |

Calendar: **2 weeks** (W13 + W14). Plan line 75 budgets W13-14 for Koopman tier + G12 keying as a 2-week slot. **Estimate +30% over the plan's implicit "Koopman held one position" framing**; the W7-9 + W10-12 pattern (5 + 5 + 4 = 14 eng-days for substrate steps) suggests 13 eng-days here is within historical step density. No HARD STOP triggered.

## 7. Next prompt — Step 1 G12 Koopman registry model-keying

```
TASK: W13-14 Step 1 — G12 Koopman registry model-keying.
     ~3 eng-days. cipher_rt_phase4 only (no kmod change).
```

Specifics for the implementation prompt:
- Re-key `cipher_recipes.cpp` (522 LOC) registry lookups to `(model_uuid, shape, …)` per audit line 279
- Re-key `cipher_kernel_table.cpp` (324 LOC) table lookups similarly
- Algebraic test `test_g12_recipe_keying` — analogous to W11 Step 2 G3 algebraic gate
- W7-12 microbench regression + N=128 30-min soak smoke
- Close tag: `week-13-step-1-g12-koopman-keying`

## 8. v1.2.3 §7 W13-14 row (line 1258)

> **13-14** | Koopman tier integration (EDMD pipeline real-input fix + recipe registry narrow-domain seeding + SUBSTITUTE-Koopman lane validation) + **G12 Koopman registry model-keying** *(held from W11-12 to W13-14 per B.1)* | v1.2.3

This scope-lock satisfies the row in full. The 3-step sequence orders G12 first (substrate edit, isolated test), then the tier integration (composes G12 + EDMD + recipe + lane), then REMEMBER consumer + validation gate.

## 9. Honest residue / open items

1. **LM head 7.43× provenance not located.** The W13-14 task prompt referenced "LM head verified 7.43×" but searching cipher-fusion-evidence + cipher_rt_phase4 + cipher_kmod produced no direct match. The figure may live in a prior session's memory anchor not in MEMORY.md, or in a non-committed evidence doc. Treated as informational historical context; Step 3 validation gate is **correctness (KL ≤ 5.5e-5), NOT a speedup target**. Real speedup measurement is W15-17 CP 5.5.
2. **Plan line 401 explicit deferral.** The audit explicitly stated "v1 deliberately defers fixing it (Goal-4 = v2)." v1.2.3 RE-ADOPTS Goal 4 in narrow-domain form per §1 line 103. The engineering challenges that motivated the deferral (registry never matches real workloads) remain open; Step 2's narrow-domain seeding is the explicit mitigation. R-W13.3 flags this as HIGH likelihood, MAJOR impact.
3. **may13 .cu kernel ports may need nvcc.** Plan line 583 listed nvcc compilation as DEFERRED to v1.5 (Koopman ports). Step 2 unlocks this if `cipher_block_sub_kernel.cu` substitute kernel needs to compile. Mitigation: host-only EDMD fallback if .cu port too disruptive in Step 2's eng-day budget.
4. **REMEMBER consumer drain rate** depends on CfC LNN forward cost; Step 1 batched drain (256 entries per CfC forward call) mitigates but not yet measured. Step 3 soak telemetry surfaces actual drain rate vs CFL throttle.
5. **CP 5.5 24-hour hybrid soak** is W15-17, NOT W13-14. Step 3 ships only N=128 30-min synthetic soak + LM-head validation; the full benchmark + ≥5 model families + cryptographic billing chain verification is the next scope-lock (W15-17).
6. **No kmod ABI change** across W13-14. cipher_kmod stays at week-9-complete (0.6.5) through Step 3.

## 10. Fingerprints expected at W13-14 close

```
cipher_rt_phase4   <new commit>     tag week-14-complete
                                    aliases week-14-step-3-remember-validate
libcipher_rt.so    <new md5>        (~ +1300 LOC across 3 steps)
cipher_kmod        unchanged        stays at week-9-complete (0.6.5)

cipher-fusion-evidence:
  WEEK_13_STEP_1_G12_KOOPMAN_KEYING.md
  WEEK_14_STEP_2_KOOPMAN_TIER.md
  WEEK_14_STEP_3_REMEMBER_VALIDATE.md (== W13-14 close summary)
```
