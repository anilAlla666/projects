# W12 Step 8: batched reframe of audit items D1, D2, D4, D10

**Date:** 2026-05-23
**Scope-drift audit closures:** D1, D2, D4, D10 of 15
**Cumulative closures:** D5, D9, D14, D3, D6, **D1, D2, D4, D10** = 9 of 15
**Open audit items:** D7, D8, D11, D12, D13, D15 (6 of 15)

## 1. Pre-condition verification

| Component | Tag / Anchor | md5 |
|-----------|--------------|-----|
| `cipher_rt_phase4` | `week-12-step-6-d3-l2-wireup` (`adbe121`) | `libcipher_rt.so` `33412ffc97b714662e0824e1de8362bd` |
| `cipher_kmod` | `week-9-complete` (0.6.5) | `cipher_kmod.ko` `8c9fdd016897436ceff382c4e9178e07` |
| `cipher_vllm_plugin` | (untagged tree) | `cipher_vllm_kv.py` `2b6cedab89387c30becd49a27313ceb4` |
| `cipher-fusion-evidence` | `2cbe7c6` | `WEEK_12_STEP_7_D6_CFL_TELEMETRY.md` present |
| `WEEK_12_SCOPE_DRIFT_AUDIT.md` | present | |
| `PLAN_KOOPMAN_RECONCILIATION.md` | present (D9 closure) | |

## 2. Rationale

`WEEK_12_SCOPE_DRIFT_AUDIT.md` partitioned its 15 items by category. D1,
D2, D4, D10 fall in categories (a) framing precision and (b) unsourced
number — the substrate is correct as shipped; the documentation needs to
match. No engineering work. Per memory `regression-discipline`, no
substrate touch. The step doc here is the documentary closure surface.

D1, D2, D4 are framing-precision items where the *substrate* in W11+W12
already does the right thing, and the W11+W12 step docs already document
the precise framing. The closure here is reconciling pitch-adjacent
references (memory anchors that carried the audit-time projection
language forward) with the W12 step-doc honest numbers.

D10 is an unsourced-number item. The W13-14 scope-lock already de-gated
the unsourced 7.43× value at `WEEK_13_14_SCOPE_LOCK.md:188` and treats
it as informational-only context. The closure here is documentary
acknowledgment that the de-gating is complete and that re-measurement
lands at W13-14 Step 3 (correctness gate KL ≤ 5.5e-5) + W15-17 CP 5.5
(production speedup gate).

## 3. Per-item reframe summary

### 3.1 D1 — G5 18× → measured 12.5× reduction

**Original framing (W6 audit-time projection):**
- `WEEK_6_G5_PATH_A_VERIFICATION.md:53`: "vs current default 100 × 80 GiB = 8000 GiB: 18× reduction"
- `WEEK_6_G5_PATH_A_VERIFICATION.md:80`: "Scenario 1 (realistic CP 5.5): 437 GiB / 100 tenants. 18× reduction vs current default"
- `WEEK_6_G5_PATH_A_VERIFICATION.md:81`: "Scenario 1 + Track 2 weight sharing: 152 GiB / 100 tenants"

**Substrate reality (W12 Step 3 measurement):**
- `WEEK_12_STEP_3_G5_L2_PERSIST.md:70`: realistic CP 5.5 mix yields `640 GiB` aggregate; not 437.
- `WEEK_12_STEP_3_G5_L2_PERSIST.md:73`: "the W6 audit's 437 GiB headline was derived from an assumed '18× reduction vs hardcoded 80 GiB default' factor. My formula achieves a real `12.5× reduction` (640 GiB / 100 tenants = 6.4 GiB/tenant avg vs 80 GiB hardcoded baseline → 12.5×)."

**References updated:**
1. `memory/g5-path-a-verified.md:3` (description line) — added "W12 Step 3 SUPERSEDED with measured 640 GiB / 12.5× reduction on the realistic mix (D1 reframe W12 Step 8)"
2. `memory/g5-path-a-verified.md:27-29` — body lines now mark the 437 / 152 GiB numbers as "W6 audit projection" with an explicit "SUPERSEDED at W12 Step 3 (D1 reframe W12 Step 8)" paragraph documenting the 640 GiB / 12.5× substrate reality
3. `memory/MEMORY.md:69` (index entry) — reflects "W6 audit projected 437 / 18×; W12 Step 3 measurement 640 GiB / 12.5× on realistic mix (D1 reframe W12 Step 8)"

**Historical references left intact** (record the projection at audit-time):
- `WEEK_6_G5_PATH_A_VERIFICATION.md` (the W6 audit doc itself)
- `WEEK_12_SCOPE_DRIFT_AUDIT.md` (the audit cites both numbers)
- `WEEK_12_STEP_3_G5_L2_PERSIST.md` (already documents the corrected framing)
- `WEEK_10_12_SCOPE_LOCK.md:147` (forward gate written *before* W12 Step 3; references the W6 audit projection as the input plan)
- `memory/w12-complete.md:53` (already correctly notes "12.5× reduction... audit's 18× was best-case")

**Corrected forward framing:**
> 12.5× reduction measured at W12 Step 3 substrate; W6 audit's 18× was optimistic best-case. Realistic CP 5.5 mix N=100 = 640 GiB (under 1 TiB host envelope). Track 2 weight-sharing compose measurement deferred to W15-17 CP 5.5.

### 3.2 D2 — L2 framing per-tenant-tensor → per-tenant-STREAM

**Substrate reality (already documented):**
- `WEEK_12_STEP_3_G5_L2_PERSIST.md:83`: "This is per-tenant-STREAM L2 policy application against a shared tensor set, NOT 'per-tenant L2 persistence' in the per-tenant-tensor sense. The existing `tensors[]` array in `CipherL2PersistState` stays global in v1; Step 3 wires each tenant's named CUDA stream to re-apply the same window policy."

**Memory already correct:**
- `memory/w12-complete.md:34`: "Advisor catch #1 — honest framing: per-tenant-STREAM L2 policy application against shared tensors[] set. NOT per-tenant tensor tracking (that's W13-14 work)"
- `memory/w12-complete.md:52`: "L2 framing is per-tenant-STREAM; per-tenant tensor tracking is W15-17 CP 5.5"

**Forward references that remain ambiguous:**
- `CIPHER_REENGINEERING_PLAN.md:799` — table cell reads "per-tenant L2 region table" as a future v2 spec, not a v1 claim. Plan-level future-spec language; left intact (the plan documents the v2 intent).
- `phase_c/track_3/TRACK_3_SC6_DESIGN_MEMO.md:47` — historical Track 3 design memo references "per-tenant L2 check" as the design surface. Historical; left intact.

**No additional memory edits needed.** The D2 reframe is fully landed in W12 Step 3 + `w12-complete`. The closure here is documentary acknowledgment.

**Corrected forward framing:**
> per-tenant-STREAM L2 policy (cudaAccessPolicyWindow attribute set per stream at REGISTER_STREAMS time). The `tensors[]` array in `CipherL2PersistState` stays global in v1. Per-tenant tensor tracking is W13-14+ work.

### 3.3 D4 — G4 Marlin "arena" → map re-key

**Substrate reality (already documented):**
- `WEEK_11_STEP_2_G3_G4_TC_SDPA.md:92`: "Scope-lock §3 Step 2 described G4 as 'per-tenant Marlin arena.' Reading the code: Marlin has no arena. It has `std::unordered_map<const void *, WeightSlot> g_weights` keyed on the FP16 weight pointer."
- `WEEK_11_STEP_2_G3_G4_TC_SDPA.md:108`: "the actual fix is the map key change"

**Memory already correct:**
- `memory/w11-step-2-g3-g4-tc-probe.md:27`: "Scope-lock's 'per-tenant arena' framing was misleading; Marlin has no arena, just `std::unordered_map<const void *, WeightSlot> g_weights`"

**Forward references that remain in scope-lock language:**
- `WEEK_10_12_SCOPE_LOCK.md:106, 117, 125` — uses old "per-tenant arena" framing. Historical scope-lock written *before* W11 Step 2 located the actual structure. Left intact.
- `memory/w10-step-1-ring-write.md:59, 62` — historical anchor with future-tense Step 2 next-prompt language. Left intact.

**No additional memory edits needed.** The D4 reframe is fully landed in W11 Step 2 + `w11-step-2-g3-g4-tc-probe`. The closure here is documentary acknowledgment.

**Corrected forward framing:**
> per-tenant Marlin `g_weights` map re-keyed on `(tenant_id, w_ptr)` tuple. Marlin has no arena; advisor reframe at W11 Step 2 landed the structurally-correct fix as a map key change. R-W11.2 HBM growth concern downgraded MAJOR → MINOR via Track 2 weight-sharing (same-model tenants share upstream `w_ptr`).

### 3.4 D10 — LM head 7.43× provenance unsourced

**Provenance search result (re-confirmed at this step):**
The audit at `WEEK_12_SCOPE_DRIFT_AUDIT.md:164` records: "Searched
`cipher-fusion-evidence/`, `cipher_rt_phase4/`, `cipher_kmod/`. Single
grep hit for `7.43` in `t4_6_3_dedup_report.md:41` is unrelated
(conversation window stats). No measurement file produces 7.43×."

This step's grep across the same trees yields the same result — every
hit for `7.43×` is in:
- `WEEK_12_SCOPE_DRIFT_AUDIT.md` (the audit citing the unsourced claim)
- `WEEK_13_14_SCOPE_LOCK.md:44, 130, 146, 188` (already de-gates the number)
- `memory/w13-14-scope-lock.md:3, 39, 45` (memory already records "provenance NOT located, treated as historical not gated")
- `memory/MEMORY.md:82` (index entry already reflects de-gating)
- `memory/scope-drift-audit.md:20, 35` (audit summary)

**De-gating already complete:**
- `WEEK_13_14_SCOPE_LOCK.md:130`: "speedup measurement INFORMATIONAL (not gated against the '7.43×' prior figure since provenance unverified)"
- `WEEK_13_14_SCOPE_LOCK.md:146`: "R-W14.2 ... Treat '7.43×' as historical context; measure actual speedup against v1.2.3 substrate as informational telemetry."
- `memory/w13-14-scope-lock.md:45`: "LM head 7.43× provenance NOT located ... Step 3 validation gates on correctness (KL ≤ 5.5e-5), NOT speedup. Real speedup measurement is W15-17 CP 5.5."

**Memory anchor #2 update (per task spec):**

The task prompt named "Memory anchor #2 goal 4" as the target for replacement of `"(4) O(1) Koopman substitution — narrow domain in v1, wide domain v2 research, LM head verified 7.43×"` with `"(4) O(1) Koopman substitution — narrow domain in v1, wide domain v2 research. v1 narrow-domain validated at W13-14 Step 3 (cryptographic gate KL ≤ 5.5e-5). Production speedup measured at W15-17 CP 5.5."`

**A.2 finding:** No memory anchor in the on-disk `/home/ubuntu/.claude/projects/-home-ubuntu/memory/` tree contains the exact literal string `"LM head verified 7.43×"`. The five product-goals language with this 7.43× substring appears nowhere in MEMORY.md or any reachable memory anchor. The closest hit, `memory/plan-v1.2.3.md:23`, lists Goal 4 as "O(1) Koopman compute substitution (RETAINED, position-shifted to W13-14)" without the 7.43× appendage. `WEEK_12_SCOPE_DRIFT_AUDIT.md:162` confirms: "User-supplied W13-14 task prompt: 'Memory #2 goal 4: O(1) Koopman substitution — narrow domain v1, wide domain v2 research, LM head verified 7.43×.' Treated as historical pitch claim."

**Closure resolution:** D10 is closed via the W13-14 scope-lock's de-gating + memory anchor `w13-14-scope-lock` recording "provenance NOT located, treated as historical not gated." No additional memory edit is possible because the 7.43× appendage exists only in the user's session-time task prompt context, not in any on-disk anchor. If the user's "Memory #2" refers to an out-of-repo pitch document, that update lands outside this step's reachable surface. **Flagging for user attention.** All in-repo / in-memory forward references already carry the de-gated framing.

**Corrected forward framing (canonical, for any future pitch artifact):**
> O(1) Koopman substitution — narrow domain in v1, wide domain v2 research. v1 narrow-domain validated at W13-14 Step 3 (cryptographic gate KL ≤ 5.5e-5 against dense baseline). Production speedup measured at W15-17 CP 5.5. The "7.43×" prior figure is informational historical context, not a gate.

## 4. Memory updates made

| Anchor | Change | Reason |
|--------|--------|--------|
| `g5-path-a-verified.md:3` | description rewritten — added "W12 Step 3 SUPERSEDED with measured 640 GiB / 12.5×" | D1 |
| `g5-path-a-verified.md:27-29` | body — marked 437 / 152 as W6 projection + added explicit SUPERSEDED paragraph | D1 |
| `MEMORY.md:69` | index entry — reflects projection-vs-measurement reframe | D1 |
| `MEMORY.md` (new entry) | one-line `d1-d2-d4-d10-reframes` pointer added | this step closure |

**No anchor changes for D2** (already correctly framed in `w12-complete`).
**No anchor changes for D4** (already correctly framed in `w11-step-2-g3-g4-tc-probe`).
**No anchor changes for D10** (already correctly framed in `w13-14-scope-lock`; the literal "Memory #2 goal 4" target is not on-disk).

## 5. No source code changes; no substrate md5 changes

Post-reframe verification:

| File | Pre-reframe md5 | Post-reframe md5 | Match |
|------|-----------------|------------------|-------|
| `libcipher_rt.so` | `33412ffc...` | (verified at step close) | YES |
| `cipher_kmod.ko` | `8c9fdd01...` | (verified at step close) | YES |
| `cipher_vllm_kv.py` | `2b6cedab...` | (verified at step close) | YES |

Git working trees in `cipher_rt_phase4` and `cipher_kmod` clean (no
modified, no untracked source).

## 6. Cumulative audit closures

| # | Item | Closure step | Mode |
|---|------|--------------|------|
| 1 | D5 — SDPA trampoline stream-fill | W12 Step 4 (`1466193`) | substrate fix |
| 2 | D9 — Plan Koopman ambiguity | W12 Step 5 prelude (`69203d3`) | plan reconciliation |
| 3 | D14 — TPS gate ±3%→±5% + env-gate | W12 Step 5 (`b360fc1`) | substrate addition |
| 4 | D3 — L2 call-site wireup | W12 Step 6 (`adbe121`) | substrate + plugin |
| 5 | D6 — CFL throttle telemetry | W12 Step 7 (`2cbe7c6`) | additive measurement |
| 6 | **D1 — G5 18×→12.5× reframe** | **W12 Step 8 (this step)** | **documentation** |
| 7 | **D2 — L2 per-stream reframe** | **W12 Step 8 (this step)** | **documentation** |
| 8 | **D4 — G4 Marlin map re-key reframe** | **W12 Step 8 (this step)** | **documentation** |
| 9 | **D10 — LM head 7.43× de-gating** | **W12 Step 8 (this step)** | **documentation** |

**9 of 15 audit items closed.**

**6 items open for user adjudication:**
- D7 — Mistral E.7 env-block (CP 5.5 deferred)
- D8 — real-LLM heterogeneous benchmark (CP 5.5 deferred)
- D11 — 24h soak (CP 5.5 deferred; current is 30-min/1h)
- D12 — cryptographic external audit (CP 5.5 deferred)
- D13 — May-13 reconstruction (W15-17 carry)
- D15 — CP 5.5 timeline (framing carry)

All 6 remaining are CP-5.5-deferred or framing-carry; none require
substrate work at W13-14.

## 7. Honest residue

1. **D10 "Memory #2" target out-of-repo.** The user's task prompt
   referenced the five-goals language with "LM head verified 7.43×" as
   the appendage. No on-disk memory anchor matches that literal. If the
   target is an external pitch artifact, this step cannot update it.
   Surfaced at §3.4 for user attention.
2. **D1 W6 path-a memory anchor now bilingual.** `g5-path-a-verified.md`
   records both the W6 audit projection (437 / 152 GiB / 18×) and the
   W12 Step 3 measurement (640 GiB / 12.5×) with explicit
   `SUPERSEDED` framing. A future reader following the chain to W12
   Step 3 lands on the correct measurement. Index entry
   `MEMORY.md:69` similarly bilingual.
3. **D2 and D4 add no new memory text.** Both have correct framing in
   their respective newer anchors (`w12-complete`, `w11-step-2-g3-g4-tc-probe`).
   This step's role is to *acknowledge* that the substrate-vs-framing
   reconciliation is complete, not to duplicate the framing.
4. **No CP 5.5-deferred item closes here.** D7, D8, D11, D12, D13, D15
   require substrate measurement or external work not yet scheduled
   (W15-17 CP 5.5 hybrid heterogeneous benchmark).
5. **Audit doc itself unchanged.** `WEEK_12_SCOPE_DRIFT_AUDIT.md` is
   the historical record of what was found; this step records what
   closed.

## 8. Files

- `cipher-fusion-evidence/WEEK_12_STEP_8_D1_D2_D4_D10_REFRAMES.md` (this doc)
- `memory/g5-path-a-verified.md` (D1 anchor reframe)
- `memory/MEMORY.md` (D1 index entry reframe + new W12 Step 8 pointer)
- `memory/d1-d2-d4-d10-reframes.md` (new closure anchor)

## 9. Anchors

| Tree | Anchor | Status |
|------|--------|--------|
| `cipher_rt_phase4` | `week-12-step-6-d3-l2-wireup` (`adbe121`) | UNCHANGED |
| `cipher_kmod` | `week-9-complete` (0.6.5) | UNCHANGED |
| `cipher_vllm_plugin/cipher_vllm_kv.py` | `2b6cedab89387c30becd49a27313ceb4` | UNCHANGED |
| `cipher-fusion-evidence` | new HEAD with this file + memory edits | doc-only commit |

No tag added (documentation-only step, no substrate move).
