# Weeks 6-12 scope drift audit

**Date:** 2026-05-23
**Scope:** v1.2.3 §7 substrate work shipped W6 through W12 (rows W6, W7-9, W10-12)
**Purpose:** identify gaps between plan + audit + pitch claims and what shipped on substrate. Surface options. User adjudicates.
**Method:** read-only walk of plan, audit, step docs, pitch artifacts. Each gap gets equal treatment. No prioritization.
**Discipline:** no remediation choices in this doc. No recommendations. No auto-degrade language.

## 1. Pre-conditions verified

| Item | Expected | Observed | OK |
|------|----------|----------|----|
| `cipher_kmod` tag | `week-9-complete` (0.6.5) | confirmed | ✓ |
| `cipher_rt_phase4` tag | `week-12-complete` (fec9cc3) | confirmed | ✓ |
| W6 step docs present | yes | WEEK_6_*.md × 9 docs (G1+G2, G5 verify, May-13 plan, transition, bench harness, kvdedup, option-1-redo, entry preflight, audit) | ✓ |
| W7-9 step docs present | yes | WEEK_7_STEP_1/2/3/4 + WEEK_9_STEP_5 + WEEK_7_9_SCOPE_LOCK | ✓ |
| W10-12 step docs present | yes | WEEK_10_STEP_1 + WEEK_11_STEP_2 + WEEK_12_STEP_3 + WEEK_10_12_SCOPE_LOCK | ✓ |

## 2. Audit scope statement

Substrate work shipped under v1.2.3 §7 W6 through W12. Audit ends at the `week-12-complete` tag. W13-14 scope-lock (cipher-fusion-evidence 0dd86d3) IS in scope as a target document for drift detection but Step 1-3 implementation has not started yet. CP 5.5 W15-17 is OUT of scope; this audit identifies items that are reframed AS W15-17 work but does not assess W15-17 itself.

Source documents walked:
- `CIPHER_REENGINEERING_PLAN.md` v1.2.3 (1907 lines)
- `WEEK_6_ARCHITECTURE_GAP_AUDIT.md` (G1-G12)
- `WEEK_6_G5_PATH_A_VERIFICATION.md`
- All `WEEK_6_*.md`, `WEEK_7_*.md`, `WEEK_9_*.md`, `WEEK_10_*.md`, `WEEK_11_*.md`, `WEEK_12_*.md`
- `WEEK_7_9_SCOPE_LOCK.md`, `WEEK_10_12_SCOPE_LOCK.md`, `WEEK_13_14_SCOPE_LOCK.md`
- `CIPHER_PLAN_EXECUTIVE_SUMMARY.md` (pitch claims)
- Memory anchors in `/home/ubuntu/.claude/projects/-home-ubuntu/memory/`

Drift categories used:
- **(a)** framing precision only — substrate delivers original intent at the original measurement
- **(b)** substrate narrower than framing — substrate ships less than the framing implies
- **(c)** substrate broader than framing — substrate ships more than the framing implies
- **(d)** commitment not shipped at all — work item carried forward or unstarted

## 3. Drift item inventory

### D1 — G5 path-a reduction factor: claimed 18×, measured 12.5×

| Field | Detail |
|-------|--------|
| Original commitment | `WEEK_6_G5_PATH_A_VERIFICATION.md:80`: "Scenario 1 (realistic CP 5.5): 437 GiB / 100 tenants. **18× reduction vs current default.**" |
| Substrate reality | `WEEK_12_STEP_3_G5_L2_PERSIST.md:73`: "The audit's 437 GiB headline was derived from an assumed '18× reduction vs hardcoded 80 GiB default' factor. My formula achieves a real **12.5× reduction**" |
| Drift description | Realistic CP 5.5 mix aggregate is 640 GiB (per `WEEK_12_STEP_3_G5_L2_PERSIST.md:70`), not 437. 12.5× reduction not 18×. Both fit 1 TiB host VA envelope. |
| Category | **(b)** substrate narrower than framing |
| Pitch impact | Memory anchor `g5-path-a-verified` referenced 437 / 152 GiB headlines. Pitch language "18× reduction" replaced by "12.5× reduction" in Step 3 step doc but not in pitch deck (if any). |
| Backfill path | Re-derive formula to land closer to audit's 18×. Audit estimate was optimistic (8000/444=18); to hit 18× requires aggregate < 444 GiB at N=100. Would need either fewer 7B-class tenants or smaller max_model_len. ~1 eng-day for formula tuning + re-verification across 5 families. Risk: formula starts to misalign with actual workload sizes; honest numbers regress. |
| Reframe path | Update pitch + memory anchor #16 to "12.5× reduction" / "6.4 GiB/tenant average" headline. Update plan §7 G5 row + WEEK_6_G5_PATH_A_VERIFICATION.md §5 to flag the original projection as best-case-optimistic + measured value as canonical. ~0.5 eng-day. |
| Defer path | Leave as-is; the 12.5× number is in the W12 step doc residue. Pitch language correction lands at W15-17 CP 5.5 alongside real-LLM density measurement. |

### D2 — L2 persistence: per-tenant tensor (framing) vs per-tenant stream (substrate)

| Field | Detail |
|-------|--------|
| Original commitment | `WEEK_10_12_SCOPE_LOCK.md:148`: "L2 persistence per-tenant policy via cudaAccessPolicyWindow." Plan §7 W10-12 row line 1258: "G5 VA pool implementation per W6 audit decision" (does not explicitly name L2; L2 came in as scope-lock blog-derived addition). |
| Substrate reality | `WEEK_12_STEP_3_G5_L2_PERSIST.md:83`: "This is per-tenant-STREAM L2 policy application against a shared tensor set, NOT 'per-tenant L2 persistence' in the per-tenant-tensor sense. The existing `tensors[]` array in `CipherL2PersistState` stays global in v1." |
| Drift description | Per-tenant TENSOR tracking would require Track 2 weight sharing to provide per-tenant weight tensor views. Shipped: same global `tensors[]` re-applied against per-tenant stream attributes via new `cipher_l2_persist_apply_for_tenant()`. |
| Category | **(b)** substrate narrower than framing |
| Pitch impact | None direct — pitch does not name L2 persistence at v1 framing depth. Memory anchor `w12-complete` documents the honest framing. |
| Backfill path | Per-tenant `CipherL2PersistState` array + per-tenant `tensors[]` populated from Track 2 weight sharing. ~5-8 eng-days substrate refactor + Track 2 integration. Risk: touches Track 2 weight sharing internals which were closed at W5; opens regression surface. |
| Reframe path | Step doc already documents honest framing. Update pitch (if any) to "per-tenant-stream L2 policy application against shared tensor set"; defer full per-tenant tensor tracking to W15-17 CP 5.5 with explicit scope-lock language. ~0 eng-days code, ~0.5 eng-day doc. |
| Defer path | Leave as-is. W13-14 scope-lock §9 #5 already flagged CFL telemetry surfacing as W14 work; per-tenant L2 tensor tracking joins that deferred list. v2 carry. |

### D3 — L2 call-site activation: API ships, plugin does not call it

| Field | Detail |
|-------|--------|
| Original commitment | `WEEK_10_12_SCOPE_LOCK.md` Step 3: "Apply at stream registration time in `cipher_vllm_kv.py` REGISTER_STREAMS path." |
| Substrate reality | `WEEK_12_STEP_3_G5_L2_PERSIST.md:108`: "the plugin doesn't currently invoke the L2 substrate directly — that wiring lives in `cipher_runtime.cpp`. Step 3 ships the API; the plugin-side invocation is a one-line follow-on in W13-14 when the Koopman tier needs L2 policy per-tenant." |
| Drift description | New API `cipher_l2_persist_apply_for_tenant()` and RING_WRITE slot 7 emission exist. No caller invokes them in v12 substrate. RING_WRITE slot 7 reservation fires zero events at v12 close. |
| Category | **(d)** commitment not shipped at all (the API exists; the wire-up does not) |
| Pitch impact | None direct. Substrate primitive is shippable on its own. |
| Backfill path | Add the one-line plugin call after `_register_streams_with_kmod()` in `cipher_vllm_kv.py`. ~0.5 eng-day including test that RING_WRITE slot 7 fires. |
| Reframe path | Step doc already names this as "one-line follow-on in W13-14." Move the one-line wire-up into W13-14 Step 3 (REMEMBER consumer scope) without re-opening W12. ~0 eng-days new work. |
| Defer path | Leave as v15-17 carry alongside per-tenant tensor tracking. |

### D4 — G4 Marlin "arena" framing (scope-lock) vs map re-key (reality)

| Field | Detail |
|-------|--------|
| Original commitment | `WEEK_10_12_SCOPE_LOCK.md:131-140`: "G4 Marlin tenant-scoped weight kit... per-tenant arena lookup keyed on `model_uuid`." Also referenced as "per-tenant Marlin arena" in `WEEK_6_ARCHITECTURE_GAP_AUDIT.md` G4 row. |
| Substrate reality | `WEEK_11_STEP_2_G3_G4_TC_SDPA.md:108`: "Marlin has no arena. It has `std::unordered_map<const void *, WeightSlot> g_weights` keyed on the FP16 weight pointer. The actual fix is the map key change." |
| Drift description | Framing implied a per-tenant arena allocator. Reality: a `std::unordered_map` re-keyed from `const void *` to `(tenant_id, w_ptr)`. Functionally equivalent for the audit's correctness intent. R-W11.2 HBM growth concern downgraded MAJOR → MINOR via Track 2 weight sharing. |
| Category | **(a)** framing precision only — substrate delivers original intent at the original audit measurement (closes G4 cross-model defect) |
| Pitch impact | None. "Per-tenant Marlin weight kit" pitch language remains defensible (the cache IS now per-tenant-keyed); the arena-vs-map distinction is implementation detail. |
| Backfill path | Build a real per-tenant arena allocator with explicit memory management. ~5-8 eng-days; touches the Marlin primary-context-pin work. Risk: re-opens the Marlin × partitioning constraint documented in `cipher-marlin-primary-ctx-pin`. |
| Reframe path | Step doc already documents the precise framing. Update plan §3.1 G4 row + scope-lock template language to "tenant-keyed cache" instead of "per-tenant arena." ~0.5 eng-day doc. |
| Defer path | Leave as-is. Map re-key delivers the correctness property; arena framing is doc-only. |

### D5 — SDPA stream-resolution: descriptor field exists, trampolines don't fill it

| Field | Detail |
|-------|--------|
| Original commitment | `WEEK_10_12_SCOPE_LOCK.md` B.3 + Step 2: "extending `cipher_rt_attn_call` with a `stream` field is a 3-line ABI tweak + 5-line plumbing change... resolver replaces `tenant_id=0` fallback." |
| Substrate reality | `WEEK_11_STEP_2_G3_G4_TC_SDPA.md:193`: "Trampolines that don't fill `call.stream` (current state — no caller does) silently fall back to tenant 0; functionally identical to W9 behavior. Future SDPA trampolines that propagate the stream get correct multi-tenant routing." `WEEK_11_STEP_2_G3_G4_TC_SDPA.md:236`: "SDPA trampolines do not fill `call.stream`. The descriptor field exists; the 3 zero-init constructors default to NULL." |
| Drift description | Step 2 added the descriptor field + the resolver call site. No caller fills the field. Multi-tenant SDPA routing is still tenant_id=0 fallback in practice. |
| Category | **(d)** commitment not shipped at all (the wire is in place; the data does not flow through it) |
| Pitch impact | Strategic — Goal 1 (100-agent heterogeneous-model multiplexing) at attention-heavy workloads still routes all SDPA to tenant 0. cuBLAS path is correctly multi-tenant via W9 Step 5 stream resolver; SDPA is not. |
| Backfill path | Touch libtorch SDPA trampoline construction sites (out of cipher_rt_phase4 scope; requires patching torch dispatch). ~3-5 eng-days. Risk: depends on libtorch internals that v1 deliberately stays decoupled from. |
| Reframe path | Update plan §7 + W7-9 Step 5 residue to "SDPA carries stream field; multi-tenant attention routing is W15-17 CP 5.5 work alongside Track 2 SDPA dispatcher port." ~0.5 eng-day doc. |
| Defer path | Leave as-is. Attention-heavy multi-tenant routing is W15-17 CP 5.5 measurement gate; if it surfaces as a critical drift, backfill at that point. |

### D6 — CFL throttle telemetry untested under realistic drain rate

| Field | Detail |
|-------|--------|
| Original commitment | `WEEK_10_STEP_1_RING_WRITE.md` Step 1 ships CFL invariant with 75% threshold. `WEEK_11_STEP_2_G3_G4_TC_SDPA.md:229`: "Documenting as residue; W12 Step 3 step doc will surface throttle telemetry." `WEEK_12_STEP_3_G5_L2_PERSIST.md:150`: "CFL throttle telemetry surfacing deferred. Step 1 harness `cipher_test_commit_n128` doesn't report `cipher_rt_ring_total_throttled`. W13-14 Koopman tier integration will add this when consumers actually drain the ring." |
| Substrate reality | `test_ring_write` Case 5 verified CFL false-positive 0% under matched producer/consumer load. No consumer drains the ring under N=128 soak (REMEMBER + CLASSIFY consumers reserved but not wired). Throttle counter `cipher_rt_ring_total_throttled` is never reported across W10-12 soaks. |
| Drift description | CFL throttle calibration was the load-bearing Step 1 risk per scope-lock R-W10.5. Synthetic Case 5 PASS; under N=128 soak the ring is producer-only (consumers not draining), CFL gate not exercised at realistic scale. |
| Category | **(b)** substrate narrower than framing (CFL works in synthetic test; real-drain regime untested) |
| Pitch impact | Indirect — pitch claim "RING_WRITE producer p99 < 150 ns" remains defensible (Step 1 microbench PASS). Pitch claim "CFL invariant prevents producer/consumer rate divergence" is unverified in production-shape load. |
| Backfill path | Wire a synthetic drain thread into `cipher_test_commit_n128` that consumes RING_WRITE entries at variable rates; soak with throttle counter surfaced. ~1 eng-day. |
| Reframe path | Document CFL as "synthetic-test-validated; production drain validated at W13-14 Step 3 when REMEMBER consumer wires." ~0.5 eng-day doc. |
| Defer path | W13-14 Step 3 scope-lock already commits to surfacing throttle telemetry at W14 close. Carry as-is. |

### D7 — Mistral E.7 DeepGEMM environmental block (4-step carry)

| Field | Detail |
|-------|--------|
| Original commitment | `WEEK_7_9_SCOPE_LOCK.md` E.7 vLLM gate: Mistral-7B B=1 decode TPS within ±3% of Step 3 baseline 163.6 tok/s. |
| Substrate reality | `WEEK_9_STEP_5_N128_SOAK.md:231`: "Part F Mistral E.7 — SKIPPED (environment block carry-forward)." `WEEK_11_STEP_2_G3_G4_TC_SDPA.md:233`: "End-to-end KL formulation is deferred until env block clears." `WEEK_12_STEP_3_G5_L2_PERSIST.md:151`: "Mistral env block carries forward from Step 4/5." |
| Drift description | Mistral graph-capture segfaults at baseline (no CIPHER injection). DeepGEMM env block carries from W7-9 Step 4 through W12 Step 3. 5 step docs flag it as residue. TinyLlama smoke is the substrate gate; Mistral measurement remains unrun. |
| Category | **(d)** commitment not shipped at all (the gate the scope-lock named was never run) |
| Pitch impact | Strategic — Goal 2 (MFU per workload-class roofline) and Goal 3 (TPW composed lift) reference Mistral-7B as the canonical mid-size representative model in plan line 1638. Without Mistral inference working, the canonical model TPS measurement is unavailable. |
| Backfill path | `pip install deep_gemm` or build from source; verify Mistral runs; re-baseline TPS at each prior step substrate; run E.7 retroactively at every step closeout. ~1-3 eng-days depending on deep_gemm install complexity. |
| Reframe path | Declare TinyLlama-1.1B + Llama-3.2-1B as the v1 measurement environment; document Mistral as "blocked at v1 environment; canonical Mistral measurement is W15-17 CP 5.5." Update each step doc + plan §7 with environment statement. ~1 eng-day. |
| Defer path | Carry to W15-17 CP 5.5 where Mistral measurement is the headline. Continue using TinyLlama as substrate gate. |

### D8 — Real-LLM heterogeneous validation deferred to CP 5.5

| Field | Detail |
|-------|--------|
| Original commitment | Plan §1 product target line 98: "100 concurrent agents per H100, each potentially running a different model architecture (Mistral / Qwen / Llama / SLMs)." Plan §7 W10-12 row line 1257 implies G3/G4/G5 substrate enables this. |
| Substrate reality | All W7-12 gates are synthetic. N=128 soak is single-process 128 thread pairs (`cipher_test_commit_n128`). No real-LLM heterogeneous workload runs. G3 algebraic gate verifies hash distinct; does not load two models. G4 map re-key passes microbench; does not exercise two model uuids in inference. G5 dry-run is `compute_va_gib` math; no model load. |
| Drift description | Real-LLM heterogeneous validation is gated on W15-17 CP 5.5. v1.2.3 §7 W7-12 substrate work is all substrate-property gates, not workload gates. The seed pitch claim "5 model families coexisting" rests entirely on W15-17 measurement. |
| Category | **(d)** commitment not shipped at all in W7-12; explicit by scope-lock design |
| Pitch impact | Strategic — the headline Goal 1 product target is unverified end-to-end. Plan + scope-locks document this as W15-17 work; pitch language must not claim it as shipped. |
| Backfill path | Run a real-LLM N=2 cross-model smoke (Mistral + Llama or TinyLlama + Phi-2 if Mistral env blocked) at W7-9 + W10-12 closeouts. ~2-4 eng-days. Surfaces real-LLM regression earlier than W15-17. |
| Reframe path | Pitch language is "substrate-validated; real-LLM heterogeneous benchmark is the W15-17 CP 5.5 deliverable." All current pitch claims that imply real-LLM heterogeneous validation already shipped need re-language. ~1 eng-day pitch + plan updates. |
| Defer path | Carry to W15-17 CP 5.5 as designed; this is the explicit plan. No new work. |

### D9 — Koopman tier: plan line 401 explicit deferral vs §1 line 103 retention

| Field | Detail |
|-------|--------|
| Original commitment | Plan §1 line 103: "**O(1) Koopman compute substitution** — RETAINED in v1 per v1.2.2 A1 and v1.2.3 §7 re-sequence... v1 scope is the narrow-domain Koopman; wide-domain Koopman remains v2 research." |
| Conflicting commitment | Plan line 401: "**This is the central engineering reality the unified runtime must address — and v1 deliberately defers fixing it (Goal-4 = v2).**" Plan line 184: "the Koopman O(1) lane is reserved in the table but **deferred to v2**." Plan line 299: "The runtime that consumes the math (`cipher_koopman_runtime.cpp`) is dead code with zero callers in either tree... The actual wire-up to live decode is **v2 work**." Plan line 314: "Goal 4 (Koopman O(1)) is **dead code today, by named choice, and v2 scope**." |
| Substrate reality | W13-14 scope-lock R-W13.3 flags this HIGH likelihood: "Goal 4 was explicitly DEFERRED by may13 audit (plan line 401: 'v1 deliberately defers fixing it'); the v1.2.3 RE-ADOPTION may surface unresolved engineering challenges." No substrate code shipped W6-W12 touches Koopman directly; W13-14 is the first attempt. |
| Drift description | Plan v1.2.3 is internally inconsistent. §1 reframe says Koopman retained in v1; §3.4 + §4 say Koopman deferred to v2. v1.2.2 ADJUDICATION A1 (line 25) moved Koopman ops to v1 scope; the deferred-to-v2 language at lines 184/299/314 is from older v1.2.0/v1.2.1 text that was NOT updated when v1.2.2 A1 moved Koopman to v1. v1.2.3 carries both narratives. |
| Category | **(d)** commitment ambiguous in source documents |
| Pitch impact | Strategic — Goal 4 status (in v1 vs v2) determines pitch positioning. Two readings produce two different pitches. |
| Backfill path | Reconcile plan §3.4 + §4 lines 184/299/314 with §1 line 103 + v1.2.2 A1 adjudication. Update all v2-deferred Koopman language to v1.2.3 narrow-domain language. ~1 eng-day plan revision. Risk: changes commitment depth in pitch artifacts. |
| Reframe path | Pitch + plan choose ONE narrative: either Koopman is v1 narrow-domain (and lines 184/299/314 reframe to "the WIDE-DOMAIN runtime is v2; the narrow-domain runtime lands W13-14") or Koopman is v2 (and §1 line 103 + v1.2.2 A1 reverse). User decides which narrative is binding. ~1 eng-day. |
| Defer path | Carry the contradiction into W13-14 implementation. W13-14 Step 3 result determines empirically whether narrow-domain Koopman ships. Plan revision lands at W14 close based on actual fire-rate measurement. |

### D10 — LM head 7.43× speedup: provenance not located

| Field | Detail |
|-------|--------|
| Original commitment | User-supplied W13-14 task prompt: "Memory #2 goal 4: O(1) Koopman substitution — narrow domain v1, wide domain v2 research, LM head verified 7.43×." Treated as historical pitch claim. |
| Substrate reality | `WEEK_13_14_SCOPE_LOCK.md:120`: "Verification reference 'LM head 7.43× substitution speedup' was mentioned in the W13-14 task prompt but the specific provenance is not located in cipher-fusion-evidence or the cipher_rt_phase4 source tree (see honest residue §9). Step 2 + Step 3 measure the actual narrow-domain speedup against the v1.2.3 substrate; the '7.43×' framing is treated as informational, not a gate." |
| Drift description | Searched `cipher-fusion-evidence/`, `cipher_rt_phase4/`, `cipher_kmod/`. Single grep hit for "7.43" in `t4_6_3_dedup_report.md:41` is unrelated (conversation window stats). No measurement file produces 7.43×. |
| Category | **(d)** commitment not shipped at all (and possibly never measured — provenance failure) |
| Pitch impact | Strategic — if "LM head 7.43×" appears in any pitch artifact, it is an unsourced number. Plan line 6 explicitly says "Numbers cited in this document trace to a specific measurement file with file path and line citations." 7.43× has no such trace. |
| Backfill path | Locate the original measurement (may be in a deleted file, prior session memory, or external doc). If located, re-run on v1.2.3 substrate to re-verify. If not located, the number was never measured. ~0.5-2 eng-days depending on locate success. |
| Reframe path | Strike "LM head 7.43×" from pitch + memory anchor #2. Replace with the W13-14 Step 3 actual measurement number (whatever it is) at week-14-complete. ~0.5 eng-day doc. |
| Defer path | Treat as user memory or external provenance; do not block W13-14 implementation. Step 3 measures actual speedup; the 7.43× framing is informational unless user adjudication confirms a real measurement. |

### D11 — 24-hour soak commitment vs 30-min / 1h actuals

| Field | Detail |
|-------|--------|
| Original commitment | Plan line 1259 W15-17 CP 5.5: "**24-hour soak**." Plan line 1605 W11-12 (now W13-14): "24-hour soak (G4) at N=15 with learning tier ON." Plan line 1614 R-W11.2: "24-hour soak before declaring Weeks 11-12 done." |
| Substrate reality | W7-9 Step 5 ran 1-hour soak per user override (`WEEK_9_STEP_5_N128_SOAK.md:5`: "user override of the spec's 24-hour figure 2026-05-23"). W10-12 Steps 1/2/3 each ran 30-min soak. None of W7-12 ran the plan-mandated 24-hour soak. Plan line 359: "24-hour soak (G4) | NEVER RUN" (pre-W7 measurement of may13 audit). |
| Drift description | The 24-hour soak commitment in plan §7 W11-12 (now W13-14) and W15-17 CP 5.5 is not satisfied by any W7-12 soak. 1h soak at W7-9 Step 5 was a user override. 30-min soaks at W10-12 are regression smokes, not commitment-satisfying soaks. |
| Category | **(b)** substrate narrower than framing (soaks ran but at shorter durations than plan-committed) |
| Pitch impact | Strategic — "production-ready" framing typically requires 24h+ soak. Pitch artifacts that claim "production-ready substrate at week-12-complete" are unverified at 24h timescale. |
| Backfill path | Run a 24h synthetic soak at week-12-complete substrate. ~24h wall + ~1 eng-day setup. Closes the W11-12 R-W11.2 commitment. |
| Reframe path | Update plan §7 W7-9 / W10-12 row language from "soak" to "regression smoke" and "1-hour soak" to "synthetic 1-hour contention soak (full 24h is W15-17 CP 5.5)." Pitch updates from "production-validated" to "1h-soak-validated; production 24h soak is W15-17." ~1 eng-day. |
| Defer path | 24h soak is W15-17 CP 5.5 work per plan line 1259. Continue 30-min smoke regression at each step; 24h lands at CP 5.5. No new work. |

### D12 — Cryptographic billing receipts: billable-grade external check

| Field | Detail |
|-------|--------|
| Original commitment | Plan line 1638 W15-17 CP 5.5: "per-tenant cryptographic billing receipts (G6 chain head verified externally by re-computing the HMAC sequence from the kmod-published ring buffer)." Plan line 1740 R-G6.1: "neocloud per-tenant billing not production-grade until G6 ships." |
| Substrate reality | `WEEK_7_STEP_3_G6_AUDIT_CHAIN.md:152`: "All 15 cross-tenant chains reproduce. This is the cryptographic gate that protects R-G6.1: external auditor with read-only access to the mmap'd ring + per-tenant key can independently re-verify the entire chain... The seed pitch's 'per-tenant cryptographic billing receipts' claim is now defensible." |
| Drift description | W7-9 Step 3 internal verifier (`test_audit_chain`) reproduces the HMAC chain. The "external auditor" check (third-party re-verification using only the mmap'd ring + key) is asserted as defensible but not executed by an independent party. Plan line 1638 (W15-17) commits to external verification; this has not been performed. |
| Category | **(b)** substrate narrower than framing (substrate is cryptographically correct; the external-audit check is a manual exercise that has not been performed) |
| Pitch impact | Strategic — "per-tenant cryptographic billing receipts" is a Goal 1 + Goal 2 framing pillar for neocloud customers. Internal self-verification PASSES; external-party verification has not been performed. Pitch claim is "defensible" not "audited." |
| Backfill path | Hand the mmap'd ring + per-tenant key to a third party (or simulate by deleting all knowledge of the chain and re-verifying via only the exported ring + keys). ~0.5 eng-day to formalize the external-audit protocol + run it. |
| Reframe path | Pitch language: "cryptographically defensible per-tenant billing chain; third-party audit protocol formalized at W15-17 alongside CP 5.5 HMAC chain export." ~0.5 eng-day doc. |
| Defer path | Plan line 1638 explicitly schedules external HMAC chain export for W15-17. Continue with internal verifier; external audit lands at CP 5.5. No new work. |

## 4. Additional drift items found during walk

### D13 — May-13 reconstruction plan: deferred work item from W6

| Field | Detail |
|-------|--------|
| Original commitment | `WEEK_6_MAY13_RECONSTRUCTION_PLAN.md` (cipher-fusion-evidence 970694b): "~9-12 eng-days in 3 phases; critical path → W7-9." Plan line 1645 W15-17: "raw per-tenant per-second telemetry parquet + HMAC chain export for external audit" requires the May-13 reconstruction to land. |
| Substrate reality | No `WEEK_*MAY13*RESULT*.md` exists. W7-12 step docs do not reference May-13 reconstruction as completed. Memory anchor `may13-reconstruction-plan` (2026-05-23) marks the plan landing; no follow-up evidence indicates execution. |
| Drift description | The May-13 POC reconstruction plan landed as a planning document. Execution status is not visible in the substrate or step docs through week-12-complete. |
| Category | **(d)** commitment not shipped (status unclear; may be intentional W15-17 carry) |
| Pitch impact | Strategic — May-13 numbers (3.617× retracted, the 76% Track 2 density, the LM head 7.43×) are partially in the v1.2.3 substrate already (G3/G4/G5/Track 2). May-13 reconstruction was framed in W6 as a POC-replay validation step before the substrate work. |
| Backfill path | Execute the May-13 reconstruction at week-12-complete substrate. ~9-12 eng-days per the plan estimate. |
| Reframe path | Document May-13 reconstruction as merged into the W10-12 substrate work (G3/G4/G5/Track 2 composition validation IS the reconstruction). ~0.5 eng-day doc. |
| Defer path | Carry to W15-17 as part of CP 5.5 substrate validation. |

### D14 — vLLM TinyLlama smoke deferred at W12

| Field | Detail |
|-------|--------|
| Original commitment | `WEEK_10_12_SCOPE_LOCK.md` Step 3 E.6: "Run bench_llm.py B=1 100 tokens with new libcipher_rt + rebuilt kv_bridge + updated cipher_vllm_kv.py. Pass criteria: TPS within ±3% of W11 Step 2 baseline." |
| Substrate reality | `WEEK_12_STEP_3_G5_L2_PERSIST.md:151`: "vLLM TinyLlama smoke deferred. Step 3 changes are in the plugin and the L2 substrate; the cipher_test_commit_n128 30-min soak is the load-bearing gate and PASSED. Mistral env block carries forward from Step 4/5." |
| Drift description | W12 Step 3 explicitly skipped the vLLM TinyLlama TPS gate. Justification: Step 3 changes are plugin + L2 substrate primitive; primitive not yet called from plugin (per D3). |
| Category | **(d)** commitment not shipped at all (the scope-lock named the gate; it was not run) |
| Pitch impact | Minor — TinyLlama is the v1 substrate-validation model. W10 + W11 closeouts include TinyLlama smoke. W12 closeout does not. Pitch language "vLLM end-to-end smoke PASS at each W10-12 step" is true for Steps 1-2, asserted-as-deferred for Step 3. |
| Backfill path | Run `bench_llm.py B=1 100 tokens` at week-12-complete substrate. ~0.5 eng-day. |
| Reframe path | Document that W12 substrate is L2 primitive + plugin compute_va_gib; vLLM TPS gate at W13-14 Step 3 alongside REMEMBER consumer wire-up. ~0.5 eng-day doc. |
| Defer path | Carry to W13-14 Step 3 closeout. |

### D15 — CP 5.5 timeline slips W11-12 → W13-14 → W15-17

| Field | Detail |
|-------|--------|
| Original commitment | v1.2.1 plan: CP 5.5 at W4-5. v1.2.2 plan: CP 5.5 at W13-14. v1.2.3 plan (line 26): "CP 5.5 headline benchmark moves from W13-14 to **W15-17**." |
| Substrate reality | Plan line 1259 reflects W15-17 CP 5.5. v1.2.3 §1 line 5 documents the slip. |
| Drift description | CP 5.5 has slipped 3 times across plan revisions. Each slip absorbed architecture-gap closure work. v1.2.3's W15-17 commitment is the third revision. |
| Category | **(a)** framing precision only — each slip was documented; substrate goal unchanged |
| Pitch impact | Strategic — investor materials referenced earlier CP 5.5 timelines. Pitch language "CP 5.5 by Week N" needs to track plan revision. |
| Backfill path | Compress W13-14 + W15-17 work to hit CP 5.5 sooner. ~unknown eng-days; depends on which scope cuts are acceptable. |
| Reframe path | Pitch language anchors to "v1 substrate complete at week-12; production-shippable CP 5.5 at W15-17 (5-week measurement + tuning window)." ~0.5 eng-day. |
| Defer path | Hold W15-17 commitment; no new work. |

## 5. Cross-cutting observations

### 5.1 Drift patterns

Of 15 inventoried items:

- **Category (a) framing precision only:** 2 items (D4 Marlin arena framing, D15 CP 5.5 timeline). Substrate delivers original intent; doc language reframes.
- **Category (b) substrate narrower than framing:** 4 items (D1 G5 18× → 12.5×, D2 L2 per-stream vs per-tenant-tensor, D6 CFL untested at scale, D11 24h soak → 1h/30min, D12 cryptographic external audit). These tend to concentrate in: estimation-then-measurement drift (D1), and ambitious framing reduced to load-bearing primitive (D2).
- **Category (c) substrate broader than framing:** 0 items.
- **Category (d) commitment not shipped at all:** 8 items (D3 L2 call-site, D5 SDPA trampolines, D7 Mistral, D8 real-LLM, D9 Koopman ambiguity, D10 LM head provenance, D13 May-13 reconstruction, D14 W12 vLLM smoke). These concentrate in: deferred-to-W15-17 (D5, D7, D8, D12, D13), and ambiguous/contradictory plan language (D9, D10).

Most common drift pattern: **substrate primitive ships; consumer/caller wire-up defers**. D3 (L2), D5 (SDPA), D6 (CFL), D14 (vLLM smoke) all share this shape.

Second most common: **measurement deferred to W15-17 CP 5.5**. D7 (Mistral), D8 (real-LLM), D12 (external audit), D11 (24h soak) all share this shape.

### 5.2 Discipline gaps

Advisor catches surfaced 3 of 15 drift items at scope-lock or implementation time:
- D2 (L2 framing) — advisor catch #1 at W12 Step 3
- D4 (Marlin arena framing) — advisor catch #2 at W11 Step 2
- D5 (SDPA trampolines) — advisor catch #3 at W11 Step 2

12 of 15 were NOT advisor-flagged. These surfaced in the step doc's "honest residue" sections (D1 G5 reduction factor in W12 Step 3, D3 L2 call-site in W12 Step 3, D6 CFL in W11 + W12, D7 Mistral in W9 + W10 + W11 + W12, D8 real-LLM by design across all step docs) or in this drift audit (D9-D15).

Pattern: advisor catches drift at implementation time when it touches the C code path being written. Drift in the framing/timing/scope dimension (D9, D11, D15) and in unsourced numbers (D10) requires document-level analysis to surface. The honest-residue discipline catches the framing drift IN the step doc itself, but does not feed back into pitch + plan revisions.

### 5.3 Plan accuracy

Plan v1.2.3 §7 row commitments matched what shipped in 7 of 15 cases. Reframed during execution in 6 cases (D1, D2, D4, D11, D14, D15). Internally inconsistent in 2 cases (D9 Koopman, plus the implicit 24h-vs-1h tension in D11).

Plan internal-consistency hot spots:
- §1 reframe (lines 100-103) does not propagate to §3.4 + §4 lines 184/299/314 (D9 Koopman)
- W6 audit projections (437 / 152 GiB) carried forward as headline numbers despite Step 3 measuring 640 / 509 GiB (D1)

### 5.4 Pitch exposure

Of 15 items, the following touch the seed pitch headline claims:

| Pitch claim | Touched by items |
|-------------|------------------|
| 100 agents × 5 model families per H100 (Goal 1) | D5, D7, D8, D9, D11 |
| 2× tok/W (Goal 3) | D7, D8, D11 |
| 85% MFU (Goal 2) | (none) — already retracted per plan line 137-161 |
| O(1) Koopman (Goal 4) | D9, D10 |
| LD_PRELOAD-only deployment (Goal 5) | (none) — preserved through W7-12 |
| per-tenant cryptographic billing receipts | D12 |
| 18× VA reduction headline | D1 |

**10 of 15 items touch at least one pitch headline.** Most-touched pitch claim: Goal 1 (100 agents × 5 families), 5 items. Second: Goal 3 (TPW), 3 items. Goals 2 (MFU) and 5 (LD_PRELOAD) untouched by W6-W12 drift.

## 6. Adjudication summary

15 drift items inventoried. Per category:
- (a) framing precision only: 2
- (b) substrate narrower than framing: 5
- (c) substrate broader than framing: 0
- (d) commitment not shipped at all: 8

Items touching seed pitch headlines: **10 of 15**.

User adjudication is the next step before any W13-14 implementation commences. Per-item options surfaced in Part 3 + 4 (each item has backfill / reframe / defer paths with eng-day estimates). No remediation has been chosen by this audit.

Adjudication-review sequence ordering is not prescribed here; the inventory in Part 3 + 4 is ordered as the task prompt's enumeration (D1-D12) plus D13-D15 found during walk. The user may take items in any sequence.

## 7. Honest residue of the audit itself

1. **D10 LM head 7.43× provenance** could not be located. The number appears in the user's W13-14 task prompt as memory context. If it lives in a session-memory anchor not in `MEMORY.md`, or in an external doc, the audit could not surface it. Flagging for user attention.
2. **W15-17 CP 5.5 has not been written yet.** Drift items D7, D8, D11, D12 reframe to "W15-17 work" but W15-17 scope-lock does not exist. Audit cannot verify W15-17 commitments will absorb these items.
3. **Pitch artifacts not enumerated.** The audit walked the executive summary + plan + step docs. Other pitch materials (investor decks, NVIDIA ICC docs, Devang communication artifacts) were not walked. Pitch-impact assessments in Part 3 are limited to what plan + memory anchors reference.
4. **No remediation paths chosen.** Each item has three paths; user picks. The audit explicitly does not recommend.
5. **Eng-day estimates in backfill paths are rough.** Each is a sketch, not a binding scope estimate. Detailed scope-lock would refine.
