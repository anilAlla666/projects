# Plan §3.4 + §4 Koopman language reconciliation

**Date:** 2026-05-23
**Closes:** `WEEK_12_SCOPE_DRIFT_AUDIT.md` item D9 (plan-internal Koopman v1-vs-v2 contradiction)
**Scope:** documentation edit only. No source code changes. No scope-lock amendments.
**User adjudication 2026-05-23:** Koopman is in v1, narrow domain. Memory anchor #2 goal 4. All 5 v1 goals non-negotiable. §1 line 103 is authoritative. §3.4 + §4 deferred-to-v2 language updates to match.

## 1. Reconciliation rationale

`WEEK_12_SCOPE_DRIFT_AUDIT.md` item D9 surfaced that `CIPHER_REENGINEERING_PLAN.md` v1.2.3 contained self-contradictory Koopman scope statements:

- **§1 line 103**: "**O(1) Koopman compute substitution** — RETAINED in v1 per v1.2.2 A1 and v1.2.3 §7 re-sequence… v1 scope is the narrow-domain Koopman; wide-domain Koopman remains v2 research."
- **§3.4 + §4 various lines**: deferred-to-v2 language (originally v1.2.0 / v1.2.1 framing; not updated when v1.2.2 ADJUDICATION A1 moved Koopman to v1 scope).

v1.2.2 ADJUDICATION A1 (2026-05-23 sealed at v1.2.2 md5 `40722374f7a9b4b93d56381add208c92`) reactivated Koopman in v1 narrow-domain scope. The §1 reframe at v1.2.3 surfaced the new framing at the top of the plan; the prior deferred-to-v2 prose deeper in the document was not propagated forward. v1.2.3 carried both narratives.

User adjudication 2026-05-23 locks the resolution: §1 line 103 is authoritative; §3.4 + §4 update to match.

## 2. Pre-edit reference table

| Line | Section | Statement | Agreement with §1 line 103 |
|------|---------|-----------|----------------------------|
| 25 | v1.2.2 audit row | "Koopman tier in v1" | ✓ agrees |
| 35 | v1.2.2 audit row | KOOPMAN_SURROGATE_ADD ioctl scoped to v1/v1.5/v2 | ✓ agrees (the ioctl itself is v2; narrow-domain in v1 doesn't need this dynamic insertion ioctl) |
| 54 | v1.2.2 §1 A1 row | "moves the 6 Koopman learning ops … into v1 scope" | ✓ agrees |
| 58 | v1.2.2 A5 row | "5 Koopman ops moved from V2-SCOPE per A1" | ✓ agrees |
| 65 | v1.2.3 B.1 row | "Koopman tier in v1 (A1 NOT reversed)" | ✓ agrees |
| 75 | v1.2.3 B.1 row | "Koopman tier RETAINED in v1" | ✓ agrees |
| 77 | v1.2.3 §1 row | "Koopman O(1) substitution (retained in v1, held to W13-14)" | ✓ agrees |
| **103** | **§1 line 103** | **"RETAINED in v1 per v1.2.2 A1… narrow-domain Koopman; wide-domain Koopman remains v2 research"** | **authoritative** |
| 106 | §1 v1.2.2 note | "6 Koopman learning ops are IN v1 scope" | ✓ agrees |
| **197** | §3.4 code inventory | "**Koopman/EDMD learner (Goal-4 substrate — dead code, zero callers):**" | ⚠ current-state observation; reads as v2 vibe without context |
| **200** | §3.4 code inventory | "cipher_koopman_runtime.cpp — L1.1 Runtime Koopman derivation; dead code (audit Section 4)." | ⚠ current-state observation |
| **401** | §4 substrate analysis | "**This is the central engineering reality the unified runtime must address — and v1 deliberately defers fixing it (Goal-4 = v2).**" | ✗ **DIRECT CONTRADICTION** |
| 440 | §4 substrate analysis | "Koopman/EDMD lane unwired (no callers of cipher_rt_substitute_v2 equivalent)" | ⚠ current-state observation |
| 448 | §4 substrate analysis | "Goal-4's Koopman runtime module is dead code with zero callers in either tree." | ⚠ current-state observation |
| **516** | §4 nvcc build | "**Adding nvcc is required only if Koopman O(1) substitute kernels … are ported. For v1 — deferred (see Section 4).**" | ✗ **DIRECT CONTRADICTION** |
| 549 | §3.5 ioctl table | "29 CIPHER_KOOPMAN_SURROGATE_ADD … v2 / research" | ✓ agrees (wide-domain dynamic insertion is v2; narrow-domain v1 uses static seeding) |
| **583** | §4 nvcc table | "DEFERRED to v1.5 (Koopman ports)" | ✗ **DIRECT CONTRADICTION** (v1.5 vs v1) |
| **680** | §4 dispatch diagram | "└─ Koopman O(1) lane (v2 — not in v1)" | ✗ **DIRECT CONTRADICTION** |
| **830** | §4 classify table | "KOOPMAN_ELIGIBLE … **v2** (Koopman O(1) lane — deferred)" | ✗ **DIRECT CONTRADICTION** |
| 1258 | §7 W13-14 row | "Koopman tier integration … + G12 Koopman registry model-keying" | ✓ agrees |
| **1328** | §8 W1 risk register | "R-W1.2 [LOW]: nvcc not added — Koopman .cu files NOT ported this week (deferred Week 5 or v1.5)." | ✗ historical W1 reframe needed |
| 1607 | §8 W11-12 verification gate | "If no shape converges within 500-launch warmup, the v1.2.2 Koopman claim is retracted" | ✓ agrees (verification gate + fallback path; baseline scope is v1) |
| 1613 | §8 R-W11.1 | "an unconvergent Koopman tier is a documented retraction, not a silent regression" | ✓ agrees (fallback condition) |
| **1671** | §8 R-A5 risk row | "Goal-4 Koopman never wired in v1 … v2 deferral explicit in Section 4.5" | ✗ **DIRECT CONTRADICTION** (already-superseded v1.2.1 language) |
| **1677** | §8 R-T1 risk row | "The DEAD CODE stays in may13." | ✗ **DIRECT CONTRADICTION** |
| 1714 | §8 deferral inventory | "v1.2.2 scope expansion moves all engineering-deferred ops (the 6 Koopman tier + 4 v1.5 overlay ops) into v1" | ✓ agrees |
| **1745** | §8 gap | "**Goal-4 Koopman O(1) substitution is NEVER built into v1.** **Superseded by v1.2.2 ADJUDICATION 1.**" | ⚠ headline language contradictory at-a-glance despite the supersede note |
| 1799 | §9 architectural summary | "Tree A contains the brain (classifier + observers + Koopman dead code)" | ⚠ current-state observation |
| 1890 | v1.2.3 B.1 audit trail | "Goal 4 Koopman retained per v1.2.2 A1, NOT reversed" | ✓ agrees |
| 1894 | v1.2.3 B.1 audit trail | "Goal 4 = O(1) Koopman compute substitution (retained from v1.2.2)" | ✓ agrees |

**Totals before edit:** 30 references. 17 agree. 9 direct contradictions (lines 401, 516, 583, 680, 830, 1328, 1671, 1677, 1745). 4 current-state observations (lines 197, 200, 440, 448, 1799). Several rows annotated as ✓ agrees-with-context.

Under the 10-item HARD STOP threshold (9 disagreements).

## 3. Edits applied

### 3.1 New reconciliation paragraph at §1 line 108

Inserted directly after the v1.2.2 ADJUDICATION A1 comment block + before "v1 op surface: 30 of 33." text. Provides authoritative inline note that:
- v1.2.2 A1 reactivated Koopman in v1 narrow-domain
- §3.4 + §4 engineering analyses remain authoritative on substrate challenges
- v1 scope addresses challenges in narrow-domain form only at W13-14 per §7 line 1258
- Wide-domain Koopman remains v2 research
- Earlier prose flagged Koopman as v2-deferred (v1.2.0/v1.2.1 framing) and is superseded
- Closes WEEK_12_SCOPE_DRIFT_AUDIT.md item D9

### 3.2 Nine disagreement edits

| Line | Edit |
|------|------|
| 401 | "v1 deliberately defers fixing it (Goal-4 = v2)" → "audit-time substrate" framing + reference to v1.2.2 ADJUDICATION A1 + W13-14 scope-lock Step 2 narrow-domain seeding |
| 516 | "For v1 — deferred (see Section 4)" → "W13-14 scope-lock R-W13.2 plans nvcc rule addition" + host-only EDMD fallback noted |
| 583 | "DEFERRED to v1.5 (Koopman ports)" → "v1 scope per v1.2.2 ADJUDICATION A1 + §7 W13-14" + host-only fallback |
| 680 | ASCII art "(v2 — not in v1)" → "(narrow-domain v1 W13-14; wide-domain v2)" |
| 830 | table "v2 (Koopman O(1) lane — deferred)" → "narrow-domain v1 (W13-14 per v1.2.2 ADJUDICATION A1 and §7 line 1258; wide-domain dynamic discovery remains v2)" |
| 1328 | "R-W1.2 [LOW]: nvcc not added — Koopman .cu files NOT ported this week (deferred Week 5 or v1.5)" → "Per v1.2.2 ADJUDICATION A1, narrow-domain Koopman ports land at Weeks 13-14 (§7 line 1258); W13-14 scope-lock R-W13.2 plans the nvcc rule addition" |
| 1671 | R-A5 row entire content rewritten to mark supersede + new W13-14 disposition |
| 1677 | "The DEAD CODE stays in may13" → "The Koopman dead code in may13 ports at W13-14 per v1.2.2 ADJUDICATION A1 + §7 line 1258" |
| 1745 | "Goal-4 Koopman O(1) substitution is NEVER built into v1" headline qualified to "narrow-domain v1 W13-14 per v1.2.2 ADJUDICATION A1 + §1 line 103" |

### 3.3 Two current-state observations annotated

| Line | Edit |
|------|------|
| 197 | "Koopman/EDMD learner (Goal-4 substrate — dead code, zero callers):" → "(Goal-4 substrate — dead code at audit-time, zero callers; narrow-domain reactivation at Weeks 13-14 per v1.2.2 ADJUDICATION A1 + §7 line 1258):" |
| 448 | "Goal-4's Koopman runtime module is dead code with zero callers in either tree." → "at audit-time; narrow-domain reactivation at Weeks 13-14 per v1.2.2 ADJUDICATION A1 + §1 line 103." |

Lines 200, 440, 1799 left as-is — current-state code observations that describe the substrate accurately at audit time. The reconciliation paragraph at line 108 provides the forward-reference context. Preserving these as-is per "minimal edit surface" guidance.

## 4. Post-edit verification

Re-grep for `koopman.*v2\b`, `defer.*koopman`, `koopman.*defer`, `v2.*koopman`, `koopman.*dead code`, `koopman.*not in v1`, `NEVER built into v1` returns 21 hits (down from 27 pre-edit). Walked:

- All v1.2.2/v1.2.3 audit-trail rows (lines 25, 35, 54, 58, 65, 75, 77, 1890, 1894) — historical v2 references describe NCCL ops, not Koopman, OR describe the supersede event itself. **Agree.**
- Line 549 KOOPMAN_SURROGATE_ADD ioctl scoped to v2/research — wide-domain dynamic insertion ioctl is genuinely v2; narrow-domain v1 uses static seeding. **Agree.**
- Lines 1607, 1613, 1615 — verification gate + fallback condition (if validation fails, retract to V2-SCOPE). Plan-honest fallback path; baseline scope is v1 narrow-domain. **Agree.**
- Lines 1538, 1745 (post-edit) — MODEL_UNKNOWN sentinel pass-through + the headline now-qualified with supersede note. **Agree.**
- Line 1799 architectural summary — current-state observation pre-reconciliation; context provided by line 108 paragraph. **Acceptable per minimal-edit-surface.**

**Total post-edit disagreements: 0.** 100% agreement with §1 line 103.

## 5. v1.2.3 §7 W13-14 row + WEEK_13_14_SCOPE_LOCK.md remain valid

§7 line 1258 W13-14 row: "Koopman tier integration (EDMD pipeline real-input fix + recipe registry narrow-domain seeding + SUBSTITUTE-Koopman lane validation) + G12 Koopman registry model-keying." Unchanged. Consistent with reconciled language.

`WEEK_13_14_SCOPE_LOCK.md` R-W13.3 (HIGH likelihood, MAJOR impact) flagged the original v1-vs-v2 contradiction at scope-lock time. With the reconciliation landed, R-W13.3 is reduced from "plan-internal contradiction risk" to "engineering risk only" — the engineering challenges that motivated the original deferral (registry never matches real workloads, Koopman runtime dead code) remain HIGH likelihood, MAJOR impact on substrate work. The scope-defining contradiction is resolved.

R-W13.3 text in `WEEK_13_14_SCOPE_LOCK.md` line 162 reads: "Goal 4 was explicitly DEFERRED by may13 audit (plan line 401: 'v1 deliberately defers fixing it'); the v1.2.3 RE-ADOPTION may surface unresolved engineering challenges." Plan line 401 has been reframed; the engineering-challenge half of R-W13.3 stands. Scope-lock remains valid; no amendment required.

## 6. Plan md5 pre/post

```
Pre-edit:  18b12c5346986309540ac246cfccefaf  CIPHER_REENGINEERING_PLAN.md (1907 lines)
Post-edit: c2eddd0a780bed9a327ab05b42d71594  CIPHER_REENGINEERING_PLAN.md (1909 lines, +2 net)
```

Total lines touched: 11 (1 new paragraph at line 108 + 9 disagreement fixes + 2 current-state-observation annotations at lines 197 and 448).

## 7. Closes WEEK_12_SCOPE_DRIFT_AUDIT.md item D9

D9 categorized as **(d) commitment ambiguous in source documents** with HIGH pitch impact (Goal 4 status determines pitch positioning). Adjudication selected the "reframe" path from D9 §6.3 ("Pitch + plan choose ONE narrative") with user-bound narrative: Koopman v1 narrow-domain, all 5 v1 goals non-negotiable.

D9 status: **CLOSED** at plan md5 `c2eddd0a780bed9a327ab05b42d71594`.

Remaining 14 scope-drift audit items (D1-D8, D10-D15) unchanged. No remediation chosen for those; user adjudication per-item still pending.
