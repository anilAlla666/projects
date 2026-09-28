# W12 Step 9: audit closure batch — D7, D8, D11, D12, D13, D15

**Date:** 2026-05-23
**Scope-drift audit closures:** D7, D8, D11, D12, D13, D15 of 15
**Cumulative closures:** all 15 of 15
**Open audit items:** none

## 1. Pre-condition verification

| Component | Tag / Anchor | md5 |
|-----------|--------------|-----|
| `cipher_rt_phase4` | `week-12-step-6-d3-l2-wireup` (`adbe121`) | `libcipher_rt.so` `33412ffc97b714662e0824e1de8362bd` |
| `cipher_kmod` | `week-9-complete` (0.6.5) | `cipher_kmod.ko` `8c9fdd016897436ceff382c4e9178e07` |
| `cipher_vllm_plugin` | (untagged tree) | `cipher_vllm_kv.py` `2b6cedab89387c30becd49a27313ceb4` |
| `cipher-fusion-evidence` | `49586db` | `WEEK_12_STEP_8_D1_D2_D4_D10_REFRAMES.md` present |
| `WEEK_12_SCOPE_DRIFT_AUDIT.md` | present | |

## 2. Closure rationale

`WEEK_12_SCOPE_DRIFT_AUDIT.md` inventoried 15 items. After W12 Steps 4-8
nine items closed (D5, D9, D14, D3, D6, D1, D2, D4, D10). Six remain:
**five are CP-5.5-deferred by design**; one (D13) needs status check on
work the W6 plan flagged as parallel to W7-9.

CP-5.5 deferral is not drift. The v1.2.3 plan §7 row at line 1261
schedules these items explicitly at **W15-17**:

> CP 5.5 hybrid headline benchmark on full unified runtime — workload-class heterogeneity (5 prefill + 80 decode + 15 burst) **+ model-architecture heterogeneity (≥5 different model families coexisting: Mistral-7B + Qwen-7B + Llama-3-8B + 2 SLMs)** at the highest density the closed gaps permit. **24-hour soak.** Per-tenant FAIRNESS + CARBON + **RECEIPT cryptographic billing chain verified.**

This step's role is documentary acknowledgment that the audit items
align with the planned W15-17 scope, not new engineering work.

## 3. Per-item closure summary

### 3.1 D7 — Mistral E.7 DeepGEMM environmental block → CP 5.5

**Original audit framing** (`WEEK_12_SCOPE_DRIFT_AUDIT.md:118-129`):
DeepGEMM env block carries from W7-9 Step 4 through W12 Step 3 across 5
step docs. Mistral graph-capture segfaults at baseline (no CIPHER
injection). TinyLlama is the substrate gate; Mistral measurement remains
unrun.

**CP-5.5 absorption:** Plan §7 W15-17 row at line 1261 names **Mistral-7B**
as the canonical large-model in the 5-family heterogeneous mix. The
DeepGEMM environment fix lands as a W15-17 pre-requisite alongside the
real-LLM measurement infrastructure stand-up. Per the audit's defer path
(line 129): "Carry to W15-17 CP 5.5 where Mistral measurement is the
headline. Continue using TinyLlama as substrate gate."

**Why this is not drift:** the W7-12 scope-locks **explicitly** named
TinyLlama as the substrate gate. Mistral measurement was never a W7-12
deliverable; the env block is a W15-17 pre-condition.

**Status: CLOSED as W15-17 pre-requisite.** Substrate gate (TinyLlama)
satisfied at every step closeout. CP 5.5 carries the Mistral fix in its
work plan.

### 3.2 D8 — Real-LLM heterogeneous validation → CP 5.5 IS this

**Original audit framing** (`WEEK_12_SCOPE_DRIFT_AUDIT.md:131-142`):
"All W7-12 gates are synthetic. N=128 soak is single-process 128 thread
pairs. No real-LLM heterogeneous workload runs. The seed pitch claim
'5 model families coexisting' rests entirely on W15-17 measurement."

**CP-5.5 absorption:** Plan §7 W15-17 row at line 1261 is **defined as**
the real-LLM heterogeneous benchmark. Audit defer path (line 142): "Carry
to W15-17 CP 5.5 as designed; this is the explicit plan. No new work."

**Why this is not drift:** the audit category (d) classification
("commitment not shipped at all in W7-12") explicitly notes "explicit by
scope-lock design." The W7-12 scope locks committed to **substrate
gates**, not workload gates. Real-LLM heterogeneous validation is W15-17
by construction.

**Pitch language guard:** any pitch artifact that implies real-LLM
heterogeneous validation has shipped is wrong. The honest framing per
the audit reframe path (line 141): "substrate-validated; real-LLM
heterogeneous benchmark is the W15-17 CP 5.5 deliverable."

**Status: CLOSED as CP 5.5 by design.**

### 3.3 D11 — 24-hour soak → CP 5.5 W15-17

**Original audit framing** (`WEEK_12_SCOPE_DRIFT_AUDIT.md:171-182`):
W7-9 Step 5 ran 1-hour soak per user override of the 24-hour spec. W10-12
Steps 1/2/3 each ran 30-min regression smokes. None of W7-12 ran the
plan-mandated 24-hour soak.

**CP-5.5 absorption:** Plan §7 W15-17 row at line 1261 explicitly names
"24-hour soak." Plan line 1259 commits CP 5.5 to "24-hour soak (G4) at
N=15 with learning tier ON." Audit defer path (line 182): "24h soak is
W15-17 CP 5.5 work per plan line 1259. Continue 30-min smoke regression
at each step; 24h lands at CP 5.5."

**Why this is not drift:** the 30-min and 1-hour soaks W7-12 ran were
**regression smokes against substrate properties** (atomicity, coherence,
fairness), not production-scale validation. The 24-hour soak is a
production-readiness gate that lands once the full substrate is built
and the real-LLM heterogeneous workloads run end-to-end.

**Pitch language guard:** "production-ready substrate at week-12-complete"
is not satisfied by W7-12 soaks. Honest framing: "1h-soak-validated
substrate; production 24h soak is W15-17 CP 5.5."

**Status: CLOSED as CP 5.5 by design.**

### 3.4 D12 — Cryptographic billing receipts billable-grade → CP 5.5

**Original audit framing** (`WEEK_12_SCOPE_DRIFT_AUDIT.md:184-195`):
W7-9 Step 3 internal verifier (`test_audit_chain`) reproduces the HMAC
chain across 15 tenants × 1000 records. The "external auditor" check
(third-party re-verification using only the mmap'd ring + per-tenant
key) is asserted as defensible but not executed by an independent party.

**CP-5.5 absorption:** Plan §7 W15-17 row at line 1261 names "Per-tenant
FAIRNESS + CARBON + **RECEIPT cryptographic billing chain verified.**"
Audit defer path (line 195): "Plan line 1638 explicitly schedules
external HMAC chain export for W15-17. Continue with internal verifier;
external audit lands at CP 5.5. No new work."

**Why this is not drift:** the substrate is cryptographically correct
(W7-9 Step 3 ships the HMAC chain + offline reproduction gate per memory
`g6-audit-chain`). The unshipped piece is the **external-audit protocol
execution**, which is a manual exercise distinct from the substrate gate.
Plan line 1638 schedules it at W15-17.

**Pitch language guard:** "cryptographically defensible" (current
state) versus "third-party audited" (CP 5.5 deliverable). The honest
framing per the audit reframe path (line 194): "cryptographically
defensible per-tenant billing chain; third-party audit protocol
formalized at W15-17 alongside CP 5.5 HMAC chain export."

**Status: CLOSED as CP 5.5 by design.**

### 3.5 D15 — CP 5.5 timeline slips W11-12 → W13-14 → W15-17

**Original audit framing** (`WEEK_12_SCOPE_DRIFT_AUDIT.md:225-236`):
"CP 5.5 has slipped 3 times across plan revisions. Each slip absorbed
architecture-gap closure work. v1.2.3's W15-17 commitment is the third
revision."

**Plan state at this step:** v1.2.3 §7 row at line 1261 holds W15-17.
Line 26 documents the slip ("CP 5.5 headline benchmark moves from W13-14
to **W15-17**"). Line 1240: "B.1 decision (2026-05-23): re-sequence not
redesign... Extend timeline 2-3 weeks. CP 5.5 reframes as hybrid
workload-class + heterogeneous-model."

**Why this is not drift:** the audit category (a) classification
("framing precision only — each slip was documented; substrate goal
unchanged") confirms this is documentation-already-done. Each timeline
revision was an explicit plan amendment with a recorded rationale.

**Pitch language guard:** any pitch artifact citing "CP 5.5 by Week N"
must use the current v1.2.3 W15-17 number, not earlier-revision
numbers. The reframe path is in pitch coordination, not plan changes.

**Status: CLOSED as documentation already-done.** Per audit defer path
(line 236): "Hold W15-17 commitment; no new work."

### 3.6 D13 — May-13 reconstruction status

**Memory baseline** (`memory/may13-reconstruction-plan`): "Phase 1
(3-4 days clean ports + bench_llm_multi.py extension) reproduces P1, P2,
P3, P5, P7 within ±10%. Phase 2 (5-7 days observability hot-path)
composes with W7-9 G6 AUDIT chain. Phase 3 (1 day) MPS comparison.
Phase 4 (0 days) P4 15→1 GPU is W15-17 CP 5.5 corollary. Total 9-12
eng-days parallel to W7-9."

**B.2 investigation findings:**

1. **WEEK_6_MAY13_RECONSTRUCTION_PLAN.md** exists at the W6 commit
   (`970694b`) — the *plan* landed. 23,042 bytes / 222 lines.
2. **No WEEK_*MAY13*RESULT*.md or follow-up evidence doc exists.**
   `ls WEEK_*MAY13*` returns only the W6 plan.
3. **W7-12 step docs do not reference May-13 reconstruction as
   completed.** The only references to "May-13 reconstruction" outside
   the plan and the audit are in W6 transition notes
   (`WEEK_6_PLAN_V1_2_3_TRANSITION.md`, `WEEK_7_9_SCOPE_LOCK.md`,
   `WEEK_6_STEP_G1_G2_CAP_BUMP.md`) recording the plan landing.
4. **`bench_llm_multi.py` extension referenced in the plan as a Phase 1
   deliverable was not located** in cipher-fusion-evidence beyond the
   plan doc itself (`grep -rn bench_llm_multi --include='*.md'` returns
   only the plan).
5. **`src/may13/` actuators** (`cipher_sense.cpp`, `cipher_comply.cpp`,
   `cipher_pulse.cpp`, `cipher_dispatch.cpp`, `cipher_oracle.cpp`,
   etc.) exist on disk in `cipher_rt_phase4` — these are pre-existing
   Phase-3-era substrate ports, NOT Phase 1-3 reconstruction
   deliverables. They predate the W6 plan landing.
6. **Git log** for `cipher-fusion-evidence` between `970694b` (May-13
   plan landing) and HEAD `49586db` shows zero commits referencing
   May-13 reconstruction execution. All commits in that window are
   W7-W12 substrate steps + closures.

**Honest finding:** Phase 1-3 reconstruction work **was not executed**
during W7-W12. The substrate work dominated the calendar.

**Partial-absorption note:** the W7-9 G6 AUDIT chain at memory
`g6-audit-chain` does absorb RECEIPT + CARBON observability wiring per
the May-13 plan's Phase 2 finding: "W7-9 G6 kmod-resident AUDIT chain
absorbs RECEIPT + CARBON wiring for free." So a sub-component of Phase
2 landed as a side-effect of W7-9 Step 3, but as G6 substrate work, not
as a May-13 reconstruction deliverable.

**Phase 4 (P4 15→1 GPU) is W15-17 CP 5.5 corollary by plan design**
(May-13 plan's own scoping at the W6 anchor).

**Closure path: option (c) reconstruction-not-started; absorbed into
W15-17 CP 5.5.**

The audit's defer path at `WEEK_12_SCOPE_DRIFT_AUDIT.md:210` reads:
"Carry to W15-17 as part of CP 5.5 substrate validation." This step
takes that path. CP 5.5 is the production-scale validation that the
May-13 reconstruction was a pre-validation step for; the work absorbs
into the W15-17 hybrid heterogeneous benchmark scope.

**Status: CLOSED as deferred-to-CP-5.5.**

**Honest residue:**
- Phase 1 (clean ports + bench_llm_multi.py extension): NOT EXECUTED;
  W15-17 carry
- Phase 2 (observability hot-path): PARTIALLY ABSORBED via W7-9 G6
  AUDIT chain (RECEIPT + CARBON wiring landed); the full hot-path
  composition with the May-13 numbers is W15-17 carry
- Phase 3 (MPS comparison): NOT EXECUTED; W15-17 carry
- Phase 4 (P4 15→1 GPU): W15-17 CP 5.5 corollary by plan design

**Pitch language guard:** any pitch artifact citing the May-13 7-problems
numbers (P1/P2/P3/P5/P7) as "reproduced on v1.2.3 substrate" is wrong
at this step's closeout. The honest framing: "May-13 POC reconstruction
is a W15-17 CP 5.5 deliverable; v1.2.3 substrate (W7-12) composes the
substrate primitives that the reconstruction will compose." Memory
`may13-reconstruction-plan` records the key load-bearing finding: "the
7-problems headline numbers are produced primarily by load-distribution
and measurement methodology, NOT by individual-actuator contributions."
This stays the canonical caveat for pitch use of the May-13 numbers
until W15-17.

## 4. Total cumulative audit closures: 15 of 15

| # | Item | Closure step | Mode |
|---|------|--------------|------|
| 1 | D5 — SDPA trampoline stream-fill | W12 Step 4 (`1466193`) | substrate fix |
| 2 | D9 — Plan Koopman ambiguity | W12 Step 5 prelude (`69203d3`) | plan reconciliation |
| 3 | D14 — TPS gate + env-gate | W12 Step 5 (`b360fc1`) | substrate addition |
| 4 | D3 — L2 call-site wireup | W12 Step 6 (`adbe121`) | substrate + plugin |
| 5 | D6 — CFL throttle telemetry | W12 Step 7 (`2cbe7c6`) | additive measurement |
| 6 | D1 — G5 18×→12.5× reframe | W12 Step 8 (`49586db`) | documentation |
| 7 | D2 — L2 per-stream reframe | W12 Step 8 (`49586db`) | documentation |
| 8 | D4 — Marlin map re-key reframe | W12 Step 8 (`49586db`) | documentation |
| 9 | D10 — LM head 7.43× de-gating | W12 Step 8 (`49586db`) | documentation |
| 10 | **D7 — Mistral env → CP 5.5** | **W12 Step 9 (this step)** | **CP-5.5 deferred** |
| 11 | **D8 — real-LLM hetero → CP 5.5** | **W12 Step 9 (this step)** | **CP-5.5 deferred** |
| 12 | **D11 — 24h soak → CP 5.5** | **W12 Step 9 (this step)** | **CP-5.5 deferred** |
| 13 | **D12 — crypto external audit → CP 5.5** | **W12 Step 9 (this step)** | **CP-5.5 deferred** |
| 14 | **D13 — May-13 reconstruction → CP 5.5** | **W12 Step 9 (this step)** | **CP-5.5 deferred** |
| 15 | **D15 — CP 5.5 timeline framing** | **W12 Step 9 (this step)** | **documentation** |

**All 15 audit items closed.**

## 5. Substrate unchanged confirmation

Post-closure verification:

| File | Pre-step md5 | Post-step md5 | Match |
|------|--------------|---------------|-------|
| `libcipher_rt.so` | `33412ffc...` | (verified at step close) | YES |
| `cipher_kmod.ko` | `8c9fdd01...` | (verified at step close) | YES |
| `cipher_vllm_kv.py` | `2b6cedab...` | (verified at step close) | YES |

Git working trees in `cipher_rt_phase4` and `cipher_kmod` clean.

## 6. v1.2.3 §7 W13-14 ready to commence

With all 15 audit items closed on a clean substrate at
`week-12-step-6-d3-l2-wireup` (`adbe121`) for `cipher_rt_phase4` and
`week-9-complete` for `cipher_kmod`, the next critical-path step is
W13-14 (Koopman tier + G12 model-keying) per `WEEK_13_14_SCOPE_LOCK.md`.

Scope-lock at memory `w13-14-scope-lock`: "3-step sequence G12 model-
keying / Koopman tier (EDMD + recipe seeding + SUBSTITUTE lane) /
REMEMBER consumer + LM-head validation. 13 eng-days. No kmod ABI change
expected. Tag chain `week-13-step-1-g12-koopman-keying` →
`week-14-complete`."

The W7-12 audit closures resolve every framing-precision item that
might block W13-14 from committing to the v1.2.3 plan. The 6 W15-17
deferrals are scoped by plan §7 line 1261 already.

## 7. Files

- `cipher-fusion-evidence/WEEK_12_STEP_9_AUDIT_CLOSURE_BATCH.md` (this doc)
- `memory/audit-closure-batch.md` (new closure anchor)
- `memory/MEMORY.md` (new index entry)

## 8. Anchors

| Tree | Anchor | Status |
|------|--------|--------|
| `cipher_rt_phase4` | `week-12-step-6-d3-l2-wireup` (`adbe121`) | UNCHANGED |
| `cipher_kmod` | `week-9-complete` (0.6.5) | UNCHANGED |
| `cipher_vllm_plugin/cipher_vllm_kv.py` | `2b6cedab89387c30becd49a27313ceb4` | UNCHANGED |
| `cipher-fusion-evidence` | new HEAD with this file + memory anchor | doc-only commit |

No tag added (documentation-only step, no substrate move).
