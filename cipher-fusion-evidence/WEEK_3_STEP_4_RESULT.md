# Week 3 Step 4 — `CIPHER_DISPATCH_LIVE=1` Flip + SENSE→DSM PROPOSE — RESULT

**Status: PARTIAL — STOPPED at sub-step 4.B. Shape-3 structural gap surfaced. Awaiting adjudication.**

Sub-step 4.A baseline verified (Mistral-7B vanilla SC6 7/7 PASS in 57s; pod ready for LIVE flip gate). Sub-step 4.B inspection revealed SENSE actuator and DSM PROPOSE ioctl are NOT in the state Wave 5 §5.5 W3 / scope-lock §2 Step 4 assumed. Per brief §4.D Shape 3 discipline ("do NOT proceed; surface for adjudication; possibly defer DSM PROPOSE to Week 4 and ship Step 4 with LIVE flip only") — STOP and surface.

No edits made post-Step-3. No commits. cipher_rt_phase4 HEAD remains `4f1a86ab` (Step 3 close). cipher_kmod HEAD remains `0ce4b8e2` (week-2-complete).

**Date:** 2026-05-20
**Pre-conditions verified:** all anchors match.

---

## 4.A — Pre-edit baseline + Mistral-7B reference — PASS

| signal | value |
| --- | --- |
| cipher_rt_phase4 HEAD | `4f1a86ab` ✓ (post-Step-3) |
| cipher_kmod HEAD | `0ce4b8e2` ✓ (week-2-complete) |
| `step4_pre` snapshots preserved | `libcipher_rt.so` + `cipher_kmod.ko` at `/tmp/week3_step4/` |
| **SC6 Mistral-7B vanilla baseline** | **PASS in 57s; 7/7 sub-checks; all4_fwd1+fwd2 bit_identical=true** |

Mistral-7B baseline reference is established. Pod is ready for the LIVE flip gate.

---

## 4.B — SENSE → DSM PROPOSE inspection — STRUCTURAL GAP (Shape 3-adjacent)

### B.1 SENSE actuator status

| signal | finding |
| --- | --- |
| `cipher_rt_sense*.c` (cipher_rt_phase4 native) | **DOES NOT EXIST** (no `cipher_rt_sense_*.c` file) |
| `src/may13/cipher_sense.cpp` (ported in Week 2 Step 2) | EXISTS; provides `cipher_sense_init()` (L154), `cipher_sense_observe(CipherRingEntry*)` (L171), `cipher_sense_report()` (L296) |
| Native callers of `cipher_sense_init` / `cipher_sense_observe` | **ZERO** (grep clean across cipher_rt_phase4/*.c/*.cpp excluding src/may13/) |

SENSE actuator's source is ported and linked into libcipher_rt.so as an inert TU (per Week 2 Step 2 "no may13 code reached by existing cipher_rt_phase4 dispatch paths"). It is NOT wired to cipher_rt_phase4's event flow. The wire-up would require:

1. Bridge layer: CUPTI launch events → may13's `CipherRingEntry` format (the may13 ring is a brain-side event log; cipher_rt_phase4 doesn't have one).
2. Per-launch call to `cipher_sense_observe(&entry)` from the CUPTI callback or via the classify observer.
3. Wait for SENSE's internal threshold to trigger a phase transition (its decide-output).
4. Pull SENSE's proposal via a new accessor function (not yet exposed by `cipher_sense.cpp`).

### B.2 DSM PROPOSE ioctl status

| signal | finding |
| --- | --- |
| `CIPHER_DSM_PROPOSE` or `PROPOSE` in `cipher_kmod/cipher_ioctl.h` | **DOES NOT EXIST** |
| Existing CP 5.4 migration ioctls | `CIPHER_CP54_SUBSCRIBE_MIGRATE` (nr 16), `_POLL_MIGRATE` (nr 17), `_START_MIGRATE` (nr 18), `_ACK_MIGRATE` (nr 19), `_COMPACT_MIGRATE` (nr 20). Migration state enum: `CIPHER_CP54_MIG_IDLE`, `_PROPOSED`, `_MIGRATING` (states, not actions) |
| Next-free ioctl nr | 26 (per Week 2 Step 6 pre-flight) |

A new `CIPHER_DSM_PROPOSE` ioctl (or fold-into-nr-20 extension) would need to be designed, implemented kmod-side with full unrolled-cleanup dispatch + handler, and wired to the existing CP 5.4 migration pipeline.

### B.3 Shape classification per brief §4.D

Brief defines three shapes:

- **Shape 1**: "both SENSE and DSM PROPOSE exist; just wire them" → not the case (DSM PROPOSE doesn't exist; SENSE not wired)
- **Shape 2**: "SENSE exists but DSM PROPOSE ioctl is new" → ambiguous if "SENSE exists" means "ported source exists" (true) vs "SENSE is integrated and actionable" (false)
- **Shape 3**: "neither exists yet" → effectively true if "exists" means "actionable"

**Verdict: Shape-3-adjacent.** SENSE's source is on disk but it has no events, no decisions, no output accessible from cipher_rt_phase4 native paths. DSM PROPOSE ioctl doesn't exist.

### B.4 Scope estimate for full Step-4-as-scoped

| sub-deliverable | estimate |
| --- | --- |
| 1. `CIPHER_DISPATCH_LIVE=1` flip (1-line default change + banner update) | 15 min |
| 2. CUPTI → may13-ring bridge | 2-3h |
| 3. cipher_sense_init() call at libcipher_rt init body | 30 min |
| 4. Per-launch cipher_sense_observe() wiring | 1-2h |
| 5. New SENSE accessor for "proposal-pending?" | 1h |
| 6. CIPHER_DSM_PROPOSE ioctl (nr 26 + struct + define) | 1h |
| 7. Kmod handler + dispatch case + unrolled cleanup | 2h |
| 8. Userspace push: cipher_sense decision → ioctl | 1h |
| 9. Step 4 SC6 TinyLlama + Mistral gates | ~10 min |
| 10. Step 4 CP 5.4 + commit/tag (2 trees) | 30 min |
| **total** | **9-12h** |

vs WEEK_3_SCOPE_LOCK.md §2 Step 4 budget of **4-6h**. Step 4 as scoped is roughly **2× over budget** if SENSE→DSM PROPOSE wiring is included.

---

## 4.C — 4.H — NOT EXECUTED

Per brief §4.D Shape 3 discipline: "do NOT proceed; surface for adjudication." No flip applied, no SENSE wiring, no kmod ioctl, no build, no SC6 LIVE-flip run, no commits, no tags.

---

## Adjudication options surfaced (no recommendation per discipline)

Three resolution paths:

### Option I — Ship Step 4 with LIVE flip ONLY; defer SENSE → DSM PROPOSE to Week 4

**Scope reduction:** drop deliverable items 2-8 above; keep only item 1 (LIVE flip) + item 9 (SC6 gate) + item 10 (commit/tag).

- ~1h of work total.
- Step 4 still ships the load-bearing change (classifier-driven routing becomes live by default).
- Mistral-7B SC6 PASS confirms bit-identity under LIVE=1 — the binding correctness gate of Step 4.
- SENSE → DSM PROPOSE wiring moves to Week 4 (which already includes the observability tier port + many other SENSE-adjacent files like cipher_loop, cipher_pipeline, cipher_pulse).
- v1.2.2 §7 Week 3 ("SENSE phase transitions trigger DSM PROPOSE for tool-idle detection") becomes partially-shipped; Week 4 closes the gap.

**Risk:** Goal 2 (100-tenant via cross-tenant SM migration) is not advanced in Week 3 the way the spec named. Mitigated: the migration ioctls (nr 16-20) are already shipped from CP 5.4; SENSE → DSM PROPOSE is the missing producer, not the consumer.

### Option II — Expand Step 4 budget to ~10-12h and ship full SENSE→DSM PROPOSE

Implements all 8 sub-deliverables above. Adds risk surfaces (new bridge layer, new ioctl ABI, new kmod handler). Mistral-7B SC6 PASS may also need to verify SENSE doesn't perturb forward pass (it shouldn't, since it's observe-and-propose, but worth a confirm).

**Risk:** larger scope = more failure surfaces; ~2× over the scope-lock budget; may slip into Week 4.

### Option III — Defer Step 4 entirely; tag Step 3 as week-3-complete with partial scope

Argues that classify_substrate (Steps 1-3) is the substantive Week 3 deliverable, and LIVE flip without SENSE wiring is incremental at best. Tag `week-3-complete` at Step 3 state; document the LIVE flip as Week 4 entry work.

**Risk:** v1.2.2 §7 Week 3 explicitly names "classifier-driven routing live" as the goal; not flipping LIVE leaves the substrate fully observe-only at Week 3 close — substantive feature shipped but not engaged.

---

## Failure-mode-of-record

Step 4 cannot proceed cleanly as scoped because the scope-lock §2 Step 4 budget (4-6h) was based on Wave 5 §5.5 W3's SENSE → DSM PROPOSE language without verifying on-disk readiness. Pre-flight verified the named symbols (`cipher_cp54_sched.c::COMPACT_MIGRATE` nr 20 + ratelimit) but did not verify that SENSE was integrated end-to-end, nor that DSM PROPOSE existed as a separate ioctl distinct from COMPACT_MIGRATE.

This is the same class as the Week 2 Step 2 audit-foundation gap: the named target exists at the cited line, but the structural surrounding wire-up does not. The right audit (would-have-been) is "verify the producer and consumer of the named integration both have their wire-up endpoints in place."

The fix going forward: any "SENSE → X" or "X → ioctl" prescription in Wave 5 should include a pre-flight check on both the producer's actionable-output exposure and the consumer's ioctl/symbol existence.

---

## Anchors at stop point

| tree | HEAD | tag |
| --- | --- | --- |
| cipher_rt_phase4 | `4f1a86ab` | `week-3-step-3-marlin-hint-volt-lock` |
| cipher_kmod | `0ce4b8e2` | `week-2-complete` |
| cipher-may13-evidence | `fc8a9ae6` | `week-2-complete` |

No edits made in Step 4. Working trees clean across all three.

---

## Telemetry on disk

- `/tmp/week3_step4/libcipher_rt.so.step4_pre` — snapshot at Step-3-close (md5 `f63e1f2e`)
- `/tmp/week3_step4/cipher_kmod.ko.step4_pre` — kmod snapshot
- `/tmp/week3_step4/sc6_pre_baseline_mistral.log` — Mistral-7B vanilla SC6 7/7 PASS in 57s

---

## Discipline notes

- Read-only at sub-step 4.B; STOP-and-surface invoked per brief §4.D Shape 3 framing.
- No code changes. No commits. No tags.
- Three options surfaced; no recommendation per discipline.
- Step 5 closeout does NOT run (Step 4 PARTIAL gates Step 5).

---

## Awaiting

User adjudication on which Option (I / II / III) to take. After adjudication:

- **Option I**: I draft a tight Step-4-LIVE-flip-only brief; ~1h to PASS or FAIL on Mistral-7B SC6.
- **Option II**: I draft a Step-4-full brief with explicit budget overrun acceptance; ~10-12h with 8 sub-deliverables.
- **Option III**: I draft a Week 3 closeout tagging `week-3-complete` at Step 3 state; document the LIVE flip + SENSE→DSM as Week 4 entry.
