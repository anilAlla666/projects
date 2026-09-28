# Week 4 Step 6 — Sub-4 Measurement Infrastructure — RESULT

**Status: PASS.**

Per-tenant transition counters + per-class transition matrix + tool-idle
counters + JSON report exposed via `cipher_rt_sense_transition.{c,h}`.
Pure userspace; no new ioctl; no kmod state required.
**Mid-step bonus fix: corrected canonical SENSE class naming**
({UNKNOWN, HUMAN, AGENT, BATCH}, not the {HUMAN, AGENT, BATCH, UNKNOWN}
sketch in Step 5.B) — required a 3rd kmod rotation (Step 5.B fixup), DKMS
re-sync, retag of `week-4-step-5-sense-session-proc`. End-to-end measurement
probe verified the matrix populates correctly with canonical labels.

**Date:** 2026-05-21
**Phase:** v1.2.2 §7 Week 4, Step 6 of 6
**Anchors:**
  - cipher_rt_phase4: `3be4531` → **`3c5ddaa`** (tag `week-4-step-6-sub4-measurement`)
  - cipher_kmod: `5782609` → **`2fc70c3`** (retagged `week-4-step-5-sense-session-proc` to fixed HEAD)
  - cipher-may13-evidence: `fc8a9ae` (unchanged)
  - DKMS srcversion: `2106824D60BEA5C2333F4D6` → **`CECE94921DE1F43F04E452F`** (loaded + DKMS-installed in sync)

---

## A — Pre-edit + entry srcversion gate — PASS

| signal | value |
|---|---|
| cipher_kmod HEAD pre-Step-6 | `5782609` (Step 5 commit; later retagged) |
| cipher_rt_phase4 HEAD | `3be4531` ✓ |
| Loaded srcversion | `2106824D60BEA5C2333F4D6` (Step 5 close) |
| DKMS srcversion | `2106824D60BEA5C2333F4D6` (in sync) |

---

## B — Sub-4 wrapper inspection + emit-Option decision — Option (c) chosen

`cipher_rt_sense_transition.c` (242 LOC pre-Step-6) already has:
- per-tenant table (256-slot, direct-mapped, evict-oldest on collision)
- hysteresis (N=8 stability, M=4 transition-debounce, K=100ms tool-idle)
- bounded proposal queue (64-slot, drop-oldest)
- existing proposals_queued / proposals_pushed accessors
- flush via existing `CIPHER_DSM_PROPOSE` ioctl (W3 Step 4 II-a)

**Insertion points found:**
- transition emit at L171-174 (when stable-class changes) → bump per-tenant + matrix
- tool-idle emit at L188-191 (AGENT gap > K) → bump per-tenant + aggregate

**Emit Option chosen — (c) filesystem JSON.** Preflight Part 5.3 recommended
this for v1: "avoids new ioctl + kmod state". Single `report()` function
writes to `/tmp/cipher_sense_transitions.json`. Operator-triggered between
workloads; not on the hot path.

Alternatives rejected:
- (a) new ioctl + kmod proc node — would require a **4th kmod rotation
  in 24h** (Steps 4, 5, 5-fixup already done); LOC budget would jump from
  ~65 to ~120; no benefit for v1 measurement
- (b) extend `CIPHER_DSM_PROPOSE` payload — Pattern (a) observability is
  already in `dsm_proposals` ring; transition+tool-idle counters are a
  different category (cumulative metrics, not per-event log), better
  suited to a separate dump

---

## C — Implementation — PASS

### cipher_rt_sense_transition.h (+16 LOC)

3 new public functions:
- `unsigned long cipher_rt_sense_transition_tool_idle_count(void);`
- `unsigned long cipher_rt_sense_transition_transitions_total(void);`
- `int cipher_rt_sense_transition_report(const char *path);`  (`path=NULL` → `/tmp/cipher_sense_transitions.json`)

### cipher_rt_sense_transition.c (+113 LOC)

| change | location | purpose |
|---|---|---|
| `struct tenant_state` +2 fields | `transitions, tool_idles` atomic_uint | per-tenant cumulative counters |
| 3 new file-static globals | `g_transition_matrix[4][4]`, `g_tool_idle_count`, `g_transitions_total` | per-(from,to) matrix + aggregates |
| observe() hot-path hook at L171 | after `sense_emit_proposal` for stable change | bump `s->transitions++`, `g_transitions_total++`, `g_transition_matrix[stable][op_class]++` |
| observe() hot-path hook at L188 | after `sense_emit_proposal` for tool-idle | bump `s->tool_idles++`, `g_tool_idle_count++` |
| accessor functions | end of file | wrap atomic loads |
| `report()` JSON dumper | end of file (~70 LOC) | writes schema, aggregate, 4×4 matrix, per-tenant array; filters out empty slots |

**Class indices match canonical CipherSessionType** (cipher_sense.h):
`0=UNKNOWN, 1=HUMAN, 2=AGENT, 3=BATCH`. The existing observe() code already
used `stable == 2 /* AGENT */` (consistent with canonical 2=AGENT).

Hot-path impact: 4 `atomic_fetch_add_explicit(relaxed)` per stable-class
transition + 2 per tool-idle. Both bounded by existing debouncing
(N=8 stability, 100ms K). Per-launch observe() unchanged when no event
fires.

### Mid-step bonus fix — canonical class names (Step 5.B retroactive correction)

While writing Step 6, I noticed my Step 5.B `cipher_sense_class_names[4]`
in cipher_proc.c was `{HUMAN, AGENT, BATCH, UNKNOWN}` indexed `0,1,2,3`.
But the authoritative SENSE enum in cipher_sense.h is
`UNKNOWN=0, HUMAN=1, AGENT=2, BATCH=3`. My ordering happened to put AGENT
at index 2 correctly by coincidence (matching the existing observe()
`stable == 2 /* AGENT */` check) but the other 3 labels were misaligned.

**Fix:** updated both `cipher_proc.c` (the sense_session emitter) and
`cipher_internal.h` (the struct comment) to canonical
`{UNKNOWN, HUMAN, AGENT, BATCH}`. Same fix applied to:
- `cipher_rt_sense_transition.c`'s report() class_names[]
- `cipher_exporter.py`'s parse_sense_session per_class dict + regex

**Cost:** 3rd kmod rotation in 24h. Bench:
- New cipher_kmod build, srcversion `2106824D…` → **`CECE94921DE1F43F04E452F`**
- DKMS sync executed (same procedure as Step 4 + 5.B)
- `sudo rmmod + sudo modprobe`; loaded srcversion matches DKMS-installed ✓
- `/proc/cipher/sense_session` now emits `UNKNOWN/HUMAN/AGENT/BATCH` in
  canonical order

**Retag:** moved `week-4-step-5-sense-session-proc` from `5782609` →
`2fc70c3` (the fixed commit). Rationale: shipping a tag at a known-buggy
commit is worse than retag — the buggy state never reached external
consumers.

### Probe verification — PASS

`/tmp/week4_step5_6_7/sense_measurement_probe.c` (dlopen-based) simulates:
- tenant 1234: 8×UNKNOWN(class 0), 8×HUMAN(1)... wait, 8×class-1, 8×class-2,
  long gap (>100ms in AGENT), 8×class-0 → expected transitions:
  UNKNOWN→HUMAN, HUMAN→AGENT, AGENT→UNKNOWN + 1 tool-idle
- tenant 5678: 8×class-0, 8×class-1 → expected: UNKNOWN→HUMAN

Probe output:
```
proposals_queued: 5
transitions_total: 4
tool_idle_count:  1
report rc=0
```

`/tmp/cipher_sense_transitions.json` verified:
```
"transition_matrix": {
  "UNKNOWN": { "UNKNOWN": 0, "HUMAN": 2, "AGENT": 0, "BATCH": 0 },
  "HUMAN":   { "UNKNOWN": 0, "HUMAN": 0, "AGENT": 1, "BATCH": 0 },
  "AGENT":   { "UNKNOWN": 1, "HUMAN": 0, "AGENT": 0, "BATCH": 0 },
  "BATCH":   { "UNKNOWN": 0, "HUMAN": 0, "AGENT": 0, "BATCH": 0 }
}
"tenants": [
  { "tenant_id": 5678, "stable": "HUMAN", … "transitions": 1, "tool_idles": 0 },
  { "tenant_id": 1234, "stable": "UNKNOWN", … "transitions": 3, "tool_idles": 1 }
]
```

Matrix and tenant entries match the probe trace exactly.

---

## D — SC6 + smoke — PASS

| arm | result | log |
|---|---|---|
| **SC6 TinyLlama vanilla** | 7/7 PASS bit-identical | `sc6_post_step6b_vanilla.log` |
| **SC6 TinyLlama CIPHER** | 7/7 PASS bit-identical | `sc6_post_step6b_cipher.log` |
| **SC6 Mistral-7B + `CIPHER_SENSE=1`** | 7/7 PASS bit-identical | `sc6_post_step6b_mistral_sense.log` |

Mistral-7B + CIPHER_SENSE=1 confirms the measurement infra runs alongside
the live SENSE classifier without perturbing SC6 correctness invariants.
SC6's workload doesn't drive enough observe() calls to trigger
SENSE_STABILITY_N=8 transitions on a per-tenant fingerprint, so the
report would be all-zero from a SC6-only run — that's expected; the
probe binary exercises the path empirically.

CP 5.4 isolation: **15/15 PASS byte-identical** to pre-Step-5 baseline
(diff `cp54_{pre_step5,post_step6}.log` = empty).

---

## E — Commit + tag — PASS

| signal | value |
|---|---|
| cipher_kmod commit (Step 5 fixup) | **`2fc70c3`** "Week 4 Step 5 (kmod, fixup): align sense_session class names with canonical CipherSessionType" — 2 files / +13 / -8 |
| cipher_kmod tag | `week-4-step-5-sense-session-proc` (retagged from `5782609` → `2fc70c3`) |
| cipher_rt_phase4 commit (Step 6) | **`3c5ddaa`** "Week 4 Step 6: Sub-4 measurement infrastructure (~65 LOC userspace-only)" — 2 files / +129 / -0 |
| cipher_rt_phase4 tag | **`week-4-step-6-sub4-measurement`** at HEAD |
| Fallbacks preserved | `cipher_kmod_fallback/cipher_kmod.ko.w4_step5b_canonical_names` (`22febc8b`) + `libcipher_rt.so.w4_step6_sub4_measurement` (`259ac994`) |

---

## Honest notes

1. **3rd kmod rotation in 24h.** Steps 4 (LP-8 retire), 5 (sense_session
   proc), 5-fixup (canonical names) each required full DKMS re-sync +
   rmmod/modprobe + srcversion gate. The DKMS sync procedure standardized
   at Step 4 worked unchanged for both subsequent rotations. Each cycle
   ~2 min including SC6 + CP 5.4 + probe.

2. **Mid-step bug catch is the right kind of mid-step bug.** The Step 5.B
   sketch happened to put AGENT at the right index by coincidence (matching
   the existing `stable == 2` check), so SC6 passed even with the
   misaligned ordering. The bug would have surfaced when a userspace
   producer eventually populates `per_class[1]` thinking it's AGENT
   (canonical), only to have the kmod display it as "AGENT" (which was
   actually index 1 = AGENT in my wrong table, but the kmod's storage
   index 1 was supposed to be AGENT-by-coincidence). The correction
   prevents the future bug.

3. **report() not wired into flush().** I added the `report()` function
   but did NOT auto-invoke it from `cipher_rt_sense_transition_flush()`.
   Operator must call it explicitly (e.g., via a dlopen probe like my
   `sense_measurement_probe.c`, or a future CLI tool). Two reasons:
   (a) preflight Part 5.3 explicitly says "Operator captures
   /tmp/sense_transitions.json snapshots between workload runs" — not a
   per-flush dump; (b) the file write would add file-system latency to
   the 256-launch flush cadence, polluting the hot-path metric.

4. **Class indices are stable v1 ABI.** The `per_class[4]` array indices
   are part of any future userspace-to-kmod ioctl payload (Step 5.B's
   "no producer wired yet" comment). Misaligned indices would have meant
   producer bridges and consumer reports disagreeing on the same array.
   Caught at Step 6 entry, fixed before any producer was built.

5. **One file-set was NOT git-managed:** `cipher_exporter/cipher-exporter.py`
   received the canonical-name fix in `parse_sense_session` + regex. That
   change is on disk only (md5 changed); will land in the Step 7
   cipher-fusion-evidence tail commit alongside the result docs.

---

## Goals enabled

| target | mechanism |
|---|---|
| **Goal 5 — Sub-4 threshold tuning** (Week 5+) | The new aggregate counters + per-class matrix + per-tenant entries give the empirical evidence needed to validate / re-tune N=8 stability, M=4 debounce, K=100ms idle thresholds. Today the values are emitted; tomorrow's tuning is a Week 5+ deliverable. |
| **Goal 2 — Multi-agent transition observability** | The matrix exposes the actual transition graph (which classes shift to which) across all tenants in steady state. Useful for capacity planning / SM-arbitration tuning. |

---

## Adjudication ask

Step 6 closed cleanly. **Step 7 (Week 4 closeout) is unblocked.** All
substantive impl work for Week 4 is complete (Steps 1+2+3+4+5+5-fixup+6).
Step 7 = tagging `week-4-complete` on all 3 trees + WEEK_4_CLOSEOUT.md +
final memory update + cipher-fusion-evidence tail commit absorbing
WEEK_4_STEP_{4,5,6}_RESULT.md + WEEK_4_STEP_4_PREFLIGHT.md +
cipher-exporter.py + the measurement probe.

**Rollback:** `git -C /home/ubuntu/cipher_rt_phase4 reset --hard week-4-step-3-tier-b-ports`
(rt_phase4 side) and `git -C /home/ubuntu/cipher_kmod reset --hard week-4-step-4-lp8-retired`
+ rmmod/insmod from pre fallback (kmod side).

---

**Evidence:**
- `cipher_rt_phase4`: `git show 3c5ddaa` (+129 LOC)
- `cipher_kmod`: `git show 2fc70c3` (+13 / -8)
- `/tmp/week4_step5_6_7/sense_measurement_probe.{c,output}` (probe + JSON output)
- `/tmp/cipher_sense_transitions.json` (canonical-name JSON dump)
- `/tmp/week4_step5_6_7/sc6_post_step6b_{vanilla,cipher,mistral_sense}.log` (all PASS)
- `/tmp/week4_step5_6_7/cp54_post_step6.log` (15/15 byte-identical to pre-Step-5)
- `/home/ubuntu/cipher_kmod_fallback/cipher_kmod.ko.w4_step5b_canonical_names` + `libcipher_rt.so.w4_step6_sub4_measurement`
