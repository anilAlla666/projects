# Week 5 Entry Pre-Flight — CP 5.5 Readiness Gate

**HEADLINE: BLOCKED-MINOR — substrate fully ready, Week 5 scope requires adjudication.**

All 9 substrate gates PASS. Week-4-complete state is intact, srcversion in
sync between loaded + DKMS, libcipher_rt.so identity preserved (75 Week-4
T-symbols all present), CP 5.4 isolation 15/15 byte-identical, SC6
TinyLlama + Mistral-7B both arms bit-identical, Sub-4 measurement probe
+ JSON valid with canonical class names.

**The blocker is Part 10 (scope) — not Gates 1-9.** v1.2.2 §7's L1389
HTML-comment amendment moves CP 5.5 from Week 5 to Weeks 13-14, repurposing
Week 5 as "KV-dedup live wire + v1 substrate consolidation." But the
document body itself wasn't fully updated — Week 5's section still has
the original v1.2.1 CP-5.5 content, and Week 6's section title duplicates
what Week 5 should become. **The WEEK_4_STEP_7_CLOSEOUT.md §6 also got
this wrong** (cited v1.2.2 but quoted v1.2.1 content). Two valid
interpretations of "Week 5" — user must pick before Step 1 drafts.

**Date:** 2026-05-21. **Read-only diagnostic; no source modified.**

---

## Gate 1 — Tree HEAD verification — PASS

| repo | HEAD | expected | dirty | result |
|---|---|---|---:|---|
| `cipher_rt_phase4` | `3c5ddaa` | `3c5ddaa` | 0 | PASS |
| `cipher_kmod` | `2fc70c3` | `2fc70c3` | 0 | PASS |
| `cipher-may13-evidence` | `fc8a9ae` | `fc8a9ae` | 0 | PASS |
| `cipher-fusion-evidence` | `270b7b1` | `270b7b1` | 13 (untracked) | WARN |

`cipher-fusion-evidence` 13 untracked entries are **pre-existing**
`WEEK_1_*` + `WAVE_5_CB2_ADJUDICATION.md` + `WAVE_5_S5_5_VERIFICATION.md`
+ `future_scope_a/FUTURE_SCOPE_A_PHASE_4_DESIGN_MEMO.md` — carried since
before this session per the user's earlier option choice (Step 3 scope
question). Not Week-5 drift. The 4 most recent Week-4 audit artifacts
all committed in `9230b5c` / `270b7b1`.

---

## Gate 2 — Tag verification — PASS (1 expected absence)

| repo | tag-at-HEAD | required tags exist? |
|---|---|---|
| `cipher_rt_phase4` | `week-4-complete`, `week-4-step-6-sub4-measurement` | ✓ all 4 Week-4 step tags placed (Step 1 `4279461`, Step 2 `b25edf7`, Step 3 `3be4531`, Step 6 `3c5ddaa`) |
| `cipher_kmod` | `week-4-complete`, `week-4-step-5-sense-session-proc` | ✓ Steps 4 (`158ad96`) + 5 (`2fc70c3`) + week-3 prerequisite (`a21a45e`) all placed |
| `cipher-may13-evidence` | `week-1-complete`, `week-1-step-1-lp7-rename`, `week-2-complete`, `week-3-complete`, `week-4-complete` | ✓ |

**Expected absence:** `cipher_rt_phase4` `week-4-step-5-prometheus` is
**not placed** — Week 4 Step 5 had no cipher_rt_phase4 source change
(only kmod + cipher_exporter/ Python). Documented in
`WEEK_4_STEP_5_RESULT.md` honest note #1; placing this tag at unchanged
HEAD `3be4531` would be semantically misleading.

---

## Gate 3 — srcversion match — PASS

| signal | value |
|---|---|
| **loaded** `/sys/module/cipher_kmod/srcversion` | `CECE94921DE1F43F04E452F` |
| **DKMS-installed** `modinfo .../updates/dkms/cipher_kmod.ko` | `CECE94921DE1F43F04E452F` |
| **source-tree** `modinfo /home/ubuntu/cipher_kmod/cipher_kmod.ko` | `CECE94921DE1F43F04E452F` |
| expected | `CECE94921DE1F43F04E452F` |

**All three in sync.** Pod reboot will load the correct LP-8-retired
+ sense_session-proc + canonical-name kmod. No DKMS drift.

---

## Gate 4 — libcipher_rt.so identity — PASS

| signal | value |
|---|---|
| md5 | `259ac994aead2da8289fc84d6116fbe9` (matches expected `259ac994`) |
| oracle_bridge T symbols | **2** (expected 2 — Step 1) ✓ |
| Tier A T symbols (loop+pipeline+pulse+continuity) | **27** (expected 27 — Step 2) ✓ |
| Tier B+ T symbols (trace+receipt+carbon+fairness+fairness_shm+guard+determinism+topology+comply) | **43** (expected 43 — Step 3) ✓ |
| Sub-4 measurement T symbols (report+tool_idle_count+transitions_total) | **3** (expected 3 — Step 6) ✓ |

**Total Week-4 T-symbols: 2+27+43+3 = 75.**

**WARN:** `WEEK_4_STEP_7_CLOSEOUT.md` headline claims "+70 new T-symbols"
in multiple places; the breakdown sums to **75** (correct count). This
is an arithmetic-slip in the closeout writeup, not a gate failure. The
substrate is intact at +75. **One-line correction recommended for the
closeout doc** (search-and-replace 70 → 75 across closeout + memory
entry; should be done before any external reader cites the wrong number).

---

## Gate 5 — cipher-exporter.py drift gate — PASS

| signal | value |
|---|---|
| live `/home/ubuntu/cipher_exporter/cipher-exporter.py` md5 | `ea62478e` |
| snapshot `cipher-fusion-evidence/exporter_w4_step5/cipher-exporter.py` md5 | `ea62478e` |
| diff lines | **0** |

**Drift gate PASS.** Live and snapshot are byte-identical.

**WARN:** `WEEK_4_STEP_7_CLOSEOUT.md` §2 records the snapshot md5 as
`83d0a3d5`; the actual on-disk md5 is `ea62478e`. The discrepancy is
between the closeout's recorded md5 (probably captured at an earlier
moment during writing) and the live/snapshot state at this preflight.
Since live == snap (diff=0), the gate semantic — *"no drift between
live and committed snapshot"* — **holds**. **One-line correction
recommended in the closeout** for the recorded md5 to match the actual
committed snapshot. Not a gate failure.

---

## Gate 6 — /dev/cipher mode + DKMS state — PASS

| signal | value |
|---|---|
| `/dev/cipher` permissions | `crw-rw-rw-` (mode **666**) ✓ — devnode callback intact per `[[cipher-devnode-codified]]` |
| `dkms status cipher-kmod` | `0.4.8, 6.8.0-1046-nvidia, x86_64: installed` ✓ |
| `lsmod cipher_kmod` Used by | **0** ✓ — no dependent module |

---

## Gate 7 — CP 5.4 isolation baseline — PASS

```
=== Phase A result: 15 PASS, 0 FAIL ===
  Test 1 (legacy nr-9 deactivated)  PASS
  Test 2 (ALLOCATE / FREE / QUERY)  4/4 PASS
  Test 3 (pool resize)              4/4 PASS
  Test 4 (do_exit reaper)           1/1 PASS
  Test 5 (disjointness)             3/3 PASS
  Test 6 (concurrent stress, 4×5)   2/2 PASS
```

`diff cp54_post_step6.log cp54_baseline.log` = empty → **byte-identical
to Week 4 Step 6 close**. Substrate not regressed.

---

## Gate 8 — SC6 4-arm baseline — PASS

| arm | result | log |
|---|---|---|
| 8.1 SC6 TinyLlama vanilla | **PASS 7/7 bit-identical** | `/tmp/week5_preflight/sc6_tinyllama_vanilla.log` |
| 8.2 SC6 TinyLlama CIPHER | **PASS 7/7 bit-identical** | `/tmp/week5_preflight/sc6_tinyllama_cipher.log` |
| 8.3 SC6 Mistral-7B vanilla | **PASS 7/7 bit-identical** | `/tmp/week5_preflight/sc6_mistral_vanilla.log` |
| 8.4 SC6 Mistral-7B CIPHER | **PASS 7/7 bit-identical** | `/tmp/week5_preflight/sc6_mistral_cipher.log` |

All 7 SC6 invariants hold across both models and both arms.

---

## Gate 9 — Sub-4 measurement infra functional — PASS

Probe `/tmp/week4_step5_6_7/sense_measurement_probe` executed cleanly:
```
proposals_queued: 5
transitions_total: 4
tool_idle_count:  1
report rc=0
```

JSON validates as `cipher_sense_transitions/v1` schema. All 4 canonical
class names present:
```
matrix keys: ['AGENT', 'BATCH', 'HUMAN', 'UNKNOWN']
aggregate keys: ['proposals_pushed', 'proposals_queued', 'tool_idle_count', 'transitions_total']
tenants count: 2
```

No naming drift — `cipher_sense_transition_report()` emits canonical
`UNKNOWN/HUMAN/AGENT/BATCH` (post-Step-6 fix is intact).

---

## Part 10 — CP 5.5 scope inspection — **MISMATCH FOUND**

### 10.1 — `CIPHER_REENGINEERING_PLAN.md` (v1.2.2) Week 5 text says CP 5.5

`§7 Week 5` at L1358 is verbatim from v1.2.1:
> **Week 5 — CP 5.5 100-tenant benchmark**
> **Goal:** Headline measurement. The marvel demonstrated.
> **Preconditions:** Nemotron Nano (or equivalent same-model agent
> workload) on the pod. **Currently NOT on the pod — install Week 4.**

Plus the full measurement target list (agents/GPU, fleet tok/W, p99,
fairness, MFU per-WL, weight HBM saved, KV-prefix dedup hit rate,
24h soak).

### 10.2 — But the L1389 HTML-comment amendment says otherwise

Immediately after the Week 5 section, an inline amendment reads:

> `<!-- v1.2.2: Weeks 6-14 below are new — they extend v1.2.1's 5-week
> sequence per ADJUDICATION 1 (Option B scope). Weeks 1-5 above are
> unchanged. Note that v1.2.2 also moves CP 5.5 from Week 5 to Weeks
> 13-14 because CP 5.5 ships on the full 30-of-33 ops surface; Week 5
> above (which originally hosted CP 5.5) is repurposed as the KV-dedup
> live wire and v1 substrate consolidation. -->`

And the doc's intro at L58 confirms:
> "Timeline. §7 extended from 5 weeks to 12-14 weeks. New Weeks 6-14:
> W6 (KV-dedup live wire — added between v1.2.1's W4 observability
> and CP 5.5), W7-8 (COMMIT primitive build), W9-10 (RING_WRITE substrate
> build), W11-12 (Koopman tier integration), W13-14 (CP 5.5 headline
> benchmark on full unified runtime with 30 of 33 ops firing)."

### 10.3 — But the document body itself was NOT fully updated

The Week 5 section header still reads "Week 5 — CP 5.5 100-tenant
benchmark" (L1358). The following section "Week 6 — KV-dedup live wire +
v1 substrate consolidation" (L1391) **has the title that should have
been Week 5's new title per the L1389 amendment**.

The L1389 amendment was added but the actual section-renumbering /
relabeling never happened. The doc is internally inconsistent at this
exact join.

### 10.4 — `WEEK_4_STEP_7_CLOSEOUT.md` §6 propagated the inconsistency

This pre-flight's predecessor (my own Week 4 closeout) §6 said:

> *"Per v1.2.2 §7: Week 5 = CP 5.5 (100-tenant benchmark + 24h soak +
> fresh-boot reproducibility in ≤ 30 min)."*

That cited "v1.2.2 §7" but copied the v1.2.1 content (Week 5 = CP 5.5),
**ignoring the L1389 amendment that says v1.2.2 moves CP 5.5 to W13-14
and Week 5 becomes KV-dedup live wire**. The closeout's 9-item gate
list at §6 was written against the wrong scope interpretation. The
gates themselves are still valid (substrate readiness, srcversion, SC6,
CP 5.4) — but the *headline label* "CP 5.5 readiness" is misnamed
under v1.2.2.

### 10.5 — Two valid interpretations of "Week 5" — user must pick

**Interpretation A — v1.2.1 / closeout literal: Week 5 = CP 5.5
(100-tenant benchmark + 24h soak + fresh-boot reproducibility).**
- Pro: matches the document body L1358. Matches the closeout §6.
- Con: ignores L1389 amendment. CP 5.5 requires Nemotron Nano (not on
  pod), full 100-tenant launcher, all 30-of-33 ops firing (current
  state is ~20 ops; COMMIT + RING_WRITE + Koopman tier not built —
  exactly what Weeks 7-12 build per v1.2.2). **Running CP 5.5 today
  measures something but not the v1.2.2-intended marvel.**

**Interpretation B — v1.2.2 amendment authoritative: Week 5 = KV-dedup
live wire + v1 substrate consolidation (CP 5.5 deferred to Weeks 13-14).**
- Pro: matches L1389 amendment + L58 timeline + L1190 "Week 5: CP 5.5
  measurement" actually refers to the final v1.2.2 Week 5 (which is
  W13-14 in the new numbering — confusing but consistent if read as
  "final Week 5 = the W13-14 numbered week").
- Con: requires interpreting the L1389 amendment as authoritative
  over the unrevised L1358 section body. Adds 7-8 weeks to v1 ship
  (Weeks 6-12 = KV-dedup + COMMIT + RING_WRITE + Koopman before
  the headline measurement).

**Interpretation C — hybrid: Week 5 = KV-dedup live wire + v1 substrate
consolidation (per L1389) AND treat W13-14 CP 5.5 as the eventual
benchmark, but skip Weeks 7-12 (COMMIT/RING_WRITE/Koopman) as v1.5+.**
- Pro: matches the amendment intent (Week 5 = KV-dedup) without
  committing to a 14-week trajectory. Ships v1 with the substrate
  consolidated but defers the full 30-op marvel.
- Con: not in any plan doc; would require a fresh adjudication.

### 10.6 — Open questions for any interpretation

Regardless of Week 5 = CP 5.5 vs KV-dedup-live-wire:

1. **Nemotron Nano model availability.** v1.2.2 §Week 5 L1363 says
   "Currently NOT on the pod — install Week 4." Week 4 didn't install
   it. If CP 5.5 (Interpretation A or eventual W13-14) is the target,
   Nemotron Nano (or a substitute "same-model agent workload") must be
   acquired. Alternative: TinyLlama-1.1B at large N (Track 2 SC6
   used N=8 successfully; scaling to N=100 is unverified).

2. **H100 concurrency limit on this pod.** Current state: 132 SMs,
   80 GB HBM, single-tenant per-process. Mistral-7B weights ≈ 14 GB
   per copy. Weight-arena (Track 2 SC5) supports up to 16 tenants
   sharing 1 copy. 100-tenant fit requires either: smaller model,
   weight sharing N=100 against 16-slot arena cap, or tiered eviction.

3. **`cipher_vllm_plugin/cipher_vllm_kv.py` reachability.** v1.2.2's
   actual-intended Week 5 (KV-dedup live wire) ports this. The file
   exists on disk (`/home/ubuntu/cipher_vllm_plugin/cipher_vllm_kv.py`,
   from CP 5.1) but its current state vs the v1.2.2 intent for Week 5
   has not been audited.

4. **Wave 5 §Week-5 (L862) says CP 5.5 too** — Wave 5 doc predates the
   v1.2.2 L1389 amendment and uses the v1.2.1 timeline. Both v1.2.2 and
   Wave 5 carry CP 5.5 / Week 5 language; only the L1389 amendment
   moves it.

---

## Part 11 — Pre-flight summary

### 11.1 — Gate results table

| gate | scope | result |
|---|---|---|
| 1 | Tree HEAD verification | PASS (1 WARN: cipher-fusion-evidence 13 untracked are pre-existing, not Week-5 drift) |
| 2 | Tag verification | PASS (1 expected absence: `week-4-step-5-prometheus` not placed — documented) |
| 3 | srcversion match | PASS (loaded == DKMS == source-tree == `CECE94921DE1F43F04E452F`) |
| 4 | libcipher_rt.so identity | PASS (md5 + all 75 Week-4 T-symbol counts match; closeout's "+70" was an arithmetic slip) |
| 5 | cipher-exporter.py drift gate | PASS (diff=0; closeout's recorded md5 `83d0a3d5` differs from actual `ea62478e` — recorded-md5 doc inaccuracy, not drift) |
| 6 | /dev/cipher + DKMS + lsmod | PASS (mode 666, DKMS installed, Used by 0) |
| 7 | CP 5.4 isolation | PASS (15/15 byte-identical to Week 4 Step 6 close) |
| 8 | SC6 4-arm baseline | PASS (TinyLlama vanilla+CIPHER + Mistral-7B vanilla+CIPHER all 7/7 bit-identical) |
| 9 | Sub-4 measurement probe | PASS (probe runs, JSON validates, canonical class names preserved) |

**All 9 substrate gates PASS.** No regression since Week 4 close.

### 11.2 — Week 5 readiness verdict

**BLOCKED-MINOR.** The substrate is fully ready for whatever Week 5 turns
out to be. **The blocker is exclusively a scope adjudication question:
Interpretation A (CP 5.5) vs B (KV-dedup live wire) vs C (hybrid).**

The substrate (libcipher_rt.so + cipher_kmod + cipher_exporter +
measurement infra) supports both Interpretations A and B:
- For Interpretation A (CP 5.5): substrate is the substrate the
  measurement will exercise. **Plus** Nemotron Nano or substitute
  workload must be acquired; 100-tenant launcher must be written.
- For Interpretation B (KV-dedup): substrate is the same; the work
  item is `cipher_vllm_plugin/cipher_vllm_kv.py` extension + kmod
  kvdedup ioctls (already shipped per v1.2.2 §1.3).

### 11.3 — Open items surfaced

Items 1-4 from §10.6:
1. Nemotron Nano availability decision (acquire / substitute /
   defer)
2. H100 concurrency limit verification at N=100 (or chosen N)
3. cipher_vllm_plugin current-state audit vs v1.2.2 Week-5
   intent
4. v1.2.2 L1389 amendment authoritative vs L1358 section body —
   the document inconsistency itself

Plus from gate WARNs:
5. WEEK_4_STEP_7_CLOSEOUT.md "70 → 75" T-symbol correction
6. WEEK_4_STEP_7_CLOSEOUT.md exporter snapshot md5 correction
   (`83d0a3d5 → ea62478e`)
7. CIPHER_REENGINEERING_PLAN.md L1358/L1391/L1389 internal
   inconsistency — the doc itself needs an editorial pass to either
   actually relabel Week 5 (per L1389 intent) or retract the L1389
   amendment.

### 11.4 — Recommended next action

**Surface Interpretations A/B/C to the user for scope adjudication
BEFORE drafting any Week 5 Step 1 prompt.** The substrate gates have
ruled out any substrate-regression-blockers; what remains is
strategic-scope.

If user picks **Interpretation B** (KV-dedup live wire — most
literal read of L1389): Step 1 prompt drafts against
`cipher_vllm_plugin/cipher_vllm_kv.py` + kmod kvdedup ioctl wiring per
v1.2.2 L1391-1398. Plus the 3 doc corrections (gate WARNs #5/#6/#7).

If user picks **Interpretation A** (CP 5.5 per closeout literal): Step 1
prompt drafts against the 100-tenant launcher + Nemotron Nano
acquisition. Plus the 3 doc corrections. Plus item #1 (workload
substitute decision if Nemotron Nano can't be acquired in-session).

If user picks **Interpretation C** (hybrid): a fresh scope-lock memo
is needed before any step draft.

---

**Evidence:**
- `/tmp/week5_preflight/cp54_baseline.log` (15/15 byte-identical)
- `/tmp/week5_preflight/sc6_{tinyllama,mistral}_{vanilla,cipher}.log` (4× PASS 7/7)
- `/tmp/week5_preflight/sub4_probe.log` + `/tmp/cipher_sense_transitions.json` (Gate 9 PASS)
- `git -C /home/ubuntu/cipher_rt_phase4 rev-parse HEAD` → `3c5ddaa`
- `git -C /home/ubuntu/cipher_kmod rev-parse HEAD` → `2fc70c3`
- `cat /sys/module/cipher_kmod/srcversion` → `CECE94921DE1F43F04E452F`
- `modinfo /lib/modules/$(uname -r)/updates/dkms/cipher_kmod.ko | grep srcversion` → `CECE94921DE1F43F04E452F`
- `md5sum /home/ubuntu/cipher_rt_phase4/libcipher_rt.so` → `259ac994aead2da8289fc84d6116fbe9`
- `diff /home/ubuntu/cipher_exporter/cipher-exporter.py /home/ubuntu/cipher-fusion-evidence/exporter_w4_step5/cipher-exporter.py` → empty
- `/home/ubuntu/cipher-fusion-evidence/CIPHER_REENGINEERING_PLAN.md` L43, L58, L1190, L1358, L1389, L1391-1398 (the scope-inconsistency citations)
- `/home/ubuntu/cipher-fusion-evidence/CIPHER_LOGIC_AUDIT_WAVE_5_FUSION_PLAN.md` L862 (Wave 5's pre-amendment Week 5 = CP 5.5 reference)
- `/home/ubuntu/cipher-fusion-evidence/WEEK_4_STEP_7_CLOSEOUT.md` §6 (closeout's interpretation that this pre-flight contradicts)
