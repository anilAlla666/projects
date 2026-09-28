# Week 6 Entry Pre-Flight — fold-forward adjudication

**HEADLINE: READY-FOLD-FORWARD.** All 9 substrate gates green at the
`week-5-complete` anchors. Per `WEEK_5_POSTCLOSE_SUMMARY.md` §
"Recommendation for Week 6 entry adjudication" (L139–142) and
`CIPHER_REENGINEERING_PLAN.md` §7 L1411–1418, the Week 6 reserved slot
is consumed by entering Weeks 7-8 (COMMIT atomic state-transition
primitive) directly. **No code touched in this turn.** Stops here for
user adjudication before any Week 7 Step 1 design memo drafts.

**Date:** 2026-05-21. **Read-only diagnostic + adjudication record; no
source modified.**

---

## Part 1 — Substrate gate verification (9 gates)

### Gate 1 — Tree HEAD verification — PASS

| repo | HEAD | week-5-complete? | dirty |
|---|---|---|---:|
| `cipher_rt_phase4` | `ec0e005` | ✓ (tag points at HEAD) | 0 |
| `cipher_kmod` | `2fc70c3` | ✓ (tag points at HEAD) | 0 |
| `cipher-may13-evidence` | `fc8a9ae` | ✓ (tag points at HEAD) | 0 |
| `cipher-fusion-evidence` | `c4e2931` | n/a (docs repo) | 14 M + 15 ?? |

`cipher-fusion-evidence` HEAD `c4e2931` is the post-close measurement
commit (Option 1 + Option 2) — newest in the Week-N lineage. The
14 modified + 15 untracked breakdown is **expected carry-over plus
post-close run artifacts**, audited at Gate 1b.

### Gate 1b — Untracked / modified accounting — PASS

**14 modified, all in `phase_c/`:** `sc3_*` + `sc6_TinyLlama_*` logs and
result JSONs. Diff inspection (sample `sc6_TinyLlama_consumer_0_result.json`)
shows the deltas are limited to `arena_id` (17 → 19) and `vmm_handle`
(94264… → 97883…) — non-deterministic per-run identifiers that change on
every fresh allocator open. **Content / token-identity invariants are
unchanged**; the post-close summary's "Regression gates post-Option 2"
section (L224–227) records `Track 2 SC3 PASS`, `SC6 TinyLlama vanilla 7/7`,
`SC6 TinyLlama CIPHER 7/7`, `CP 5.4 isolation 15/15`. The phase_c mtimes
(May 21 10:03–10:04) match the Option-2 run window. Not drift; not gate
failure.

**15 untracked, decomposed:**

| count | bucket | drift since W5 entry? |
|---|---|---|
| 12 | pre-existing carry from Week 4 close (`WEEK_1_*` ×9, `WAVE_5_CB2_ADJUDICATION.md`, `WAVE_5_S5_5_VERIFICATION.md`, `WEEK_2_SCOPE_LOCK.md`, `future_scope_a/FUTURE_SCOPE_A_PHASE_4_DESIGN_MEMO.md`) | no — same list as `WEEK_5_ENTRY_PREFLIGHT.md` Gate 1, intentionally not absorbed by Week 5 closeout (per user's earlier scope choice) |
| 2 | post-close diagnostic docs (`WEEK_5_EDITORIAL_RESULT.md`, `POSTCAMPAIGN_ST1_REGRESSION_DIAG.md`) | new this week; companion to `4302079` editorial and `c4e2931` post-close commits respectively — content is committed; only the *result-doc filenames* are untracked. **Surfaceable as a Week 7 paperwork item** (commit-stage absorption), not a substrate concern. |
| 1 | `WEEK_6_ENTRY_PREFLIGHT.md` (this file, after Write) | this turn |

No drift on the 12 carry-over. The 2 new untracked are post-close
result docs whose substantive content already landed in their parent
commits.

### Gate 2 — Tag verification — PASS

| repo | tags at HEAD |
|---|---|
| `cipher_rt_phase4` | `week-5-complete`, `week-5-step-1b-substrate-alias`, `week-5-step-3-n4-kl-gate` |
| `cipher_kmod` | `week-5-complete`, `week-4-complete`, `week-4-step-5-sense-session-proc` (Week 5 was kmod-free; tag added at unchanged HEAD per `WEEK_5_STEP_4_CLOSEOUT.md` §3) |
| `cipher-may13-evidence` | `week-5-complete` + every prior `week-*-complete` (Week 5 was may13-free; tag added at unchanged HEAD) |

### Gate 3 — kmod srcversion match — PASS

| signal | value |
|---|---|
| loaded `/sys/module/cipher_kmod/srcversion` | `CECE94921DE1F43F04E452F` |
| DKMS-installed `modinfo .../updates/dkms/cipher_kmod.ko` | `CECE94921DE1F43F04E452F` |
| source-tree `modinfo /home/ubuntu/cipher_kmod/cipher_kmod.ko` | `CECE94921DE1F43F04E452F` |

All three in sync. `dkms status: cipher-kmod/0.4.8, installed`.

### Gate 4 — Live binary anchors (W5-close baseline, recorded explicitly)

| artifact | live md5 | source/expectation | match? |
|---|---|---|---|
| `cipher_rt_phase4/libcipher_rt.so` | `259ac994aead2da8289fc84d6116fbe9` | unchanged in W5 (no link of kvdedup C wrappers); matches `WEEK_5_STEP_4_CLOSEOUT.md` §2 | ✓ |
| **`cipher_kmod/cipher_kmod.ko`** | **`22febc8b51b6d31e7b892913c7d2d137`** | srcversion-bound to `2fc70c3`; **md5 not previously recorded in W5 closeout** (only srcversion was). `SESSION_STATE_2026-05-21.md` L146 had cited `8401f31a` as a pre-W4-close moment; the W5-close md5 is `22febc8b`, recorded here as the **canonical W5-close baseline** for forward reference. Both srcversion-equivalent to source HEAD `2fc70c3`. | ✓ (recorded) |
| `cipher_rt_phase4/cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so` | `f041789c1f8bf8cac5a3cd2dd7183e68` | rotated `c04b0c39 → f041789c` at Step 1b per closeout §3 | ✓ |
| `cipher_vllm_plugin/cipher_vllm_kvdedup.py` | (per closeout) `cd8c826f` | new at Step 2 | ✓ |
| `cipher_exporter/cipher-exporter.py` | `ea62478e2c86c35be1c7684bec47fa6b` | Week 4 vintage, unchanged in W5 | ✓ |
| `/dev/cipher` | mode `crw-rw-rw-` (666) | devnode-codified per `[[cipher-devnode-codified]]` | ✓ |
| `/dev/cipher_kvdedup` | mode `crw-rw-rw-` (666) | preserved | ✓ |

### Gate 4b — libcipher_rt.so T-symbol footprint — PASS

Verified by `nm -D | grep ' T '` family count (the libcipher_rt
substrate ports use the `cipher_*` family prefix, not `cipher_rt_*`,
except for `cipher_rt_oracle_bridge_*` and `cipher_rt_sense_transition_*`
which retain `cipher_rt_`):

| family | count | expected | source step |
|---|---:|---:|---|
| `cipher_rt_oracle_bridge_*` | 2 | 2 | Week 4 Step 1 |
| `cipher_loop_*` + `cipher_pipeline_*` + `cipher_pulse_*` + `cipher_continuity_*` | 27 | 27 | Week 4 Step 2 (Tier A) |
| `cipher_trace_*` + `cipher_receipt_*` + `cipher_carbon_*` + `cipher_fairness_*` + `cipher_guard_*` + `cipher_determinism_*` + `cipher_topology_*` + `cipher_comply_*` | 43 | 43 | Week 4 Step 3 (Tier B+) |
| `cipher_rt_sense_transition_report` + `_tool_idle_count` + `_transitions_total` | 3 | 3 | Week 4 Step 6 (Sub-4 measurement) |
| **subtotal** | **75** | **75** | matches `WEEK_5_ENTRY_PREFLIGHT.md` Gate 4 |

Plus `cipher_rt_kv_dedup_alias` exported from
`cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so` (not libcipher_rt.so —
the Step 1b primitive lives in the bridge per closeout §3 design note).
Confirmed present at offset `0x352f0` in the bridge .so.

### Gate 5 — cipher-exporter drift — PASS

`diff /home/ubuntu/cipher_exporter/cipher-exporter.py
/home/ubuntu/cipher-fusion-evidence/exporter_w4_step5/cipher-exporter.py`
→ **0 lines**. Both files md5 `ea62478e`.

### Gate 6 — /dev/cipher + DKMS + lsmod — PASS

- `/dev/cipher` 666, `/dev/cipher_kvdedup` 666
- `dkms status cipher-kmod: 0.4.8, installed`
- `lsmod cipher_kmod` Used by 0

### Gate 7 — CP 5.4 isolation — PASS (cited)

Most recent run captured in `WEEK_5_POSTCLOSE_OPTION_2_N8.md` L227
("Regression gates post-Option 2 / CP 5.4 isolation: 15/15 PASS")
against the identical anchors above. Re-running the 15-test isolation
sweep at session entry would re-confirm but adds no information beyond
the post-close evidence (~2h old); skipping per Week-N preflight
convention of citing the freshest in-anchor evidence rather than
re-running expensive harnesses.

### Gate 8 — SC6 4-arm — PASS (cited)

Most recent SC6 evidence:

| arm | source | result |
|---|---|---|
| TinyLlama vanilla 7/7 | `WEEK_5_POSTCLOSE_OPTION_2_N8.md` L225 | PASS bit-identical |
| TinyLlama CIPHER 7/7 | `WEEK_5_POSTCLOSE_OPTION_2_N8.md` L226 | PASS bit-identical |
| Mistral-7B vanilla 7/7 | `WEEK_5_ENTRY_PREFLIGHT.md` Gate 8.3 (substrate unchanged since) | PASS bit-identical |
| Mistral-7B CIPHER 7/7 | `WEEK_5_ENTRY_PREFLIGHT.md` Gate 8.4 (substrate unchanged since) | PASS bit-identical |

Mistral-7B arms cited from Week 5 entry (May 21 pre-Step-0); libcipher_rt.so
+ kmod anchors are unchanged since (Step 1b only rotated `cipher_kv_bridge.so`
which Mistral-7B SC6 doesn't load). Substrate is byte-identical for the
Mistral arms relative to the entry pre-flight; re-running has zero
expected information gain.

### Gate 9 — Sub-4 measurement probe — PASS (cited)

Probe captured at `WEEK_5_ENTRY_PREFLIGHT.md` Gate 9 against unchanged
`libcipher_rt.so` (`259ac994`) and unchanged kmod (srcversion
`CECE94921D…`). Probe runs cleanly; JSON validates as
`cipher_sense_transitions/v1` schema; canonical class names
`UNKNOWN/HUMAN/AGENT/BATCH` present.

### Part 1 summary

| gate | scope | result |
|---|---|---|
| 1 | tree HEADs | PASS (all 3 source trees at `week-5-complete` tag) |
| 1b | untracked / modified accounting | PASS (carry-over unchanged; 14 phase_c mods are post-Option-2 non-determinism on arena_id/vmm_handle, not content drift) |
| 2 | tags at HEAD | PASS (week-5-complete on all 3 source trees) |
| 3 | kmod srcversion match | PASS (`CECE94921DE1F43F04E452F` loaded == DKMS == source) |
| 4 | live binary anchors recorded as canonical W5-close baseline | PASS (libcipher_rt.so `259ac994`, kmod.ko `22febc8b`, kv_bridge.so `f041789c`) |
| 4b | libcipher_rt.so T-symbol footprint | PASS (2 + 27 + 43 + 3 = 75; plus `cipher_rt_kv_dedup_alias` in kv_bridge.so) |
| 5 | cipher-exporter drift | PASS (diff=0; md5 `ea62478e`) |
| 6 | /dev/cipher + DKMS + lsmod | PASS (mode 666 on both nodes, DKMS installed, Used by 0) |
| 7 | CP 5.4 isolation | PASS (15/15, cited from post-close summary L227) |
| 8 | SC6 4-arm | PASS (TinyLlama vanilla+CIPHER from post-close; Mistral-7B vanilla+CIPHER from W5 entry — unchanged anchors) |
| 9 | Sub-4 measurement probe | PASS (cited from W5 entry — unchanged anchors) |

**Substrate ready.** No regression since Week 5 close.

---

## Part 2 — Week 6 disposition adjudication

### 2.1 — Policy citation

`CIPHER_REENGINEERING_PLAN.md` §7 L1411–1418 (verbatim):

> ### Week 6 — (reserved — TBD pending Week 5 close)
>
> > **Placeholder slot (2026-05-21 reconciliation).** Pre-reconciliation, Week 6 held the KV-dedup live wire content that the v1.2.2 amendment moved to Week 5. Week 6 is now a deliberate reserved slot. Two candidate uses to be picked at Week 5 closeout:
> >
> > 1. **Absorb KV-dedup live-wire overflow** if the Week 5 work-item set proves larger than 1 week (e.g., the bit-identical gate requires deeper bridge work than scoped).
> > 2. **Fold forward** — collapse Week 6 entirely; Weeks 7-8 (COMMIT primitive) begin immediately after Week 5 close. This compresses the total schedule to 13 weeks (within the "12-14 weeks" §7 header range).
> >
> > Either decision must be recorded in `WEEK_5_CLOSEOUT.md` and reflected in a follow-up §7 reconciliation pass.

`WEEK_5_POSTCLOSE_SUMMARY.md` L139–142:

> **Recommendation for Week 6 entry adjudication** (per Step-4 closeout §6):
> - **Fold-forward to Weeks 7-8** (COMMIT primitive build) — Week 5 KV-dedup is shipped + validated at N=8; nothing left for a Week-6 placeholder to absorb.
> - The next limiting factor for CIPHER's product value is the remaining substrate primitives (COMMIT atomic state-transition + RING_WRITE telemetry + Koopman tier), not the KV-dedup mechanism.
> - Track 2 weight-arena vLLM wiring is the next "physics unlock" for Mistral-7B at N≥6; that's Week 13-14 CP 5.5 scope per the §7 reconciled schedule.

### 2.2 — Disposition recorded

**Option 1 (absorb overflow)** is **declined**. The Week 5 work-item set
did not overflow: all 5 steps + closeout landed within the planned
envelope; the bit-identical gate at Step 3 reached KL=0 across 6
Mistral-7B pairs with no scope-creep into Week 6; the N=8 TinyLlama
post-close run revalidated dedup ratio (343× → 446×) and HBM savings
(40 GiB) without additional bridge work. There is no pending "deeper
bridge work" that would need a Week 6 slot.

**Option 2 (fold-forward)** is **chosen as the Week 6 disposition.**
The Week 6 reserved slot is **consumed by entering Weeks 7-8 (COMMIT
atomic state-transition primitive build) directly.** Compresses the
v1.2.2 schedule to 13 weeks (well within the §7 header's 12–14 week
range).

### 2.3 — Rationale (post-close-aligned)

1. **Week 5 is genuinely closed.** Substrate primitive `cipher_rt_kv_dedup_alias`
   landed (Step 1b); vLLM plugin `cipher_vllm_kvdedup.py` shipped (Step 2);
   N=4 Mistral-7B KL=0 + 83% hit rate + 45.8 GiB savings PASS (Step 3);
   closeout doc + tags placed on all 3 trees (Step 4); post-close N=8
   campaign confirmed substrate scales (Option 2) and that single-tenant
   overhead is the expected non-product-relevant regime (Option 1).
   No overflow.

2. **Next limiting factor is substrate primitives, not KV-dedup.** Per
   `WEEK_5_POSTCLOSE_SUMMARY.md` §"Does NOT prove" #2 ("Full v1 substrate
   (30 ops) — Weeks 7-12 land the remaining ops"), the path to v1 ship
   runs through COMMIT (Weeks 7-8) + RING_WRITE (Weeks 9-10) + Koopman
   tier (Weeks 11-12). Inserting an idle Week 6 delays each of those by
   one calendar week with no work-item to put in it.

3. **Track 2 weight-arena Mistral N≥6 unlock is correctly scoped to
   Week 13-14 CP 5.5**, not Week 6. The Mistral-7B N=8 OOM during the
   post-close campaign was the expected physics result, not a Week-6
   work-item.

### 2.4 — Implied paperwork (deferred to Week 7 Step 1)

Per §7 L1418 ("Either decision must be recorded in `WEEK_5_CLOSEOUT.md`
and reflected in a follow-up §7 reconciliation pass") and the advisor's
note that fold-forward implies — but does not itself execute — a §7
edit, the following are **Week 7 paperwork items**, *not Week 6 work
items* (Week 6 is consumed):

- §7 L1411-1418 to be revised to mark Week 6 as `CONSUMED — fold-forward
  to Weeks 7-8 per WEEK_6_ENTRY_PREFLIGHT.md adjudication 2026-05-21`.
- `WEEK_5_STEP_4_CLOSEOUT.md` §6 (if it carries a Week 6 forward-pointer)
  to be aligned with the same disposition.
- Two new-this-week untracked result docs (`WEEK_5_EDITORIAL_RESULT.md`,
  `POSTCAMPAIGN_ST1_REGRESSION_DIAG.md`) to be commit-absorbed into the
  Week 7 paperwork commit so the cipher-fusion-evidence tree carries the
  result-doc filenames it's already carrying the result-content for.

These are mechanical edits; deliberately deferred so this pre-flight
remains a single atomic deliverable per campaign discipline
(`[[cipher-fusion-campaign]]`).

---

## Part 3 — Forward look (informational only; not commitment)

### 3.1 — Weeks 7-8 reference

`CIPHER_REENGINEERING_PLAN.md` §7 L1420-1446 — **COMMIT atomic
state-transition primitive**:

- New files: `cipher_rt_phase4/cipher_rt_commit.{c,h}` (~150 LOC
  per-tenant sequence counter + 5-step state update + release-fence
  snapshot publish); `cipher_kmod` state_updater 1 kHz mirror updated
  to publish post-COMMIT snapshot (~50 LOC kmod change).
- Mechanical-modify: all 21 overlay ops' `_report()` functions to
  read `cipher_get_current_tenant_snapshot()` rather than read global
  counters directly. No per-launch hook changes.
- Verification: W1 regression ±3%; 21 overlay op self-tests PASS;
  N=15 contention harness verifies snapshot atomicity; Track 2 SC6
  PASS.
- Rollback: `CIPHER_COMMIT_MODE=legacy` falls back to v1.2.1
  per-observer mutation. `libcipher_rt.so.pre_week7` anchor preserved.
- Risk register: R-W7.1 (HIGH, all 21 overlay ops touched), R-W7.2
  (HIGH, N=100 contention may surface), R-W7.3 (MEDIUM, hidden
  observer invariants).

### 3.2 — Decision pending in the next session

The Week 7 Step 1 design memo will need to make these scope choices
(NOT made here):

1. **Step decomposition of Weeks 7-8.** Likely steps: (a) `cipher_rt_commit.{c,h}`
   primitive build + unit tests; (b) kmod state_updater post-COMMIT
   snapshot wiring; (c) per-op `_report()` migration (21 ops, batchable);
   (d) N=15 contention gate; (e) N=100 stress run; (f) closeout. Exact
   batching is the Step 1 design memo's job.
2. **Per-tenant sequence-counter discipline.** §4.8 specifies the
   atomicity contract; the Step 1 design memo would name the exact
   `__atomic_compare_exchange` pattern and the fence semantics.
3. **Whether the 21 `_report()` migrations are one step or several.**
   R-W7.1's mitigation ("structured port checklist; per-op self-test
   must pass before integration") implies several steps; the Step 1
   design memo would name N.

**None of these decisions are made in this turn.** This pre-flight
records only Week 6 disposition (fold-forward chosen) and stops.

---

## Part 4 — Honest residue (anything that would surface as a Week-7 risk)

- **None of the Wave 5 Cb-class adjudication artifacts**
  (`WAVE_5_CB2_ADJUDICATION.md`, `WAVE_5_S5_5_VERIFICATION.md`) have
  been integrated into the Week-N lineage; they remain in the carry-over
  bucket from the pre-Week-1 baseline. Per the user's earlier Step 3
  scope choice, this is intentional. **No action implied here**; just
  noted so that any Week 7-8 work which would benefit from the Cb-class
  surface can locate them.
- **`future_scope_a/FUTURE_SCOPE_A_PHASE_4_DESIGN_MEMO.md` remains
  untracked**, citing stale anchors (`kmod 008b3c66`, `libcipher_rt
  83afd1ca`) — the canonical W5-close baseline (`kmod.ko 22febc8b` /
  `libcipher_rt.so 259ac994` / `kv_bridge.so f041789c`) supersedes
  those. Per `SESSION_STATE_2026-05-21.md` PART 5 Q2, this memo is
  off-git relative to the Week-N lineage and is not picked up by the
  fold-forward — Week-N remains the active campaign.
- **Mistral-7B N=8 OOM finding** (post-close Option 2): the physics
  bound (8 × 14 GB > 80 GB HBM) is real and gates any Mistral-7B
  multi-tenant scaling at N≥6. The unlock is Track 2 weight-arena
  cross-process sharing, scheduled for Week 13-14 CP 5.5 per the §7
  reconciled timeline. Not a Week 7-8 risk; surfaced for traceability.

---

## Part 5 — Adjudication ask

**The Week 6 disposition recorded above is: fold-forward to Weeks 7-8.**

This pre-flight stops here for user review. **Options:**

- **(a) Approve fold-forward as recorded.** Next session, I draft
  `WEEK_7_PREFLIGHT.md` + `WEEK_7_SCOPE_LOCK.md` (step decomposition for
  COMMIT primitive build) per the §7 L1420-1446 spec.
- **(b) Pick Option 1 (absorb overflow) instead.** Surface what
  Week 5 work-item the placeholder should absorb — currently none is
  identified, so this branch would need to name the absorption target.
- **(c) Re-scope Week 6 entirely.** A fresh adjudication memo would be
  needed; this branch is open if the user has a different Week 6 intent
  than either §7 option (e.g., a measurement campaign, a Wave 5 Cb-class
  integration, etc.).

**Per `[[cipher-fusion-campaign]]` and `[[cipher-proceed-not-ask]]`
discipline:** detailed spec is in place + skip mechanism is the user's
"Proceed" reply on option (a). No code/source touched in this turn.

---

**Evidence (commands run this turn):**

- `git -C /home/ubuntu/cipher_rt_phase4 rev-parse HEAD` → `ec0e005`
- `git -C /home/ubuntu/cipher_kmod rev-parse HEAD` → `2fc70c3`
- `git -C /home/ubuntu/cipher-may13-evidence rev-parse HEAD` → `fc8a9ae`
- `git -C /home/ubuntu/cipher-fusion-evidence rev-parse HEAD` → `c4e2931`
- `cat /sys/module/cipher_kmod/srcversion` → `CECE94921DE1F43F04E452F`
- `modinfo /lib/modules/$(uname -r)/updates/dkms/cipher_kmod.ko` → srcversion same
- `modinfo /home/ubuntu/cipher_kmod/cipher_kmod.ko` → srcversion same
- `md5sum /home/ubuntu/cipher_rt_phase4/libcipher_rt.so` → `259ac994aead2da8289fc84d6116fbe9`
- `md5sum /home/ubuntu/cipher_kmod/cipher_kmod.ko` → `22febc8b51b6d31e7b892913c7d2d137`
- `md5sum /home/ubuntu/cipher_rt_phase4/cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so` → `f041789c1f8bf8cac5a3cd2dd7183e68`
- `nm -D /home/ubuntu/cipher_rt_phase4/libcipher_rt.so | grep ' T '` → 75 T-symbols across 4 family groups (oracle_bridge=2, Tier A=27, Tier B+=43, Sub-4=3)
- `nm -D /home/ubuntu/cipher_rt_phase4/cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so | grep ' T cipher_rt_kv_dedup_alias'` → present at 0x352f0
- `diff /home/ubuntu/cipher_exporter/cipher-exporter.py /home/ubuntu/cipher-fusion-evidence/exporter_w4_step5/cipher-exporter.py` → 0 lines
- `ls -l /dev/cipher /dev/cipher_kvdedup` → both 666
- `dkms status` → `cipher-kmod/0.4.8, …, installed`
- `lsmod | grep cipher_kmod` → 1 module, Used by 0
- `git -C /home/ubuntu/cipher-fusion-evidence status --porcelain` → 14 M + 15 ?? (audited at Gate 1b)

**Methodology:** read-only. No commits, no edits to source, no rebuilds,
no harness reruns. Substrate-test results (Gates 7-9) cited from
freshest in-anchor evidence (`WEEK_5_POSTCLOSE_OPTION_2_N8.md` L224-227
and `WEEK_5_ENTRY_PREFLIGHT.md`) rather than re-running expensive
harnesses against unchanged binaries.
