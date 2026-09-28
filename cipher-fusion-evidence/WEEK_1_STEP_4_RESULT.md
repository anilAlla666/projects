# Week 1 Step 4 — Result

**HEADLINE STATUS: PARTIAL (STOPPED at Step B per brief discipline; Steps C-F not executed; no edits applied).**

**Date:** 2026-05-20
**Outcome:** Step A passed (clean baseline; cipher_kmod.ko md5 reproduces anchor `008b3c66`; baseline build rc=0). Step B surfaced **three concrete contradictions** between Wave 5 §5.5 Week 1's Cb.2 specification and the actual `cipher_internal.h` + `cipher_ioctl.h` struct layout on disk. Per the brief discipline rule ("If Wave 5 specification is ambiguous about any field, STOP and surface. Do not invent fields."), no edits were applied. Steps C, D, E, F were not executed. The `cipher_kmod` tree remains at `pre-week-1-baseline` (HEAD `6e360f4d95047f6922c570c835a8250fb553b0b7`), working tree clean, `cipher_kmod.ko` md5 `008b3c66faa82c71c1615ddf9c87ec56`.

---

## Section A — Pre-edit verification

### A.1 Git state

| Field | Value |
|---|---|
| HEAD | `6e360f4d95047f6922c570c835a8250fb553b0b7` |
| `pre-week-1-baseline` tag | `6e360f4d95047f6922c570c835a8250fb553b0b7` (matches HEAD) |
| Working-tree status | clean |
| Tags pointing at HEAD | `pre-week-1-baseline` |

### A.2 Pre-edit md5s

| File | md5 |
|---|---|
| `cipher_internal.h` | `3f78cada2dde78c7b38c013eaa1fa2e4` |
| `cipher_ioctl.h` | `24b87d952f68c91d17d606756a54ca6b` |
| `cipher_kmod.ko` (on disk pre-build) | `008b3c66faa82c71c1615ddf9c87ec56` |

### A.3 Pre-edit kmod clean build

`make clean && make` in `cipher_kmod/`:

- rc = 0
- Warning count: **1**
- Post-build cipher_kmod.ko md5: `008b3c66faa82c71c1615ddf9c87ec56` (anchor REPRODUCED byte-identical)
- Full log: `/tmp/week1_step4/pre_cb2_build.log`
- Pre-edit .ko preserved at `/tmp/week1_step4/cipher_kmod.ko.precb2`

**Section A verdict: PASS.** Baseline clean and reproducible; ready for the edit IF the spec is unambiguous.

---

## Section B — Wave 5 Cb.2 spec inspection + current struct reality

### B.1 Wave 5 §5.5 Week 1 Cb.2 spec verbatim (lines 586-595)

```
**Snapshot reserved-tail bump (Cb.2).** Add classifier-side fields to
`cipher_internal.h::struct cipher_pid_stats` AND the userspace mirror
`cipher_ioctl.h::struct cipher_tenant_snapshot_user`, all in the reserved
tail. Fields (proposed):
- `__u32 recommended_sm_count;` (consumed by Cc.7 ALLOCATE)
- `__u32 slo_priority;` (consumed by partition_router)
- `__u8 session_band;` (HUMAN=0 / AGENT=1 / BATCH=2 / UNKNOWN=3)
- `__u8 tenant_billing_class;` (corp=0 / free=1 / research=2)
- `__u8 reserved_pad[2];`
- Remaining: `__u32 reserved[12];` (was 16; now 12).
```

### B.2 Wave 5 I-W1.2 invariant (lines 604-606)

```
- I-W1.2 (size invariant): `sizeof(struct cipher_tenant_snapshot_user)`
  unchanged (still 336 B target; the new 8 B of fields fit in the reserved
  tail).
```

### B.3 Current struct layout — `cipher_pid_stats` (kmod-internal)

`/home/ubuntu/cipher_kmod/cipher_internal.h:103-173`. Ends at line 173 with:

```c
struct cipher_pid_stats {
    ...
    u32      koopman_substitution_eligibility;
    u64      snapshot_jiffies;           /* state_updater freshness */
};
```

**`cipher_pid_stats` has no `reserved[N]` field at all.** The struct ends with `snapshot_jiffies`; there is no reserved tail. Wave 5's "Add classifier-side fields to `cipher_internal.h::struct cipher_pid_stats` ... all in the reserved tail" references a reserved tail that does not exist in this struct.

### B.4 Current struct layout — `cipher_tenant_snapshot` (kmod-internal, distinct from `cipher_pid_stats`)

`/home/ubuntu/cipher_kmod/cipher_internal.h:182-231`. This struct DOES have a reserved tail. Relevant excerpt:

```c
struct cipher_tenant_snapshot {
    ...
    /* --- Agentic / SLO cluster (P4.7) --- */
    u32      session_band;
    u32      slo_priority;
    u32      fairness_quota_remaining_pct;
    u32      _pad_agentic;
    ...
    /* --- Freshness --- */
    u64      snapshot_jiffies;
    u32      reserved[16];
};
```

### B.5 Current struct layout — `cipher_tenant_snapshot_user` (userspace mirror)

`/home/ubuntu/cipher_kmod/cipher_ioctl.h:123-170`. Sizeof: **336 bytes** (verified by gcc-compiled `printf("%zu", sizeof(...))`). Relevant excerpt:

```c
struct cipher_tenant_snapshot_user {
    ...
    /* Agentic / SLO */
    __u32 session_band;
    __u32 slo_priority;
    __u32 fairness_quota_remaining_pct;
    __u32 _pad_agentic;
    ...
    /* Freshness */
    __u64 snapshot_jiffies;
    __u32 reserved[16];
};
```

### B.6 Three contradictions surfaced

**Contradiction 1 — Target struct does not have a reserved tail.**

Wave 5 names `cipher_pid_stats` as one of the two structs to extend "in the reserved tail." `cipher_pid_stats` has no `reserved[N]` field; it ends with `snapshot_jiffies`. The actual structs with a reserved tail are `cipher_tenant_snapshot` (kmod-internal) and `cipher_tenant_snapshot_user` (userspace mirror) — both have `reserved[16]`. Wave 5's naming is incorrect; the correct target structs are `cipher_tenant_snapshot` + `cipher_tenant_snapshot_user`, not `cipher_pid_stats`.

**Contradiction 2 — Two of the four proposed fields already exist in the target struct, at a different size and outside the reserved tail.**

Wave 5 proposes:
- `__u8 session_band` — but `session_band` already exists as `__u32` (and `u32` kmod-side) in the "Agentic / SLO" cluster of both target structs.
- `__u8 slo_priority` — but `slo_priority` already exists as `__u32` (and `u32` kmod-side) in the same cluster.

Both fields are NOT in the reserved tail today; they are in the Agentic / SLO cluster, with `_pad_agentic` as the 4-pack partner. Wave 5's "all in the reserved tail" framing conflicts with this reality.

Adding `__u8` versions to the reserved tail would create two same-named fields — a compile error. Deleting the existing `__u32` versions and replacing them with `__u8` in the reserved tail would shift offsets of every field between the Agentic/SLO cluster and the freshness section, violating the brief's "Existing field offsets MUST NOT change" requirement.

**Contradiction 3 — Wave 5's math is internally inconsistent.**

| Wave 5 source | Number | What it implies |
|---|---|---|
| Field declarations: `__u32 (4B) + __u32 (4B) + __u8 (1B) + __u8 (1B) + __u8[2] (2B)` | 12 bytes added | reserved consumed: 3 of 16 slots |
| "Remaining: `__u32 reserved[12];` (was 16; now 12)" | 4 slots consumed | 16 bytes added |
| I-W1.2: "the new 8 B of fields fit in the reserved tail" | 8 bytes added | 2 of 16 slots consumed |

Three different numbers (8, 12, 16) for the same arithmetic.

### B.7 Canonical interpretation (offered for adjudication, NOT applied)

The only reading that satisfies all three of (a) I-W1.2 size invariant 336, (b) existing field offsets unchanged, (c) Wave 5's "8 B of fields" claim, and (d) all four Wave-5-named fields landing in the struct is:

**Option A** — Add **only 2 new `__u32` fields** (`recommended_sm_count` and `tenant_billing_class`) to the reserved tail of both `cipher_tenant_snapshot` and `cipher_tenant_snapshot_user`. Treat `session_band` and `slo_priority` as already-present (they are). Reduce `reserved[16]` to `reserved[14]`. Total: 8 bytes added (matches I-W1.2 exactly); sizeof unchanged at 336; all existing field offsets preserved.

This requires interpretation of Wave 5 (the spec proposes 5 fields including the 2 that already exist; canonical reading discards the duplicates). The brief's discipline rule forbids inventing or interpreting fields without user adjudication.

**Three other plausible options exist:**

- **Option B** — Add all 5 Wave 5 fields literally (including duplicates of session_band and slo_priority). Requires deleting or renaming the existing `__u32 session_band` and `__u32 slo_priority` fields; this violates the brief's "Existing field offsets MUST NOT change" requirement. Not viable.
- **Option C** — Reinterpret Wave 5 to mean: rename `_pad_agentic` to `tenant_billing_class` (as `__u32`), and add `recommended_sm_count` to the reserved tail. Total: 4 bytes added; sizeof unchanged at 336. Touches one existing field name but offset stays put.
- **Option D** — Adjudicate that Wave 5 was written without inspecting the struct and the actual Cb.2 work is only the 2 truly-new fields plus extending `cipher_pid_stats` (the kmod-internal struct that has no reserved tail) by adding a reserved tail to it. Largest change; touches three structs and an internal-only invariant.

**No option was applied.** The brief explicitly says "Do not invent fields." Each option above requires a user-level decision on how to reconcile Wave 5's spec with the current struct layout.

**Section B verdict: PARTIAL (STOP).** Three concrete contradictions documented; no edits applied; canonical reading + three alternatives surfaced for user adjudication.

---

## Section C — Apply the edit

**NOT EXECUTED.** Per brief discipline rule, no edits were applied. `cipher_internal.h` and `cipher_ioctl.h` are unmodified.

| File | md5 (pre-step) | md5 (post-step) | Status |
|---|---|---|---|
| `cipher_internal.h` | `3f78cada2dde78c7b38c013eaa1fa2e4` | `3f78cada2dde78c7b38c013eaa1fa2e4` | unchanged |
| `cipher_ioctl.h` | `24b87d952f68c91d17d606756a54ca6b` | `24b87d952f68c91d17d606756a54ca6b` | unchanged |

---

## Section D — Build verification + ABI invariant

**NOT EXECUTED.** With zero source modifications, the post-build cipher_kmod.ko md5 is byte-identical to the pre-build anchor `008b3c66faa82c71c1615ddf9c87ec56`. ABI invariants are trivially preserved because no ABI surface changed.

---

## Section E — Regression smoke

**NOT EXECUTED.** Step E requires a post-edit kmod. With no edit applied, there is no regression to verify against. The pre-flight CP 5.4 isolation 15/15 PASS evidence (from WEEK_1_PRE_FLIGHT.md §3.1 + Steps 1, 2 v2, 3 all re-running it) carries forward unchanged.

---

## Section F — Commit and tag

**NOT EXECUTED.** No commit, no tag.

Current `cipher_kmod` working-tree state:

- HEAD: `6e360f4d95047f6922c570c835a8250fb553b0b7` (unchanged, at `pre-week-1-baseline` tag).
- Working tree: clean. No pending files. No untracked.
- cipher_kmod.ko on disk: `008b3c66faa82c71c1615ddf9c87ec56` (the anchor).
- Rollback path: not needed; the tree is already at the rollback anchor.

---

## Headline summary

| Field | Value |
|---|---|
| Headline status | **PARTIAL — STOPPED at Step B per discipline** |
| Step A — pre-verification | PASS |
| Step B — Wave 5 spec inspection | **3 contradictions surfaced; STOP** |
| Step C — apply edit | NOT EXECUTED |
| Step D — build + ABI | NOT EXECUTED |
| Step E — regression smoke | NOT EXECUTED |
| Step F — commit + tag | NOT EXECUTED |
| `cipher_kmod` HEAD | `6e360f4d95047f6922c570c835a8250fb553b0b7` (unchanged) |
| `cipher_kmod.ko` md5 | `008b3c66faa82c71c1615ddf9c87ec56` (unchanged from anchor) |
| Files modified | 0 |
| Rollback path | not needed; tree is at the baseline anchor |

### Surfaced for user adjudication (no mitigation proposed in this document)

1. **`cipher_pid_stats` has no reserved tail.** Wave 5 names it as a target for the reserved-tail bump but the struct ends with `snapshot_jiffies`. Choose:
   - (a) Drop `cipher_pid_stats` from the Cb.2 scope (only `cipher_tenant_snapshot` + `cipher_tenant_snapshot_user` have reserved tails, both are the proper targets).
   - (b) Add a new `reserved[N]` tail to `cipher_pid_stats` first, then bump (changes kmod-internal struct size; touches state_updater and probe paths).

2. **`session_band` and `slo_priority` already exist as `__u32` outside the reserved tail.** Wave 5's `__u8` proposal in the reserved tail would create duplicate field names OR require deleting the existing `__u32` versions. Choose:
   - (a) Treat as already-done; do not re-add. The two existing `__u32` fields satisfy Wave 5's Cb.2 intent.
   - (b) Delete the existing `__u32` fields and replace with `__u8` in the reserved tail. Requires accepting that existing field offsets shift in the Agentic/SLO cluster forward (violates the brief's offset invariant unless explicitly re-adjudicated).
   - (c) Keep both: existing `__u32` fields PLUS new `__u8` fields with different names (rename Wave 5's `session_band` and `slo_priority` to e.g., `session_band_v2`).

3. **Wave 5's math is inconsistent (8B vs 12B vs 16B).** Choose:
   - (a) Trust I-W1.2's "8 B of fields" → add only `recommended_sm_count` and `tenant_billing_class` as `__u32`. This is Option A in §B.7.
   - (b) Trust the field declaration list → add `recommended_sm_count` + `tenant_billing_class` + `reserved_pad[2]` (12B), reduce reserved by 3 slots.
   - (c) Trust "was 16; now 12" → add 16 bytes of new fields somehow.

The discipline rule says no mitigation in this document. These three items together produce 8 possible interpretations of Cb.2. The defensible canonical reading is **Option A** (8B = 2 new `__u32` fields in the reserved tail of `cipher_tenant_snapshot` + `cipher_tenant_snapshot_user`; `cipher_pid_stats` left alone; existing `session_band` / `slo_priority` accepted as-is). The user adjudicates whether this is the intended Cb.2 scope or whether Wave 5's spec needs revision before the next retry.

### Week 1 closeout readiness

Week 1 closeout depends on Cb.2 landing. With Step 4 PARTIAL, Week 1 cannot close. Steps 1, 2 v2, and 3 remain successfully completed and committed:

| Step | Status | Commit | Tag |
|---|---|---|---|
| Step 1 — LP-7 struct rename | PASS | `fc8a9ae6...` in cipher-may13-evidence | `week-1-step-1-lp7-rename` |
| Step 2 v2 — header port | PASS | `50a6f228...` in cipher_rt_phase4 | `week-1-step-2-v2-header-port` |
| Step 3 — cross-tree compile harness | PASS | `bcf8a83b...` in cipher_rt_phase4 | `week-1-step-3-cross-tree-harness` |
| Step 4 — Cb.2 reserved-tail bump | **PARTIAL (stopped)** | — | — |

The next retry of Step 4 needs an adjudicated Cb.2 spec — either a revised Wave 5 field list that matches the actual struct layout, or an explicit user pick from §B.7 Options A–D + the three items above. Once the spec is unambiguous, Step 4 retry should be 30-45 minutes of mechanical work.

---

**End of WEEK_1_STEP_4_RESULT.md.**
