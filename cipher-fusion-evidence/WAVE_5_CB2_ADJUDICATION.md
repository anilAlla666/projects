# Wave 5 §5.5 Cb.2 Audit-Correction Trail

**Date:** 2026-05-20
**Purpose:** Record the discovery, adjudication, and resolution of three contradictions between the Wave 5 §5.5 Week 1 Cb.2 specification and the on-disk struct layout in `cipher_kmod/`. This document is the audit trail for Option A — the adjudicated reading that was applied in Week 1 Step 4 v2.
**Source documents:**
- Wave 5 §5.5 Week 1 (`CIPHER_LOGIC_AUDIT_WAVE_5_FUSION_PLAN.md` L583-595, md5 reference within Wave 5).
- Week 1 Step 4 attempt (`WEEK_1_STEP_4_RESULT.md`, md5 `558869e65a0c960995e2f26027aa7d05`) — surfaced PARTIAL with the contradictions documented in §B.5-B.7.
- Week 1 Step 4 v2 retry (`WEEK_1_STEP_4_V2_RESULT.md`, md5 `9d88eca12b51895b991e229cf8afd446`) — Option A applied, PASS.

---

## Section 1 — Original Wave 5 §5.5 Cb.2 text verbatim

Lines 586-595 of `CIPHER_LOGIC_AUDIT_WAVE_5_FUSION_PLAN.md`:

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

Plus the I-W1.2 invariant (L604-606):

```
- I-W1.2 (size invariant): `sizeof(struct cipher_tenant_snapshot_user)`
  unchanged (still 336 B target; the new 8 B of fields fit in the reserved
  tail).
```

---

## Section 2 — Three contradictions surfaced by Step 4

### Contradiction 2a — Target struct: `cipher_pid_stats` has no reserved tail

Wave 5 names two target structs:
1. `cipher_internal.h::struct cipher_pid_stats`
2. `cipher_ioctl.h::struct cipher_tenant_snapshot_user`

For both, Wave 5 says "all in the reserved tail."

**Disk evidence:** `cipher_pid_stats` (in `cipher_internal.h:103-173`) **has no `reserved[N]` field at all.** The struct ends with `u64 snapshot_jiffies;` at L172. There is no reserved tail in this struct. Wave 5's named target #1 does not have the reserved-tail placeholder Wave 5 assumes.

The struct with a reserved tail on the kmod-internal side is `cipher_tenant_snapshot` (defined at `cipher_internal.h:182-231`, distinct from `cipher_pid_stats`). This struct does have `u32 reserved[16];` at L231.

### Contradiction 2b — Field duplication: `session_band` and `slo_priority` already exist as `__u32`

Wave 5 proposes adding:
- `__u8 session_band` (HUMAN=0 / AGENT=1 / BATCH=2 / UNKNOWN=3)
- `__u8 slo_priority`

**Disk evidence:** Both fields already exist in `cipher_tenant_snapshot` AND `cipher_tenant_snapshot_user` — at full `__u32` size — in the "Agentic / SLO" cluster, **not** in the reserved tail. From `cipher_ioctl.h:165-170`:

```c
    /* Agentic / SLO */
    __u32 session_band;
    __u32 slo_priority;
    __u32 fairness_quota_remaining_pct;
    __u32 _pad_agentic;
```

Wave 5's "add `__u8` session_band/slo_priority to the reserved tail" would either:
- (a) create duplicate field names (compile error), or
- (b) require deleting the existing `__u32` versions — which shifts the offset of every field after the Agentic/SLO cluster, violating the offset-invariant the brief requires.

Neither path is mechanically applicable as Wave 5 wrote it.

### Contradiction 2c — Internal math: 8B / 12B / 16B contradiction

Wave 5 declares 5 fields totaling 12 bytes:

| Field | Size |
|---|---|
| `__u32 recommended_sm_count` | 4 |
| `__u32 slo_priority` | 4 |
| `__u8 session_band` | 1 |
| `__u8 tenant_billing_class` | 1 |
| `__u8 reserved_pad[2]` | 2 |
| **Total** | **12** |

Wave 5 also says: "Remaining: `__u32 reserved[12];` (was 16; now 12)." This implies **16 bytes** (4 slots × 4 bytes) consumed — but the field declarations sum to **12** bytes (3 slots × 4 bytes).

Wave 5 also says (I-W1.2): "the new 8 B of fields fit in the reserved tail" — implying **8 bytes** (2 slots × 4 bytes).

Three different byte counts (8, 12, 16) for the same arithmetic. Without further clarification it is not possible to apply Wave 5's spec literally.

---

## Section 3 — On-disk evidence

### 3.1 — `cipher_pid_stats` (kmod-internal, `cipher_internal.h:103-173`)

```c
struct cipher_pid_stats {
    struct hlist_node node;
    struct rcu_head   rcu;
    pid_t             pid;
    pid_t             tgid;
    char              comm[TASK_COMM_LEN];
    char              tenant_id[CIPHER_TENANT_ID_LEN];
    atomic64_t        per_slot[CIPHER_NV_IOCTL_COUNT];
    atomic64_t        other_count;
    atomic64_t        total;
    atomic64_t        errors;
    u64               first_seen_jiffies;
    u64               last_seen_jiffies;
    /* ... phase 3 telemetry ... */
    /* ... phase 4 derived state ... */
    u32               session_band;
    u32               slo_priority;
    u32               fairness_quota_remaining_pct;
    /* ... attention cluster ... */
    u64               snapshot_jiffies;
};
```

End of struct: no `reserved[N]` field. Sizeof: determined by accumulation of all listed fields.

### 3.2 — `cipher_tenant_snapshot` (kmod-internal, `cipher_internal.h:182-231`)

```c
struct cipher_tenant_snapshot {
    /* Identity, telemetry, partition, DVFS, L2, KV, weight, fusion ... */
    /* --- Agentic / SLO cluster (P4.7) --- */
    u32      session_band;
    u32      slo_priority;
    u32      fairness_quota_remaining_pct;
    u32      _pad_agentic;
    /* --- Attention cluster (P4.6 secondary) --- */
    u32      graph_capture_state;
    u32      koopman_substitution_eligibility;
    /* --- Freshness --- */
    u64      snapshot_jiffies;
    u32      reserved[16];      /* <-- this is the actual reserved tail */
};
```

### 3.3 — `cipher_tenant_snapshot_user` (userspace mirror, `cipher_ioctl.h:123-179`)

```c
struct cipher_tenant_snapshot_user {
    /* (same field set as cipher_tenant_snapshot, with __u32/__u64 typed) */
    /* Agentic / SLO */
    __u32 session_band;
    __u32 slo_priority;
    __u32 fairness_quota_remaining_pct;
    __u32 _pad_agentic;
    /* Attention */
    __u32 graph_capture_state;
    __u32 koopman_substitution_eligibility;
    /* Freshness */
    __u64 snapshot_jiffies;
    __u32 reserved[16];          /* <-- userspace mirror reserved tail */
};
```

**Verified `sizeof(struct cipher_tenant_snapshot_user) = 336`** via gcc-compiled test program (offsets recorded in WEEK_1_STEP_4_V2_RESULT.md §B.3).

---

## Section 4 — Option A adjudication

The user adjudicated **Option A** as the canonical reading. The adjudication satisfies all three Wave 5 constraints (I-W1.2 byte count, offset invariant, sizeof preservation) by treating the spec as:

### 4.1 — What Option A does

- **Target structs:** `cipher_tenant_snapshot` (kmod-internal) **and** `cipher_tenant_snapshot_user` (userspace mirror). NOT `cipher_pid_stats` (which has no reserved tail and would require a much larger change to extend).
- **Fields added:** 2 new fields, both `__u32`:
  - `recommended_sm_count` (consumed by Cc.7 ALLOCATE; classifier-derived partition hint)
  - `tenant_billing_class` (operator config: corp / free / research)
- **Field placement:** at the HEAD of the reserved tail (consuming the first 2 of the existing 16 reserved `__u32` slots).
- **Reserved shrinkage:** `reserved[16]` → `reserved[14]` to absorb the 8 bytes for the 2 new fields.
- **Pre-existing fields NOT touched:** `session_band` and `slo_priority` already exist as `__u32` in the Agentic/SLO cluster; they are left alone. The "`__u8 session_band` + `__u8 slo_priority` + `__u8 reserved_pad[2]`" subset of Wave 5's spec is treated as already-accomplished (at `__u32` size).

### 4.2 — Why Option A satisfies all three Wave 5 constraints

| Constraint | Option A satisfies |
|---|---|
| I-W1.2 ("new 8 B of fields") | **Exactly 8 bytes added.** 2 × `__u32` = 8 B. |
| sizeof unchanged at 336 | **Held.** 8 bytes added in new fields; 8 bytes removed from `reserved[16]→reserved[14]`. Net zero. |
| Existing offsets unchanged | **Held.** New fields placed at the HEAD of the reserved tail (former `reserved[0]` / `reserved[1]` positions). Every non-reserved existing field stays at the same offset. |
| Wave 5's named fields all accounted for | **3 of 4 satisfied.** `recommended_sm_count` and `tenant_billing_class` added; `session_band` and `slo_priority` already in struct (at `__u32` rather than `__u8`); `reserved_pad[2]` becomes moot at `__u32` alignment. |

### 4.3 — Three alternatives that Option A rejects

The Step 4 result doc surfaced these for completeness:

- **Option B** — Add all 5 Wave 5 fields literally. Requires deleting existing `__u32 session_band` and `__u32 slo_priority` in the Agentic/SLO cluster. **Rejected** because deleting those shifts every offset after the cluster, violating the offset-invariant.
- **Option C** — Rename `_pad_agentic` to `tenant_billing_class` (as `__u32`), add `recommended_sm_count` to the reserved tail. **Rejected** because it touches an existing field name; while the offset stays put, the field-name change risks breaking external readers that key on `_pad_agentic`.
- **Option D** — Add a `reserved[N]` field to `cipher_pid_stats` first, then apply Cb.2. **Rejected** because it expands the kmod-internal struct unnecessarily and triggers re-verification of every read path; out of scope for a Week 1 reserved-tail bump.

Option A is the least-disturbing of the four options and the one that mechanically satisfies all three invariants.

---

## Section 5 — Implementation outcome

Option A was applied in Week 1 Step 4 v2 (`WEEK_1_STEP_4_V2_RESULT.md`):

| Step | Outcome |
|---|---|
| A — pre-verification | PASS (clean baseline at `pre-week-1-baseline`, ko md5 = anchor `008b3c66`) |
| B — pre-edit layout | PASS (sizeof 336, 10 key offsets captured) |
| C — Option A edits applied | PASS (2 edits to `cipher_internal.h` + `cipher_ioctl.h`, 24 lines added, 2 removed) |
| D — build + ABI invariants | PASS (rc=0, sizeof preserved at 336, all non-reserved offsets unchanged, new fields at 272 / 276, ko md5 `008b3c66` → `09c6ded5...`) |
| E — regression smoke | PASS (kmod rmmod + insmod clean, CP 5.4 isolation 15/15 byte-identical to Step 3 baseline) |
| F — commit + tag | PASS (`f8572ecf422050a488f1fb45f74534ecd1bde678`, tag `week-1-step-4-cb2-reserved-tail`) |

The `reserved[14]` tail still leaves 56 bytes (14 × 4 B) of capacity for future Cb.x extensions if needed.

---

## Section 6 — Lesson encoded

**Wave 5 is synthesis. When Wave 5 conflicts with on-disk evidence, on-disk evidence wins.**

The Cb.2 case has three subtypes of Wave 5/disk drift:

1. **Wrong target named.** Wave 5 said `cipher_pid_stats` had a reserved tail; it does not. Future §5.5 entries that name a target need to be verified on disk before execution.
2. **Duplicate field names.** Wave 5 proposed fields that already exist (at a different size). The synthesis assumed the struct didn't have them already.
3. **Internal math inconsistency.** Wave 5's own arithmetic (12 B fields + 4 reserved slots consumed + 8 B in I-W1.2) contradicted itself.

The corrective protocol going forward:

- Before executing any Wave 5 §5.5 entry, **verify the named target exists at the named location with the named structure** (file path, struct/function name, field set, line numbers if cited). This is the "WAVE_5_S5_5_VERIFICATION.md sweep" that landed alongside this document.
- If a target is missing or its structure differs, **STOP and surface for adjudication** before applying any edit.
- If Wave 5's math doesn't balance, **trust the explicit invariant (I-W1.x style)** over the implicit count drift, and surface the discrepancy.
- Wave 5 paraphrases (`flash_call` for the C++-mangled SDPA trampoline; `_tick()` for `cipher_partition_tick()`) are acceptable as long as the target function or struct can be unambiguously identified from the actual disk content.

The Cb.2 contradiction would have been caught earlier if the §5.5 sweep had been done at plan v1.2.2 sealing time rather than at Step 4 execution time. Going forward, the sweep is part of the closeout artifact for each week.

---

**End of WAVE_5_CB2_ADJUDICATION.md.**
