# Week 1 Step 4 v2 — Result (Option A retry)

**HEADLINE STATUS: PASS.**

**Date:** 2026-05-20
**Scope:** Retry of Cb.2 reserved-tail bump under user-adjudicated **Option A**. Step 4 surfaced 3 contradictions in Wave 5 §5.5 Cb.2 spec vs. on-disk struct layout (see `WEEK_1_STEP_4_RESULT.md` md5 `558869e65a0c960995e2f26027aa7d05` §B.6). Option A scope: add 2 new `__u32` fields (`recommended_sm_count`, `tenant_billing_class`) to the reserved tail of `cipher_tenant_snapshot` + `cipher_tenant_snapshot_user`; reduce `reserved[16]` → `reserved[14]`; preserve sizeof at 336; `cipher_pid_stats` and the existing `session_band` / `slo_priority` fields untouched.
**Outcome:** All six steps PASS. Both struct sizeofs preserved at 336; all existing field offsets unchanged; new fields land at offsets 272 and 276 as planned; the `reserved` field itself shifts by +8 bytes by design; cipher_kmod.ko rebuilds with rc=0 and warning delta zero; kmod rmmod + insmod cycle clean; CP 5.4 isolation 15/15 PASS byte-identical to Step 3 baseline. Commit `f8572ecf422050a488f1fb45f74534ecd1bde678` landed and tagged `week-1-step-4-cb2-reserved-tail`.

---

## Section A — Pre-edit verification

### A.1 Git state at entry

| Field | Value |
|---|---|
| HEAD | `6e360f4d95047f6922c570c835a8250fb553b0b7` |
| `pre-week-1-baseline` tag | `6e360f4d95047f6922c570c835a8250fb553b0b7` (matches HEAD) |
| Working-tree status | clean (Step 4 attempt left the tree untouched) |

### A.2 Pre-edit md5s

| File | md5 |
|---|---|
| `cipher_internal.h` | `3f78cada2dde78c7b38c013eaa1fa2e4` |
| `cipher_ioctl.h` | `24b87d952f68c91d17d606756a54ca6b` |
| `cipher_kmod.ko` | `008b3c66faa82c71c1615ddf9c87ec56` (matches anchor) |

**Section A verdict: PASS.** Clean baseline; anchor confirmed; no drift from Step 4 attempt.

---

## Section B — Pre-edit struct layout

### B.1 `cipher_tenant_snapshot` (kmod-internal, `cipher_internal.h`)

```c
struct cipher_tenant_snapshot {
    char     tenant_id_str[CIPHER_TENANT_ID_LEN];
    u64      tenant_session_fp;
    u32      tenant_handle_u32;
    ...
    /* --- Attention cluster (P4.6 secondary) --- */
    u32      graph_capture_state;
    u32      koopman_substitution_eligibility;

    /* --- Freshness --- */
    u64      snapshot_jiffies;
    u32      reserved[16];      /* <-- target for Cb.2 bump */
};
```

### B.2 `cipher_tenant_snapshot_user` (userspace mirror, `cipher_ioctl.h`)

```c
struct cipher_tenant_snapshot_user {
    char  tenant_id_str[CIPHER_TENANT_ID_LEN];
    __u64 tenant_session_fp;
    __u32 tenant_handle_u32;
    ...
    /* Attention */
    __u32 graph_capture_state;
    __u32 koopman_substitution_eligibility;

    /* Freshness */
    __u64 snapshot_jiffies;
    __u32 reserved[16];         /* <-- target for Cb.2 bump */
};
```

### B.3 Pre-edit sizeof + offsets (verified via gcc-compiled test program)

```
sizeof(struct cipher_tenant_snapshot_user) = 336
offsetof tenant_id_str          = 0
offsetof tenant_session_fp      = 64
offsetof sm_util_pct            = 88
offsetof sm_partition_mask      = 120
offsetof predicted_hot_regions  = 152
offsetof session_band           = 240
offsetof slo_priority           = 244
offsetof graph_capture_state    = 256
offsetof snapshot_jiffies       = 264
offsetof reserved               = 272
```

Reserved tail starts at offset 272 and runs 64 bytes (16 × 4) to 336. Preserved at `/tmp/week1_step4_v2/pre_offsets.txt`.

**Section B verdict: PASS.** Both target structs end with `__u32 reserved[16];` exactly as expected.

---

## Section C — Apply Option A edit

### C.1 Edit to `cipher_internal.h` (cipher_tenant_snapshot)

Replaced 3 lines with 13 lines (10 new):

```diff
 	/* --- Freshness --- */
 	u64      snapshot_jiffies;
-	u32      reserved[16];
+
+	/* --- Cb.2 reserved-tail bump (Wave 5 §5.5 Week 1, Option A adjudication) ---
+	 * Two new __u32 fields land at the HEAD of the reserved tail.
+	 * Existing offsets before this point are unchanged; sizeof preserved
+	 * at 336 B; reserved[16] -> reserved[14] absorbs the 8 bytes.
+	 * Consumer: COMMIT primitive (Weeks 7-8) for atomic per-tenant state
+	 * update — see CIPHER_REENGINEERING_PLAN.md v1.2.2 §4.8 and
+	 * WEEK_1_STEP_4_V2_RESULT.md. */
+	u32      recommended_sm_count;     /* classifier-derived partition hint; consumed by Cc.7 ALLOCATE */
+	u32      tenant_billing_class;     /* corp=0 / free=1 / research=2 (operator config -> state_updater) */
+	u32      reserved[14];
 };
```

### C.2 Edit to `cipher_ioctl.h` (cipher_tenant_snapshot_user)

Replaced 3 lines with 15 lines (12 new):

```diff
 	/* Freshness */
 	__u64 snapshot_jiffies;
-	__u32 reserved[16];
+
+	/* Cb.2 reserved-tail bump (Wave 5 §5.5 Week 1, Option A adjudication).
+	 * Two new __u32 fields land at the HEAD of the reserved tail; existing
+	 * offsets before this point are unchanged; sizeof preserved at 336 B;
+	 * reserved[16] -> reserved[14] absorbs the 8 bytes.
+	 * Consumer: COMMIT primitive (Weeks 7-8) for atomic per-tenant state
+	 * update — see CIPHER_REENGINEERING_PLAN.md v1.2.2 §4.8 and
+	 * WEEK_1_STEP_4_V2_RESULT.md. Userspace mirror of the kmod-internal
+	 * struct cipher_tenant_snapshot (cipher_internal.h) — names + offsets
+	 * match exactly. */
+	__u32 recommended_sm_count;
+	__u32 tenant_billing_class;
+	__u32 reserved[14];
 };
```

Both edits preserve the exact indentation style of the surrounding code (tabs for `cipher_internal.h`, tabs for `cipher_ioctl.h`).

**Section C verdict: PASS.** Two edits applied; both structures identical in scope (recommended_sm_count + tenant_billing_class + reserved[14]); names + offsets matched.

---

## Section D — Build + ABI invariants

### D.1 Post-edit kmod clean build

`make clean && make` in `cipher_kmod/`:

- rc = 0
- Warning count: **1** (delta zero vs pre-edit; the BTF-skip note is the same pre-existing warning surfaced in WEEK_1_PRE_FLIGHT.md §2.2)
- Build log: `/tmp/week1_step4_v2/post_cb2_v2_build.log`
- Full link line emitted; cipher_kmod.ko linked + MODPOST passed

### D.2 cipher_kmod.ko md5 transition

| State | md5 |
|---|---|
| Pre-edit (anchor) | `008b3c66faa82c71c1615ddf9c87ec56` |
| Post-edit | `09c6ded53e33d1ee418dcdc6d8605cd4` |

The md5 change is expected — the struct layout extension affects every TU that includes `cipher_internal.h` or `cipher_ioctl.h` (which is most of them).

### D.3 Post-edit sizeof + offsets

```
sizeof(struct cipher_tenant_snapshot_user) = 336    [INVARIANT HELD]
offsetof tenant_id_str          = 0                 (unchanged)
offsetof tenant_session_fp      = 64                (unchanged)
offsetof sm_util_pct            = 88                (unchanged)
offsetof sm_partition_mask      = 120               (unchanged)
offsetof predicted_hot_regions  = 152               (unchanged)
offsetof session_band           = 240               (unchanged)
offsetof slo_priority           = 244               (unchanged)
offsetof graph_capture_state    = 256               (unchanged)
offsetof snapshot_jiffies       = 264               (unchanged)
offsetof recommended_sm_count   = 272               (NEW)
offsetof tenant_billing_class   = 276               (NEW)
offsetof reserved (post-edit)   = 280               (was 272; +8 by design)
sizeof reserved (post-edit)     = 56                (was 64; absorbs 8 bytes for new fields)
```

`diff pre_offsets.txt post_offsets.txt` shows exactly one difference: the `reserved` field offset moved from 272 → 280. **This shift is by design** and is exactly what Option A specified — the 8 bytes absorbed for the new fields come from the head of `reserved`. Every other field's offset is unchanged.

### D.4 ABI invariants verified

| Invariant | Status |
|---|---|
| sizeof(cipher_tenant_snapshot_user) unchanged at 336 | **HELD** |
| All non-reserved existing field offsets unchanged | **HELD** (10 offsets verified identical pre vs post) |
| reserved[N] field shifted by +8 bytes (by design) | **EXPECTED** |
| All 24 wired ioctl NR defines (CIPHER_REGISTER_TENANT through CIPHER_ARENA_QUERY) | **UNCHANGED** (verified via grep of `#define CIPHER_*` in cipher_ioctl.h) |
| Reserved ioctl nrs 2/3/4 still return -ENOSYS | **HELD** (`cipher_dev.c:112` and `cipher_dev.c:300` retain the -ENOSYS returns) |
| Build rc=0 | PASS |
| Warning delta zero | PASS (1 → 1) |

**Section D verdict: PASS.** All ABI invariants held. Option A's size-neutral edit confirmed at offset level.

---

## Section E — Regression smoke

### E.1 Kmod rmmod + insmod cycle (re-load with the new .ko)

```
$ sudo rmmod cipher_kmod           ; rc=0
$ lsmod | grep cipher_kmod         ; (empty — unloaded)
$ sudo insmod cipher_kmod.ko       ; rc=0
$ lsmod | grep cipher_kmod         ; cipher_kmod 1163264 0  (loaded)
```

### E.2 Post-reload sanity

- `/dev/cipher` and `/dev/cipher_kvdedup` re-created (major 511 and 510).
- All `/proc/cipher/*` nodes present: `arenas`, `bar0_state`, `flops`, `gpu_state`, `migrations`, `stats`.
- dmesg shows clean init sequence:
  - `cipher_kmod: kprobes detached, alloc_failures=0 reaped=477` (graceful unload)
  - `cipher_kmod: unloaded cleanly`
  - `cipher_kmod: loading (Phase 4.2 T4.2.1 — lock-free SM partition allocator)`
  - `cipher_partition_allocator: lock-free atomic slots; sm_count=132 slots=32 (4 SMs each), cap 8 per tenant`
  - `cipher_kmod: CP 5.4 arbitration ledger — 15 × 8-SM groups (120 SMs)`
  - `cipher_kmod: Track 2 SC5 weight-arena registry — 16 slots, 5s liveness reaper`
  - `cipher_kmod: CP 3.3 FLOP telemetry ready`
  - `cipher_kmod: /dev/cipher ready (major=511, REGISTER_TENANT + GPU_STATE/PROCESS_UTIL/LAUNCH_STATS live)`
  - `cipher_kmod: /dev/cipher_kvdedup ready (major=510, T4.6.4 cross-tenant KV dedup)`
  - `cipher_bar0: bound to 0000:07:00.0 [10de:2330] BAR0=0x6002000000 size=16 MB`
  - `cipher_kmod: kprobes attached at 0xffffffffc05f6300; do_exit reaper at 0xffffffffa2b0cfc0`
  - `cipher_kmod: loaded ok; nvidia_unlocked_ioctl @ 0xffffffffc05f6300 hooked`
  - `cipher_state_updater: kthread starting (cadence 1000 Hz)`

No errors. No WARN. No oops. Clean init.

### E.3 CP 5.4 isolation 15/15 against the new kmod

Re-ran `cipher-fusion-evidence/cp_5_4/step1_3/cp54_isolation_test`:

| Test | Sub-assertions | Result |
|---|---|---|
| Test 1 — legacy nr-9 deactivated | 1 | PASS |
| Test 2 — ALLOCATE / FREE / QUERY | 4 | 4× PASS |
| Test 3 — pool resize | 4 | 4× PASS |
| Test 4 — do_exit reaper | 1 | PASS |
| Test 5 — disjointness | 3 | 3× PASS |
| Test 6 — concurrent stress | 2 | 2× PASS |
| **Total** | **15** | **15 PASS, 0 FAIL** |

### E.4 Byte-identical regression check

```
$ diff /tmp/week1_step3/cp54_isolation.log /tmp/week1_step4_v2/cp54_isolation.log
(no output — diff empty)
```

CP 5.4 isolation output **byte-identical** to the Step 3 baseline. The Cb.2 struct extension has zero observable effect on substrate runtime behavior.

**Section E verdict: PASS.** Kmod reload clean; CP 5.4 isolation 15/15 byte-identical to baseline.

---

## Section F — Commit and tag

### F.1 Diff stat

```
 cipher_internal.h | 12 +++++++++++-
 cipher_ioctl.h    | 14 +++++++++++++-
 2 files changed, 24 insertions(+), 2 deletions(-)
```

Two files modified; 24 insertions / 2 deletions. Matches the brief's expectation ("2 files, small line counts").

### F.2 Commit + tag

| Field | Value |
|---|---|
| Commit SHA | `f8572ecf422050a488f1fb45f74534ecd1bde678` |
| Tag | `week-1-step-4-cb2-reserved-tail` (lightweight) at the same SHA |
| Previous tag preserved | `pre-week-1-baseline` at `6e360f4d...` |
| Working tree post-commit | clean (zero pending) |
| Author | `Anil <anil.0666369@gmail.com>` |
| Co-Authored-By | `Claude Opus 4.7 (1M context) <noreply@anthropic.com>` |

### F.3 Rollback path

- **Roll back this step:** `git reset --hard pre-week-1-baseline` in `cipher_kmod/` returns the tree to the pre-Step-4 state.
- **Tarball:** `/home/ubuntu/cipher-baselines/pre_week_1_cipher_kmod_20260520.tar.gz` (md5 `23bf73a40e5f06bcff1e3c26256da1d8`, including `.git/`) is the belt-and-suspenders artifact.

**Section F verdict: PASS.** Commit landed, tag placed, working tree clean.

---

## Headline summary

| Field | Value |
|---|---|
| Headline status | **PASS** |
| Step A — pre-verification | PASS |
| Step B — pre-edit layout capture | PASS (sizeof 336, 10 offsets recorded) |
| Step C — apply Option A edit | PASS (2 files, 24 insertions / 2 deletions) |
| Step D — build + ABI invariants | PASS (rc=0, warning delta zero, sizeof preserved, all non-reserved offsets unchanged) |
| Step E — regression smoke | PASS (kmod reload clean, CP 5.4 isolation 15/15 byte-identical to baseline) |
| Step F — commit + tag | PASS (SHA `f8572ecf...`, tag `week-1-step-4-cb2-reserved-tail`) |
| Files changed | 2 (`cipher_internal.h`, `cipher_ioctl.h`) |
| Lines added | 24 |
| sizeof transition (cipher_tenant_snapshot_user) | 336 → 336 (unchanged) |
| cipher_kmod.ko md5 transition | `008b3c66...` → `09c6ded53e33d1ee418dcdc6d8605cd4` |
| Rollback path | `git reset --hard pre-week-1-baseline` |

### Surfaced (informational, no mitigation needed)

1. **The `reserved` field offset shifted by +8 bytes** (from 272 to 280). This is BY DESIGN per Option A: the 8 bytes absorbed for `recommended_sm_count` (offset 272) and `tenant_billing_class` (offset 276) come from the head of the previous `reserved[16]` array. The field is called `reserved` precisely because its slot positions are not load-bearing; any consumer that reads `reserved[0]` was reading the placeholder space that's now occupied by the named fields.

2. **`cipher_pid_stats` was deliberately not touched** per Option A (Wave 5 named it as a target but it has no reserved tail; the target structs are the snapshot structs). If a future Cb.2 round needs to extend `cipher_pid_stats`, that is a separate (and larger) change because the kmod-internal struct does not have a reserved-tail placeholder.

3. **Wave 5 spec ↔ reality reconciliation** documented in this round closes out the 3 contradictions surfaced by Step 4. The on-disk struct + the v1.2.2 plan + the COMMIT primitive's design (Weeks 7-8) all now reference the same 4 fields: `recommended_sm_count`, `tenant_billing_class`, `session_band` (pre-existing), `slo_priority` (pre-existing).

### Week 1 closeout readiness

All four Step 4 v2 sections of Week 1 (per plan v1.2.2 §7) are now complete and committed:

| Step | Status | Commit | Tag |
|---|---|---|---|
| Step 1 — LP-7 struct rename | PASS | `fc8a9ae6...` in cipher-may13-evidence | `week-1-step-1-lp7-rename` |
| Step 2 v2 — header port (12-header closure) | PASS | `50a6f228...` in cipher_rt_phase4 | `week-1-step-2-v2-header-port` |
| Step 3 — cross-tree compile harness | PASS | `bcf8a83b...` in cipher_rt_phase4 | `week-1-step-3-cross-tree-harness` |
| Step 4 v2 — Cb.2 reserved-tail bump | **PASS** | `f8572ecf...` in cipher_kmod | `week-1-step-4-cb2-reserved-tail` |

Week 1 is closeable. **Week 1 Step 5 (closeout) prompt can be sent.**

---

**End of WEEK_1_STEP_4_V2_RESULT.md.**
