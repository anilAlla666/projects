# Week 4 Step 4 — Pre-flight — LP-8 Allocator Retirement + C4 Naming-Drift

**Headline: SCOPE-AS-WRITTEN.** Step 4 ships as a clean ~520-LOC kmod
deletion (one file removed + ~30 LOC of caller/decl/Makefile cleanup) +
one paperwork erratum for the C4 Wave-5 mis-naming. No caller cascade,
no ABI change at the ioctl-mask level, no observable runtime change.
**First kmod rotation since pre-Step-1 baseline.**

**Date:** 2026-05-21. **Read-only diagnostic.** No source modified.
**Anchors verified at this preflight:**
- cipher_rt_phase4 HEAD `3be4531` (tag `week-4-step-3-tier-b-ports`)
- cipher_kmod HEAD `a21a45e` (tags `week-3-complete`, `week-3-step-4-opt2a-dsm-propose`)
- cipher-may13-evidence HEAD `fc8a9ae` (unchanged)
- cipher-fusion-evidence HEAD `d4ef628` (Step 2+3 audit tail commit landed)
- Loaded `cipher_kmod.ko` srcversion `F6B9227C41E5439BD8F1B02` (baseline for the post-Step-4 reload gate; will change)

---

## Part 1 — LP-8 surface identification

### 1.1–1.3 What LP-8 actually is

**LP-8 ≠ slab/kmem_cache allocator.** The user-spec phrasing
("slab allocator") would have matched standard Linux memory infrastructure
in `cipher_proc.c`, `cipher_clock.c`, `cipher_probe.c`, etc. — none of
those are the LP-8 target. **LP-8 = `cipher_partition_allocator.c`** — the
legacy SM-partition slot-array allocator (Phase 4.2 T4.2.1). The word
"allocator" refers to **SM-partition allocation** (`cipher_partition_request`
returns u32 bitmask of SM-block slots), not memory allocation.

Wave 5 verbatim (line 494, 780, 1255):
> `cipher_partition_allocator.c` is REFACTOR-RETIRE: userspace `nr 9`
> dead, but `state_updater::tick` (5 s rebalance) and
> `do_exit::release_slots_only` live. Confusion risk. … Retire cleanly
> in Week 4 …

### 1.4 The surface

| file | LOC | purpose |
|---|---|---|
| `cipher_kmod/cipher_partition_allocator.c` | **482** | The whole LP-8 implementation. Slot array, request/release, idle reap, rebalance, init/exit. |
| (no .h sibling — no public header exists; declarations live in `cipher_internal.h`) | — | — |

**Public functions exported (`EXPORT_SYMBOL_GPL`):**

| function | line | export | live caller? |
|---|---|---|---|
| `cipher_partition_request(pid, hint, …)` | 124 | EXPORT_SYMBOL_GPL @ 130 | **DEAD** — only callable via ioctl nr 9 which returns -ENOSYS since CP 5.4 close |
| `cipher_partition_request_v2(pid, hint, flags, …)` | ~140 | EXPORT_SYMBOL_GPL @ 255 | **DEAD** — same path |
| `cipher_partition_release(pid)` | 260 | EXPORT_SYMBOL_GPL @ 277 | **DEAD** — only called via ioctl nr 10 (also -ENOSYS path; companion to the dead `_request`) |
| `cipher_partition_release_slots_only(pid)` | 282 | EXPORT_SYMBOL_GPL @ 293 | **LIVE** — `cipher_probe.c:210` (do_exit reaper) |
| `cipher_partition_tick(void)` | 429 | (no EXPORT — internal API) | **LIVE** — `cipher_state_updater.c:176` (1 kHz kthread) |
| `cipher_partition_allocator_init(void)` | 448 | (no EXPORT) | **LIVE** — `cipher_main.c:51` (module init) |
| `cipher_partition_allocator_exit(void)` | 476 | (no EXPORT) | **LIVE** — `cipher_main.c:118` (module exit) |

**Internal (file-static) functions** (zero callers outside the file —
all go away with the file):

| function | line | role |
|---|---|---|
| `cipher_partition_find_tenant_rcu` | 95 | helper |
| `cipher_partition_writeback_cache` | 106 | writes `cipher_pid_stats::sm_partition_mask/count` |
| `cipher_partition_reap_idle` | 298 | called by tick |
| `cipher_load_cmp_desc` | 343 | sort comparator |
| `cipher_partition_rebalance` | 352 | called by tick |

**Storage:**

| name | location | size | role |
|---|---|---|---|
| `struct cipher_partition_slot { atomic_t state; }` | `cipher_partition_allocator.c:88` | sizeof(atomic_t), cache-aligned to 64 B | per-slot type |
| `cipher_slots[CIPHER_PARTITION_SLOTS_MAX]` | `:91` | 32 × 64 B = 2048 B static BSS | the slot array (`CIPHER_PARTITION_SLOTS_MAX = 32` macro at `:66`) |
| `cipher_partition_slots` (int) | `:78` | 4 B | runtime slot count (clamped to 32 in `_init` at `:461-462`) |

---

## Part 2 — LP-8 caller inventory

### 2.1–2.2 Resolved call graph

```
USERSPACE (cipher_rt_phase4, vLLM workers, …)
     │ ioctl nr 9 → -ENOSYS (since CP 5.4 close; stub at cipher_dev.c:107)
     ▼
KMOD
     ├─ cipher_main.c:51   cipher_partition_allocator_init()    (LIVE)
     ├─ cipher_main.c:118  cipher_partition_allocator_exit()    (LIVE)
     ├─ cipher_state_updater.c:176  cipher_partition_tick()     (LIVE — 1 kHz)
     ├─ cipher_probe.c:210  cipher_partition_release_slots_only(pid)  (LIVE — do_exit)
     └─ ioctl handlers for nr 9/10 → already wrappers that return -ENOSYS
```

**Live call sites: 4.** All in cipher_kmod itself, none in userspace
(cipher_rt_phase4 has zero refs to `cipher_partition_*` in current .c/.cpp;
the only matches are `.pre_T4_2_4d` historical snapshots, see Part 4.5).

### 2.3 Caller-purpose mapping

| caller | path | replaces with |
|---|---|---|
| `cipher_main.c:51, :118` | module init/exit | **delete** — slot array storage goes away with the .c file |
| `cipher_state_updater.c:176 cipher_partition_tick()` | 1 kHz kthread tick | **delete the call**. Wave 5 §785 suggests "move the 30 s idle reclaim into state_updater directly as a 'free slot reaper' leg." Investigation shows the legacy idle-reclaim **had nothing to reclaim** — `cipher_partition_request*` (the only writer of `cipher_slots[]`) has never been called since CP 5.4 deactivated nr 9, so the slot array has been all-zero. Reclaim of empty slots = no-op. **Recommendation: delete the call, not the file.** State this explicitly in Step 4 result doc — does not match Wave 5's "move semantics" wording verbatim but achieves the same final state. |
| `cipher_probe.c:210 cipher_partition_release_slots_only(pid)` | do_exit reaper | **delete the call**. Same reasoning: slots are always zero, release-slots-only is a no-op walk. The `cipher_cp54_release(pid)` call at `cipher_probe.c:198` handles all live partition state. |
| ioctl nr 9 handler (`cipher_dev_request_sm_partition` at `cipher_dev.c:107-113`) | -ENOSYS stub | **delete the function**; replace `case CIPHER_REQUEST_SM_PARTITION: return cipher_dev_request_sm_partition(arg);` (cipher_dev.c:265-266) with a one-liner `case CIPHER_REQUEST_SM_PARTITION: return -ENOSYS;`. **The struct `cipher_partition_request` in `cipher_ioctl.h:232` STAYS** (ABI surface preservation per T-W4.5). |

**No caller cascade.** Replacements are all deletions or one-line inlines.

---

## Part 3 — CP 5.4 + Week 2-3 ioctl interaction check

### 3.1 CP 5.4 path inventory

CP 5.4 lives in `cipher_kmod/cipher_cp54_sched.c` with its own 15-group
8-SM-block ledger (totally independent of LP-8's 32-slot 4-SM-block array).
Wave 5 §101 verbatim: *"the legacy 4-SM-slot allocator and the CP 5.4
8-SM-group ledger (nrs 13/14/15, cipher_cp54_sched.c) address the same
132 physical SMs."*

### 3.2 Per-ioctl handler — touches LP-8?

| ioctl nr | name | handler file | calls `cipher_partition_*`? |
|---|---|---|---|
| 9 | `CIPHER_REQUEST_SM_PARTITION` (LP-8) | cipher_dev.c | **stub only — returns -ENOSYS** |
| 13 | `CIPHER_CP54_ALLOCATE` | cipher_cp54_sched.c | **no** |
| 14 | `CIPHER_CP54_FREE` | cipher_cp54_sched.c | **no** |
| 15 | `CIPHER_CP54_QUERY` | cipher_cp54_sched.c | **no** |
| 16–20 | `CIPHER_CP54_{SUBSCRIBE,POLL,START,ACK,COMPACT}_MIGRATE` | cipher_cp54_sched.c | **no** |
| 21–24 | `CIPHER_ARENA_{REGISTER,IMPORT,LEAVE,QUERY}` (Track 2 SC5) | cipher_weight_arena.c | **no** |
| 25 | `CIPHER_PUSH_CLASSIFY_STATS` (Week 2 Step 6) | cipher_classify_proc.c | **no** |
| 26 | `CIPHER_DSM_PROPOSE` (Week 3 Step 4 II-a) | cipher_cp54_sched.c | **no** |

**No CP 5.4 / Week 2-3 / Track 2 ioctl path touches LP-8.** CP 5.4
isolation 15/15 will be byte-identical post-deletion.

### 3.3 do_exit reaper — both ledgers, independently

`cipher_probe.c::cipher_do_exit_pre()` calls:
- **L198:** `cipher_cp54_release(pid)` — CP 5.4 8-SM-group release. **Stays.**
- **L210:** `cipher_partition_release_slots_only(pid)` — LP-8 4-SM-slot release. **Deletes.**

The two are independent: `cipher_cp54_release` walks the CP 5.4 group ledger
(`cipher_cp54_sched.c::cp54_release_groups`); `cipher_partition_release_slots_only`
walks the LP-8 slot array. Removing the LP-8 leg cannot affect CP 5.4 state.

---

## Part 4 — C4 cipher_partition_slot naming-drift

### 4.1–4.3 Resolved

The user-spec C4 was already documented as a Wave-5 erratum in
`WAVE_5_S5_5_VERIFICATION.md` §C4 (L85-95) before this preflight.
The Wave 5 doc text says "Remove the slot array `cipher_partition_slot[32]`
and all writes to it" — but in the code:

- **`struct cipher_partition_slot`** = the per-slot **type** (`cipher_partition_allocator.c:88`)
- **`cipher_slots`** = the actual **array** of that type (`:91`)
- **`CIPHER_PARTITION_SLOTS_MAX`** = the size macro = **32** (`:66`) — Wave 5's "32" is correct (the clamp at `:461-462` enforces ≤ 32 from the file-header comment's 33 = 132/4)

There is no variable in the code named `cipher_partition_slot[32]`. Wave 5
conflated the struct type with the array variable.

### 4.4 Category

**PAPERWORK-ONLY**, and **mooted by LP-8 file deletion**. After Step 4:
- `struct cipher_partition_slot` ceases to exist (file deleted)
- `cipher_slots[]` ceases to exist (file deleted)
- `CIPHER_PARTITION_SLOTS_MAX` macro ceases to exist (file deleted)

The Wave 5 text remains technically incorrect but **points at a deleted
entity** after Step 4 — like a stale link. Minimal fix is a one-line
erratum in the Step 4 commit message + result doc, citing
`WAVE_5_S5_5_VERIFICATION.md §C4` as the canonical correction. **Do NOT
fold a code rename into Step 4** — there is no code to rename; the
target is being deleted.

### 4.5 Out-of-tree consumer audit (advisor flag ADV-2)

- **No loaded kernel module depends on `cipher_kmod`** — `lsmod | grep cipher_kmod` shows `Used by 0`.
- **No `cipher_rt_km*` binary exists** on the pod — the Wave-5-era comment "EXPORT_SYMBOL_GPL on request/release so future `cipher_rt_km` can call" was forward-looking; `cipher_rt_km` was never built.
- **No `cipher_rt_phase4` C/C++ source uses `sm_partition_mask`/`sm_partition_count`** at runtime: only the struct field declarations exist in `cipher_rt_tenant.h:53-54` for ABI mirror, and one comment-only ref in a `.pre_T4_2_4d` historical snapshot (`cipher_rt_green_ctx.c.pre_T4_2_4d:14` — pre-rotation, not live).

The five `EXPORT_SYMBOL_GPL` lines (`cipher_partition_request`,
`_request_v2`, `_release`, `_release_slots_only`, and the tick — no, tick
isn't exported; only the four request/release variants) **disappear
without consequence**.

---

## Part 5 — Step 4 scope decision

### 5.1 LP-8 retirement scope (confirmed)

| item | scope | LOC |
|---|---|---|
| **Delete** `cipher_kmod/cipher_partition_allocator.c` | full file | **−482** |
| **Delete** declarations from `cipher_kmod/cipher_internal.h` | L379-396 (8 prototypes) | **−10** |
| **Delete** call `cipher_partition_allocator_init()` in `cipher_main.c:51` | 1 line + 1 blank | **−2** |
| **Delete** call `cipher_partition_allocator_exit()` in `cipher_main.c:118` | 1 line + comment | **−2** |
| **Delete** call `cipher_partition_tick()` in `cipher_state_updater.c:176` + surrounding comment | ~3 lines | **−3** |
| **Delete** call `cipher_partition_release_slots_only(pid)` in `cipher_probe.c:210` + surrounding comment | ~6 lines | **−6** |
| **Delete** `cipher_dev_request_sm_partition` function in `cipher_dev.c:98-113` (15 lines incl. comment) | 15 lines | **−15** |
| **Inline** `case CIPHER_REQUEST_SM_PARTITION: return -ENOSYS;` in `cipher_dev.c:265` | 1 line (replaces 2-line case calling the deleted handler) | **−1** |
| **Keep** `struct cipher_partition_request` in `cipher_ioctl.h:232` (ABI; T-W4.5) | 0 | **0** |
| **Keep** `_IOWR(CIPHER_IOCTL_MAGIC, 9, ...)` macro in `cipher_ioctl.h:244` (ABI; T-W4.5) | 0 | **0** |
| **Keep** `cipher_pid_stats::sm_partition_mask/count` fields in `cipher_internal.h:198-199` (I-W4.4 + ABI mirror in tenant_snapshot ioctl) | 0 | **0** |
| **Update** kbuild/Makefile in `cipher_kmod/Kbuild` to drop `cipher_partition_allocator.o` | 1 line | **−1** |
| **Net** | | **≈ −522 LOC** |

### 5.2 I-W4.4 invariant resolution (advisor flag ADV-1)

Wave 5's I-W4.4 invariant as written:
> *"LP-8 cleanup leaves `cipher_pid_stats::sm_partition_mask`
> populated by CP 5.4 — ioctl nr 8 snapshot still returns valid masks."*

This is **wrong as written**. CP 5.4 does **not** call
`cipher_set_sm_partition_mask` anywhere. Grep across the kmod shows
**zero callers** of that function. The mask field has been BSS-zero for
every tenant since CP 5.4 close (because the writer in
`cipher_partition_writeback_cache` is only triggered from
`cipher_partition_request*`, which is only callable via ioctl nr 9,
which returns -ENOSYS).

**Resolution: option 1 ("vacuously satisfied") — VERIFIED SAFE.** The
field is already always zero. Post-deletion: still always zero. Snapshot
returns the same value pre-deletion as post-deletion: zero.

Advisor-flagged check (does any userspace consumer read non-zero?):
**no userspace consumer reads `sm_partition_mask`/`sm_partition_count`
at all** (cipher_rt_phase4 has only declarations in `cipher_rt_tenant.h`,
zero readers in any live .c/.cpp). Adopting option 1 is a positive
finding, not a workaround.

**Reformulated invariant for the result doc:** *"I-W4.4 (corrected):
`cipher_pid_stats::sm_partition_mask/count` were already always zero on
the live build (nr 9 -ENOSYS since CP 5.4 close → writeback path never
fires); LP-8 deletion preserves the zero-value behavior; ioctl nr 8
snapshot returns the same (zero) field values pre/post. No userspace
consumer depends on non-zero values."*

### 5.3 C4 fold-in

**Defer the C4 fix to the Step 4 commit message + result doc as a
one-line erratum** pointing at `WAVE_5_S5_5_VERIFICATION.md §C4`. No
code rename — the renamed entities are being deleted.

### 5.4 srcversion reload gate (advisor flag ADV-3)

**First kmod rotation since pre-Step-1.** Current loaded srcversion:
`F6B9227C41E5439BD8F1B02`. Step 4 cadence must include:

1. `make` in cipher_kmod → produces new .ko with new srcversion
2. DKMS reload (`dkms install` or equivalent) → swaps the loaded module
3. `cat /sys/module/cipher_kmod/srcversion` → verify the NEW srcversion
   is loaded (not the stale F6B9227C…)
4. **THEN** run CP 5.4 isolation 15/15 — against the freshly-loaded module
5. **THEN** run SC6 TinyLlama + Mistral-7B both arms

Without step 3, a kmod build could "succeed" but the test would still
run against the stale loaded module — invalid gate. Hard requirement.

### 5.5 Estimated Step 4 size

| component | LOC | time |
|---|---|---|
| LP-8 deletion | ~−522 | 1–1.5 h |
| C4 erratum note | +5 lines doc | 5 min |
| kmod build + DKMS reload + srcversion verify | — | 15 min |
| CP 5.4 isolation 15/15 | — | 5 min |
| SC6 TinyLlama vanilla + CIPHER | — | 5 min |
| SC6 Mistral-7B vanilla + CIPHER | — | 6 min |
| Commit + tag + result doc + memory update | — | 30 min |
| **Total** | | **~2.5–3 h** |

Matches scope-lock "minor (cleanup)" estimate.

### 5.6 Decision

**SCOPE-AS-WRITTEN** — Step 4 ships as ~520-LOC kmod deletion + C4
erratum note.

| spec headline option | match? |
|---|---|
| SCOPE-AS-WRITTEN: ~80 LOC kmod cleanup | **YES** (actual is ~520 LOC because it's a whole-file delete, larger than the spec's "~80 LOC" guess; the deletion is still a cleanup-only with no caller cascade — matches the spirit) |
| SCOPE-EXPANDED: caller cascade | NO — no cascade |
| SCOPE-COMPRESSED: C4 defers | C4 is paperwork-only in this step (the desired rename target is being deleted), so this category technically applies if "C4 defer to its own paperwork step" is preferred. **Recommendation: fold C4 erratum into Step 4 result doc** — it's a one-line citation. |
| ADJUDICATION-REQUIRED | NO — both load-bearing checks pass: I-W4.4 vacuously satisfied; no out-of-tree consumers of the EXPORT_SYMBOL_GPL surface |

### 5.7 Hard-stop gates Step 4 must clear

1. **Build clean** rc=0, warning delta ≤ 0 vs pre-Step-4 baseline.
2. **srcversion verified** to change post-reload (`cat /sys/module/cipher_kmod/srcversion` ≠ `F6B9227C41E5439BD8F1B02`).
3. **CP 5.4 isolation 15/15 byte-identical** vs Step 3 baseline.
4. **SC6 TinyLlama** vanilla 7/7 + CIPHER 7/7 bit-identical.
5. **SC6 Mistral-7B** vanilla 7/7 + CIPHER 7/7 bit-identical.
6. **ABI T-W4.5:** ioctl nr 9 still returns -ENOSYS (verified via a tiny
   userspace strace probe or direct ioctl call).
7. **Snapshot ABI I-W4.4:** ioctl nr 8 still returns valid `cipher_tenant_snapshot`
   bytes with `sm_partition_mask/count` = 0 (unchanged from pre-Step-4).

---

## Adjudication ask

**SCOPE-AS-WRITTEN.** Step 4 execution prompt may be drafted with this
scope. The 5.7 hard-stop gates are the close criterion. The corrected
I-W4.4 reformulation should be included in the Step 4 result doc so
future audits don't trip on the Wave 5 mistext.

**Recommended Step 4 commit message subject:**
`Week 4 Step 4: LP-8 allocator retirement (cipher_partition_allocator.c deleted) + W5 C4 erratum`

---

**Evidence read for this preflight:**
- `/home/ubuntu/cipher-fusion-evidence/CIPHER_LOGIC_AUDIT_WAVE_5_FUSION_PLAN.md` §494, §780-860, §1255 (Wave 5 LP-8 retirement plan)
- `/home/ubuntu/cipher-fusion-evidence/WAVE_5_S5_5_VERIFICATION.md` §C4 (the resolved Wave-5 naming erratum)
- `/home/ubuntu/cipher_kmod/cipher_partition_allocator.c` (482 LOC; all 9 functions enumerated)
- `/home/ubuntu/cipher_kmod/cipher_internal.h:198-199, 379-396`
- `/home/ubuntu/cipher_kmod/cipher_ioctl.h:140-141, 232, 244`
- `/home/ubuntu/cipher_kmod/cipher_dev.c:98-113, 265-266`
- `/home/ubuntu/cipher_kmod/cipher_main.c:51, 118`
- `/home/ubuntu/cipher_kmod/cipher_probe.c:184-225` (do_exit reaper)
- `/home/ubuntu/cipher_kmod/cipher_state_updater.c:170-180`
- `/home/ubuntu/cipher_kmod/cipher_tenant_snapshot.c:11-12, 70-71, 207-235`
- `/home/ubuntu/cipher_kmod/cipher_cp54_sched.c:200-202` (cp54_release_groups)
- `/home/ubuntu/cipher_kmod/cipher_kmod.mod.c:38-50` (KSYMTAB exports)
- `lsmod` (no out-of-tree consumer)
- `/sys/module/cipher_kmod/srcversion` (baseline `F6B9227C41E5439BD8F1B02`)
- cipher_rt_phase4 grep for `sm_partition_mask`/`cipher_partition_*` (zero live consumers)
