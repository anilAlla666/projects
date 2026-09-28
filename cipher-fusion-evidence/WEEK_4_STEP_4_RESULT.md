# Week 4 Step 4 — LP-8 Partition Allocator Retirement — RESULT

**Status: PASS.**

`cipher_partition_allocator.c` deleted (482 LOC) + 4 caller sites cleaned
+ ABI nr 9 preserved as -ENOSYS stub + Wave 5 §I-W4.4 invariant corrected
to its empirically-true form. **First kmod rotation since pre-Step-1
baseline.** srcversion verified changed `F6B9227C → 180A1D4...`. CP 5.4
isolation 15/15 byte-identical; SC6 TinyLlama + Mistral-7B bit-identical
across both arms; new kmod loaded cleanly with /dev/cipher mode 666
preserved. Net diff: 7 files / +19 / -528.

**Date:** 2026-05-21
**Phase:** v1.2.2 §7 Week 4, Step 4 of 6
**Anchors:**
  - cipher_rt_phase4: `3be4531` (tag `week-4-step-3-tier-b-ports`) — unchanged (Step 4 is kmod-only)
  - cipher_kmod: `a21a45e` → **`158ad96`** (tag `week-4-step-4-lp8-retired`)
  - cipher-may13-evidence: `fc8a9ae` (unchanged)
  - cipher-fusion-evidence: `d4ef628` (Step 2+3 audit tail commit) — this Step 4 result + commit-doc landings tracked separately

---

## A — Pre-edit verification + baselines — PASS

| signal | value |
|---|---|
| cipher_kmod HEAD | `a21a45e` ✓ matches Step 3 baseline |
| cipher_rt_phase4 HEAD | `3be4531` ✓ matches Step 3 close |
| Both trees clean | ✓ |
| Snapshot pre | `/tmp/week4_step4/cipher_kmod.ko.step4_pre` md5 `8401f31a` |
| Loaded srcversion pre | **`F6B9227C41E5439BD8F1B02`** (from `/sys/module/cipher_kmod/srcversion`) — written to `/tmp/week4_step4/srcversion_pre.txt` |
| DKMS status | `cipher-kmod/0.4.8 … installed` (DKMS-installed module is from May 16, srcversion `E427CAFA…`; **loaded module came from local build at `/home/ubuntu/cipher_kmod/cipher_kmod.ko`**, not from DKMS — this divergence existed pre-Step-4 and is flagged for follow-up) |
| Used by | 0 (rmmod safe) |
| /dev/cipher consumers | none (lsof/fuser empty) |
| sudo available | yes, no-prompt |
| Pre-edit CP 5.4 isolation | **15/15 PASS** (`/tmp/week4_step4/cp54_pre.log`) |
| Pre-edit SC6 TinyLlama vanilla | **PASS 7/7 bit-identical** (`/tmp/week4_step4/sc6_pre_vanilla.log`) |

---

## B — LP-8 deletion in cipher_kmod — PASS

| sub-step | action | result |
|---|---|---|
| **B.1** | `rm cipher_partition_allocator.c` | 482 LOC file deleted |
| **B.2** | check `.h` sibling | none exists (declarations were in `cipher_internal.h`) |
| **B.3** | Kbuild edit | removed `cipher_partition_allocator.o` from `cipher_kmod-y` |
| **B.4** | 4 caller sites cleaned | see table below |
| **B.5** | nr 9 -ENOSYS preserved | `case CIPHER_REQUEST_SM_PARTITION: return -ENOSYS;` (`cipher_dev.c:255-256`) |
| **B.6** | orphan `EXPORT_SYMBOL` refs | **0** in source (KSYMTAB in mod.c will regenerate at build) |
| **B.7** | `cipher_set_sm_partition_mask` callers | **still 0** (Wave 5 §I-W4.4 false-as-written verified) |

### B.4 caller-cleanup detail

| caller | action | replacement |
|---|---|---|
| `cipher_main.c:51` `cipher_partition_allocator_init()` | deleted | (none — slot array static storage went away with the .c file) |
| `cipher_main.c:118` `cipher_partition_allocator_exit()` | deleted | (none) |
| `cipher_state_updater.c:176` `cipher_partition_tick()` | deleted from 1 kHz kthread tick | comment explaining the legacy reap had nothing to reap (writer was nr 9, deactivated) |
| `cipher_probe.c:210` `cipher_partition_release_slots_only(pid)` | deleted from do_exit reaper | comment pointing at the live release path (`cipher_cp54_release(pid)` at L198) |
| `cipher_dev.c:107-113` `cipher_dev_request_sm_partition()` function | deleted (15 LOC) | inline `return -ENOSYS;` in the switch case statement |
| `cipher_internal.h:378-396` 8 function prototypes | replaced with single retirement-comment block | (none) |
| `cipher_main.c:49` `pr_info("loading (Phase 4.2 T4.2.1 — lock-free SM partition allocator)")` | updated | `pr_info("loading (Phase 4 — CP 5.4 8-SM-group arbitration; W4 Step 4 retired LP-8)")` |
| `cipher_dev.c:98-106` comment block above the case | rewritten | points at CP 5.4 (nr 13 CIPHER_CP54_ALLOCATE) as replacement |

---

## C — Build + DKMS reload + srcversion gate — PASS

| sub-step | result |
|---|---|
| **C.1** clean build | rc=0; **1 warning** (`the compiler differs from the one used to build the kernel` — generic kbuild noise unrelated to LP-8); 0 errors. *No kmod-build pre-baseline was captured this step (kmod wasn't rebuilt in Steps 1-3, so the kbuild log from Step A.3 is a cipher_rt_phase4 userspace build, not a kmod build); the framing "delta 0" cannot be claimed against a pre baseline that wasn't taken. The 1 post-warning is structural kbuild noise and is the only warning the kmod compile produces.* |
| **C.2** `cipher_kmod.ko` built | md5 `15a0d50dc846dfffd1094bf94b88e41b` (size 2,435,048 B) |
| **C.2b** `modinfo` srcversion | **`180A1D412429A77A73B464D`** — differs from pre `F6B9227C…` ✓ |
| **C.3** reload procedure (local-build pattern — DKMS source not updated this step) | `sudo rmmod cipher_kmod` (rc=0, clean per dmesg: `kthread stopping`, `kprobes detached alloc_failures=0 reaped=504`, `unloaded cleanly`) → `sudo insmod /home/ubuntu/cipher_kmod/cipher_kmod.ko` (rc=0) |
| **C.4** loaded srcversion | **`180A1D412429A77A73B464D`** (saved to `/tmp/week4_step4/srcversion_post.txt`) — matches the new build ✓ |
| **Load-bearing srcversion gate** | **`F6B9227C41E5439BD8F1B02` → `180A1D412429A77A73B464D` (CHANGED)** — PASS |

### Post-reload dmesg (clean banners)

```
cipher_state_updater: kthread stopping
cipher_kmod: kprobes detached, alloc_failures=0 reaped=504
cipher_kmod: unloaded cleanly
cipher_kmod: loading (Phase 4 — CP 5.4 8-SM-group arbitration; W4 Step 4 retired LP-8)
cipher_kmod: CP 5.4 arbitration ledger — 15 × 8-SM groups (120 SMs); legacy nr-9 4-SM allocator deactivated
cipher_kmod: Track 2 SC5 weight-arena registry — 16 slots, 5s liveness reaper
cipher_kmod: CP 3.3 FLOP telemetry ready
cipher_kmod: /dev/cipher ready (major=511, REGISTER_TENANT + GPU_STATE/PROCESS_UTIL/LAUNCH_STATS live)
cipher_kmod: /dev/cipher_kvdedup ready
cipher_kmod: kprobes attached at 0xffffffffc05f6300; do_exit reaper at 0xffffffffa2b0cfc0
cipher_kmod: loaded ok; nvidia_unlocked_ioctl hooked
cipher_state_updater: kthread starting (cadence 1000 Hz)
```

`/dev/cipher` mode **`666`** preserved per `[[cipher-devnode-codified]]`.

### DKMS-state divergence — flagged AND resolved post-commit

**Pre-sync state:** `dkms status cipher-kmod` showed `0.4.8 … installed`,
but the **DKMS-installed `.ko`** at
`/lib/modules/.../updates/dkms/cipher_kmod.ko` had srcversion
`E427CAFA4E94D548233DC7A` (from a May-16 build) — **not** the post-Step-4
build. The reload procedure used `insmod /home/ubuntu/cipher_kmod/cipher_kmod.ko`
directly (the actual cadence on this pod since pre-Step-1).

**Sync executed 2026-05-21 post-commit-158ad96** (advisor follow-up; user
adjudicated "sync now"):
1. `sudo dkms remove cipher-kmod/0.4.8 --all` — removed stale entry
2. `sudo rm -rf /usr/src/cipher-kmod-0.4.8` + recreated with current
   sources (.c .h Kbuild Makefile only; no build artifacts / no .git)
3. `dkms.conf` recovered from
   `/tmp/cipher-deb-build/cipher-platform/usr/src/cipher-kmod-0.4.8/dkms.conf`
   (CP 2.5 .deb-build artifact)
4. `sudo dkms add/build/install cipher-kmod/0.4.8` — clean rebuild + install
5. **Verified:** `modinfo /lib/modules/$(uname -r)/updates/dkms/cipher_kmod.ko | grep srcversion`
   reports `180A1D412429A77A73B464D` — **matches loaded** ✓.

**Post-sync state:** DKMS-installed and loaded module srcversions are now
identical (`180A1D412429A77A73B464D`). **Pod reboot is now safe** — boot
will load the LP-8-retired module. The follow-up flagged in the original
preflight is closed.

---

## D — ABI preservation checks — PASS

Probe binary: `/tmp/week4_step4/abi_probe2.c` (compiled with real
`cipher_ioctl.h` headers; v1 used incorrect struct mirrors and got
ENOTTY from size-mismatch in `_IOWR`). Output:
`/tmp/week4_step4/abi_probe2_output.log`.

| check | result |
|---|---|
| **D.1** nr 9 `CIPHER_REQUEST_SM_PARTITION` | rc=-1 errno=**38 (ENOSYS)** ✓ — ABI preserved per T-W4.5 |
| **D.2a** REGISTER_TENANT (sanity) | rc=0 (snapshot has a fresh tenant to read back) |
| **D.2b** nr 8 `CIPHER_GET_TENANT_SNAPSHOT` | rc=0; `pid=1851556 tenant_id_str=step4_d_probe`; **`sm_partition_mask=0x00000000 sm_partition_count=0`** ✓ — I-W4.4 corrected invariant verified empirically (fields are BSS-zero as expected) |
| **D.3** all other ioctl nrs preserved | verified via `grep -nE 'case CIPHER_'` in `cipher_dev.c` — switch dispatches unchanged for nrs 1-7 (REGISTER + SUBMIT_* + SNAPSHOT), 10-12 (SET_CLOCK_MHZ, FLOP family), 13-20 (CP 5.4 family), 21-24 (ARENA family), 25-26 (PUSH_CLASSIFY_STATS, DSM_PROPOSE), Phase-6-reserved (SNAPSHOT/RESET/GET_VERSION → -ENOSYS) |

---

## E — SC6 + CP 5.4 regression gates — PASS

| arm | result | log |
|---|---|---|
| **E.1 CP 5.4 isolation** | **15/15 PASS byte-identical** to pre | `/tmp/week4_step4/cp54_post.log` (`diff cp54_{pre,post}.log` = empty) |
| **E.2a SC6 TinyLlama vanilla** | **7/7 PASS bit-identical** | `/tmp/week4_step4/sc6_post_tinyllama_vanilla.log` |
| **E.2b SC6 TinyLlama CIPHER** | **7/7 PASS bit-identical** | `/tmp/week4_step4/sc6_post_tinyllama_cipher.log` |
| **E.3a SC6 Mistral-7B vanilla** | **7/7 PASS bit-identical** | `/tmp/week4_step4/sc6_post_mistral_vanilla.log` |
| **E.3b SC6 Mistral-7B CIPHER** | **7/7 PASS bit-identical** | `/tmp/week4_step4/sc6_post_mistral_cipher.log` |

All 7 invariants of the SC6 7-check matrix held across all 4 SC6 arms:
`all4_fwd1_bit_identical`, `all4_fwd2_bit_identical_after_producer_death`,
`all4_arena_backed`, `arena_survived_producer_sigkill`,
`producer_pid_cleared`, `all4_consumers_held_arena`,
`arena_reaped_after_last_participant`.

---

## F — Commit + tag — PASS

| signal | value |
|---|---|
| F.1 commit | **`158ad96093f98c2dbb96c6eb29028832e8d23031`** |
| F.1 author | Anil &lt;anil.0666369@gmail.com&gt; (per-commit via `git -c`, no global config) |
| F.1 message | "Week 4 Step 4: LP-8 partition allocator retirement" |
| F.1 diff stat | **7 files changed, +19 / -528** |
| F.2 tag | **`week-4-step-4-lp8-retired`** ✓ points at HEAD |
| F.3 fallback `.ko`s preserved | `cipher_kmod_fallback/cipher_kmod.ko.pre_w4_step4` (`8401f31a`) + `cipher_kmod_fallback/cipher_kmod.ko.w4_step4_lp8_retired` (`15a0d50d`) |

### Files committed (7)

```
Kbuild                       -1   (cipher_partition_allocator.o removed)
cipher_dev.c                +6/-16  (handler deleted, case inlined, comment updated)
cipher_internal.h           +5/-18  (8 prototypes → 1 retirement comment)
cipher_main.c               +1/-3   (init+exit calls dropped, banner updated)
cipher_partition_allocator.c -482  (full file deleted)
cipher_probe.c              +5/-5   (release_slots_only call dropped, comment updated)
cipher_state_updater.c      +3/-2   (tick call dropped, comment updated)
```

---

## G — Memory update — PASS

| step | action |
|---|---|
| **G.1** | wrote `~/.claude/projects/-home-ubuntu/memory/week4-step4-pass.md` as new ACTIVE pointer |
| **G.2** | flipped `~/.claude/projects/-home-ubuntu/memory/week4-step3-pass.md` header to HISTORICAL with forward-link to `[[week4-step4-pass]]` |
| **G.3** | updated `~/.claude/projects/-home-ubuntu/memory/MEMORY.md` index — Step 3 line set to "(closed)", Step 4 line appended as tail breadcrumb |

---

## Wave 5 corrections (paperwork, captured in commit + this doc)

### I-W4.4 invariant reformulation

**Wave 5 §831 said:** *"LP-8 cleanup leaves `cipher_pid_stats::sm_partition_mask`
populated by CP 5.4 — ioctl nr 8 snapshot still returns valid masks."*

**Empirical reality discovered during preflight:** `cipher_set_sm_partition_mask`
has **zero callers** in the kmod (CP 5.4 was never wired into it). The
mask field has been BSS-zero for every tenant since CP 5.4 deactivated
nr 9 (because the only writer path —
`cipher_partition_writeback_cache` from `cipher_partition_request*` —
was unreachable). Post-Step-4: still BSS-zero. Same value pre/post.

**Empirical confirmation (D.2):** `abi_probe2` registered as
tenant `step4_d_probe` (pid 1851556) and read back via nr 8 snapshot:
`sm_partition_mask=0x00000000 sm_partition_count=0`. The "valid masks"
condition is satisfied vacuously — the values are valid (zero), just
not populated. No userspace consumer reads non-zero values (verified
via grep of cipher_rt_phase4 for `sm_partition_mask`/`_count`: only
struct-field declarations in `cipher_rt_tenant.h:53-54`, zero live
readers).

**Corrected invariant:** *"I-W4.4 (corrected, W4 Step 4 2026-05-21):
`cipher_pid_stats::sm_partition_mask/count` were already always zero on
the live build because nr 9 has been -ENOSYS since CP 5.4 close → the
writeback path (`cipher_partition_writeback_cache`) never fired. LP-8
deletion preserves the zero-value behavior. ioctl nr 8 snapshot
returns the same (zero) field values pre/post. No userspace consumer
depends on non-zero values."*

### C4 cipher_partition_slot[32] misnaming

**Wave 5 §812 said:** *"Remove the slot array `cipher_partition_slot[32]`
and all writes to it."*

**Reality:** No on-disk variable named `cipher_partition_slot[32]`
existed. The type was `struct cipher_partition_slot`
(`cipher_partition_allocator.c:88`); the array variable was
`cipher_slots[CIPHER_PARTITION_SLOTS_MAX]` (`:91`); the size macro was
`CIPHER_PARTITION_SLOTS_MAX = 32`. The pre-existing
`WAVE_5_S5_5_VERIFICATION.md` §C4 already documented this.

**Resolution:** **MOOTED by file deletion.** After Step 4, neither
`struct cipher_partition_slot` nor `cipher_slots` nor
`CIPHER_PARTITION_SLOTS_MAX` exists in the source tree. Wave 5 §812's
text remains incorrect but now points at a deleted entity, like a stale
link. The Wave 5 audit can cite this commit + §C4 of S5.5 as the
canonical resolution.

---

## Honest notes

1. **DKMS divergence was real and pre-existing — now resolved.** Loaded
   module came from local build (`/home/ubuntu/cipher_kmod/cipher_kmod.ko`),
   not from DKMS. DKMS-installed `.ko` was May-16 vintage (srcversion
   `E427CAFA…`). Step 4 originally preserved this cadence — reloaded the
   local build. A boot would have re-loaded the stale May-16 DKMS build
   (which still had LP-8) → Step 5's planned kmod additions would have
   landed against an LP-8-present base. **Advisor flagged this as
   blocking-ish for Step 5; user chose option (a) "sync DKMS now"; sync
   executed post-commit-158ad96 (see §C above). DKMS and loaded module
   are now both at srcversion `180A1D412429A77A73B464D`.**

2. **One warning, not LP-8-related — but I don't have a kmod-build
   pre-baseline to call "delta 0".** Build report says `warning: the
   compiler differs from the one used to build the kernel`. This is a
   generic kbuild check (gcc version mismatch between kernel-build and
   module-build). I did NOT rebuild the kmod during Steps 1-3, so I
   don't have a pre-Step-4 kmod build log to compare against. The
   "delta 0" framing in earlier sections was sloppy. Defensible
   framing: **post-build has 1 structural kbuild warning, no LP-8-
   related warnings, and the .c file deletion can only remove
   warnings, not add them** — the warning-count cannot have gone up
   on a deletion commit.

3. **Net diff math.** +19/-528 ≈ -509 LOC net. Preflight estimated ~522
   LOC; close enough (the comment-replacement blocks added back ~12 lines
   of explanatory text that the bare-count estimate didn't include).

4. **Snapshot of nr 8 returns zero — by design now.** Userspace consumers
   that expect non-zero `sm_partition_mask` from snapshot are looking at
   the wrong field for SM-partition state — they should be querying CP
   5.4's per-tenant group ledger (cipher_cp54_ioctl_query, nr 15) for
   the actual partition assignment. The legacy field is preserved for
   ABI stability but is permanently zero.

5. **abi_probe2 vs v1.** First probe attempt used hand-mirrored structs
   with wrong sizes; `_IOWR` encoded the wrong ioctl nr; kernel rejected
   with ENOTTY. v2 includes the real `cipher_ioctl.h` headers and works
   correctly. Both probes preserved in `/tmp/week4_step4/` for the audit
   trail.

---

## Goals enabled / preserved (substrate hygiene)

| layer | status |
|---|---|
| **CP 5.4** (15 × 8-SM groups, nrs 13-15) | live; 15/15 isolation byte-identical |
| **CP 5.4 migration** (nrs 16-20) | unchanged dispatch |
| **Track 2 SC5** (arena nrs 21-24) | unchanged dispatch |
| **Week 2 PUSH_CLASSIFY_STATS** (nr 25) | unchanged dispatch |
| **Week 3 DSM_PROPOSE** (nr 26) | unchanged dispatch |
| **LP-8 (nr 9)** | retired; case statement returns -ENOSYS inline; ABI nr preserved |
| **Phase 6 reserved** (SNAPSHOT/RESET/GET_VERSION) | unchanged -ENOSYS |
| **/proc/cipher/*** | unchanged (no /proc node owned by LP-8) |
| **CP 3.3 FLOP telemetry** | unchanged |
| **/dev/cipher mode** | 666 preserved |

---

## Adjudication ask

Step 4 closed cleanly. **Step 5 (Prometheus exporter additions) is
unblocked** per `WEEK_4_SCOPE_LOCK.md` §2. Step 5 entry pre-flight
should:
1. Confirm the Prometheus exporter surface (per Wave 5 §LP-something +
   section 7 verbatim — small ~50 LOC adds for the new observability
   metrics).
2. Verify the exporter wires into existing `/proc/cipher/*` patterns
   without ABI surface changes.
3. Check whether Step 5 touches kmod (likely userspace-only, but
   confirm).

**Rollback:**
```
git -C /home/ubuntu/cipher_kmod reset --hard week-3-step-4-opt2a-dsm-propose
sudo rmmod cipher_kmod
sudo insmod /home/ubuntu/cipher_kmod_fallback/cipher_kmod.ko.pre_w4_step4
cat /sys/module/cipher_kmod/srcversion   # should report F6B9227C41E5439BD8F1B02
```

---

**Evidence:**
- `/tmp/week4_step4/cipher_kmod.ko.step4_pre` (pre snapshot, md5 `8401f31a`)
- `/tmp/week4_step4/srcversion_{pre,post}.txt`
- `/tmp/week4_step4/cp54_{pre,post}.log` (15/15 PASS both, byte-identical)
- `/tmp/week4_step4/sc6_pre_vanilla.log`
- `/tmp/week4_step4/sc6_post_tinyllama_{vanilla,cipher}.log` (PASS both)
- `/tmp/week4_step4/sc6_post_mistral_{vanilla,cipher}.log` (PASS both)
- `/tmp/week4_step4/build.log` (clean rc=0)
- `/tmp/week4_step4/abi_probe2.c` + `abi_probe2_output.log` (D-gate PASS)
- `/home/ubuntu/cipher_kmod_fallback/cipher_kmod.ko.{pre_w4_step4,w4_step4_lp8_retired}` (fallback .kos)
- `/home/ubuntu/cipher-fusion-evidence/WEEK_4_STEP_4_PREFLIGHT.md` (the pre-flight that adjudicated SCOPE-AS-WRITTEN)
- `git show 158ad96` (the LP-8 retirement commit)
