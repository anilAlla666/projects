# CP 5.4 Step 1.3 — Phase A: kmod correction 16 → 15 — REPORT

**Date:** 2026-05-18. **Verdict: PASS.** The CP 5.4 kmod ledger is corrected
from a hard-coded 16 groups to the hardware-measured 15. Kmod anchor rotates
**`7f467de4` → `8d777dfb`**. All Phase A gates passed; isolation 15/15,
W1 regression within ±3%, dmesg clean.

---

## Why

Phase 1.3a (`PHASE_1_3A_PROBE.md`) measured `cuDevSmResourceSplitByCount(minCount=8)`
on this H100 80GB SXM5: **15 groups + 12-SM remainder**, not the 16 + 4 the
scope memo and the Steps 1.1–1.2 kmod assumed. The kmod's
`CIPHER_CP54_NUM_GROUPS = 16` had a phantom group 15 with no hardware backing
(`ALLOCATE(POOL)` would claim it; a client materialising it would index a
nonexistent split group). Founder adjudication: correct the kmod (Option 1).

## Source change

**`cipher_kmod/cipher_cp54_sched.c`** — one constant + comments:
- `#define CIPHER_CP54_NUM_GROUPS 16` → `15` (with a comment citing the probe).
- File-header comment: "16 × 8-SM groups (128 of 132 SMs — 4-SM remainder)" →
  "15 × 8-SM groups (120 of 132 SMs — 12-SM remainder)" + probe reference.
- §5 comment "The 16-group atomic array" → "15-group"; the `state[]` comment
  "16 × 8-SM groups" → "15".
- `static_assert(CIPHER_CP54_NUM_GROUPS <= 32, …)` — still holds (15 ≤ 32).
- The `cipher_cp54_sched_init` `pr_info` is parameterised on the macros, so the
  load line auto-updates to "15 × 8-SM groups (120 SMs)".

**`cipher_kmod/cipher_ioctl.h`** — comments only: the CP 5.4 block "16 × 8-SM
groups" → "15 × 8-SM groups (120 of 132 SMs — H100 hardware split)"; the
`grp_mask_out` field comment de-hardcoded.

No ABI change: struct layouts, ioctl nrs (13/14/15), and `grp_mask` width (u32)
all unchanged — 15 groups still fit. The legacy nr-9 path, the do_exit reaper,
and the partition allocator are untouched.

**Isolation test** — `step1_3/cp54_isolation_test.c` (copied from the Step 1.2
version, Step 1.2's original left intact). Assertion updates, each commented
with a `PHASE_1_3A_PROBE.md` reference: every "all 16 free" / pool-count
assertion 16 → 15; Test 3's post-shrink pool 14 → 13 (15 − 2). Per-group
arithmetic (8/16/24 SMs → 1/2/3 groups) unchanged.

## Build

`make` in `cipher_kmod/` — clean (only the pre-existing benign "compiler
differs" notice). New kmod **md5 `8d777dfb4c60d128b77c49a8082b9ee2`**,
srcversion `7A8E9D5E3EB8F0C904613D1`. `7f467de4` preserved as
`cipher_kmod_fallback/cipher_kmod.ko.cp5_4_step1_pre15fix` **before** the source
change.

## Reload

Exporter stopped; pre-`rmmod` checks clean (refcount 0, no `/dev/cipher` fds,
GPU clear). `rmmod` `7f467de4` → `insmod` `8d777dfb`. dmesg confirms:

```
cipher_kmod: CP 5.4 arbitration ledger — 15 × 8-SM groups (120 SMs);
             legacy nr-9 4-SM allocator deactivated
```

`/dev/cipher` re-created 0666; exporter restarted (pid 1604693).

## Verification

| gate | result |
|---|---|
| isolation re-verify (15-group assertions) | **15 PASS / 0 FAIL** (`isolation_phase_a.log`) |
| — Test 2 "all 15 free", Test 3 "pool 13", Tests 4/5/6 "all 15 free" | all PASS |
| W1 regression — 1 rep, Phase B N=8 TinyLlama batched | **PASS** |
| — agg tok/s | 495.34 (baseline 491.8; gate [477.0, 506.5]) ✓ |
| — tok/W | 3.593 (baseline 3.54; gate [3.434, 3.646]) ✓ |
| — teacher-forced KL_max (per round) | ≤ 5.5e-5 (gate ≤ 0.1) ✓ |
| dmesg | **clean** — no WARNING/BUG/Oops/error/failed/leak from any cipher module |

## Anchors

- **kmod `7f467de4` → `8d777dfb`** (rotated — Phase A passed). Loaded.
- Fallbacks (outside the kbuild dir, [[cipher-kbuild-clean-wipes-ko]]):
  `cipher_kmod.ko.cp5_4_step1_15fix` (`8d777dfb`),
  `cipher_kmod.ko.cp5_4_step1_pre15fix` (`7f467de4`),
  `cipher_kmod.ko.pre_cp5_4` (`e2f50452`), + `cipher_kmod_src_cp5_4_step1_15fix.tar.gz`.
- `libcipher_rt` `a7ac8e97`, `libcipher_v2` `86618c30`, `cipher_kv_bridge`
  `fca6843d` — unchanged.
- `cipher_anchors_manifest.txt` updated.

**Phase A PASS → proceeding to Phase B (re-run the Q3 probe on `8d777dfb`).**
