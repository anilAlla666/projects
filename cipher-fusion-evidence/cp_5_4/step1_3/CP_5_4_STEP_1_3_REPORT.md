# CP 5.4 — Step 1.3: libcipher_rt client integration — CLOSURE REPORT

**Date:** 2026-05-18. **Verdict: Step 1.3 COMPLETE — all four phases PASS.**
The green context is now driven by the CP 5.4 kmod group ledger. Two anchors
rotated, both verified: kmod `7f467de4` → `8d777dfb`, libcipher_rt
`a7ac8e97` → `ebc0baaa`. POOL/executor wiring is the descoped Step 1.3b'.

---

## Phase results

| phase | scope | verdict |
|---|---|---|
| A | kmod correction `CIPHER_CP54_NUM_GROUPS` 16 → 15 | **PASS** — isolation 15/15, W1 regression in ±3%, dmesg clean |
| B | re-run Q3 split-order probe on the corrected kmod | **PASS** — 15 groups, determinism holds (`%smid` identical to 1.3a) |
| C | libcipher_rt client build | **PASS** — clean build, symbol diff = −5 ARB +1 cp54 |
| D | integration tests A–F | **PASS** — all six |

## What Step 1.3 delivered

**Kmod (`8d777dfb`, Phase A).** `CIPHER_CP54_NUM_GROUPS` 16 → 15 — the CP 5.4
Step 1.3a probe measured `cuDevSmResourceSplitByCount(minCount=8)` on this
H100 as 15 8-SM groups + 12 remainder, not 16 + 4. One constant + comments; no
ABI change. The phantom group 15 is gone.

**libcipher_rt (`ebc0baaa`, Phase C).** Three sub-components, each compiled in
isolation before the next:
1. `cipher_rt_green_ctx.c` — green context is kmod-driven. New
   `cipher_rt_green_ctx_cp54_init()` issues `CIPHER_CP54_ALLOCATE` (nr 13) and
   caches the granted `grp_mask`; `ensure()` selects the set-bit groups from
   `cuDevSmResourceSplitByCount` and builds a **variable-size** green context.
   The `pid ^ tenant_handle` hash-pick is retained only as the ALLOCATE-failure
   fallback.
2. `cipher_inject.c` — `cp54_init()` is called at injection-init, on the
   long-lived `pthread_once` thread, so the kmod ledger entry's lifetime tracks
   the process (Q4).
3. ARB retired — the 30 s poll thread drove ioctl nr 9 (now `-ENOSYS`); its
   output had zero consumers. `cipher_rt_arbitrate.o` dropped from the build.

**qos_class via env (Q2).** `CIPHER_QOS_CLASS` (partition/shared/pool, default
shared) + `CIPHER_SM_COUNT`. `libcipher_v2` (`86618c30`) is **unchanged** — the
kmod's CP 5.4 ledger takes qos directly through the ALLOCATE ioctl, so no
`REGISTER_TENANT` ABI change was needed.

## Verification highlights (Phase D)

- **Test A (SHARED):** ALLOCATE returns an empty mask → no green context → runs
  on the primary context. KL vs `a7ac8e97` baseline = 0.000, gen ids match.
- **Test B (PARTITION 16):** ALLOCATE → `grp_mask=0x0003` (2 groups);
  variable-size green context verified at 16 SMs; `cp54_query` confirms
  `n_partitions=1 free_grp_count=13`. KL = 0.000.
- **Test C (reaper):** a PARTITION tenant's groups are reclaimed by the do_exit
  reaper on exit (`free_grp_count` returns to 15) and cleanly reused.
- **Tests D/E/F:** dmesg clean; SC2 6/6 gates PASS (KL_max 0.0); W1 Phase B
  regression within ±3% (504.3 tok/s, 3.630 tok/W).

The split-order determinism probe (Phase 1.3a, re-validated Phase B) certifies
the grp_mask bridge: kmod group g ↔ the same physical SMs in every process.

## Anchors

| anchor | before Step 1.3 | after | fallback |
|---|---|---|---|
| kmod | `7f467de4` | **`8d777dfb`** | `cipher_kmod_fallback/cipher_kmod.ko.cp5_4_step1_pre15fix` (`7f467de4`), `.pre_cp5_4` (`e2f50452`) |
| libcipher_rt | `a7ac8e97` | **`ebc0baaa`** | `cipher_rt_phase4/` + `cipher_rt_fallback/libcipher_rt.so.pre_cp5_4_step3` (`a7ac8e97`) |
| libcipher_v2 | `86618c30` | `86618c30` (unchanged) | — |
| cipher_kv_bridge | `fca6843d` | `fca6843d` (unchanged) | — |

`cipher_anchors_manifest.txt` updated. Both rotations are post-verification
(kmod after Phase A isolation; libcipher_rt after Phase D).

## Scope deferred

- **Step 1.3b'** — POOL / batch-executor green-context binding. Descoped from
  Step 1.3 by Q1 adjudication: the Phase B executor loads `libcipher_v2.so`
  (no green-ctx code), so binding its launch stream to a pool green context is
  separate work. The libcipher_rt POOL path is written (a POOL ALLOCATE builds
  a green ctx over the granted mask) but untested — no POOL tenant exists yet.
- **Step 1.4** — confined-pool batch-lift curve — gates on Step 1.3b'.

## Adjudication ask

Step 1.3 is complete and verified; both anchors rotated. Per campaign
discipline (one atomic step, WAIT for adjudication), **stopping here.**
Recommend: authorize **Step 1.3b'** — POOL/executor green-context binding
(the Q1-descoped item) — which Step 1.4 depends on. Or direct otherwise.
