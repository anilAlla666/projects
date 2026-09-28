# CP 5.4 — Step 1.2 RE-ATTEMPT: kmod reload + isolation + regression — REPORT

**Date:** 2026-05-18. **Verdict: PASS** — Phase 1 PASS / Phase 2 PASS (15/15) /
Phase 3 PASS. The one-line `cipher_probe.c` reaper-placement bug from the first
attempt is fixed and empirically verified by the **unchanged** isolation suite.
Kmod anchor rotates **`e2f50452` → `7f467de4`**.

---

## Context

The first Step 1.2 attempt (`CP_5_4_STEP_1_2_REPORT.md`) failed Phase 2: the
`do_exit` reaper early-returned for any pid with no `cipher_pid_stats` entry, so
`cipher_cp54_release` (wired *below* that guard in Step 1.1) never ran for a
process that ALLOCATEd CP 5.4 groups without ever calling `REGISTER_TENANT` →
permanent SM-group leak (Test 4), contaminating Tests 5/6. Rolled back to
`e2f50452` per procedure. The user authorized the one-line fix; the safety of
the placement move was verified structurally in `STEP_1_2_REATTEMPT_PREFLIGHT.md`.

## The fix (applied)

`cipher_probe.c:cipher_do_exit_pre()` — `cipher_cp54_release(pid)` moved
**above** the `if (likely(!e)) return 0;` fast-path guard, with an explanatory
comment. The legacy `cipher_partition_release_slots_only(pid)` stays below the
guard (nr-9 is deactivated; no unregistered legacy caller can exist). Rebuilt:

| build | md5 | note |
|---|---|---|
| first-attempt (broken) | `90103f82` | reaper bug — discarded |
| **re-attempt (fixed)** | **`7f467de4`** | this report's subject |
| legacy fallback | `e2f50452` | kmod 0.4.8, rollback target — preserved |

Loaded-module `srcversion` `2CF15040D2DBF9C724EF453`.

## Verdict table

| phase | result |
|---|---|
| Phase 0 — pre-reload baseline | reused (`PRE_RELOAD_BASELINE.md`, still valid — no kmod change since) |
| Phase 1 — kmod reload | **PASS** — `e2f50452` → `7f467de4` reloaded clean |
| Phase 2 — isolation tests | **PASS** — 15/15, unchanged test binary |
| Phase 3 — regression | **PASS** — W1/W2/W3 all within ±3% of baseline |
| anchor | **rotates `e2f50452` → `7f467de4`** |

## Phase 1 — reload (PASS)

Pre-`rmmod` checks: `cipher_kmod` refcount 0, no open `/dev/cipher` fds
(`lsof`), `nvidia-smi` compute-apps empty (GPU 0 % / 0 MiB), `cipher-exporter`
stopped first (user-authorized, as in the first attempt). `rmmod cipher_kmod`
detached clean (`alloc_failures=0 reaped=152`). `insmod
cipher_kmod/cipher_kmod.ko` (`7f467de4`). dmesg confirmed the CP 5.4 init line:

```
cipher_kmod: CP 5.4 arbitration ledger — 16 × 8-SM groups (128 SMs);
             legacy nr-9 4-SM allocator deactivated
```

In-memory size 94208, `/dev/cipher` re-created `crw-rw-rw-` (0666 devnode
callback intact). `cipher-exporter` restarted (pid 1602727, `/metrics` serving).
Reload mechanics: clean.

## Phase 2 — isolation tests (PASS — 15/15)

Harness `cp54_isolation_test` — **the exact binary from the first attempt
(17:43), source unchanged**, which is the report's stated fix-verification
criterion ("if they pass without test edits, the diagnosis is complete"). Full
output captured to `isolation_reattempt.log`:

```
Test 1 — legacy nr-9 deactivated:        PASS
Test 2 — ALLOCATE / FREE / QUERY:        PASS ×4
Test 3 — pool resize:                    PASS ×4
Test 4 — do_exit reaper:                 PASS   <-- the bug, now fixed
Test 5 — disjointness:                   PASS ×3
Test 6 — concurrent stress:              PASS ×2
=== Phase 2 result: 15 PASS, 0 FAIL ===
```

**Test 4** — the first attempt's failure — now passes: the forked child opens
`/dev/cipher`, `CIPHER_CP54_ALLOCATE`s 24 SMs (3 groups), and `_exit()`s
**without FREE** and **without `REGISTER_TENANT`**; the `do_exit` reaper
reclaimed all 3 groups. **Tests 5 and 6** (which the first attempt failed only
via downstream contamination from Test 4's leak) now pass their "all 16 groups
free" assertions — confirming the leak was the single upstream fault and the
diagnosis was complete. dmesg during the suite: only the expected benign
`REQUEST_SM_PARTITION (nr 9) deactivated` line (Test 1 exercises nr-9) — no
`pr_err`/`pr_warn`.

## Phase 3 — regression (PASS)

Captured on the fixed kmod `7f467de4` immediately after the re-attempt build
(17:55–17:58). These ran on a prior load instance of the **byte-identical**
`7f467de4` .ko (regression behaviour is binary-determined; the 18:07 reload
loaded the same file) — they are valid evidence for the anchor. Gates from
`PRE_RELOAD_BASELINE.md`, applied mean-vs-mean.

**W1 — Phase B N=8 TinyLlama cross-tenant batched (3 reps)**
(`cp_5_6/phase_b/session1/arm3_postreload_rep{1,2,3}`)

| rep | agg tok/s | mean W | tok/W |
|---|---|---|---|
| 1 | 485.14 | 137.68 | 3.524 |
| 2 | 499.17 | 138.92 | 3.593 |
| 3 | 495.81 | 139.09 | 3.565 |
| **mean** | **493.37** | **138.56** | **3.561** |

Baseline mean tok/s 491.8 (±3% [477.0, 506.5]) → **493.37 PASS**; baseline mean
tok/W 3.54 (±3% [3.434, 3.646]) → **3.561 PASS**. Total tokens 4224/rep (= 8
tenants complete).

**W2 — CP 5.3 STEP 2B Arm-c F1-fixed spec-decode** (`cp_5_3/step2b/arm_c.json`)

`accept_rate` **0.37963** (41/108) — *bit-identical* to baseline 0.37963;
gate ±3% [0.368, 0.391] → **PASS**. 22 rounds, decode text coherent and
identical to baseline ("…63 million square miles. It is bordered by Asia and
Australia…"). `wall_s` 4.52 vs 4.56 baseline.

**W3 — Track 2 SC2 VMM weight arena** (`phase_c/sc2_result.json`)

All 5 gates PASS (`gate_1_vmm_backed` … `gate_5_export`), teacher-forced
`kl_max` **0.0**, `SC2_PASS` true — identical to baseline. Binary gate → **PASS**.

**dmesg regression check:** across the whole kernel ring buffer, every
`cipher_kmod` reaper detach reports `alloc_failures=0`; there are no `pr_err`,
`pr_warn`, "leak", "bug", "oops", or "NULL"/"panic" lines from any cipher
module. No regression signal.

## Anchors & artifacts

- **kmod anchor rotates: `e2f50452` → `7f467de4`.** `7f467de4` is the working
  CP 5.4 kmod (Step 1.1 ledger + slot→SM bridge + 3 ioctls + the Step 1.2
  reaper fix). This is §9's planned "rotates after Step 1" event; it lands at
  Step 1.2 because that is when the kmod is first verified.
- **libcipher_rt `a7ac8e97` — unchanged** (CP 5.4 does not touch it until
  Step 1.3). libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d` — unchanged.
- Fallbacks preserved **outside the kbuild dir** ([[cipher-kbuild-clean-wipes-ko]]):
  - `cipher_kmod_fallback/cipher_kmod.ko.cp5_4_step1` (`7f467de4`) — new anchor copy
  - `cipher_kmod_fallback/cipher_kmod.ko.pre_cp5_4` (`e2f50452`) — legacy rollback
  - `cipher_kmod_fallback/cipher_kmod_src_cp5_4_step1.tar.gz` — fixed kmod source
- Evidence: `isolation_reattempt.log`, `STEP_1_2_REATTEMPT_PREFLIGHT.md`,
  `PRE_RELOAD_BASELINE.md`, `cp54_isolation_test.c`, this report; Phase 3
  result JSONs under `cp_5_6/phase_b/session1/arm3_postreload_rep*`,
  `cp_5_3/step2b/arm_c.json`, `phase_c/sc2_result.json`.

## What the procedure caught

Step 1.1's structural assumption ("the existing reaper pattern extends
cleanly") was wrong; the empirical isolation suite — required by the user over
structural reasoning — caught it on a rolled-back-able kmod, not in production.
The fix verified on the **unchanged** test, so the diagnosis is closed: the
reaper-guard leak was the only fault.

## Adjudication ask

Step 1.2 PASS — kmod `7f467de4` is verified and loaded; anchor rotated.
Per campaign discipline (one atomic STEP, WAIT for adjudication), **stopping
here.** Recommend: authorize **Step 1.3** — libcipher_rt client integration
(kmod-driven variable green ctx; `qos_class`; batch-executor pool registration
+ launch-stream binding; libcipher_rt anchor `a7ac8e97` rotates on that step).
Or direct otherwise.
