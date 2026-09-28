# CP 5.4 — Step 1.2: kmod reload + isolation + regression — REPORT

**Date:** 2026-05-18. **Verdict: Phase 1 PASS / Phase 2 FAIL / Phase 3 not
reached.** One real kmod bug, caught by the isolation suite; **rolled back to
`e2f50452`**. Anchors unchanged. The bug is a one-line fix; re-attempt is the
user's call.

---

## Verdict

| phase | result |
|---|---|
| Phase 0 — pre-reload baseline | done (`PRE_RELOAD_BASELINE.md`) |
| Phase 1 — kmod reload | **PASS** — `e2f50452` → `90103f82` reloaded clean |
| Phase 2 — isolation tests | **FAIL** — 12 PASS / 3 FAIL (1 real bug, 2 downstream) |
| Phase 3 — regression | **not reached** (Phase 2 gate) |
| rollback | **complete** — `e2f50452` reloaded, exporter restored |

## Phase 1 — reload (PASS)

Pre-reload checks: GPU clear, no `/dev/cipher` fds, refcount 0, fallback md5
`e2f50452`, new kmod md5 `90103f82` — all clean (the `cipher-exporter`
monitoring daemon was stopped first, user-authorized). `rmmod cipher_kmod`
(unloaded cleanly, `reaped=5500 alloc_failures=0`) → `insmod
cipher_kmod.ko` (`90103f82`). dmesg confirmed the CP 5.4 init line:

```
cipher_kmod: CP 5.4 arbitration ledger — 16 × 8-SM groups (128 SMs);
             legacy nr-9 4-SM allocator deactivated
```

Module size 81920 → 94208, `/dev/cipher` re-created 0666. Reload mechanics:
clean.

## Phase 2 — isolation tests (FAIL: 12 PASS / 3 FAIL)

Harness `cp54_isolation_test.c` — direct `ioctl()` on `/dev/cipher`, distinct
PIDs via `fork()`.

```
Test 1 — legacy nr-9 deactivated:        PASS
Test 2 — ALLOCATE/FREE/QUERY:            PASS (×4)
Test 3 — pool resize:                    PASS (×4)
Test 4 — do_exit reaper:                 FAIL
Test 5 — disjointness:                   PASS ×2, FAIL ×1
Test 6 — concurrent stress:              PASS ×1, FAIL ×1
```

The CP 5.4 ledger mechanics that were exercised **work**: nr-9 returns
`-ENOSYS`; ALLOCATE/FREE/QUERY round-trip; pool resize (POOL holds 16 →
PARTITION takes 2 → pool 14); two partitions get **disjoint** masks
(`m1 & m2 == 0`); concurrent 4×5 ALLOCATE/FREE does not crash. The failures
are all one bug.

## Root cause — the `do_exit` reaper never reaches the CP 5.4 release

**Test 4** forks a child that opens `/dev/cipher` and calls
`CIPHER_CP54_ALLOCATE` (24 SMs → 3 groups), then `_exit()`s **without FREE** —
the reaper must reclaim. It did not.

`cipher_probe.c:cipher_do_exit_pre()`:

```c
e = cipher_pid_lookup_rcu(pid);
...
if (likely(!e)) return 0;          /* <-- early return */
...
cipher_partition_release_slots_only(pid);
cipher_cp54_release(pid);          /* <-- Step 1.1 placed it HERE */
```

The reaper **early-returns for any pid with no `cipher_pid_stats` entry** —
i.e. any process that never called `CIPHER_REGISTER_TENANT` or submitted
telemetry. Step 1.1 wired `cipher_cp54_release(pid)` *after* that early
return. Test 4's child only does `open` + `ALLOCATE` — it never
`REGISTER_TENANT`s — so it has no `cipher_pid_stats` entry, the reaper returns
at the guard, and `cipher_cp54_release` never runs. **The child's 3 groups
leak permanently** (held by a dead pid).

### Why Tests 5 and 6 also failed — downstream contamination

Tests 5/6 are **not independent failures** — they inherit Test 4's 3-group
leak. After Test 4: only **13** of 16 groups free.
- **Test 5**: its two PARTITION-2 holders allocate and *explicitly FREE* fine
  ("each get 2 groups" ✓, "disjoint" ✓) — only the "all 16 free after release"
  check fails, because 13 ≠ 16.
- **Test 6**: 4×5 concurrent ALLOCATE/FREE runs without crashing ("no crash"
  ✓) — only "all 16 free, 0 partitions" fails, again 13 ≠ 16.

The FAILs are *exclusively* the free-count assertions; every mechanic check
passed. That pattern is the signature of a single upstream leak, not three
bugs.

### Why Test 3 passing is not contradictory

Test 3's POOL holder is also a non-registered child — but it is released via
an **explicit `CIPHER_CP54_FREE`** (pipe-triggered), not via the reaper. Test 3
therefore exercises FREE, not the reaper, and is silent on this bug. FREE
works; the reaper path is the defect.

## This is a real bug, not a test artifact

In production every tenant registers via `libcipher_v2` (`REGISTER_TENANT`) and
*would* have a `cipher_pid_stats` entry — so the reaper would fire. But:

- The CP 5.4 V1 memo §7 promised **"full crash recovery via the do_exit
  reaper."** As built, that recovery is silently conditional on the tenant
  *also* having called a *separate* ABI (`REGISTER_TENANT`). A tenant that
  ALLOCATEs groups and then crashes during early init — before it registers —
  **leaks its SM groups permanently.** That is a real crash-safety hole.
- The CP 5.4 ledger's correctness must not depend on a different ABI being
  invoked first. The coupling is fragile; the isolation test exposed it.

## The fix — one line

Move `cipher_cp54_release(pid)` (and its comment) in `cipher_do_exit_pre`
to **before** the `if (likely(!e)) return 0;` guard. `cipher_cp54_release` is
lock-free, idempotent, and a no-op for a pid that owns nothing (it scans 16
groups + 64 metadata slots and finds nothing) — it is safe to call
unconditionally on every `do_exit`. (The legacy `cipher_partition_-
release_slots_only` may stay where it is — nr 9 is deactivated, so no
unregistered legacy caller exists.)

**Test of the fix:** re-run `cp54_isolation_test.c` **unchanged** after the
fix — all 6 tests must pass, *including Tests 5 and 6*, with no change to the
test code. If they pass without test edits, the diagnosis is complete (the
leak was the only fault); if they need test edits, it was patched-around.

## Rollback (complete)

`rmmod` `90103f82` (refcount 0, no tenants) → `insmod` the preserved
`cipher_kmod.ko.pre_cp5_4` (`e2f50452`). Verified: module size back to 81920,
dmesg load line is the legacy `"lock-free SM partition allocator"` with **no**
"CP 5.4 arbitration ledger" line, `/dev/cipher` present 0666. The
`cipher-exporter` was restarted (`nohup … cipher-exporter.py`, pid 1599356,
listening `0.0.0.0:9402`) and confirmed reading `/proc/cipher/stats` against
the rolled-back kmod.

## Anchors & artifacts

- **kmod: unchanged — `e2f50452` loaded** (rollback target). The anchor does
  **not** rotate (Step 1.2 did not pass).
- **libcipher_rt `a7ac8e97` — unchanged** (never touched in Step 1.2).
- Fallback `cipher_kmod.ko.pre_cp5_4` (`e2f50452`) preserved, authoritative
  rollback target.
- The broken `90103f82` build remains at `/home/ubuntu/cipher_kmod/
  cipher_kmod.ko`; the fix-up `make` will overwrite it. Step 1.1 sources
  (`cipher_cp54_sched.c` etc.) are in place; the fix is the `cipher_probe.c`
  one-line move above.
- Artifacts: `cp54_isolation_test.c`, `PRE_RELOAD_BASELINE.md`, this report.

## Note

The user required empirical isolation tests rather than structural reasoning.
Step 1.1's structural assumption ("the existing reaper pattern extends
cleanly") was wrong — the reaper's `cipher_pid_stats` guard silently gates the
CP 5.4 release. The advisor's pre-reload review caught three races; the
isolation suite caught a fourth. **The procedure worked** — the bug surfaced
in a test, on a rolled-back-able kmod, not in production.

## Adjudication ask

Phase 2 failed → rolled back per procedure. Recommend: authorize the
one-line `cipher_probe.c` fix + rebuild + re-attempt Step 1.2 from Phase 1
(Phase 0 baseline still valid; isolation suite re-run unchanged is the
fix-verification). Or direct otherwise.
