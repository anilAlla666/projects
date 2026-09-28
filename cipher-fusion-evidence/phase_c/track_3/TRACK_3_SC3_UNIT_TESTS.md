# Track 3 SC3-2 — UNIT + INTEGRATION TESTS

**Date:** 2026-05-19. **Under test:** libcipher_rt `83afd1ca` (SC3 build) +
kmod `98da2d1f` (SC2). Driver: `sc3_run.py` → `sc3_tenant.py` +
`cipher_migrate.py`. **Verdict: OVERALL PASS (7/7 + e2e-migrated).**

---

## Reachability gate (prerequisite)

`cp54_reachability_test.py` — does a Python tenant reach the *injected*
libcipher_rt instance? **PASS.** `CDLL(None)` FAILS (`undefined symbol` — the
CUDA driver loads the injection lib RTLD_LOCAL, symbols do not enter the global
namespace). `CDLL(path, RTLD_NOLOAD)` and `CDLL(path)` both resolve to the
injected instance — `is_initialized=1, sm_count=16`, identical symbol address
`0x7b04dd292e80`. The handler uses `RTLD_NOLOAD`. No fallback needed.
Evidence: `sc3_reachability_result.json`.

## Unit tests

| test | result | evidence |
|---|---|---|
| **self_migrate** — primitive in isolation | **PASS** | `migrate(0x3→0x6000)` rc=0; cur_mask 0x3→0x6000; 16 old SMs → 16 new SMs, **disjoint** |
| **verify_fault** — L1 verify fails | **PASS** | `CIPHER_SC3_FAULT=verify` → `migrate` rc=−1; cur_mask **unchanged** (0x3); SM set unchanged. Log: *"L1 verify FAILED … abort, old context intact (NACK)"* |
| **destroy_fault** — post-swap destroy fails | **PASS** | `CIPHER_SC3_FAULT=destroy` → `cuGreenCtxDestroy rc=999`; migration **COMMITTED** (cur_mask=0x6000); old handle leaked (bounded); post-migration GPU op succeeds. Log: *"migration is COMMITTED; leaking the old handle (bounded)"* |
| **optin_yes** — `CIPHER_MIGRATABLE=1` | **PASS** | `SUBSCRIBE_MIGRATE(1) rc=0`; COMPACT → POLL = PROPOSED |
| **optin_no** — env unset | **PASS** | no SUBSCRIBE; COMPACT → POLL = IDLE (tenant pinned) |

self_migrate exercises `cipher_rt_green_ctx_migrate()` directly (no kmod
proposal): drain → build → L1 verify → swap → release, kernels then provably
run on the new SM set (`%smid` probe). verify_fault and destroy_fault confirm
the two failure contracts (B-floor pre-swap NACK; commit-and-leak post-swap).

## End-to-end integration — 2 migratable decode tenants + COMPACT controller

Two TinyLlama decode tenants (`CIPHER_MIGRATABLE=1`, qos=partition, 16 SM each),
each running `MigrateHandler.step()` at every decode-round boundary; a
controller thread issued **103 `COMPACT_MIGRATE`** calls during the run.

| tenant | result | migration | KL gate |
|---|---|---|---|
| **e2e A** | **PASS** | 1 proposed, **1 committed** — green ctx → `0x6000` (group 13), SM set moved, L1+L2 ok | kl_max 5.50e-05 ✓ |
| **e2e B** | **PASS** | 1 proposed, **1 committed** — green ctx → `0x1800` | kl_max 5.50e-05 ✓ |

Both tenants received a real kmod PROPOSAL, drove `START_MIGRATE` → the
libcipher_rt primitive → L2 `%smid` probe → `ACK_MIGRATE(1)` → kmod COMMIT,
and **continued decoding correctly on the new SM set** (TFGATE KL ≤ 0.1). The
full PROPOSE→START→MIGRATING→ACK→COMMIT path works end-to-end across the kmod
(SC2) and libcipher_rt (SC3). Log confirms: *"GREEN/MIGRATE: committed — now
on mask 0x6000 (16 SMs, lowest group 13)"*.

## Two-layer verification (Item-2 PUSH) — exercised

- **L1 structural** (libcipher_rt, pre-swap): `cuGreenCtxGetDevResource` count
  == old count, not full set. verify_fault confirms L1 gates the swap.
- **L2 `%smid` probe** (Python handler, post-swap): observed SM set ⊆
  `mask_to_sms(target)`. Every committed e2e migration passed L2 (`l2_ok=true`,
  `l2_outside=[]`).

## dmesg

**Clean** — no WARNING/BUG/Oops/error/failed/leak from any cipher module
across all scenarios. (libcipher_rt `GREEN/MIGRATE:` lines are userspace
`cipher_log` to tenant stderr, not dmesg.)

---

## Verdict

**SC3-2 unit + integration tests: 7/7 PASS, both e2e tenants migrated
end-to-end.** The tenant-side migration primitive + handler work; the failure
contracts (B-floor NACK, commit-and-leak) hold; opt-in plumbing works;
correctness (KL) is preserved across a live migration.
Evidence: `TRACK_3_SC3_UNIT_TESTS_result.json`, `sc3_*_result.json`, `sc3_*.log`.
