# Track 3 SC2 — REGRESSION SMOKE TEST

**Date:** 2026-05-19. **Kmod under test:** SC2 build `cipher_kmod.ko`
md5 `98da2d1fd906a1b58c978a6996021539` (anchor `98da2d1f`), loaded.
**Reference:** `TRACK_3_SC2_PRE_BASELINE.md` (kmod `8d777dfb`).
**Gate:** W1 within ±3 % mean-vs-mean of the pre-baseline; W2 disjointness +
KL; W3 `SC2_PASS`; dmesg clean.

---

## W1 — Phase B N=8 TinyLlama batched (`run_arm3.sh`)

| metric | pre-baseline | SC2 `98da2d1f` | Δ | gate | verdict |
|---|---|---|---|---|---|
| agg tok/s | 497.093 | **489.342** | **−1.56 %** | [482.18, 512.01] (±3 %) | **PASS** |
| tok/W | 3.571 | 3.515 | −1.57 % | within ±3 % | PASS |
| TFGATE KL_max | ≤ 5.5e-5 | ≤ 5.5e-5 (r3 2.0e-5, r4 3.1e-5) | — | ≤ 0.1 | PASS |

−1.56 % is inside the ±3 % band and within normal run-to-run variance for this
workload (the `8d777dfb` 15-fix run itself measured 495.34; pre-baseline 497.09
— a ~1.6 % spread with no code change). Evidence: `arm3_sc2smoke_repsc2smoke/`.

## W2 — mixed deployment, 1.6B-2A pattern (PARTITION + POOL)

| check | pre-baseline | SC2 `98da2d1f` | verdict |
|---|---|---|---|
| disjointness pass | True | **True** | PASS |
| clause1_fails | [] | **[]** | PASS |
| clause2_fails | [] | **[]** | PASS |
| partition KL gates | PASS (5.5e-5) | **PASS** (5.5e-5) | PASS |
| POOL ok | True | **True** (agg 110.591) | PASS |

`cp54_s16_orchestrator.py --partitions 16,16 --pool-clients 2`. part0
`grp_mask=0x1800`, part1 `grp_mask=0x6000` — identical placement to the
pre-baseline. Evidence: `cp_5_4/step1_6/sc2smoke_b2a/`.

## W3 — Track 2 SC2 VMM weight arena (`sc2_verify.py`)

| gate | pre-baseline | SC2 `98da2d1f` | verdict |
|---|---|---|---|
| SC2_PASS (all 6 gates) | True | **True** | PASS |
| teacher-forced KL_max | 0.0 | **0.0** | PASS |

Kmod-independent path; reproduces exactly. Evidence: `phase_c/sc2_result.json`,
`track_3/w3_sc2smoke.log`.

## dmesg

**Clean.** No `WARNING`/`BUG`/`Oops`/`error`/`failed`/`leak` from any cipher
module across the build, isolation tests, 29 unit-test assertions, and the
W1/W2/W3 smoke run. The SC2 migration path emits `pr_info` "CP54 migrate
PROPOSE …" lines when COMPACT/eval proposes a migration (260 such lines from
the unit-test COMPACT calls) — these are informational, not faults, and the
smoke workloads (all-pinned, zero `migratable` tenants) emit none.

---

## Verdict — SMOKE TEST PASS

All verified primitives reproduce on the SC2 kmod `98da2d1f`:
- W1 within ±3 % (−1.56 %) of the pre-baseline;
- W2 disjointness PASS, clause1 = clause2 = 0, KL gates PASS;
- W3 `SC2_PASS = True`, KL 0.0;
- dmesg clean.

**No regression.** The structural guarantee held: with zero `migratable`
tenants the migration mechanism is a side-effect-free scan, so the unchanged
ledger path is exercised byte-for-byte. SC2 may rotate the kmod anchor
`8d777dfb` → `98da2d1f` and proceed to closeout.

## Rollback path (not needed — smoke PASSED; documented per discipline)

```
sudo rmmod cipher_kmod
sudo insmod /home/ubuntu/cipher_kmod_fallback/cipher_kmod.ko.pre_track3   # 8d777dfb
```
(`.pre_sc2` is byte-identical to `.pre_track3` — SC2 is the first Track 3 code
change.) Then re-run W1/W2/W3 to confirm pre-Track-3 levels.
