# Track 2 SC3 — PRE-BASELINE

**Date:** 2026-05-19. Captured before any Track 2 SC3 source modification.

**Substrate (md5-verified):** kmod `285d102e`, libcipher_rt `83afd1ca`,
cipher_kv_bridge `fca6843d` — the Track 3-close substrate, unchanged.
**Anchor preserved:** `cipher_kv_bridge.so` `fca6843d` →
`cipher_rt_phase4/cipher_kv_bridge.so.pre_track2_sc3` (md5-verified).
SC3 rollback baseline.

---

## Baseline

| primitive | pre-SC3 | post-SC3 smoke gate |
|---|---|---|
| **W3** — Track 2 SC2 VMM weight arena (`sc2_verify.py`) | **`SC2_PASS=True`, KL_max 0.0**, arena 2.05 GiB | `SC2_PASS=True`, KL 0.0 |
| isolation — `cp54_isolation_test` | **15/15 PASS** | 15/15 |
| **W1** — Phase B N=8 (`run_arm3.sh`), 3 reps | **468.1 / 468.6 / 493.7** tok/s — mean **476.8**, range [468.1, 493.7] | within W1's intrinsic variance band (see note) |

## W1 variance — methodology note (surfaced)

W1 measured **468.1 / 468.6 / 493.7** across 3 back-to-back reps on the
**md5-unchanged** substrate (GPU 39 °C, SM clock 1980 MHz — not throttled).
The run-to-run spread is **~±5 %**, which **exceeds the ±3 % mean-vs-mean
regression gate**. This is intrinsic noise in the Phase B batched-executor
aggregate tok/s (variable per-round generation lengths — one round is a
16-token round vs 128-token rounds), not substrate drift.

**Consequence for the SC3 gate:** W1 exercises `run_arm3.sh` →
`cipher_batch_executor_gen.py` on **libcipher_rt + kmod** — it does **not**
load `cipher_kv_bridge`. Track 2 SC3 rotates **`cipher_kv_bridge` only**
(kmod + libcipher_rt untouched), so W1 **cannot** be affected by an SC3
change by construction. The SC3 W1 smoke check is therefore "W1 lands within
its own measured variance band [≈468, ≈494]" — confirming the untouched
kmod+libcipher_rt path — not a tight ±3 % against a single noisy rep.

**W3 is the SC3-meaningful regression gate** — `sc2_verify.py` is the workload
that *does* load `cipher_kv_bridge` (the artifact SC3 rotates). Its gate is
binary (`SC2_PASS`, KL 0.0) — no variance problem.

## Scope note — Track 3 substrate

W2 + the Track 3 DSM tests (SC3-DSM e2e, SC5 sweep) exercise kmod `285d102e`
+ libcipher_rt `83afd1ca`, which Track 2 SC3 does **not** touch. They are
verified-PASS on this exact md5-identical substrate at Track 3 SC6; an SC3
`cipher_kv_bridge` change cannot regress a kmod/libcipher_rt-only test.
Re-running them pre-SC3 on the provably-unchanged substrate adds nothing.

## dmesg

Clean — no WARNING/BUG/Oops/error/failed/leak.

---

## Verdict

Pre-SC3 baseline established. The SC3 post-build smoke must reproduce:
**W3 `SC2_PASS=True` + KL 0.0** (the cipher_kv_bridge gate); isolation 15/15;
W1 within [≈468, ≈494] (its measured variance band — cipher_kv_bridge-
independent); dmesg clean.
