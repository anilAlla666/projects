# Track 3 SC3-2 — REGRESSION SMOKE TEST

**Date:** 2026-05-19. **Under test:** libcipher_rt `83afd1ca` (SC3) + kmod
`98da2d1f` (SC2). **Reference:** `TRACK_3_SC3_PRE_BASELINE.md`.
**Verdict: SMOKE TEST PASS — no regression.**

## W1 — Phase B N=8 TinyLlama batched

| metric | pre-SC3 | SC3 `83afd1ca` | Δ | gate | verdict |
|---|---|---|---|---|---|
| agg tok/s | 498.466 | **498.996** | **+0.11 %** | [483.51, 513.42] | **PASS** |
| tok/W | 3.574 | 3.587 | +0.36 % | ±3 % | PASS |
| TFGATE KL_max | ≤ 5.5e-5 | ≤ 5.5e-5 | — | ≤ 0.1 | PASS |

Evidence: `cp_5_6/phase_b/session1/arm3_sc3smoke_repsc3smoke/`.

## W2 — 1.6B-2A mixed deployment (PARTITION + POOL)

| check | pre-SC3 | SC3 | verdict |
|---|---|---|---|
| disjointness pass | True | **True** | PASS |
| clause1_fails / clause2_fails | [] / [] | **[] / []** | PASS |
| partition KL gates | PASS | **PASS** | PASS |
| POOL ok | True | **True** (agg 110.786) | PASS |

Evidence: `cp_5_4/step1_6/sc3smoke_b2a/`.

## W3 — Track 2 SC2 VMM weight arena

| gate | pre-SC3 | SC3 | verdict |
|---|---|---|---|
| SC2_PASS (6 gates) | True | **True** | PASS |
| teacher-forced KL_max | 0.0 | **0.0** | PASS |

Evidence: `phase_c/sc2_result.json`, `track_3/w3_sc3smoke.log`.

## dmesg

**Clean** — no WARNING/BUG/Oops/error/failed/leak across build, reachability
gate, 7 unit/integration scenarios, and the W1/W2/W3 smoke run.

## Verdict — SMOKE TEST PASS

All verified primitives reproduce on libcipher_rt `83afd1ca`: W1 +0.11 %
(inside ±3 %); W2 disjoint + clause1=clause2=0 + KL; W3 `SC2_PASS=True`; dmesg
clean. The no-regression guarantee held — the all-pinned smoke workloads never
subscribe (`CIPHER_MIGRATABLE` unset), so `cipher_rt_green_ctx_ensure()` runs
unchanged and the migrate path is never entered. SC3 may rotate the
libcipher_rt anchor `ebc0baaa` → `83afd1ca`.

## Rollback path (not needed — smoke PASSED; documented per discipline)

```
cp /home/ubuntu/cipher_rt_fallback/libcipher_rt.so.pre_sc3 \
   /home/ubuntu/cipher_rt_phase4/libcipher_rt.so          # restore ebc0baaa
```
kmod `98da2d1f` stays loaded. Then re-run W1/W2/W3 to confirm pre-SC3 levels.
