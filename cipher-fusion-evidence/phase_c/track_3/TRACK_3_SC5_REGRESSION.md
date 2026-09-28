# Track 3 SC5-2 — REGRESSION SMOKE TEST

**Date:** 2026-05-19. **Under test:** SC5 kmod `285d102e` (SC4 fold +
constrained POOL grant), loaded; + libcipher_rt `83afd1ca`.
**Reference:** `TRACK_3_SC5_PRE_BASELINE.md`.
**Verdict: SMOKE TEST PASS — no regression. Run BEFORE any measurement work
(per the SC5 discipline — a regression stops SC5 at the build, not after
wasted measurement).**

| gate | pre-SC5 | SC5 `285d102e` | verdict |
|---|---|---|---|
| build | — | clean, no warnings | PASS |
| **isolation** `cp54_isolation_test` | 15/15 | **15 / 15 PASS** | PASS |
| **SC2 unit** `cp54_migrate_test` | 29/29 | **29 / 29 PASS** | PASS |
| **W1** Phase B N=8 | 491.186 tok/s | **495.257 tok/s** (+0.83 %, gate [476.45, 505.92]) | PASS |
| — TFGATE KL_max | ≤ 5.5e-5 | ≤ 5.5e-5 | PASS |
| **W2** 1.6B-2A PARTITION+POOL | disjoint, c1=c2=0 | **disjoint, clause1=clause2=0, KL PASS** | PASS |
| **W3** Track 2 SC2 weight arena | `SC2_PASS` | **`SC2_PASS=True`, KL 0.0** | PASS |
| **SC3 unit + e2e** `sc3_run.py` | 7/7 | **7 / 7 PASS** (self_migrate, verify_fault, destroy_fault, optin×2, e2e_A, e2e_B; both e2e tenants migrated) | PASS |
| dmesg | clean | **clean** | PASS |

## Notes

- **isolation 15/15 is the meaningful gate for the constrained POOL grant** —
  Test 3 (pool resize) and Test 5 (disjointness) exercise the POOL claim path;
  a gross `cp54_pool_claim_low_prefix` regression would fail them. They pass:
  the constrained grant is correct on the static / well-ordered cases. Its
  behaviour under *out-of-order partition churn* is what SC5-3 measures (the
  5 seeds) — by construction W1/W2/W3 do not exercise that.
- **The all-pinned guarantee held:** W1/W2/W3 set no `CIPHER_MIGRATABLE`, so
  the migration path never fires regardless of the new module-param values;
  and the constrained grant matches the old behaviour for the static
  partition layouts these workloads use.
- SC3 `sc3_run.py` re-run in full (not referenced) — the kmod changed, so the
  migration end-to-end path was actually re-verified on `285d102e`.

## Verdict — SMOKE TEST PASS

All verified primitives reproduce on SC5 kmod `285d102e`. SC5-2 rotates the
kmod anchor `98da2d1f` → `285d102e` and stops for adjudication before SC5-3
measurement.

## Rollback path (not needed — smoke PASSED; documented per discipline)

```
sudo rmmod cipher_kmod
sudo insmod /home/ubuntu/cipher_kmod_fallback/cipher_kmod.ko.pre_sc5   # 98da2d1f
```
libcipher_rt `83afd1ca` stays. Then re-run the smoke to confirm pre-SC5 levels.
