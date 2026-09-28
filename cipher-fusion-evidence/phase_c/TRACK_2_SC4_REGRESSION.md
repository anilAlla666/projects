# Track 2 SC4-2 — REGRESSION SMOKE TEST

**Date:** 2026-05-19. **Verdict: SMOKE TEST PASS — no regression.**

SC4 is **pure Python** — `cipher_model_fingerprint.py` + new `sc4_*.py` files;
**no existing binary modified**. The regression surface is therefore
structurally null; the smoke confirms it.

| gate | pre-SC4 | SC4 | verdict |
|---|---|---|---|
| anchor md5 — kmod | `285d102e` | **`285d102e`** | byte-identical |
| anchor md5 — libcipher_rt | `83afd1ca` | **`83afd1ca`** | byte-identical |
| anchor md5 — cipher_kv_bridge | `c04b0c39` | **`c04b0c39`** | byte-identical |
| anchor md5 — libcipher_v2 | `86618c30` | **`86618c30`** | byte-identical |
| W3 — `sc2_verify.py` | `SC2_PASS`, KL 0.0 | **`SC2_PASS=True`, KL 0.0** | PASS |
| isolation — `cp54_isolation_test` | 15/15 | **15/15 PASS** | PASS |
| SC3 integration — `sc3_run.py` | PASS | PASS (verified at the SC4 pre-baseline, this session, on this md5-identical substrate; the `sc3_*.py` scripts are unmodified by SC4) | PASS |
| dmesg | clean | **clean** | PASS |

**All four anchor md5s byte-identical** — the definitive proof that SC4
rotated nothing. W3 (the cipher_kv_bridge-dependent workload) and isolation
(the kmod workload) both pass; dmesg clean across the build, unit tests, and
the two integration scenarios.

## Verdict — SMOKE TEST PASS, NO ANCHOR ROTATION

SC4 introduced the model-identity safety primitive with **zero substrate
change**. kmod `285d102e`, libcipher_rt `83afd1ca`, cipher_kv_bridge
`c04b0c39`, libcipher_v2 `86618c30` — unchanged. No rollback artifact is
required (no binary changed); the SC3 scripts are preserved by being
unmodified.
