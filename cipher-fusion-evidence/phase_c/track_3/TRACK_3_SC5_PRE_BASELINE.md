# Track 3 SC5 — PRE-BASELINE

**Date:** 2026-05-19. **Purpose:** the reference the SC5-2 post-build
regression smoke compares against. Captured **before any SC5 source
modification**.

**Substrate under test:** kmod `98da2d1f` (loaded) + libcipher_rt `83afd1ca` +
libcipher_v2 `86618c30` + cipher_kv_bridge `fca6843d`. This is the byte-
identical SC3-closeout substrate.

**Anchor preserved:** kmod `98da2d1f` → `cipher_kmod_fallback/cipher_kmod.ko.pre_sc5`
(md5-verified). SC5 rollback baseline.

---

## Verified-primitive baseline

| primitive | pre-SC5 result | SC5 smoke gate |
|---|---|---|
| **W1** — Phase B N=8 TinyLlama | **491.186 tok/s** (tok/W 3.519) | within ±3 % → **[476.45, 505.92]** |
| — TFGATE KL_max | ≤ 5.5e-5 | ≤ 0.1 |
| **W2** — 1.6B-2A PARTITION+POOL | disjoint PASS, clause1=clause2=0, KL PASS | same |
| **W3** — Track 2 SC2 weight arena | `SC2_PASS=True`, KL 0.0 | `SC2_PASS=True` |
| **isolation** — `cp54_isolation_test` | **15 / 15 PASS** | 15/15 |
| **SC2 unit** — `cp54_migrate_test` | **29 / 29 PASS** (state machine 28 + latency-gate 1) | 29/29 |
| **SC3 unit + e2e** — `sc3_run.py` | **7/7 PASS** at SC3 closeout — substrate byte-identical (kmod `98da2d1f` + libcipher_rt `83afd1ca` unchanged since); re-run in the SC5-2 post-build smoke (kmod changes) | 7/7 |

W1/W2/W3 + isolation + SC2-unit run fresh here. SC3's `sc3_run.py` (~15 min,
2× TinyLlama e2e) is verified-PASS at SC3 closeout on this exact substrate and
is re-run in the post-build smoke — where the kmod has actually changed and a
regression could appear. Re-running it pre-build, on a provably-unchanged
substrate, would add nothing.

Evidence: `arm3_sc5pre_repsc5pre/`, `cp_5_4/step1_6/sc5pre_b2a/`,
`track_3/w3_sc5pre.log`.

## dmesg

Clean — no WARNING/BUG/Oops/error/failed/leak.

---

## Verdict

**Pre-SC5 baseline established — all verified primitives PASS.** The SC5-2
post-build regression smoke (after SC4 fold + `cp54_pool_claim_low_prefix` +
harness) must reproduce: W1 ∈ [476.45, 505.92]; W2 disjoint + clause1=clause2=0;
W3 `SC2_PASS`; isolation 15/15; SC2-unit 29/29; SC3 7/7; dmesg clean.
