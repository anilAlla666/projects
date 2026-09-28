# CP 5.4 — Step 1.6B-2A — RE-VALIDATION ON CURRENT ANCHORS

**Date:** 2026-05-19. **Verdict: PASS — the substrate composes correctly
through Tracks 2 + 3.** This is the load-bearing pre-build gate for 1.6B-3
(`CP_5_4_STEP_1_6B_3_4_DESIGN_MEMO.md` §0 / §6).

---

## Why this gate exists

1.6B-1 (partition-tenant runner) and 1.6B-2/1.6B-2A (mixed-deployment
orchestrator) were built and sanity-passed against substrate **kmod
`8d777dfb` / libcipher_rt `ebc0baaa`**. Since then, on the same calendar day:

- **Track 3 (DSM)** rebuilt **libcipher_rt** (`ebc0baaa → 83afd1ca`).
- **Track 2** rebuilt the **kmod** twice (`→ 285d102e → 008b3c66`) and
  cipher_kv_bridge (`fca6843d → c04b0c39`).

The substrate that 1.6B-2A was validated against no longer exists. Generic
regression (W3, isolation 15/15, SC6) does **not** exercise the libcipher_rt
green-ctx-init → `CIPHER_CP54_ALLOCATE` path the PARTITION tenants depend on.
This gate re-runs 1.6B-2A on the current anchors to prove that path survived
the cross-track rotations — before 1.6B-3 is built on top of it.

## Run

```
cp54_s16_orchestrator.py --partitions 16,16 --pool-clients 2 --rounds 5 --tag b2a_reval
```

2 × 16-SM PARTITION + POOL, 5 rounds, per-round lockstep barrier — identical
configuration to the original `b2_op5_b2a` run. Output: `b2_b2a_reval/`.

**Anchors at run time (verified by `md5sum`, before and after — unchanged):**
kmod `008b3c66`, libcipher_rt `83afd1ca`, cipher_kv_bridge `c04b0c39`,
libcipher_v2 `cc0479b8`.

## Result — PASS, and numerically consistent with the original

| signal | original `b2_op5_b2a` (anchors `8d777dfb`/`ebc0baaa`) | re-validation `b2_b2a_reval` (anchors `008b3c66`/`83afd1ca`) |
|---|---|---|
| orchestration mechanics | PASS | **PASS** — POOL-first launch, both partitions READY, barrier release, all rc=0 |
| POOL grp_mask | low prefix (11 groups) | **`0x7fff` claimed → resized to 11 groups / 88 SM low prefix** |
| partition grp_masks | groups 11–14, high end | **part0 `0x1800`, part1 `0x6000`** — groups 11–14, high end |
| disjointness clause 1 (runtime enforcement) | 0 fails | **0 fails** |
| disjointness clause 2 (pairwise) | 0 fails | **0 fails** |
| per-tenant KL gate | PASS, kl_max 5.5e-5 | **PASS, kl_max 5.50e-5** |
| POOL aggregate tok/s | 111.4 | **111.308** |
| per-tenant decode CV | ≈ 0.31 | **0.3144 / 0.3112** |
| all 5 rounds lockstep-synced | yes | **yes** |
| dmesg | clean | **clean** (no oops/warn/fault) |

The re-validation reproduces the original behaviour within run-to-run noise —
identical disjointness result, byte-identical KL, matching POOL throughput and
decode CV. The libcipher_rt green-ctx → `CIPHER_CP54_ALLOCATE` path and the
kmod CP 5.4 group ledger compose correctly on the post-Track-2/3 substrate.

## Verdict

**PASS.** No silent cross-track regression. Track 3's libcipher_rt rebuild and
Track 2's two kmod rebuilds were additive — the CP 5.4 PARTITION path is
intact. The 1.6B-3 build precondition is satisfied.

**Per the SC6-3-pattern stop discipline, this gate result is surfaced for
adjudication before the 1.6B-3 harness build proceeds** (the §0 gate is a
load-bearing precondition — its result, PASS or FAIL, is adjudicated).

Evidence: `b2_b2a_reval/orchestrator_result.json`, `part0.json`, `part1.json`,
`pool_t0.json`, `pool_t1.json`, `executor.log`.
