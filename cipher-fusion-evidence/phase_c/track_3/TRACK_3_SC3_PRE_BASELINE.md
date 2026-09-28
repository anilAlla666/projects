# Track 3 SC3 — PRE-BASELINE (W1/W2/W3 on kmod `98da2d1f` + libcipher_rt `ebc0baaa`)

**Date:** 2026-05-19. **Purpose:** the reference the SC3 regression smoke test
compares against (±3 % mean-vs-mean). Captured **before any SC3 source
modification**.

**Substrate under test:**
- kmod `cipher_kmod/cipher_kmod.ko` = `98da2d1f` (SC2, loaded)
- libcipher_rt `cipher_rt_phase4/libcipher_rt.so` = `ebc0baaa4e771abbdd20f6c4af645040`
- libcipher_v2 `86618c30`, cipher_kv_bridge `fca6843d` — unchanged

**Anchor preserved:** `ebc0baaa` → `cipher_rt_fallback/libcipher_rt.so.pre_sc3`
(md5-verified identical to `.pre_track3`). This is the SC3 rollback baseline.

Each workload run once (confirm current state — no re-tuning).

---

## W1 — Phase B N=8 TinyLlama batched (`run_arm3.sh`)

| metric | pre-SC3 baseline | SC3 smoke gate (±3 %) |
|---|---|---|
| **agg tok/s** | **498.466** | **[483.51, 513.42]** |
| tok/W | 3.574 (secondary) | within ±3 % |
| TFGATE KL_max | ≤ 5.50e-05 | ≤ 0.1 |

Evidence: `cp_5_6/phase_b/session1/arm3_sc3pre_repsc3pre/`.

## W2 — 1.6B-2A mixed deployment (PARTITION + POOL)

| check | pre-SC3 baseline | SC3 smoke gate |
|---|---|---|
| disjointness pass | True | True |
| clause1_fails / clause2_fails | [] / [] | 0 / 0 |
| partition KL gates | PASS | PASS |
| POOL ok | True (agg 111.603) | True |

`cp54_s16_orchestrator.py --partitions 16,16 --pool-clients 2`.
Evidence: `cp_5_4/step1_6/sc3pre_b2a/`.

## W3 — Track 2 SC2 VMM weight arena (`sc2_verify.py`)

| gate | pre-SC3 baseline | SC3 smoke gate |
|---|---|---|
| SC2_PASS (6 gates) | True | True |
| teacher-forced KL_max | 0.0 | ≤ 0.1 |

Evidence: `phase_c/sc2_result.json`, `track_3/w3_sc3pre.log`.

## dmesg

**Clean** — no WARNING/BUG/Oops/error/failed/leak from any cipher module.

---

## Verdict

**Pre-SC3 baseline established — W1/W2/W3 all PASS** on kmod `98da2d1f` +
libcipher_rt `ebc0baaa`. The SC3 regression smoke (after the SC3 build) must
reproduce: W1 ∈ [483.51, 513.42] tok/s; W2 disjoint + clause1=clause2=0 + KL;
W3 `SC2_PASS=True`; dmesg clean.
