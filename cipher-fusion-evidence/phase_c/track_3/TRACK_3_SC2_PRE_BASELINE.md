# Track 3 SC2 — PRE-BASELINE (W1/W2/W3 reference on kmod `8d777dfb`)

**Date:** 2026-05-19. **Purpose:** the reference point the SC2 regression
smoke test compares against (±3 % mean-vs-mean). Captured **before any SC2
source modification**, on the currently-loaded pre-Track-3 kmod.

**Kmod under test:** `cipher_kmod/cipher_kmod.ko` md5
`8d777dfb4c60d128b77c49a8082b9ee2` (= anchor `8d777dfb`, loaded —
`lsmod` shows `cipher_kmod`). libcipher_rt `ebc0baaa`, libcipher_v2 `86618c30`,
cipher_kv_bridge `fca6843d` — unchanged.

Each workload run **once** ("confirm current state", per the SC2 plan — no
re-tuning). These three numbers are the SC2 smoke-test reference, **not** the
historical 491.8 figure.

---

## W1 — Phase B N=8 TinyLlama batched (`run_arm3.sh`, gen executor)

| metric | pre-baseline | historical ref | SC2 smoke gate (±3 % of pre-baseline) |
|---|---|---|---|
| **agg tok/s** | **497.093** | 491.8 baseline / 495.339 on `8d777dfb` 15-fix | **[482.18, 512.01]** |
| tok/W | 3.571 (mean power 139.2 W, 47 samples) | 3.54 baseline | secondary |
| TFGATE KL_max (per round) | ≤ 5.50e-05 | gate ≤ 0.1 | ≤ 0.1 |

Harness: `cipher-fusion-evidence/cp_5_6/phase_b/session1/run_arm3.sh`
(`N=8`, `cipher_batch_executor_gen.py`, TinyLlama-1.1B, WL01 gold).
Evidence: `arm3_sc2pre_repsc2pre/` (executor_result.json, power.csv, logs).

## W2 — one mixed deployment, 1.6B-2A pattern (PARTITION + POOL concurrent)

| check | pre-baseline | SC2 smoke gate |
|---|---|---|
| disjointness pass | **True** | True |
| clause1_fails (partition ⊆ kmod alloc) | **[]** (0) | 0 |
| clause2_fails (pairwise SM intersection empty) | **[]** (0) | 0 |
| partition KL gates | **PASS** (kl_max 5.50e-05 both partitions) | PASS |
| POOL ok | **True** (agg_tok_s 111.494) | True |

Config: `cp54_s16_orchestrator.py --partitions 16,16 --pool-clients 2`
(2 × 16-SM PARTITION + 2-client POOL, 5 rounds, per-round lockstep barrier).
part0 `grp_mask=0x1800`, part1 `grp_mask=0x6000`, POOL low prefix.
Evidence: `cipher-fusion-evidence/cp_5_4/step1_6/sc2pre_b2a/`.

## W3 — Track 2 SC2 VMM weight arena (`sc2_verify.py`)

| gate | pre-baseline | SC2 smoke gate |
|---|---|---|
| 1 VMM-backed (`page_info kind='weight'`) | **PASS** | PASS |
| 2 `page_info` VMM-handle present | **PASS** (`vmm_handle=0x622dfb5c4330`) | PASS |
| 3 DtoH memcmp bytes-identical | **PASS** | PASS |
| 4 teacher-forced KL ≤ 0.1 | **PASS** (KL_max 0.0) | PASS |
| 5 exportable VMM handle | **PASS** (`export_fd=44`, fstat ok) | PASS |
| copy `torch.equal` all 201 tensors | **PASS** | PASS |
| **SC2_PASS** | **True** | True |

Harness: `cipher-fusion-evidence/phase_c/sc2_verify.py` (TinyLlama-1.1B,
201 weight tensors / 2.20 GB rebound into the VMM arena). Kmod-independent —
exercises `cipher_kv_bridge` `fca6843d` only.
Evidence: `phase_c/sc2_result.json`, `track_3/w3_prebaseline.log`.

## dmesg

**Clean.** No `WARNING`/`BUG`/`Oops`/`error`/`failed`/`leak` from any cipher
module. Only expected info lines present (kmod load, kprobe/reaper attach,
`/dev/cipher_kvdedup` ready, BAR0 bind, and the informational
`REQUEST_SM_PARTITION (nr 9) deactivated` notice — a `pr_warn_once` that is
expected behaviour, not a fault).

---

## Verdict

**Pre-baseline established — W1/W2/W3 all PASS on kmod `8d777dfb`.** SC2's
regression smoke test (run after the SC2 build) must reproduce:
- W1 agg tok/s within **[482.18, 512.01]**;
- W2 disjointness PASS, clause1=clause2=0, KL gates PASS;
- W3 `SC2_PASS=True`;
- dmesg clean.

Next: SC2 kmod source implementation (task #2).
