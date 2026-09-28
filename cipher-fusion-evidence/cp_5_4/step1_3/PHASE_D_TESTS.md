# CP 5.4 Step 1.3 — Phase D: integration tests A–F — REPORT

**Date:** 2026-05-18. **Verdict: ALL TESTS PASS (A–F).** The new libcipher_rt
(`ebc0baaa`) drives green contexts from the CP 5.4 kmod ledger correctly for
SHARED and PARTITION tenants, with byte-identical numerics vs the `a7ac8e97`
baseline, and does not regress SC2 or Phase B.

---

## Harness

- `cp54_tenant.py` — single-tenant TinyLlama-1.1B fp16; deterministic
  teacher-forced forward + 24-token greedy decode; saves logits + gen ids.
  `FAST=1` = load + one forward + exit (CP 5.4 tenant for the reaper test).
  qos declared purely via env (`CIPHER_QOS_CLASS` / `CIPHER_SM_COUNT`, Q2).
- `cp54_query.c` — `/dev/cipher` `CIPHER_CP54_QUERY` ledger snapshot tool.
- `cp54_kl.py` — KL(baseline‖test) gate (≤ 0.1) + exact gen-id match.
- Each tenant loads libcipher_rt via `CUDA_INJECTION64_PATH`. Baseline =
  canonical `a7ac8e97`; Tests = `libcipher_rt.so.new` (`ebc0baaa`).

## Test A — SHARED qos_class (default) — PASS

`libcipher_rt.so.new`, no qos env. libcipher_rt log:
`GREEN/CP54: ALLOCATE ok — qos=1 sm_count=0 -> grp_mask=0x0000 grp_count=0`
then `GREEN: CP 5.4 qos=shared — tenant owns no SM groups; no green context
created (runs on primary context …)`. No ARB lines (ARB retired — confirmed).
**KL vs baseline: max 0.000e+00, gen_ids match.** A SHARED tenant correctly
gets no green context and runs unconfined; numerics identical.

## Test B — PARTITION qos_class — PASS

`CIPHER_QOS_CLASS=partition CIPHER_SM_COUNT=16`. libcipher_rt log:
`ALLOCATE ok — qos=0 sm_count=16 -> grp_mask=0x0003 grp_count=2`,
`GREEN: CP 5.4 grp_mask=0x0003 → 2 group(s), lowest id 0`,
`GREEN: green context ready — 2 group(s), 16 SMs (… CP 5.4 kmod-driven)`.
Concurrent `cp54_query` while the tenant held its allocation:
`n_partitions=1 pool_grp_count=0 free_grp_count=13` (15 − 2). **KL vs
baseline: max 0.000e+00, gen_ids match.** The green context is kmod-driven
(grp_mask-selected, not hash-picked), variable size (16 SMs / 2 groups), and
the kmod ledger reflects the allocation.

## Test C — PARTITION disjointness (sequential) + reaper — PASS

| step | ledger (`cp54_query`) |
|---|---|
| before | `n_partitions=0 free_grp_count=15` |
| Tenant A `PARTITION sm_count=24` → `grp_mask=0x0007` (3 groups, 24-SM green ctx) | — |
| **after A exits** | `n_partitions=0 free_grp_count=15` — **reaper reclaimed A's 3 groups** |
| Tenant B `PARTITION sm_count=16` → `grp_mask=0x0003` (2 groups, 16-SM green ctx) | while alive: `n_partitions=1 free_grp_count=13` |
| after B exits | `n_partitions=0 free_grp_count=15` |

The do_exit reaper reclaims a real injected tenant's groups on exit; the
freed groups are cleanly reused by the next tenant.

## Test D — dmesg cleanliness — PASS

Across Tests A/B/C: no `WARNING`/`BUG`/`Oops`/`panic`/`error`/`failed`/`leak`
from any cipher module; kmod `alloc_failures=0`. No kernel oops.

## Test E — SC2 regression with new libcipher_rt — PASS

`sc2_verify.py` run with `CUDA_INJECTION64_PATH=libcipher_rt.so.new`. The new
lib loads (qos=shared → no green ctx, runs on primary). **SC2_PASS = true**,
all gates (`gate_1_vmm_backed`…`gate_5_export` + `copy_equal`) true,
teacher-forced `kl_max = 0.0` — identical to the SC2 baseline. The new
libcipher_rt does not break the Track 2 VMM weight arena.

## Test F — Phase B regression with new kmod — PASS

W1 (Phase B N=8 TinyLlama batched), 1 rep on kmod `8d777dfb`:
agg tok/s **504.33** (gate [477.0, 506.5] ✓), tok/W **3.630** (gate
[3.434, 3.646] ✓), TFGATE KL ≤ 3.1e-5. Note: the Phase B batch executor loads
`libcipher_v2.so` (not libcipher_rt) — `libcipher_rt.so.new` is not in W1's
path; this re-confirms the corrected kmod + system are regression-free for
Phase B. (libcipher_rt↔executor integration is the descoped Step 1.3b'.)

## Verdict

All Phase D tests **PASS**. SHARED and PARTITION green-context behavior is
correct and kmod-driven; numerics are byte-identical to `a7ac8e97`
(KL = 0.000); the reaper and ledger work for real injected tenants; SC2 and
Phase B do not regress. → Phases A+B+C+D all PASS → libcipher_rt anchor
rotates `a7ac8e97` → `ebc0baaa`.
