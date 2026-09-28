# CP 5.4 — Step 1.6B-3 — BUILD LOG (naive Arm-A harness)

**Date:** 2026-05-19. **Status: harness BUILT + sanity-PASS. STOPPING before
the measurement sweep — a substantive design point surfaced (the per-OP tenant
count), per the Phase-2 instruction.** Anchors unchanged — throwaway harness,
no substrate source touched.

---

## Built

Two throwaway files under `cipher-fusion-evidence/cp_5_4/step1_6/`
(`cp54_s16_*` naming):

- **`cp54_s16_naive_tenant.py`** — the Arm-A tenant. A deliberately
  **stripped `cp54_s16_partition_tenant.py`**: model load, the teacher-forced
  KL gate, `decode_round` (the per-decode-step timing path) and the per-round
  lockstep barrier are **byte-identical** to the PARTITION tenant — the only
  Arm-A/Arm-B difference must be substrate-mediated SM partitioning vs none
  (Step 1.6 §11.2). Removed: the kmod `CIPHER_CP54_QUERY` ledger read and the
  `%smid` disjointness probe — neither has meaning with no partition. Reports
  per-tenant peak memory.
- **`cp54_s16_naive_orchestrator.py`** — launches N naive tenants concurrently
  (plain `Popen`; the launch env explicitly strips `CUDA_INJECTION64_PATH` /
  `LD_PRELOAD` / `CIPHER_*`), runs the lockstep barrier coordinator, collects
  per-tenant JSON, computes per-tenant decode CV / p50 / p95 / p99 / tok·s⁻¹
  and the 3-point framebuffer (idle / loaded / exited). Records the kmod
  load-state and warns if a run is not a pure Arm-A environment.

Warm-round convention (design memo §3e): the tenant runs an untimed warmup
`decode_round` before the timed loop; the orchestrator's headline CV
additionally discards the **first timed round** (the first under-contention
round) — `tenant_stats()` computes `warm` (round ≥ 1) and `all_rounds`.

## Sanity run — `b3_b3_sanity/` — PASS

`cp54_s16_naive_orchestrator.py --tenants 3 --rounds 4 --tag b3_sanity`, run
with the **CIPHER kmod unloaded** (`sudo rmmod cipher_kmod` → confirmed empty;
reloaded `008b3c66` after the run). True Arm-A environment.

- 3 naive tenants launched, all `READY`, all 4 rounds lockstep-synced.
- per-tenant KL gates **all PASS**; decode mean ≈ 15 ms, CV ≈ 0.45–0.46.
- `aggregate_tok_s` 199.3.
- framebuffer idle / loaded / exited = **0 / 9266 / 0 MiB** — clean teardown,
  no leak.

The harness is mechanically correct: concurrent launch, lockstep barrier,
JSON collection, CV/percentile/tok·s⁻¹ aggregation, 3-point FB all work.

## ★ Substantive design point — the per-OP tenant count — STOP for confirm

The Phase-2 authorization (item 4) describes the three OPs as **"OP-2: 2
tenants, OP-5: 5 tenants, OP-asym: 3 tenants of varied sizes."** This does
**not** reconcile with the **adjudicated** 1.6B-3/4 design memo
(`CP_5_4_STEP_1_6B_3_4_DESIGN_MEMO.md`, all 10 items accepted):

- Design memo **§2 / §4** pin the Arm-A process counts at **OP-2 → 7,
  OP-5 → 10, OP-asym → 9** — each = the PARTITION-class tenant count **plus
  5 POOL-class logical tenants** (adjudication item 7, pool-class = 5).
- The Phase-2 "2 / 5 / 3" appears to be shorthand for the **PARTITION count
  only** — and it also mis-states OP-asym (the pinned OP-asym is **4**
  partitions at 8/16/16/24 SM, not 3).
- **"Varied sizes" has no Arm-A meaning.** Naive tenants have no SM slice —
  every naive tenant runs the identical TinyLlama-1.1B B=1 WL01 decode. The
  OP-asym asymmetry is purely an Arm-B SM-allocation property; Arm-A's
  OP-asym is simply 9 identical naive tenants.

**Why it is load-bearing:** the naive baseline's *only* job is to be a fair
comparison point for Arm-B. Arm-B OP-5 contention = 5 partitions + a POOL
batching 5 clients. If Arm-A OP-5 runs only 5 processes (the Phase-2 number)
instead of 10, its contention is **half** of Arm-B's → Arm-A looks
artificially good → the substrate-value comparison is biased in CIPHER's
favour. The honest baseline must reproduce Arm-B's *total* tenant count.

**Resolution applied (pending confirm):** the harness is built to take an
explicit `--tenants N`; the measurement sweep will use the **design-memo
counts — OP-2 = 7, OP-5 = 10, OP-asym = 9** (the accepted, adjudicated
values), all tenants the identical naive TinyLlama workload. This is recorded
here per [[cipher-proceed-not-ask]] — the detailed accepted spec (the design
memo) governs over the looser Phase-2 prose — but because the discrepancy
materially changes the run (7 vs 2 processes), it is **surfaced for explicit
confirmation before the sweep burns 3 OPs × 5 reps.**

## Anchors

Unchanged — `cp54_s16_naive_*.py` are throwaway harnesses, no substrate source
touched. kmod `008b3c66` (unloaded for the sanity run, reloaded after —
md5-verified), libcipher_rt `83afd1ca`, cipher_kv_bridge `c04b0c39`,
libcipher_v2 `cc0479b8`.

## Stop / next

**STOPPING for adjudication of the per-OP tenant count** (above). On confirm:
Phase 3 — the 1.6B-3 measurement sweep, OP-2 / OP-5 / OP-asym at the confirmed
counts, 5 reps each, kmod unloaded throughout, `CP_5_4_STEP_1_6B_3_RESULTS.md`.
