# CP 5.4 — Step 1.6B-4 — BUILD LOG (Arm-B CIPHER harness)

**Date:** 2026-05-19. **Status: harness BUILT + sanity-PASS. STOPPING before
the measurement sweep, per the Phase-4 instruction.** Anchors unchanged —
throwaway harness + one preserved orchestrator edit.

---

## Built / extended

The 1.6B-2/2A mixed-deployment orchestrator already exists and was
re-validated on the current substrate (`CP_5_4_STEP_1_6B_2A_REVALIDATION.md`).
1.6B-4 **extends** it (no fork) and adds a sweep driver:

- **`cp54_s16_orchestrator.py`** — edited (preserved as
  `cp54_s16_orchestrator.py.pre_1_6b4`, md5 `e6719507`). Additions, all
  throwaway-harness:
  - **3-point framebuffer** — `nvidia-smi memory.used` at idle (pre-launch) /
    loaded (all tenants resident) / exited (post-teardown).
  - **NVML power sampler** — a 100 ms-cadence `pynvml` thread (design memo
    §3d); GPU-aggregate, summarised `{mean,min,max}_w`. **Descriptive only —
    never a substrate-lift claim** ([[cipher-lift-framing]]).
  - **`--marlin-trace`** (D4) — LD_PRELOADs the Step 1.5 CUPTI cuModuleLoad
    counter `libcumod_count.so` into the POOL executor; `parse_marlin_trace`
    reads the executor log for cuModuleLoad count and `[cipher_v2] MARLIN`
    lines.
  - **CV comparability fix** — the orchestrator's per-tenant CV previously
    used `decode_ms[1:]` per round (drop first *step* of every round) +
    `pstdev`. Changed to **discard the first *round*** (design memo §3e) and
    `stdev` — now **identical** to the Arm-A naive orchestrator's
    `tenant_stats()`. Arm-A and Arm-B CV must be computed the same way or the
    comparison is invalid. (p50/p95/p99 percentile methods still differ
    slightly between the two orchestrators — secondary metrics, noted; the
    headline is CV.)
- **`cp54_s16_cipher_sweep.py`** — new sweep driver. Per OP: **5 headline
  reps** (no trace — the pristine CV/throughput population) **+ 1 trace rep**
  (`--marlin-trace`, characterization, **excluded** from the headline
  statistics — the LD_PRELOAD'd CUPTI counter is kept out of the headline
  reps). Between every rep: a **`CIPHER_CP54_QUERY` clean-ledger check** (D1)
  — `n_part==0 ∧ pool==0 ∧ free==15`; a non-drained ledger **ABORTS** the
  sweep. Headline aggregation computes the **bootstrap-95 % CI of the mean
  per-tenant CV** (`bootstrap_ci`, 10 000 resamples — design memo §11.2; the
  Arm-B-vs-Arm-A CI-separation test is the 1.6D comparison).

## Operating points (pinned)

| OP | `--partitions` | POOL | logical tenants |
|---|---|---|---|
| OP-2 | `16,16` | 11 grp / 88 SM | 2 PART + 5 POOL = 7 |
| OP-5 | `16,16,16,16,16` | 5 grp / 40 SM | 5 + 5 = 10 |
| OP-asym | `8,16,16,24` | 7 grp / 56 SM | 4 + 5 = 9 |

All match the Arm-A 1.6B-3 logical-tenant counts (7/10/9). `--pool-clients 5`.

## Sanity run — OP-2, 1 headline + 1 trace rep, 4 rounds — PASS

`cp54_s16_cipher_sweep.py --ops op2 --headline-reps 1 --rounds 4`, kmod
**loaded** (`008b3c66`). Every Phase-4 mechanism fired:

| mechanism | observed |
|---|---|
| orchestrator mixed deployment | 2 PARTITION (`0x6000`/`0x1800`, 16 SM each) + POOL (5 clients), barrier all 4 rounds, all rc=0 |
| disjointness probe | **PASS — clause1=0, clause2=0** (5 round-checks) |
| 3-point framebuffer | idle/loaded/exited = **0 / 9078 / 0 MiB** — clean teardown |
| NVML power sampler | **178 samples**, mean 150.6 W (min 123.8 / max 185.6) |
| `CIPHER_CP54_QUERY` clean-ledger check | **CLEAN** before and after every rep (`n_part=0 pool=0 free=15`) |
| Marlin trace (D4) | `cumod_load_count=2`, **`marlin_engine_initialised=False`** — no `[cipher_v2] MARLIN` line in the executor log |
| bootstrap CI | partition-CV mean 0.3911, 95 % CI [0.3892, 0.3930] |

Per-tenant KL gates PASS (kl_max 5.5e-5); POOL `agg_tok_s` 279.

**Preliminary D4 read (sanity only — full answer is the sweep's trace reps):**
the Marlin engine did **not** initialise in the OP-2 sanity run (no `MARLIN`
log line; the 2 cuModuleLoad calls are non-Marlin). On the honest D4 framing
(design memo §4.5): *Marlin engine not initialised → Marlin is out of the
measured path.* The full-sweep trace reps confirm this per OP.

## Anchors

Unchanged — `cp54_s16_*` are throwaway harnesses; the orchestrator edit is a
test-harness edit (`.pre_1_6b4` preserved). kmod `008b3c66`, libcipher_rt
`83afd1ca`, cipher_kv_bridge `c04b0c39`, libcipher_v2 `cc0479b8` — md5-verified
unchanged after the sanity run. (`cipher_kv_bridge` is present but **not
engaged** — Track 2 weight sharing is OFF for the 1.6B-4 headline, design memo
§3c.)

## Stop / next

**Phase 4 build complete, sanity-PASS — STOPPING for adjudication before
Phase 5** (the full Arm-B sweep: OP-2/OP-5/OP-asym, 5 headline reps + 1 trace
rep each, 10 rounds, kmod loaded). On adjudication → Phase 5 → the 1.6D
Arm-A-vs-Arm-B comparison (`CP_5_4_STEP_1_6B_4_RESULTS.md`).
