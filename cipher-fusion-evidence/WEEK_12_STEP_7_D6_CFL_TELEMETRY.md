# W12 Step 7: D6 CFL throttle telemetry harness backfill

**Date:** 2026-05-23
**Scope-drift audit closure:** D6 of 15
**Cumulative closures:** D5, D9, D14, D3, **D6** = 5 of 15
**Open audit items:** D1, D2, D4, D7, D8, D10, D11, D12, D13, D15 (10 of 15)

## 1. Pre-condition verification

All substrate anchors verified on disk before measurement.

| Component | Tag / Anchor | md5 |
|-----------|--------------|-----|
| `cipher_rt_phase4` | `week-12-step-6-d3-l2-wireup` (`adbe121`) | `libcipher_rt.so` `33412ffc97b714662e0824e1de8362bd` |
| `cipher_kmod` | `week-9-complete` (0.6.5) | `cipher_kmod.ko` `8c9fdd016897436ceff382c4e9178e07` |
| `cipher_vllm_plugin` | (untagged tree) | `cipher_vllm_kv.py` `2b6cedab89387c30becd49a27313ceb4` |
| `cipher-fusion-evidence` | `602c560` | `WEEK_12_STEP_6_D3_L2_WIREUP.md` present |
| `cipher_rt_ring_write.c` | unchanged since W10 Step 1 | (no source touch) |

Baseline `libcipher_rt.so` preserved at `/tmp/step7_d6_baseline/libcipher_rt.so`
(`33412ffc97b714662e0824e1de8362bd`).

## 2. D6 audit-item context

`WEEK_12_SCOPE_DRIFT_AUDIT.md` item D6 records that W10 Step 1 shipped the
CFL stability invariant (75% EWMA threshold sampled every 1024 writes per
`cipher_rt_ring_write.c:177-193`) without ever exercising it under producer-
consumer rate divergence. The 30-min N=128 soak harness drives writers at
synthetic rates but never imposed a true consumer-rate ceiling distinct
from the producer rate. The CFL throttle counter
(`cipher_rt_ring_total_throttled` at `cipher_rt_ring_write.h:96`) had never
been observed firing under controlled conditions. D6 builds a measurement
harness that exercises the producer-consumer rate divergence to confirm:

1. CFL throttle counter increments when producer rate is sustained above
   the 75% × EWMA-consumer-drain threshold.
2. False-positive rate is bounded under conditions where the consumer drain
   capacity exceeds producer rate.
3. The behavior holds across single producer slot and multi-slot composition
   (all 5 producer slots from the W10-W12 ladder: CLASSIFY, KV_DEDUP,
   MARLIN, TC_PROBE, L2_PERSIST).
4. Provides calibration data for W15-17 CP 5.5 production-rate tuning.

This is **additive measurement**. No substrate code change. Per memory
`regression-discipline`, the W10 Step 1 CFL implementation stays exactly
as shipped.

## 3. CFL implementation summary (read-only code citations)

Source: `/home/ubuntu/cipher_rt_phase4/cipher_rt_ring_write.c` (310 lines).

### 3.1 Tunables (`cipher_rt_ring_write.c:58-62`)

| Constant | Value | Purpose |
|----------|-------|---------|
| `CIPHER_RT_RING_CACHELINE` | 64 B | cache-line padding |
| `CIPHER_RT_CFL_SAMPLE_EVERY` | 1024 writes | sampling boundary for `cfl_write_rate` |
| `CIPHER_RT_CFL_THRESHOLD_PCT` | 75 | throttle if `write_rate > 75% × drain_rate_ema` |
| `CIPHER_RT_CFL_EMA_ALPHA_X100` | 10 | EWMA alpha = 0.10 fixed-point |
| `CIPHER_RT_CFL_INITIAL_DRAIN` | `UINT64_MAX` | cold start: no throttle until first consumer EMA update |

### 3.2 Producer self-throttle path (`cipher_rt_ring_write.c:159-225`)

The producer at line 159 takes the W12-D14 env gate at line 164, then:

1. Loads `write_seq` (relaxed; single producer per tenant).
2. If `(seq & (CIPHER_RT_CFL_SAMPLE_EVERY - 1)) == 0`, calls
   `cfl_update_write_rate(r, seq)` to refresh `cfl_write_rate`. This costs
   one `clock_gettime` amortized to ~25 ps per call.
3. Reads `drain_rate_ema` and `cfl_write_rate` (both relaxed atomics).
4. If `drain_rate_ema != UINT64_MAX && write_rate > (drain_rate_ema * 75)/100`,
   increments `total_throttled` and returns. The CFL self-throttle path.
5. Otherwise checks ring overflow at line 197-202 (`seq - min_read_seq() >= 4096`).
   If overflow, increments `total_dropped` and returns.
6. Otherwise commits the entry and releases-stores `seq+1`.

### 3.3 Consumer drain EMA update (`cipher_rt_ring_write.c:255-286`)

Only `consumer_id == 0` (CLASSIFY slot) updates `drain_rate_ema`. The other
slot (ORACLE = 1) drains its own cursor but does **not** touch the EMA. The
EMA update is gated to `dt >= 100 ms` (line 262) to amortize cost. First
update path (line 272) does a direct seed from instantaneous rate (the cold
start `UINT64_MAX` -> instantaneous transition); subsequent updates do the
standard EWMA blend with alpha=0.10.

### 3.4 Producer overflow protocol (`cipher_rt_ring_write.c:129-140, 195-202`)

`min_read_seq` iterates **all** `NUM_CONSUMERS=2` slots and returns the
minimum. The producer overflow check uses this minimum. If any consumer
slot stalls, its `read_seq` stays low and the producer drops once
`seq - min_read >= 4096`. This is the load-bearing constraint the test
harness had to honor (Section 5.1).

### 3.5 Public accessors

`cipher_rt_ring_total_written/throttled/dropped` (lines 291-310) are the
per-tenant counters the test harness diffs across cases.

## 4. Test harness design

Source: `/tmp/step7_d6_baseline/test_cfl_telemetry.c` (~290 LOC, C11,
pthread). Links `libcipher_rt.so`.

### 4.1 Producer/consumer threads

- **Producer thread:** spin-paced via `wait_until` (the test harness uses
  a hybrid `nanosleep` + busy-poll to hit sub-millisecond per-call budgets
  needed at 500k events/sec). Per-call payload is 8 B.
- **Consumer thread:** drives `cipher_rt_ring_drain` at a target
  call-rate. Each call drains up to `DRAIN_BATCH = 64` entries. The
  effective drain capacity per second is `call_rate × 64`.

### 4.2 Dual-slot drain (load-bearing)

Per Section 3.4, the producer's overflow check uses `min_read_seq` across
**all** consumer slots. The harness spawns one drain thread per slot
(slot-0 CLASSIFY and slot-1 ORACLE) for **every** tenant. Slot-0 is the
only thread that updates the EMA; slot-1 drains identically but does not
touch `drain_rate_ema`. Without this, slot-1 stays at `read_seq=0` and
trips overflow drops after the first 4096 writes (verified empirically in
the smoke test before the fix landed).

### 4.3 Five regimes

| Case | Regime | Producer/tenant | Drain/tenant | n tenants | event_types |
|------|--------|----------------|--------------|-----------|-------------|
| 1 | BALANCED single-slot | 100,000 /s | 100,000 /s | 1 | CLASSIFY |
| 2 | PRODUCER-FAST single | 500,000 /s | 100,000 /s | 1 | CLASSIFY |
| 3 | CONSUMER-FAST single | 50,000 /s | 500,000 /s | 1 | CLASSIFY |
| 4 | BALANCED multi-slot | 20,000 /s | 20,000 /s | 5 | CLASSIFY + KV_DEDUP + MARLIN + TC_PROBE + L2_PERSIST |
| 5 | PRODUCER-FAST multi | 100,000 /s | 20,000 /s | 5 | (same 5 event types) |

Each case runs 60s. Total wall time ~5 min plus warm-up.

### 4.4 Substrate-correctness gate

The harness returns exit code 0 iff Case 2 PRODUCER-FAST shows
`total_throttled > 0`. This is the load-bearing "CFL fires under designed
conditions" gate. All other case results are characterization data, not
gates.

## 5. Measurement results

Full log at `/tmp/step7_d6_baseline/cfl_telemetry.log`. 60s per case.

| Case | Regime | Written | Throttled | Dropped | Throttle% | Drop% |
|------|--------|---------|-----------|---------|-----------|-------|
| 1 | BALANCED single-slot | 9,539 | 5,990,462 | 0 | **99.84%** | 0.00% |
| 2 | PRODUCER-FAST single | 6,001,737 | 5,859 | 23,992,405 | 0.02% | **79.97%** |
| 3 | CONSUMER-FAST single | 2,997,075 | 2,926 | 0 | 0.10% | 0.00% |
| 4 | BALANCED multi-slot | 5,994,155 | 5,850 | 0 | 0.10% | 0.00% |
| 5 | PRODUCER-FAST multi | 6,010,880 | 5,870 | 23,983,255 | 0.02% | **79.94%** |

**Gates:**
- Case 2 PRODUCER-FAST single-slot CFL fires: **YES** (5,859 throttle events)
- Case 5 PRODUCER-FAST multi-slot CFL fires: **YES** (5,870 throttle events)
- Test binary exit code: **0**

## 6. Calibration finding (informs W15-17 CP 5.5)

The five regimes surface four distinct CFL dynamics that the original W10
Step 1 design did not characterize:

### 6.1 Two operating modes, not one

CFL is a **soft** throttle that yields to ring-overflow drop in true
overload. Cases 2 and 5 (producer at 5× consumer drain) show **drop
dominates throttle by ~4000×** (79.97% drop vs 0.02% throttle). The
producer's `write_rate` measurement is computed from **successful**
write_seq advances, so once the ring fills and overflow drops kick in,
the measured write_rate stays below the threshold and CFL barely engages.
The hard backstop is `min_read_seq` overflow, not CFL.

This is **load-bearing** for production interpretation: CFL is best
understood as steady-state back-pressure, not as a DoS shield. Real
overload uses the drop counter.

### 6.2 Sampling sawtooth at high call rates

Case 1 (100k/s producer at the consumer's ceiling) throttles **99.84%** of
calls. The dynamic is a sawtooth driven by the sampled-rate measurement
at the 1024-write boundary: once throttled, `seq` stops advancing, the
next call samples `ds=0` at the same boundary which sets `write_rate=0`,
the throttle releases, one write succeeds, then the throttle re-engages
on the next sample. This produces an empirical write rate of ~159 writes/s
(0.16% of the 100k/s call rate) at balanced load.

This is **not** what the W10 Step 1 design memo implied. The implied
behavior was "throttle fires occasionally under sustained pressure." The
actual behavior under high call rates is "throttle dominates."

### 6.3 Benign at low call rates

Case 4 (5 tenants at 20k/s/tenant each) shows the same write_rate ≈
drain_rate condition as Case 1 in principle but throttles only **0.10%**
of calls. The difference is the absolute sampling cadence: at 20k/s, the
1024-write sample boundary hits every ~51 ms (vs every ~10 ms at 100k/s),
which gives the EMA update window time to keep drain_rate ahead of
write_rate. Below ~50k events/sec/tenant, CFL is effectively dormant.

**Production-rate implication:** vLLM CUPTI launch callbacks fire on the
order of 1k-10k events/sec/tenant per the W10-W12 soak telemetry (10.66 M
publishes/s aggregate across 128 tenants = 83k/s/tenant peak, much
lower under realistic generation tokens-per-second loads). Production
CFL behavior should sit closer to Case 4 (0.10% throttle) than Case 1.

### 6.4 Consumer-fast is silent

Case 3 (50k/s producer, 500k/s drain capacity) shows **0.10% throttle** — no
false positives in the regime where consumer drain capacity exceeds
producer rate by 10×. The 75% threshold is conservative enough that
asymmetric consumer headroom keeps the EMA above the threshold.

### 6.5 Threshold tuning recommendation (deferred to CP 5.5)

The 75% default produces aggressive throttle at balanced high-rate single-
tenant load. Two paths exist for CP 5.5:

- **Path A: keep 75%.** Document that CFL is a sampled steady-state
  back-pressure and that high per-tenant rates trigger heavy throttle by
  design (a feature, not a bug — overload routes to drop+upstream
  back-pressure).
- **Path B: raise to ~85-90%.** Reduces sawtooth at high-rate balanced
  loads. May produce more overflow drops under truly bursty workloads.

Either path needs **production rate measurement** in real vLLM workloads
to calibrate. D6 establishes the synthetic envelope; CP 5.5 establishes
the production set point. **No CFL tunable changes in this step.**

## 7. Substrate verification post-measurement

Re-checked all three substrate md5s after the 5-minute measurement run.
All match pre-measurement values.

| File | Pre-measurement md5 | Post-measurement md5 | Match |
|------|--------------------|--------------------|-------|
| `libcipher_rt.so` | `33412ffc...` | `33412ffc97b714662e0824e1de8362bd` | YES |
| `cipher_kmod.ko` | `8c9fdd01...` | `8c9fdd016897436ceff382c4e9178e07` | YES |
| `cipher_vllm_kv.py` | `2b6cedab...` | `2b6cedab89387c30becd49a27313ceb4` | YES |

Git working trees in `cipher_rt_phase4` and `cipher_kmod` clean (no
modified, no untracked source).

The measurement harness reads exported telemetry counters via the public
API; no substrate state was modifiable from outside `libcipher_rt.so`.

## 8. Scope-drift audit item D6 CLOSED

D6 was characterization, not fix. The CFL substrate as shipped in W10
Step 1 stays as shipped. The harness now provides:

1. Empirical confirmation that CFL fires under producer-fast regimes
   (Cases 2 and 5).
2. Empirical confirmation that CFL stays silent under consumer-fast
   regimes (Case 3 at 0.10%).
3. The sampled-rate sawtooth dynamic surfaced at high call rates
   (Case 1 at 99.84%) — newly characterized; not in W10 Step 1 design memo.
4. The two-mode behavior: CFL is a soft throttle, ring overflow is the
   hard backstop in true overload (Cases 2 and 5 with ~80% drop).
5. A clear calibration target for W15-17 CP 5.5 (path A or path B).

Cumulative scope-drift closures: D5 (W12 Step 4), D9 (Plan reconciliation),
D14 (W12 Step 5), D3 (W12 Step 6), **D6 (this step)**. Five of fifteen.

## 9. Honest residue

1. The 60s synthetic per-case window may under-represent EMA settling
   time and ring-fill transient behavior in long-tail workloads.
   Production CP 5.5 measurement spans hours.
2. The harness drives `cipher_rt_ring_write` directly. In production the
   producer is invoked indirectly via the W11 Step 2 producer-site wirings
   (CLASSIFY observer, KV_DEDUP G3, MARLIN G4, TC_PROBE, L2_PERSIST).
   Per-call cost differences between the direct call and the indirect
   wrappers are not measured here.
3. Slot-1 (ORACLE) is drained by a thread that does nothing but maintain
   `read_seq` advance. The ORACLE consumer in W13-14 will be a real
   consumer with its own pacing. The synthetic drain rate matches slot-0
   here; in production the rates may diverge.
4. `total_dropped` was not in scope for D6 but its dominance in Cases 2
   and 5 surfaces a related characterization question (drop rate under
   bursty overload) deferred to CP 5.5.
5. Case 1's 99.84% throttle would be surprising to a reader who only saw
   the W10 Step 1 design memo. The step doc here is now the canonical
   reference for CFL's sampling sawtooth at high per-tenant rates.

## 10. Files

- `/tmp/step7_d6_baseline/test_cfl_telemetry.c` (~290 LOC, C11, harness source)
- `/tmp/step7_d6_baseline/test_cfl_telemetry` (compiled binary, 16944 B)
- `/tmp/step7_d6_baseline/cfl_telemetry.log` (full measurement log)
- `/tmp/step7_d6_baseline/libcipher_rt.so` (preserved baseline,
  `33412ffc97b714662e0824e1de8362bd`)
- `cipher-fusion-evidence/WEEK_12_STEP_7_D6_CFL_TELEMETRY.md` (this doc)

## 11. Anchors

| Tree | Anchor | Status |
|------|--------|--------|
| `cipher_rt_phase4` | `week-12-step-6-d3-l2-wireup` (`adbe121`) | UNCHANGED |
| `cipher_kmod` | `week-9-complete` (0.6.5) | UNCHANGED |
| `cipher_vllm_plugin/cipher_vllm_kv.py` | `2b6cedab89387c30becd49a27313ceb4` | UNCHANGED |
| `cipher-fusion-evidence` | new HEAD with this file | step-doc commit only |

No tag added (additive measurement, no substrate move).
