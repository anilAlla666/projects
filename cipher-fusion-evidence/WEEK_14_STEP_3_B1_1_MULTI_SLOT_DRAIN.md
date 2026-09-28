# W14 Step 3 S3.B1.1 — REMEMBER Consumer Multi-Slot Drain

**Date:** 2026-05-24
**Tag:** `week-14-step-3-b1-1-multi-slot-drain`
**Substrate anchor:**
- cipher_rt_phase4 `68cc9c3` tag `week-14-step-3-b1-1-multi-slot-drain`
- cipher-fusion-evidence pre-doc HEAD `ef8b830` (S3.C KL gate addendum)
- cipher_kmod `8c643fc` tag `week-9-complete` (unchanged)
- libcipher_rt.so md5 `097cf8d907a7e866a3e3640eb0993003` (was `93d291a8cf6c370c126c5a16fa706617` at S3.B1)

## 1. Latent bug

S3.B1 shipped the REMEMBER consumer draining only slot 2 (`CIPHER_RT_RING_CONSUMER_REMEMBER`). The CLASSIFY (slot 0) and ORACLE (slot 1) read_seq cursors stayed at 0 forever because v1 has no production drainer for those slots — only producers exist:

- `cipher_rt_classify_observer.c:102` emits `CIPHER_RT_RING_EVENT_CLASSIFY`
- `cipher_rt_cublas_shim.c:172` emits `CIPHER_RT_RING_EVENT_TC_PROBE`
- `cipher_rt_kv_alloc.c:591` emits `CIPHER_RT_RING_EVENT_KV_DEDUP_LOCAL`

The producer overflow check at `cipher_rt_ring_write.c:198` uses `min_read_seq()` over all `CIPHER_RT_RING_NUM_CONSUMERS=3` slots:

```c
uint64_t min_read = min_read_seq(r);
if (seq - min_read >= CIPHER_RT_RING_ENTRIES) {
    atomic_fetch_add_explicit(&r->total_dropped, 1, ...);
    return;
}
```

With `read_seq[0] = read_seq[1] = 0`, `min_read = 0` and the 4097th emit per tenant gets silently dropped. The REMEMBER consumer's slot-2 cursor advances normally but is bypassed by the producer's stricter min check.

## 2. Diagnosis

Surfaced during W14 S3.C LM-head harness build. The harness fired one Koopman substitution on tenant 0; expected `cipher_rt_remember_consumer_drained` to advance by 1 within 50 ms. Observed:

- `cipher_rt_koopman_remember_emits` advanced by 1 (engine counter increments unconditionally)
- `cipher_rt_remember_consumer_drained` did not advance
- Consumer thread was confirmed running (191130 total cycles, 141 active)

Late check at 500 ms showed `drained_total = 4096` (one tenant's worth of producer-time entries; capped at ring capacity). The capping pattern at the exact ring buffer size pointed at the overflow check. Verified by tracing: during TinyLlama autoreg the classify_observer emitted to tenant 0 at high rate, filled the ring to capacity, dropped everything subsequent including the Pass I REMEMBER emit.

## 3. Fix

`cipher_rt_remember_consumer.cpp`: the consumer also drains slots 0 and 1 (CLASSIFY + ORACLE) via `cipher_rt_ring_drain` in both `coldstart_sync_all_tenants()` and `consumer_loop()`. Drained entries are not processed by `process_entry` (which filters on `event_type == REMEMBER`); the drain calls only advance `read_seq[0]` and `read_seq[1]` to track the producer's `write_seq`.

```cpp
/* v1 housekeeping: advance CLASSIFY + ORACLE cursors via
 * drain-and-discard. Without this the producer's overflow
 * check drops emits after 4096 per tenant. */
(void)cipher_rt_ring_drain(t, CIPHER_RT_RING_CONSUMER_CLASSIFY, batch, BATCH_PER_TENANT);
(void)cipher_rt_ring_drain(t, CIPHER_RT_RING_CONSUMER_ORACLE,   batch, BATCH_PER_TENANT);
```

This is a v1-only patch. v2 will introduce per-slot consumer registration so the producer's overflow check considers only ACTIVE consumers, preserving CLASSIFY/ORACLE entries for future real consumers.

## 4. Effects on S3.B1 self-test

Rerun of `test_step3_b1_consumer`:

| Case | Before fix | After fix |
|---|---|---|
| ENVGATE OFF | PASS (drained 0) | PASS (drained 0) |
| COLD START | PASS (10000 posted, 4096 accepted, 4096 drained, 4096 lnn_calls) | PASS (same) |
| COMPOSE | PASS (32000 intended, 31000 accepted, drop_pct 3.12%, 31000 drained) | PASS (32000 intended, **32000 accepted, drop_pct 0.00%, 32000 drained**) |

COMPOSE strictly improves: with the multi-slot drain housekeeping the producer never hits the overflow drop, all 32000 events are accepted and drained.

## 5. Effect on S3.D N=128 soak (forecast)

Per kickoff Q3 adjudication, S3.D runs both modes:

- **Mode A `CIPHER_REMEMBER=0`** (consumer dormant): no consumer behavior change; slots 0/1/2 all stay at 0; producer overflow drops after 4096 per tenant; matches W7-9 baseline 10.66 M/s within ±10% gate
- **Mode B `CIPHER_REMEMBER=1`** (consumer active): consumer advances all 3 slots; producer overflow does NOT fire; throughput shifts above the W7-9 baseline. The ±10% gate may FAIL in Mode B. If so, S3.D surfaces this as a measurement of "v1 REMEMBER-active substrate throughput" alongside the W7-9 baseline; the gate criterion is then a discriminator between the two modes, not a regression test against W12

The forecast is verified at S3.D runtime; Mode B's deviation tells us how much the consumer's drain-all changes producer behavior.

## 6. References

- Latent-bug-introducing commit: cipher_rt_phase4 `70f389b` (S3.B1)
- Producer overflow check: `cipher_rt_ring_write.c:194-202`
- min_read_seq: `cipher_rt_ring_write.c:131-140`
- Consumer fix: `cipher_rt_remember_consumer.cpp` `coldstart_sync_all_tenants()` + `consumer_loop()`
- S3.B1 self-test: `cipher_rt_phase4/test_step3_b1_consumer.cpp`
- Diagnosis surfaced by: S3.C harness Pass I (next commit `week-14-step-3-c-lmhead-validate`)
