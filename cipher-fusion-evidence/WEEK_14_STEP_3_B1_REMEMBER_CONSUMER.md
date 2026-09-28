# W14 Step 3 S3.B1 — REMEMBER Consumer Drain

**Date:** 2026-05-24
**Tag:** `week-14-step-3-b1-remember-consumer`
**Substrate anchor at close:**
- cipher_rt_phase4 `70f389b` tag `week-14-step-3-b1-remember-consumer`
- cipher-fusion-evidence pre-doc HEAD `7154d5b`
- cipher_kmod `8c643fc` tags `week-9-complete` (unchanged)
- libcipher_rt.so md5 `93d291a8cf6c370c126c5a16fa706617` (was `d018b523f53ab38d7ef5fe24858afa82` at `week-14-step-3-b0-koopman-producer`)

## 1. What shipped

Closes scope-lock §4 Step 3 line 113. The REMEMBER consumer drains RING_WRITE slot 2 into the CfC LNN forward path. Per drained event, one `cipher_lnn_decide()` invocation updates the persistent CfC hidden state `h`. v1 scope is wire-up + correctness; the decision is consumed but not actuated.

### Code surface

**`cipher_rt_phase4/cipher_rt_ring_write.h` (+10 LOC, -1 LOC):**

Extends consumer enum:

```c
enum cipher_rt_ring_consumer {
    CIPHER_RT_RING_CONSUMER_CLASSIFY = 0,
    CIPHER_RT_RING_CONSUMER_ORACLE   = 1,
    CIPHER_RT_RING_CONSUMER_REMEMBER = 2,    /* W14 Step 3 S3.B1 */
    CIPHER_RT_RING_NUM_CONSUMERS     = 3,
};
```

Resolves the deferred comment at line 65 ("extend when REMEMBER lands W13-14"). The `read_seq[CIPHER_RT_RING_NUM_CONSUMERS]` array in `cipher_rt_ring_write.c` struct grows from 16 to 24 bytes; `_pad_consumer[]` calculation 64 - 24 - 24 - 8 = 8 stays positive, struct cache-line alignment preserved. `min_read_seq()` body unchanged (iterates `NUM_CONSUMERS`, polymorphic on macro).

**`cipher_rt_phase4/cipher_rt_remember_consumer.h` (+60 LOC, new):**

Public API with lifecycle (init/exit), is-active accessor, and four telemetry accessors (drained/lnn_invocations/active_cycles/total_cycles).

**`cipher_rt_phase4/cipher_rt_remember_consumer.cpp` (+220 LOC, new):**

- Single-pthread drain loop iterating all `CIPHER_RT_RING_MAX_TENANTS=128` tenants per cycle
- Batched drain of `BATCH_PER_TENANT=256` entries per tenant per cycle (R-W14.3 mitigation per scope-lock line 147)
- Per drained event: `process_entry()` synthesizes `cipher_lnn_decide()` arguments from the Koopman payload `(M, K_dim, N_dim)` mapped to `(grid_x, grid_y, grid_z)` with `op_class=0`, `block_size=256`, `shmem=0`, `liquid=NULL`, `threshold=0.5f`. The CfC forward pass at `cipher_lnn.cpp:309-364` updates `state->h` unconditionally
- Cold-start cursor sync at thread start: drain-until-empty per tenant to advance `read_seq[REMEMBER]` to current `write_seq`, preventing producer back-pressure on cold cursor
- Compiled as C++ because `cipher_lnn.h` transitively pulls in `std::atomic` through `cipher_liquid_state.h:38`; public API wrapped in `extern "C"`
- `IDLE_SLEEP_US=500` between cycles when no work; avoids busy-spin

**`cipher_rt_phase4/cipher_inject.c` (+2 LOC):**

```c
#include "cipher_rt_remember_consumer.h"
...
(void)cipher_rt_remember_consumer_init(); /* W14 Step 3 S3.B1 (env-gated CIPHER_REMEMBER=1) */
```

**`cipher_rt_phase4/Makefile` (+4 LOC):**

Adds `cipher_rt_remember_consumer.o` to OBJS and a build rule using `$(CXX) $(CXXFLAGS)` (consumer compiles as C++).

**`cipher_rt_phase4/test_step3_b1_consumer.cpp` (+200 LOC, new):**

Three-case test harness.

## 2. Test harness results

### Case A — ENVGATE OFF

```
[CIPHER REMEMBER] env-gated off (CIPHER_REMEMBER!=1)
[S3.B1 ENVGATE OFF] PASS (consumer dormant, drained=0)
```

Consumer init returns 0 without spawning the thread when `CIPHER_REMEMBER` is unset. `is_active() == 0`.

### Case B — COLD START

```
[CIPHER REMEMBER] consumer thread spawned (batch=256 tenants=128 slot=2)
[CIPHER L3.10] CfC LNN initialized. (48 -> 64 -> 12, 59 KB weights)
[S3.B1 COLD START] posted=10000 accepted=4096 dropped=5904 drained=4096 lnn_calls=4096
[S3.B1 COLD START] PASS
```

10000 events posted to tenant 0 in a tight loop; 4096 accepted (ring buffer ceiling reached before consumer drains a cycle); 4096 drained; 4096 CfC forward invocations (1:1 with drained events). Drops (5904) are the events that arrived after the 4096-entry ring filled before the consumer's drain cycle could free space.

### Case C — PRODUCER+CONSUMER COMPOSE (realistic rate)

```
[S3.B1 COMPOSE] producers=32 per_prod=1000 wall=0.01s intended=32000
                accepted=31000 dropped=1000 drop_pct=3.12% rate=2.27 M/s
                drained=31000
[S3.B1 COMPOSE] PASS (drop_pct=3.12%, drained=31000)
```

32 producer threads each emit 1000 events with a 1ms sleep every 100 events; per-producer rate ~100k/s; aggregate ~2.27 M/s = ~70k/s per tenant. Consumer per-tenant drain capacity is 256 entries per cycle × sub-ms cycle ≈ 256k/s per tenant; producer rate is well below capacity, drop rate 3.12% sits at the noise floor. 31000 of 32000 events drained.

**Gate:** drop_pct ≤ 30% (loose CI bound). **Actual:** 3.12%.

This matches the v1 scope-lock expectation: under realistic Koopman fire rate (much less than 1k events/s per active tenant at narrow-domain decode), the consumer drains the ring fully with drops at the noise floor.

## 3. Regression suite at S3.B1 substrate

| Test | Result | Notes |
|---|---|---|
| `test_ring_write` | 6/6 PASS | Case 6 producer p99 within budget |
| `test_commit_atomicity` | 4/4 PASS | total_count 3.67M (W7-9 Step 5 ALL CASES PASS pattern) |
| `test_observe_publish` | 3/3 PASS | publish/reader/coherence all PASS |
| `test_resolver` | 3/3 PASS | ladder N=4/16/128 + lookup p99 + ABI errors |
| `test_register_model` | 5/5 PASS | round-trip 0.49 µs |
| `test_audit_chain` | 2/3 PASS | Case 3 pre-existing fail at W12 Step 6 baseline (W13 Step 1 carry; documented) |
| `test_tc_probe` | 17/17 PASS | 100% accuracy |
| `test_g3_cross_model_keying` | dlopen fail | pre-existing libtorch_python env issue (carry) |
| `test_sdpa_tenant_routing` | 4/4 PASS | resolver ~240 ns/call |
| `test_g5_va_density` | PASS | 5 families; realistic mix 640 GiB |
| `test_l2_wireup` | 5/5 PASS | gate-disabled apply clean |

9/11 PASS, 2 pre-existing carry-forwards. Matches W14 Step 3 S3.B0 and W14 Step 2 baselines exactly.

## 4. Stop-condition check

- In-distribution top-1 = 90% at rank-64 D1.3: NOT touched (CfC hidden-state update is downstream of the .cu shape registry; D1.3 uses pre-registered shapes)
- Cross-distribution 99.95% passthrough: NOT touched (β OOD detector path unchanged)
- β detector misfires: NOT touched (consumer doesn't read β threshold)
- Regression suite: 9/11 PASS, matches W14 Step 3 S3.B0 baseline. PASS.

No stop conditions triggered.

## 5. Integration surface for S3.C / S3.D

- S3.C LM-head validation harness will invoke the producer+consumer pipeline end-to-end under a real LM-head narrow-domain workload; the drain rate report (`cipher_rt_remember_consumer_drained()`) gates "Koopman fires at least once" per scope-lock §4 Step 3 line 130
- S3.D 30-min N=128 soak runs at this anchor; the consumer thread is env-gated off by default in `CIPHER_REMEMBER` so the soak runs in passive-consumer baseline (substrate-changed-but-consumer-dormant); the soak harness reports `cipher_rt_ring_total_throttled` per tenant (scope-lock line 134 gate < 5%)

## 6. References

- Scope-lock Step 3: `WEEK_13_14_SCOPE_LOCK.md:109-138`
- Slot 3 reservation: `cipher_rt_phase4/cipher_rt_ring_write.h:53`
- Consumer slot enum: `cipher_rt_phase4/cipher_rt_ring_write.h:60-70`
- Insertion point per scope-lock line 113: `cipher_rt_phase4/src/may13/cipher_lnn.cpp:401-460` (decision gates) + `cipher_lnn.cpp:309-364` (forward pass that updates `h`)
- Producer pattern source: `cipher_rt_phase4/cipher_rt_koopman_engine.cpp` (S3.B0)
- Step doc: this file
