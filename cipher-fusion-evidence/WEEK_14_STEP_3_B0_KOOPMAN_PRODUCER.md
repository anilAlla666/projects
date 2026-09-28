# W14 Step 3 S3.B0 — Slot-3 RING_WRITE Producer

**Date:** 2026-05-24
**Tag:** `week-14-step-3-b0-koopman-producer`
**Substrate anchor at close:**
- cipher_rt_phase4 `4abb138` tag `week-14-step-3-b0-koopman-producer`
- cipher-fusion-evidence pre-step doc HEAD `6be1d4f`
- cipher_kmod `8c643fc` tags `week-9-complete` + `week-9-step-5-n128-soak` (unchanged)
- libcipher_rt.so md5 `d018b523f53ab38d7ef5fe24858afa82` (was `20f308d9670d3e82440b0f4c6901b200` at `week-14-step-2-koopman-tier`)

## 1. What shipped

Closes the W14 Step 2 sub-element residue documented in `WEEK_14_STEP_2_KOOPMAN_TIER_ADDENDUM.md`. The slot-3 RING_WRITE producer named at `WEEK_13_14_SCOPE_LOCK.md:99` ("RING_WRITE emission on each Koopman-eligible launch (slot REMEMBER), weak-linked cipher_rt_ring_write, ~30 LOC") did not ship in Step 2; it ships here as substep S3.B0 of W14 Step 3.

### Code surface

**`cipher_rt_phase4/cipher_rt_koopman_engine.cpp` (+38 LOC):**

- Weak-linked extern declaration of `cipher_rt_ring_write` (defensive; the engine builds clean if the ring is absent in some future link config, though v1 Makefile pulls `cipher_rt_ring_write.o` in at line 70)
- Payload struct `cipher_rt_koopman_event { uint32_t M, K_dim, N_dim, op_class; uint64_t ptr_B, ptr_C, reserved; }` with `static_assert(sizeof == 40)` to match `cipher_rt_ring_entry.payload[40]` (`cipher_rt_ring_write.h:44`)
- Producer call sited on the HANDLED success path of `maybe_handle_koopman()`, after the `g_calls_handled` increment and before the verbose log. Fires only when the .cu shape registry returns success (line 118) — i.e., only on actual Koopman substitution, not on PASSTHROUGH or OOD-skip
- `tenant_id=0u` single-tenant default, matching W10 Step 1 classify_observer at `cipher_rt_classify_observer.c:102` and W11+W12 kv_alloc at `cipher_rt_kv_alloc.c:591`. Stream-keyed resolver wire-up is deferred (the W11 Step 2 comment in classify_observer notes the same deferral)
- `event_subtype=1u` (handled), `commit_seq=0u` (matches classify_observer pattern; the W7-9 Step 2 COMMIT framework wire-up is a separate hook deferred per scope-lock §3)
- New telemetry counter `g_remember_emits` and accessor `cipher_rt_koopman_remember_emits()`

**`cipher_rt_phase4/cipher_rt_koopman.h` (+1 LOC):**

- Public accessor `cipher_rt_koopman_remember_emits()` for downstream consumers and the S3.B1 REMEMBER consumer thread

**`cipher_rt_phase4/test_step3_b0_producer.cpp` (+143 LOC, new):**

- Standalone microbench linking against `libcipher_rt.so`
- Test A: P99 publish cadence on a single tenant with 100k calls after 1024 warmup
- Test B: N=128 thread concurrency smoke with 10k events per thread

## 2. Microbench results

### Test A — P99 publish cadence (single-tenant)

```
[S3.B0 P99 CADENCE] N=100000 mean=81 p50=80 p99=82 p999=223 max=15080 (ns)
[S3.B0 P99 CADENCE] PASS
```

**Gate:** p99 ≤ 200 ns (W10 Step 1 RING_WRITE budget 150 ns; +50 ns measurement overhead absorbs std::chrono::steady_clock vs CLOCK_MONOTONIC_RAW).

**Actual:** p99 = 82 ns. Comparable to W10 Step 1 `test_ring_write` Case 6 result (mean 47 ns / p99 61 ns, batched 1000). Single-call measurement here includes more harness overhead.

### Test B — N=128 thread concurrency smoke

```
[S3.B0 N=128 SMOKE] threads=128 per_thread=10000 wall=0.05s intended=1280000
                    accepted=520192 dropped=856736 throttled=0 rate=25.27 M/s
[S3.B0 N=128 SMOKE] PASS (all 1280000 writes issued; ring bookkeeping coherent)
```

**Gate:** all 1.28M writes issued by 128 threads (`g_writes == intended`), ring bookkeeping coherent (accepted + dropped + throttled adds up; no segfault; no abort).

**Actual:** all 1.28M issued. Drops are expected in v1 since no consumer drains the ring; `min_read_seq` stays at 0 for all tenants, so `seq - min_read >= 4096` triggers drop on the 4097th and subsequent writes per tenant. S3.B1 wires the REMEMBER consumer slot 2 and resolves the drop floor.

## 3. Regression suite at S3.B0 substrate (post-rebuild, pre-commit verification)

| Test | Result | Notes |
|---|---|---|
| `test_ring_write` | 6/6 PASS | Case 6 p99=61 ns |
| `test_commit_atomicity` | 4/4 PASS | Case 4 p99=67 ns |
| `test_observe_publish` | 3/3 PASS | publish p99=93 ns |
| `test_resolver` | 3/3 PASS | lookup p99=37 ns |
| `test_register_model` | 5/5 PASS | mean 0.49 µs round-trip |
| `test_audit_chain` | 2/3 PASS | Case 3 pre-existing fail at W12 Step 6 baseline byte-identical .so; documented at W13 Step 1 close |
| `test_tc_probe` | 17/17 PASS | 100% accuracy |
| `test_g3_cross_model_keying` | dlopen fail | pre-existing libtorch_python env issue at W12 Step 6 baseline; documented at W13 Step 1 close |
| `test_sdpa_tenant_routing` | 4/4 PASS | resolver ~240 ns/call |
| `test_g5_va_density` | PASS | 5 families; realistic mix 640 GiB |
| `test_l2_wireup` | 5/5 PASS | gate-disabled apply clean |

9/11 PASS, 2 pre-existing carry-forwards. Matches W14 Step 2 baseline exactly.

## 4. Stop-condition check (per kickoff)

- In-distribution top-1 = 90% at rank-64 D1.3: NOT touched by this step (S3.B0 adds producer emission only; the Koopman .cu kernel and EDMD calibration pipeline are unchanged)
- Cross-distribution 99.95% passthrough: NOT touched by this step (β OOD detector and PASSTHROUGH path unchanged)
- β detector misfires: NOT touched by this step (OOD threshold init at line 99 unchanged; producer emission is downstream of β check)
- Regression suite green: 9/11 PASS, 2 pre-existing carry-forwards. PASS.

No stop conditions triggered.

## 5. What S3.B1 will add

The S3.B1 REMEMBER consumer drain (next substep) will:

1. Extend `CIPHER_RT_RING_NUM_CONSUMERS` from 2 to 3 in `cipher_rt_ring_write.h:65` and add `CIPHER_RT_RING_CONSUMER_REMEMBER = 2` to the enum, addressing the deferred comment at `cipher_rt_ring_write.h:65` ("extend when REMEMBER lands W13-14")
2. Initialize `read_seq[2]` to current `write_seq` at consumer-thread start, so the producer does not back-pressure on a not-yet-running consumer
3. Spawn the consumer thread from `cipher_inject.c` gated by `CIPHER_REMEMBER=1` env var
4. Batch-drain 256 entries per CfC LNN forward (R-W14.3 mitigation per scope-lock line 147)
5. Feed `(M, K_dim, N_dim)` shape and `(ptr_B, ptr_C)` pointers into `cipher_lnn_decide` gate at `cipher_lnn.cpp:428-458`

## 6. References

- Scope-lock sub-element: `WEEK_13_14_SCOPE_LOCK.md:99`
- Step 2 residue record: `WEEK_14_STEP_2_KOOPMAN_TIER_ADDENDUM.md` (cipher-fusion-evidence `6be1d4f`)
- Slot 3 reservation: `cipher_rt_phase4/cipher_rt_ring_write.h:53`
- Engine producer call site: `cipher_rt_phase4/cipher_rt_koopman_engine.cpp` HANDLED path
- Microbench: `cipher_rt_phase4/test_step3_b0_producer.cpp`
- Pattern source: `cipher_rt_phase4/cipher_rt_classify_observer.c:102`
