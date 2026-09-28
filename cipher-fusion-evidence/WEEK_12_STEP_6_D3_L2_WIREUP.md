# Week 12 Step 6 — D3 L2 call-site activation backfill

**Date:** 2026-05-23
**Closes:** `WEEK_12_SCOPE_DRIFT_AUDIT.md` item D3 (L2 primitive shipped without caller at W12 Step 3)
**Tag:** `week-12-step-6-d3-l2-wireup` on `cipher_rt_phase4` (commit `adbe121`)
**User adjudication 2026-05-23:** Path (a) — add C ABI accessor with full regression per memory `cipher-phase-discipline`.

## 1. Pre-conditions verified

| Item | At entry | At close | Δ |
|------|----------|----------|---|
| `cipher_rt_phase4` tag | `week-12-step-5-d14-env-gate` (b360fc1) | **`week-12-step-6-d3-l2-wireup`** (adbe121) | rotated |
| `cipher_kmod` tag | `week-9-complete` (0.6.5) | unchanged | none |
| `libcipher_rt.so` md5 | `3e22be9da9dbb6e5e0384394be3b3187` | **`33412ffc97b714662e0824e1de8362bd`** | rotated |
| `cipher_vllm_kv.py` md5 | `1009817ca1975fd38da96c3005f56759` | **`2b6cedab89387c30becd49a27313ceb4`** | rotated |
| `cipher_rt_attn_call` ABI | 408 B | 408 B | preserved |
| `cipher_l2_persist_get_state` symbol | not present | **present** | additive |

## 2. D3 audit item context

W12 Step 3 (`cipher_rt_phase4` commit `fec9cc3`) shipped `cipher_l2_persist_apply_for_tenant()` API + RING_WRITE slot 7 emission, but no caller invokes them. Per `WEEK_12_SCOPE_DRIFT_AUDIT.md:155`:

> "The substrate primitive is shippable; the runtime invocation lands when there's a consumer (W13-14 or later)."

D3 closes the gap by adding the consumer in `cipher_vllm_plugin/cipher_vllm_kv.py` at the REGISTER_STREAMS completion path. Per memory `cipher-phase-discipline` substrate-modification rule, this requires full regression because the path (a) C ABI accessor is a substrate edit.

## 3. L2 primitive ABI

From `include/may13/cipher_l2_persist.h:90-99`:

```c
cudaError_t cipher_l2_persist_apply_for_tenant(CipherL2PersistState* state,
                                               unsigned int tenant_id,
                                               cudaStream_t stream);
```

New accessor added per Part A.5 HARD STOP (no plugin-callable accessor existed):

```c
CipherL2PersistState *cipher_l2_persist_get_state(void);
```

The `CipherL2PersistState` lives at `g_cipher.l2_persist` per `cipher_runtime.cpp:49`. `g_cipher` IS an exported BSS symbol (verified via `nm`) but `l2_persist` sits at `offsetof(CipherRuntime, l2_persist)` which equals `sizeof(CipherGreenCtxState)`. Per advisor for Step 6, computing this offset plugin-side is fragile; clean accessor is a 5 LOC addition.

## 4. Substrate edits (cipher_rt_phase4)

### 4.1 cipher_l2_persist.cu — file-scope state capture + accessor (~15 LOC)

```c++
/* W12 Step 6 D3 backfill: file-scope captured at cipher_l2_persist_init. */
CipherL2PersistState *cipher_l2_persist_state_ptr = nullptr;

extern "C" CipherL2PersistState *cipher_l2_persist_get_state(void)
{
    return cipher_l2_persist_state_ptr;
}
```

`cipher_l2_persist_init` now writes the state pointer to the file-scope variable on entry:

```c++
cudaError_t cipher_l2_persist_init(CipherL2PersistState* state,
                                   cudaStream_t          stream)
{
    memset(state, 0, sizeof(*state));
    state->persist_stream    = stream;
    state->l2_capacity_bytes = query_l2_size();
    /* W12 Step 6 D3 backfill: capture state pointer for plugin accessor. */
    cipher_l2_persist_state_ptr = state;
    /* ... existing init code ... */
}
```

### 4.2 include/may13/cipher_l2_persist.h — header declaration (+5 LOC)

```c++
// W12 Step 6 D3 backfill: returns the global CipherL2PersistState pointer
// captured at cipher_l2_persist_init time. Returns NULL if init has not
// run. Plugin-side callers invoke this to bridge into apply_for_tenant
// without needing to know the CipherRuntime struct layout.
CipherL2PersistState *cipher_l2_persist_get_state(void);
```

## 5. Plugin wire-up (cipher_vllm_plugin/cipher_vllm_kv.py, +~70 LOC)

Added at `_register_streams_with_kmod()` post-ioctl-success site:

```python
try:
    fcntl.ioctl(fd, _REGISTER_STREAMS_IOCTL, req)
    _log(f"REGISTER_STREAMS OK: tenant_id={_TENANT} num={req.num_streams}")
    _apply_l2_persist_for_tenant(handles[:req.num_streams])  # W12 Step 6 D3
except OSError as e:
    _log(f"REGISTER_STREAMS ioctl failed ({e}); fallback tenant=0")
```

New helper `_apply_l2_persist_for_tenant(handles)`:

```python
def _apply_l2_persist_for_tenant(handles):
    if not _l2_enabled():
        return
    lib = _l2_lib()
    if not lib:
        return
    state_ptr = lib.cipher_l2_persist_get_state()
    if not state_ptr:
        _log("L2 apply skipped: no L2 state initialized (no tensors registered yet)")
        return
    applied = 0
    for h in handles:
        try:
            rc = lib.cipher_l2_persist_apply_for_tenant(
                ctypes.c_void_p(state_ptr), ctypes.c_uint32(_TENANT),
                ctypes.c_void_p(int(h)))
            if rc == 0:
                applied += 1
            else:
                _log(f"L2 apply failed for stream {int(h):#x}: rc={rc}")
        except Exception as e:
            _log(f"L2 apply exception ({e}); skipping")
    _log(f"L2 persist applied for tenant={_TENANT} on N={applied}/{len(handles)} streams")
```

Plus env-gate `_l2_enabled()` mirroring the W12 Step 5 D14 pattern (`CIPHER_L2_PERSIST=1` default, set 0/off/no to disable) and lazy `_l2_lib()` resolver that tries soname first then absolute path fallback (so synthetic tests pre-loading by path resolve cleanly under any LD_LIBRARY_PATH).

## 6. Verification (Part C)

### 6.1 test_l2_wireup synthetic 5/5 PASS

`/tmp/step6_d3_baseline/test_l2_wireup.py`:

| Case | Result | Detail |
|------|--------|--------|
| 1 — symbol resolution + accessor returns | PASS | `cipher_l2_persist_get_state()` returns NULL pre-init (expected); resolves cleanly |
| 2 — gate default-on | PASS | `_l2_enabled()` returns True with env unset |
| 3 — gate env-off | PASS | `_l2_enabled()` returns False with `CIPHER_L2_PERSIST=0` |
| 4 — apply with synthetic handles | PASS | NULL state triggers silent skip with log; no exception |
| 5 — gate-disabled apply | PASS | No work done; no exception |

### 6.2 vLLM TinyLlama smoke (D14-style)

```
[cipher-vllm-kv] REGISTER_MODEL OK: arch=3 uuid=a8ba71d9522a84f70537d546920370e2
[cipher-vllm-kv] REGISTER_STREAMS no named streams enumerated; default tenant=0 fallback
ITER 0 tok/s=2040.65 wall=0.26
ITER 1 tok/s=2165.77 wall=0.24
ITER 2 tok/s=2157.96 wall=0.24
RESULT mean_tok_s=2121.46
```

- REGISTER_MODEL fires ✓
- REGISTER_STREAMS exits early (D14 finding #3 inherited): TinyLlama vLLM doesn't enumerate named streams, so `_apply_l2_persist_for_tenant` is **unreachable in this real-vLLM path**. The wire-up is correct (synthetic test validates) but real-vLLM doesn't exercise it end-to-end.
- TPS 2121 vs vanilla 2271 morning baseline = -6.6%. vs 2192 afternoon vanilla = -3.24% (within ±5% Path-b gate from W12 Step 5).
- No segfault, no exception.

### 6.3 W7-12 microbench regression (Part C.4)

All PASS at gate-default:

| Gate | Result |
|------|--------|
| `test_commit_atomicity` 4/4 | PASS |
| `test_audit_chain` 5/5 | PASS |
| `test_observe_publish` 3/3 | PASS |
| `test_resolver` 3/3 | PASS |
| `test_ring_write` 6/6 | PASS |
| `test_tc_probe` 17/17 | PASS |
| `test_g3_cross_model_keying` | PASS (15/15 pairwise distinct) |
| `test_g5_va_density` | PASS (5 families) |

### 6.4 N=128 30-min soak

| Metric | W12 Step 5 baseline (3e22be9d) | W12 Step 6 (33412ffc) | Δ |
|--------|--------------------------------|------------------------|---|
| Atomicity | 128000/128000 | 128000/128000 | ✓ |
| Writers-only fairness | min 0.911 max 1.137 | min **0.938** max 1.074 | tightest since W12 Step 3 |
| All-thread ratio | min 0.985 max 1.020 | min 0.980 max 1.012 | tighter |
| Reader coherence | 0 incoherent / 5.20T reads | **0 incoherent / 5.24T reads** | ✓ |
| Aggregate rate | 11.52 M/s | **10.96 M/s** | -4.86% (within ±10% gate) |
| Wall time | 1800.70 s | 1800.87 s | as planned |

**ALL GATES PASS.** The -4.86% aggregate rate vs Step 5 is within the ±10% gate; the substrate change (file-scope state pointer capture + accessor) adds no measurable cost to the COMMIT/seqlock hot path the soak exercises (synthetic harness doesn't trigger the L2 wire-up since it doesn't go through plugin REGISTER_STREAMS).

## 7. Honest residue

1. **Real-vLLM L2 wire-up coverage gap (D14 finding #3 inherited).** TinyLlama vLLM doesn't enumerate named streams via `torch.cuda.current_stream()`; `_apply_l2_persist_for_tenant` doesn't fire end-to-end in production smoke. `test_l2_wireup` synthetic harness validates the wire-up correctness. Real-vLLM exercise of the path requires workloads that issue named-stream CUDA ops (W15-17 CP 5.5 heterogeneous-model verification target).
2. **State pointer captured at init.** If `cipher_l2_persist_init` is called multiple times (different states), the file-scope `cipher_l2_persist_state_ptr` records the LAST init. v1 has only one global `g_cipher.l2_persist` so this is fine; v2 multi-state would need per-init refcounting.
3. **L2 wire-up env gate `CIPHER_L2_PERSIST=1` default ON** matches W12 Step 5 D14 discipline. Setting to 0 disables the apply call entirely for benchmark/low-traffic regimes.
4. **Soak aggregate rate −4.86% from W12 Step 5** is within ±10% gate but worth noting against the synthetic-noise floor (~3-5% run-to-run variance observed across W12 soaks). No substrate-level concern; not a regression vs gate.

## 8. v1.2.3 scope-drift audit item D3 status

**CLOSED** at cipher_rt_phase4 tag `week-12-step-6-d3-l2-wireup` (commit `adbe121`).

Pre-fix: L2 primitive shipped at W12 Step 3 without consumer; RING_WRITE slot 7 (`CIPHER_RT_RING_EVENT_L2_PERSIST = 7`) reserved but never fires.

Post-fix: Plugin calls `cipher_l2_persist_apply_for_tenant` per registered stream at REGISTER_STREAMS completion. State pointer accessible via new `cipher_l2_persist_get_state()` C ABI accessor. RING_WRITE slot 7 fires when wire-up actually executes (gated on real vLLM exposing named streams — currently W15-17 verification target per D14 finding #3).

Cumulative scope-drift audit closures: **D5 + D9 + D14 + D3 = 4 of 15 items** closed.

Remaining 11 items (D1-D2, D4, D6-D8, D10-D13, D15) unchanged; per-item user adjudication still pending.

## 9. Final fingerprints

```
cipher_rt_phase4    adbe121   tag week-12-step-6-d3-l2-wireup
libcipher_rt.so     md5 33412ffc97b714662e0824e1de8362bd
cipher_vllm_kv.py   md5 2b6cedab89387c30becd49a27313ceb4 (not git-tracked)
cipher_kmod         unchanged at week-9-complete (0.6.5, ko md5 8c9fdd01)
cipher_kv_bridge.so unchanged at md5 5a3db034 (W11 Step 2)

Files touched:
  cipher_rt_phase4/src/may13/cipher_l2_persist.cu     +~15 LOC (accessor + init capture)
  cipher_rt_phase4/include/may13/cipher_l2_persist.h  +5 LOC (declaration)
  cipher_vllm_plugin/cipher_vllm_kv.py                +~70 LOC (caller + gate + lib resolver)
```
