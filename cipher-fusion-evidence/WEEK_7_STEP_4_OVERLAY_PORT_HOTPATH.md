# Week 7-9 Step 4 — Overlay-op _report() port + COMMIT hot-path wire + token-boundary AUDIT

**Date:** 2026-05-23
**Spec:** CIPHER_REENGINEERING_PLAN.md v1.2.3 §7 W7-9 Step 4
**Locked decisions (user 2026-05-23):**
- Option **β**: per-token-boundary AUDIT (Route a, vLLM execute_model post-step)
- Option **(i)**: port ALL 23 surface entries (19 active, 4 orthogonal-skip)
- Option **A**: ABI 0.6.0 already in place from Step 3

## 1. Anchors

| Tree | Pre-Step-4 anchor | Post-Step-4 anchor | Δ |
|------|-------------------|---------------------|---|
| `cipher_rt_phase4` | `24f906d` (tag `week-7-step-2-commit-primitive`) | **`b39702c`** (tag **`week-7-step-4-overlay-port-hotpath`**) | rotated |
| `cipher_kmod`      | `57d96cc` (tag `week-7-step-3-g6-audit-chain`) | `57d96cc` (unchanged) | none |
| `cipher_vllm_plugin/cipher_vllm_kv.py` (not git-tracked) | md5 `0870311e376004ac83038c4d7e7ea882` | md5 **`70662763468ee871e128ee02ee41e090`** | rotated |
| `cipher_rt_phase4/libcipher_rt.so` | md5 `f858f17f98be4e0e170e373dc897eacf` | md5 **`6afa5a4b7c4c515cbdae8e316b760662`** | rotated |
| `cipher_kmod/cipher_kmod.ko` | md5 `bb42dc1fc8a74d4641030008bd804ecb` | unchanged | none |

## 2. Substrate changes

### 2.1 Snapshot extension (`cipher_rt_commit.h`)

`struct cipher_rt_snapshot`: **9 fields → 24 fields**. The 15 new fields are
overlay-op counter mirrors that the ports read under the seqlock acquire
pattern:

```
determinism_xor              (cipher_determinism_hash)
determinism_count            (cipher_determinism_count)
guard_leak_count             (weak cipher_guard_leak_count)
guard_session_count          (weak cipher_guard_session_count)
loop_runaway_count           (weak cipher_loop_runaway_count)
loop_session_count           (weak cipher_loop_session_count)
pulse_ecc_inject             (weak cipher_pulse_ecc_inject_total)
sense_session_count          (cipher_sense_current_session)
pipeline_edge_count          (weak cipher_pipeline_edge_count)
trace_written                (cipher_trace_written)
trace_dropped                (cipher_trace_dropped)
kernel_table_size            (weak cipher_kt_size)
topo_device_count            (weak cipher_topology_device_count)
struct_override_count        (weak cipher_struct_override_count)
continuity_session_count     (weak cipher_continuity_session_count)
oracle_flops_substituted     (weak cipher_oracle_flops_substituted_total)
oracle_total_decisions       (weak cipher_oracle_total_decisions)
token_boundary_count         (per-tenant ioctl issue counter)
```

Snapshot size **192 B** per tenant (was 72 B); padded to 256 B per slot
across 128 tenants = **32 KiB** total BSS footprint.

### 2.2 C/C++ ABI clash fix

The Step 2 header used `atomic_uint_least64_t` (from `<stdatomic.h>`). The
Step 4 may13 `.cpp` ports include the header; libstdc++'s
`shared_ptr_atomic.h` already brought in `<atomic>` and the two clash via
macro expansion. Resolved by switching `seq` to plain `uint64_t` and using
`__atomic_*_n` GCC builtins (same machine code, both C and C++ safe). All
readers/writers in `cipher_rt_commit.c` and `cipher_rt_snapshot_acquire`
use `__atomic_load_n` / `__atomic_store_n` with explicit memory orders.

### 2.3 `cipher_rt_snapshot_acquire` (header-inline)

Coherent snapshot copy under acquire-fence reader pattern. After the
implementation pass exposed a subtle bug:

> The do-while form `do { ...; if (s1 & 1) continue; memcpy; s2 = load; }
> while (s1 != s2);` is **broken** under odd-s1 retries. The `continue`
> skips both `memcpy` and the `s2` load; the while-clause `s1 != s2`
> then checks against stack garbage from prior iterations. If garbage
> happens to equal the new s1, the loop falsely exits with an
> incoherent (memcpy'd) snapshot.

Rewritten as `for (;;) { ... if (s1 & 1) { if (++spins>1024) bail; continue; }
memcpy; s2 = load; if (s1 == s2) break; }`. Explicit break, no reliance on
stale stack state. Post-loop `__atomic_store_n(&out->seq, s1, RELAXED)`
normalizes the seq field — without `__atomic_store_n` the compiler proves
the store equivalent to memcpy's snap->seq load and elides it (verified
via objdump on the test binary).

### 2.4 `cipher_rt_commit_observe_and_publish` (hot-path writer helper)

Queries each observer accessor (weak-linked; absent → 0), packs a
`cipher_rt_commit_fields`, calls `begin` + `end`. Single seqlock cycle.
Env-gated via `CIPHER_COMMIT_HOTPATH` (default ON).

### 2.5 `cipher_rt_commit_token_boundary` (kmod AUDIT bridge)

Lazy-opens `/dev/cipher` (CAS-cached fd), issues `CIPHER_AUDIT_RECORD`
ioctl (NR 28, ABI 0.6.0) with `actuator_id=COMMIT(1)`. Increments the
per-tenant `g_token_count[]` counter (mirrored into the snapshot
`token_boundary_count` field via the next observe_and_publish).

### 2.6 cuBLAS shim wire (`cipher_rt_cublas_shim.c`)

Post `cipher_rt_matmul_dispatch` return:
```c
cublasStatus_t rc = cipher_rt_matmul_dispatch(&call, g_real_gemmEx);
(void)cipher_rt_commit_observe_and_publish((uint32_t)0);
return rc;
```

### 2.7 SDPA dispatch wire (`cipher_rt_attn_dispatch.cpp`)

Same call at both HANDLED and PASSTHROUGH return points in `route()`.

### 2.8 vLLM per-token AUDIT hook (`cipher_vllm_kv.py`)

Wraps `GPUModelRunner.execute_model` via ctypes call into
`cipher_rt_commit_token_boundary`. One ioctl per decode step. Env-gated
via `CIPHER_TOKEN_AUDIT` (default ON when KV-alloc plugin enabled).

## 3. Overlay-op ports — 19 active

Mechanical pattern per port: `#include "cipher_rt_commit.h"` + 4-line
acquire prelude at `_report()` body entry:

```cpp
struct cipher_rt_snapshot _snap;
cipher_rt_snapshot_acquire(0u, &_snap);
(void)_snap;
```

Anchor port (`cipher_carbon.cpp`) additionally emits `commit_seq` and
`commits_total` in the JSON output to demonstrate the snapshot fields are
consumable downstream. The 18 batched-port files only insert the acquire
prelude (no JSON changes) — this is sufficient to verify the substrate
compiles and links across all 19 ops without rewriting each report's emit
logic per the W7-9 scope. Field-surface JSON expansion is W10-12 work.

### 3.1 Classification (per W7-9 inventory)

| Class | Count | Files |
|-------|-------|-------|
| MECHANICAL | 11 | carbon, continuity, determinism, fairness, guard, loop, oracle_billing (in oracle.cpp), pipeline, pulse, topology, trace + kernel_table:269 |
| NON-MECHANICAL | 2 | oracle (phase detector state), receipt (HMAC chain) |
| SUB-DISPATCHER | 3 | comply (aggregator), dispatch:525 (LAYER3), runtime:107 (F1 root) |
| INFRA-PORTABLE | 3 | green_ctx, l2_persist, sense |
| INFRA-ORTHOGONAL (skipped) | 4 | liquid_state, recipes, telemetry, kernel_table:312 atexit_dump |

The 4 orthogonal skips were locked by the inventory in Part A:
- `liquid_state.cu` — device-local _report, no tenant state coupling
- `recipes.cpp` — registration-time only, no per-tenant counter
- `telemetry.cpp` — HW trajectory (SM clock, ECC), no tenant coupling
- `kernel_table:312` (atexit_dump) — wraps the line-269 entry already
  ported

## 4. Smokes & gates

### 4.1 `test_observe_publish` (new, this step)

| Case | Result | Detail |
|------|--------|--------|
| A — publish overhead 100k iters | **PASS** | mean 81 ns / p50 80 / **p99 95** / max 13 µs (budget p99 ≤ 500 ns) |
| B — reader overhead 100k iters | **PASS** | mean 34 ns / p50 34 / **p99 35** / max 11 µs (budget p99 ≤ 200 ns) |
| C — coherence 1M reads × 1 writer | **PASS** | **0 incoherent** of 1,000,000 reads |

### 4.2 `test_commit_atomicity` (Step 2 regression)

| Case | Result |
|------|--------|
| 1 — 10k commits, final seq even, fields coherent | **PASS** |
| 2 — N=15 × 10k commits, no cross-tenant interference | **PASS** |
| 3 — 1 writer + 2 readers, seqlock coherent | **PASS** |
| 4 — 100k cycles p99 ≤ 200 ns | **PASS** (mean 50 ns / p99 52 / max 10 µs) |

### 4.3 vLLM E.5 — TinyLlama-1.1B B=1 decode

| Config | Mean tok/s | Iters |
|--------|------------|-------|
| Baseline (no cipher) | 2146.22 | [2055.4, 2182.5, 2200.7] |
| Step 4 substrate + REGISTER_MODEL + AUDIT hook | **2136.67** | [2045.5, 2181.3, 2183.2] |
| **Delta** | **−0.45%** | within ±3% HARD STOP ✓ |

REGISTER_MODEL fires (`arch=3 uuid=f4cecfd7b465bf5a2801fb975c4c28eb` for
TinyLlama). AUDIT hook installed log line present. No segfaults, no errors.

### 4.4 vLLM E.5 — Mistral-7B B=1 decode (BLOCKED, environmental)

`VLLM_USE_DEEP_GEMM=0` workaround unblocks TinyLlama but Mistral
**still segfaults during graph capture at baseline** (no cipher injection).
This is environmental — the same Mistral run produced 163.6 tok/s in
Step 3 earlier on 2026-05-23 (same pod, same models directory). Crash
reproduces with `CUDA_INJECTION64_PATH` unset, ruling out Step 4.
Documented as residue; not a Step 4 regression.

## 5. Honest residue

1. **Mistral-7B graph-capture segfault** at baseline (pre-Step-4 reproducible).
   Environmental issue surfaced between the Step 3 successful 163.6 tok/s
   measurement (earlier today) and the Step 4 smoke attempt. Not
   attributable to Step 4. Mistral gate deferred to environment fix.
2. **Multi-tenant routing not wired.** `cipher_v2_current_tenant_id` is not
   exposed; cublas/SDPA hot-path writes use `tenant_id=0` (single-tenant).
   Multi-tenant routing is Step 5 N=128 soak scope.
3. **JSON emit expansion is partial.** Only `cipher_carbon.cpp` emits the
   snapshot's `commit_seq`/`commits_total` in its JSON; the 18 batched-port
   files insert the acquire prelude but don't yet emit new fields. The
   field-surface expansion is W10-12 RING_WRITE work.
4. **Token-boundary AUDIT plugin needs libcipher_rt.so on ldconfig path.**
   When run via `CUDA_INJECTION64_PATH` only, the worker subprocess may not
   resolve `libcipher_rt.so` by name; setting `LD_LIBRARY_PATH=cipher_rt_phase4`
   unblocks it. Without it the hook silently no-ops (documented log line).
5. **observe_and_publish weak-linked accessors return 0** for fields whose
   accessors aren't exported by the observer files (e.g.
   `cipher_carbon_session_count` exists but
   `cipher_guard_leak_count` does not). Step 4 does NOT add those
   accessors; observers expose them as follow-on per-op (W10-12 RING_WRITE).
6. **observe_and_publish call uses tenant_id=0.** The single-tenant scope.
   At N>1 tenants, the call needs the active tenant resolver.

## 6. v1.2.3 §7 W7-9 progression

| Step | Status | Anchor |
|------|--------|--------|
| ✓ Step 1 G10 ABI scaffolding | shipped | kmod 0.5.5 |
| ✓ Step 2 COMMIT primitive core | shipped | cipher_rt_phase4 24f906d |
| ✓ Step 3 G6 kmod-resident AUDIT chain | shipped | kmod 0.6.0 (57d96cc) |
| ✓ **Step 4 23 overlay-ops port + COMMIT hot-path + AUDIT token-boundary** | **SHIPPED** | **cipher_rt_phase4 b39702c** |
| Step 5 — N=128 contention soak → `week-9-complete` | NEXT | — |

## 7. Final fingerprints

```
cipher_rt_phase4         b39702c              tag week-7-step-4-overlay-port-hotpath
cipher_kmod              57d96cc              tag week-7-step-3-g6-audit-chain (unchanged)
cipher_kmod.ko           md5 bb42dc1fc8a74d4641030008bd804ecb (unchanged)
libcipher_rt.so          md5 6afa5a4b7c4c515cbdae8e316b760662
cipher_vllm_kv.py        md5 70662763468ee871e128ee02ee41e090
```
