# Week 2 Step 3 — Telemetry / Substrate Impedance Check

**HEADLINE STATUS: COMPOSE-CLEAN. Step 3 prompt unblocked.**

Step 3's planned scope (a priority-ordered classifier-actuator registry mirroring `cipher_rt_matmul_dispatch.{c,h}`, with no actuator chain yet) is a thin wrapper around the pure-function `cipher::classify_launch()`. The may13 telemetry emission surface (recorder hooks, oracle billing, liquid-state updates, CUPTI/NVML background sampler) fires **downstream of classification**, not within it. The substrate registry has no telemetry consumer responsibility. Zero impedance at the Step 3 boundary.

One informational nuance is documented (§3.4) — not a blocker — about the deferred-init model in `cipher_rt_phase4` and which dispatch.cpp branch Step 6 will wire through.

**Date:** 2026-05-20
**Tree state:** unchanged (`23014c1e` after Step 2 v3 D1 PASS). Read-only diagnostic.

---

## Producer / consumer summary table

| # | producer (may13) | consumer (Step 3+) | category | scope |
| ---:| --- | --- | --- | --- |
| P1 | `cipher::classify_launch(fn, gx,gy,gz, bx,by,bz, shared)` → `ClassifyResult{op, confidence, cache_hit}` | `cipher_rt_classify_dispatch(call, out)` | **CLEAN-COMPOSE** | Step 3 |
| P2 | `cipher_oracle_decide(state, query)` → `CipherOracleResult{decision}` | downstream (not Step 3) | N/A for Step 3 | Step 6 |
| P3 | `cipher_registry_lookup(reg, op_class, sh, arch)` → `const CipherRegistryEntry*` | downstream (not Step 3) | N/A for Step 3 | Step 6 |
| P4 | `cipher_oracle_bill_gemm/nongemm`, `cipher_liquid_record_*` — telemetry side-effects, void | downstream (not Step 3) | N/A for Step 3 | Step 6 |
| P5 | `cipher_telemetry_record_gemm(M, N, K, ns)` — telemetry, void | downstream (not Step 3) | N/A for Step 3 | Step 6 |
| P6 | `cipher_telemetry_sample_sync` + 2 ms background NVML thread | NEVER STARTS in cipher_rt_phase4 (no `cipher_init()` call) | inactive | — |
| P7 | `cipher_edmd_live_collect(...)` — weak undefined | inactive (null pointer; guarded) | inactive | — |
| P8 | `cipher_intercept_stats()` → `CipherInterceptStats*` | H1 stub provides zero counters | already-resolved | — |

8 producer interfaces inspected. **Exactly one** (P1) is in scope for Step 3, and it composes cleanly with the planned consumer signature.

---

## Step 1 — may13 telemetry emission surface

### 1.1 — `cipher_telemetry.{cpp,h}` public surface

5 public functions declared in `include/may13/cipher_telemetry.h`:

| function | signature | role |
| --- | --- | --- |
| `cipher_telemetry_init` | `(state*, device_ordinal, liquid_mgr*) → int` | starts background pthread that writes `CipherHwTrajectory` every 2 ms via CUPTI PM + NVML |
| `cipher_telemetry_destroy` | `(state*) → void` | stops thread, frees state |
| `cipher_telemetry_sample_sync` | `(state*, hw_out*) → int` | force one sync sample (testing only) |
| `cipher_telemetry_report` | `(const state*) → void` | stderr telemetry dump |
| `cipher_telemetry_record_gemm` | `(M, N, K, kernel_start_ns) → void` | per-GEMM MFU bookkeeping, O(1) |

Public struct `CipherTelemetryState` (lines 52-81) contains opaque CUPTI state (512 B), pthread handle, double-buffered `CipherHwTrajectory[2]`, sample counters, NVML handle.

**Threading model:** `cipher_telemetry_init` spawns a pthread at 500 Hz (2 ms cadence). Writes hardware-trajectory snapshot into the `CipherLiquidStateMgr*` passed at init. *Async producer*; consumer reads the liquid state on demand.

### 1.2 — Recorder hooks (`cipher_liquid_record_*`)

6 hooks declared in `include/may13/cipher_liquid_state.h` (lines 213-234):

```c
bool  cipher_liquid_record_substitution(CipherLiquidStateMgr*, int layer);
void  cipher_liquid_record_passthrough (CipherLiquidStateMgr*, int layer);
void  cipher_liquid_update_grad_ema    (CipherLiquidStateMgr*, float norm);
void  cipher_liquid_update_hw          (CipherLiquidStateMgr*, ...);
void  cipher_liquid_record_nccl        (CipherLiquidStateMgr*, ...);
void  cipher_liquid_record_op          (CipherLiquidStateMgr*, uint8_t op);
```

All void/bool returns. **Pure side-effect on the liquid-state ring buffer.** Sync, hot-path compatible (atomics inside). Threading: read by the telemetry background thread; written by `cipher_dispatch.cpp` on every kernel launch.

`cipher_flow_recorder_*` is NOT in the A4 closure — those functions live in `cipher_flow_recorder.cpp` (excluded from the 11-file port). The H1-stub regime doesn't surface them as undef because nothing in our ported set references them. They were a v3-PARTIAL discovery that became moot once the closure stabilized.

### 1.3 — ROOT-variant live-EDMD telemetry

The ROOT `cipher_dispatch.cpp` defines a weak hook at `src/may13/cipher_dispatch.cpp:196`:

```c
extern "C" __attribute__((weak)) bool cipher_edmd_live_collect(
    int M_py, int K_dim, int N_dim,
    int weight_dtype, const void* weight_gpu,
    int activation_dtype, const void* activation_gpu,
    int output_dtype, const void* output_gpu);
```

Called from `edmd_live_post_relaunch_hook()` at L212-240 after `cipher_tls_relaunch()` materializes the ground-truth output:

```c
if (!cipher_edmd_live_collect || !cipher_tls_get_gemm_ptrs
    || !cipher_tls_get_gemm_types) return;
```

In our build, all three are weak-undefined (no provider): nm shows `w cipher_edmd_live_collect`, `w cipher_tls_get_gemm_ptrs`, `w cipher_tls_get_gemm_shape`, `w cipher_tls_relaunch`. The pre-guards short-circuit; **the live-EDMD path is inactive in cipher_rt_phase4 builds by design** (it would only activate if a future actuator provided these symbols). No impedance.

### 1.4 — Dispatch flow telemetry emissions (the actual hot-path side-effects)

`cipher_dispatch()` at `src/may13/cipher_dispatch.cpp:338` — the entry point. The classify-only branch (L339-426) fires:

```c
cipher_oracle_decide(&g_oracle, &oq)         // gate
cipher_registry_lookup(&g_registry, ...)     // recipe table
apply_recipe(entry, desc, layer_idx)         // substitution
cipher_oracle_bill_gemm(&g_oracle, gm, gn, gk, success)   // telemetry
cipher_oracle_bill_nongemm(&g_oracle, success)            // telemetry
```

These are all DOWNSTREAM of classification. The substrate registry Step 3 designs (a wrapper around `classify_launch`) does not emit telemetry — telemetry fires later, in the oracle/registry chain Step 6 will wire.

---

## Step 2 — Planned classifier registry consumer signature

### Reference shape from `cipher_rt_matmul_dispatch.h`

```c
struct cipher_rt_matmul_call {     // call descriptor — filled by shim
    cipher_rt_cublas_handle_t handle;
    int transa, transb, m, n, k;
    const void *alpha, *A, *B, *beta;
    void *C;
    int Atype, Btype, Ctype, lda, ldb, ldc, computeType, algo;
    cipher_rt_cuda_stream_t stream;
    uint64_t reserved[4];
};

enum cipher_rt_matmul_result {
    CIPHER_RT_MATMUL_HANDLED     = 0,
    CIPHER_RT_MATMUL_PASSTHROUGH = 1,
    CIPHER_RT_MATMUL_ERROR       = 2,
};

struct cipher_rt_matmul_actuator {
    const char *name;
    int priority;
    int (*maybe_handle)(const struct cipher_rt_matmul_call *call,
                        int *out_cublas_status);
};

int cipher_rt_matmul_register_actuator(const struct cipher_rt_matmul_actuator *);
int cipher_rt_matmul_dispatch(const struct cipher_rt_matmul_call *,
                               cipher_rt_cublasGemmEx_passthrough_t);
```

Pattern: priority-ordered 16-slot registry; per-actuator callback returns HANDLED / PASSTHROUGH / ERROR; substrate iterates and either dispatches the substitute or falls through to passthrough_fn.

### Step 3 planned shape (per `WEEK_2_SCOPE_LOCK.md` §3 + Wave 5 §5.5 W1 L570-573)

```c
// cipher_rt_classify_substrate.h (planned)
struct cipher_rt_classify_call {        // kernel descriptor
    const void *fn;                      // function pointer
    unsigned    grid_x, grid_y, grid_z;
    unsigned    block_x, block_y, block_z;
    unsigned    shared_bytes;
    uint64_t    reserved[4];
};

struct cipher_rt_classify_out {         // classification result
    uint8_t  op_class;       // OpClass enum value: 0=GEMM..6=ITERATIVE_CUSTOM
    uint8_t  confidence;     // 0-100
    uint8_t  cache_hit;
    uint8_t  reserved_pad[5];
};

enum cipher_rt_classify_result {
    CIPHER_RT_CLASSIFY_HANDLED     = 0,
    CIPHER_RT_CLASSIFY_PASSTHROUGH = 1,
    CIPHER_RT_CLASSIFY_REDIRECTED  = 2,
    CIPHER_RT_CLASSIFY_ERROR       = 3,
};

struct cipher_rt_classifier_t {
    const char *name;
    int         priority;
    int       (*classify)(const struct cipher_rt_classify_call *,
                          struct cipher_rt_classify_out *);
};

int cipher_rt_classify_register(const struct cipher_rt_classifier_t *);
int cipher_rt_classify_dispatch(const struct cipher_rt_classify_call *,
                                struct cipher_rt_classify_out *);
```

No actuator chain in Step 3 — just the registry + dispatch entry. The fallback classifier in Step 3 will wrap `cipher::classify_launch()` (the may13 default classifier) so that `cipher_rt_classify_dispatch()` always produces a result.

---

## Step 3 — Compose check

### P1: `classify_launch()` → `cipher_rt_classify_dispatch()` — CLEAN-COMPOSE

| signal | may13 producer | Step 3 consumer | match |
| --- | --- | --- | --- |
| input fields | `fn, gx,gy,gz, bx,by,bz, shared` | `fn, grid_*, block_*, shared_bytes` | ✓ identical |
| input type | parameter list (7 scalars + ptr) | struct `cipher_rt_classify_call` | trivial struct-pack |
| output fields | `OpClass op, uint8_t confidence, bool cache_hit` | `uint8_t op_class, confidence, cache_hit` | ✓ identical |
| output type | C++ struct `ClassifyResult` | C struct `cipher_rt_classify_out` | trivial conversion (`OpClass→uint8_t` is uint8_t-backed enum) |
| side effects | none (pure function) | none expected | ✓ |
| threading | hot-path, lock-free read | hot-path, lock-free | ✓ |
| latency | ~0.65 ns cache-hit, <1 ns cold | mirror | ✓ |

The Step 3 default classifier:

```c
static int may13_classifier(const struct cipher_rt_classify_call *call,
                            struct cipher_rt_classify_out *out) {
    cipher::ClassifyResult r = cipher::classify_launch(
        call->fn, call->grid_x, call->grid_y, call->grid_z,
        call->block_x, call->block_y, call->block_z,
        call->shared_bytes);
    out->op_class   = static_cast<uint8_t>(r.op);
    out->confidence = r.confidence;
    out->cache_hit  = r.cache_hit ? 1 : 0;
    return CIPHER_RT_CLASSIFY_HANDLED;
}
```

~10 LOC. No shim, no struct repack beyond the obvious 1:1 field copy. **CLEAN-COMPOSE.**

### P2–P5: NOT in Step 3 scope

The oracle/registry/recipe chain and the telemetry recorder hooks all fire **after** classification, in code Step 6 will wire through `cipher_dispatch.cpp`'s classify-only branch. The Step 3 substrate registry has no responsibility for them.

`cipher_oracle_decide`, `cipher_registry_lookup`, `cipher_oracle_bill_*`, `cipher_liquid_record_*`, `cipher_telemetry_record_gemm` are all available at link time (defined in ported may13 sources). They will be reached by Step 6 hot-path wiring, not by Step 3.

### P6: `cipher_telemetry_init` background thread — INACTIVE in cipher_rt_phase4

`cipher_telemetry_init` is called from `cipher_runtime.cpp::cipher_init()` (L65). `cipher_init()` is the only entry that flips `g_cipher.initialized = true`.

Verified: no `cipher_init()` call exists in cipher_rt_phase4's native code (grep for `cipher_init\b` in cipher_rt_phase4/*.c/.cpp excluding src/may13/ → no matches). The runtime is never fully initialized; the telemetry background thread never spawns; the 2 ms NVML sampler is dormant.

This is the cipher_rt_phase4 design intent — keep the may13 brain linked-but-inert until Step 6 wires the classify-only path. **No impedance.**

### P7: `cipher_edmd_live_collect` — weak null in our build

Confirmed `w cipher_edmd_live_collect` in `nm libcipher_rt.so`. The runtime guard at `cipher_dispatch.cpp:213` short-circuits when the pointer is null. **Inactive, no impedance.**

### P8: `cipher_intercept_stats` — H1 stub already provides zero counters

Already in tree at `src/cipher_may13_stubs.cpp`. PURE-LOG consumer at `cipher_runtime.cpp:123`. **Resolved.**

### 3.4 — Informational nuance (not a blocker)

`cipher_dispatch.cpp` has TWO branches based on `g_cipher.initialized`:

- **L339-426 "classify-only path"** — runs when `g_cipher.initialized == false`. Does classify → oracle → registry → apply_recipe → bill, but without the F2/F3/F4/F5 hardware-coupled actuators that need `cipher_init()`. **This is the path Step 6 will wire through.**
- **L428+ "full path"** — runs when `g_cipher.initialized == true`. Would activate green-context split, L2 persist, liquid-state background thread, etc. **cipher_rt_phase4 never reaches this branch** (no `cipher_init()` call).

The two paths are functionally equivalent for classification + dispatch — the classify-only branch does the same `classify_launch` + `cipher_oracle_decide` + `cipher_registry_lookup` + `apply_recipe`. The only difference is the missing hardware-coupled actuators, which cipher_rt_phase4 explicitly excludes per CP 5.4 substrate design (green-ctx is owned by cipher_rt_green_ctx.c, L2-persist by cipher_rt_partition_router.c, etc.).

Step 6 wiring should target the classify-only branch as the entry. This is a Step 6 design choice, not a Step 3 blocker.

---

## Step 4 — Observer-side impedance

`cipher_rt_classify_observer.c` (Step 4 deliverable) is an **empty stub** per `WEEK_2_SCOPE_LOCK.md` §3:

> "Empty stub. Registers against matmul/attn registries in Step 6."

It registers itself as an observer-priority-0 entry into the matmul and attn registries (mirroring `cipher_rt_attn_test_actuator.c` which already registers with priority 0 on the attn substrate). The observer's `maybe_handle` callback returns PASSTHROUGH; its only side effect is incrementing a per-call counter for the proc node (Step 5).

Step 4's observer interface signature is the **same** as the matmul/attn actuator callback signatures (returns PASSTHROUGH, takes the substrate's call descriptor). No new producer-consumer pair, no impedance.

---

## Step 5 — `/proc/cipher/classify_stats` data flow

### Existing pattern in `cipher_kmod/cipher_proc.c`

Six proc nodes already exist (lines 472-537):

```c
proc_create(CIPHER_PROC_STATS,      0444, dir, &cipher_proc_stats_fops);
proc_create(CIPHER_PROC_BAR0,       0444, dir, &cipher_proc_bar0_fops);
proc_create(CIPHER_PROC_GPU_STATE,  0444, dir, &cipher_proc_gpu_state_fops);
proc_create(CIPHER_PROC_FLOPS,      0444, dir, &cipher_proc_flops_fops);
proc_create(CIPHER_PROC_MIGRATIONS, 0444, dir, &cipher_proc_migrations_fops);
proc_create(CIPHER_PROC_ARENAS,     0444, dir, &cipher_proc_arenas_fops);
```

Each node has a `_show` handler that reads from kmod-side state (per-tenant counters, migration ledger, arena state). Userspace pushes data into kmod state via ioctl (no per-kernel ioctl traffic — only batched pushes).

### Step 5 data source for `classify_stats`

The /proc node will report per-class counts (GEMM, ATTENTION, ELEMENTWISE, etc.). Two options for the data source (Step 5 will decide):

- **Option (i)** — kmod-side counter, updated via batched ioctl from userspace (e.g., `CIPHER_PUSH_CLASSIFY_STATS` ioctl that takes a per-class count array, called every N seconds from the substrate). Mirrors how cipher_proc_stats updates today.
- **Option (ii)** — substrate-side counter exposed via `cipher_rt_classify_dispatch_calls_*()` diagnostic accessors (mirroring `cipher_rt_matmul_calls_total/handled/passthrough` in matmul_dispatch.h:120-123). The proc node would then need a sysfs-like path that pulls from the userspace .so — which requires a /dev/cipher ioctl bridge.

Either is feasible; this is a Step 5 design choice. **Not an impedance issue for Step 3** — the substrate doesn't need to know about the proc node design at Step 3 time. Step 3 can add a 16-counter array in the registry (one per priority slot), and Step 5 either reads it directly via ioctl or accumulates pushes.

---

## Step 6 — Recommendation

**COMPOSE-CLEAN.**

The only producer-consumer pair in scope for Step 3 (P1: `classify_launch` → `cipher_rt_classify_dispatch`) is a 1:1 field mapping. The substrate is a thin C-wrapper around a C++ namespace function — about 10-20 LOC for the default classifier registration.

Downstream telemetry emissions (P2-P5) are not in Step 3's scope and will be exercised by Step 6 hot-path wiring with no new impedance constraint — the emissions are void side-effects into may13-internal state (liquid mgr, oracle state, telemetry counters) that cipher_rt_phase4's deferred-init model leaves dormant until and unless Step 6 explicitly calls `cipher_init()` (which the current scope does not require).

The inactive surfaces (P6 background thread, P7 weak-EDMD hook, P8 H1-stubbed intercept_stats) are accounted for and do not require Step 3 design accommodation.

**Step 3 prompt is unblocked.** No shim required, no redesign needed, no adjudication question outstanding.

---

## Telemetry on disk

- `/home/ubuntu/cipher_rt_phase4/cipher_rt_matmul_dispatch.h` — reference substrate shape (read-only)
- `/home/ubuntu/cipher_rt_phase4/include/may13/cipher_classify.hpp` — may13 classify_launch + ClassifyResult definitions
- `/home/ubuntu/cipher_rt_phase4/include/may13/cipher_telemetry.h` — telemetry API surface
- `/home/ubuntu/cipher_rt_phase4/include/may13/cipher_liquid_state.h` — 6 recorder hook decls
- `/home/ubuntu/cipher_rt_phase4/src/may13/cipher_dispatch.cpp` — dispatch flow + edmd_live weak hook at L196-240; classify-only branch L339-426
- `/home/ubuntu/cipher_rt_phase4/src/may13/cipher_runtime.cpp` — cipher_init() at L24 (NOT called from cipher_rt_phase4 native)
- `/home/ubuntu/cipher_kmod/cipher_proc.c` — proc-node emit pattern reference for Step 5
- `/home/ubuntu/cipher-fusion-evidence/WEEK_2_SCOPE_LOCK.md` §3 — Step 3 spec
- `/home/ubuntu/cipher-fusion-evidence/CIPHER_LOGIC_AUDIT_WAVE_5_FUSION_PLAN.md` — Wave 5 §5.5 W1 L570-573 substrate API spec

---

## Discipline notes

- Read-only diagnostic. No source-tree changes.
- No commit on `cipher_rt_phase4`. Tree at `23014c1e` (post Step 2 v3 D1).
- Step 3 prompt can be sent without redesign or shim work.
- The informational nuance in §3.4 (classify-only vs full path) is a Step 6 design surface; Step 3's substrate is path-agnostic and works for both.

---

## Awaiting

Step 3 brief. The substrate registry can be designed as scoped — no impedance-driven design changes required.
