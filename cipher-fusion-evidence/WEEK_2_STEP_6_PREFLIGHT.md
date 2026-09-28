# Week 2 Step 6 — Pre-flight: Wiring Sites + ioctl ABI + Flow Trace

**HEADLINE STATUS: SCOPE-COMPRESSED.**

Sites 3 (CUPTI) and 4 (ioctl) wire cleanly. Sites 1 (cuBLAS GemmEx) and 2 (SDPA trampolines) have a **structural geometry gap**: both call descriptors live at API layers ABOVE cuLaunchKernel, so neither has `fn`/`grid_*`/`block_*`/`shared_bytes` to feed `cipher_rt_classify_call`. CLASSIFY is fundamentally a cuLaunchKernel-level tool — the natural single wiring site is CUPTI.

**Surfaced for Step-6 scope decision before brief is drafted.**

| site | wiring? | category | notes |
| --- | --- | --- | --- |
| 1 cuBLAS GemmEx | no kernel geometry available | **DESIGN-MISMATCH** | `cipher_rt_matmul_call` exposes M/N/K + A/B/C + types; no `fn`/grid/block. cuBLAS is above cuLaunchKernel. |
| 2 SDPA trampolines | no kernel geometry available | **DESIGN-MISMATCH** | `cipher_rt_attn_call` exposes Q/K/V tensors + dropout + scale; no `fn`/grid/block. ATen op is above cuLaunchKernel. |
| 3 CUPTI callback | grid/block/fn available | **CLEAN-WIRE** | `cipher_cupti.c:134-159` already extracts gridDim+blockDim from `cu/cudaLaunchKernel_v7000_params`. `func` + `sharedMem` also in those structs but not yet read by cipher_cupti.c. |
| 4 ioctl bridge | next-free nr=25; shape sized | **CLEAN-WIRE** | nrs 0/8/9/12/13/21/22 also free, but 25 is the lowest contiguous extension. Reserved nrs 2/3/4 still `-ENOSYS` per cipher_dev.c:297-300 — discipline memory holds. |

**Date:** 2026-05-20
**Tree state:** unchanged. Read-only diagnostic. No source modifications.

---

## Part 1 — Four wiring-site geography

### SITE 1 — cuBLAS GemmEx interception (DESIGN-MISMATCH)

**Files:** `cipher_rt_matmul_dispatch.h:42-68`, `cipher_rt_matmul_dispatch.c:85-116`, `cipher_rt_cublas_shim.c:82+`

`cipher_rt_matmul_call` struct fields:

```c
struct cipher_rt_matmul_call {
    cipher_rt_cublas_handle_t handle;
    int transa, transb, m, n, k;
    const void *alpha, *A;  int Atype; int lda;
    const void *B;          int Btype; int ldb;
    const void *beta;
    void *C;                int Ctype; int ldc;
    int computeType, algo;
    cipher_rt_cuda_stream_t stream;
    uint64_t reserved[4];
};
```

**Fields needed by `cipher_rt_classify_call` but ABSENT here:**
- `const void *fn` (CUDA kernel function pointer)
- `uint32_t grid_x/y/z, block_x/y/z`
- `uint32_t shared_bytes`

**Why the gap:** `cublasGemmEx` is a cuBLAS API entry, not a cuLaunchKernel. Inside `cublasGemmEx`, NVIDIA dispatches one or more CUDA kernels — the launches happen below the visibility of `cipher_rt_cublas_shim.c`. By the time the shim sees the call, the actual kernel geometry is hidden inside the cuBLAS implementation. The classifier expects the geometry; the cuBLAS shim does not have it.

**Inferred op_class:** at this site the op is *definitionally* GEMM (op_class=0). Routing through `cipher_rt_classify_route()` here would either need a synthesized geometry-free call descriptor, or it could short-circuit by hard-coding `op_class=GEMM` without consulting the classifier. The latter is the natural integration but is NOT what Step 6 was scoped to do.

**Calling-thread context:** the workload's own thread (cuBLAS shim runs synchronously from the caller). Safe for mutex.
**Hot-path criticality:** per-GEMM-call (very hot in inference workloads).

### SITE 2 — SDPA trampolines (DESIGN-MISMATCH)

**File:** `cipher_rt_attn_dispatch.cpp` — 3 trampolines (flash L289, efficient L324, cudnn L362). Post-Step-1 LP-2 refactor, each has `int r = route(c); if (r == HANDLED) abort();` followed by `return orig(...)`.

`cipher_rt_attn_call` struct fields:

```c
struct cipher_rt_attn_call {
    enum cipher_rt_attn_backend backend;       /* flash/eff/cudnn */
    struct cipher_rt_attn_tensor q, k, v;       /* Q/K/V tensor descriptors */
    double dropout_p; int is_causal;
    int return_debug_mask, compute_log_sumexp;
    double scale; int scale_is_set;
    const void *attn_bias_data;
    int64_t     attn_bias_sizes[4]; int attn_bias_rank;
    void       *out_status_devptr;  /* Step 1 LP-2 reserved-tail consumer */
    uint64_t    reserved[5];
};
```

**Same gap as Site 1:** no `fn`/grid/block/shared. SDPA dispatcher trampolines run at the ATen op level — *above* the launch of any CUDA kernel inside libtorch's SDPA implementation.

**Inferred op_class:** at this site the op is *definitionally* ATTENTION (op_class=1). Same options as Site 1: synthesize or short-circuit.

**Calling-thread context:** workload's own thread (trampolines run synchronously). Safe.
**Hot-path criticality:** per-SDPA-call.

### SITE 3 — CUPTI callback (CLEAN-WIRE)

**File:** `cipher_cupti.c:134-159` — the per-launch callback already pulls `gridDim` + `blockDim`. Adding `func` + `sharedMem` reads is mechanical.

CUPTI params already define what's needed:

```c
typedef struct cudaLaunchKernel_v7000_params_st {
    const void *func;          /* ← classify_call->fn */
    dim3 gridDim;              /* ← grid_x/y/z */
    dim3 blockDim;             /* ← block_x/y/z */
    void **args;
    size_t sharedMem;          /* ← shared_bytes */
    cudaStream_t stream;
} cudaLaunchKernel_v7000_params;

typedef struct cuLaunchKernel_params_st {
    CUfunction f;              /* ← classify_call->fn */
    unsigned int gridDimX/Y/Z, blockDimX/Y/Z;  /* ← grid/block */
    unsigned int sharedMemBytes;               /* ← shared_bytes */
    CUstream hStream;
    void **kernelParams, **extra;
} cuLaunchKernel_params;
```

Both runtime-API (`cudaLaunchKernel_v7000`, `cudaLaunchKernelExC_v11060`) and driver-API (`cuLaunchKernel`) paths expose the same set, so a single shared classify-call population block can handle all three callback variants.

**Wiring site:** new block immediately after `cipher_rt_smp_observe(...)` at `cipher_cupti.c:161` (existing grid/block read at L134-159). Add a `cipher_rt_classify_call call_desc = {...}` populated from `p->func` / dims / sharedMem, call `cipher_rt_classify_route(&call_desc, &out)`, then `cipher_rt_classify_observer_observe(&call_desc, &out, result)`.

**Calling-thread context:** workload's own thread (CUPTI callback fires synchronously on the thread calling cuLaunchKernel). The pthread_mutex in `cipher_rt_classify_substrate.c` is safe.
**Hot-path criticality:** per-kernel-launch (extreme). The substrate's `route()` is unbatched; the per-call cost is one mutex-snapshot (`g_classify.n_actuators` read), one function pointer call, and three atomic increments. Expected ~100-200 ns.

**Optimization opportunity (defer to a future revision, not Step 6):** the existing `cipher_v2_flush_mask` pattern at L186-188 batches the ioctl every 256 launches; the classify_route call could be sampled at the same rate if per-launch cost is a problem. Step 6 should ship per-launch first; sampling can be added later.

### SITE 4 — ioctl bridge for CIPHER_PUSH_CLASSIFY_STATS (CLEAN-WIRE)

**Ioctl nr survey** (`cipher_kmod/cipher_ioctl.h`):

| nr | name | status |
| ---:| --- | --- |
| 1 | CIPHER_REGISTER_TENANT | live |
| **2** | **CIPHER_SNAPSHOT** | **-ENOSYS (reserved per Cb.2 discipline)** |
| **3** | **CIPHER_RESET** | **-ENOSYS** |
| **4** | **CIPHER_GET_VERSION** | **-ENOSYS** |
| 5 | CIPHER_SUBMIT_GPU_STATE | live |
| 6 | CIPHER_SUBMIT_PROCESS_UTIL | live |
| 7 | CIPHER_SUBMIT_LAUNCH_STATS | live |
| 8 | (free) | — |
| 9 | (free) | — |
| 10-11 | live | |
| 12-13 | (free) | — |
| 14-20 | live | |
| 21-22 | (free) | — |
| 23-24 | live | |
| **25+** | **(free — next-contiguous)** | — |

`-ENOSYS` handlers at `cipher_dev.c:297-300`:

```c
case CIPHER_SNAPSHOT:
case CIPHER_RESET:
case CIPHER_GET_VERSION:
    return -ENOSYS;
```

Per the auto-memory `cipher-abi-rule`: "reserved nrs (2/3/4) return -ENOSYS; new ioctls take fresh nrs." nr=25 is the natural next-contiguous extension.

**Kmod-side dispatch:** `cipher_dev.c:254+` switch over `cmd`. Add a new `case CIPHER_PUSH_CLASSIFY_STATS:` clause that calls a new `cipher_push_classify_stats_ioctl(arg)` handler.

**Userspace caller:** libcipher_rt.so background thread (new) OR piggyback on existing `cipher_v2_flush` ioctl-7 path at `cipher_cupti.c:188`. Either choice is fine; Part 5 recommends timer-based separate thread to decouple from the kernel-launch cadence.

---

## Part 2 — Ioctl ABI sizing

### Payload struct

The observer-side struct (`cipher_rt_classify_observer_stats`) is 4 + 16 + 8 = 28 uint64_t (~224 B):

```c
struct cipher_rt_classify_observer_stats {
    uint64_t total_classifications;     /* offset  0 */
    uint64_t handled;                   /* offset  8 */
    uint64_t passthrough;               /* offset 16 */
    uint64_t per_op_class_counts[16];   /* offset 24..151 */
    uint64_t reserved[8];               /* offset 152..215 */
};                                      /* total 216 B */
```

Wait — recount: 3 head fields + 16 array + 8 reserved = 27 uint64_t × 8 = 216 B. (The "4 + 16 + 8" earlier mis-counted.)

For the ioctl payload, mirror exactly:

```c
struct cipher_push_classify_stats {
    __u64 total;
    __u64 handled;
    __u64 passthrough;
    __u64 per_op_class[16];
    __u64 reserved[8];
};                                      /* 216 B */

#define CIPHER_PUSH_CLASSIFY_STATS \
    _IOW(CIPHER_IOCTL_MAGIC, 25, struct cipher_push_classify_stats)
```

**Alignment:** 8-byte natural for all `__u64`. Total size 216 B is well below the kernel ioctl payload limit (8 KB).
**Endianness:** x86_64 only for now; no swap required. Document but don't enforce until cross-arch ports.
**Packing:** no `__attribute__((packed))` needed — natural layout matches between userspace (`uint64_t`) and kmod (`__u64` is `unsigned long long` on Linux x86_64).
**32/64-bit:** the kmod is built for the host arch; libcipher_rt.so likewise. No compat shim needed.

### Push semantics — surfaced (Part 5 picks)

Two viable semantics:

**SET semantics** — kmod counter ← observer counter (replace):
- Push handler: `atomic64_set(&cipher_classify_stats.total, payload.total)` etc.
- Simpler. kmod counter mirrors observer at last push.
- Downside: loses kmod-side counter independence; concurrent observers (multi-process LD_PRELOAD) would overwrite each other.

**ADD semantics** — kmod counter ← kmod counter + delta:
- Userspace tracks "last pushed values" per counter; pushes deltas.
- Multi-process safe: deltas from N processes accumulate correctly in kmod.
- Downside: userspace must maintain "last pushed" state; adds ~216 B per process for the delta table.

For the v1.2.2 Phase 5 single-tenant scope, **SET semantics are sufficient**. ADD is the future-proof multi-tenant choice.

---

## Part 3 — End-to-end flow trace

| stage | what | exists post-Step-5? | thread context | gap? |
| --- | --- | --- | --- | --- |
| 1 | cuLaunchKernel from workload | yes (CUPTI catches) | workload thread | — |
| 2 | Site 3 calls `cipher_rt_classify_route(call, out)` | **NO — Step 6 wires** | workload thread (CUPTI ENTER) | depends on Step 6 |
| 3 | route() iterates registry, may13_default fires | yes (Step 3) | workload thread | — |
| 4 | may13_default calls `cipher::classify_launch()` → ClassifyResult | yes (Step 3) | workload thread | — |
| 5 | route() returns HANDLED, populates `out` | yes (Step 3) | workload thread | — |
| 6 | Caller calls `cipher_rt_classify_observer_observe(call, out, result)` | **NO — Step 6 wires** | workload thread | depends on Step 6 |
| 7 | observer atomic-increments g_stats counters | yes (Step 4) | workload thread | — |
| 8 | Periodic ioctl push (CIPHER_PUSH_CLASSIFY_STATS) flushes snapshot | **NO — Step 6 designs** | new pthread or piggyback | thread choice = Part 5 |
| 9 | kmod ioctl handler updates `cipher_classify_stats` atomics | **NO — Step 6 implements kmod handler** | per-call ioctl context | — |
| 10 | Operator `cat /proc/cipher/classify_stats` | yes (Step 5) | reader thread | — |

### Latency budget check

The substrate hot-path call (stage 2-7):
- pthread_mutex unused on the read path (only on register; route reads snapshotted n_actuators)
- 1 function-pointer call (may13_default)
- `cipher::classify_launch` cache hit (~0.65 ns per may13 banner) or cold (~1 ns)
- 3 atomic increments in observer

Estimated end-to-end stage-2-through-7 latency: **~100-200 ns per launch** (substrate route + classify + observe). Per may13 design `<100ns` is the cache-hit-only target; we will measure post-wire.

For a typical inference workload at ~10K kernel launches/sec, this adds ~1-2 ms/sec overhead. **Acceptable for Step 6.**

### Verified data flow no-loss

- Stages 7-9 use atomics; no race-condition data loss.
- Stage 8 SET semantics: between two pushes, observer counters accumulate; the next push pushes the *current* observer total, so cumulative counts are correctly carried forward (even though intermediate steps aren't visible to proc).
- Stage 8 ADD semantics: delta is computed from "last pushed" and pushed; same guarantee but explicit.

### `cipher::classify_launch` on deferred-init g_cipher — VERIFIED SAFE

`classify_launch` is `inline` in `cipher_classify.hpp` (per the file's docstring: "Ships as a standalone header — no CUDA dependency"). Pure geometry-based: reads `g.fn` for cache lookup, computes fingerprint from grid/block/shared. Does NOT touch `g_cipher` runtime singleton. Safe to call without `cipher_init()` (which is exactly cipher_rt_phase4's design intent).

---

## Part 4 — Composability per site

### SITE 1 (cuBLAS GemmEx) — DESIGN-MISMATCH

**The gap:** Step 6 cannot mechanically wire `cipher_rt_classify_route()` from `cipher_rt_matmul_dispatch.c` because `cipher_rt_matmul_call` does not carry the geometry the classifier needs.

**Adjudication options (no recommendation):**

- **1A — Synthesize geometry from cuBLAS params** — fabricate `grid_x = m/bx, block_x = bx` for a notional TF32/wmma tile (`bx=128, ty=8`). Result: classifier returns "GEMM" (which it would anyway). The classifier learns nothing it doesn't already know; the synthesis introduces lie-by-omission.
- **1B — Short-circuit: set `out->op_class = GEMM (0)` without calling route()** — Site 1 marks the matmul shim's output as "we know this is GEMM"; observer counts get incremented per shim call. Honest but skips the classifier (defeats the substrate's pluggability).
- **1C — Drop Site 1 from Step 6** — only wire CUPTI (Site 3); cuBLAS shim doesn't carry op_class. Any future cuBLAS-level actuator that wants op-class can consult a per-stream classification cache populated by CUPTI.
- **1D — Restructure matmul_dispatch to carry geometry** — out of Step 6 scope; would need to bridge cuBLAS-level call to kernel-level. Significant.

The CLASSIFY substrate is fundamentally a cuLaunchKernel-level tool (per `cipher_classify.hpp` design — geometry-based classifier). Option **1C** is the cleanest structural choice.

### SITE 2 (SDPA trampolines) — DESIGN-MISMATCH (same shape as 1)

**The gap:** identical to Site 1. `cipher_rt_attn_call` carries Q/K/V tensor shapes; the classifier expects launch geometry.

**Adjudication options (no recommendation):**

- **2A — Synthesize geometry from attention shape** — fabricate a notional grid from `(S, H, D)`. Lie-by-omission again.
- **2B — Short-circuit: set `out->op_class = ATTENTION (1)`** — same honest-but-skip pattern as 1B.
- **2C — Drop Site 2 from Step 6** — only wire CUPTI; SDPA trampolines stay focused on their substrate-actuator role.
- **2D — Restructure attn_dispatch** — out of scope.

Same recommendation rationale as Site 1: **2C** is structurally cleanest.

### SITE 3 (CUPTI) — CLEAN-WIRE

The CUPTI callback at `cipher_cupti.c:134-159` already extracts grid/block. Adding `fn` (`p->func` / `p->config->func`) + `sharedMem` is mechanical. Step 6 lands `cipher_rt_classify_call` population + `cipher_rt_classify_route()` + `cipher_rt_classify_observer_observe()` in ~20 LOC at the existing extraction site.

The existing T4.2.4d enforcement (cuCtxSetCurrent to green) and SM_PACKER observation (cipher_rt_smp_observe) precede the proposed CLASSIFY block — no ordering conflict. Add classify between SMP observe and the existing flush block.

### SITE 4 (ioctl) — CLEAN-WIRE

- Free nr=25 confirmed.
- Payload struct 216 B sized cleanly to atomic64_t array shape on kmod side.
- Reserved nrs 2/3/4 ABI discipline preserved.
- Kmod handler addition is ~20 LOC in `cipher_dev.c` + `cipher_main.c` (push_classify_stats_ioctl function).

---

## Part 5 — Step 6 scope recommendation

**SCOPE-COMPRESSED: drop Sites 1 and 2 (or defer them).**

### Recommended Step 6 scope

**3 wiring landings, not 5:**

1. **CUPTI callback wiring** (Site 3): populate `cipher_rt_classify_call`, call `cipher_rt_classify_route()`, call `cipher_rt_classify_observer_observe()`. ~20 LOC in `cipher_cupti.c`.
2. **Periodic ioctl push** (userspace half of Site 4): new pthread in `cipher_inject.c` init body (or piggyback CIPHER_SUBMIT_LAUNCH_STATS path at `cipher_cupti.c:188`); calls `cipher_rt_classify_observer_snapshot()` and pushes via `ioctl(/dev/cipher, CIPHER_PUSH_CLASSIFY_STATS, &payload)`. ~30 LOC.
3. **Kmod ioctl handler** (kmod half of Site 4): new `case CIPHER_PUSH_CLASSIFY_STATS:` in `cipher_dev.c`, atomic64_set per-field into `cipher_classify_stats`. ~20 LOC.

### Surfaced for Step-6 brief (decision points)

- **Site 1 + Site 2 disposition:** drop entirely from Step 6 (option 1C/2C), OR pick 1B/2B as honest short-circuits that exercise observer paths. Recommend dropping; observers already increment from CUPTI which fires for every cuBLAS-internal launch too (cuBLAS does call cuLaunchKernel under the hood, just hidden from cuBLAS shim — CUPTI sees those launches).

- **Push semantics:** **SET** for v1.2.2 single-tenant scope. ADD is multi-tenant-correct but adds userspace state tracking. The Step 5 proc node will show "last known observer values" with SET semantics — consistent with the "(no producer wired yet)" hint until Step 6, then mirror observer counters thereafter.

- **Push trigger:** **timer-based new pthread** (1 Hz cadence; rate matches the cipher_thermal_feedback NVML sampler pattern at 100ms cadence — too aggressive for classify; classify_stats doesn't need sub-second latency). Alternative: piggyback `cipher_cupti.c:188` flush every 256 launches — couples cadence to launch rate, which is fine.

### Estimated Step 6 LOC

```
cipher_cupti.c            ~20 LOC  (CLASSIFY call site)
cipher_inject.c           ~30 LOC  (push thread or piggyback)
cipher_dev.c              ~10 LOC  (case clause)
cipher_main.c (handler)   ~20 LOC  (push_classify_stats_ioctl impl)
cipher_ioctl.h            ~10 LOC  (struct + _IOW define)
                          ─────────
                           ~90 LOC
```

3-4 hour single-step scope. CP 5.4 + LP-2 + Step-3 substrate registration banners all unchanged; classify counters become non-zero post-wire.

---

## Telemetry on disk

- `cipher_rt_phase4/cipher_rt_matmul_dispatch.{c,h}` — Site 1 call descriptor (no geometry)
- `cipher_rt_phase4/cipher_rt_attn_dispatch.{cpp,h}` — Site 2 call descriptor (no geometry)
- `cipher_rt_phase4/cipher_cupti.c:134-159` — Site 3 wiring landing site (grid/block extraction exists)
- `cipher_rt_phase4/cipher_rt_classify_substrate.h` — substrate API
- `cipher_rt_phase4/cipher_rt_classify_observer.h` — observer API
- `cipher_kmod/cipher_ioctl.h` — ioctl nr space; nr=25 next-free
- `cipher_kmod/cipher_dev.c:254-300` — ioctl dispatch + -ENOSYS for nrs 2/3/4
- `cipher_kmod/cipher_main.c` — kmod-side `cipher_classify_stats` global
- `/usr/include/generated_cuda_meta.h` — CUPTI param structs reference

---

## Discipline notes

- Read-only diagnostic; no source-tree changes.
- The DESIGN-MISMATCH at Sites 1+2 is an inherent property of the substrate design (classifier = cuLaunchKernel-level), not a bug or oversight. Wave 5 §5.5 W2's hot-path code likely wires CLASSIFY only at CUPTI; the brief's 4-site list appears to have been an over-broad enumeration.
- Step 6's payoff (observer counters become non-zero) is preserved at full quality by Site 3 alone — CUPTI observes every cuLaunchKernel including those launched internally by cuBLAS GemmEx and torch SDPA. Dropping Sites 1+2 loses nothing observability-wise.

---

## Awaiting

Step 6 brief, with:
- **3 wiring landings** (Site 3 + ioctl userspace + ioctl kmod) — not 5
- **SET push semantics** (or explicit ADD if multi-tenant safety required)
- **Timer-based or 256-launch-piggyback push trigger** (brief picks one)
- **Sites 1+2 dropped (1C+2C)** or explicitly addressed with op_class short-circuits (1B+2B)

The pre-flight artifact (this doc) gives the Step 6 brief author the data needed to make these structural choices cleanly.
