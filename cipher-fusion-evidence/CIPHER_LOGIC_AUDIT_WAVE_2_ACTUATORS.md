# CIPHER Logic Audit — Wave 2: Actuator Substrate (cipher_rt_phase4 + libcipher_v2)

**Scope.** Deep logic-level audit of the deployed Phase-4 runtime substrate
(`/home/ubuntu/cipher_rt_phase4/`, anchor `c2c5d313` per
`cipher-cp25-closed`) and the earlier-version libcipher_v2 substrate
(`/home/ubuntu/libcipher_v2/`, anchor `86618c30`). Companion to Wave 1
(`CIPHER_LOGIC_AUDIT_WAVE_1_CLASSIFIER.md`), which covered the
classifier brain in `/home/ubuntu/cipher-may13-evidence/`.

**Method.** Each file has nine sections (PURPOSE / PUBLIC SURFACE /
CONTROL FLOW / STATE / CONCURRENCY / DEPENDENCIES / LOGIC-AS-CODED VS
LOGIC-AS-DOCUMENTED / CONTRIBUTION TO SYSTEM / FUSION POINTS). Citations
are file-relative line numbers. Numeric constants and behaviour claimed
by the task prompt are verified or refuted in-line and again in the
"Specific Anchors" cross-check at the end.

**Convention for fusion classes.**
PORT-AS-IS = compile with no logic change.
SHIM-REQUIRED = add adapter from classifier output to this actuator's
input, no behavioural change to either side.
REFACTOR-REQUIRED = substantive code change on this side.
VESTIGIAL = retired; not in OBJS.

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_inject.c

### PURPOSE
CUDA-driver injection entry point. The CUDA driver, on `cuInit`, reads
`CUDA_INJECTION64_PATH`, dlopens the named .so, and dlsyms one of
`InitializeInjection` (with table-ptr arg) or `InitializeInjection2`
(no arg). Both are exported here (L64, L72) and route through a single
`pthread_once`-gated init body (L34). The init body brings up every
substrate in declared order and runs the GOT patcher last (L57). Return
value is always 1 — failures degrade individual subsystems without
blocking `cuInit` (L58-60).

### PUBLIC SURFACE
- `int InitializeInjection(void *pfnGetExportTable)` — L64.
- `int InitializeInjection2(void)` — L72.

Both exported with default visibility per the Makefile rule
`cipher_inject.o: ... -fvisibility=default` (Makefile L70).

### CONTROL FLOW
`InitializeInjection*` → `pthread_once(&cipher_v2_init_once, cipher_v2_init_body)`.
`cipher_v2_init_body` runs 14 steps in order (L36-57):

1. `cipher_v2_tenant_register` — Phase 2 (`ioctl nr 1`).
2. `cipher_rt_green_ctx_cp54_init` — CP 5.4 Step 1.3 (`ioctl nr 13`).
3. `cipher_rt_smp_init` — SM_PACKER counters.
4. `cipher_rt_pr_init` — PARTITION_ROUTER.
5. `cipher_v2_cupti_init` — Phase 3 (`ioctl nr 7`) + Phase 4 hot-path glue.
6. `cipher_rt_volt_init` — DVFS.
7. `cipher_rt_matmul_dispatch_init` — matmul substrate.
8. `cipher_rt_marlin_init` — Marlin actuator (env-gated).
9. `cipher_rt_attn_dispatch_init` — attention substrate.
10. `cipher_rt_attn_test_actuator_init` — smoke actuator (env-gated).
11. `cipher_rt_audit_init` — AUDIT (env-gated).
12. `cipher_rt_cublas_shim_register_got` — register cuBLAS trampolines.
13. `cipher_rt_attn_register_got` — register SDPA trampolines.
14. `cipher_rt_got_patch_init` — apply GOT patches to every loaded ELF.

All return values are explicitly cast to void; the init body always
returns 1 to the driver.

### STATE
Single module-static: `cipher_v2_init_once` (L32). No other globals.

### CONCURRENCY
Driver is single-threaded at `cuInit`; `pthread_once` is the belt-and-
suspenders guard against double init via both entrypoints.

### DEPENDENCIES
- INBOUND: CUDA driver. No in-tree callers.
- OUTBOUND: 13 substrate init functions (above). External: pthread.
- kmod ioctls: indirectly via init bodies (nr 1, 7, 13).

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- Phase 4 architecture (`PHASE_4_ARCHITECTURE.md`) and the CP 2.5
  memo agree: init runs on the driver's load thread, GOT patch is the
  last step. Code matches (L55-57).
- Wave-1's `cipher_dispatch.cpp` has no analogous file; the
  pthread_once + 14-step init pattern is new for Wave 2.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING. Without this file, `libcipher_rt.so` does not initialize.

### FUSION POINTS
- (1) Classifier consumption: a classifier-brain port would need an
  init slot between step 7 (matmul substrate) and step 8 (Marlin) so
  the classifier observer registers before the substitution actuators.
- (2) Fallback today: no classifier; substrate inits happen in the
  fixed order above.
- (3) Insertion point: add `cipher_rt_classify_init()` between
  `cipher_rt_matmul_dispatch_init()` and `cipher_rt_marlin_init()`,
  and again between `cipher_rt_attn_dispatch_init()` and
  `cipher_rt_attn_test_actuator_init()`. The classifier registers as
  a priority-5 observer on both substrates (above the priority-0
  AUDIT, below substitution priorities).
- Fusion class: REFACTOR-REQUIRED (small — two new lines plus the
  classifier's own init function). Severity: blocking for the v1
  classifier-brain port; trivial to mechanize.

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_rt_matmul_dispatch.c + .h

### PURPOSE
Phase 4.5 matmul-routing substrate. cuBLAS `cublasGemmEx` calls land
here through the GOT-patched shim; the substrate walks a priority-
ordered registry of actuators (Marlin first today; FP8, dedup, classify
in future). The substrate is the canonical extension point — actuators
plug in via `cipher_rt_matmul_register_actuator()` without touching the
shim. Documented in `.h` L7-26.

### PUBLIC SURFACE
From `.h`:
- `int cipher_rt_matmul_dispatch_init(void)` — L118.
- `int cipher_rt_matmul_register_actuator(const struct cipher_rt_matmul_actuator*)` — L92.
- `int cipher_rt_matmul_dispatch(const struct cipher_rt_matmul_call*, cipher_rt_cublasGemmEx_passthrough_t)` — L113.
- `unsigned long cipher_rt_matmul_calls_total/handled/passthrough(void)` — L121-123.
- Enum `cipher_rt_matmul_result { HANDLED=0, PASSTHROUGH=1, ERROR=2 }` — `.h` L70-74. **3-value enum confirmed.**

### CONTROL FLOW
`cipher_rt_matmul_dispatch_init`: atomic-exchange guard (L42),
register `cipher_rt_matmul_atexit_diag` (L43), log, return 0.
`cipher_rt_matmul_register_actuator`: NULL-validate (L57-58); lock
`g_disp.reg_lock`; bail if registry full (max 16, L62); insertion-sort
by priority (L70-74), preserving stable order; release; log; return 0.
`cipher_rt_matmul_dispatch` (HOT PATH):

```
atomic_fetch_add(&g_disp.total, 1);          /* L91 */
n = g_disp.n_actuators;                       /* L94 — snapshot, NO LOCK */
for (i = 0; i < n; i++) {
    result = g_disp.actuators[i].maybe_handle(call, &status);
    if (result == HANDLED) {
        atomic_fetch_add(&g_disp.handled, 1);
        return status;                        /* L99 */
    }
    if (result == ERROR) {
        cipher_dbg("MATMUL: '%s' ERROR; falling through", name);
        break;                                /* L104 — fall to real cublas */
    }
    /* PASSTHROUGH → next actuator */
}
atomic_fetch_add(&g_disp.passthrough, 1);
return passthrough_fn(... 18 GEMM args ...);  /* L110-115 */
```

Worst-case path: 16 actuators × one `maybe_handle`. With Marlin only,
≤1 hit per call. ERROR semantics: **break to real cublas, do NOT try
remaining actuators** (L104). This differs from the attention
substrate.

### STATE
Module-static `g_disp` (L17-29):
- `inited` (atomic_int)
- `reg_lock` (PTHREAD_MUTEX_INITIALIZER)
- `n_actuators` (int)
- `actuators[16]` (cipher_rt_matmul_actuator[])
- `total/handled/passthrough` (atomic_ulong) — telemetry confirmed.

Mutations: `register_actuator` under `reg_lock`; counters via atomics.

### CONCURRENCY
Registration is mutex-protected. Dispatch is lock-free on the hot path
(reads `n_actuators` as a plain int — append-only post-init means a
torn read can only under-count, never crash, and a non-NULL slot is
guaranteed because registration writes the slot before bumping the
counter). Telemetry counters use C11 atomics.

### DEPENDENCIES
- INBOUND: `cipher_rt_cublas_shim.c::cipher_rt_cublasGemmEx_impl` (L122).
- OUTBOUND: actuator callbacks; the passthrough fn (real cublas).
- kmod ioctls: none directly.
- External: pthread, libc atomics.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- `.h` L100-105 declares the substrate behaviour exactly as coded.
- `AUDIT_REPORT_2026_05_16.md` §2 confirms shipped status.
- Wave 1's `cipher_dispatch.cpp:480` returns `CIPHER_PASS_THROUGH`
  at the end of its hot path; this substrate's equivalent is L109's
  `atomic_fetch_add(&g_disp.passthrough, 1)` + the real-cublas call.
  Behaviourally identical for the passthrough branch; the surrounding
  registry walk is the new piece.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING. Without this file, Marlin (and any future GEMM
substitution) has nowhere to register.

### FUSION POINTS
- (1) Classifier consumption: classifier observer would publish
  `(op_class=GEMM, recipe_id, decision)` to a TLS slot for downstream
  actuators to read.
- (2) Fallback today: each actuator decides independently from the
  call descriptor (Marlin: shape + STABILITY_THRESHOLD).
- (3) Insertion point: a new priority-5 actuator `classify_matmul`
  that calls into `cipher_rt_classify_launch()` (Wave-1 port) and
  publishes the result to TLS. Marlin reads the TLS hint if present
  to short-circuit observation counting. AUDIT reads it to chain the
  decision.
- Contract: the registry append-only invariant and the 3-value enum
  port-as-is.
- Fusion class: PORT-AS-IS for the substrate; SHIM-REQUIRED for
  hooking the classifier observer in. Severity: minor.

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_rt_attn_dispatch.cpp + .h

### PURPOSE
Phase 4.6.1 attention-routing substrate, the SDPA-side twin of
`cipher_rt_matmul_dispatch`. Interposes on libtorch_cpu.so's three SDPA
dispatcher `::call` entries (flash / efficient / cuDNN) via plain
LD_PRELOAD-style trampolines that the GOT patcher installs (CP 2.5).
Each trampoline builds an attn-call descriptor, walks the actuator
registry, and **always falls through to the real backend in T4.6.1**.
File header L12-13: "T4.6.1 ships PASSTHROUGH only. HANDLED/REDIRECTED
paths are wired but exercised by future actuators".

### PUBLIC SURFACE
From `.h`:
- `int cipher_rt_attn_register_actuator(const cipher_rt_attn_actuator*)` — L112.
- `int cipher_rt_attn_dispatch_init(void)` — L120.
- 5 diagnostic accessors (`tramp_calls`, `tramp_fake`, `calls_total/handled/passthrough/redirected/by_backend`) — `.cpp` L258-267, `.h` L122-127.
- `void cipher_rt_attn_register_got(void)` — `.cpp` L411.
- Enum `cipher_rt_attn_result { HANDLED=0, PASSTHROUGH=1, REDIRECTED=2, ERROR=3 }` — `.h` L91-96. **4-value enum confirmed**, incl. REDIRECTED.
- Three hidden trampoline symbols matching the mangled SDPA names (`.cpp` L289, L324, L362). `#pragma GCC visibility push(hidden)` at L286.

### CONTROL FLOW

**Trampoline (flash example, L289-322)**:
```
g_tramp_calls.fetch_add(1);                /* L295: counted unconditionally */
orig = resolve_lazy(g_flash, MANGLED_FLASH); /* L296 → dlopen(libtorch_cpu.so, RTLD_NOLOAD) + dlsym */
if (!orig) abort();                        /* L297-301 self-call would infinite loop */
if (tensors_are_real(q,k,v)) {             /* L303 — skip FakeTensor/Meta */
    try {
        c = build descriptor;
        route(c);                          /* L313 */
    } catch (...) { g_tramp_fake.fetch_add(1); }
} else g_tramp_fake.fetch_add(1);
return orig(q,k,v,...);                    /* L321 — ALWAYS falls through */
```

**`route(call)` — registry walk under snapshot lock (L135-177)**:
```
g_total.fetch_add(1);
g_per_backend[idx].fetch_add(1);          /* by backend tag */
pthread_mutex_lock(&g_reg.mu);             /* L145 */
n = g_reg.n;
memcpy(snap, g_reg.entries, ...);          /* L147 — SNAPSHOT under lock */
pthread_mutex_unlock(&g_reg.mu);           /* L148 — release before calling actuators */
for (i = 0; i < n; ++i) {
    r = snap[i].maybe_handle(&call);
    switch (r) {
    case HANDLED:    g_handled++; return HANDLED;        /* L154-156 */
    case REDIRECTED: g_redirected++; stderr-warn; break; /* L157-165 — treats as passthrough */
    case ERROR:      stderr-warn; break;                  /* L166-169 — continue to next */
    case PASSTHROUGH: default: break;
    }
}
g_passthrough.fetch_add(1); return PASSTHROUGH;          /* L175-176 */
```

**ERROR semantic split confirmed.** matmul ERROR → exit registry walk
(matmul L104 `break`). attn ERROR → break the `switch`, continue the
for-loop (L168 inside switch case + the fall-through goes to next
actuator). This is a divergence between the two substrates.

**HANDLED is discarded by every trampoline** because each trampoline
calls `orig(...)` unconditionally after `route()` returns (L321, L359,
L398). T4.6.1's PASSTHROUGH-only scope is honoured by construction; a
classifier-driven substitution would not currently take effect.

**`resolve_lazy` (L190-205)**: dlopen `libtorch_cpu.so` with RTLD_NOLOAD
(picks up the already-loaded copy), dlsym for the mangled name, CAS
into the atomic slot. CP 2.5 note (L181-188): RTLD_NEXT doesn't work
because libcipher_rt is loaded LAST by the CUDA driver; resolving
through libtorch_cpu's handle directly bypasses our own hidden
trampoline.

**`cipher_rt_attn_register_got` (L411-421)**: registers all three
mangled symbols against the trampoline addresses via
`cipher_rt_got_register()`. No `save_real` (NULL) — substrates use
dlopen+dlsym to resolve originals, not the saved slot.

### STATE
File-static (anonymous namespace):
- `Registry g_reg` — pthread mutex + 16-entry actuator array + count (L63).
- `g_flash`, `g_eff`, `g_cudnn` — atomic function pointers (L65-67).
- `g_init_announced` — once-only init log (L68).
- Telemetry atomics (L70-74): total / handled / passthrough / redirected / per_backend[4].
- Coverage instrumentation (L83-84): `g_tramp_calls`, `g_tramp_fake` — count trampoline entries before any guard, vs FakeTensor exemptions.

### CONCURRENCY
Registration under `g_reg.mu`. Dispatch snapshots the registry under
lock at L145-148, then releases before invoking actuators — actuators
may call back into substrate APIs without lock recursion. All counters
are C11 atomics.

### DEPENDENCIES
- INBOUND: GOT patcher writes addresses into libtorch_cpu's GOT slots.
- OUTBOUND: dlopen libtorch_cpu.so; actuator callbacks; real SDPA ops.
- External: pthread, ATen headers, c10 core, libdl.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- File header L12-13 honest about PASSTHROUGH-only.
- `AUDIT_REPORT_2026_05_16.md:186` and `:289` flag the
  REDIRECTED → PASSTHROUGH treatment explicitly: "substitution
  mechanism deferred until L1 actuator drives the choice". Code
  matches (L157-165).
- Wave-1's `cipher_dispatch.cpp` has no attention path; the analog
  is the matmul/Marlin lane. No cross-tree contract for attn HANDLED
  yet.
- **DRIFT vs matmul substrate**: ERROR semantic is opposite
  (continue-on-ERROR here vs break-on-ERROR there). The .h does not
  flag this difference. A unified fusion classifier would have to
  know which substrate it sits on.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING for telemetry/coverage of attn. **NOT-YET-LOAD-BEARING
for substitution** — every HANDLED is structurally discarded.

### FUSION POINTS
- (1) Classifier consumption: classifier observer publishes
  `(op_class=ATTENTION, decision, backend_hint)` to a TLS slot.
  Actuators (L1 dedup, paged-attention adapter) read it.
- (2) Fallback today: routes by `backend` tag only; HANDLED discarded.
- (3) Insertion point: priority-5 observer actuator that publishes
  TLS hint. **Substitution requires REFACTOR**: the trampoline must
  branch on `route()`'s return value and skip `orig(...)` when HANDLED
  (or call into a hot-substitute path the actuator stored). See
  Wave-1 `apply_recipe` SUBSTITUTE contract.
- Fusion class: SHIM-REQUIRED for observer; REFACTOR-REQUIRED for
  the substitute lane. Severity: blocking for substitute lane (until
  trampolines respect HANDLED), minor for observation.

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_rt_marlin_actuator.c + cipher_rt_marlin.h

### PURPOSE
Marlin INT4 GEMM as the first actuator on the matmul substrate. Gates
eligible `cublasGemmEx` calls (FP16 dtypes, shape constraints), lazy-
quantizes weights on stable observation, dispatches the Marlin kernel
on hit. File header L4-19.

### PUBLIC SURFACE
From `.h`:
- `int cipher_rt_marlin_init(void)` — L26.
- `int cipher_rt_marlin_is_active(void)` — L29.
- `unsigned long cipher_rt_marlin_calls_total/handled/weights_quantized(void)` — L30-32.

### CONTROL FLOW
**`cipher_rt_marlin_init` (L176-208)**:
- Read `CIPHER_MARLIN` env; set `g_enabled`.
- Read `CIPHER_MARLIN_VERBOSE`.
- If not enabled: log + return 0.
- `cipher_rt_marlin_engine_init` — resolve NVRTC + driver + cudart.
- `cipher_rt_matmul_register_actuator(&g_marlin_actuator)` at L198 — priority 10 (L172).
- Log gate constants and return.

**`maybe_handle_marlin` (L56-168) — the hot path** (called from substrate dispatch):

```
if (!g_enabled) return PASSTHROUGH;                                  /* L92 */
if (Atype/Btype/Ctype != CUDA_R_16F) return PASSTHROUGH;             /* L95 */

/* Marlin's M is PyTorch's batch dim — call->n in cuBLAS column-major */
marlin_M = call->n;                                                   /* L101 */
marlin_N = call->m;
marlin_K = call->k;
weight_ptr = call->A;

if (marlin_M <= 0 || marlin_M > 64) return PASSTHROUGH;              /* L106 — M_GATE=64 confirmed */
if (marlin_N < 1024 || marlin_K < 1024) return PASSTHROUGH;          /* L114 */
if ((marlin_K & 127) != 0 || (marlin_N & 63) != 0) return PASSTHROUGH;/* L118 — K%128 + N%64 */

g_calls_total++;

hits = cipher_rt_marlin_engine_observe_weight(weight_ptr);           /* L125 */
if (!is_ready(weight_ptr)) {
    if (hits < STABILITY_THRESHOLD=4) return PASSTHROUGH;            /* L128 — threshold=4 confirmed */
    if (engine_quantize_repack(weight_ptr, K, N) != 0)
        return PASSTHROUGH;                                          /* L132 */
}

if (!engine_lookup(weight_ptr, &B, &S, &K, &N, &G))
    return PASSTHROUGH;                                              /* L146 */

rc = engine_dispatch(call->B, B, S, call->C, M, N, K, G, call->stream);
if (rc != 0) return ERROR;                                           /* L162 — actuator-level ERROR */

*out_status = 0;                /* CUBLAS_STATUS_SUCCESS */
return HANDLED;                                                       /* L167 */
```

### STATE
- `g_enabled` (atomic_int) — `CIPHER_MARLIN` env.
- `g_verbose` (atomic_int) — `CIPHER_MARLIN_VERBOSE`.
- `g_calls_total/handled/skipped` (atomic_ulong) — telemetry.
- `g_marlin_actuator` (static const) — registration descriptor.

### CONCURRENCY
All counters atomic. The Marlin engine handles its own per-weight
mutex (`g_weight_mu`) and per-stream workspace mutex (`g_ws_mu`) in
`marlin_engine.cpp`.

### DEPENDENCIES
- INBOUND: `cipher_rt_matmul_dispatch.c` (via registry callback).
- OUTBOUND: 8 C-ABI engine functions (L32-44) into
  `cipher_rt_marlin_engine.cpp`.
- kmod ioctls: none directly (engine path is pure CUDA driver/runtime).
- External: stdatomic.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- `.h` L9 says "M ≤ 8" but code (L47, L106) gates at M ≤ 64. **DRIFT**:
  header comment is stale; the actuator description in the .c file
  header (L8) says "M ≤ MAX_M_MARLIN = 4*16 = 64", which is the
  correct value.
- File header L13 says STABILITY_THRESHOLD default 4 — matches L48.
- `cipher-cp53-closed` STEP 2A passes with the actuator + the green-
  ctx grid sizing in engine.cpp L828-829. Code matches the memo.
- `cipher-marlin-primary-ctx-pin` documents that the engine pins the
  GEMM dispatch to the primary context when no green ctx is active
  (engine.cpp L1021-1063). Actuator-level passes the stream through;
  the pin is engine-internal. Memo and code agree.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING for the GEMM substitute lane on Mistral-7B-style decode
B≥8 workloads. SUPPORTING (regime-limited) on B=1 — per
`cipher-t46-attn-substrate` and the F1 closeout, "TinyLlama B=1 was a
loop artifact; 3.617× single-instance no-perf-loss RETRACTED".

### FUSION POINTS
- (1) Classifier consumption: would consume `op_class==GEMM` and a
  `recipe_id` from a Wave-1-style registry hit to short-circuit
  STABILITY_THRESHOLD observation. Could also consume `inference_mode`
  to skip Marlin during training (today the env is a coarse on/off).
- (2) Fallback today: shape + dtype gate + own STABILITY counter.
- (3) Insertion point: read a TLS hint published by the classify
  observer actuator (added at priority 5, above audit, below Marlin).
- Fusion class: PORT-AS-IS for the actuator's main logic; SHIM-REQUIRED
  to read the TLS hint. Severity: minor.

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_rt_marlin_engine.cpp + supporting files

### PURPOSE
Distilled Marlin engine from `cipher-may13-evidence/src/cipher_weight_compress.cpp`
(Marlin-only subset). Provides NVRTC compile of 10 kernel entry points,
per-weight lazy quantization (FP16 → INT4+scales via GPU kernels),
host-side Marlin XOR-swizzled repack using LUTs from
`cipher_rt_marlin_perms.h`, and GEMM dispatch with grid=SM_count.
File header L1-18.

### PUBLIC SURFACE
8 C-ABI functions (declared in actuator L32-44):
- `cipher_rt_marlin_engine_init` — L945.
- `cipher_rt_marlin_engine_ensure_compiled` — L955.
- `cipher_rt_marlin_engine_observe_weight` — L960.
- `cipher_rt_marlin_engine_is_ready` — L968.
- `cipher_rt_marlin_engine_lookup` — L976.
- `cipher_rt_marlin_engine_quantize_repack` — L992.
- `cipher_rt_marlin_engine_dispatch` — L999.
- `cipher_rt_marlin_engine_weights_count` — L1065.
- `cipher_rt_marlin_engine_ws_slots` — L1075.

Plus cross-TU symbols:
- `__thread int cipher_rt_in_marlin_quant` (L45) — read by CUPTI callback.
- `unsigned int cipher_rt_green_ctx_sm_count(void)` (extern, L52) — read at dispatch time.

### CONTROL FLOW
**`resolve_api` (L130-199)**: dlopen `libnvrtc.so.13|.12|.so`,
`libcuda.so.1`, `libcudart.so.13|.12|.so` (RTLD_NOLOAD first), dlsym
NVRTC + driver + runtime fn-pointers. Records ok=true if all critical
pointers resolved. Includes CP 5.6 event API for cross-context
ordering (L172-178).

**`ensure_marlin_compiled` (L316-368)**: state machine 0=cold, 1=
compiling, 2=ready, 3=failed. CAS to 1; force CUDA ctx via
`cudaFree(0)`; NVRTC compile sm_90; cuModuleGetFunction for all 10
entry points; set MAX_DYN_SHARED_SIZE=96K per entry; store(2) on
success or store(3) on failure.

**`ensure_quant_compiled` (L433-475)**: parallel state machine for
the inline quant kernels source (`cipher_rt_quant_kernel_src_str`,
L240-308): transpose + per-group scale + pack passes.

**`quantize_fp16_to_int4_groupwise_gpu` (L512-596)**: GPU-side quant.
Allocate transposed-weight scratch buf; launch 3 kernels (transpose,
scales, pack); cudaMemcpy DtoH the INT4 + scales; free scratches.
The DtoH copy is synchronous and serves as the implicit barrier.

**`marlin_repack_host` (L603-695)**: host-side six-step repack —
unpack signed nibbles to unsigned-biased (L611-625); reshape to
(K/16, 16, N/16, 16) and permute (L627-640); apply 1024-element
`cipher_rt_marlin_perm` LUT (L642-651); pack 8 nibbles per int32
(L653-665); apply 64-element `cipher_rt_marlin_scale_perm` per group
(L668-676); HtoD the result.

**`marlin_gemm_launch` (L771-838)**: shape guards (M ≤ 64, N % 64,
K % 128, G ∈ {128, -1}); pick `thread_k/n` by M; lookup the right
cubin entry (L794); fall back to 132 SMs if cuDeviceAttr fails
(L802-803); per-stream workspace via `marlin_ws_for_stream` (L813);
zero the workspace prefix on the launch stream (L817); set grid =
green-ctx SM count if green ctx active, else g_sm_count (L828-829);
launch with block=256, dyn_shared=96K (L831).

**`cipher_rt_marlin_engine_dispatch` (L999-1063)** — the cross-context
fix. When no green ctx active:
- `PrimaryCtxGuard` makes the primary context current.
- `marlin_gemm_launch` runs on the GEMM's launch stream.
- Create an event (0x2 CU_EVENT_DISABLE_TIMING), record on the
  launch stream.
- ~PrimaryCtxGuard restores the caller's context.
- `cuStreamWaitEvent(stream, done, 0)` enqueues the dependency on
  the now-caller-context stream.
- Destroy event.

This is the CP 5.6 P1 fix for finding F1 (cross-context race —
producer on primary, consumer on tenant's context). When a green ctx
IS active, no pin: launch directly with the partition-sized grid.

**`PrimaryCtxGuard` (L873-893)**: RAII guard. Sets
`cipher_rt_in_marlin_quant=1` so the CUPTI callback skips green-ctx
enforcement; saves caller ctx (cuCtxGetCurrent); sets primary as
current (cuCtxSetCurrent); destructor reverses and clears the TLS
flag.

**Per-stream workspace (L702-769)**: 256-slot table, LRU eviction
when full. Hit path is O(slots) but sub-µs. Miss allocates one
`MARLIN_WS_BYTES = 32768/128 * 16 * sizeof(int) = 16384` device
buffer. Comment L716 makes the LRU-safety argument: with slots ≥
max-concurrent streams, eviction victim is always a slot whose stream
was destroyed — race-free without a `cuStreamDestroy` callback.

### STATE
- `g_api` (Api struct, L92-127) — resolved fn ptrs.
- `g_api_mu` — guards resolution.
- `g_marlin_fns[10]` (L212) — cubin table.
- `g_marlin_module`, `g_marlin_state` (L229) — compile state machine.
- `g_quant_module/_fn_*/g_quant_state` (L310-314) — quant compile state.
- `g_weight_mu`, `g_weights` (L489-490) — pointer-keyed quantized
  weight cache.
- `g_sm_count` (L700) — set once at first dispatch.
- `g_ws_slots[256]`, `g_ws_mu`, `g_ws_tick` (L725-727) — per-stream
  workspace registry.
- `g_primary_ctx`, `g_primary_once` (L859-860) — once-only primary
  ctx retain.

### CONCURRENCY
Compile state machines via CAS + 1-ms spin. Weight cache and workspace
registry under separate mutexes. `cipher_rt_in_marlin_quant` is
`__thread`, never raced.

### DEPENDENCIES
- INBOUND: `cipher_rt_marlin_actuator.c` (via 8 C-ABI extern decls).
- OUTBOUND: NVRTC, driver, runtime API; `cipher_rt_green_ctx_sm_count`.
- External: libnvrtc, libcuda, libcudart, libstdc++.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- File header lists the Marlin XOR-swizzled repack lineage; matches.
- `cipher-marlin-primary-ctx-pin` memo: confirmed in CP 2.4 Fix A
  comment block (L1003-1019) and CP 5.6 P1 in the event-wait block
  (L1021-1059). Both the green-ctx-active branch (L1061) and the
  primary-pin branch (L1035-1060) are in the same `dispatch` function.
- `cipher-f1-fullgpu-marlin-broken` memo correctly identifies the
  cross-context race fix; matches code.
- The kernel-source string (L240-308) for the inline quant kernels is
  faithful to the may13 `kQuantSrc`.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING for INT4 GEMM substitution. The primary-ctx + event
trick is a load-bearing correctness fix for the cross-context race
(F1 finding). Without it, the GEMM (producer) and the caller's
consumer race independently.

### FUSION POINTS
- (1) Classifier consumption: could read `op_class==GEMM` to pre-warm
  the NVRTC compile or to skip the compile for non-decode paths.
- (2) Fallback today: own state machines on first eligible call.
- (3) Insertion point: no direct insertion; engine is downstream of
  the actuator. The classifier insertion is at the actuator (see
  Section /cipher_rt_marlin_actuator.c).
- Fusion class: PORT-AS-IS. The engine is logic that doesn't need a
  classifier signal. Severity: none for the engine itself.

(`cipher_rt_marlin_kernel_src.{cpp,h}` and `cipher_rt_marlin_perms.h`
are pure data: NVRTC kernel source string and LUTs. PORT-AS-IS, no
classifier interaction.)

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_rt_volt.c + cipher_rt_volt.h

### PURPOSE
T4.3.1/T4.3.2 DVFS clock-lock actuator. Ports the Phase B path from
`cipher-may13-evidence/src/cipher_volt.cpp`: signal-safe restore,
dlsym-cached NVML fn pointers, probe-then-actuate. Adds a kmod-ioctl
fall-through (T4.3.2) for non-root contexts. File header L4-10.

### PUBLIC SURFACE
- `int cipher_rt_volt_init(void)` — L253.
- `int cipher_rt_volt_mode(void)` — L362 (returns CipherRtVoltMode enum).
- `unsigned int cipher_rt_volt_locked_mhz(void)` — L367.
- `const char *cipher_rt_volt_status_string(void)` — L372.

### CONTROL FLOW

**`cipher_rt_volt_init` (L253-360)**:
- atomic-exchange `init_done` guard.
- Read `CIPHER_VOLT`. Off → return OFF.
- Read `CIPHER_VOLT_MHZ` or `CIPHER_VOLT_BATCH` → `target_mhz`. Batch
  LUT (L54-63): 1→1000, 8→1600, 32→1980, 64→1980, default→0. **LUT confirmed.**
- Clamp [210, 1980]; outside → return OFF (L293). **Range confirmed.**
- `resolve_nvml` (L84-132): dlopen libnvidia-ml.so.1 + dlsym 6 fns;
  `nvmlInit_v2`; `nvmlDeviceGetHandleByIndex_v2`; cache base clock;
  cache driver version.
- `probe_actuation` (L135-147): no-op `SetGpuLockedClocks(cur, cur)`
  + reset.
- If NVML probe succeeds → path=NVML; else `kmod_ioctl_open` +
  `CIPHER_SET_CLOCK_MHZ(0)` probe (L317) → path=KMOD_IOCTL.
- Both fail → DEGRADED, return.
- Install signal handlers BEFORE applying lock (L333) — comment L331
  explains: "so a crash between apply and exit still restores".
- Apply lock via the chosen path. On failure → DEGRADED.
- Store locked_mhz, mode=ACTIVE, log, return ACTIVE.

**Signal handler (`cipher_rt_volt_signal_handler`, L218-236)**:
- Guarded by atomics: only if ACTIVE && locked_mhz != 0 &&
  !atomic_exchange(&restored, 1).
- Calls `cipher_rt_volt_do_restore` (L189-200): NVML path uses
  pre-cached `fn_reset_locked`; ioctl path uses pre-cached fd. Both
  are async-signal-safe.
- Reset to default handler + re-raise.

**Signal coverage (L241)**: `SIGTERM, SIGINT, SIGSEGV, SIGABRT, SIGBUS`. **5 signals confirmed.**
**`atexit` (L250)** also calls restore via `cipher_rt_volt_atexit_restore` (L203-215).

### STATE
- `g_volt` struct (L65-81): init_done, mode, locked_mhz, NVML handle
  + 5 fn ptrs, driver_version, base_clock_mhz, restored flag, path,
  cipher_fd.

### CONCURRENCY
All state via C11 atomics or write-once (fn ptrs after `resolve_nvml`).
Signal handler reads only atomics + pre-cached pointers/fd — async-
signal-safe.

### DEPENDENCIES
- INBOUND: `cipher_inject.c::cipher_v2_init_body` (L43).
- OUTBOUND: libnvidia-ml.so (NVML), `/dev/cipher` ioctl
  `CIPHER_SET_CLOCK_MHZ`.
- kmod ioctl: `CIPHER_SET_CLOCK_MHZ` — defined in `cipher_ioctl.h`.
- External: libdl, signal.h, fcntl, ioctl.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- `cipher-t431-volt-shipped`, `cipher-t432-kmod-volt-ioctl`,
  `cipher-t43-envelope` all agree: 2 paths (NVML direct + kmod ioctl
  nr 10), [210, 1980] MHz clamp, signal-safe restore. Code matches.
- The envelope memo notes 7B+ regime regressions — that is a regime/
  performance claim, not a code-correctness claim; the actuator
  itself works as documented.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING for the +55% tok/W decode lift in the memory-bandwidth-
bound regime (`cipher-t43-envelope`).

### FUSION POINTS
- (1) Classifier consumption: per-batch-shape DVFS would consume
  current batch hint (M for cuBLAS GEMM, B for SDPA Q dim) per
  launch. Today's static env lock would become a launch-time clock
  retarget.
- (2) Fallback today: env-static at init.
- (3) Insertion point: a TLS hint from the classify observer fed
  into a new CUPTI-callback branch that calls
  `nvmlDeviceSetGpuLockedClocks` (or the kmod ioctl) on
  shape-change. NB: NVML set is not async-signal-safe in the strict
  sense but IS thread-safe; would need a per-launch budget probe
  before wiring inline.
- Fusion class: PORT-AS-IS for the static path; REFACTOR-REQUIRED
  for per-launch dynamic clock retarget. Severity: minor (per-launch
  is v1.5+).

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_rt_audit.c + cipher_rt_audit.h

### PURPOSE
Op 9 AUDIT ported from `op31-prod cipher_10ops_impl.cpp`. Maintains
an HMAC-SHA256 tamper-evident hash chain over every operation the
substrate dispatches. Registers priority-0 observers on BOTH the
matmul and attention substrate registries (confirmed). On exit, writes
the ring + chain head to `CIPHER_AUDIT_DUMP` for offline verification
by `audit_verify.py`. File header `.h` L4-19.

### PUBLIC SURFACE
- `int cipher_rt_audit_init(void)` — L178.
- `void cipher_rt_audit_record(uint32_t op_class, uint64_t call_hash, uint8_t decision)` — L78.
- `void cipher_rt_audit_chain_head(uint8_t out[32])` — L199.
- `uint64_t cipher_rt_audit_count(void)` — L206.
- `int cipher_rt_audit_enabled(void)` — L207.
- `size_t cipher_rt_audit_dump(struct cipher_rt_audit_entry *out, size_t max)` — L209.
- Enums `cipher_rt_audit_op_class { MATMUL=0, ATTENTION=1, SUBSTITUTE=2 }` and `cipher_rt_audit_decision { OBSERVED=0, SUBSTITUTED=1 }` (.h L31-40).

### CONTROL FLOW
**`cipher_rt_audit_init` (L178-197)**:
- If already armed: return 0.
- env-gate via `env_on(getenv("CIPHER_AUDIT"))` — accepts 1/y/Y/t/T/o/O (L141-143).
- Zero `chain[32]`, count=0, prev_ts=0. Set `g.armed=1`.
- Register matmul observer (priority 0, name "audit", L132-134).
- Register attn observer (priority 0, name "audit", L135-137).
- **Priority-0 on BOTH substrates confirmed.**

**`cipher_rt_audit_record(op, call_hash, decision)` (L78-97)**:
- Bail if not armed (L81).
- Lock global `g.mu`.
- seq = count++; ts = now_ns(); delta = ts - prev_ts.
- Write entry into ring slot `seq % 8192`.
- prev_ts = ts.
- **`audit_chain_update(g.chain, e)` (L61-76)**: canonical 96-byte
  block (chain[32] || seq[8] || ts[8] || delta[8] || call_hash[8] ||
  op_class[4] || decision[1] || pad[27]); HMAC-SHA256 over the block
  writes the new chain. **HMAC-SHA256 synchronous on hot path under
  global mutex confirmed.**

**Observer callbacks (L101-130)**:
- `audit_matmul_handle`: hash 10-uint64 fingerprint of the call
  (m/n/k/A/B/C ptrs + 3 dtypes + algo, L105-111), record, return
  `MATMUL_PASSTHROUGH`.
- `audit_attn_handle`: hash 8-uint64 fingerprint (backend +
  q/k/v ptrs + q-tensor sizes), record, return `ATTN_PASSTHROUGH`.

**`__attribute__((destructor)) cipher_rt_audit_dump_at_exit` (L148-176)**:
- If `g.armed` and `CIPHER_AUDIT_DUMP` set: open path, write magic
  "CIPHAUD1" || count[8] || chain[32] || min(count, 8192) entries.

### STATE
Single global `g` (L30-37): `armed`, `mu`, `chain[32]`, `count`,
`prev_ts`, `ring[8192]`. **AUDIT_RING=8192 ring confirmed.**

### CONCURRENCY
Single global mutex `g.mu` serializes every record() call. HMAC-SHA256
runs under that lock. The task prompt notes ~50 ns/call; under heavy
multi-thread launch (e.g. vLLM N=16) this single mutex serializes
hot-path launches.

### DEPENDENCIES
- INBOUND: substrate registries (via observer callbacks).
- OUTBOUND: OpenSSL HMAC/SHA256.
- kmod ioctls: none.
- External: libcrypto.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- `.h` L13-15 says HMAC-SHA256 is ~50 ns/call under SHA-NI and "no
  shadow-thread / ring-consumer machinery needed". Code matches —
  the record path is inline.
- AUDIT_KEY (L21-26): first byte 0xC1 + "NEURALDY" suffix matches the
  op31-prod port. Production-comment L20 notes TPM is the target.
- AUDIT_RING=8192 (L28) — fixed-size FIFO ring; older entries
  overwritten on overflow.
- The 96-byte canonical block layout (L57-76) MUST match
  `audit_verify.py` byte-for-byte; the .h L42-52 entry layout is
  declared stable. Drift between record/verify would silently
  invalidate the chain.

### CONTRIBUTION TO SYSTEM
SUPPORTING. Op 9 is an observer-only fusion op (env-gated by default).
Without it, the substrate runs identical to today; with it, the chain
head is the integrity witness.

### FUSION POINTS
- (1) Classifier consumption: would chain `(OP_SUBSTITUTE, recipe_id,
  SUBSTITUTED)` events when Wave-1's recipe lane fires.
- (2) Fallback today: only `OBSERVED` events. `SUBSTITUTED` reserved
  (.h L34, L39) but never emitted.
- (3) Insertion point: any substitute-lane actuator calls
  `cipher_rt_audit_record(SUBSTITUTE, recipe_hash, SUBSTITUTED)` once
  HANDLED is returned. No code change needed — the API is already
  exported.
- Fusion class: PORT-AS-IS for the substrate; SHIM-REQUIRED only for
  upstream call sites in the eventual substitute actuator. Severity:
  none for AUDIT itself.

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_rt_attn_test_actuator.c

### PURPOSE
Smoke-test actuator for the attention substrate (T4.6.1). Registers
at libcipher_rt init if `CIPHER_ATTN_TEST` is set; counts per-backend
calls, logs the first 4, always returns PASSTHROUGH. File header L3-13.

### PUBLIC SURFACE
- `int cipher_rt_attn_test_actuator_init(void)` — L90 (called from
  `cipher_inject.c:47`).

### CONTROL FLOW
`cipher_rt_attn_test_actuator_init` (L90-99):
- Read `CIPHER_ATTN_TEST` env; if unset or not on, log "disabled"
  + return 0.
- Else `cipher_rt_attn_register_actuator(&g_test_actuator)`.

`test_maybe_handle` (L46-70):
- Increment per-backend counter.
- Log first 4 calls with shapes + dtype.
- Always return `CIPHER_RT_ATTN_PASSTHROUGH`.

### STATE
Four atomic_ulongs (L22-25): seen_flash/eff/cudnn + logged.
Actuator descriptor `g_test_actuator` (L72-76) — priority 0.

### CONCURRENCY
All counters atomic. Stateless apart from counters.

### DEPENDENCIES
- INBOUND: `cipher_inject.c:47`.
- OUTBOUND: attention substrate registry.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- Matches its own header L9-13: a coverage probe, not an actuator.
- Like the AUDIT observer, priority 0 means it sees every call before
  any substitution actuator. Both priority-0 observers fire on the
  same call ordering — neither has a documented coordination contract.

### CONTRIBUTION TO SYSTEM
SUPPORTING (env-gated test). Used to verify T4.6.1 substrate sees real
attention dispatch.

### FUSION POINTS
- (1)/(2)/(3): no classifier dependency; pure observer.
- Fusion class: PORT-AS-IS. Severity: none.

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_rt_cublas_shim.c

### PURPOSE
T4.5.1 cuBLAS interception. Originally a `.symver`-tagged LD_PRELOAD
shim; CP 2.5 replaced that with GOT patching. The exported symbol is
now `cipher_rt_cublasGemmEx_impl` (no `cublasGemmEx@libcublas.so.13`
versioned alias). The GOT patcher writes its address into every
loaded module's GOT slot for `cublasGemmEx`. The shim resolves the
real cuBLAS via `dlopen(libcublas.so.13|.12) + dlvsym`. File header
L2-9 + L125-133.

### PUBLIC SURFACE
- `cublasStatus_t cipher_rt_cublasGemmEx_impl(...18 GEMM args...)` — L73.
- `void cipher_rt_cublas_shim_register_got(void)` — L134.
- `unsigned long cipher_rt_cublas_shim_calls(void)` — L142.

### CONTROL FLOW

**`cipher_rt_cublasGemmEx_impl` (L73-123)**:
- Lazy-init real fn pointer via `shim_lazy_init` (L50-71):
  `resolve_real_in("libcublas.so.13|.12", "cublasGemmEx")` — RTLD_NOLOAD
  first, then RTLD_NOW; `dlvsym` then `dlsym` (L40-48).
- If real fn not resolved → return 15 (CUBLAS_STATUS_NOT_SUPPORTED).
- `g_cublasGetStream(handle, &stream)` — best-effort.
- Build `cipher_rt_matmul_call` descriptor (L100-120).
- `cipher_rt_matmul_dispatch(&call, g_real_gemmEx)`.

**`cipher_rt_cublas_shim_register_got` (L134-140)**: one call to
`cipher_rt_got_register("cublasGemmEx", impl, NULL)`. No `save_real`
because the substrate uses dlopen+dlvsym to resolve the real fn
(comment L128-132 explains the lazy-PLT robustness argument).

### STATE
- `g_real_gemmEx` (cublasGemmEx_fn) — resolved by lazy_init.
- `g_cublasGetStream` (cublasGetStream_v2_fn) — same.
- `g_shim_init_done` (atomic_int).
- `g_shim_calls` (atomic_ulong).

### CONCURRENCY
Lazy-init via atomic flag; multiple threads may race resolve but the
real fn is process-wide and the result is the same pointer either way.
Counter atomic.

### DEPENDENCIES
- INBOUND: GOT patcher writes our impl into PyTorch's GOT slot;
  PyTorch calls us instead of cuBLAS.
- OUTBOUND: libcublas, `cipher_rt_matmul_dispatch`.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- CP 2.5 memo: shim no longer exports a versioned `cublasGemmEx`;
  exported symbol is `cipher_rt_cublasGemmEx_impl`. Code matches
  (no `.symver` directive). Makefile L63-65 explicitly drops the
  `cublas_version.map` version-script.
- The cublas.h-free typedefs (L23-31) mean the shim doesn't pull in
  cublas headers — important for compile decoupling.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING. Without this shim, no cuBLAS call ever reaches the
matmul substrate.

### FUSION POINTS
- (1) Classifier consumption: indirectly via the matmul substrate.
- (2) Fallback today: pure passthrough on `g_real_gemmEx`.
- (3) Insertion point: substrate registry, not the shim.
- The Wave-1 note (Wave-1 L252) about `cipher_tls_get_gemm_shape` —
  Tree B's shape is read directly from the call descriptor inside the
  shim; no TLS needed. The shim already has M/N/K at L75-77, exposed
  to actuators via `cipher_rt_matmul_call`.
- Fusion class: PORT-AS-IS. Severity: none.

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_rt_got_patch.c + cipher_rt_got_patch.h

### PURPOSE
CP 2.5 GOT/PLT patcher. `dl_iterate_phdr` walks every loaded ELF
object, parses PT_DYNAMIC relocation tables, and rewrites GOT slots
whose symbol names match registered targets. RELRO-safe via
`/proc/self/maps` page-protection probe + mprotect-write-restore.
File header L2-27.

### PUBLIC SURFACE
- `int cipher_rt_got_register(const char *symname, void *trampoline, void **save_real)` — L58.
- `int cipher_rt_got_patch_apply(void)` — L212.
- `int cipher_rt_got_patch_init(void)` — L222 (apply + log).
- `unsigned long cipher_rt_got_slots_patched(void)`, `_modules_scanned(void)` — L231-232.

### CONTROL FLOW

**`cipher_rt_got_register` (L58-76)**: append target entry under
mutex; max 16 targets (L43); record symname, trampoline, save_real.

**`cipher_rt_got_patch_apply` (L212-220)**: lock + `dl_iterate_phdr(phdr_cb, &ctx)`.

**`phdr_cb` (L128-210)**:
- Locate PT_DYNAMIC (L135-142).
- Walk dynamic entries: collect DT_SYMTAB, DT_STRTAB, DT_JMPREL,
  DT_PLTRELSZ, DT_RELA, DT_RELASZ (L151-160). Comment: glibc
  relocates d_ptr to absolute addresses.
- Walk both `.rela.plt` (JUMP_SLOT) and `.rela.dyn` (GLOB_DAT)
  (L167-208).
- For each Rela entry where type is `R_X86_64_JUMP_SLOT` or
  `R_X86_64_GLOB_DAT` (L176-177): get symbol name from strtab; for
  every registered target with matching name:
  - slot = info->dlpi_addr + r->r_offset.
  - Skip if already trampoline (idempotent).
  - If save_real provided and not yet saved: `*save_real = *slot`.
  - `write_got_slot(slot, trampoline)`.

**`write_got_slot` (L105-124)**:
- Get page address (mask off PAGE_SIZE-1).
- `region_prot(slot)` reads `/proc/self/maps` for the current
  protection flags (L80-102).
- `want = orig | PROT_WRITE`.
- `mprotect(page, pgsz, want)`.
- `*slot = val`.
- Restore only if orig was known AND different from want (L121).
  Comment L117-120: never downgrade to read-only on a guess (a
  later lazy resolution of another slot on the page would fault).

**`cipher_rt_got_patch_init` (L222-229)**: call apply + log
"patch applied — N slots across M modules; T targets registered".

### STATE
- `g_targets[16]` (L52).
- `g_n_targets`, `g_mu`, `g_slots_patched`, `g_modules_scanned`.

### CONCURRENCY
Mutex around register + apply. dl_iterate_phdr itself takes the
runtime linker's lock internally.

### DEPENDENCIES
- INBOUND: `cipher_inject.c:55-57`; `cipher_rt_cublas_shim_register_got`;
  `cipher_rt_attn_register_got`.
- OUTBOUND: link.h, elf.h, /proc/self/maps.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- File header L2-27 describes the algorithm exactly as coded.
- RELRO note matches L80-124. Lazy-PLT note matches L18-23.
- x86-64 / glibc only (RELA + d_ptr relocation) — explicit caveat
  L25-27. Code does not portably support other arches.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING. CP 2.5 retired LD_PRELOAD interposition for
`libcipher_rt.so`; without GOT patching, no shim symbol can be
reached from PyTorch's call sites.

### FUSION POINTS
- (1)/(2)/(3): no classifier dependency; pure mechanism.
- Fusion class: PORT-AS-IS. Severity: none.

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_rt_sm_packer.c + .h

### PURPOSE
T4.2.3 SM_PACKER detection-only build. Counts consecutive small kernel
launches per stream to identify "packing opportunities". Actual
substitution requires a persistent kernel (T4.2.4 PERSIST_ENGINE) —
not yet wired. File header `.h` L3-17.

### PUBLIC SURFACE
- `int cipher_rt_smp_init(void)` — L33.
- `void cipher_rt_smp_observe(gridX,Y,Z, blockX,Y,Z, stream_handle)` — L41.
- `unsigned long cipher_rt_smp_total_launches()`, `_small_launches()`, `_longest_streak()` — L104-117.
- Constant `CIPHER_RT_SMP_SMALL_THRESHOLD = 16 * 256 = 4096` (`.h` L28). **Threshold confirmed.**

### CONTROL FLOW
`smp_init`: zero `g_smp_streams[32]` (L35), log, return.
`smp_observe`:
- atomic-increment total.
- compute total threads = grid·block (L53-54).
- if ≥ threshold: lock; reset streak for matching stream slot;
  unlock; return (L56-67).
- else: increment small_launches.
- lock; find or insert stream slot in 32-slot cache (linear probe).
- bump current_streak; update longest_streak via CAS loop.

### STATE
- `g_smp_streams[32]` (L26) — open-addressed cache; per-slot
  `{handle, current_streak}`.
- `g_smp_lock` (pthread mutex).
- `g_total_launches`, `g_small_launches`, `g_longest_streak`
  (atomic_ulongs).

### CONCURRENCY
Streak cache under `g_smp_lock`. Counters atomic.

### DEPENDENCIES
- INBOUND: `cipher_cupti.c:161`.
- OUTBOUND: none.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- `.h` says detection only (L9-13). Code matches.
- 32 stream slots, threshold = 16 × 256 = 4096 threads — both confirmed.

### CONTRIBUTION TO SYSTEM
SUPPORTING (telemetry only). Not yet wired to a substitution path.

### FUSION POINTS
- (1) Classifier consumption: would feed `slo_priority` or workload
  pattern to decide pack-vs-not.
- (2) Fallback today: counters only.
- (3) Insertion: if a PERSIST_ENGINE ever lands, the observed streak
  is the trigger.
- Fusion class: PORT-AS-IS (with the caveat that streak data is
  unused). Severity: none.

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_rt_partition_router.c + .h

### PURPOSE
T4.2.2 PARTITION_ROUTER. On first observation of each CUDA stream from
the CUPTI launch callback, sets `CU_STREAM_ATTRIBUTE_PRIORITY` (per
launches-quartile) and `CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP` (per
tenant). Subsequent launches on the same stream are an O(1) lock-free
cache check. File header `.h` L4-18.

### PUBLIC SURFACE
- `int cipher_rt_pr_init(void)` — L133.
- `int cipher_rt_pr_observe_stream(void *stream_handle)` — L264.
- `unsigned long cipher_rt_pr_observed_count()`, `_configured_count()` — L281-289.
- Also exports `unsigned int cipher_rt_tenant_handle_u32_for_self(void)` — L75 (used by green_ctx hash-pick fallback).

### CONTROL FLOW
**`pr_init` (L133-151)**: CAS init flag; zero 256-slot table; refresh tenant snapshot; log.
**`pr_observe_stream(handle)` (L264-279)**:
- increment observed_count.
- `cipher_rt_pr_lookup(handle)` (L155-170): linear probe, hash =
  `(handle * 2654435761) & (CIPHER_RT_PR_CACHE_SLOTS-1)`; acquire-load
  stream_handle; if matches AND configured-flag set → return slot.
  If empty slot → return -1.
- if hit: return (hot path, lock-free).
- else: lock `g_pr_lock`; `cipher_rt_pr_configure_locked`; unlock.

**`pr_configure_locked` (L173-262)**:
- Re-check lookup (another thread may have configured concurrently).
- Find empty slot via linear probe; warn-once on capacity exceedance.
- Refresh tenant snapshot if older than 500 ms (TENANT_REFRESH_INTERVAL_NS).
- CP 5.4 Step 1.3 note (L214-218): ARBITRATE ioctl nr 9 retired; SM
  allocation now goes through `CIPHER_CP54_ALLOCATE` at injection-init.
- `derive_priority_from_launches`: launches > 1M → -5 else 0.
- `cuStreamSetAttribute(stream, CU_STREAM_ATTRIBUTE_PRIORITY, val)`.
- `derive_sync_domain_map`: comment L116-128 explains the previous
  per-tenant `tenant_handle % 2` attempt regressed WL14 by -5.7%
  tok/s — current code is no-op (default=0, remote=0).
- `cuStreamSetAttribute(stream, CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, val)`.
- Write stream_handle first, then configured (release order — concurrent
  reader either sees in-progress or fully configured).

**`cipher_rt_tenant_handle_u32_for_self` (L75-85)**: cached tenant
handle for green_ctx hash-pick fallback. Lazily refreshes if 0.

### STATE
- `g_slots[256]` (L48) — `{_Atomic uintptr_t stream_handle, _Atomic int configured}`.
- `g_pr_lock` (pthread mutex).
- `g_observed_count`, `g_configured_count` (atomic_ulongs).
- `g_pr_initialized`, `g_pr_capacity_warned` (atomic_int).
- `g_cached_tenant_handle`, `g_cached_launches_total`,
  `g_tenant_last_refresh_ns` — cached tenant snapshot fields.

### CONCURRENCY
Lookup is lock-free (atomic loads). Configure under `g_pr_lock`.
Tenant cache refreshed under same lock, 500 ms cadence.

### DEPENDENCIES
- INBOUND: `cipher_cupti.c:160`; `cipher_rt_green_ctx.c` reads
  `cipher_rt_tenant_handle_u32_for_self`.
- OUTBOUND: `cuStreamSetAttribute`; `cipher_rt_tenant_query_by_pid`.
- kmod ioctls: indirectly via `cipher_rt_tenant_query_by_pid` (nr 8).

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- 256-slot cache and lock-free dispatch confirmed.
- Sets `CU_STREAM_ATTRIBUTE_PRIORITY` (L226-227) and
  `CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP` (L242-243) confirmed.
- Sync-domain map is currently a no-op (L128-131) — `.h` L9-11
  still implies it is active. **DRIFT**: docs imply per-tenant
  domain spread; code is no-op pending per-tenant access-pattern
  telemetry (comment L116-127).
- ARBITRATE retirement (L214-218) consistent with
  `cipher-cp54-step1-3` memo.

### CONTRIBUTION TO SYSTEM
SUPPORTING. Sets stream priority. Sync-domain spread is dead code
behind the API.

### FUSION POINTS
- (1) Classifier consumption: `slo_priority` / `session_band` would
  set priority directly instead of the launches-quartile heuristic.
- (2) Fallback today: launches > 1M → -5 else 0.
- (3) Insertion: read `slo_priority` from the classifier-published
  TLS hint inside `derive_priority_from_launches` or its caller.
- Fusion class: SHIM-REQUIRED (small). Severity: minor.

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_rt_green_ctx.c + .h

### PURPOSE
Per-process Green Context with a real SM partition. Three modes:
PARTITION (kmod-granted variable-size mask), POOL (kmod-granted),
SHARED (no own ctx; runs on primary, will bind to pool's ctx in CP
5.4 Step 1.3b'). The kmod's CP 5.4 group ledger is the authority; if
unavailable, falls back to a per-pid hash-pick of one 8-SM group.
File header L1-40.

Also exposes Track 3 SC3 Dynamic SM Migration (`cipher_rt_green_ctx_migrate`).

### PUBLIC SURFACE
- `int cipher_rt_green_ctx_cp54_init(void)` — L137.
- `int cipher_rt_green_ctx_ensure(void)` — L224.
- `int cipher_rt_green_ctx_push(void)` — L563.
- `void cipher_rt_green_ctx_pop(void)` — L582.
- `int cipher_rt_green_ctx_make_current(void)` — L207 (T4.2.4d).
- `int cipher_rt_green_ctx_migrate(unsigned new_mask)` — L417 (Track 3 SC3).
- `unsigned cipher_rt_green_ctx_cur_mask(void)` — L395.
- 4 diagnostic accessors: `is_initialized`, `streams_observed`,
  `sm_count`, `group_id`, `handle_voidp`.

### CONTROL FLOW

**`cipher_rt_green_ctx_cp54_init` (L137-203)**:
- atomic-exchange one-shot guard.
- Read `CIPHER_QOS_CLASS` (partition/pool/shared, default shared)
  + `CIPHER_SM_COUNT`.
- `open("/dev/cipher", O_RDWR)`; on failure → return -1 (hash-pick fallback).
- `ioctl(fd, CIPHER_CP54_ALLOCATE, &a)` (ioctl nr 13 confirmed). Fill
  `a.qos_class`, `a.sm_count`.
- Track 3 SC3 opt-in (L171-186): if PARTITION AND `CIPHER_MIGRATABLE`
  is set, also `ioctl(fd, CIPHER_CP54_SUBSCRIBE_MIGRATE, &one)` on
  same fd before close.
- Close fd; cache `g_cp54_qos`, `g_cp54_grp_mask`, `g_cp54_grp_count`;
  set `g_cp54_ok=1`.

**`cipher_rt_green_ctx_ensure` (L224-393)**: 7-step (matches header):
1. `cuDeviceGetDevResource(0, &full_sm_res, CU_DEV_RESOURCE_TYPE_SM)` — L269.
2. `cuDevSmResourceSplitByCount(groups, &nb_groups, &full, &remaining, 0, MIN_SM=8)` — L278. nb_groups = 15 on this H100 per file-header L29-30.
3. Choose selected groups: if CP 5.4 mask available, set bit-by-bit (L290-308); else hash-pick fallback (L313-327).
4. `cuDevResourceGenerateDesc(&desc, selected, nsel)` — L330.
5. `cuGreenCtxCreate(&g_green_ctx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM)` — L339.
6. `cuCtxFromGreenCtx(&g_green_cuctx, g_green_ctx)` — L348.
7. VERIFY via `cuGreenCtxGetDevResource`; abort if SM count equals the
   full set (L368-374).

If `g_cp54_ok && g_cp54_grp_mask == 0` (SHARED): early-return -1
(L260-266) — no green ctx; runs on primary until pool binding lands.

**`cipher_rt_green_ctx_make_current` (L207-222)** — T4.2.4d enforcement:
- atomic-load init succeeded; if not → return -1.
- `cuCtxGetCurrent(&cur)`; if cur == g_green_cuctx → return 0 (cheap).
- Else `cuCtxSetCurrent(g_green_cuctx)`; return 0 on success.

**`cipher_rt_green_ctx_migrate(new_mask)` (L417-561)** — Track 3 SC3:
- Reject if no init, mask=0, or popcount mismatch (L433-449).
- Lock `g_green_lock`.
- Step A drain: `cuCtxSetCurrent(g_green_cuctx)` + `cuCtxSynchronize`.
- Step B build into LOCALS: cuDeviceGetDevResource + split + selected
  + GenerateDesc + cuGreenCtxCreate + cuCtxFromGreenCtx — every failure
  pre-swap aborts with old context intact (L463-507).
- Step C L1 structural verify: count-preserving and not the full set
  (L508-529). `CIPHER_SC3_FAULT=verify` test hook (L517-521).
- Step D atomic swap (no in-flight kernels by construction): retarget
  `g_green_ctx/cuctx/sm_count/group_id/cur_mask` + cuCtxSetCurrent
  to new ctx (L531-541).
- Step E post-commit release of old ctx — failure logged-and-leaked
  per `cipher-incident-2-bar0-exit` rule (L543-554).
- `CIPHER_SC3_FAULT=destroy` test hook (L547-551).

**`cipher_rt_green_ctx_push/pop` (L563-587)**: per-thread
`cuCtxPushCurrent` / `cuCtxPopCurrent` — used by the CUPTI
cuStreamCreate hook (cipher_cupti.c:97-99).

### STATE
File-static:
- `g_green_ctx` (CUgreenCtx), `g_green_cuctx` (CUcontext).
- `g_green_init_attempted/succeeded` (atomic_int).
- `g_streams_observed` (atomic_ulong).
- `g_green_lock` (pthread mutex).
- `g_green_group_id`, `g_green_sm_count`, `g_green_cur_mask` (unsigned).
- `g_cp54_init_done` (atomic_int), `g_cp54_ok`, `g_cp54_qos`,
  `g_cp54_grp_mask`, `g_cp54_grp_count` — CP 5.4 ledger cache.

### CONCURRENCY
Migration takes `g_green_lock`; lookups and `make_current` rely on
the atomic `g_green_init_succeeded` flag for sequential consistency.
Per Step D's comment (L531-534), the tenant is expected to be at a
safe point during migrate — no in-flight kernels — so the CUPTI
launch callback (which reads `g_green_cuctx` via `make_current`) does
not run concurrently.

### DEPENDENCIES
- INBOUND: `cipher_cupti.c:96,123-127` (ensure + push/pop +
  make_current per launch); `cipher_rt_marlin_engine.cpp:52` (reads
  `sm_count` to size GEMM grid).
- OUTBOUND: cuda.h driver APIs, `/dev/cipher` ioctls
  `CIPHER_CP54_ALLOCATE` (nr 13) and `CIPHER_CP54_SUBSCRIBE_MIGRATE`.
- External: cuda.h.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- File header L1-40 enumerates Step 1.3 behaviour and the fallback
  exactly as coded.
- `cipher-cp54-15groups` memo: H100 green-ctx split = 15 8-SM groups
  not 16. CIPHER_RT_GREEN_MAX_GROUPS=16 is array capacity (L63),
  actual live count is whatever the split returns. Memo and code agree.
- `cipher-cp54-step1-3` SHARED tenant returns -1 (no green ctx) —
  matches L260-266.
- `cipher-track3-dsm` memo's "drain → build → L1 verify → swap →
  release" sequence is implemented at L417-561 exactly.
- T4.2.4d "persistent cuCtxSetCurrent per-launch" — `make_current`
  is called from the CUPTI callback at every launch
  (cipher_cupti.c:124).

### CONTRIBUTION TO SYSTEM
LOAD-BEARING. Without `cipher_rt_green_ctx_make_current` called from
the CUPTI callback, PyTorch's NULL-stream launches do NOT execute on
the partition. The T4.2.4c diagnostic (cipher-t424c-partition-not-enforced)
shows this concretely.

### FUSION POINTS
- (1) Classifier consumption: would consume `slo_priority`,
  `session_band`, and `kv_cache_size_mb` from the tenant snapshot to
  size the partition mask. Today these are env-driven
  (CIPHER_QOS_CLASS, CIPHER_SM_COUNT).
- (2) Fallback today: env→ioctl path or pid-hash fallback.
- (3) Insertion: nothing in this file changes — the upstream change
  is in the kmod's CP 5.4 scheduler that decides masks.
- Fusion class: PORT-AS-IS. The kmod is the policy seat for green
  ctx, not this file. Severity: none.

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_tenant.c

### PURPOSE
Tenant identity registration with the kmod. Reads `CIPHER_TENANT_ID`,
opens `/dev/cipher`, issues `CIPHER_REGISTER_TENANT` (ioctl nr 1),
closes. Single function, no global state. File header L1-7.

### PUBLIC SURFACE
- `int cipher_v2_tenant_register(void)` — L21.

### CONTROL FLOW
- Read `CIPHER_TENANT_ID`; if unset → skip with debug log.
- `open(/dev/cipher, O_RDWR)`; failure → log + return -1.
- Build payload: `pid = gettid()` (LWP), `tgid = getpid()`, tenant_id
  string (bounded copy, null-terminated).
- `ioctl(fd, CIPHER_REGISTER_TENANT, &payload)`; failure → log +
  close + return -2.
- Success → log + close + return 0.

### STATE
None.

### CONCURRENCY
Single-thread caller (init body). No concurrency considerations.

### DEPENDENCIES
- INBOUND: `cipher_inject.c:37`.
- OUTBOUND: `/dev/cipher` ioctl nr 1.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- `cipher_v2_internal.h:7` says ioctl nr 1 (REGISTER_TENANT). Matches.
- `pid = gettid()` per `PHASE_4_ARCHITECTURE.md` "Binding deployment
  requirement: per-thread fd". Matches.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING. Without registration, downstream actuators have no
tenant_id to use for per-tenant decisions.

### FUSION POINTS
- (1) Classifier: no direct dependency.
- (2) Fallback: env-driven.
- (3) Insertion: none.
- Fusion class: PORT-AS-IS. Severity: none.

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_cupti.c

### PURPOSE
CUPTI subscriber + kernel-launch callback. Originally Phase 3 Task 5
(populates LAUNCHES column via ioctl nr 7); Phase 4 added per-launch
green-ctx enforcement, partition-router observation, sm_packer
observation, and detailed diagnostic counters by green-ctx category.
File header L1-19.

### PUBLIC SURFACE
- `int cipher_v2_cupti_init(void)` — L223.

### CONTROL FLOW

**`cipher_v2_cupti_init` (L223-298)**:
- Register `atexit` for diag log.
- `open("/dev/cipher", O_WRONLY)` — long-lived fd.
- `cuptiSubscribe(&g_subscriber, cb, NULL)`.
- `cuptiEnableCallback` for 3 launch CBIDs (cudaLaunchKernel_v7000,
  cudaLaunchKernelExC_v11060, cuLaunchKernel) and 5 stream-create
  CBIDs (driver: cuStreamCreate, cuStreamCreateWithPriority; runtime:
  cudaStreamCreate_v3020, cudaStreamCreateWithFlags_v5000,
  cudaStreamCreateWithPriority_v5050).

**`cipher_v2_cupti_cb` (L67-206)** — the hot path:
- If stream-create CBID: `green_ctx_ensure + push` on ENTER, `pop` on
  EXIT, return (L83-102).
- If not API_ENTER: return.
- **L109-127** — T4.2.4d enforcement gate:
  - Skip if `cipher_rt_in_marlin_quant` TLS set (Marlin's own
    quant/repack pin).
  - else `cipher_rt_green_ctx_ensure()`; if 0 →
    `cipher_rt_green_ctx_make_current()`; if 0 → atomic-increment
    `g_ctx_swaps_to_green`.
  - **Per-launch enforcement gated by `cipher_rt_in_marlin_quant` TLS confirmed.**
- L134-159: extract gridDim/blockDim/stream from CB params (3 CBIDs).
- L160: `cipher_rt_pr_observe_stream(stream)`.
- L161: `cipher_rt_smp_observe(grid, block, stream)`.
- L163-164: atomic-increment `g_launches_total`.
- L166-183: diagnostic — classify launch by `cuStreamGetGreenCtx`:
  on null stream / null green / our green / other green; bucket counters.
- L185-188: if not flush-tick (every 256), return.
- Build `cipher_launch_stats` and `ioctl(g_cipher_fd, CIPHER_SUBMIT_LAUNCH_STATS, &ls)`.

### STATE
- 5 atomic launch counters (L43-48): on_green, on_other_grn, on_null_grn,
  on_null_strm, ctx_swaps_to_green.
- `g_launches_total`, `g_grid_ops_total`.
- `g_cipher_fd` (-1 if unopened).
- `g_subscriber` (CUpti_SubscriberHandle).
- Cross-TU extern: `__thread int cipher_rt_in_marlin_quant` (declared L60).

### CONCURRENCY
The callback fires on the workload's own thread. All counters atomic.
The Marlin TLS flag is per-thread.

### DEPENDENCIES
- INBOUND: CUPTI subscribed at init.
- OUTBOUND: many — `cipher_rt_pr_observe_stream`, `cipher_rt_smp_observe`,
  `cipher_rt_green_ctx_ensure/push/pop/make_current/handle_voidp`,
  `cuStreamGetGreenCtx`, ioctl nr 7.
- kmod ioctls: `CIPHER_SUBMIT_LAUNCH_STATS` (nr 7).

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- File header L1-19: NO flush thread (workload thread for anti-spoof);
  flush every 256; degrade quietly. Matches.
- `cipher-t424d-enforcement-fixed` memo: per-launch
  cuCtxSetCurrent(green) enforces partition. Matches L123-127.
- `cipher-cp53-closed` (CP 5.3 STEP 2A): green-ctx grid sizing wired
  via cipher_rt_marlin_engine.cpp reads `green_ctx_sm_count`. Verified
  in marlin_engine.cpp L828.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING. T4.2.4d enforcement is what makes partitioning hold on
NULL-stream launches. Stream-create hook is what binds PyTorch streams
to the green ctx.

### FUSION POINTS
- (1) Classifier consumption: per-launch shape hint + decision could
  be published from this callback to TLS for downstream actuator
  consumption. Tree A's Stage-0 ring write is the analog (per Wave-1
  `cipher_dispatch.cpp:1483-style intercept`). Today's CUPTI gives us
  the call surface but no shape data (CUPTI launch params are
  grid/block + opaque function handle, not GEMM M/N/K).
- (2) Fallback today: counters only.
- (3) Insertion: a classifier observer that runs after the existing
  per-launch enforcement, publishing op_class hint by kernel name.
- Fusion class: REFACTOR-REQUIRED (substantive). The Wave-1 path
  reads M/N/K from `cipher_tls_get_gemm_shape` weak symbol; in Tree B
  the cuBLAS shim already has M/N/K (cuBLAS path) but generic CUPTI
  launches don't. Severity: requires-mitigation. The CUPTI callback
  is the path for non-cuBLAS kernels; for cuBLAS the matmul substrate
  is the better classifier insertion point.

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_rt_tenant.cpp + .h

### PURPOSE
Actuator-side API for tenant-aware fusion. Three access modes:
- Mode 1 (HOT, ioctl) — ~5 µs per call; per-launch decisions.
- Mode 2 (COLD, /proc poll) — 100 ms refresh; ARBITRATE / Stage 2.
- Mode 3 (CACHED, TLS) — sub-µs; per-launch hot path.

File header L1-18. Per-thread fd to `/dev/cipher` per `PHASE_4_BACKLOG.md`
B1 fix (2026-05-13).

### PUBLIC SURFACE
- `int cipher_rt_tenant_open(void)` — L83.
- `void cipher_rt_tenant_close(void)` — L96.
- `int cipher_rt_tenant_query_by_pid(pid_t pid, struct cipher_tenant_snapshot *out)` — L118.
- `int cipher_rt_tenant_query_by_id(const char *tenant_id, struct cipher_tenant_snapshot *out)` — L139.
- `int cipher_rt_tenant_enumerate(struct cipher_tenant_snapshot *out, int max)` — L167.
- `int cipher_rt_tenant_refresh_cached(void)` — L221.
- `const struct cipher_tenant_snapshot *cipher_rt_tenant_cached(void)` — L235.
- `uint64_t cipher_rt_tenant_cached_age_ns(void)` — L241.

Ioctl nr 8 baked locally: `CIPHER_GET_TENANT_SNAPSHOT _IOWR('C', 8, struct cipher_tenant_snapshot_query)` (L26-28).

### CONTROL FLOW
**Mode 1 query**: `fd_ensure` (lazy per-thread open at L109-116); fill
`q.target_pid` (or `target_tenant_id` if pid==0); `ioctl(fd, CIPHER_GET_TENANT_SNAPSHOT, &q)`; copy snapshot to out.

**Mode 2 enumerate**: open `/proc/cipher/stats`; walk lines until the
"Per-PID summary" header; sscanf 16 fields per line (pid, tgid, comm,
tenant_id, total, 6 unused, sm, mem, launches, age); fill the struct
(only the partial fields available from /proc — other fields filled
by Mode 1 ioctl if needed). Skip "cipher-gpustate" pseudo-row.

**Mode 3 cached**: `refresh_cached` calls `query_by_pid(my_tid())`
into `__thread g_tls_cache.snap`, sets valid + timestamp.
`cached` returns pointer (NULL if invalid).
`cached_age_ns` returns elapsed since last refresh.

### STATE
Two thread-local statics (L53-54): `t_cipher_fd` (int, -1 sentinel),
`t_cipher_fd_initialized` (int, 0 sentinel). Plus
`g_tls_cache` (struct cipher_rt_tls_cache, L65): snap + valid + timestamp.

### CONCURRENCY
All state is `__thread` — no locks. Each thread opens its own
`/dev/cipher` fd lazily on first call.

### DEPENDENCIES
- INBOUND: `cipher_rt_partition_router.c::cipher_rt_pr_refresh_tenant` (L92).
- OUTBOUND: `/dev/cipher` ioctl nr 8; `/proc/cipher/stats`.
- External: pthread (TLS), syscall(SYS_gettid).

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- `.h` L17-18 says "STAGED: not yet integrated into cipher_rt build".
  **DRIFT**: file IS in Makefile OBJS L52 — actively compiled. Header
  is stale.
- The full `cipher_tenant_snapshot` struct (.h L38-84) has 27 fields,
  many unused by current actuators (predicted_hot_regions[8],
  graph_capture_state, koopman_substitution_eligibility, fusion_recipe_id,
  etc.). Per-thread-fd fix matches `cipher-contention-gate-context`
  memo's "33t, per-thread fd" finding.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING for per-tenant state in the actuators that consume it
(today: partition_router). Tenant snapshot is also the substrate for
future SLO-aware actuators.

### FUSION POINTS
- (1) Classifier consumption: the cached snapshot's
  `slo_priority` + `session_band` + `predicted_hot_regions[]` +
  `koopman_substitution_eligibility` fields are the natural inputs
  for a Wave-1-style oracle (cipher_oracle.cpp Gate 6).
- (2) Fallback today: actuators consume what they need (partition_router
  reads launches_total + tenant_handle_u32).
- (3) Insertion: a classifier observer reads
  `cipher_rt_tenant_cached()` to feed an oracle-style gate. No code
  change to this file.
- Fusion class: PORT-AS-IS. Severity: none.

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_rt_kv_alloc.c + .h

### PURPOSE
Phase 4.6.2 KV page allocator (CUDA VMM) + Phase 4.6.3 content-hash
dedup + Phase C SC2 weight arena + Track 2 SC3 weight import. Reserves
one large VA pool, carves per-(tenant,seq,layer,head_kv,role) slabs,
maps physical 2 MiB pages on demand. Each physical page carries a tag.
The dedup path is kmod-backed (T4.6.4) — cipher_rt computes xxhash64,
exports/imports cuIpc POSIX handles, memcmp-verifies; the kmod holds
refcounts. File header L1-23, L383-390, L682-689.

### PUBLIC SURFACE
KV slab API:
- `int cipher_rt_kv_alloc_init(size_t va_pool_bytes)` — L77.
- `void cipher_rt_kv_alloc_shutdown(void)` — L355.
- `cipher_rt_kv_slab_t *cipher_rt_kv_slab_create(...)` — L200.
- `int cipher_rt_kv_slab_ensure(slab, needed_bytes)` — L272.
- `void cipher_rt_kv_slab_free(slab)` — L291.
- `int cipher_rt_kv_page_info(devptr, out_tag)` — L316.
- `void cipher_rt_kv_alloc_get_stats(out)` — L346.

Dedup API:
- `int cipher_rt_kv_dedup_init(void)` — L472.
- `int cipher_rt_kv_dedup_put(content, out_devptr)` — L522.
- `void cipher_rt_kv_dedup_free(devptr)` — L634.
- `void cipher_rt_kv_dedup_get_stats(out)` — L660.

Weight arena API (Phase C SC2 / Track 2 SC3):
- `int cipher_rt_weight_arena_create(tenant_id, bytes, out_base)` — L714.
- `int cipher_rt_weight_arena_export(base, out_fd)` — L781.
- `int cipher_rt_weight_arena_info(devptr, out_tenant, out_handle)` — L803.
- `void cipher_rt_weight_arena_free(base)` — L825.
- `int cipher_rt_weight_arena_import(fd, want_base, bytes, out_base)` — L842.

### CONTROL FLOW

**`kv_alloc_init` (L77-129)**:
- cuInit, cuDeviceGet(0), cuDevicePrimaryCtxRetain + cuCtxSetCurrent.
- `cuMemGetAllocationGranularity` (2 MiB on H100).
- `cuMemAddressReserve(pool_base, pool, 0, 0, 0)`.
- `calloc(n_pages, sizeof(page_slot))`.
- Set stats.

**`slab_create` (L200-270)**:
- Page-round; first-fit walk for `want` consecutive FREE pages.
- Find free slab record; set state to RESERVED on the range.
- `map_pages(s, 0, init_pages)` — see `map_one`.
- Return slab handle + base devptr.

**`map_one(gi, shared)` (L145-180)**:
- If `shared == 0` (MISS): `cuMemCreate(... requestedHandleTypes=POSIX_FD)`
  — exportable.
- `cuMemMap(va, page_size, 0, h, 0)`.
- `cuMemSetAccess(... READWRITE)`.
- If fresh: `cuMemsetD8(va, 0, page_size)` — A3 invariant
  (cross-tenant zero on miss; HIT skips the zero).
- Record state=MAPPED, handle.

**`dedup_init` (L472-511)**:
- `open("/dev/cipher_kvdedup", O_RDWR | O_CLOEXEC)` — **L481 confirmed**.
- `ioctl(d.kvd_fd, CIPHER_KVDEDUP_INIT, &ini)` — **NR INIT confirmed**.
- Reserve `d.scratch_va` 2 MiB; `cuMemHostAlloc` pin buffer; allocate
  `d.pooloff` (per-page offset map).

**`dedup_put(content, out_devptr)` (L522-632)**:
- `h = xxh64(content, page_size)` (L529, custom xxhash64 L405-425).
- `carve_free_page()` and `map_one(gi, 0)` — fresh exportable page.
- `cuMemcpyHtoD(va, content, page_size)`.
- `cuMemExportToShareableHandle(&export_fd, handle, POSIX_FD, 0)`.
- `ioctl(d.kvd_fd, CIPHER_KVDEDUP_PUT, &p)` (L563) — **NR PUT confirmed**.
- If MISS (L570-576): close export_fd; store pool_offset; return.
- If HIT: `cuMemImportFromShareableHandle(&shared, p.candidate_fd, POSIX_FD)`;
  `dedup_verify(content, shared)` — `cuMemMap` scratch + `cuMemcpyDtoH` to
  pin_buf + memcmp.
  - If matched: unmap+release own page; `map_one(gi, shared)` — swap
    onto shared page; `ioctl(CIPHER_KVDEDUP_CONFIRM, &cf)` (L602) —
    **NR CONFIRM confirmed**.
  - If collision (xxh equal but memcmp differs): release shared;
    re-issue ioctl PUT with `CIPHER_KVDEDUP_FLAG_FORCE_NEW`.

**`dedup_free(devptr)` (L634-658)**: range-check; if MAPPED:
`ioctl(CIPHER_KVDEDUP_FREE, &f)` (L650) — **NR FREE confirmed**;
`cuMemUnmap`+`cuMemRelease`.

**`dedup_get_stats(out)` (L660-680)**: `ioctl(CIPHER_KVDEDUP_STATS, &s)`
(L669) — **NR STATS confirmed**.

**`weight_arena_create(tenant_id, bytes, out_base)` (L714-779)**:
- Find free arena slot; round size to page; `cuMemCreate` exportable;
  `cuMemAddressReserve` + `cuMemMap` + `cuMemSetAccess RW` +
  `cuMemsetD8(0)`.
- Record `{in_use, tenant_id, va, size, handle}`.

**`weight_arena_export(base, out_fd)` (L781-801)**: lookup arena by
base; `cuMemExportToShareableHandle(&fd, handle, POSIX_FD, 0)`.

**`weight_arena_import(fd, want_base, bytes, out_base)` (L842-907)**:
- `cuMemImportFromShareableHandle(&h, fd, POSIX_FD)`.
- `cuMemAddressReserve(&va, sz, 0, want_base, 0)` (prefer same VA);
  fallback to any VA.
- `cuMemMap` + `cuMemSetAccess READ-ONLY` (consumer never writes).
- Track `imported=1`.

### STATE
- `g` (file-static, L43-58): `init_done`, `dev`, `primary_ctx`,
  `pool_base`, `pool_bytes`, `page_size`, `n_pages`, `pages[n_pages]`,
  `slabs[MAX_SLABS=4096]`, `mu`, `stats`.
- `d` (file-static, L432-439): dedup state — `init_done`, `kvd_fd`,
  `tenant_id`, `scratch_va`, `pin_buf`, `pooloff[n_pages]`.
- `w_arenas[MAX_WEIGHT_ARENAS=64]` (L701).
- Single mutex `g.mu` guards ALL of the above. phase2_design.md A1
  hazard.

### CONCURRENCY
Single global mutex `g.mu`. All operations serialize on it — including
the memcmp-verify (which is 2 MiB of host-side memcmp under lock).

### DEPENDENCIES
- INBOUND: `cipher_kv_bridge.cpp` (Python entry).
- OUTBOUND: cuda.h VMM API (cuMemCreate, cuMemAddressReserve, cuMemMap,
  cuMemSetAccess, cuMemExport/Import, cuMemcpy*, cuMemset*, cuMemUnmap,
  cuMemRelease, cuMemAddressFree). `/dev/cipher_kvdedup` ioctls
  INIT/PUT/CONFIRM/FREE/STATS — **5 NRs confirmed**.
- kmod: `cipher_kvdedup.h` ABI.
- External: stdint, pthread.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- `cipher-t46-kv-dedup-discovery` memo: cuIpc cross-process verified;
  attention substrate; impl 2.5–4 weeks. The dedup path here is the
  L1 in-process dedup (single-tenant; cuIpc handles allow cross-process,
  but the live ABI is one tenant ↔ one kmod fd).
- `cipher-cp4656-closed` 100-proc dedup test references this code.
- `cipher-track2-weight-sharing` (Track 2 SC5): weight arena
  create/export/info/free is here; SC3 import added (L842). Memo matches.
- The dedup uses xxhash64 (custom impl L405-425) — NOT FNV (header
  L48-50 says "FNV-1a" for `content_hash`). **DRIFT** between the slab
  page-tag header comment and the dedup implementation. The slab tag's
  `content_hash` is filled by T4.6.3 (.h L49) — the dedup path
  doesn't actually fill the slab tag (it manages its own xxhash table
  in the kmod). So the header comment is stale; not a code bug.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING for vLLM KV integration (CP 5.1) and cross-tenant weight
sharing (Track 2). The slab API is the underlay; dedup and weight
arena are additive.

### FUSION POINTS
- (1) Classifier consumption: the slab tag's
  `(tenant_id, seq_id, layer, head_kv, role, content_hash)` could be
  filled by a KV classifier that recognizes hot prefixes; today the
  Python caller fills these.
- (2) Fallback today: tags from the Python caller; xxhash dedup.
- (3) Insertion: none in C; insertion point is the Python KV cache
  manager that decides slab tags.
- Fusion class: PORT-AS-IS. Severity: none for the C side.

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_kv_bridge.cpp

### PURPOSE
Python/libtorch C++ extension (pybind11). Wraps the C VMM page
allocator and the weight arena as torch.Tensor objects via
`at::from_blob`. The seam between CIPHER's tagged physical memory and
PyTorch's tensor type system. File header L2-13.

### PUBLIC SURFACE (Python)
- `init(va_pool_bytes) -> bool`.
- `vmm_zeros(shape, itemsize, dtype, tenant, seq, layer, role) -> Tensor`.
- `page_info(devptr) -> dict | None`.
- `get_stats() -> dict`.
- `class WeightArena` with `alloc(shape, itemsize, dtype)`,
  `view(shape, itemsize, dtype, offset)`, `export_fd()`, properties
  base/size/cursor/tenant.
- `weight_arena_create(nbytes, tenant) -> WeightArena`.
- `weight_arena_import(fd, want_base, nbytes) -> WeightArena`.

### CONTROL FLOW

**`vmm_zeros` (L50-81)**:
- Build `cipher_rt_kv_page_tag` from args; `head_kv = 0xFFFF`
  ("whole-tensor slab spans all KV heads").
- `cipher_rt_kv_slab_create(&tag, nbytes, nbytes, &devptr)` — fully
  mapped (StaticLayer pre-allocates).
- `torch::from_blob(devptr, shape, deleter=slab_free, opts)` —
  deleter releases the CIPHER slab on tensor destruction.

**`weight_arena_create` (L100-116)**: `cipher_rt_weight_arena_create`;
wrap in `shared_ptr<WeightArena>`; destructor calls
`cipher_rt_weight_arena_free`.

**`WeightArena::alloc` (L119-137)**: bump-allocate inside the arena
(512-byte aligned); `from_blob` with no-op deleter — arena owns memory.

**`WeightArena::view` (L178-196)**: same but at an explicit offset
(Track 2 SC3 manifest rebinding; no bump pointer change).

**`page_info(devptr)` (L198-223)**: try KV slab info; if not, try
weight arena info; return Python dict or None.

**`dtype_from_string` (L29-39)**: maps "float16/half/bfloat16/float/
float32/int8" — vLLM v1 needs int8 for the raw KV buffer (L33-37).

### STATE
None beyond the per-instance WeightArena objects.

### CONCURRENCY
Inherits the C side's `g.mu` for all backing-store operations. Python's
GIL serializes the bridge calls themselves.

### DEPENDENCIES
- INBOUND: Python (cipher_kv_cache.py, vLLM hooks).
- OUTBOUND: `cipher_rt_kv_alloc.h` C API; libtorch + pybind11.
- kmod ioctls: none directly (those land in cipher_rt_kv_alloc.c).
- External: torch/extension.h, cuda.h, pybind11.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- `cipher-cp51-closed` (vLLM KV integration, Option A
  buffer-ownership hook): the int8 dtype branch is the load-bearing
  type for vLLM v1 — matches.
- `cipher-track2-weight-sharing` SC2/SC3: `weight_arena_create` /
  `weight_arena_import` / `view` form the producer/consumer pair —
  matches.

### CONTRIBUTION TO SYSTEM
LOAD-BEARING for Python-side vLLM integration and Track 2 weight
sharing.

### FUSION POINTS
- (1) Classifier: no direct dependency; Python caller already knows
  KV/weight role.
- (2)/(3): N/A.
- Fusion class: PORT-AS-IS. Severity: none.

---

## Section /home/ubuntu/cipher_rt_phase4/cipher_rt_arbitrate.c (VESTIGIAL)

**File-header (L1-14) declares it retired at CP 5.4 Step 1.3
(2026-05-18).** Not built (Makefile L48-50 confirms `cipher_rt_arbitrate.o`
is NOT in OBJS). ARB issued ioctl nr 9 (CIPHER_REQUEST_SM_PARTITION),
which the CP 5.4 kmod returns -ENOSYS for; SM allocation is now the
CP 5.4 group ledger (`CIPHER_CP54_ALLOCATE`). The 30 s poll-thread
B7 fix from T4.2.2 (file-header L26-32) is dead code. Kept in-tree
for historical reference. Fusion class: VESTIGIAL.

---

## Section /home/ubuntu/libcipher_v2/cipher_inject.c

### PURPOSE
Earlier-version inject entrypoint (anchor 86618c30). Phase 3 substrate
only: REGISTER_TENANT + CUPTI launch counter. No green ctx, no Marlin,
no audit, no GOT patcher.

### PUBLIC SURFACE
- `int InitializeInjection(void *)` — L33.
- `int InitializeInjection2(void)` — L41.

### CONTROL FLOW
Same `pthread_once`-gated init pattern as Phase 4. Init body (L22-30)
runs ONLY:
1. `cipher_v2_tenant_register` (ioctl nr 1).
2. `cipher_v2_cupti_init` (ioctl nr 7).

### STATE
`cipher_v2_init_once` — single pthread_once.

### CONCURRENCY
Same single-shot pattern.

### DEPENDENCIES
Same — CUDA driver inbound, two substrate inits outbound.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- `cipher-phase3-shipped` confirms this is the Phase 3 baseline.
- The Phase 4 file has 14 init steps vs. the v2's 2 — the divergence
  is entirely new substrate (Phase 4 actuators + CP 2.5 GOT patcher).

### CONTRIBUTION TO SYSTEM
HISTORICAL BASELINE.

### FUSION POINTS
Not a fusion target itself; the Phase 4 version is what the classifier
brain hooks into. Fusion class: PORT-AS-IS (do not modify).

---

## Section /home/ubuntu/libcipher_v2/cipher_tenant.c

Byte-identical to `cipher_rt_phase4/cipher_tenant.c` (md5 matches by
inspection; same 59 LOC + same logic). Phase 3 tenant registration
path was preserved verbatim. PURPOSE / SURFACE / CONTROL FLOW /
DEPENDENCIES / etc. all match the Phase 4 section above. No drift.

Fusion class: PORT-AS-IS.

---

## Section /home/ubuntu/libcipher_v2/cipher_cupti.c

### PURPOSE
Earlier-version CUPTI integration (Phase 3 Task 5 only). Counts kernel
launches, batched-flushes via `CIPHER_SUBMIT_LAUNCH_STATS` (ioctl nr 7)
every 256. **No Phase 4 actuator side-effects.**

### PUBLIC SURFACE
- `int cipher_v2_cupti_init(void)` — L89.

### CONTROL FLOW
`cipher_v2_cupti_cb` (L45-87):
- Filter API_ENTER only.
- Atomic-increment total.
- If not flush-tick (every 256): return.
- Build stats and ioctl.

Init (L89-136):
- Open `/dev/cipher`.
- `cuptiSubscribe + cuptiEnableCallback` for cudaLaunchKernel_v7000,
  cudaLaunchKernelExC_v11060, cuLaunchKernel — **only 3 launch CBIDs**.
- No stream-create hooks.

### STATE
- `g_launches_total`, `g_grid_ops_total` (atomic).
- `g_cipher_fd`, `g_subscriber`.

### CONCURRENCY
Workload thread, counters atomic.

### DEPENDENCIES
- INBOUND: `cipher_inject.c::cipher_v2_init_body`.
- OUTBOUND: CUPTI + ioctl nr 7.

### LOGIC-AS-CODED VS LOGIC-AS-DOCUMENTED
- The Phase 4 `cipher_cupti.c` is a strict superset:
  - Adds green-ctx ensure/push/pop on stream-create.
  - Adds T4.2.4d enforcement gate (cipher_rt_in_marlin_quant TLS
    + make_current).
  - Adds partition_router + sm_packer observe.
  - Adds 5 launch-bucket diagnostic counters.
  - Adds 5 stream-create CBIDs.
- **Central divergence**: this v2 file has no actuator side-effects;
  the Phase 4 file is the actuator hot path's primary call surface.

### CONTRIBUTION TO SYSTEM
HISTORICAL BASELINE.

### FUSION POINTS
Not a fusion target itself. Fusion class: PORT-AS-IS (do not modify).

---

## Section /home/ubuntu/libcipher_v2/cipher_v2_internal.h

40-line internal header. Same MACRO scaffolding as
`cipher_rt_phase4/cipher_v2_internal.h` minus the Phase 4 actuator
forward decls (no `cipher_rt_pr_init`, no `cipher_rt_smp_init`, no
green ctx decls). PORT-AS-IS as historical baseline.

---

# Cross-File Synthesis (Wave 2 specific)

## Hot-path call graph in cipher_rt_phase4 (verified by inspection)

```
CUDA driver -> InitializeInjection2
   -> pthread_once -> cipher_v2_init_body (cipher_inject.c:34)
        -> tenant register (ioctl nr 1)
        -> green_ctx_cp54_init (ioctl nr 13)
        -> smp_init + pr_init
        -> cupti_init (subscribes 8 CBIDs; opens /dev/cipher O_WRONLY)
        -> volt_init (NVML or ioctl nr 10; signal handlers)
        -> matmul_dispatch_init
        -> marlin_init (registers actuator priority 10)
        -> attn_dispatch_init
        -> attn_test (env-gated, priority 0)
        -> audit_init (env-gated, priority 0 on BOTH substrates)
        -> cublas_shim_register_got + attn_register_got
        -> got_patch_init (dl_iterate_phdr; patches every loaded ELF)

CUPTI cuStreamCreate callback ENTER ->
   cipher_rt_green_ctx_ensure + push
EXIT -> pop                                 (cipher_cupti.c:96-101)

CUPTI launch callback API_ENTER (every cudaLaunchKernel*) ->
   if !cipher_rt_in_marlin_quant:
        cipher_rt_green_ctx_make_current   (cipher_cupti.c:123-127)
   cipher_rt_pr_observe_stream             (cipher_cupti.c:160)
   cipher_rt_smp_observe                   (cipher_cupti.c:161)
   classify launch by stream's green ctx   (cipher_cupti.c:166-183)
   every 256 launches -> ioctl nr 7        (cipher_cupti.c:205)

PyTorch cublasGemmEx (GOT-patched) ->
   cipher_rt_cublasGemmEx_impl              (cipher_rt_cublas_shim.c:73)
        -> cipher_rt_matmul_dispatch       (matmul_dispatch.c:85)
              foreach actuator in priority order:
                 audit observer (env)      (audit.c:101) -> PASSTHROUGH (+ HMAC chain)
                 marlin gates + dispatch   (marlin_actuator.c:56) -> HANDLED / ERROR / PASSTHROUGH
              fall-through -> g_real_gemmEx  (libcublas.so via dlvsym)

PyTorch ATen SDPA (GOT-patched flash/eff/cudnn ::call) ->
   trampoline (attn_dispatch.cpp:289/324/362)
        -> route()                          (attn_dispatch.cpp:135)
              snapshot registry under lock; release; iterate actuators:
                 audit observer (env)      (audit.c:117) -> PASSTHROUGH
                 attn_test (env)            (attn_test.c:46) -> PASSTHROUGH
              -> g_total++/g_passthrough++
        -> ALWAYS call orig(...)            (attn_dispatch.cpp:321/359/398)
              HANDLED is structurally discarded in T4.6.1.

Python/vLLM -> cipher_kv_bridge.vmm_zeros ->
   cipher_rt_kv_slab_create  (kv_alloc.c:200)
        -> cuMemAddressReserve / cuMemCreate exportable /
           cuMemMap / cuMemSetAccess RW / cuMemsetD8(0)

Python -> cipher_kv_bridge.weight_arena_create ->
   cipher_rt_weight_arena_create (kv_alloc.c:714)
   weight_arena_export -> POSIX fd (kv_alloc.c:781)
   weight_arena_import (consumer) -> READ-ONLY map (kv_alloc.c:842)
```

## Substrate divergences worth flagging for fusion

1. **ERROR semantic split.** matmul ERROR exits the registry walk and
   falls through to real cuBLAS (matmul_dispatch.c L101-104). attn
   ERROR breaks the switch but continues the for-loop to the next
   actuator (attn_dispatch.cpp L166-169). A unified fusion classifier
   would need to honour both. Severity: requires-mitigation.

2. **attn HANDLED discarded.** Every trampoline calls orig(...)
   unconditionally (attn_dispatch.cpp L321/359/398). T4.6.1 honours
   this as documented PASSTHROUGH-only scope; a classifier-driven
   substitute actuator returning HANDLED produces NO behavioural
   change. Severity: blocking for the substitute lane; mitigation is
   a small trampoline refactor.

3. **Single mutex on AUDIT.** Every HMAC-SHA256 record() takes
   `g.mu` (audit.c L83). Under 16-tenant launch rate, this is the
   serialization point. Today AUDIT is env-gated and not on by
   default; if it ever becomes the SUBSTITUTE chain target, the
   throughput floor warrants a reconsideration. Severity:
   requires-mitigation (deferred).

4. **POOL executor absent.** Kmod has `CIPHER_CP54_QOS_POOL=2`
   (cipher_rt_green_ctx.c L125 reads it from env). cipher_rt's POOL
   path is treated like PARTITION (file header L25, code L290-308
   handles `g_cp54_grp_mask != 0`). The userspace POOL executor that
   binds SHARED tenants into the pool's green ctx is NOT in this
   tree — `cipher_rt_green_ctx_ensure` returns -1 for SHARED tenants
   (L260-266) so they run on primary context until "Step 1.3b'"
   lands. **Documented absence per the task prompt.**

5. **cipher_rt_arbitrate.c is VESTIGIAL.** Retired at CP 5.4 Step 1.3;
   not in OBJS; ARB ioctl nr 9 returns -ENOSYS. The file is
   kept-in-tree for history. Fusion should ignore it.

6. **cipher_rt_tenant.h drift.** Header L17 says "STAGED: not yet
   integrated into cipher_rt build" but it IS in Makefile OBJS L52.
   Header comment is stale; code is the source of truth.

7. **Sync-domain map dead code.** partition_router.c L116-131:
   `derive_sync_domain_map` returns `{0, 0}` after the per-tenant
   spread regressed WL14 by -5.7%. The .h L9 still advertises
   sync-domain spreading. Drift between docs and code, but code is
   the safer baseline.

8. **CUPTI v2 vs Phase 4 divergence.** libcipher_v2/cipher_cupti.c
   is the Phase 3 telemetry-only baseline. Every Phase 4 actuator
   hot-path side-effect (green-ctx enforcement, partition router,
   sm_packer, marlin TLS gate, stream-create hooks, 5 launch
   bucket counters) lives in the Phase 4 cipher_cupti.c only.

9. **`xxh64` vs `FNV-1a` (kv_alloc.h L48-50 vs kv_alloc.c L405-425).**
   Header says page tag's `content_hash` is "FNV-1a of contents". The
   dedup engine uses xxhash64. The dedup path doesn't fill the slab
   tag's `content_hash` field; the kmod manages its own table keyed
   by xxhash64. Header comment is stale; not a code bug.

10. **Marlin .h M ≤ 8 vs .c M ≤ 64.** marlin.h L9 says "M ≤ 8"; the
    actuator gate is M ≤ 64 (marlin_actuator.c L47, L106). Header
    comment is stale.

## Specific anchors — verified or refuted

| Claim | Verdict | Cite |
|---|---|---|
| matmul: 3-value enum, no-lock dispatch, break-on-ERROR, total/handled/passthrough atomic_ulong | VERIFIED | matmul_dispatch.h L70-74; matmul_dispatch.c L85-116 (esp. L94 snapshot, L101-104 ERROR break); L24-26 telemetry |
| attn: 4-value enum incl. REDIRECTED, snapshot-under-lock, continue-on-ERROR (break in switch case), trampoline ALWAYS calls orig | VERIFIED | attn_dispatch.h L91-96; attn_dispatch.cpp L143-148 snapshot; L166-169 case ERROR break (inside switch); L321/359/398 unconditional orig |
| audit: priority 0 on BOTH substrates, env-gated CIPHER_AUDIT, HMAC-SHA256 synchronous on hot path under global mutex | VERIFIED | audit.c L132-137; L182 env; L83-96 HMAC under g.mu |
| marlin_actuator L198 registration; L47/L106/L118/L48 gates M≤64, K%128, STABILITY_THRESHOLD=4 | VERIFIED | exact line refs hold |
| volt: NVML + ioctl nr 10 paths, [210, 1980] MHz clamp, signal-safe atexit + SIGTERM/INT/SEGV/ABRT/BUS handlers, batch LUT B=1→1000 B=8→1600 B≥32→1980 | VERIFIED | volt.c L54-63 LUT; L293 clamp; L241 5 signals; L189-200 path-aware restore |
| got_patch: dl_iterate_phdr, JUMP_SLOT/GLOB_DAT, RELRO-safe mprotect via /proc/self/maps | VERIFIED | got_patch.c L128-210; L176-177 reloc types; L80-124 region_prot + write_got_slot |
| green_ctx: ALLOCATE at injection-init (ioctl nr 13), split MIN_SM=8, cuGreenCtxCreate, cuCtxFromGreenCtx, T4.2.4d per-launch make_current | VERIFIED | green_ctx.c L137-203 (L169 ioctl nr 13); L278 split; L339 create; L348 cuCtxFromGreenCtx; L207-222 make_current |
| cupti L106-127 per-launch enforcement gated by cipher_rt_in_marlin_quant TLS, calls make_current | VERIFIED | cipher_cupti.c L60 extern TLS; L119-127 gate + make_current |
| kv_alloc opens /dev/cipher_kvdedup at L481; issues 5 NRs INIT/PUT/CONFIRM/FREE/STATS; Track 2 SC5/Phase C SC2 weight arena create/export/info/free/import | VERIFIED | L481 open; L490 INIT; L563/L621 PUT; L602 CONFIRM; L650 FREE; L669 STATS; L714/L781/L803/L825/L842 arena 5-fn family |

All 9 anchors verified.

## v1 fusion summary (Wave 2 actuators)

| File | Action | Severity | Rationale |
|---|---|---|---|
| `cipher_inject.c` | REFACTOR (small) | minor | Add one classifier-init step between matmul and Marlin |
| `cipher_rt_matmul_dispatch.{c,h}` | PORT-AS-IS + SHIM | minor | Substrate stable; add classify observer as priority-5 actuator |
| `cipher_rt_attn_dispatch.{cpp,h}` | REFACTOR (medium) | blocking | Trampoline must respect HANDLED for substitute lane |
| `cipher_rt_marlin_actuator.c` | PORT-AS-IS + SHIM | minor | Read TLS hint from classify observer; preserve own gates |
| `cipher_rt_marlin_engine.cpp` + supporting | PORT-AS-IS | none | Pure engine; no classifier dependency |
| `cipher_rt_volt.{c,h}` | PORT-AS-IS for v1; REFACTOR for per-launch | minor (deferred) | Per-launch dynamic clock retarget is v1.5+ |
| `cipher_rt_audit.{c,h}` | PORT-AS-IS | none | API already exports `record(SUBSTITUTE, ...)` |
| `cipher_rt_attn_test_actuator.c` | PORT-AS-IS | none | Pure observer |
| `cipher_rt_cublas_shim.c` | PORT-AS-IS | none | Substrate seam, no classifier dependency |
| `cipher_rt_got_patch.{c,h}` | PORT-AS-IS | none | Pure mechanism |
| `cipher_rt_sm_packer.{c,h}` | PORT-AS-IS | none | Detection only |
| `cipher_rt_partition_router.{c,h}` | PORT-AS-IS + SHIM | minor | Read slo_priority from classifier hint instead of launches quartile |
| `cipher_rt_green_ctx.{c,h}` | PORT-AS-IS | none | Policy lives in kmod CP 5.4 scheduler |
| `cipher_tenant.c` | PORT-AS-IS | none | Stable; ioctl nr 1 |
| `cipher_cupti.c` | REFACTOR (medium) | requires-mitigation | Hot-path insertion point for non-cuBLAS classify observer |
| `cipher_rt_tenant.{cpp,h}` | PORT-AS-IS | minor (stale header) | Header L17 "STAGED" is stale; code is in OBJS |
| `cipher_rt_kv_alloc.{c,h}` | PORT-AS-IS | none | KV/weight backing store; classifier insertion is on the Python side |
| `cipher_kv_bridge.cpp` | PORT-AS-IS | none | Python seam |
| `cipher_rt_arbitrate.c` | VESTIGIAL | n/a | Retired CP 5.4 Step 1.3; not in OBJS |
| `libcipher_v2/cipher_inject.c` | PORT-AS-IS (historical) | n/a | Phase 3 baseline |
| `libcipher_v2/cipher_tenant.c` | PORT-AS-IS (historical) | n/a | Identical to Phase 4 |
| `libcipher_v2/cipher_cupti.c` | PORT-AS-IS (historical) | n/a | Phase 3 baseline; Phase 4 cupti.c is the strict superset |
| `libcipher_v2/cipher_v2_internal.h` | PORT-AS-IS (historical) | n/a | Phase 3 internal header |

## The load-bearing fact (re-verified, Wave 2)

The deployed Phase 4 substrate has every primitive a classifier-brain
port needs:

- **A per-launch hot path with side-effects** — cipher_cupti.c L106-127.
- **Two registry substrates (matmul, attn) with priority-ordered
  actuator chains** — matmul_dispatch.c L85, attn_dispatch.cpp L135.
- **A per-thread tenant snapshot in TLS** — cipher_rt_tenant.cpp L65.
- **Audit chain with a `record(SUBSTITUTE, recipe_hash, SUBSTITUTED)`
  contract already exported** — audit.h L34-39.
- **GOT-patch interposition that works under
  `CUDA_INJECTION64_PATH`-only loading** — got_patch.c L210.

What is missing for fusion:

- **No classifier observer** — must be added at priority 5 on BOTH
  substrate registries.
- **No substitute lane on attn** — trampolines always call orig
  (attn_dispatch.cpp L321/359/398); blocking until refactored.
- **No POOL executor** — SHARED tenants run on primary context;
  green_ctx.c L260-266 returns -1.
- **No per-launch dynamic clock retarget** — volt.c is env-static.

The honest engineering posture for v1: port the classifier brain as
a priority-5 substrate observer; defer the attn substitute lane to
v1.5; preserve the existing PASSTHROUGH-only contract; ship MFU
measurement on the matmul substrate.
