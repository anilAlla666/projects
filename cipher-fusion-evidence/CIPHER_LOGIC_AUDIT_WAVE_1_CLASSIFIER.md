# CIPHER Logic Audit -- Wave 1: Classifier Brain

**Scope.** Ten files in `/home/ubuntu/cipher-may13-evidence/` that together make up the may13 *classifier brain*: the hot-path dispatcher, the safety oracle, the geometry classifier, the kernel-name table, the structural rule table, the recipe library, the cubin parameter recovery, and three Stage-1 observers (SENSE, PREDICT, DETERMINISM). The deployed runtime in `cipher_rt_phase4/` contains none of this brain (one comment-only stub aside); fusion is a port. This audit reconstructs each file's control flow at a level a porter could re-implement from the description, surfaces drift between code and the planning docs, and names the contract that must hold at each fusion point.

**Build context.** Per `Makefile:29-31`, `src/cipher_dispatch.cpp` and `src/cipher_oracle.cpp` are silently `filter-out`-excluded from `RT_CPP_SRC`; the canonical compiled-into-`libcipher_rt.so` versions live at top-level (`/home/ubuntu/cipher-may13-evidence/cipher_dispatch.cpp` = 543 LOC, `/home/ubuntu/cipher-may13-evidence/cipher_oracle.cpp` = 539 LOC). All line citations in this audit refer to the top-level files. The src/ shadows (616 LOC and similar) exist but are dead code in the production build.

The hot-path entry is `cipher_dispatch(CipherKernelDesc*)`, which `src/cipher_intercept_cudart.cpp:1419-1428` resolves via `dlsym(RTLD_DEFAULT, "cipher_dispatch")` and invokes per launch (`cipher_intercept_cudart.cpp:1483`). The classifier brain runs on the launching thread synchronously -- there is no Stage-1 thread in this code path; the Stage-1 observers (SENSE/PREDICT/DETERMINISM) are driven from a ring-buffer drain inside `cipher_10ops_impl.cpp:509,542,548`, not from `cipher_dispatch` itself.

---

## Section /home/ubuntu/cipher-may13-evidence/cipher_dispatch.cpp

### PURPOSE
From the file header (lines 5-43): *"Layer 3 Dispatch Engine. Replaces the Phase 0 passthrough stub in cipher_runtime.cpp. This IS the hot path. Every GPU kernel launch hits this function."* The decision flow is documented as: classify -> struct_lookup -> oracle -> registry -> recipe -> SUBSTITUTED. The header claims `~160ns` for the cache-hit no-substitute path and `~1.8us` for a substitution with LNN forward pass. This is the live spine of CIPHER's may13 classifier brain: the only function the LD_PRELOAD hook (`cipher_intercept_cudart.cpp`) actually calls per kernel launch.

### PUBLIC SURFACE
- `int cipher_layer3_init(void)` -- `cipher_dispatch.cpp:110-124`. Called from `src/cipher_runtime.cpp:73`. Returns `0`. Initialises `g_oracle`, calls `cipher_struct_lookup_init`, calls `cipher_registry_init(&g_registry)`, sets `g_layer3_initialized = true`. The integer return is currently a "always 0" -- no error path.
- `CipherDispatchResult cipher_dispatch(CipherKernelDesc* desc)` -- `cipher_dispatch.cpp:338-516`. Returns `CIPHER_PASS_THROUGH` (0) or `CIPHER_SUBSTITUTED` (1). This is the hot path.
- `CipherOracleState* cipher_get_oracle(void)` -- `cipher_dispatch.cpp:522`. Returns `&g_oracle`. Consumed by `cipher_telemetry.cpp:331` (via dlsym) and `cipher_intercept_cudart.cpp:3000`.
- `CipherRegistry* cipher_get_registry(void)` -- `cipher_dispatch.cpp:523`.
- `void cipher_layer3_report(void)` -- `cipher_dispatch.cpp:525-531`. Called from `src/cipher_runtime.cpp:121`.
- `void cipher_training_step_hook(const float* grad_norms, uint32_t num_layers, uint32_t global_step)` -- `cipher_dispatch.cpp:535-543`. Wired only from external test harness -- no in-tree caller resolved during grep.

### CONTROL FLOW

**`cipher_dispatch(desc)` -- pseudo-code reconstruction:**

```
if !g_cipher.initialized:                       # L339 -- pre-init "classify-only" branch
    cr = classify_launch(desc->fn, grid, block, shared)
    desc->op_class   = cr.op
    desc->confidence = cr.confidence
    if !g_layer3_initialized:                   # L351 -- lazy oracle/registry init
        cipher_registry_init(&g_registry)
        cipher_oracle_init(&g_oracle, NULL, &DEFAULT_CONFIG)
        g_layer3_initialized = true
    {
        infer_layer_context(desc, &layer_idx, &is_backward)   # L361
        oq = OracleQuery{layer_idx, total=80, op_class, confidence, name=NULL, is_backward, is_optimizer=false}
        if cipher_oracle_decide(&g_oracle, &oq) == DENY:
            return CIPHER_PASS_THROUGH          # L373
    }
    if desc->op_class == 0 (GEMM):
        sh = gemm_shape_hash(desc, &gm, &gn, &gk)   # L379
        entry = cipher_registry_lookup(&g_registry, 0, sh, arch=90)
        if entry:
            log "Registry HIT"
            if apply_recipe(entry, desc, 0):
                cipher_oracle_bill_gemm(true)
                return CIPHER_SUBSTITUTED       # L388
        cipher_oracle_bill_gemm(false)          # L391
    elif desc->op_class in {3 ELEMENTWISE, 4 REDUCTION}:
        elem_hash = grid_x ^ (block_x << 16)    # NOT fnv_shape
        entry = cipher_registry_lookup(&g_registry, op_class, elem_hash, arch=90)
        if entry and entry->recipe_type == 2:
            if apply_recipe(entry, desc, 0):
                bill_nongemm(true); return CIPHER_SUBSTITUTED   # L403
        # synthetic Chebyshev attempt
        synth = {op_class, recipe_type=2, error_bound=0.01, active=true}
        if apply_recipe(&synth, desc, 0):
            bill_nongemm(true); return CIPHER_SUBSTITUTED       # L418
        bill_nongemm(false)
    else:
        bill_nongemm(false)
    return CIPHER_PASS_THROUGH                   # L425

# Main path (g_cipher.initialized == true)
cr = classify_launch(desc->fn, grid, block, shared)        # L430
desc->op_class  = cr.op
desc->confidence = cr.confidence
cipher_liquid_record_op(&g_cipher.liquid, desc->op_class)  # L440 -- feeds workload rhythm

if cr.op == ITERATIVE_CUSTOM and cr.confidence < 60:
    return CIPHER_PASS_THROUGH                   # L445 -- fast exit

if !g_layer3_initialized:
    return CIPHER_PASS_THROUGH                   # L448

infer_layer_context(desc, &layer_idx, &is_backward)        # L453
oq = OracleQuery{layer_idx, total=80, op_class, confidence, name=NULL,
                  is_backward, is_optimizer=false}
oracle = cipher_oracle_decide(&g_oracle, &oq)              # L465
if oracle.decision == DENY:
    return CIPHER_PASS_THROUGH                   # L468

arch = g_hw_profile.architecture  # = 90 (H100 Hopper)
sh = gemm_shape_hash(desc, &_gm, &_gn, &_gk)               # L473
entry = cipher_registry_lookup(&g_registry, op_class, sh, arch=90)
if !entry:
    return CIPHER_PASS_THROUGH                   # L480 -- THE LOAD-BEARING PATH
if entry->error_bound > 0.01f:
    return CIPHER_PASS_THROUGH                   # L486

substituted = apply_recipe(entry, desc, layer_idx)         # L490
if !substituted:
    # billing
    if desc->op_class == 0: bill_gemm(false) else bill_nongemm(false)
    return CIPHER_PASS_THROUGH                   # L500

cipher_oracle_record_substitution(&g_oracle, layer_idx)    # L504 -- updates N<=4 counter
# billing
if desc->op_class == 0: bill_gemm(true) else bill_nongemm(true)
return CIPHER_SUBSTITUTED                        # L515 -- the lone success
```

**`apply_recipe(entry, desc, layer_idx)` -- pseudo-code (L242-331):**

```
switch entry->recipe_type:
  case 0 (GEMM):
    gemm_shape_hash(desc, &gm,&gn,&gk)
    if any < 16: return false                    # L250
    gcfg = cipher_recipe_gemm(gm,gn,gk, &g_hw_profile)
    # Koopman-converged branch
    if cipher_get_edmd_pipeline:
        edmd = cipher_get_edmd_pipeline(0)
        if edmd && edmd->status == EDMD_SOLVED && edmd->koopman.fit_error < 0.05:
            log "SUBSTITUTE: Koopman converged"
            if cipher_tls_relaunch: cipher_tls_relaunch()
            edmd_live_post_relaunch_hook(gm,gn,gk)
            return true                          # L277
    # Fallback relaunch
    if cipher_tls_relaunch:
        rc = cipher_tls_relaunch()
        if rc == 0:
            log "SUBSTITUTE: relaunch"
            edmd_live_post_relaunch_hook(gm,gn,gk)
            return true                          # L294
    return false                                 # L297
  case 2 (Chebyshev):
    if op_class != 4 && op_class != 3: return false           # L306
    s_cheb_opportunity++
    log every 500th occurrence
    return false                                  # L320 -- Python wrapper handles real substitution
  case 4 (HyperFlux): return true                # L325 -- always accept (no actuator)
  default: return false                          # L329
```

**`gemm_shape_hash(desc, &m, &n, &k)` -- pseudo-code (L143-177):**

```
*m = *n = *k = 0
if cipher_tls_get_gemm_shape:                    # weak from libcipher_hook.so
    cipher_tls_get_gemm_shape(&m,&n,&k,&valid)
    if valid: return fnv_shape(m,n,k)
if grid_z==1 && block_x==256 && block_y==1:
    m = grid_x * 256; n = grid_y * 128; k = (m+n)/2          # H100 cuBLAS tile inference
    return fnv_shape(m,n,k)
# Generic fallback
return fnv1a(grid_x, grid_y, block_x, shared_bytes)
```

**`infer_layer_context(desc, &layer, &backward)` -- L81-96:** Pure thread-local counter increment; `layer = (t_step_kernels / 60) % 80`. Backward flag never set by anyone -- `t_is_backward` is only initialized and reset to `false`. Worst-case latency is one increment + one divide + one modulo: a few nanoseconds.

**Branch determinants.**
- L444 ITERATIVE_CUSTOM gate: determined by `classify_launch()` output. ITERATIVE_CUSTOM has hard-coded confidence 40 (`cipher_classify.hpp:216`); the threshold here is 60 -- so EVERY ITERATIVE_CUSTOM kernel exits at L444 in the main path (cannot pass).
- L448 init gate: determined by `g_layer3_initialized` static. After first successful `cipher_layer3_init`, never re-evaluates.
- L467 oracle DENY: see `cipher_oracle_decide` flow below.
- L477 registry MISS: determined by whether `(op_class, fnv_shape(m,n,k), arch=90)` matches a registry entry. With the 32-entry day-one seed (`cipher_recipes.cpp:346`), real-workload M/N/K never hash to a seed entry -- see drift note below.
- L484 error_bound > 0.01f: NEVER triggers under current seed entries (all are 0.001-0.005, see Section /src/cipher_recipes.cpp).
- L490 substituted: false only if `cipher_tls_relaunch` is null OR returns non-zero. The weak symbol is defined in `libcipher_hook.so` (see Section /src/cipher_recipes.cpp / Section actuator coupling).

**Worst-case path length.** Header claims `~160ns` cache-hit, `~1.8us` substitute. Path length on the empirically common path (L480 registry miss) is: `classify_launch` (~10 ns cache hit per `cipher_classify.hpp:178`) + `cipher_liquid_record_op` + 2 atomic increments in `infer_layer_context` + `cipher_oracle_decide` (~50 ns claimed) + `gemm_shape_hash` (FNV1a over 3 u32s) + `cipher_registry_lookup` linear scan over 32+ entries (~32 x ~5 cycles = ~50 ns) + return = roughly 150-200 ns. The substitution path adds Koopman pipeline read, `cipher_tls_relaunch`, and the EDMD-live post-hook (cudaMemcpy and DTYPE introspection).

### STATE
**Module-level state owned here:**
- `static CipherOracleState g_oracle` (L62) -- 257 layers x 4 floats = ~5 KB of EMA state plus counters; mutated by `cipher_oracle_*` family.
- `static CipherRegistry g_registry` (L63) -- 256 entries x 80 bytes = ~20 KB; populated once at init.
- `static CipherHwProfile g_hw_profile = CIPHER_H100_PROFILE` (L64) -- read-only after init.
- `static bool g_layer3_initialized` (L65) -- single-write barrier.
- `static __thread uint32_t t_layer_counter, t_step_kernels` (L77-78) -- per-thread.
- `static __thread bool t_is_backward` (L79) -- never written non-false anywhere in this file.

**State read from elsewhere:**
- `g_cipher.initialized`, `g_cipher.liquid` (extern `CipherRuntime` from `cipher.h`).
- Weak externs: `cipher_tls_get_gemm_shape`, `cipher_tls_relaunch`, `cipher_get_edmd_pipeline`, `cipher_tls_get_gemm_ptrs`, `cipher_tls_get_gemm_types`, `cipher_edmd_live_collect` (L131-205). Defined respectively in `libcipher_hook.so` (the three TLS accessors), `libcipher_rt.so/cipher_edmd.cpp` (the EDMD pipeline), and `libcipher_rt.so/cipher_edmd_live.cpp:415` (the live calibration sink).

**Mutated by:** `cipher_layer3_init` (one-shot), `cipher_dispatch` (per launch -- increments TLS counter, updates oracle counters via callees, updates registry only at init), `cipher_training_step_hook` (fans into oracle update).

### CONCURRENCY
**Threading model:** hot-path-sync. Every launching thread calls `cipher_dispatch` directly. No background thread, no signal handler.

**Locks:** none acquired in this file. `g_oracle` and `g_registry` are unguarded -- multi-thread updates to oracle counters (`sub_counter[i]`, `force_passthrough[i]`, EMA arrays) race. The N<=4 counter increment at `cipher_oracle.cpp:394` is `state->sub_counter[layer_idx]++` -- non-atomic. Acceptable because the failure mode is at-worst missing a substitution opportunity (the N<=4 counter underflows or doesn't quite fire when it should). Not signal-safe.

**Lock-free reads.** TLS accessors `t_step_kernels`, `t_layer_counter`, `t_is_backward` are thread-local: no synchronization needed.

### DEPENDENCIES
**INBOUND** (who calls into this file):
- `cipher_dispatch` -- called from `src/cipher_intercept_cudart.cpp:1483` (resolved via `dlsym(RTLD_DEFAULT, "cipher_dispatch")` at L1419).
- `cipher_layer3_init` -- `src/cipher_runtime.cpp:73`.
- `cipher_layer3_report` -- `src/cipher_runtime.cpp:121`.
- `cipher_get_oracle` -- `src/cipher_telemetry.cpp:331` (dlsym), `src/cipher_intercept_cudart.cpp:3000` (dlsym).
- `cipher_get_registry`, `cipher_training_step_hook` -- no in-tree caller, exported in `exports.map` but not wired.

**OUTBOUND** (what this file calls):
- `cipher::classify_launch` (inline header -- `cipher_classify.hpp:227`).
- `cipher_liquid_record_op` (defined in `cipher_liquid_state.cu` / `cipher_lnn.cpp`).
- `cipher_oracle_init`, `_decide`, `_record_substitution`, `_bill_gemm`, `_bill_nongemm` -- defined in `cipher_oracle.cpp`.
- `cipher_struct_lookup_init` -- defined in `cipher_structural_lookup.cpp` (called via `cipher_oracle_init` chain at L104 of oracle).
- `cipher_registry_init`, `_lookup` -- defined in `cipher_recipes.cpp`.
- `cipher_recipe_gemm` -- defined in `cipher_recipes.cpp`.
- Weak: `cipher_tls_get_gemm_shape`, `cipher_tls_relaunch`, `cipher_tls_get_gemm_ptrs`, `cipher_tls_get_gemm_types` -- defined in `cipher_intercept_cudart.cpp`. The weak attribute means link succeeds even when libcipher_hook.so is absent; at runtime they resolve via the GOT once libcipher_hook.so loads.
- Weak: `cipher_get_edmd_pipeline` -- defined in `src/cipher_edmd.cpp`.
- Weak: `cipher_edmd_live_collect` -- defined in `src/cipher_edmd_live.cpp:415`.

**kmod ioctls:** none. The dispatcher is pure CPU.

**External libraries:** none directly (uses dlfcn.h header but no dlopen calls).

### LOGIC-AS-CODED vs LOGIC-AS-DOCUMENTED

**Plan section 1.1 (CIPHER_REENGINEERING_PLAN.md:83-85)** claims: *"Five PASS_THROUGH early-exits all return CIPHER_PASS_THROUGH: line 541 (Layer 3 not initialized), 560-561 (ORACLE DENY), 570-573 (registry MISS -- 'L3.5 EDMD pipeline -- wired in Week 4-5'), 577-579 (entry->error_bound > 0.01), 584 (substitute failed)."*

**Drift (already documented in DEEP_INSPECTION_REPORT D.1).** Those citations match the src/ shadow file (616 LOC). The TOP-LEVEL file (543 LOC, the one compiled into libcipher_rt.so per Makefile L29-31) has the corresponding lines at L448, L468, L480, L486, L500. This audit's CONTROL FLOW section above uses the correct top-level numbers per the user's anchor instruction.

**Additional drift the plan undercounts (verified by re-reading file):** The TOP-LEVEL file contains an **entire second dispatch path** at L338-426 -- the "classify-only" branch fired when `!g_cipher.initialized`. That branch has:
- 1 PASS_THROUGH at L373 (oracle DENY in classify-only path)
- 1 SUBSTITUTED at L388 (GEMM registry HIT in classify-only path)
- 1 SUBSTITUTED at L403 (non-GEMM Chebyshev registry HIT)
- 1 SUBSTITUTED at L418 (synthetic-entry Chebyshev success)
- 1 PASS_THROUGH at L425 (default fall-through)

So the total tally is **8 PASS_THROUGH + 4 SUBSTITUTED** return points, not the plan's 5+1. The DEEP_INSPECTION_REPORT D.1.NEW already calls this out; this audit confirms.

**Drift -- the load-bearing L480 path.** The comment at L478-479 reads: *"No surrogate yet. If this op appears frequently, queue for EDMD. (L3.5 EDMD pipeline -- wired in Week 4-5)"*. The "Week 4-5" wiring never happened in tree. There is no `cipher_edmd_queue_kernel` or equivalent -- the registry can only grow if some external code calls `cipher_registry_insert` (defined `cipher_recipes.cpp:502` but called from nowhere in tree per grep). The PLAN section 3 (Goal-4) calls this out correctly as the central deferred reality. CODE matches docs; the *behaviour* (Goal-4 doesn't fire) is correct.

**Drift -- `infer_layer_context` is structurally wrong for inference.** The function assumes 80 layers (Llama-3 70B) hardcoded `CIPHER_INFERRED_TOTAL_LAYERS`. Real Llama-3 8B is 32 layers, Mistral-7B is 32, the only Llama with 80 is 70B. Pass `total_layers=80` into oracle on Mistral and the oracle's *last-3-layers* rule (`cipher_structural_lookup.cpp:221-226`) only blocks layers 77/78/79 -- which are layer-modulo'd to 80, so the inferred layer_idx never hits 77 for a 32-layer model (it visits layers 0..79). Net effect: structural last-3-layers protection silently disabled outside the Llama-3-70B regime.

**Drift -- Chebyshev synthetic-entry billing.** L416-419 creates a stack-local `CipherRegistryEntry synth` and passes it to `apply_recipe`. `apply_recipe`'s GEMM branch (case 0) reads `entry->error_bound` only after `recipe_type` switch, but the synth entry has `recipe_type=2`. So it hits the Chebyshev case (L300-321), which returns `false` (Python wrapper handles real substitution). The `apply_recipe(&synth, ...)` call therefore always returns false at L416. The conditional `if (apply_recipe(&synth, desc, 0))` at L416 is effectively dead -- control falls through to L421 `bill_nongemm(false)`. So L418's "synthetic-entry Chebyshev SUBSTITUTED" return is **unreachable**.

**Drift -- `apply_recipe` HyperFlux branch (case 4) returns true unconditionally with no actuator.** Per L323-326: `case 4: return true;` -- there is no kernel relaunch, no logging, no metric. If a registry entry with `recipe_type=4` ever matched (the 4 HyperFlux entries 22-25 with `op_class=6 (ITERATIVE_CUSTOM)` and `hw_arch=86 (Ampere)`), it would report "CIPHER_SUBSTITUTED" while the actual kernel was already passed through to the device by the LD_PRELOAD hook BEFORE `cipher_dispatch` was called. Net effect: spurious SUBSTITUTED count. But: never fires on H100 (hw_arch mismatch).

**Drift -- claimed `~160ns` cache-hit latency.** The path traverses `classify_launch` (lock-free), `cipher_liquid_record_op` (lock-free atomic increment), oracle (5 gates with branch+counters), and registry linear scan over ~32 entries. Modern x86 with branch prediction warm: probably 80-150 ns. The 160 ns claim is plausible but not verified in tree.

### CONTRIBUTION TO SYSTEM
**LOAD-BEARING.** This file is the only entry point that ties together CLASSIFY (op 1), ORACLE (op 2), and the registry-driven SUBSTITUTE (op 3). Without `cipher_dispatch`, no kernel ever enters CIPHER's classifier brain. Removing this file: every kernel passes through unmodified -- no observation, no billing, no substitution opportunity. **The hot-path spine.**

But: in current state, the dispatcher's downstream consumers (registry, EDMD pipeline) are starved. Every real workload exits at L480 (registry MISS). So today, `cipher_dispatch`'s contribution to *actuation* is zero -- but its contribution to *substrate* (the hooks that telemetry/billing read via `cipher_get_oracle`) is load-bearing for the Stage-1/2 observers and downstream reporting.

### FUSION POINTS

**Hot-path entry -- Tree B has no equivalent.**
- **Contract:** `cipher_dispatch(CipherKernelDesc*)` is what the LD_PRELOAD hook calls per kernel launch. Tree B's deployed runtime is LD_PRELOAD-free (per `cipher-cp25-closed` memo); there is no analogous hook. The fusion target is to add a pre-launch callback inside the existing actuator-registry substrates (`cipher_rt_matmul_dispatch.c`, `cipher_rt_attn_dispatch.cpp`) and route through `cipher_rt_dispatch.cpp` (the unified port).
- **Honored on both sides?** No. Tree B's runtime has no Stage-0 entry. **Fusion requires** adding a new substrate file `cipher_rt_dispatch_substrate.cpp` mirroring the matmul-dispatch shape.
- **Severity: blocking.** Without this entry point, the entire classifier-brain port has nowhere to attach.
- **Action: REFACTOR-REQUIRED.** Port `cipher_dispatch` to `cipher_rt_phase4/cipher_rt_dispatch.cpp` but rename to e.g. `cipher_rt_classify_pre_launch(const cipher_rt_call*)`. The internal logic ports as-is.

**Weak symbol `cipher_tls_get_gemm_shape` (L132).**
- **Contract:** Returns `(M, N, K, valid)` from TLS slot set by `cublasGemmEx`/`cublasLtMatmul` shim. Defined in `libcipher_hook.so` (specifically `src/cipher_intercept_cudart.cpp:209`).
- **Honored on both sides?** Tree B has `cipher_rt_cublas_shim.c` but it does not export `cipher_tls_get_gemm_shape`. The cuBLAS interception path in Tree B sits inside the cuBLAS shim itself; M/N/K are known at call time without needing TLS.
- **Action: SHIM-REQUIRED.** Provide an inline `cipher_rt_get_current_gemm_shape` inside `cipher_rt_cublas_shim.c` that returns the call-site M/N/K when invoked from the shim's actuator-registered classifier. Severity: requires-mitigation.

**Weak symbol `cipher_tls_relaunch` (L185).**
- **Contract:** Re-executes the captured kernel launch under CIPHER control, with TLS-stored arg buffer. Used in `apply_recipe` GEMM branch to actually run the kernel after the registry hit. Returns 0 on success.
- **Honored on both sides?** Tree B's actuators (Marlin, cuBLAS shim) are themselves the "relaunch" -- they invoke the actuator-resolved kernel directly. So in Tree B, the classifier's "apply_recipe" doesn't need a relaunch path; it returns `SUBSTITUTE` and the actuator-registry's high-priority entry takes over.
- **Action: REFACTOR-REQUIRED.** Replace the `cipher_tls_relaunch` weak-symbol call with a "return SUBSTITUTE and let actuator chain handle it" contract. Severity: requires-mitigation; the test plan must show byte-identical output through Tree B's actuator path.

**Weak symbol `cipher_get_edmd_pipeline` (L186-187).**
- **Contract:** Returns pointer to an `EdmdPipeline` for op_class. Used to check Koopman convergence in the GEMM branch.
- **Honored on both sides?** Tree B has no EDMD module. The deferred Koopman lane (Goal-4) was never wired.
- **Action: PORT-AS-IS or DEFER.** Port the symbol but stub to return NULL until Goal-4 is wired in v2. Severity: minor (current behaviour is "always fallback to relaunch").

**Weak symbol `cipher_edmd_live_collect` (L196-205).**
- **Contract:** Post-relaunch hook that collects (A, B, C, alpha, beta, dtypes) and adds the GEMM as a snapshot for live Koopman calibration. Reads TLS pointers via `cipher_tls_get_gemm_ptrs/types`.
- **Honored on both sides?** Tree B has no analogous collection path. The EDMD-live calibration sink lives in `src/cipher_edmd_live.cpp:415` which is NOT compiled into `cipher_rt_phase4/libcipher_rt.so`.
- **Action: DEFER for v1.** The post-hook fires twice in `apply_recipe` (L275 and L292); both calls become no-ops in Tree B until EDMD-live is ported. Severity: minor.

**`g_cipher.liquid` reference (L113, L440).**
- **Contract:** Pass `&g_cipher.liquid` to oracle init and call `cipher_liquid_record_op(&g_cipher.liquid, op_class)` to feed the workload-rhythm classifier (LNN).
- **Honored on both sides?** Tree B has no `cipher_liquid_state` module. `cipher_liquid_record_op` is dead in the production tree.
- **Action: REFACTOR-REQUIRED.** Either port the liquid-state ring buffer (small -- `cipher_lnn.cpp` is well-bounded), or pass `nullptr` to `cipher_oracle_init` (the oracle null-checks `state->liquid` at every use site -- `cipher_oracle.cpp:197, 248, 374, 396, 408`). Choose null-pass for v1 to minimize port surface. Severity: minor.

---

## Section /home/ubuntu/cipher-may13-evidence/cipher_oracle.cpp

### PURPOSE
From `include/cipher_oracle.h:1-37`: three coordinated safety mechanisms (L3.6 N<=4 substitution rate-limit, L3.7 EMA gradient divergence detector, L3.9 phase detector for warmup suppression). All gates must PERMIT for substitution to proceed. The file also implements topological auto-inference of inference-mode (`topo_detect_inference`) so substitution can fire without an explicit training-step hook calling `cipher_oracle_update_gradients`.

### PUBLIC SURFACE
- `void cipher_oracle_init(state, liquid, cfg)` -- L80-119. Initializes EMA sigmas to 1.0, calls `cipher_struct_lookup_init`, sets `state->initialized=true`.
- `void cipher_oracle_update_gradients(state, grad_norms, num_layers, global_step)` -- L265-287. Called every 100 training steps (per header comment).
- `CipherOracleResult cipher_oracle_decide(state, query)` -- L293-384. The hot-path gate.
- `void cipher_oracle_record_substitution(state, layer_idx)` -- L390-399.
- `void cipher_oracle_record_passthrough(state, layer_idx)` -- L402-411.
- `void cipher_oracle_set_phase(state, phase)` -- L417-423.
- `void cipher_oracle_report(state)` -- L429-463.
- `void cipher_oracle_update_mfu(state, mfu_fraction)` -- L469-483.
- `void cipher_oracle_bill_gemm(state, M, N, K, substituted)` -- L489-502.
- `void cipher_oracle_bill_nongemm(state, substituted)` -- L504-507.
- `void cipher_oracle_billing_report(state)` -- L509-539.

### CONTROL FLOW

**`cipher_oracle_decide(state, q)` -- pseudo-code (L293-384):**

```
PERMIT = {PERMIT, "ok"}                                # static const at L296
# Lazy-resolve CIPHER_FORCE_PERMIT env var (cached after first call)
if first call:
    s_force_permit = (getenv("CIPHER_FORCE_PERMIT") == "1") ? 1 : 0
if s_force_permit:
    state->total_decisions++
    state->permitted++
    return PERMIT                                       # L307

if !state->initialized:
    return DENY("oracle-not-init")                      # L311

state->total_decisions++

# Topological phase auto-detect (replaces explicit gradient-hook requirement)
if state->phase.detected_phase == 0:
    topo_detect_inference(state, q->op_class)           # L319

# Gate 1: Phase
if state->phase.detected_phase == 0:
    state->denied_warmup++
    return DENY("warmup")                               # L325

# Gate 2: Minimum confidence (MFU-adjusted)
effective_conf = clamp(state->cfg.min_confidence + state->mfu_confidence_adjust, 20, 95)
if q->confidence < effective_conf:
    state->denied_low_confidence++
    return DENY("low-confidence")                       # L337

# Gate 3: Structural lookup
sctx = {kernel_name, layer_idx, total_layers, op_class,
        training_phase=state->phase.detected_phase,
        is_backward, is_optimizer_step}
slr = cipher_struct_lookup(&sctx)                       # L351
if slr.result == CIPHER_STRUCT_FULL_PRECISION:
    state->denied_structural++
    return DENY("structural-rule")                      # L354

# Gate 4: EMA permanent demotion
if layer_idx < 256 && state->ema.permanently_demoted[layer_idx]:
    state->denied_ema_demotion++
    return DENY("ema-demoted")                          # L361

# Gate 5: N<=4 rule
if layer_idx < 256:
    cnt = state->sub_counter[layer_idx]
    if cnt >= state->cfg.n_max:                         # default n_max = 4
        state->sub_counter[layer_idx] = 0
        state->force_passthrough[layer_idx] = 1
        state->denied_n4++
        cipher_liquid_record_passthrough(state->liquid, layer_idx)
        return DENY("n4-rule")                          # L377

state->permitted++
return PERMIT                                            # L383
```

**`topo_detect_inference(state, op_class)` -- pseudo-code (L30-74):**

```
state->topo_class_counts[op_class & 0x7]++
total = state->total_decisions
if (total & 0x3F) != 0: return false                   # only every 64th decision
if state->phase.detected_phase >= 1: return false      # already escaped warmup
has_training_ops = (topo[5] > 0) || (topo[6] > 0)      # MEMCPY_TRANSPOSE or ITERATIVE_CUSTOM
if !has_training_ops && total >= 200:                  # TOPO_WINDOW
    state->phase.detected_phase = 1
    state->phase.phase_entry_step = total
    log "inference mode auto-detected"
    return true
# Secondary: 64+ kernels, only GEMM/ELEMENTWISE/REDUCTION
if total >= 64:
    infer_only = topo[0] + topo[3] + topo[4]
    if infer_only == total:
        state->phase.detected_phase = 1
        log "inference mode"
        return true
return false
```

**`update_phase_detector(state, grad_norms, num_layers, global_step)` -- L125-200:** Advances `step_count` always (even when manually overridden, so EMA baseline can be established). Hard warmup gate `< warmup_steps=500` keeps phase=0. After warmup: compute grad mean and variance, EMA both at alpha=0.01. Transition WARMUP->CONVERGENCE when `cv = sqrt(var)/mean < 0.15`. Transition CONVERGENCE->FINETUNE when `cv < 0.05 && mean < 0.001`. Sync `detected_phase` to `liquid->device->phase`.

**`update_ema_monitor(state, grad_norms, num_layers)` -- L206-259:** Per-layer EMA at kappa=cfg.ema_kappa (default 0.999). Establish baseline at `step_count >= ema_baseline_steps` (default 1000): `baseline = ema`, `sigma = 0.1*baseline + 1e-4`. Track running sigma after baseline: `sigma = 0.99*sigma + 0.01*|g-baseline|`. Divergence test: `g > baseline + 2sigma*sigma AND g > 5*baseline + 1e-4`. On divergence: permanently demote -- set `permanently_demoted[i] = true`, increment counter, set `liquid->device->layer[i].perm_passthrough = 1`, call `cipher_struct_override_layer(i, true)`.

**Branch determinants.** `CIPHER_FORCE_PERMIT` env var (cached on first call, never re-evaluated -- set CIPHER_FORCE_PERMIT=1 to bypass all 5 gates and PERMIT everything). `state->initialized` (set once). `state->phase.detected_phase` (advances by gradient signal or topological auto-detect). `state->ema.permanently_demoted[layer_idx]` (per-layer one-shot, never cleared). `state->sub_counter[layer_idx]` (rate limit).

**Worst-case path length.** PERMIT case touches: env-cache load (1 cycle), total_decisions++ (1 atomic), topo_detect (mod-64 fast-skip), 5 branch comparisons, structural lookup (claimed <10 ns per `cipher_structural_lookup.h:25`), 1 array deref (permanently_demoted), 1 array deref (sub_counter), counter increment. Roughly ~30-50 ns. Header claim of "~50ns" is reasonable.

### STATE
**Module-level state owned (file-static):**
- `static int s_force_permit = -1` (L299 inside `cipher_oracle_decide`) -- env cache.

**State owned via state pointer:**
- All of `CipherOracleState` (`cipher_oracle.h:101-137`): 256-element arrays of `float ema[]`, `float baseline[]`, `float sigma[]`, `bool baseline_set[]`, `bool permanently_demoted[]`, `uint8_t sub_counter[]`, `uint8_t force_passthrough[]`; 8-element `topo_class_counts[]`; aggregate counters (`total_decisions`, `permitted`, `denied_*`); MFU feedback (`last_mfu_fraction`, `mfu_confidence_adjust`); billing counters; `CipherPhaseState phase`.

**State read from elsewhere:**
- `state->liquid` (`CipherLiquidStateMgr*`) -- read at L197-199, L249-250, L374, L396, L408 to sync phase, record passthrough/substitution, and demote layer. NULL-checked at every use; safe to pass null.
- `state->liquid->device->layer[i].perm_passthrough` mutated at L250.

**Mutated by:**
- Counter increments throughout `cipher_oracle_decide`.
- `cipher_oracle_record_substitution`: `sub_counter[i]++`, `force_passthrough[i]=0`.
- `cipher_oracle_record_passthrough`: `sub_counter[i]=0`, `force_passthrough[i]=0`.
- `cipher_oracle_update_gradients` -> `update_phase_detector` -> `update_ema_monitor`: writes per-layer EMA/baseline/sigma/permanently_demoted.

### CONCURRENCY
**Threading model:** hot-path-sync (`cipher_oracle_decide` runs on the launching thread). `update_gradients` and `update_mfu` are training-loop hooks, called on the optimizer-step thread (typically single-threaded).

**Locks:** none. All counter increments are non-atomic. Race conditions:
1. `state->sub_counter[layer_idx]++` (L394) races with the `cnt >= n_max` read at L367 -- possible double-fire.
2. `topo_class_counts[op_class & 0x7]++` (L32) races across threads, can under-count.
3. `denied_*` counters race -- reports are approximate.

This is acceptable because (a) the failure modes degrade gracefully (one extra substitution at most, or undercounting in reports), (b) the oracle is a safety gate not a correctness gate.

**Signal-safety:** no. Calls `getenv` (not signal-safe), `fprintf(stderr, ...)` (not signal-safe). Atexit handler not present.

### DEPENDENCIES
**INBOUND:**
- `cipher_oracle_init`, `_decide`, `_record_substitution`, `_bill_gemm`, `_bill_nongemm` -- `cipher_dispatch.cpp:113, 354, 371, 387, 391, ...` (all branches).
- `cipher_oracle_update_mfu` -- `cipher_telemetry.cpp:321` (dlsym; periodic call after MFU sample).
- `cipher_oracle_set_phase`, `_record_passthrough` -- no in-tree callers; reserved for test harnesses.
- `cipher_oracle_update_gradients` -- only called from `cipher_dispatch.cpp:539` (`cipher_training_step_hook`), which itself has no in-tree caller.

**OUTBOUND:**
- `cipher_struct_lookup_init`, `cipher_struct_lookup`, `cipher_struct_override_layer` -- `cipher_structural_lookup.cpp`.
- `cipher_liquid_record_passthrough`, `cipher_liquid_record_substitution`, `cipher_liquid_update_grad_ema` -- `cipher_lnn.cpp` / `cipher_liquid_state.cu`.
- `getenv`, `memset`, `fprintf`, `sqrtf`, `fabsf` -- libc / libm.

**kmod ioctls:** none. **External libraries:** libc, libm.

### LOGIC-AS-CODED vs LOGIC-AS-DOCUMENTED

**Plan section 3.0 (CIPHER_REENGINEERING_PLAN.md:293)** lists `cipher_oracle_decide` at `L293-377`. **Correction:** actual range is L293-384 (return PERMIT is at L383). Minor citation drift.

**Plan section 3.0 (line 86)** says: *"5-gate logic per audit_section_1a (phase / min-confidence / structural-lookup / EMA-demotion / N<=4)."* -- **Verified.** The 5 gates match the code at L323/335/352/358/365.

**Drift -- header `cipher_oracle.h:38-40` claims:**
> *"L3.9: Warmup (steps 0-499): gradient variance HIGH -> all substitution DISABLED."*

Code path: `topo_detect_inference` (called BEFORE the warmup gate, L319) can flip `detected_phase` from 0 to 1 (CONVERGENCE) without ever receiving a gradient signal -- just from observing 64+ kernels of pure GEMM/ELEMENTWISE/REDUCTION classes (L57-71) or 200+ kernels with no training-ops. **So in inference mode, substitution turns on at the 64th kernel; the documented "0-499 disabled" is overridden by the topological detector.** This is *intentional* per the L17-25 file header comment, but the `cipher_oracle.h` header doc-comment was not updated to reflect it. Drift: minor -- code is intentional; header doc is stale.

**Drift -- MFU feedback documentation.** `cipher_oracle.h:213-214`: *"MFU feedback -- call periodically with current MFU from telemetry. Adjusts oracle aggressiveness based on observed hardware utilization."* The implementation (L478-483) maps `mfu < 0.5 -> -15`, `< 0.65 -> -10`, `< 0.75 -> -5`, `< 0.85 -> 0`, `>= 0.85 -> +5`. The min_confidence default is 60, ITERATIVE_CUSTOM kernels get confidence 40. With mfu=0.4 -> effective_conf=45 -> ITERATIVE_CUSTOM still denied (40 < 45). With mfu=0.4 and floor at 20: effective_conf=45 (`if effective < 20: 20`; 60-15=45, not below floor). To let ITERATIVE_CUSTOM pass, would need mfu below ~0.5 AND min_confidence already low. **Net effect**: in practice the MFU feedback gates GEMM (conf=85) only when very high -- once min_confidence rises above 85, GEMM stops too. This needs careful production tuning. Drift: minor; not a contract break.

**Drift -- `update_phase_detector` advances `step_count` even in manual override.** Comment at L131 says: *"Always advance step counter -- even in manual override mode so EMA baseline establishment is not blocked."* Looks intentional. Verified -- L132 unconditional, L134 returns early after.

**Drift -- `cipher_struct_override_layer` is called from inside `update_ema_monitor`** (L252) but **only when liquid->initialized is true** is `liquid->device->layer[i].perm_passthrough = 1` also set (L249-250). If `liquid` is null (the fusion port may pass null), the layer is still flagged in oracle's `permanently_demoted[]` and in structural_lookup's `g_layer_overrides[]`, but not in liquid. **Fusion implication:** Tree B port with null liquid still gets the safety effect from oracle + struct_lookup; the liquid sync is lost but liquid isn't ported anyway. OK to null-pass.

### CONTRIBUTION TO SYSTEM
**LOAD-BEARING.** Without the oracle gates, every classified kernel would attempt substitution. The structural lookup gate (Gate 3) at L351 -- particularly the *Last-3-layers* rule via `cipher_structural_lookup.cpp:221-226` -- is what prevents CIPHER from corrupting model output by substituting the final logit projection. The N<=4 rule (L365-378) is what prevents catastrophic Lipschitz amplification of per-op error.

Even in current state where the registry is starved and almost everything ports through, the oracle's **billing** functions feed the telemetry layer (`cipher_telemetry.cpp:321`), giving operators a per-launch view of GEMM/non-GEMM dispatch counts. The MFU feedback (L469-483) is a thoughtful piece of design that adapts the oracle's aggressiveness to current hardware load -- it should survive the port verbatim.

### FUSION POINTS

**Oracle decision call site -- Tree B has no equivalent.**
- **Contract:** `cipher_oracle_decide(state, query) -> {PERMIT, DENY}`. Query carries `(layer_idx, total_layers, op_class, confidence, kernel_name, is_backward, is_optimizer)`.
- **Honored on both sides?** Tree B has no oracle. The actuator registry pattern (`cipher_rt_matmul_dispatch.c:52`) currently has no equivalent of "ask oracle before actuating."
- **Action: PORT-AS-IS + add call site.** Port `cipher_oracle.cpp` into `cipher_rt_phase4/cipher_rt_oracle.cpp` with no internal logic changes. Add a call to `cipher_rt_oracle_decide()` inside the new classifier substrate (mirroring Plan section 7 Week-2 instruction). Severity: minor; the logic is self-contained.

**Liquid-state coupling (`state->liquid` pointer).**
- **Contract:** Optional. NULL-safe everywhere in oracle code (verified at L197, L248, L374, L396, L408).
- **Action: pass NULL** to `cipher_oracle_init` in Tree B port. The downstream `cipher_liquid_record_*` calls become no-ops. Severity: minor.

**Structural lookup coupling.**
- **Contract:** `cipher_struct_lookup(&sctx)` is called inside `cipher_oracle_decide` at L351. Must port `cipher_structural_lookup.{h,cpp}` together.
- **Action: PORT-AS-IS.** Severity: none. (Audited separately below.)

**EMA divergence demotion path (L252) calls `cipher_struct_override_layer(i, true)`.**
- **Contract:** A divergent layer is permanently demoted in BOTH oracle state (L246) AND structural lookup's layer-override table. Two-write protocol.
- **Honored on both sides?** Yes, as long as `cipher_structural_lookup.cpp` ports as well.
- **Severity:** minor.

**`CIPHER_FORCE_PERMIT` env var.**
- **Contract:** Set to "1" to bypass all gates. Cached on first call.
- **Action: PORT-AS-IS** -- survives unchanged into Tree B. Useful for testing the actuator-substitution path without the safety gates blocking. Severity: none.

---

## Section /home/ubuntu/cipher-may13-evidence/include/cipher_classify.hpp

### PURPOSE
From header (L1-30): *"Operation Classification Engine. Classifies intercepted GPU kernels into 7 mathematical families using only the geometry exposed by cuLaunchKernel: grid dims, block dims, shared mem, and kernel function pointer. Zero training data. Zero ML inference. Ships as a standalone header -- no CUDA dependency. CACHE: 512-slot open-addressing hash on function pointer. Lock-free reads. Last-writer-wins on collision. Hot path: 0.65ns. Cold path: <1ns. SUCCESS CRITERION: 95%+ correct on Llama-3 workload. <100ns latency."* This is the L3.1 substrate.

### PUBLIC SURFACE
- `enum class cipher::OpClass : uint8_t { GEMM=0, ATTENTION=1, CONVOLUTION=2, ELEMENTWISE=3, REDUCTION=4, MEMCPY_TRANSPOSE=5, ITERATIVE_CUSTOM=6, UNCLASSIFIED=0xFF }` -- L45-54.
- `struct cipher::ClassifyResult { OpClass op; uint8_t confidence; bool cache_hit; }` -- L57-61.
- `struct cipher::KernelGeom { uint32_t gx,gy,gz,bx,by,bz,shared_bytes; const void* fn; }` -- L64-69.
- `inline ClassifyResult cipher::classify(const KernelGeom& g) noexcept` -- L198-224.
- `inline ClassifyResult cipher::classify_launch(fn, gx,gy,gz, bx,by,bz, shared_bytes) noexcept` -- L227-234. Wrapper over `classify`.
- `inline const char* cipher::opclass_name(OpClass op) noexcept` -- L237-249.

### CONTROL FLOW

**`classify(g)` -- pseudo-code (L198-224):**

```
key = (uintptr_t)g.fn
slot = detail::cache_slot(key)             # slot index = key & 511
# Hot path: cache check
if slot.key.load(relaxed) == key:
    cached_op = slot.op.load(relaxed)
    if cached_op != 0xFF:
        return {OpClass(cached_op), slot.conf.load(relaxed), cache_hit=true}
# Cold path
op = detail::fingerprint(g)
conf = (op == ITERATIVE_CUSTOM) ? 40 : 85
slot.key.store(key, relaxed)
slot.op.store(uint8(op), relaxed)
slot.conf.store(conf, relaxed)
return {op, conf, cache_hit=false}
```

**`detail::fingerprint(g)` -- pseudo-code (L94-174):**

```
threads = bx * by * bz
grid_vol = gx * gy

# ATTENTION (checked BEFORE GEMM)
if is_2d_grid && bx==128 && shared_bytes>=32768 && aspect>=8.0:
    return ATTENTION
if is_2d_grid && shared_bytes>=49152 && bx<=256 && by==1 && aspect>=4.0:
    return ATTENTION

# GEMM
if is_2d_grid && is_1d_block && bx>=64 && large_shared && grid_vol>=16 && aspect<32:
    return GEMM
if is_2d_grid && shared_bytes>=16384 && 128<=threads<=512 && by<=4:
    return GEMM

# ELEMENTWISE
if is_1d_block && tiny_shared && gy==1 && gz==1: return ELEMENTWISE
if gz==1 && gy==1 && shared_bytes==0 && bx>=128: return ELEMENTWISE

# MEMCPY/TRANSPOSE
if is_2d_grid && square_block:
    tile2 = bx*by
    if shared_bytes == tile2*4 || tile2*2 || tile2*8:
        return MEMCPY_TRANSPOSE
if gz==1 && gy>1 && shared_bytes==0 && bx==256 && by==1:
    return MEMCPY_TRANSPOSE

# REDUCTION
exp_f32 = bx * 4; exp_f16 = bx * 2; exp_f64 = bx * 8
reduction_shared = (shared in {exp_f32, exp_f16, exp_f64})
if bx>=128 && is_1d_block && reduction_shared: return REDUCTION
if bx<=64 && gy==1 && gz==1 && reduction_shared && gx>=64: return REDUCTION

# CONVOLUTION
if is_3d_grid && gz>=2 && 512<=shared<=16384: return CONVOLUTION
if is_2d_grid && by>=2 && !square_block && gz==1 && bx==32 && shared>=4096:
    return CONVOLUTION

# ITERATIVE_CUSTOM (fallback)
return ITERATIVE_CUSTOM
```

**`detail::cache_slot(key)` -- L187-190:** `static std::array<CacheSlot, 512> tbl` (function-local static, zero-initialized on first call). Returns `tbl[key & 511u]`. Lifetime is process-wide.

### STATE
**Module-level state:** the 512-slot cache `detail::cache_slot::tbl` (L188). Each slot is `{atomic<uintptr_t> key, atomic<uint8_t> op, atomic<uint8_t> conf}` = 16 bytes, total 8 KB. Zero-initialized on first call (function-local static).

### CONCURRENCY
**Threading model:** lock-free hot path. Every relaxed atomic load/store. No mutex.

**Race semantics:** last-writer-wins on collision. If two distinct function pointers hash to the same slot, the second observer overwrites -- the next observation of the first function pointer falls into the cold path and re-fingerprints. *Safe but possibly inefficient under load.*

**Signal-safety:** YES -- pure C++ with only `std::atomic` ops, no allocations after first call. Could be called from a signal handler if needed (though no such use site exists).

### DEPENDENCIES
**INBOUND:**
- `cipher_dispatch.cpp:342, 430` -- top-level dispatch calls `classify_launch`.
- `src/cipher_dispatch.cpp:406, 523` -- the silently-excluded src/ shadow has the same calls.

**OUTBOUND:** none -- pure header library. Uses `<cstdint>`, `<cstring>`, `<array>`, `<atomic>` only.

**External libraries:** none.

### LOGIC-AS-CODED vs LOGIC-AS-DOCUMENTED

**Header claim "0.65ns hot path":** A single relaxed atomic load on x86 is ~1-2 cycles. With branch mispredict miss, ~10 cycles. 0.65 ns at 3.5 GHz = 2.3 cycles. Plausible for a hit on a hot cache line. Drift: marketing -- believable but unverified.

**Header claim "95%+ correct on Llama-3 workload":** Cannot be verified from code alone. The fingerprint heuristic is hand-tuned against H100 GEMM tile sizes (`tile_M=256, tile_N=128, block=(256,1,1)`) -- robust within that envelope, fragile outside it.

**Drift -- ATTENTION rules are H100-specific.** The two ATTENTION rules (L102-110) require `shared_bytes >= 32768` (32 KB) and `>= 49152` (48 KB). H100 has 228 KB max shared per SM. A100 has 164 KB. On older GPUs that compile FlashAttention with smaller shared mem, this rule misfires. Drift: minor -- the deployed target is H100.

**Drift -- ELEMENTWISE rules can collide with REDUCTION.** Both check `is_1d_block && gy==1 && gz==1`. ELEMENTWISE requires `tiny_shared (<=128)`. REDUCTION requires `shared == bx * {2|4|8}`. So for `bx=64` REDUCTION needs shared `{128, 256, 512}`. Shared=128 satisfies BOTH `tiny_shared` and `bx*2` -- ELEMENTWISE checks first (L124), so this kernel classifies as ELEMENTWISE. **For RMSNorm with bx=64 and shared=128, it gets ELEMENTWISE not REDUCTION.** Drift: borderline -- RMSNorm in PyTorch typically uses bx=256 or bx=1024, which lands clean in REDUCTION at L155. But bespoke RMSNorm kernels at bx<=64 will misclassify. The Op-14 SiLU/RMSNorm fused kernels (per CLAUDE.md Phase 3 audit) launch at `cipher_rmsnorm_fp16` with grid (32, 1) block (256, 1) -- that lands at L155 (REDUCTION) safely.

**Drift -- `is_2d_block` at L81** returns true if `by >= 2 OR bx == by`. This is wrong-feeling (the OR is broad -- at by=1 bx=1 it's true via bx==by). But the helper isn't used in fingerprint (`square_block` is used instead at L85). Vestigial helper.

**Drift -- confidence values are ONLY two: 40 (ITERATIVE_CUSTOM) or 85 (everything else).** No graduated confidence. Plan section 3.0 implies a continuous confidence; the implementation is binary-ish. Minor.

### CONTRIBUTION TO SYSTEM
**LOAD-BEARING.** Without `classify_launch`, `cipher_dispatch` can't choose a path. The 7-class taxonomy is also consumed by the structural lookup (`g_opclass_default_fp[7]` at `cipher_structural_lookup.cpp:92`) -- six of the seven defaults are baked into structural lookup logic. Removing this header: every downstream consumer breaks.

### FUSION POINTS

**Port unchanged -- Tree B has no equivalent.**
- **Contract:** Standalone C++17 header, no external dependencies. Can be `#include`'d directly into Tree B.
- **Action: PORT-AS-IS** to `cipher_rt_phase4/cipher_rt_classify.h`. The header is self-contained. Severity: none.
- **Renaming note (Plan section 7 W1):** The plan calls for renaming functions to `cipher_rt_*` prefix. For an `inline` header function in namespace `cipher`, prefer keeping the namespace and adding a thin wrapper `extern "C" cipher_rt_classify_result_t cipher_rt_classify_launch(...)` in `cipher_rt_classify.cpp` so the C-only cuBLAS shim can call it.

**Cache hot-path budget.** Tree B's matmul-dispatch latency budget per Plan section 5 is ~12 ns per launch. Classify cache hit is ~5-10 ns; cold path is the full fingerprint walk (~30-50 ns). Hot/cold mix in Tree B's workload (steady-state ~99% hit) keeps the budget. Severity: none.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_kernel_table.cpp

### PURPOSE
From header (L1-6) and `include/cipher_kernel_table.h:1-21`: caches `CUfunction -> (name, classification, grid/block, param_count, param_offsets, param_sizes)` for every observed kernel. Resolves names via `cuFuncGetName` first, falls back to `dladdr`. Probes `cuFuncGetParamInfo` but accepts that it returns 0 params for runtime-API kernels (registered via `__cudaRegisterFunction`). Dumps the full table to JSON at process exit. **DUAL-BUILT:** compiled into both `libcipher_hook.so` and `libcipher_rt.so` per `Makefile:29` (only 3 hook-only files are filtered out; this is not one of them). Verified by `DEEP_INSPECTION_REPORT.md:76`.

### PUBLIC SURFACE
- `extern "C" const CipherKernelEntry* cipher_kt_observe(fn, gx,gy,gz, bx,by,bz, smem)` -- L222-251.
- `extern "C" unsigned cipher_kt_size(void)` -- L253-255.
- `extern "C" const char* cipher_kt_name_for(void* fn_handle)` -- L257-267.
- `extern "C" int cipher_kt_dump_json(const char* path)` -- L269-307.
- `__attribute__((destructor)) static void cipher_kt_atexit_dump(void)` -- L311-317.

### CONTROL FLOW

**`cipher_kt_observe(fn, gx,gy,gz, bx,by,bz, smem)` -- pseudo-code (L222-251):**

```
if !fn: return NULL
probe = hash_fn(fn)                          # splitmix64 mod KT_SIZE
for i in 0..KT_SIZE:
    slot = (probe + i) & KT_MASK
    e = g_kt[slot]
    if e.fn_handle == fn:
        e.observe_count++                    # non-atomic
        return &e
    if e.fn_handle == NULL: break            # miss
# Slow path under coarse mutex (function-static mu)
lock(mu)
slot = insert_locked(fn, gx,gy,gz, bx,by,bz, smem)
unlock(mu)
if slot < 0: return NULL                     # table full
return &g_kt[slot]
```

**`insert_locked(fn, gx,gy,gz, bx,by,bz, smem)` -- L178-218:**

```
probe = hash_fn(fn)
for i in 0..KT_SIZE:
    slot = (probe + i) & KT_MASK
    e = g_kt[slot]
    if e.fn_handle == NULL:
        e.fn_handle = fn
        nm = resolve_name(fn)              # cuFuncGetName then dladdr fallback
        if nm: strncpy(e.name, nm, 127); e.name[127]=0
        else:  snprintf(e.name, "<unresolved-%p>", fn)
        e.category = classify(nm)          # the strstr-based name classifier
        e.param_count = probe_param_info(fn, e.param_offsets, e.param_sizes)
        e.first_grid_x = gx; ... (capture geometry)
        e.observe_count = 1
        g_kt_used.fetch_add(1, relaxed)
        if env CIPHER_KERNEL_TABLE_VERBOSE: stderr log
        return slot
    if e.fn_handle == fn: return slot       # race won by another thread
return -1
```

**`classify(nm)` -- pure name-prefix-matching (L61-107):** Hard-coded order with 18 categories. Mostly `strstr` (substring) not strict prefix. Catches FlashAttention variants first, then GEMM family (cutlass3x, wgmma, gemm, splitkreduce, nvjet_*), then norm/activation kernels (rms_norm, layer_norm, RoPE, silu, gelu, softmax), then embedding/elementwise/copy/transpose/cast. Returns `CIPHER_KT_UNKNOWN` on no match.

**`resolve_libcuda()` -- L37-49:** Once-resolve cached behind atomic + mutex. dlopen(libcuda.so.1) with RTLD_NOW|RTLD_NOLOAD first (don't load if absent), then RTLD_LAZY. Resolves `cuFuncGetName`, `cuFuncGetParamInfo`, `cuFuncGetAttribute`.

**`probe_param_info(fn, offsets[16], sizes[16])` -- L160-174:** Iterates ordinals 0..15, calls `cuFuncGetParamInfo(fn, i, &off, &sz)`. First failing slot is the parameter count. *Per the CLAUDE.md note in the codebase*, this returns 0 for kernels registered via `__cudaRegisterFunction` (i.e. every PyTorch nn.Linear/RMSNorm/etc kernel).

**`cipher_kt_dump_json(path)` -- L269-307:** Walks `g_kt[]`, escapes `name` for JSON, writes `{fn, name, category, category_id, param_count, grid, block, smem_bytes, observe_count}` per entry. Returns count written.

**`cipher_kt_atexit_dump` -- L311-317:** Destructor priority unspecified (defaults to "after all priority-N destructors"). Gates on `CIPHER_KERNEL_TABLE_VERBOSE`. Writes `/tmp/cipher_kernel_table.json` on exit.

### STATE
- `static CipherKernelEntry g_kt[4096]` (L22) -- BSS-initialized; ~4096 x ~240 bytes = ~1 MB. Hash table with linear probing.
- `static std::atomic<unsigned> g_kt_used` -- count of populated slots.
- `static std::atomic<int> g_resolved` + `static std::mutex g_resolve_mu` -- once-resolve gate.
- `static std::mutex mu` (function-static at L245) -- insert mutex.
- Function pointers `g_get_name`, `g_get_paraminfo`, `g_get_attribute` -- populated by `resolve_libcuda`.

### CONCURRENCY
**Threading model:** lock-free read fast path, coarse-mutex slow path. The hot path (`cipher_kt_observe` first loop) does no locking -- just linear probing.

**Race semantics:** `e.observe_count++` is non-atomic (L237). Acceptable -- count is informational.

**Insert race:** `cipher_kt_observe` falls into the function-static `std::mutex mu` (L245-246) under contention. `insert_locked` re-walks the probe chain inside the mutex, so two threads inserting the same fn handle resolve to one slot.

**Signal-safety:** atexit destructor reads `getenv`, calls `fprintf`, calls `cipher_kt_dump_json` which calls `fopen/fprintf/fclose`. NOT signal-safe; relies on normal exit.

### DEPENDENCIES
**INBOUND:**
- `cipher_kt_observe` -- `src/cipher_intercept_cudart.cpp:2463, 2610` (every `cuLaunchKernel`/`cuLaunchKernelEx` intercept).
- `cipher_kt_name_for` -- `src/cipher_flow_patterns.cpp:194`, `src/cipher_flow_recorder.cpp:42`.
- `cipher_kt_size`, `cipher_kt_dump_json` -- `tests/test_kernel_table.py`.

**OUTBOUND:** `dlopen("libcuda.so.1")`, `dlsym` for `cuFuncGetName`/`cuFuncGetParamInfo`/`cuFuncGetAttribute`, `dladdr`, libc (`strncpy`, `strstr`, `snprintf`, `fopen`, `fprintf`, `fclose`, `getenv`).

**kmod ioctls:** none.

**External libraries:** libcuda.so.1 (lazy-resolved), libdl, libc.

### LOGIC-AS-CODED vs LOGIC-AS-DOCUMENTED

**DEEP_INSPECTION_REPORT.md:76 confirms:** *"5 source files are built into BOTH `libcipher_hook.so` AND `libcipher_rt.so` (because the Makefile excludes only 3 hook-side files from RT_CPP_SRC, not all 7): `cipher_kernel_table.cpp`, ..."* **Verified.** This is a v1 fusion concern -- the unified runtime drops libcipher_hook.so, so each dual-built file must port to exactly one location.

**Plan section 3.0 (line 99):** *"cipher_kernel_table.cpp -- kernel-name recognition (FLASHATTN/GEMM/RMSNORM/...); CLASSIFY backing (DUAL-BUILT, hook+RT)."* -- **Verified.**

**Drift -- `classify()` uses `strstr` not strict prefix.** The header doc (`cipher_structural_lookup.cpp:14`) talks about "kernel name prefix table"; the kernel_table classify uses `strstr` (substring). They are different classifiers -- kernel_table maps to category labels (`CIPHER_KT_*`), structural_lookup maps to FULL_PRECISION/SUBSTITUTABLE. Different surfaces. No drift between the two, but they overlap conceptually.

**Drift -- table size 4096 may overflow.** Modern PyTorch loads ~1500-2500 kernels at startup (per the CLAUDE.md's reference to kernel-table dumps). 4096 slots with linear probing is comfortable but not infinite. Under heavy plugin load (Triton, Inductor JIT), could fill. The `insert_locked` returns -1 on full table; observers handle gracefully.

**Drift -- duplicated `CipherKernelEntry` typedef name.** `include/cipher_kernel_table.h:53` defines `struct CipherKernelEntry { void* fn_handle; char name[128]; ... }`. `include/cipher_param_recovery.h:30` defines a DIFFERENT `struct CipherKernelEntry { const void* host_fun; const char* device_fun; ... }`. **Symbol collision risk:** both headers cannot be `#include`'d into the same translation unit. In practice, `cipher_intercept_cudart.cpp` uses both -- by forward-declaring or guarding. Severity: minor -- but worth resolving in port (rename one).

### CONTRIBUTION TO SYSTEM
**LOAD-BEARING for downstream observability.** Without `cipher_kt_observe`, `cipher_flow_patterns` and `cipher_flow_recorder` lose their name lookups. With it, the system gets a per-kernel name + category map for telemetry/debugging.

Within `cipher_dispatch`'s actuation path: `cipher_kt_observe` is NOT called from `cipher_dispatch`. The dispatcher uses geometry-only classify (`cipher_classify.hpp`). So kernel_table is a parallel-name-classifier substrate, not a hot-path actuator. **Could be deferred to v1.5** if needed.

### FUSION POINTS

**Dual-build resolution.**
- **Contract:** kernel_table must live in exactly one DSO in Tree B. The current dual-build was tolerated by LD_PRELOAD-position resolution (hook wins).
- **Action: PORT-AS-IS to `cipher_rt_phase4/cipher_rt_kernel_table.cpp`,** drop the libcipher_hook.so copy. Severity: minor.

**`CipherKernelEntry` rename to avoid collision with `cipher_param_recovery.h`.**
- **Contract:** Both headers define a struct with the same name but different fields. In the unified runtime, both port together (kernel_table for category labels, param_recovery for cubin metadata).
- **Action: REFACTOR-REQUIRED.** Rename one struct (e.g. `CipherKtEntry` and `CipherParamEntry`). Severity: minor; one-pass rename across the port.

**Lazy libcuda resolution.**
- **Contract:** `dlopen("libcuda.so.1", RTLD_NOLOAD)` then RTLD_LAZY. Honored on both sides -- libcuda.so.1 is always present where CIPHER runs.
- **Action: PORT-AS-IS.** Severity: none.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_structural_lookup.cpp

### PURPOSE
From `include/cipher_structural_lookup.h:1-27`: *"Zero-compute, zero-latency table of operations that are ALWAYS run at full precision regardless of what the classification engine says."* L3.8 in the L3.x naming scheme. Hardcoded transformer-architecture rules: MHSA, last-3-layers, loss kernels, optimizer step, embedding/first-layer kernels all force FULL_PRECISION. FFN/intermediate LayerNorm/RMSNorm are SUBSTITUTABLE. Backed by a prefix-rule table + a 512-slot FNV-prefix name cache.

### PUBLIC SURFACE
- `void cipher_struct_lookup_init(void)` -- L169-188.
- `CipherStructLookupResult cipher_struct_lookup(const CipherStructContext* ctx)` -- L194-263.
- `void cipher_struct_override_layer(uint32_t layer_idx, bool full_precision)` -- L269-275.
- `void cipher_struct_lookup_report(void)` -- L281-300.

### CONTROL FLOW

**`cipher_struct_lookup(ctx)` -- pseudo-code (L194-263):**

```
out = {SUBSTITUTABLE, NONE, false}
# Gate 1: warmup -> always FP
if ctx->training_phase == 0:
    return {FULL_PRECISION, WARMUP, false}                    # L199-203
# Gate 2: optimizer step -> always FP
if ctx->is_optimizer_step:
    return {FULL_PRECISION, WEIGHT_UPDATE, false}             # L206-210
# Gate 3: per-layer override
if ctx->layer_idx < 256 && g_layer_overrides[ctx->layer_idx]:
    return {FULL_PRECISION, USER_OVERRIDE, false}             # L213-218
# Gate 4: last-3-layers
if ctx->total_layers > 3 && ctx->layer_idx >= ctx->total_layers - 3:
    return {FULL_PRECISION, LAST_LAYERS, false}               # L221-226
# Gate 5: kernel-name prefix
if ctx->kernel_name:
    cached = name_cache_lookup(ctx->kernel_name)
    if cached.cache_hit: return cached
    # walk g_name_rules[] (linear scan)
    for rule in g_name_rules until sentinel:
        if strncmp(ctx->kernel_name, rule.prefix, strlen(rule.prefix)) == 0:
            out = {rule.force_fp ? FP : SUB, rule.reason, false}
            name_cache_insert(ctx->kernel_name, out.result, out.reason)
            return out
# Gate 6: op-class default
if ctx->op_class < 7:
    out.result = g_opclass_default_fp[op_class] ? FP : SUB
    if op_class == 1: out.reason = MHSA
    return out                                                # L250-258
# Gate 7: unknown
return {UNKNOWN, NONE, false}                                 # L261
```

**`cipher_struct_lookup_init()` -- L169-188:** Clears name cache and layer override array. **Pre-warms** the cache by inserting each prefix-rule into the name cache (L178-183). This is a *weird* prewarming: the cache key is `fnv1a_prefix(rule.prefix)` over up to 32 bytes -- so the cache stores the rule's PREFIX. The cache hit at lookup time computes `fnv1a_prefix(ctx->kernel_name)` over the first 32 bytes of the actual kernel name. **For the cache hit to fire, the first 32 bytes of the kernel name must equal the rule prefix.** If the kernel name is longer than the prefix (typical: `flash_attn_fwd_kernel_v2`), the first 32 bytes differ -- cache miss, fall through to strncmp walk.

**Net effect**: the prewarm is largely ineffective. The walk at L234-246 is the real path.

**`name_cache_lookup(name)`:** FNV-1a over first 32 bytes of `name`, slot = hash & 511, check `valid && name_hash == h`. Returns `{result, reason, cache_hit=true}` on hit.

**`cipher_struct_override_layer(layer_idx, full_precision)`:** Sets `g_layer_overrides[layer_idx] = 1` or 0. Logs to stderr.

**Branch determinants.**
- L199 warmup: from oracle's `state->phase.detected_phase`. With topological auto-detect (`cipher_oracle.cpp:30-74`), this flips to 1 (CONVERGENCE) after 64+ inference-only kernels.
- L206 optimizer_step: from oracle's query (`cipher_oracle.cpp:349`). Always false in `cipher_dispatch.cpp:463` (set to `is_optimizer = false`). **Permanently false on inference workload.**
- L221 last-3-layers: based on `layer_idx >= total_layers - 3`. With `total_layers=80` hardcoded in `cipher_dispatch.cpp:76` (`CIPHER_INFERRED_TOTAL_LAYERS`), the last-3-layers gate fires only for `layer_idx in {77,78,79}`. Inferred layer ranges over `[0, 79]` per `(t_step_kernels/60) % 80`. So a long-enough run hits 77-79 about 3/80 = 3.75% of the time.
- L229 kernel_name: ctx->kernel_name is set to NULL in `cipher_dispatch.cpp:367, 461` ("Phase 2: cuFuncGetName()"). **Permanently NULL today.** Gate 5 never fires.
- L250 op-class default: this is the gate that actually decides today's workload. op_class=1 (ATTENTION) -> FP; op_class=6 (ITERATIVE_CUSTOM) -> FP; else SUBSTITUTABLE.

### STATE
- `static const KernelNameRule g_name_rules[]` (L23-82) -- read-only prefix table with sentinel.
- `static const bool g_opclass_default_fp[7]` (L92-100) -- read-only.
- `static uint8_t g_layer_overrides[256]` (L107) -- mutated by `cipher_struct_override_layer`.
- `static NameCacheSlot g_name_cache[512]` (L125) -- mutated on cache miss insert.

### CONCURRENCY
**Threading model:** hot-path-sync, called inside `cipher_oracle_decide`.

**Locks:** none. Name cache writes (L160-162) race; last-writer-wins. Acceptable -- cache is informational, lookups always have correctness via the linear-scan fallback.

**Signal-safety:** yes for `cipher_struct_lookup` (no syscalls, no allocations). No for `cipher_struct_override_layer` (fprintf).

### DEPENDENCIES
**INBOUND:**
- `cipher_struct_lookup_init` -- `cipher_oracle.cpp:104`, `cipher_dispatch.cpp:116, 354`.
- `cipher_struct_lookup` -- `cipher_oracle.cpp:351`.
- `cipher_struct_override_layer` -- `cipher_oracle.cpp:251` (EMA divergence demotion).
- `cipher_struct_lookup_report` -- `cipher_dispatch.cpp:530`.

**OUTBOUND:** libc (`memset`, `strncmp`, `strlen`, `fprintf`), no libcuda, no kmod.

### LOGIC-AS-CODED vs LOGIC-AS-DOCUMENTED

**Header claim "<10ns lookup":** Realistic for the early-gate path (gates 1-4 are simple compares). Gate 5 (string walk) is slower -- strncmp over up to ~50 rules x ~16 chars each = ~800 cycles = ~250 ns. Drift: minor -- the 10 ns claim is for the cache-hit path; cold path is much slower.

**Drift -- ineffective prewarm (described above in CONTROL FLOW).** The prewarm at L178-183 stores rule.prefix hashes (e.g. FNV1a("flash_attn", 32 bytes mostly null)), but real lookups hash actual kernel names like `flash_attn_fwd_kernel_<long_suffix>` -- different first-32-byte hashes. The prewarm only hits when the kernel name IS the prefix exactly (rare). Severity: minor; the fallback walk is functionally correct, just slower than advertised.

**Drift -- `ctx->kernel_name` is always NULL in current call sites.** `cipher_dispatch.cpp:367, 461` explicitly set `.kernel_name = NULL`. So Gate 5 never fires; the only working gates are 1-4 and 6. **Severity:** semi-significant -- half of the rule table (the kernel-name prefix matching) is dead code on the current call path. Fusion port should wire `cuFuncGetName` result into `oq.kernel_name` to activate the rule table.

**Drift -- `g_name_rules` table contains LOSS / OPTIMIZER / GRAD_ACCUM rules that are inference-irrelevant.** Lines 25-67 list many training-only kernel names (nll_loss, adam, weight_update, grad_accum, etc.). For an inference-only target (which is what topo_detect_inference declares the workload to be), these rules never fire. They're correct but unused.

**Drift -- `op_class==1 (ATTENTION) -> FP` (L92-100, L255).** This is the rule that protects MHSA causal masking. But `cipher_dispatch.cpp:444` already kills ITERATIVE_CUSTOM (which is a subset). For pure GEMM/ELEM/REDUCE workloads, no ATTENTION classification fires at the dispatcher -- `g_opclass_default_fp[1]=true` is a backstop.

### CONTRIBUTION TO SYSTEM
**LOAD-BEARING for safety.** The Gate-4 last-3-layers rule and Gate-2 optimizer-step rule are what keep CIPHER from silently corrupting the last logit projection during training or the optimizer-step kernels during weight update. Without this file, oracle Gate 3 (structural-lookup) always returns SUBSTITUTABLE; safety drops to N<=4 alone.

But: many of its rules are inactive today because `ctx->kernel_name` is null at all call sites. The active value is from Gates 1, 2, 4, 6.

### FUSION POINTS

**Port-as-is.**
- **Contract:** Pure logic, no external state, no libcuda. C-callable. NULL-safe input (handles `ctx->kernel_name == NULL`).
- **Action: PORT-AS-IS** to `cipher_rt_phase4/cipher_rt_structural_lookup.cpp`. Severity: none.

**Activating Gate 5 (kernel-name rules).**
- **Contract:** Tree B has `cuFuncGetName` via `cipher_rt_kernel_table.cpp` (post-port). The classifier substrate should populate `oq.kernel_name = cipher_kt_name_for(fn)` before calling `cipher_oracle_decide`.
- **Action: REFACTOR-REQUIRED (small).** Add the kernel_name lookup at the new pre-launch hook. Severity: minor.

**Override table coupling to oracle EMA demotion.**
- **Contract:** `cipher_struct_override_layer(i, true)` is called from `cipher_oracle.cpp:251` on EMA divergence.
- **Honored on both sides?** Yes if both files port together.
- **Severity:** none.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_param_recovery.cpp

### PURPOSE
From file header (L1-26): *"Builds a hostFun -> kernel-info registry by intercepting cudart's two internal symbols `__cudaRegisterFatBinary` and `__cudaRegisterFunction`."* Parses fatbin/cubin sections directly (no driver query) to extract `EIATTR_KPARAM_INFO` records (`(ordinal, offset, size_align)` 4xu32 records inside `.nv.info.<funcname>` ELF sections). Two complementary indices: `by_host[host_fun]` (set by `__cudaRegisterFunction` interception) and `by_name[kernel_name]` (set by post-hoc DSO scanning of libtorch_cuda.so / libcublas.so). Also a `by_func[cufunc]` index populated via `cipher_param_register_cufunc` after `cuModuleGetFunction`.

### PUBLIC SURFACE
- `int cipher_param_recovery_init(void)` -- L538-542.
- `int cipher_param_recovery_enabled(void)` -- L544-546.
- `void cipher_param_register_fatbin(void* handle, const void* fatbin_blob)` -- L548-563.
- `void cipher_param_register_function(void* fat_handle, const void* host_fun, const char* device_fun, const char* device_name)` -- L565-583.
- `const CipherKernelEntry* cipher_param_lookup(const void* host_fun)` -- L585-600.
- `const CipherKernelEntry* cipher_param_lookup_by_name(const char* kernel_name)` -- L455-464.
- `int cipher_param_scan_dso(const char* path)` -- L400-436.
- `int cipher_param_scan_pytorch(void)` -- L438-453.
- `int cipher_param_ingest_image(const void* image)` -- L468-514.
- `void cipher_param_register_cufunc(const void* cufunc, const char* name)` -- L516-525.
- `const CipherKernelEntry* cipher_param_lookup_cufunc(const void* cufunc)` -- L527-536.
- `void cipher_param_table_dump(int to_file)` -- L611-665.
- `void cipher_param_stats(CipherParamStats* out)` -- L602-609.
- `__attribute__((constructor(113))) static void cipher_param_recovery_autoinit()` -- L667-668.

### CONTROL FLOW

**`cipher_param_register_fatbin(handle, fatbin_blob)` -- L548-563:**

```
if !handle: return
blob = fatbin_blob
if FatbinWrapper at blob has magic 0x466243B1:
    blob = wrap->data         # unwrap to inner FatbinHeader
lock(s.table_mu); s.fatbin_blob[handle] = blob
s.fatbins_seen++
```

**`cipher_param_register_function(fat_handle, host_fun, device_fun, device_name)` -- L565-583:**

```
if !host_fun: return
lock(s.table_mu)
if s.by_host.count(host_fun): return    # idempotent
k = new KernelEntry
k->e = {host_fun, device_fun, device_name, fat_handle, param_count=0, parsed_ok=0}
s.by_host[host_fun] = k
unlock; s.functions_seen++
```

**`cipher_param_lookup(host_fun)` -- L585-600:**

```
if !host_fun: return NULL
lock(s.table_mu)
it = s.by_host.find(host_fun)
if not found: return NULL
k = it->second; unlock
# Lazy parse
lock(k->parse_mu)
ensure_parsed_locked(k)
unlock(k->parse_mu)
return &k->e
```

**`ensure_parsed_locked(k)` -- L373-396:**

```
if k->e.parsed_ok != 0 || param_count > 0: return    # already parsed
blob = lookup(s.fatbin_blob, k->e.fat_handle)
if !blob: parse_failed++; return
fname = k->e.device_fun ?: k->e.device_name
if !fname: parse_failed++; return
if parse_fatbin_for_kernel(blob, fname, k):
    parsed_ok++
else:
    parse_failed++
```

**`parse_fatbin_for_kernel(blob, func_name, out)` -- L338-371:**

```
hdr = (FatbinHeader*)blob
if hdr->magic != 0xBA55ED50: return false
p   = blob + hdr->headerSize
end = blob + headerSize + fatSize
while p + 8 < end:
    kind = read u16 le
    hsize = read u32 at p+4
    psize = read u64 at p+8
    if !valid: return false
    if kind == 2 (ELF cubin):
        if parse_cubin_for_kernel(p + hsize, psize, func_name, out): return true
    p += hsize + psize
return false
```

**`parse_cubin_for_kernel(elf, elf_size, func_name, out)` -- L109-185:**

```
ehdr = ELF64 header at elf
verify ELF magic, ELFCLASS64, e_shoff/e_shnum non-zero, e_shstrndx valid
strtab = elf + shdr[e_shstrndx].sh_offset
target = ".nv.info." + func_name
for i in 0..ehdr->e_shnum:
    sec_name = strtab + shdr[i].sh_name
    if strcmp(sec_name, target) != 0: continue
    p = elf + shdr[i].sh_offset; end = p + sh_size
    n_params = 0
    while p + 4 <= end:
        fmt = p[0]; attr = p[1]; vsize = p[2]|p[3]<<8; p+=4
        adv = (fmt: NVAL=0, BVAL=1, HVAL=2, SVAL=vsize, default=vsize)
        if p + adv > end: break
        if attr == EIATTR_KPARAM_INFO (0x17) && fmt == EIFMT_SVAL && adv >= 16:
            # Parse 4xu32: (w0_skipped, w1, w2, w3)
            w1 = u32 at p+4; w2 = u32 at p+8; w3 = u32 at p+12
            ordinal = (w2 >> 16) & 0xFFFF; offset = w2 & 0xFFFF
            if ordinal >= 16:                    # alternate encoding
                ordinal = (w1 >> 16) & 0xFFFF; offset = w2 & 0xFFFF
            size = w3 & 0xFFFF
            if ordinal < 16 && n_params < 16:
                if ordinal + 1 > n_params: n_params = ordinal + 1
                out->e.params[ordinal] = {offset, size}
        p += adv
    out->e.param_count = n_params
    out->e.parsed_ok = 1
    return n_params > 0
return false
```

**`parse_cubin_all_kernels(payload, payload_size, s)` -- L189-304:** Variant that walks every `.nv.info.<name>` section in the ELF and inserts into `s.by_name`. Strips up to 4-byte prefix before ELF magic (NVCC cubin variation). Uses `e_shentsize` from file rather than `sizeof(Elf64_Shdr)` (cubins extend section headers). Bounded by `safety < 4096` loop counter to prevent runaway on malformed cubin.

**`parse_fatbin_all_kernels(base, base_size, s)` -- L307-335:** Same walker as `parse_fatbin_for_kernel` but calls `parse_cubin_all_kernels` for every section.

**`cipher_param_scan_dso(path)` -- L400-436:** mmap the DSO, scan for `FATBIN_HEADER_MAGIC = 0xBA55ED50`, validate header, call `parse_fatbin_all_kernels`. Step by 4 bytes on miss.

**`cipher_param_scan_pytorch()` -- L438-453:** Hardcoded paths: `/usr/lib/python3/dist-packages/torch/lib/libtorch_cuda.so`, `libtorch_cuda_linalg.so`, `libcublas.so.12`, `libcublasLt.so.12`. Sum results across.

**`cipher_param_ingest_image(image)` -- L468-514:** Default OFF (env `CIPHER_PARAM_INGEST=1` to enable). Detects raw ELF cubin vs fatbin, calls appropriate parser.

**`cipher_param_register_cufunc(cufunc, name)`:** Links a CUfunction handle to an existing `by_name` entry; populates `by_func` map.

**Branch determinants.**
- `cipher_param_ingest_image` defaults to OFF (env `CIPHER_PARAM_INGEST`); current sessions skip the cubin parse path.
- `cipher_param_recovery_init` defaults `enabled=1`, no env gate.

### STATE
- `static State s` (function-static singleton, L88-91) holding 4 hash maps (`by_host`, `by_name`, `by_func`, `fatbin_blob`), 4 atomic counters (`fatbins_seen`, `functions_seen`, `parsed_ok`, `parse_failed`), one `enabled` flag, one `table_mu` mutex.
- Per-`KernelEntry`: `parse_mu` mutex (lazy-parse coordination).
- Heap-allocated `KernelEntry`s (created via `new`, leaked at process exit; no destructor).

### CONCURRENCY
**Threading model:** registration is called from cudart init thread; lookups from launching threads.

**Locks:** coarse `table_mu` (one mutex per State); per-entry `parse_mu` for lazy fatbin parsing. Both are `std::mutex`.

**Race semantics:** insert race in `parse_cubin_all_kernels` (L289-296) re-checks `by_name.count(key)` under the table lock; if another thread inserted first, this thread `free`/`delete`s its tentative entry.

**Signal-safety:** no. Uses `unordered_map`, `new`/`delete`, mutexes, mmap/munmap, dlopen.

### DEPENDENCIES
**INBOUND:**
- `cipher_param_register_fatbin` -- `src/cipher_intercept_cudart.cpp:1605` (via dlsym in the `__cudaRegisterFatBinary` shim).
- `cipher_param_register_function` -- `src/cipher_intercept_cudart.cpp:1646`.
- `cipher_param_ingest_image` -- `src/cipher_intercept_cudart.cpp:1685, 1714, 1777`.
- `cipher_param_register_cufunc` -- `src/cipher_intercept_cudart.cpp:1739, 1835`.
- `cipher_param_lookup`, `_by_name`, `_cufunc` -- wired but no in-tree downstream consumer was located by grep (used externally by the Phase 1 probe scripts).

**OUTBOUND:** mmap/munmap (sys/mman.h), open/close (fcntl), fstat, libc (memcmp, strncmp, strcmp, strdup, free, new/delete), ELF parsing (elf.h types only), no libcuda call (parses cubins directly).

**kmod ioctls:** none. **External libraries:** libc, libstdc++.

### LOGIC-AS-CODED vs LOGIC-AS-DOCUMENTED

**Plan section 3.0 (L109):** *"cipher_param_recovery.cpp -- Stage 1 fatbin/cubin EIATTR_KPARAM_INFO parser; CLASSIFY backing."* -- **Verified.**

**Drift -- `cipher_param_ingest_image` defaults OFF (L474-479).** Comment says: *"we now use cuKernelGetParamInfo via the driver instead of parsing cubins ourselves (set CIPHER_PARAM_INGEST=1 to re-enable the parsing path, which is significantly slower at startup)."* So in default config, no cubin ingestion happens via `cuModuleLoadData`. The `by_host` table is still populated via `__cudaRegisterFunction` interception; the `by_name` table requires explicit `cipher_param_scan_pytorch()` or `cipher_param_scan_dso()` calls.

**Drift -- `CipherKernelEntry` struct name collides with `cipher_kernel_table.h`.** Verified -- `include/cipher_param_recovery.h:30-38` has *different fields* than `include/cipher_kernel_table.h:53-64` under the same name. Translation units that include both will fail to compile. The codebase resolves this by including only one per .cpp. Fusion concern: rename one to avoid the trap.

**Drift -- `EIATTR_KPARAM_INFO` parsing has two encoding fallback paths (L162-170, L266-275).** The "canonical" encoding puts ordinal in `(w2 >> 16) & 0xFFFF`; the "alternate" in `(w1 >> 16) & 0xFFFF`. Heuristic: if ordinal >= 16 in canonical, try alternate. This works empirically per the in-tree tests but is undocumented CUDA internals -- *fragile across CUDA major versions*. Severity: known fragility -- defer to CUDA driver query `cuKernelGetParamInfo` per the L472-473 comment, which is the post-Phase-1 approach. The cubin-parse path is preserved as fallback.

### CONTRIBUTION TO SYSTEM
**SUPPORTING.** Param recovery feeds the FlashAttention / KV-redirect / RoPE quantization work (per CLAUDE.md Stage 8b/c notes). It is not on the `cipher_dispatch` critical path -- the dispatcher uses `cipher_classify.hpp` geometry classification, not the param table.

For v1 fusion: the param recovery substrate is *prerequisite* to the L3.5 EDMD pipeline ever firing (EDMD needs to know which arg of cuLaunchKernel is the weight, activation, output -- requires param layout from the cubin). But EDMD is deferred to v2. So param_recovery can also be deferred to v1.5.

### FUSION POINTS

**Port-as-needed (deferred for v1).**
- **Contract:** Pure CPU table; standalone -- no kmod, no libcuda. Bidirectional API (register from hook, lookup from any caller).
- **Action: DEFER to v1.5 or PORT-AS-IS once EDMD-live is ready.** Severity: minor (no v1 consumer).

**`__cudaRegisterFatBinary`/`__cudaRegisterFunction` interception is the call site.**
- **Contract:** These are cudart internal symbols. Tree B's LD_PRELOAD-free deploy doesn't intercept them.
- **Action: SHIM-REQUIRED.** For Tree B port, replace fatbin interception with explicit `cipher_param_scan_pytorch()` at init time (post-cudart-load). Severity: requires-mitigation (loses the `by_host` index, retains `by_name`).

**`cipher_param_register_cufunc`/`_lookup_cufunc` for cuModule-loaded kernels.**
- **Contract:** Set when `cuModuleGetFunction` returns; read when classifier needs param layout.
- **Honored on both sides?** Tree B's NVRTC pipeline (`cipher_rt_marlin_engine.cpp` uses NVRTC for Marlin) does call `cuModuleGetFunction` but does not register with param_recovery. Wiring would be additive.
- **Severity:** minor.

**Struct-name collision (`CipherKernelEntry`).** See kernel_table fusion point -- REFACTOR-REQUIRED to rename one.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_recipes.cpp

### PURPOSE
From `include/cipher_recipes.h:1-34` and header comments here: three closed-form recipe families (L3.2 GEMM roofline, L3.3 FAVOR+ attention, L3.4 Chebyshev nonlinearities), plus the L1.3 substitution registry -- a fixed-size array of `CipherRegistryEntry` seeded at init with 32+ shapes. The registry is what `cipher_dispatch.cpp:475` queries to decide whether a (op_class, shape_hash, hw_arch) tuple has a known recipe.

### PUBLIC SURFACE
- `CipherGemmConfig cipher_recipe_gemm(uint32_t M, uint32_t N, uint32_t K, const CipherHwProfile* hw)` -- L45-106.
- `CipherAttentionConfig cipher_recipe_attention(seq_len, head_dim, num_heads, batch, epsilon, causal, hw)` -- L124-170.
- `bool cipher_recipe_attention_safe(cfg, max_acceptable_error)` -- L172-177.
- `CipherChebyshevConfig cipher_recipe_chebyshev(CipherNonlinType nonlin, uint32_t degree)` -- L214-331.
- `float cipher_chebyshev_eval(cfg, float x)` -- L192-208.
- `void cipher_registry_init(CipherRegistry* reg)` -- L346-484.
- `const CipherRegistryEntry* cipher_registry_lookup(reg, op_class, shape_hash, hw_arch)` -- L486-500.
- `bool cipher_registry_insert(reg, entry)` -- L502-508.
- `void cipher_registry_report(reg)` -- L510-522.

### CONTROL FLOW

**`cipher_recipe_gemm(M, N, K, hw)` -- L45-106:** Pure analytical roofline.

```
dtype_bytes = 2 (BF16/FP16)
flops = 2*M*N*K
bytes = (M*K + K*N + M*N) * dtype_bytes
ai = flops / bytes
ridge = peak_tflops_fp16 * 1e12 / (hbm_gbps * 1e9)
memory_bound = (ai < ridge)
l2_budget = l2_size * 0.30
tile_k = 64
tile_side = clamp(round_up_pow2(sqrt(l2_budget / (tile_k * dtype_bytes))), 32, 256)
if memory_bound && tile_side < 256: tile_m = tile_n = clamp(tile_side*2, 64, 256)
else tile_m = tile_n = tile_side
clamp tile_m to M, tile_n to N
warps_per_block = 8     # hardcoded for Hopper
shared_per_block_kb = (tile_m + tile_n) * tile_k * dtype / 1024
pipeline_stages = (shared<64KB:4, <96KB:3, else:2)
occupancy = memory_bound ? 0.85 : 0.92
roofline_efficiency = occupancy
predicted_tflops = peak * occupancy * min(1, ai/ridge)
return cfg
```

**`cipher_recipe_attention(seq_len, head_dim, num_heads, batch, epsilon, causal, hw)` -- L124-170:** Closed-form FAVOR+ feature count.

```
d_f = head_dim
d_req = ceil(d_f * log(d_f + 1) / epsilon^2)
D = round_up_pow2(d_req)
if D < head_dim: D = round_up_pow2(head_dim)
if D > 512: D = 512
favor_wins = (seq_len^2 > seq_len * D * 2)
num_features = favor_wins ? D : 0
feature_scaling = favor_wins ? 1/sqrt(D) : 1
error_bound = D > 0 ? 1/sqrt(D) * sqrt(log(1/0.01)) : 0
return cfg
```

**`cipher_recipe_chebyshev(nonlin, degree)` -- L214-331:** Switch over `nonlin`; for each (GELU, SILU, LAYERNORM, RMSNORM, SOFTMAX, GELU_TANH) populates `domain_lo/hi`, `max_error`, `degree`, and 9-11 pre-computed `coeffs[]` values. Default case sets `cfg.valid = false`. **The `degree` argument is IGNORED** -- the function overrides it from a per-nonlinearity hardcoded value. Calling with `degree=5` gets `degree=8` back.

**`cipher_chebyshev_eval(cfg, x)` -- L192-208:** Clenshaw recurrence:
```
if !cfg->valid: return x
span = domain_hi - domain_lo
xc = (2*x - (lo+hi)) / span        # map x to [-1, 1]
b_prev = b_curr = 0
for k from degree down to 1:
    b_next = 2*xc*b_curr - b_prev + cfg->coeffs[k]
    b_prev = b_curr; b_curr = b_next
return xc * b_curr - b_prev + cfg->coeffs[0]
```

**`cipher_registry_init(reg)` -- L346-484:** Seeds 32+ entries. Pattern: `e = {0}` zero-init, then assign fields, then `reg->entries[reg->count++] = e`. Contents:
- Entry 0: GEMM 4096x4096x4096 (Llama-3 70B square), arch=90, error=0.005, conf=0.96
- Entry 1: GEMM 4096x4096x128 (Llama-3 70B attn QK^T), arch=90, error=0.005, conf=0.955
- Entry 1 (duplicate index, comment): GEMM 4096x28672x8192 (Llama-3 70B FFN up), arch=90
- Entry 2: GEMM 4096x8192x28672 (Llama-3 70B FFN down)
- Entry 3: GEMM 43x128x128 (SOMA motor control), arch=80, error=0.001
- Entries 4-9: 6 more Llama-3 70B shapes (decode-qk, prefill-2k, seq8k-proj, etc.)
- Entries 10-15: RMSNorm Chebyshev shapes at `shape_hash = 4096 << i` for i=0..5
- Entries 16-21: 6 elementwise activations (gelu/silu) at `shape_hash = 0xAC000 + i`
- Entries 22-25: 4 HyperFlux entries at `shape_hash = 0xBE000 + i`, **arch=86** (Ampere only)
- Entries 26-31: 6 A100 versions of the Llama-3 shapes, arch=80
- Then 10 "shape-parametric" entries at the end (L457-479): `hash_shape(0, K, N)` -- i.e. M=0 baked into the hash. Names like `gemm-Kx4096-N4096`, `gemm-Kx14336-N4096`, `gemm-Kx8192-N28672`, etc.

**`cipher_registry_lookup(reg, op_class, shape_hash, hw_arch)` -- L486-500:** Linear scan, return first entry matching `(op_class, shape_hash)` and (`hw_arch == arch || entry->hw_arch == 0`).

**Branch determinants.**
- Registry seed entries 0-31 use `arch in {86, 90, 80}`; only the arch=90 entries match the H100 default. The arch=86 HyperFlux entries (Ampere) never match on H100 (`hw_arch=90`).
- Entries 32-41 use `arch=0` (any), so they match for any caller -- but the shape_hash is `hash_shape(0, K, N)`, with M=0, which the lookup never produces (M comes from runtime, always > 0).

### STATE
- No file-static state. All state lives in the `CipherRegistry* reg` argument owned by the dispatcher (`g_registry` in `cipher_dispatch.cpp:63`).
- `s_cheb_opportunity` static counter in `apply_recipe` is in `cipher_dispatch.cpp:310`, not here.

### CONCURRENCY
**Threading model:** registry init runs once at startup; lookups are read-only on hot path.

**Locks:** none. The registry is read-only after init; safe for concurrent read.

**Signal-safety:** yes for `cipher_registry_lookup` (pure read, no syscalls); no for `cipher_registry_init` (calls fprintf).

### DEPENDENCIES
**INBOUND:**
- `cipher_recipe_gemm` -- `cipher_dispatch.cpp:252` (top-level), `src/cipher_dispatch.cpp:210` (shadow).
- `cipher_recipe_chebyshev` -- `src/cipher_block_sub_kernel.cu:947` via dlsym.
- `cipher_registry_init` -- `cipher_dispatch.cpp:119, 352`.
- `cipher_registry_lookup` -- `cipher_dispatch.cpp:381, 398, 475` (top-level), `src/cipher_dispatch.cpp:506, 509, 568` (shadow).
- `cipher_registry_insert` -- no in-tree caller.

**OUTBOUND:** libm (sqrtf, ceilf, logf), libc (memset, strncpy, snprintf, fprintf).

**kmod ioctls:** none. **External libraries:** libc, libm.

### LOGIC-AS-CODED vs LOGIC-AS-DOCUMENTED

**DEEP_INSPECTION_REPORT D.3 already documents the registry-seed mismatch.** Quoting: *"The shape-parametric block at the end (entries 32-41) is keyed on `hash_shape(0, K, N)` -- i.e. with M=0. The lookup at `cipher_dispatch.cpp:153` calls `fnv_shape(m, n, k)` with the actual M from runtime. Since real workloads always have M >= 1, these 10 shape-parametric entries never match in practice."* Verified.

**Drift -- `cipher_dispatch.cpp:391, 398, 475` calls `cipher_registry_lookup` with `arch=90` (H100 Hopper)**. The registry has:
- 4 entries at `arch=86` (HyperFlux) -- never match on H100
- 6 entries at `arch=80` (A100) -- never match on H100
- ~22 entries at `arch=90` (H100) -- could match if shape_hash hits
- 10 entries at `arch=0` (any) but with M=0 in hash -- never match in practice

So on H100, the only theoretically-matching set is ~22 entries with very specific (M,N,K) tuples. **Real Mistral-7B shapes are 4096x14336x4096 / 14336x4096x4096 / etc. -- none of these are seeded explicitly with M=4096; the shape-parametric `(K=14336, N=4096)` entries have M=0 baked into the hash and don't match.** Verified: registry MISS at L480 is the empirically-observed path.

**Drift -- `cipher_chebyshev_eval` ignores the `degree` parameter passed in.** L218 hardcodes `cfg.degree = 8` initially, then each switch case overrides -- e.g. LAYERNORM sets to 10. The caller's `degree` argument is ignored. Drift: minor -- design intent is "always use the optimal degree per nonlinearity"; the arg is misleading.

**Drift -- GELU coefficient `coeffs[0] = 1.2312642265f` is computed for `[-4, 4]` domain (L225-237).** Header comment notes: *"max_abs=0.0125, max_rel(|f|>0.05)=23%"*. **The 23% relative error near zero crossings means this approximation is correct away from zero but loses several significant figures near x=0** -- fine for activations (GELU(0) is approximately 0), poor for general use. Drift: matches design intent; just be aware.

**Drift -- `cipher_recipe_attention_safe` returns true for `num_features == 0`** (L173-175). num_features=0 means "use exact attention", which is always safe -- but then the recipe isn't doing anything. Logical OK.

**Drift -- Comment at L356 calls the registry "32 day-one entries"**, but the actual count is more (32 fixed + 10 shape-parametric = 42 if `CIPHER_REGISTRY_MAX_ENTRIES >= 42`, which is 256 -- so all 42 fit). Plan section 3.0 line 98 says "32 hardcoded Llama-3-70B-scale shapes", which the DEEP_INSPECTION_REPORT D.3 already corrected.

### CONTRIBUTION TO SYSTEM
**LOAD-BEARING (in design) but STARVED (in practice).** The registry is supposed to be the substitute-decision substrate. Today, real workloads never hash to seeded entries, so `cipher_registry_lookup` returns NULL -> L480 PASS_THROUGH -> no substitution -> no measurable effect from this file's recipes on real workloads.

The Chebyshev recipes feed `src/cipher_block_sub_kernel.cu` via dlsym (L947) -- used by Stage-6 NVRTC substitution path for activations. The roofline GEMM recipe `cipher_recipe_gemm` is used cosmetically (for log messages in `cipher_dispatch.cpp:286-291`).

**Removing this file:** breaks dispatch (registry_lookup goes missing), breaks block_sub_kernel.cu (Chebyshev gone). But: removing the registry seed entries would have **no observable runtime effect** because no real workload matches them.

### FUSION POINTS

**Port-as-is -- registry needs reseeding for real shapes.**
- **Contract:** `cipher_registry_lookup(reg, op_class, fnv_shape(M,N,K), arch)` -> entry or NULL.
- **Honored on both sides?** Tree B has the actuator (`cipher_rt_matmul_dispatch.c`) but no registry of "which shapes have a known recipe." Tree B implicitly decides "any shape > B>=8 -> Marlin actuator engages" -- no registry consultation.
- **Action: PORT-AS-IS + RESEED.** Port the registry mechanism unchanged but seed it with shapes that match Tree B's actuator competence envelope (Mistral-7B FFN sizes, attention QK^T sizes). Severity: minor (additive; Tree B's current actuator routing is independent and can run in parallel).

**`cipher_recipe_chebyshev` dlsym path from block_sub_kernel.cu.**
- **Contract:** `dlsym(RTLD_DEFAULT, "cipher_recipe_chebyshev")` -> function pointer with the C ABI.
- **Honored on both sides?** Tree B does NOT have `cipher_block_sub_kernel.cu` in its build (per the cipher_rt_phase4/ file list). The dlsym caller goes away in v1.
- **Action: PORT-AS-IS the function** (cheap); the dlsym caller is deferred.

**Chebyshev evaluation: pure CPU function.**
- **Contract:** Standalone closed-form. No external state.
- **Action: PORT-AS-IS.** Severity: none.

**Hardware profile coupling (CipherHwProfile).**
- **Contract:** GEMM recipe and attention recipe both take an `hw` argument. Tree B has hardware queries scattered (`cipher_rt_volt.c` reads NVML, etc.). For v1 port, hardcode H100 profile via `CIPHER_H100_PROFILE` macro (header L63-72) -- Tree B's deployment target is H100.
- **Action: PORT-AS-IS with hardcoded H100 profile** until v1.5 hw-detection. Severity: minor.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_sense.cpp

### PURPOSE
From file header (L1-15) and `include/cipher_sense.h:1-9`: *"Op 13 SENSE -- Session classification (Stage 1). Reads ring entries observed by Stage 1, infers session boundaries from inter-event idle gaps, fingerprints sessions by the first 8 distinct GEMM shape proxies, and re-classifies session_type periodically."* Classifies sessions into HUMAN_INTERACTIVE / AGENT_AUTONOMOUS / BATCH_BACKGROUND / UNKNOWN using only ring-buffer timing and shape sequences. Default OFF (`CIPHER_SENSE=on` to enable).

### PUBLIC SURFACE
- `extern "C" int cipher_sense_init(void)` -- L154-169.
- `extern "C" void cipher_sense_observe(const CipherRingEntry* ev)` -- L171-273.
- `extern "C" CipherSessionType cipher_sense_get_type(uint64_t fingerprint)` -- L275-286.
- `extern "C" uint64_t cipher_sense_current_session(void)` -- L288-290.
- `extern "C" unsigned cipher_sense_session_count(void)` -- L292-294.
- `extern "C" void cipher_sense_report(void)` -- L296-340.

### CONTROL FLOW

**`cipher_sense_observe(ev)` -- pseudo-code (L171-273):**

```
if !g_sense_enabled.load(relaxed): return     # cheap exit when disabled
if !ev: return
now = ev->timestamp_ns
if now == 0:
    clock_gettime(CLOCK_MONOTONIC_RAW); now = ns
prev = g_last_ts_ns.exchange(now, relaxed)
slot = g_current_slot.load(relaxed)
start_new = (slot < 0) || (prev != 0 && now > prev && (now - prev) > 200ms)
if start_new:
    seed = fnv1a64(&now, 8, 0xcbf...^func_ptr_hash)
    slot = allocate_slot(seed)                # linear CAS scan
    if slot < 0: return                       # table full
    g_current_slot.store(slot, relaxed)
    g_current_fingerprint.store(seed, relaxed)
    g_sessions[slot].first_seq = ev->sequence
    g_sessions[slot].first_ts_ns = now
s = g_sessions[slot]
s.last_seq = ev->sequence
s.last_ts_ns = now
# Inter-event timing
if !start_new && prev != 0 && now > prev:
    gap = now - prev
    update s.inter_event_min/max/sum, n_inter_event++
# GEMM-class signals
if ev->kernel_class == 0 (GEMM):
    unpack_mkn(ev->params_hash, M, K, N)      # tag 0xC top nibble
    batch = (M && N) ? min(M,N) : (M | N)
    is_decode  = (batch > 0 && batch <= 8)
    is_prefill = (batch >= 256)
    if is_decode:
        s.decode_count++
        s.decodes_since_classify++
    if is_prefill:
        s.prefill_count++
        s.decodes_since_classify++
        if s.last_prefill_ts_ns > 0:
            update s.prefill_gap_sum/sum_sq, prefill_arrival_n++
        s.last_prefill_ts_ns = now
    # Track up to 8 distinct shape proxies; lock fingerprint when reached
    if s.n_shapes_seen < 8:
        sp = ev->params_hash ?: ev->func_ptr_hash
        if sp != 0 && not in s.first_8_shapes:
            s.first_8_shapes[n++] = sp
            if n_shapes_seen == 8:
                fp = fnv1a64(s.first_8_shapes, 64, seed)
                s.fingerprint.store(fp, release)
                g_current_fingerprint.store(fp, relaxed)
# Re-classify every 10 decodes
if s.decodes_since_classify >= 10:
    s.decodes_since_classify = 0
    classify(s)
```

**`classify(s)` -- L129-150:**

```
is_agent = (decode_count > 500)
is_batch = (prefill_count >= 3 && decode_count <= 200)
is_human = (prefill_count in [1,2] && decode_count in [5, 500])
t = is_agent ? AGENT : (is_batch ? BATCH : (is_human ? HUMAN : UNKNOWN))
s.session_type.store(t, release)
```

**`allocate_slot(seed)` -- L100-127:** Linear CAS scan over 1024 slots; first slot with `fingerprint == 0` is claimed via `compare_exchange_strong`. On success, initialize fields and increment `g_session_count`. On full table, return -1 silently.

**Branch determinants.**
- `g_sense_enabled` -- set in `cipher_sense_init` based on `CIPHER_SENSE=on|1|ON`. Default OFF.
- `IDLE_THRESHOLD_NS = 200 ms` for session boundary detection.
- `RECLASS_DECODE_EVERY = 10` -- reclassify every 10 decode events.

### STATE
- `alignas(64) SessionState g_sessions[1024]` (L65) -- ~32 KB per session (atomic counters + 8 shape slots + timings); table is 32 MB.
- `std::atomic<int> g_sense_enabled, g_sense_initialized` (L67-68).
- `std::atomic<unsigned> g_session_count`, `g_current_fingerprint`, `g_current_slot`, `g_last_ts_ns` (L69-72).

### CONCURRENCY
**Threading model:** designed for Stage 1 background thread (per file header), but per `cipher_10ops_impl.cpp:509`, it's called from the ring drain. Written defensively for concurrency (CAS, atomic accessors), but in practice "only one Stage 1 thread today" (L99 comment).

**Locks:** none. CAS on slot allocation; atomic loads/stores for fingerprint/type.

**Race semantics:** `g_current_slot.load(relaxed)` + `g_last_ts_ns.exchange(now, relaxed)` is racy across multiple Stage-1 threads (none exist) -- boundary detection becomes approximate under concurrency.

**Signal-safety:** uses `clock_gettime(CLOCK_MONOTONIC_RAW)` (signal-safe), `getenv` (not signal-safe -- only in init). `fprintf` in init and report. Hot path is signal-safe.

### DEPENDENCIES
**INBOUND:**
- `cipher_sense_observe` -- `src/cipher_10ops_impl.cpp:509`.
- `cipher_sense_init` -- typically a constructor or explicit init call; **no in-tree caller found by grep.** Either `cipher_sense_init` is called by an external test or `g_sense_initialized` defaults to off and the observer is a no-op until enabled.
- `cipher_sense_report`, `_get_type`, `_current_session`, `_session_count` -- used by Op-13 reporting telemetry (likely in `cipher_telemetry.cpp` or test harness).

**OUTBOUND:** libc (clock_gettime, fnv1a, strcmp, getenv, fprintf, fopen).

**kmod ioctls:** none. **External libraries:** libc, libpthread (atomics), no libcuda.

### LOGIC-AS-CODED vs LOGIC-AS-DOCUMENTED

**Plan section 3.0 (L318):** *"SENSE | classifier | `cipher_sense.cpp:171-273` | not present | WORKING in may13 | universal (session classifier)"* -- **Verified.** Lines 171-273 cover `cipher_sense_observe`.

**Drift -- header comment "v1 limitation still standing: no stream_id in ring entries -- concurrent sessions on one GPU collapse into one logical session"** (L13-15). Read the `CipherRingEntry` struct at `include/cipher_10ops.h:23-37`: no stream_id field. Confirmed. The session classifier can't distinguish concurrent vLLM tenants. Future work.

**Drift -- `unpack_mkn` requires tag 0xC in top nibble of params_hash.** Per `cipher_sense.cpp:91-96`: `if ((ph >> 60) != 0xC) { M=K=N=0; return; }`. The tag must be set by whoever wrote `params_hash` into the ring. This is the "Op-14 geometry plumb-through" mentioned in the file header. **Honored?** Per `src/cipher_intercept_cudart.cpp` (Op-14 ring writes), check whether they set tag 0xC. *Not verified in this audit pass.* If the tag isn't set, M/K/N are all zero, batch=0, neither is_decode nor is_prefill fires -- session never classifies beyond UNKNOWN.

**Drift -- `cipher_sense_init` never called in-tree.** No constructor-priority attribute (unlike `cipher_param_recovery_autoinit` at priority 113). If no caller invokes init, `g_sense_enabled` remains 0 and the observer is always-off. Severity: moderate -- if Tree B port wires this without an init call, SENSE never fires.

**Drift -- comment vs code in `classify()`.** L130-133 comment says: *"AGENT_AUTONOMOUS: long sustained decode chain (>500 decode steps)"*. L134 code: `bool is_agent = (decode_count > 500)`. Match. Similar for BATCH and HUMAN. No drift here; just verification.

### CONTRIBUTION TO SYSTEM
**SUPPORTING.** SENSE feeds downstream ops SHIELD, FAIRNESS, LOOP (per `CIPHER_REENGINEERING_PLAN.md:372`) by tagging the current session with `HUMAN/AGENT/BATCH`. Today, none of those downstream consumers run in production (Tree B has no SHIELD/FAIRNESS/LOOP ports).

**Removing this file:** SHIELD/FAIRNESS/LOOP lose their session-band signal -- but those ops aren't deployed in Tree B yet.

For v1 fusion: SENSE is a classifier that can run *in parallel* with the actuator-substitution path. It contributes nothing to compute/perf -- but it's foundational for multi-tenant fairness (CP 5.4 -> 5.5 work).

### FUSION POINTS

**Port-as-needed for v1.5.**
- **Contract:** Ring-buffer drain hook + session-table reader. Standalone classifier; no actuator coupling.
- **Honored on both sides?** Tree B has no ring buffer. The actuator substrates already see every cuBLAS/SDPA call directly.
- **Action: REFACTOR-REQUIRED.** Either port the ring buffer (large dep -- `cipher_10ops.h` ring infrastructure), or rewrite `cipher_sense_observe` to take a synthetic `CipherRingEntry` constructed at the actuator hook. Severity: requires-mitigation.

**Init call site.**
- **Contract:** `cipher_sense_init` must be called once to enable.
- **Action: REFACTOR-REQUIRED.** Wire into the Tree B `InitializeInjection` callback chain (Plan section 7 W2). Severity: minor.

**`params_hash` tag-0xC packing convention.**
- **Contract:** Whoever writes ring entries must pack `(M, K, N)` into 60 bits below the tag. In Tree B, the new pre-launch hook needs to do this packing -- or `cipher_sense_observe` needs a new signature.
- **Action: REFACTOR-REQUIRED (small).** Change `cipher_sense_observe` to take `(M, K, N)` explicitly. Severity: minor.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_predict.cpp

### PURPOSE
From file header (L1-10): *"Op 17 PREDICT -- v1 observer. Per-shape dispatch counter + short-gap reuse counter. At report time, emits hot shapes (count >= HOT_COUNT AND short_gap/count >= 0.5) as preload candidates. No CUDA, no actuation."* Also includes a **Stage-3 pointer-reuse tracker** (PREDICT instrumentation upgrade) that promotes hot pointers into the persistence engine via dlsym. Default OFF (`CIPHER_PREDICT=on`).

### PUBLIC SURFACE
- `int cipher_predict_init(void)` -- L143-158.
- `void cipher_predict_observe(const CipherRingEntry* ev)` -- L160-179.
- `void cipher_predict_observe_ptr(void* ptr, unsigned long bytes)` -- L193-245.
- `unsigned cipher_predict_shape_count(void)` -- L181-183.
- `unsigned cipher_predict_candidate_count(void)` -- L185-187.
- `unsigned cipher_predict_hot_ptr_count(void)` -- L189-191.
- `void cipher_predict_report(void)` -- L259-333.

### CONTROL FLOW

**`cipher_predict_observe(ev)` -- pseudo-code (L160-179):**

```
if !g_enabled.load(relaxed): return
if !ev: return
key = ev->params_hash ?: ev->func_ptr_hash
if key == 0: return
t = ev->timestamp_ns ?: now_ns()
idx = probe_or_insert(key, t)
if idx < 0: idx = evict_and_install(key, t)
e = g_table[idx]
if e.count > 0 && t > e.last_ts_ns && (t - e.last_ts_ns) < 10ms:
    e.short_gap_count++
e.count++
e.last_ts_ns = t
```

**`probe_or_insert(key, t)` -- L94-116:** Open-address linear probe (mix64 -> 4096 slots). CAS to claim empty slot. Returns slot or -1 on full table.

**`evict_and_install(key, t)` -- L119-139:** Min-count eviction with oldest-last_ts_ns tiebreak. Scans all 4096 slots.

**`cipher_predict_observe_ptr(ptr, bytes)` -- L193-245:**

```
if !enabled || !ptr || bytes < 64KB: return
key = (uint64)ptr; h = mix64(key); t = now_ns()
# Probe/insert (same pattern as shape table but on g_ptr_table[1024])
idx = ...
if idx < 0: return                            # full table -- silent drop
e = g_ptr_table[idx]
if bytes > e.bytes: e.bytes = bytes
e.count++; e.last_ts_ns = t
# Promote on threshold
if !e.registered && e.count >= 100:
    e.registered = 1
    g_ptr_hot_count++
    resolve_persist_register()                # one-time dlsym
    fn = g_persist_register
    if fn:
        dt_s = max(0.001, (last - first) / 1e9)
        score = count / dt_s
        fn(ptr, bytes, score)                 # cipher_persist_engine_register
```

**`cipher_predict_report()` -- L259-333:** Scans `g_table[]`, filters by `count >= HOT_COUNT (200)` AND `short_gap_count/count >= 0.5`. Insertion-sorts top 32 by count. Writes JSON to `/tmp/cipher_predict_report.json` with candidates + global stats.

**Branch determinants.**
- `g_enabled` from `CIPHER_PREDICT=on` env.
- `HOT_COUNT = 200`, `HOT_GAP_NS = 10ms`, `HOT_RATIO = 0.5`, `HOT_PTR_COUNT = 100`, `HOT_PTR_MIN_BYTES = 64 KB`.

### STATE
- `alignas(64) ShapeEntry g_table[4096]` (L42) -- ~64 bytes per entry, ~256 KB.
- `alignas(64) PtrEntry g_ptr_table[1024]` (L59) -- ~32 bytes per entry, ~32 KB.
- Atomic counters: `g_enabled, g_initialized, g_shape_count, g_eviction_count, g_candidate_count, g_ptr_seen_count, g_ptr_hot_count`.
- `g_persist_register` (atomic function pointer) + `g_persist_resolved` (atomic flag).

### CONCURRENCY
**Threading model:** ring-drain caller for `_observe`, intercept-hook caller for `_observe_ptr`.

**Locks:** none. Slot claim via `compare_exchange_strong`. Eviction is a non-atomic scan -- under concurrent eviction, two threads can evict different slots for the same incoming key; the later writer wins.

**Race semantics:** `e.count++` is non-atomic (L177, L228). Acceptable -- counts are statistical.

**Signal-safety:** hot path mostly signal-safe; report path is not (fopen/fprintf).

### DEPENDENCIES
**INBOUND:**
- `cipher_predict_observe` -- `src/cipher_10ops_impl.cpp:542`.
- `cipher_predict_observe_ptr` -- `src/cipher_intercept_cudart.cpp:421` (dlsym in cuBLAS shim).
- `cipher_predict_init` -- no in-tree caller found.

**OUTBOUND:**
- `clock_gettime`, `dlsym(RTLD_DEFAULT, "cipher_persist_engine_register")` -- for hot-pointer promotion into the persist engine (Stage 3).
- libc primitives.

**kmod ioctls:** none. **External libraries:** libdl (dlsym), libc.

### LOGIC-AS-CODED vs LOGIC-AS-DOCUMENTED

**Plan section 3.0 (L322):** *"PREDICT | actuator | `cipher_predict.cpp:160-245` | not present | WORKING -- hot-pointer promotion into persist engine | memory-bound"* -- **Verified.**

**Drift -- "v1 observer ... No CUDA, no actuation"** (file header L1-3) **vs** *the Stage-3 pointer promotion calls `cipher_persist_engine_register`* (L237-242). The header says no actuation; the implementation does actuate (via dlsym promotion). The PLAN's "WORKING -- hot-pointer promotion into persist engine" is the truth; the file header is stale. Severity: minor -- documentation drift only.

**Drift -- `HOT_PTR_MIN_BYTES = 64 KB` filters out small tensors.** Real KV-cache regions are 1 MB+ per layer; OK. Activation slabs are similar. Severity: none -- intentional.

**Drift -- eviction policy may starve "freshly hot" shapes.** If a brand-new key arrives when the table is full, `evict_and_install` finds the minimum-count slot, which is often a slot that was *just inserted* (count=1) by another fresh key. Effectively, when the working set exceeds 4096, eviction thrashes. Severity: minor -- only relevant under pathological shape diversity.

### CONTRIBUTION TO SYSTEM
**SUPPORTING -- but the pointer-promotion path is functionally on the L2-persistence hot path.** In may13, `cipher_predict_observe_ptr` is called from the cuBLAS shim with each (A, B, C) pointer; when one crosses HOT_PTR_COUNT=100 reuses, it's auto-registered with `cipher_persist_engine_register`, which sets up the L2 persistence window for that pointer (per CLAUDE.md Stage 3 notes: "M4 run = 48,848 window_hits / 48,797 launches in 10 s").

Without PREDICT's pointer-promotion, the L2 persist engine never gets its inputs; the 48K window_hits/sec drops to zero.

**Removing the shape-observation path (the `_observe` ring drain):** the JSON report becomes empty; no actuation impact.

### FUSION POINTS

**Pointer-promotion path is a v1-relevant actuator hook.**
- **Contract:** `cipher_predict_observe_ptr(ptr, bytes)` from cuBLAS shim -> `cipher_persist_engine_register(ptr, bytes, score)` via dlsym.
- **Honored on both sides?** Tree B has its own KV-persistence path (`cipher_rt_kv_alloc.c`) using a different mechanism (cuMemExport, not cudaAccessPolicyWindow). The PREDICT-driven L2-persist path is a may13-only substrate.
- **Action: DEFER for v1** (Tree B's KV path supersedes). Severity: minor.

**Ring-observe path.**
- **Contract:** Stage-1 ring drain calls `_observe(ev)`.
- **Action: SHIM-REQUIRED** (synthesize CipherRingEntry from actuator hook) OR DEFER. Severity: minor.

---

## Section /home/ubuntu/cipher-may13-evidence/src/cipher_determinism.cpp

### PURPOSE
From file header (L1-5) and `include/cipher_determinism.h:1-9`: *"Op 21 DETERMINISM -- v1 dispatch-sequence fingerprint. Mixes the ordered params_hash stream into a 64-bit accumulator. No CUDA, no allocation."* For identical deterministic workloads (same seed, shapes, call order), the reported FNV-mix hash matches across reruns -- a fingerprint of execution. Default OFF (`CIPHER_DETERMINISM=on`).

### PUBLIC SURFACE
- `int cipher_determinism_init(void)` -- L33-48.
- `void cipher_determinism_observe(const CipherRingEntry* ev)` -- L50-63.
- `uint64_t cipher_determinism_hash(void)` -- L65-67.
- `uint64_t cipher_determinism_count(void)` -- L69-71.
- `void cipher_determinism_report(void)` -- L73-88.

### CONTROL FLOW

**`cipher_determinism_observe(ev)` -- pseudo-code (L50-63):**

```
if !g_enabled.load(relaxed): return
if !ev: return
ph = ev->params_hash
if ph == 0: return                          # ASLR-stable only
m = mix64(ph)                               # splitmix64
do:
    prev = g_hash.load(relaxed)
    next = (prev XOR m) * FNV_PRIME         # 0x100000001b3
while !g_hash.compare_exchange_weak(prev, next, release, relaxed)
g_count.fetch_add(1, relaxed)
```

This is a lock-free CAS-loop accumulator: each observation mixes a per-shape `mix64(params_hash)` into the running 64-bit hash via XOR+multiply.

**`cipher_determinism_report()`:** Writes `{dispatch_hash, dispatch_count}` JSON to `/tmp/cipher_determinism_report.json`.

**Branch determinants.**
- `g_enabled` set by `CIPHER_DETERMINISM` env.
- `ev->params_hash == 0` skip -- only ASLR-stable hashes contribute (avoids non-reproducibility from address-derived hashes).

### STATE
- `std::atomic<int> g_enabled, g_initialized` (L19-20).
- `std::atomic<uint64_t> g_hash` (L21) -- FNV-offset seeded at 0xcbf29ce484222325.
- `std::atomic<uint64_t> g_count` (L22).

### CONCURRENCY
**Threading model:** ring-drain caller.

**Locks:** none. The hash update is a CAS-loop; safe across concurrent observers.

**Race semantics:** the CAS loop is order-sensitive -- observers landing in a different order produce different hashes. *This is by design* -- the goal is to detect order changes.

**Signal-safety:** hot path is signal-safe. Init and report are not.

### DEPENDENCIES
**INBOUND:**
- `cipher_determinism_observe` -- `src/cipher_10ops_impl.cpp:548`.
- `cipher_determinism_init`, `_hash`, `_count`, `_report` -- by test harness (`tests/test_*`), not in-tree dispatch.

**OUTBOUND:** libc (`getenv`, `strcmp`, `fopen`, `fprintf`, `fclose`).

**kmod ioctls:** none. **External libraries:** libc.

### LOGIC-AS-CODED vs LOGIC-AS-DOCUMENTED

**Plan section 3.0 (L326):** *"DETERMINISM | observability | `cipher_determinism.cpp:50-63` | not present | WORKING -- FIRES (dispatch_count=36450 in stress2) | universal"* -- **Verified.** Lines 50-63 cover `cipher_determinism_observe` exactly. The "dispatch_count=36450 in stress2" claim refers to a stress test in `cipher-may13-evidence/stress2/`.

**Drift -- the FNV-mix-with-prior-state hash is order-dependent BUT also count-dependent.** A workload that runs N times the same kernels in different orders produces different hashes -- correct. A workload that runs N+1 kernels produces a different hash from N -- also correct. So this is a "exact replay" fingerprint, not a "same-set-of-kernels" fingerprint. The header doc says "identical deterministic workloads" -- implies exact replay. Match.

**Drift -- `ph == 0` skip.** The ring-write code in `cipher_10ops_impl.cpp` (not audited here) determines whether `params_hash` is ASLR-stable or address-derived. If consistently 0 for ASLR-unstable kernels, this is fine. Severity: depends on Op-14 plumb-through correctness; not verified.

### CONTRIBUTION TO SYSTEM
**SUPPORTING.** Pure observability. Used for regression testing and reproducibility verification. Removing this file: no actuation change.

For v1 fusion: a small, self-contained observer with zero coupling. Ports cleanly.

### FUSION POINTS

**Port-as-is, deferrable for v1.5.**
- **Contract:** `_observe(ev)` from any source that provides a CipherRingEntry with a stable params_hash.
- **Honored on both sides?** Tree B has no ring buffer.
- **Action: REFACTOR-REQUIRED (tiny).** Change signature to take `uint64_t params_hash` directly. Severity: minor.

---

# Cross-File Synthesis

## Hot-path call graph (verified in this audit)

```
LD_PRELOAD libcipher_hook.so::cuLaunchKernel intercept
       (src/cipher_intercept_cudart.cpp:1483)
       v
cipher_dispatch(CipherKernelDesc*)                        [TOP-LEVEL cipher_dispatch.cpp:338]
   |
   +-> classify_launch(...)                            [cipher_classify.hpp:227]
   |     +-> detail::fingerprint(g)                    [cipher_classify.hpp:94]
   |
   +-> cipher_liquid_record_op(&g_cipher.liquid, op)   [cipher_lnn.cpp]
   |
   +-> infer_layer_context(desc, &layer, &back)        [cipher_dispatch.cpp:81]
   |
   +-> cipher_oracle_decide(&g_oracle, &oq)            [cipher_oracle.cpp:293]
   |     +-> topo_detect_inference(state, oc)          [cipher_oracle.cpp:30]
   |     +-> cipher_struct_lookup(&sctx)               [cipher_structural_lookup.cpp:194]
   |           +-> name_cache_lookup
   |           +-> g_name_rules walk
   |
   +-> gemm_shape_hash(desc, &m, &n, &k)               [cipher_dispatch.cpp:143]
   |     +-> weak cipher_tls_get_gemm_shape            [src/cipher_intercept_cudart.cpp]
   |
   +-> cipher_registry_lookup(&g_registry, oc, h, 90)  [cipher_recipes.cpp:486]
   |
   +-> apply_recipe(entry, desc, layer)                [cipher_dispatch.cpp:242]
   |     +-> cipher_recipe_gemm(M,N,K,&hw)             [cipher_recipes.cpp:45]
   |     +-> weak cipher_get_edmd_pipeline(0)          [src/cipher_edmd.cpp]
   |     +-> weak cipher_tls_relaunch()                [src/cipher_intercept_cudart.cpp]
   |     +-> edmd_live_post_relaunch_hook              [cipher_dispatch.cpp:212]
   |           +-> weak cipher_edmd_live_collect       [src/cipher_edmd_live.cpp:415]
   |
   +-> cipher_oracle_record_substitution(...)          [cipher_oracle.cpp:390]
   +-> cipher_oracle_bill_gemm/_nongemm                [cipher_oracle.cpp:489/504]
```

The Stage-1 observers (SENSE/PREDICT/DETERMINISM) live on a **parallel** drain path, NOT on the hot path:

```
LD_PRELOAD libcipher_hook.so::cuLaunchKernel intercept
   writes a CipherRingEntry to the lock-free ring buffer
   v
cipher_10ops_impl.cpp ring-drain thread
   +-> cipher_sense_observe(ev)                 [cipher_sense.cpp:171]
   +-> cipher_predict_observe(ev)               [cipher_predict.cpp:160]
   +-> cipher_determinism_observe(ev)           [cipher_determinism.cpp:50]

Separately, in-shim:
cipher_intercept_cudart.cpp::cublasGemmEx shim
   +-> cipher_predict_observe_ptr(A,B,C, bytes) [cipher_predict.cpp:193]
         +-> cipher_persist_engine_register  (via dlsym)

cipher_intercept_cudart.cpp::__cudaRegisterFunction shim
   +-> cipher_param_register_fatbin              [cipher_param_recovery.cpp:548]
   +-> cipher_param_register_function            [cipher_param_recovery.cpp:565]
   +-> cipher_param_ingest_image (env-gated)     [cipher_param_recovery.cpp:468]

cipher_intercept_cudart.cpp::cuLaunchKernel/Ex shim
   +-> cipher_kt_observe(fn, grid, block, smem)  [cipher_kernel_table.cpp:222]
```

## The empirically-observed runtime path

Based on the cross-file analysis:

1. Every kernel launch calls `cipher_dispatch`.
2. `classify_launch` returns one of 7 OpClass values with confidence 40 (ITERATIVE_CUSTOM) or 85 (else).
3. The main path's L444 fast-exit kills every ITERATIVE_CUSTOM launch (confidence 40 < min_confidence default 60).
4. `cipher_oracle_decide` runs Gate 1 (phase). Topological auto-detect flips to CONVERGENCE after ~64-200 kernels.
5. Gate 5 (kernel-name structural lookup) is INACTIVE because `oq.kernel_name = NULL` at all call sites.
6. Gate 6 (op-class default) is the deciding factor: ATTENTION/ITERATIVE_CUSTOM -> DENY; GEMM/CONV/ELEM/REDUCE/XPOSE -> PERMIT.
7. For permitted GEMM: `gemm_shape_hash` -> `cipher_registry_lookup(arch=90)`. The 32+ seed entries (~22 at arch=90, ~10 at arch=0-but-M=0-baked) almost never match real Mistral/Llama shapes.
8. Therefore: **L480 PASS_THROUGH is the dominant path. The Koopman lane (`apply_recipe` case 0 Koopman branch) and the relaunch lane (`apply_recipe` case 0 fallback) never fire on real workloads.**
9. Billing accumulates: `billing_gemm_passthrough` counts up; `billing_gemm_substituted` stays at zero.

This is the central engineering reality.

## v1 fusion summary

| File | Action | Severity | Rationale |
|---|---|---|---|
| `cipher_dispatch.cpp` (top-level) | REFACTOR | blocking | Tree B has no hot-path entry; new substrate file needed |
| `cipher_oracle.cpp` (top-level) | PORT-AS-IS + add call site | minor | Standalone; null-pass `liquid` |
| `include/cipher_classify.hpp` | PORT-AS-IS | none | Header-only library |
| `src/cipher_kernel_table.cpp` | PORT-AS-IS + drop hook copy | minor | Resolve dual-build |
| `src/cipher_structural_lookup.cpp` | PORT-AS-IS | none | Pure logic |
| `src/cipher_param_recovery.cpp` | DEFER to v1.5 | minor | Prerequisite to EDMD only |
| `src/cipher_recipes.cpp` | PORT-AS-IS + reseed registry | minor | Reseed to match Tree B actuator envelope |
| `src/cipher_sense.cpp` | REFACTOR | requires-mitigation | Needs new call surface (no ring in Tree B) |
| `src/cipher_predict.cpp` | DEFER | minor | Tree B's KV path supersedes |
| `src/cipher_determinism.cpp` | REFACTOR (tiny) | minor | Change signature to take `params_hash` |

## Cross-file struct collision (must fix during port)

**`CipherKernelEntry`** is defined in BOTH:
- `include/cipher_kernel_table.h:53-64` (fn_handle, name, category, ...)
- `include/cipher_param_recovery.h:30-38` (host_fun, device_fun, device_name, params, ...)

These structs have **different fields under the same name** and cannot co-exist in one translation unit. Fusion port must rename one (e.g., `CipherKtEntry` and `CipherParamEntry`).

## The load-bearing fact (re-verified)

`cipher_dispatch.cpp:480` is `return CIPHER_PASS_THROUGH` triggered when `cipher_registry_lookup` returns NULL. Today, every real-workload kernel exits here. The PLAN's Goal-4 = "wire the EDMD pipeline so the registry grows from observation" is the path forward; it is deferred to v2. **Until then, the classifier brain has zero actuation effect on real workloads.** Telemetry and billing fire, but the substitute lane stays dark.

The honest engineering posture for v1: port the classifier brain as substrate, wire it into the actuator-registry pattern, ship MFU measurement at CP 5.5 (per Plan section 5), defer Koopman discovery to v2.
