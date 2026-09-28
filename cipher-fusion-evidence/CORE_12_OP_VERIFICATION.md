# Core 12 Op Verification

**Date:** 2026-05-20
**Scope:** Per-op verification of every Core 12 op (plus ARBITRATE = #13), beyond the tier-level mapping in CIPHER_REENGINEERING_PLAN.md v1.2.1 §2.1.
**Mode:** Verification only. No code changes. Every claim cites the file:line in the source tree or in a Wave audit document.
**Source documents:** v1.2.1 plan §2.1 / §2.2; Wave 1 (CLASSIFIER), Wave 2 (ACTUATORS), Wave 3 (KMOD), Wave 4 (OBSERVERS), Wave 5 (FUSION_PLAN); the actual source files in `cipher-may13-evidence/`, `cipher_rt_phase4/`, `cipher_kmod/`; the may13 build-state CLAUDE.md.

---

## Op-name aliasing surfaced up front

The brief lists 13 op names. The actual op set is 12, because two pairs of names refer to the same op:

| Brief name | Plan §2.1 name | Status |
|---|---|---|
| COMMIT (#5) and ORCHESTRATE (#4) | "COMMIT / ORCHESTRATE" (Op #4) | Same op. ORCHESTRATE was the legacy may13 spelling (also surfaced as a print label at `cipher_10ops_impl.cpp:972` and as a hint sink at `cipher_straggler.h:29`); COMMIT is the unified-runtime spelling. One implementation, two names. |
| GENERATE (#6) | "SAMPLE / GENERATE" (Op #5) | Same op. Plan calls it SAMPLE because the only thing it does in production today is collect tensor norm samples for the EDMD pipeline. GENERATE is the legacy may13 spelling (mentioned in CLAUDE.md's 12-op list); SAMPLE is the v1.2.1 unified-runtime spelling. |

Each appears once below, with both names listed.

---

## 1. CLASSIFY

### A. Current implementation

| Field | Value |
|---|---|
| Definition file | `cipher-may13-evidence/include/cipher_classify.hpp:94-224` |
| Production caller (when ported) | n/a in deployed runtime today |
| Tree | may13 only |
| LOC of the classifier function `fingerprint()` | ~95 lines (L99-180 ish) |
| Public symbols | `cipher::classify_launch()`, `cipher::classify()`, `cipher::detail::fingerprint()`, `cipher::OpClass` enum |
| Hot-path budget | "~160 ns cache-hit" per the file header (Wave 1 measured this honestly; Wave 5 LP-14 rejected the prior 12 ns claim) |

### B. What it actually does

Reads `(fn, gx, gy, gz, bx, by, bz, shared_bytes)` from the cuLaunchKernel arguments. Applies a hand-tuned geometry-fingerprint decision tree (`detail::fingerprint`, `cipher_classify.hpp:103-181`) that maps each launch to one of seven kernel classes: GEMM, ATTENTION, ELEMENTWISE, MEMCPY_TRANSPOSE, REDUCTION, CONVOLUTION, ITERATIVE_CUSTOM. The function pointer is hashed into a 512-slot lock-free cache (`detail::cache_slot`, L195-208); on cache hit (the common case) the result is a relaxed atomic load plus a comparison, approximately 0.65 ns of read; on miss the fingerprint runs and the result is written back. Confidence is binary: 85 for any non-ITERATIVE class, 40 for ITERATIVE_CUSTOM (the only uncertain class). This binary confidence is the load-bearing detail that Wave 5 LP-11 calls out: with ORACLE's default `min_confidence = 60`, ITERATIVE_CUSTOM kernels exit early at `cipher_dispatch.cpp:444`, which is by-design lossless against any substitution.

Inputs consumed: kernel descriptor (fn pointer + grid + block + shared bytes). Outputs produced: `ClassifyResult { OpClass op; uint8_t confidence; bool from_cache; }`. Hot path: synchronous, per launch, in-thread.

### C. Where it fires in production today

**Not as a CLASSIFY op.** The substantive classifier logic is not ported into `cipher_rt_phase4`. The only mention of "CLASSIFY" in the deployed runtime is a comment in `cipher_rt_phase4/cipher_cupti.c:166` ("T4.2.4d diagnostic: classify the stream this launch is on") and that comment is about classifying the CUDA stream identity (null / green / other-green), not the kernel class.

In may13, CLASSIFY fires from `cipher_dispatch.cpp:343` (the classify-only branch when runtime is not initialized) and `cipher_dispatch.cpp:430` (the main spine in the post-init path). Both branches call `cipher::classify_launch(...)`.

### D. Integration plan

**Port from may13 in Week 1.** New file `cipher_rt_phase4/cipher_rt_classify.h` mirrors the header. New substrate file `cipher_rt_phase4/cipher_rt_classify_substrate.cpp` implements the priority-ordered classifier registry per plan v1.2.1 §4.0 (4-value enum HANDLED/PASSTHROUGH/REDIRECTED/ERROR, single-arg `maybe_handle`, snapshot-under-lock, break-on-ERROR). The hot-path call site is in `cipher_inject.c`'s GOT-patched cuLaunchKernel intercept (added in Week 2).

Fusion contract with adjacent ops:
- CLASSIFY produces `ClassifyResult` → feeds ORACLE as `CipherOracleQuery.op_class` and `.confidence`.
- The cache write-back is shared TLS, read by SUBSTITUTE and observers via Mode 3 TLS per plan §4.4.
- Wave 5 contracts touched: Ca.6 (classify_launch header port), Ca.7 (classifier observer on matmul substrate), Ca.8 (classifier observer on attn substrate), Ca.11 (header port), Ca.12 (kernel_name plumb into structural_lookup).

### E. Wave audit findings

- **Wave 1** (CLASSIFIER) is the primary deep-dive. CLASSIFY appears in Wave 1 throughout the control-flow analysis: the fingerprint switch is exhaustively walked, the cache slot semantics are verified, and the binary-confidence finding is what produced Wave 5 LP-11. Wave 1 also flagged F1-STRUCT-COLLISION (LP-7), which is between two adjacent headers (cipher_kernel_table.h and cipher_param_recovery.h) but does not break CLASSIFY itself; it blocks any future TU that pulls both.
- **Wave 5** (FUSION_PLAN) §5.1.a catalogues 12 classifier-to-actuator contracts (Ca.1 through Ca.12); §5.4 surfaces LP-11 (binary confidence), LP-14 (honest hot-path budget 100-200 ns), and LP-6 (80-layer drift, a hardcoded total_layers=80 that should be env-readable for non-Llama-3-70B models).
- No wave-vs-wave disagreement on CLASSIFY.

### F. Risk for integration

**requires-shim.** The classifier substrate is new code (the registry pattern + the wrapper for `classify_launch`). Compile blocker LP-7 must precede the port (Week 1 Step 1). Otherwise the port is mechanical: drop the header, register the observer, call from the cuLaunchKernel intercept. Budget honestly at 100-200 ns/launch; alarm at 500 ns/launch (per plan v1.2.1 §7 Week 2).

---

## 2. ORACLE

### A. Current implementation

| Field | Value |
|---|---|
| Definition file | `cipher-may13-evidence/cipher_oracle.cpp:293-377` (the `cipher_oracle_decide` function) |
| Header | `cipher-may13-evidence/include/cipher_oracle.h` |
| Tree | may13 only |
| LOC of `cipher_oracle_decide` | ~85 lines |
| Public symbols | `cipher_oracle_init`, `cipher_oracle_decide`, `cipher_oracle_record_substitution`, `cipher_oracle_bill_gemm`, `cipher_oracle_bill_nongemm`, `cipher_oracle_set_phase`, `cipher_oracle_report`, `cipher_oracle_update_gradients` |

### B. What it actually does

Five-gate safety decision walked sequentially on every dispatch call. Inputs: `CipherOracleQuery { layer_idx, total_layers, op_class, confidence, kernel_name, is_backward, is_optimizer }`. Outputs: `CipherOracleResult { decision: PERMIT/DENY, reason: const char* }`.

The five gates per `cipher_oracle.cpp:293-377`:

1. **Phase gate** (L324-328) — if `state->phase.detected_phase == 0` (warmup, not steady), return DENY "warmup". Topological phase auto-detection runs before this on every call (L317).
2. **Minimum confidence gate** (L330-342) — compares `q->confidence` against `cfg.min_confidence + mfu_confidence_adjust`, clamped to [20, 95]. Default min_confidence is 60. Returns DENY "low-confidence" on failure.
3. **Structural lookup gate** (L344-360) — calls `cipher_struct_lookup` with the (kernel_name, layer_idx, total_layers, op_class, phase, is_backward, is_optimizer_step) context; returns DENY "structural-rule" if the lookup says CIPHER_STRUCT_FULL_PRECISION.
4. **EMA permanent demotion gate** (L362-367) — checks `state->ema.permanently_demoted[layer_idx]`; returns DENY "ema-demoted" if set.
5. **N≤4 substitution rate-limit gate** (L369-...) — checks `sub_counter[layer_idx] >= cfg.n_max`; if hit, resets the counter, sets `force_passthrough[layer_idx]`, returns DENY "n4-rule".

If all five pass, returns PERMIT. `CIPHER_FORCE_PERMIT=1` env bypasses everything for testing.

Hot path: synchronous, per launch, in-thread. Stage 0 timing budget contributor.

### C. Where it fires in production today

**Not firing.** Like CLASSIFY, ORACLE lives only in may13. Not ported to `cipher_rt_phase4`. No production caller. In may13, ORACLE is called from `cipher_dispatch.cpp:373` (classify-only path) and `cipher_dispatch.cpp:468` (post-init main spine).

### D. Integration plan

**Port from may13 in Week 1.** New file `cipher_rt_phase4/cipher_rt_oracle.{cpp,h}` (port the top-level live file, not the silently-excluded src/ shadow). Calling convention stays unchanged. Wire into the hot path immediately after CLASSIFY in Week 2.

Fusion contract:
- Receives `op_class` + `confidence` from CLASSIFY.
- Receives `(layer_idx, total_layers, is_backward, is_optimizer)` from the kernel-context inference helper (`infer_layer_context` at `cipher_dispatch.cpp:454`).
- Returns `decision` and `reason` to the SUBSTITUTE dispatcher.
- Wave 5 contracts: Cb.4 (per-pid hook into kprobe pre-handler — Cb-side, optional), and reads structural rules from `cipher_structural_lookup` (port-as-is).

### E. Wave audit findings

- **Wave 1** §5-gate decomposition exhaustively walks every gate, including the topo phase detector. Wave 1 also confirmed `CIPHER_FORCE_PERMIT` is the documented bypass.
- **Wave 5** §5.4 LP-6 (80-layer hardcode) intersects ORACLE because the EMA gate uses `layer_idx < CIPHER_INFERRED_TOTAL_LAYERS` arithmetic. For Mistral-7B (32 layers), this works fine; for non-Llama-3-70B models, LP-6 surfaces the hardcoded constant.
- No disagreement.

### F. Risk for integration

**minor.** Standalone module; no Tree B dependencies; no nvcc; no kmod ABI changes. Ports as a single .cpp + .h pair. Risk is the honesty of the five-gate semantics in production; the gates are well-understood and Week 2's behavioral test verifies ORACLE stays in STEADY after warmup.

---

## 3. SUBSTITUTE — SPECIAL FOCUS (TWO PATHS: Marlin vs Koopman)

### A. Current implementation

The plan §2.1 names two parallel paths for SUBSTITUTE. They differ in tree, in mechanism, and in v1 scope.

#### Path A — Marlin INT4 GEMM (production, deployed runtime)

| Field | Value |
|---|---|
| Definition file | `cipher_rt_phase4/cipher_rt_marlin_actuator.c:170` (`g_marlin_actuator`) |
| Engine | `cipher_rt_phase4/cipher_rt_marlin_engine.cpp` (1078 LOC) |
| Kernel source | `cipher_rt_phase4/cipher_rt_marlin_kernel_src.cpp` (773 LOC, Apache-2.0 IST-DASLab kernel as a string literal) |
| Tree | cipher_rt_phase4 |
| Public symbols | `cipher_rt_marlin_init`, `cipher_rt_marlin_is_active`, `cipher_rt_marlin_calls_total`, `cipher_rt_marlin_calls_handled`, `cipher_rt_marlin_weights_quantized` |
| Registry registration | `cipher_rt_matmul_register_actuator(&g_marlin_actuator)` with priority 10 (`cipher_rt_marlin_actuator.c:207`) |
| Gate | Env `CIPHER_MARLIN` must be `on`/`1`/`ON`; FP16; K%128==0; N%64==0; stability-threshold observations passed; M<=MARLIN_MAX_M_GATE |

#### Path B — Koopman O(1) substitute (may13, dead code)

| Field | Value |
|---|---|
| Definition file | `cipher-may13-evidence/cipher_dispatch.cpp:242` (`apply_recipe` function) |
| Recipe library | `cipher-may13-evidence/src/cipher_recipes.cpp:346` (the 32+ shape registry) |
| Koopman runtime | `cipher-may13-evidence/src/cipher_koopman_runtime.cpp` (the L1.1 Runtime Koopman module — **dead code, zero callers** per plan §1.1) |
| Block kernel | `cipher-may13-evidence/src/cipher_block_sub_kernel.cu` |
| Attention kernel | `cipher-may13-evidence/src/cipher_attn_koopman_kernel.cu` |
| EDMD pipeline | `cipher-may13-evidence/src/cipher_edmd.cpp` + `cipher_edmd_live.cpp` |
| Tree | may13 only |
| Public symbols (not exported into deployed runtime) | `cipher_dispatch`, `apply_recipe`, `cipher_get_edmd_pipeline`, `cipher_edmd_live_collect` |

### B. What it actually does

**Path A (Marlin).** On every `cublasGemmEx` call routed through the matmul substrate, the Marlin actuator's `maybe_handle_marlin` (`cipher_rt_marlin_actuator.c:~110-180`) checks the call shape and weight pointer. If the weight has been observed STABILITY_THRESHOLD times and the call satisfies the regime gates (M<=cap, FP16, K%128==0, N%64==0), the actuator pre-shuffles the weight into Marlin's XOR-swizzled int32 layout (`marlin_repack` in `cipher_weight_compress.cpp`), allocates scales, and dispatches the Marlin kernel via `cipher_rt_marlin_engine_dispatch`. On success, returns `CIPHER_RT_MATMUL_HANDLED` and writes `CUBLAS_STATUS_SUCCESS` into `*out_status`. The substrate respects HANDLED and skips the real `cublasGemmEx`.

**Path B (Koopman).** On every may13 dispatch, after ORACLE permits, the `cipher_recipes` registry is queried by (op_class, shape_hash, arch). On a hit, `apply_recipe` is called. For GEMM (`recipe_type == 0`, `cipher_dispatch.cpp:248-300`), it checks whether the EDMD pipeline for that op_class has SOLVED to fit_error < 0.05 (`cipher_dispatch.cpp:259-272`); if yes, the Koopman surrogate replaces the GEMM. If no, the recipe-defined GEMM config is used. For elementwise/reduction (`recipe_type == 2`), the Chebyshev substitution path runs. In production, the registry is loaded with 32+ entries from `cipher_recipes.cpp:346` (spanning Llama-3-70B, SOMA motor, RMSNorm, HyperFlux, A100 shapes, and 10 shape-parametric entries) but **no real workload's M/N/K hashes match those entries**, so `apply_recipe` never runs. Plan §2.1 verified this: "no real workload's M/N/K hash matches them; therefore `apply_recipe()` and the Koopman substitution branch NEVER RUN."

### C. Where it fires in production today

**Path A (Marlin):** Fires on every `cublasGemmEx` call that satisfies the regime gates, when `CIPHER_MARLIN=on`. Verified to handle Mistral-7B Marlin lane at batch >= 8 prefill (the historical 1.38× headline at batch=8 per may13 CLAUDE.md Phase 2 measurements). When `CIPHER_MARLIN` is unset (the default), the actuator returns 0 from init and registers nothing.

**Path B (Koopman):** Zero substitutions ever fire on any deployed configuration. Plan §1.6 "Test coverage map" row: "Goal-4 Koopman substitutions: All stress/stress2 logs — **0 substitutions ever fire** (zero matches in [L3.2] SUBSTITUTE, [O(1)-block], etc.)"

### D. Integration plan

**v1 scope: Path A stays in cipher_rt_phase4 as-is. Path B remains dead code in may13, NOT ported.**

The unified runtime keeps Marlin as the canonical SUBSTITUTE actuator. The dispatch lane in plan v1.2.1 §4.5 "Routing tables" reserves a "Koopman O(1) lane" row in the dispatch table but marks it explicitly as "v2 — Koopman O(1) lane — deferred." The mathematics (EDMD, LNN, KEN) stays in may13 unbuilt against the deployed runtime; the runtime module (`cipher_koopman_runtime.cpp`) stays dead in may13 with zero callers.

Fusion contract:
- Path A: takes the matmul `cipher_rt_matmul_call` (handle, M/N/K, A/B/C ptrs, alpha/beta/stream), returns HANDLED + status, or PASSTHROUGH, or ERROR. Documented in plan v1.2.1 §4.0.
- Path B: would need EDMD live wiring + a registry repopulation pass; both are explicit v2 deferrals.

The `apply_recipe` branch in may13's `cipher_dispatch.cpp` does NOT port. When Week 1 ports `cipher_dispatch.cpp` as `cipher_rt_phase4/cipher_rt_dispatch.cpp`, the apply_recipe call site can be carried as a stub that returns PASS_THROUGH (consistent with empty registry behavior); the v2 wire-up would replace the stub with a real EDMD-pipeline check.

### E. Wave audit findings

- **Wave 2** (ACTUATORS) is the deep-dive for the Marlin path. The full Marlin engine, kernel source, scales, repack, and dispatch pipeline are documented contract-by-contract. Wave 2 identified F2-X HANDLED-DISCARD on attn (Wave 5 LP-2) but matmul HANDLED is respected.
- **Wave 5** §5.1.a Ca.9 (Marlin TLS-hint read), §5.1.c Cc.7 (CP54 ALLOCATE-side handshake when Marlin × partitioning becomes Phase 6 work). §5.2.1 (cuBLAS path through the substrate) explicitly names Marlin as the production-default SUBSTITUTE actuator. §5.4 LP-1 (relaunch contract) — the matmul substrate's existing chain pattern IS the relaunch mechanism Tree A intended, so the weak-symbol re-entry contracts (Ca.2/Ca.3) are superseded by the substrate.
- The Koopman path is named in plan §1.1 / §2.1 / §5.6 R-A5 as a v2 explicit deferral, not a v1 lane. No wave covered the Koopman pipeline end-to-end because it has zero callers; Wave 5 LP-1's resolution explicitly subsumes Tree A's intended Koopman-relaunch mechanism into the substrate.

### F. Risk for integration

**minor for v1.** Marlin stays where it is and works. The structural Marlin × partitioning composition (Marlin is pinned to primary context per Fix A; cannot compose with 8-SM green-context partitions) is a Phase 6 Song Han engagement item, scoped explicitly out of v1 per plan §8 R-C3.

**out-of-v1-scope for the Koopman lane (v2).** Wiring Path B to a registry that actually matches real workloads, plus an EDMD live pipeline that converges, plus a Koopman surrogate that meets the bit-identical correctness gate, is a v2 research program (plan §8 honest gap #1). Not a v1 deliverable.

---

## 4. COMMIT (= ORCHESTRATE)

### A. Current implementation

| Field | Value |
|---|---|
| Definition file | `cipher-may13-evidence/cipher_dispatch.cpp` (the dispatch return path; the SUBSTITUTE-success branch returns `CIPHER_SUBSTITUTED`, the PASS_THROUGH branch returns `CIPHER_PASS_THROUGH`) |
| Production usage | implicit in `cipher_rt_phase4/cipher_rt_matmul_dispatch.c::cipher_rt_matmul_dispatch` (`L100-122`); the loop returns the chosen actuator's `*out_status` directly, no separate COMMIT op |
| Tree | both: may13's COMMIT is a phase inside cipher_dispatch.cpp; rt_phase4's COMMIT is implicit in the substrate dispatch return |
| LOC | ~30 lines of return-path glue |
| Public symbols | none distinct; COMMIT is a label, not a function. There is a Track 3 DSM state-machine TOKEN named `CIPHER_CP54_MIGOUT_COMMITTED` at `cipher_cp54_sched.c:395` and an ACK-commit at L383 — that COMMIT is the migration-FSM commit, scoped to DSM, not the per-launch COMMIT op |

### B. What it actually does

COMMIT is the dispatch return: once SUBSTITUTE has succeeded (or PASS_THROUGH has been chosen), COMMIT launches the chosen kernel and returns the result to the caller. In may13, this is the `return CIPHER_SUBSTITUTED;` / `return CIPHER_PASS_THROUGH;` at the bottom of `cipher_dispatch`. In `cipher_rt_phase4`'s matmul substrate, COMMIT is the actuator chain's HANDLED short-circuit (`cipher_rt_matmul_dispatch.c:108-111`):

```c
if (result == CIPHER_RT_MATMUL_HANDLED) {
    atomic_fetch_add(&g_disp.handled, 1);
    return status;
}
```

And the all-PASSTHROUGH fallthrough (`L118-122`) that calls the real `passthrough_fn` for the cuBLAS call.

Inputs: the actuator's return enum + `*out_status`. Outputs: the cuBLAS-status code returned to the cuBLAS shim's caller. Hot path: per launch, synchronous.

The ORCHESTRATE name is a legacy may13 spelling. It survives at `cipher_10ops_impl.cpp:972` only as a print-label string ("SUBSTITUTE+ORCHESTRATE+GENERATE+RING_WRITE"), and at `cipher_straggler.h:29` as a comment about NCCL algo-hint sinks. There is no separate ORCHESTRATE function in any tree.

### C. Where it fires in production today

**Implicitly on every routed matmul call** (the substrate dispatch return at `cipher_rt_matmul_dispatch.c:108-122`). **Implicitly on every routed SDPA call** (the analogous `cipher_rt_attn_dispatch.cpp:211-242` substrate return). The launch counter `g_disp.handled` increments per HANDLED, the `g_disp.passthrough` counter increments per all-PASSTHROUGH; the totals are reported at process exit via `cipher_rt_matmul_atexit_diag` (`cipher_rt_matmul_dispatch.c:30-37`).

The Track 3 DSM `CIPHER_CP54_MIGOUT_COMMITTED` token (`cipher_cp54_sched.c:395`) is a separate, in-kernel state-machine commit for the migration FSM, unrelated to per-launch COMMIT. Plan v1.2.1 §2.3 corrected an earlier over-claim that "COMMIT" lived as both an op token in cp54_sched AND a weight-arena commit-on-publish; the v1.2.1 correction restricted COMMIT-as-FSM-token to `cipher_cp54_sched.c` only.

### D. Integration plan

**Stays in cipher_rt_phase4 as-is.** No port needed. The unified runtime's COMMIT is the matmul / attn / classifier substrate's return path; when the classifier substrate is added in Week 1, COMMIT-equivalent is the substrate dispatch return semantics for that registry.

Fusion contract:
- Receives: actuator return enum + `*out_status` (matmul) or `cipher_rt_attn_result` (attn).
- Returns: cuBLAS status (matmul) or `void` with `orig()` being called or skipped (attn).
- Plan v1.2.1 §4.0 defines the chosen contract per substrate (matmul: 3-value enum with break-on-ERROR; attn: 4-value enum with HANDLED currently structurally discarded per LP-2).

### E. Wave audit findings

- **Wave 2** (ACTUATORS) §registry-pattern documents the matmul + attn substrate dispatch returns as the canonical COMMIT pattern.
- **Wave 5** §5.1.c Cc.7 (CP54 ALLOCATE handler) covers the DSM COMMIT token (the migration FSM commit), distinct from per-launch COMMIT.
- No wave-vs-wave disagreement; v1.2.1 §2.3 already corrected the prior over-claim.

### F. Risk for integration

**minor.** COMMIT is fully production-grade in the deployed runtime. The integration is purely cosmetic: the unified runtime documents COMMIT as the per-launch dispatch return contract, with one canonical implementation per substrate.

---

## 5. SAMPLE (= GENERATE) — SPECIAL FOCUS (the questioned identity)

### A. Current implementation

| Field | Value |
|---|---|
| Definition file | `cipher-may13-evidence/src/cipher_intercept_cudart.cpp:2353-2425` |
| Tree | may13 only (in `libcipher_hook.so`, the LD_PRELOAD intercept) |
| LOC | ~73 lines |
| Public symbols | `cipher_read_sample` (`L2371-2380`), `cipher_sample_wseq` (`L2382-2385`) |
| Static functions | `compute_sample_norm` (`L2389-2405`), `maybe_collect_sample` (`L2408-2425`) |
| Data structures | `CipherSampleEntry { float x[1]; float y[1]; uint8_t op_class; uint8_t valid; }`; `g_sample_ring[512]`; `g_sample_wseq` atomic write counter |

### B. What it actually does

After each intercepted kernel launch, `maybe_collect_sample(op_class)` is called (the post-launch hook in `cipher_intercept_cudart.cpp`). It:

1. Increments a static `s_launch_count` counter.
2. **Hard-stops after 500 launches** (`L2411`: `if (s_launch_count > 500) return;`).
3. Checks `tls_shape_valid && tls_A && tls_C` (TLS-resident GEMM tensor pointers populated by the cuBLAS shim).
4. Calls `compute_sample_norm(tls_A, M*K)` and `compute_sample_norm(tls_C, M*N)`. Each computes the normalized Frobenius norm of a 256-element sample copied from device memory via `cudaMemcpy` (`L2389-2405`). The sample size is hard-coded at 256 floats per tensor.
5. Skips writing if both norms are below 1e-10 (degenerate / zero-input launches).
6. Writes one `CipherSampleEntry { x, y, op_class, valid=1 }` into `g_sample_ring[idx % 512]` and advances `g_sample_wseq` with release-order.

The function exists for the ADAPT thread in `libcipher_rt.so`: ADAPT reads samples via `cipher_read_sample(seq, x_out, y_out, op_class_out)` (`L2371`) and feeds them into the EDMD pipeline. This is the **GENERATE step**: the post-launch sampler that produces (input_norm, output_norm) pairs as training data for the EDMD-then-Koopman learning pipeline.

Why two names: the historical may13 CLAUDE.md listed it as GENERATE (the 12-core-ops nomenclature, paired with RING_WRITE as the per-launch observability twin); the v1.2.1 plan §2.1 lists it as "SAMPLE / GENERATE" because the production function reads as a sampler.

Hot path: yes during warmup (the first 500 launches per process). Cold after warmup.

### C. Where it fires in production today

**Fires on every cuLaunchKernel intercept during the first 500 launches per process, then never again.** The hard-stop at `L2411` is the warmup limiter. Plan §2.1 explicitly notes "PARTIAL — warmup-only" for this op.

The sample ring is read by the may13 Stage 2 ADAPT thread (`cipher_10ops_impl.cpp:597-765`). In production, the ADAPT thread does not spawn because the observer thread is lazy-gated (plan §1.1: "at line 949-965, threads spawn only if `observers_enabled > 0` (sum of 20 observer `_init()` returns); default-off observers → no threads → 6 core ops NEVER FIRE in production"). So even when SAMPLE produces 500 entries per process during warmup, no consumer reads them in production.

### D. Integration plan

**Defer to v2 with the rest of the learning tier.** SAMPLE only has product value when ADAPT consumes its output to produce a converged EDMD Koopman surrogate, which then populates the SUBSTITUTE registry. Without ADAPT firing and without the Koopman lane being wired, SAMPLE produces data nothing reads. Plan §4.4 places SAMPLE at "Mode 3 TLS / sync / per launch (warmup)" with the per-tenant ring as state; the unified v1 runtime ships with SAMPLE not wired.

If the v2 learning tier is wired, SAMPLE ports as a Stage 0 inline hook in the cuLaunchKernel intercept (already implemented in may13; carrying it over to `cipher_inject.c` is mechanical). The hard-stop at 500 launches is a warmup heuristic; v2 would either remove the stop (continuous training) or replace it with a workload-class-aware re-sampling cadence.

Fusion contract (when wired in v2):
- Reads from: TLS GEMM tensor pointers (set by the cuBLAS shim, contract Ca.9-style).
- Writes to: per-tenant sample ring (Mode 3 TLS).
- Consumed by: ADAPT (which then feeds EDMD).

### E. Wave audit findings

- **Plan §2.1** is the primary cite (no Wave deep-dives this; SAMPLE is not on the v1 hot path).
- **Wave 5** §5.4 LP-1 (Tree A weak-symbol re-entry) intersects because the historical re-entry contract `tls_get_gemm_shape` is the way SAMPLE reads the GEMM operand sizes. The v1.2.1 substrate-actuator pattern supersedes the re-entry contract; if SAMPLE ports in v2, it ports as an actuator on the matmul substrate (priority 0, always PASSTHROUGH, side-effects the sample ring), exactly the AUDIT template.
- No disagreement.

### F. Risk for integration

**out-of-v1-scope (v2 learning tier).** The op is fully functional in may13; carrying it into the unified runtime is mechanical when the v2 learning tier wires up. Plan §8 honest gap #1 names this: "Goal-4 Koopman O(1) substitution is NEVER built into v1. The math exists (cipher_edmd, cipher_lnn). The runtime module exists (cipher_koopman_runtime — dead code, zero callers). The hot-path lane is reserved (Section 4.5 dispatch table). v2 work."

---

## 6. RING_WRITE

### A. Current implementation

| Field | Value |
|---|---|
| Definition | `cipher-may13-evidence/include/cipher_10ops.h:83-100` (the `cipher_ring_write` inline function) |
| Ring data structure | `cipher_10ops.h:CipherRing` (65,536-slot Disruptor SPMC) |
| Call sites | `cipher-may13-evidence/src/cipher_intercept.cpp:169-184` (Stage 0 post-dispatch) and `:374-376` (Stage 0 from cuBLAS GEMM path) |
| Tree | may13 only |
| LOC | ~18-line inline function + 9-line struct |
| Public symbols | `cipher_ring_write` (header inline); `g_cipher_10ops.ring` (global runtime object); `cipher_feed_ring` (`cipher_10ops_impl.cpp:1061`) |

### B. What it actually does

A single-producer multi-consumer (SPMC) lock-free Disruptor-pattern ring buffer (header `cipher_10ops.h:83-100`). The ring has 65,536 entries (a power of two for fast `& MASK` indexing). Each entry is a `CipherRingEntry` carrying `(sequence, timestamp_ns, timestamp_delta, kernel_class, grid_x/y/z, block_x/y/z, func_ptr_hash, confidence, decision, params_hash)`. The write path uses memory-order-relaxed loads of the read sequences from both Stage 1 and Stage 2 consumers, picks the slower one, and if the ring is not full does a memcpy of the entry plus a release-order store of `write_seq+1`. On x86 TSO this is a plain MOV. On full ring (consumers fell behind), the write is dropped with no error (lossy).

The op is named "RING_WRITE" because it is the Stage 0 producer's sole job per kernel launch: write one ring entry of metadata for downstream consumers.

Inputs: a populated `CipherRingEntry` filled by the caller. Outputs: 1 (written) or 0 (ring full, dropped). Hot path: per launch, synchronous, ~10 ns budget.

### C. Where it fires in production today

**Fires on every successful Stage 0 dispatch path in may13** (`cipher_intercept.cpp:169-184`, the post-`cipher_dispatch` call). Also fires on the cuBLAS GEMM intercept side path (`cipher_intercept.cpp:368-376`) that packs `(M, K, N)` into the params_hash tail for SENSE/SHIELD to recover. Counters are kept in `g_cipher_10ops.ring_writes`.

**Does NOT fire in the deployed runtime.** `cipher_rt_phase4` does not have a ring writer; the launch-stats path uses CUPTI directly via `cipher_cupti.c`'s 1 kHz daemon (which submits a different aggregate via kmod ioctl nr 7 SUBMIT_LAUNCH_STATS).

### D. Integration plan

**Port from may13 in Week 2.** The ring buffer is the substrate the Stage 1 shadow thread (REMEMBER/VALIDATE/AUDIT/SPECULATE) and Stage 2 background thread (ADAPT/ARBITRATE) read from. If those threads port (they do not in v1; see plan §1.1 "Stage 1/2 threads NEVER spawn in production"), the ring needs to port too.

For v1, the deployed runtime already records per-launch metadata via CUPTI; the may13 ring is functionally redundant with the CUPTI path. A pragmatic v1 stance is: **do not port the ring in v1**. The CUPTI launch counter daemon (`cipher_cupti.c`) already produces the per-launch metadata that the ioctl nr 7 telemetry consumes; the kmod's per-tenant snapshot (ioctl nr 8) is the cross-process consumer of this metadata. Adding a redundant userspace ring buffer in the unified runtime would be Wave 5 LP-1-style: replacing the existing substrate with an older alternative.

Plan §2.3 places RING_WRITE in the observability tier as "WORKING — when 10ops initialized" — meaning it works when the 10ops runtime is initialized but is not the production telemetry path. v1.2.1's dispatch pipeline (§4.2 step 2) places "RING_WRITE → SPMC ring entry (timestamp, params_hash, class)" inline, but a careful reading shows this step is what the unified runtime would do *if* the ring is ported; if the CUPTI path stays, this step is functionally absorbed into CUPTI's per-launch counter.

Fusion contract (if ported):
- Reads from: CLASSIFY output + ORACLE decision + intercept timestamps.
- Writes to: SPMC ring (global) for shadow-thread consumption.
- Wave 5 Cd.7 (Stage-1 SPMC ring fan-out): observer-to-classifier contract.

### F. Risk for integration

**minor / requires-supersession-decision.** No technical blocker. The decision is whether v1 wants a separate userspace ring (port RING_WRITE) or wants the existing CUPTI path to remain the sole launch-telemetry sink. Plan §1.6 already labels the Stage 1/2 path "NEVER FIRES in production"; v1.2.1 §4.4 places the ring under "per-tenant ring (Mode 3 TLS)." The cleanest v1 stance is: ring stays in may13 as a dependency of the (deferred) v2 learning tier; the unified v1 runtime uses CUPTI as today.

---

## 7. REMEMBER

### A. Current implementation

| Field | Value |
|---|---|
| Definition file | `cipher-may13-evidence/src/cipher_10ops_impl.cpp:428-458` (REMEMBER update inside the stage1_shadow thread) |
| Backing module | `cipher_lnn.cpp` (CfC Liquid Neural Network forward) |
| Tree | may13 only |
| LOC | ~30 lines of REMEMBER update logic + shared use of the LNN module |
| Public symbols | none (REMEMBER is inline inside the stage1 thread); state in `s_shadow_lnn` (`cipher_10ops_impl.cpp:82`); ring entry consumed |
| Header constant | `KEN_R=8`, `KEN_N=16` for the spectral subspace dimensions of the companion Koopman Eigenfunction Network |

### B. What it actually does

REMEMBER maintains a per-process shadow CfC (Closed-form Continuous-time) recurrent network state of dimension 64. On each ring entry the Stage 1 shadow thread reads, REMEMBER captures the LNN hidden state `h_before`, calls `cipher_lnn_forward(&s_shadow_lnn, ...)`, captures `h_after`, and writes a per-op slot for Stage 2 ADAPT to read. The hidden state encodes a sliding picture of recent kernel-class history. REMEMBER's outputs are consumed by:

- SPECULATE: reads `s_shadow_lnn.recipe_type` to predict the next kernel class (`cipher_10ops_impl.cpp:477-505`).
- ADAPT: reads `(h_before, h_after)` pairs as the EDMD snapshot input (`cipher_10ops_impl.cpp:562-572`).

Per `cipher_10ops_impl.cpp:5-13` block comment: "REMEMBER → cipher_lnn_forward() on dedicated shadow CfC (64-dim hidden)". The one-step shadow-lag formal bound noted in the file header is `error = O(L * delta_t)` with `L = Lipschitz constant of CfC dynamics`; at the documented 1-100 µs kernel intervals, the bound is negligible.

Hot path: **no.** REMEMBER runs in the Stage 1 shadow thread, not on the kernel-launch thread.

### C. Where it fires in production today

**Never.** Plan §1.1 confirms: "Stage 1+2 background threads... default-off observers → no threads → 6 core ops NEVER FIRE in production." Plan §2.1 confirms "WORKING code, NEVER FIRES (Stage 1 thread not spawned)."

### D. Integration plan

**Out of v1 scope.** v1 ships the substrate without the Stage 1 shadow thread. Plan §4.1 explicitly: "No Stage 1/2 background threads in v1." If v2 ships the learning tier, REMEMBER ports as part of that program.

Fusion contract (when v2 ports):
- Reads from: per-tenant SPMC ring (consumed via `read_seq_s1` advance).
- Writes to: `s_shadow_lnn` per-tenant state; per-op-slot hidden states for ADAPT.

### F. Risk for integration

**out-of-v1-scope (v2 learning tier).** Plan §8 R-T2 deliberately defers: "Stage 1/2 ops (REMEMBER/VALIDATE/ADAPT/SPECULATE) were never tested in production; porting them risks introducing unknown failure modes."

---

## 8. VALIDATE

### A. Current implementation

| Field | Value |
|---|---|
| Definition file | `cipher-may13-evidence/src/cipher_10ops_impl.cpp:460-475` (the VALIDATE step inside stage1_shadow) |
| Backing structure | `RStats` (per-class Welford online statistics; `cipher_10ops_impl.cpp:325-340`) |
| Tree | may13 only |
| LOC | ~16 lines |
| Public symbols | none (inline inside the stage1 thread) |

### B. What it actually does

VALIDATE runs Welford online (mean, variance) statistics on the SPMC ring's `decision` field per kernel class. The Welford updater is `rs_update` at `cipher_10ops_impl.cpp:327` and the 3-sigma anomaly detector is `rs_ok` at `:334`. After every ring read, the Stage 1 shadow thread runs:

```c
rs_update(&class_stats[class], x);   // online mean, variance
// (3-sigma detection rs_ok is INSTANTIATED but NOT CALLED on the live path)
```

Plan §2.1 notes: "PARTIAL — Welford stats real; `rs_ok` 3-σ detector never called." Plan §2.2 confirms VALIDATE as a partial op.

In production, the HMAC step at the bottom of this block (`cipher_10ops_impl.cpp:472-474`) is commented out: "AUDIT: skip HMAC on detached thread (OpenSSL not thread-safe here)."

Hot path: no (Stage 1 shadow thread).

### C. Where it fires in production today

**Never.** Same reason as REMEMBER: the Stage 1 thread does not spawn.

### D. Integration plan

**Defer to v2 with the learning tier**, with one nuance: the Welford accumulator part of VALIDATE could ship as a priority-0 observability actuator on the matmul / attn substrates if the kernel-class anomaly signal becomes a customer-visible feature. Plan §4.4 places VALIDATE at "shadow-thread" — v1 does not ship it.

### F. Risk for integration

**out-of-v1-scope (v2).** Same as REMEMBER. The instantiated-but-never-called `rs_ok` detector is a known partial; v1.5 could ship the detector behind an env gate without changing the math.

---

## 9. AUDIT

### A. Current implementation

| Field | Value |
|---|---|
| may13 definition | `cipher-may13-evidence/src/cipher_10ops_impl.cpp:341-378` (`audit_chain_update`) and `:472-474` (call site, commented out) |
| Production definition | `cipher_rt_phase4/cipher_rt_audit.c` (222 lines) — full HMAC-SHA256 chain implementation; AUDIT_KEY embedded; ring + dump-at-exit |
| Tree | both: may13 (commented-out shadow-thread path) + cipher_rt_phase4 (production, env-gated) |
| Public symbols (rt_phase4) | `cipher_rt_audit_init`, `cipher_rt_audit_record`, `cipher_rt_audit_armed`, `cipher_rt_audit_count` |
| Registers on substrates | matmul (priority 0, `audit_matmul_act`); attn (priority 0, `audit_attn_act`) |
| Env gate | `CIPHER_AUDIT`; optional `CIPHER_AUDIT_DUMP=path` for offline verification by `audit_verify.py` |

### B. What it actually does

AUDIT maintains an HMAC-SHA256 tamper-evident chain over a stream of `(op_class, call_hash, decision, timestamp_ns, timestamp_delta, sequence)` tuples per process. The chain link is canonical 96 bytes: `chain[32] || seq[8] || ts[8] || delta[8] || call_hash[8] || op_class[4] || decision[1] || pad[27]`. On every call to `cipher_rt_audit_record(op_class, call_hash, decision)`, the function locks `g.mu`, fills the next ring slot (8192-deep), updates `chain` via HMAC-SHA256 with `AUDIT_KEY`, and increments `g.count`. The AUDIT_KEY at `cipher_rt_audit.c:21-26` is "C110E0A0 D170E000 ... NEURALDY"; production deploys are expected to load from TPM or secure enclave per the file comment.

At process exit (`cipher_rt_audit_dump_at_exit`, `cipher_rt_audit.c:148-180`), if `CIPHER_AUDIT_DUMP` names a path, the chain head + ring entries are written in a file format that `audit_verify.py` can replay offline to verify the chain.

AUDIT is registered as a priority-0 observer on BOTH the matmul substrate and the attn substrate. On every dispatch, the actuator's `audit_matmul_handle` / `audit_attn_handle` is called; both compute a `fnv1a` hash of the call descriptor and call `cipher_rt_audit_record`. Both return PASSTHROUGH so the actual cuBLAS / SDPA call proceeds normally.

Hot path: yes (per-launch HMAC-SHA256 on the audit chain when `CIPHER_AUDIT=on`).

### C. Where it fires in production today

**Conditionally — env-gated.** When `CIPHER_AUDIT=on`, AUDIT fires on every `cublasGemmEx` and every SDPA call routed through the substrates. Plan §1.6 and §2.3 mark AUDIT as one of the 6 ops that fire on Tree B's hot path today.

**Not in may13 shadow thread:** the call site at `cipher_10ops_impl.cpp:472-474` is commented out with "skip HMAC on detached thread (OpenSSL not thread-safe here)." This is a known partial in may13; production AUDIT runs in `cipher_rt_audit.c` instead.

### D. Integration plan

**Keep in cipher_rt_phase4 as-is.** The unified runtime's AUDIT is the deployed rt_phase4 path. Plan §3.1 row 3 ("HMAC chain types... pick one — rt_phase4's AUDIT is leaner and already shipped. Retire may13's AUDIT internals."). The may13 `audit_chain_update` and inline shadow-thread path are NOT ported.

Wave 5 LP-4 names the v1.5 concern: single global mutex on `g.mu` serializes the per-launch HMAC. At 16 tenants × 10K launches/s/tenant = 160K record()/s × 50 ns HMAC, ~8 ms/s wall is consumed; under contention worst-case ~50% utilization of one core. Today the cost is 0 because the default is `CIPHER_AUDIT=` (unset, OFF). Plan v1.2.1 §7 Week 4 schedules the AUDIT lockless refactor for the N=100 scale gate; if Week 4 measurement does not show the contention, the refactor flips to v1.5 per Wave 5 §5.5.

Fusion contract:
- Reads: matmul / attn call descriptors via the substrate actuator interface.
- Writes: HMAC chain head + ring; per-tenant counters.
- Wave 5 Cd.8 (AUDIT record() shim for substitute-lane): the AUDIT contract upgrades when AUDIT becomes the substitute-decision recorder (today it observes; with Cd.8, it records SUBSTITUTED vs PASSTHROUGH per call).

### E. Wave audit findings

- **Wave 3** (KMOD) §observability covers AUDIT on the kmod side (telemetry path nr 8 GET_TENANT_SNAPSHOT exposes per-tenant audit counters).
- **Wave 4** (OBSERVERS) is the deep-dive for AUDIT-userspace. Documents the chain, the AUDIT_KEY, the dump-at-exit format, and the offline verification tool. Wave 4 §FUSION POINTS surfaced the single-mutex hot-path concern.
- **Wave 5** §5.1.d Cd.8 (substitute-lane recorder upgrade) and §5.4 LP-4 (audit lock, v1.5 sharding plan).
- No wave-vs-wave disagreement.

### F. Risk for integration

**minor (requires-mitigation at N=100 scale).** The op is fully production-grade. The mutex contention risk is named, has a v1.5 mitigation path (per-tenant chain sharding), and is gated by Week 4 N=30 stress measurement before CP 5.5.

---

## 10. SPECULATE

### A. Current implementation

| Field | Value |
|---|---|
| Definition file | `cipher-may13-evidence/src/cipher_10ops_impl.cpp:477-505` (SPECULATE write inside stage1_shadow) |
| Hot-path check site | `cipher-may13-evidence/src/cipher_intercept.cpp:124-149` (the Stage 0 SPECULATE look-aside check, ~2 ns budget) |
| Backing structure | `g_cipher_10ops.look_aside` (a hash-keyed look-aside buffer; see `cipher_10ops.h`) |
| Tree | may13 only |
| Public symbols | `cipher_lookaside_check`, `g_cipher_10ops.look_aside` (header globals); `g_cipher_10ops.speculate_hits` / `.speculate_total` counters |

### B. What it actually does

SPECULATE has two sub-ops, both noted in the may13 CLAUDE.md ("SPECULATE r/w" = read + write):

**SPECULATE-write (Stage 1 shadow thread, `:477-505`):** after REMEMBER updates the hidden state, the Stage 1 thread reads `s_shadow_lnn.recipe_type` (the CfC-decided predicted class) and writes a (key, predicted_class, confidence) entry into the look-aside buffer. The key is the kernel class of the latest observed launch; the entry is the predicted-class hint for the next launch.

**SPECULATE-check (Stage 0 hot path, `cipher_intercept.cpp:124-149`):** before `cipher_dispatch` is called, the Stage 0 path checks `cipher_lookaside_check(&g_cipher_10ops.look_aside, (int)desc.op_class)`. On hit, the precomputed dispatch result is treated as SUBSTITUTED and the kernel launch is skipped (`cipher_intercept.cpp:147`). On miss, the standard dispatch path runs.

Hot path: yes for the check; no for the write.

### C. Where it fires in production today

**Never.** The Stage 1 thread never spawns, so SPECULATE-write never runs, so the look-aside is empty, so SPECULATE-check always misses, so SPECULATE-as-skip never fires. Counters in `g_cipher_10ops.speculate_hits` stay at 0 in every production stress log.

The check is harmless when the look-aside is empty (one atomic load returns "not found" in ~2 ns).

### D. Integration plan

**Out of v1 scope.** SPECULATE has no value without the shadow thread populating the look-aside. Carrying the check into the unified hot path without the write is dead code. v2 with the learning tier ports the pair together.

### F. Risk for integration

**out-of-v1-scope (v2).** Same as REMEMBER / VALIDATE / ADAPT.

---

## 11. ADAPT

### A. Current implementation

| Field | Value |
|---|---|
| Definition file | `cipher-may13-evidence/src/cipher_10ops_impl.cpp:597-765` (the ADAPT block inside stage2_background) |
| EDMD pipeline | `cipher_10ops_impl.cpp:87` declares `s_adapt_edmd[7]` (one per op class) |
| KEN backing | `cipher_10ops_impl.cpp:99-211` (Koopman Eigenfunction Network — encoder/evolver/decoder forward + eigenfunction loss) |
| EDMD module | `cipher-may13-evidence/src/cipher_edmd.cpp` (the EDMD solver itself) |
| Tree | may13 only |
| LOC | ~170 lines for the ADAPT block + the KEN + EDMD modules |
| Public symbols | `cipher_get_edmd_pipeline` (`:292`); `g_cipher_10ops.adapt_swaps` counter |

### B. What it actually does

ADAPT is a Stage 2 background thread block that:

1. Reads (h_before, h_after) snapshots from REMEMBER per op class (`:605-637`).
2. Feeds them into `s_adapt_edmd[class]` via `cipher_edmd_pipeline_feed_sample`.
3. When a pipeline accumulates enough samples (the EDMD solver fires), reads the resulting Koopman operator + fit error.
4. If fit_error < 0.05, marks the pipeline as `CIPHER_EDMD_SOLVED` and the SUBSTITUTE branch in `cipher_dispatch.cpp:259-272` is then permitted to use it.

Plan §2.1 critically notes: "PARTIAL — EDMD solver real but fed degenerate `h_before==h_after` synthetic input (`:693-701, :663-669`); NEVER FIRES." That is, the EDMD math works on real inputs but the snapshot feeder path passes synthetic constant data, so the solver finds a trivial identity operator and never produces a useful Koopman map.

ADAPT also has a "direct sample ring path" (`:743-761`) that pulls SAMPLE data from `cipher_intercept_cudart.cpp`'s ring; this is the GENERATE → ADAPT data flow that closes the learning loop if both sides actuate.

Hot path: no (Stage 2 background thread).

### C. Where it fires in production today

**Never.** Stage 2 thread does not spawn (default-off observers); even if it spawned, the synthetic input path makes the EDMD solver converge to identity, which the SUBSTITUTE branch then declines (the fit_error gate is satisfied but the surrogate is the identity matrix and doesn't substitute anything meaningful). Plan §2.1: "NEVER FIRES."

### D. Integration plan

**Out of v1 scope.** ADAPT requires:
1. The Stage 2 thread to spawn (which requires Stage 1 / observer infrastructure).
2. The snapshot feeder bug (synthetic constant `h_before == h_after`) to be fixed.
3. A real EDMD solver convergence on real GEMM-shape data.
4. The SUBSTITUTE path's Koopman lane to be wired (currently the `apply_recipe` Koopman branch checks the EDMD status, but with empty registry the lookup never reaches there).

All four are v2 work. v1.2.1 §8 R-T2 explicitly defers.

### F. Risk for integration

**out-of-v1-scope (v2 learning tier).** Plan §1.6 "0 substitutions ever fire" applies here too.

---

## 12. ARBITRATE

### A. Current implementation

| Field | Value |
|---|---|
| Legacy may13 definition | `cipher-may13-evidence/src/cipher_10ops_impl.cpp:767-814` (writes a SHM demand signal; SM rebalance was TODO) |
| Production definition | `cipher_kmod/cipher_cp54_sched.c` (894 LOC: SM-arbitration ledger, 15 × 8-SM groups, two-clause disjointness, do_exit reaper, Track 3 DSM state machine) |
| Tree | both; Tree A (may13) is the predecessor that wrote SHM demand only; Tree B2 (cipher_kmod) is the production state-machine that actually allocates and migrates |
| LOC | rt-side: 47 lines (may13 demand-only); kmod-side: 894 lines (production) |
| Public symbols (kmod) | `cipher_dev_cp54_ioctl` dispatcher (nrs 13-20); `cipher_cp54_release` (FREE + do_exit reaper hook); migration FSM tokens `CIPHER_CP54_MIGOUT_*` |
| Userspace API | ioctl nrs 13 (CP54_ALLOCATE), 14 (CP54_FREE), 15 (CP54_QUERY), 16 (CP54_SUBSCRIBE_MIGRATE), 17 (CP54_POLL_MIGRATE), 18 (CP54_START_MIGRATE), 19 (CP54_ACK_MIGRATE), 20 (CP54_COMPACT_MIGRATE) |

### B. What it actually does

ARBITRATE is the cross-process SM-allocation authority. On `CP54_ALLOCATE` (nr 13), the kmod acquires `cipher_cp54_lock` (mutex), claims `ceil(sm_count/8)` 8-SM groups via per-group atomic_cmpxchg from a 15-entry atomic array, records the assignment in a metadata table slot keyed by PID + QoS class (PARTITION / POOL / SHARED), and returns the group mask to the caller. The two-clause disjointness invariant is enforced at runtime: no two PARTITION tenants share a group; no group is both held by a PARTITION tenant and the POOL holder.

On `CP54_FREE` (nr 14) or `do_exit` (the kprobe reaper at `cipher_cp54_sched.c:359`), the kmod releases groups via lock-free atomic_cmpxchg (the do_exit path runs in atomic context and cannot sleep). The reaper handles concurrent crashes correctly per Track 3 SC1.

Track 3 DSM (nrs 16-20) layers a migration state machine on top: SUBSCRIBE_MIGRATE / POLL_MIGRATE / START_MIGRATE / ACK_MIGRATE / COMPACT_MIGRATE drive a PROPOSED → MIGRATING → COMMITTED / ABORTED FSM that lets the substrate move a migratable tenant's 8-SM groups to compact the POOL under churn (~1.26 ms per migration, ~70% pool stranding reduction).

The legacy may13 ARBITRATE (`cipher_10ops_impl.cpp:767-814`) is a SHM demand-writer that signals "tenant X needs more SMs" via POSIX shared memory. The "SM rebalance" was a TODO in may13; production rebalance is now in the kmod state machine.

Hot path: no (ALLOCATE / FREE / MIGRATE are cold-path ioctl calls; the per-launch hot path reads the snapshot via ioctl nr 8). The per-group atomic_cmpxchg on FREE is itself lock-free.

### C. Where it fires in production today

**The kmod path fires on every tenant register/teardown** and on every Track 3 DSM event. Plan §1.6 confirms "CP 5.4 isolation 15/15 PASS" verified in the post-Week-1 pre-flight at `cp_5_4/step1_3/cp54_isolation_test`.

**The may13 SHM path never fires in production** because the Stage 2 thread does not spawn. Plan v1.2.1 §1 Tree B2 confirms the kmod allocator supersedes the may13 op.

### D. Integration plan

**Keep in cipher_kmod as-is.** The unified v1 ships the existing 15 × 8-SM ledger + Track 3 DSM. The may13 ARBITRATE op is NOT ported. Plan §3.7 row 11 (Track 2 SC5 supersession analog) extends to ARBITRATE: kmod-resident ARBITRATE supersedes the may13 SHM signal.

Fusion contract:
- Reads from: userspace `CIPHER_CP54_ALLOCATE` call (sm_count, qos hint, migratable flag).
- Writes to: 15-group atomic array + per-PID metadata table; per-tenant `cipher_pid_stats::sm_partition_mask/count` (consumed by every actuator via the snapshot).
- Wave 5 contracts: Cc.7 (CP54 ALLOCATE), Cc.8 (CP54 SUBSCRIBE/POLL/START/ACK/COMPACT), Ce.5 (CP54 lock vs lock-free reaper invariant).

### E. Wave audit findings

- **Wave 3** (KMOD) is the deep-dive for ARBITRATE. The state machine, the two-clause disjointness invariant, the do_exit reaper atomic-context discipline, the 15-not-16 group count are all documented. Wave 3 §FUSION POINTS produced LP-8 (the legacy `cipher_partition_allocator.c` is REFACTOR-RETIRE: userspace nr 9 is dead but state_updater::tick and do_exit::release_slots_only still live; clean retirement scheduled for v1 Week 4).
- **Wave 5** §5.1.c Cc.7 / Cc.8 / §5.1.e Ce.5 / §5.4 LP-8 (allocator retirement) and LP-15 (memory ordering between classifier and CP54).
- No wave-vs-wave disagreement; v1.2.1 §1.3 confirms the kmod-resident ARBITRATE is the production authority.

### F. Risk for integration

**minor.** The op is fully production-grade. The LP-8 cleanup is scheduled for Week 4 (retire the legacy partition_allocator.c userspace path entirely; the kmod-side state_updater::tick + do_exit reclaim continue under their existing lock-free discipline).

---

## Closing table

| Op | Fires in production today | Files | v1 integration approach | Risk |
|---|---|---|---|---|
| 1. CLASSIFY | Comment-only at `cipher_cupti.c:166` (stream classification, not kernel-class); substantive logic stranded in may13 | `cipher-may13-evidence/include/cipher_classify.hpp:94-224` | Port in Week 1 with new `cipher_rt_classify_substrate.cpp` (priority-ordered registry per plan §4.0) | requires-shim |
| 2. ORACLE | Not firing | `cipher-may13-evidence/cipher_oracle.cpp:293-377` | Port in Week 1 as `cipher_rt_oracle.{cpp,h}` | minor |
| 3. SUBSTITUTE (Marlin path) | Fires per `cublasGemmEx` when `CIPHER_MARLIN=on` and gates pass | `cipher_rt_phase4/cipher_rt_marlin_actuator.c` + engine + kernel src | Marlin stays as-is (priority 10 on matmul substrate) | minor |
| 3. SUBSTITUTE (Koopman path) | Never fires (zero substitutions in any production log) | may13 `cipher_dispatch.cpp:200-395` + `cipher_recipes.cpp:346` + `cipher_block_sub_kernel.cu` + `cipher_attn_koopman_kernel.cu` + `cipher_koopman_runtime.cpp` (dead code, zero callers) | NOT ported in v1; v2 work | out-of-v1-scope (v2) |
| 4. COMMIT (= ORCHESTRATE) | Implicit per launch on every matmul / attn substrate return | `cipher_rt_matmul_dispatch.c:100-122` + `cipher_rt_attn_dispatch.cpp:211-242` | Stays as-is | minor |
| 5. SAMPLE (= GENERATE) | Fires for first 500 launches per process (hard-stop at `cipher_intercept_cudart.cpp:2411`), then never | `cipher-may13-evidence/src/cipher_intercept_cudart.cpp:2353-2425` | Not ported in v1 (no v1 consumer); v2 with learning tier | out-of-v1-scope (v2) |
| 6. RING_WRITE | Not firing in deployed runtime (CUPTI path supersedes); ring-write code present but consumer threads never spawn | `cipher-may13-evidence/include/cipher_10ops.h:83-100` + `src/cipher_intercept.cpp:169-184,374-376` | NOT ported in v1; CUPTI path is the production telemetry; ring belongs to v2 learning tier | requires-supersession-decision |
| 7. REMEMBER | Never fires (Stage 1 thread does not spawn) | `cipher-may13-evidence/src/cipher_10ops_impl.cpp:428-458` + `cipher_lnn.cpp` | Not ported in v1; v2 work | out-of-v1-scope (v2) |
| 8. VALIDATE | Never fires; `rs_ok` 3-σ detector instantiated but never called even in may13 | `cipher-may13-evidence/src/cipher_10ops_impl.cpp:460-475` | Not ported in v1; v2 work | out-of-v1-scope (v2) |
| 9. AUDIT | Fires per matmul + per attn call when `CIPHER_AUDIT=on`; default OFF | Production: `cipher_rt_phase4/cipher_rt_audit.c` (222 LOC); may13 `cipher_10ops_impl.cpp:341-378` is commented out | Stays as-is; AUDIT lockless refactor scheduled in plan v1.2.1 §7 Week 4 if N=100 stress shows contention | minor (requires-mitigation at scale) |
| 10. SPECULATE | Never fires (Stage 1 write never happens; Stage 0 check always misses) | `cipher-may13-evidence/src/cipher_10ops_impl.cpp:477-505` (write) + `src/cipher_intercept.cpp:124-149` (check) | Not ported in v1; v2 work | out-of-v1-scope (v2) |
| 11. ADAPT | Never fires; EDMD solver fed degenerate synthetic input | `cipher-may13-evidence/src/cipher_10ops_impl.cpp:597-765` + `cipher_edmd.cpp` + KEN module | Not ported in v1; v2 work | out-of-v1-scope (v2) |
| 12. ARBITRATE | Fires per tenant register / teardown via kmod; Track 3 DSM fires per migration event | Production: `cipher_kmod/cipher_cp54_sched.c` (894 LOC) | Stays as-is in kmod; LP-8 cleanup (retire legacy partition_allocator.c userspace path) in v1 Week 4 | minor |

**v1 ship-set:** 5 of the 12 ops actively fire in the deployed runtime today (Marlin SUBSTITUTE conditional on env, COMMIT implicit, AUDIT conditional on env, ARBITRATE kmod, plus CLASSIFY's stream-classification comment which is structural-only). Week 1 ports CLASSIFY (substantive) and ORACLE, bringing the v1 active set to 7 active hot-path ops (CLASSIFY, ORACLE, SUBSTITUTE-Marlin, COMMIT, AUDIT, ARBITRATE, plus the GENERATE/SAMPLE warmup-only behavior if ported, which is not planned for v1).

**v2 ship-set (deferred):** SUBSTITUTE-Koopman, GENERATE/SAMPLE wiring, RING_WRITE consumer threads, REMEMBER, VALIDATE, SPECULATE, ADAPT. Plan §8 honest-gap-1 names this whole tier as deferred together (the Goal-4 Koopman O(1) substitution): mathematics ships in may13, the runtime is dead code with zero callers, and the dispatch table reserves the lane.

---

## Cross-cutting findings surfaced during this verification

1. **The "all 12 core ops in cipher_10ops_impl.cpp" claim in may13 CLAUDE.md is loose.** Six of the 12 (REMEMBER, VALIDATE, AUDIT-may13-side, SPECULATE-write, ADAPT, ARBITRATE-may13-side) are in that file. CLASSIFY lives in `include/cipher_classify.hpp`. ORACLE lives in `cipher_oracle.cpp`. SUBSTITUTE-Marlin lives in `cipher_rt_phase4/cipher_rt_marlin_actuator.c`. SUBSTITUTE-Koopman lives in `cipher_dispatch.cpp` + `cipher_recipes.cpp` + the .cu files + dead-code `cipher_koopman_runtime.cpp`. COMMIT is a dispatch-return phase, not a function. SAMPLE / GENERATE lives in `cipher_intercept_cudart.cpp`. RING_WRITE is an inline header function plus call sites in `cipher_intercept.cpp`. ARBITRATE-production lives in `cipher_kmod/cipher_cp54_sched.c`. Only the Stage 1 / Stage 2 ops are actually inside `cipher_10ops_impl.cpp`.

2. **The 12-op count is consistent across plan §2.1 and may13 CLAUDE.md** once you treat COMMIT≡ORCHESTRATE as one op and SAMPLE≡GENERATE as one op. The brief's 13 names is correct only if you count both aliases of each. With aliases collapsed, the canonical set is 12.

3. **Of the 12 ops, only 5 to 6 actually fire in production today.** The rest are either stranded in may13 (CLASSIFY, ORACLE, SAMPLE warmup-only, REMEMBER, VALIDATE, SPECULATE, ADAPT) or in dead-code modules (SUBSTITUTE-Koopman). v1 brings the active set to 7 by porting CLASSIFY and ORACLE. v2 brings the full Koopman / learning tier online.

4. **No wave-vs-wave disagreement was surfaced for any of the 12 ops.** The five waves cover the substrate cleanly:
   - **Wave 1** owns CLASSIFY, ORACLE (the classifier brain).
   - **Wave 2** owns SUBSTITUTE-Marlin, COMMIT (the actuator substrate).
   - **Wave 3** owns ARBITRATE, AUDIT-kmod-side (the kmod authority).
   - **Wave 4** owns AUDIT-userspace, the 14 observability ops including SAMPLE / VALIDATE / RING_WRITE consumer side.
   - **Wave 5** ties them together as 43 fusion contracts (12 Ca + 5 Cb + 11 Cc + 8 Cd + 7 Ce) plus 16 lossy points (LP-1 through LP-16, severity rollup 2 blocking / 5 requires-mitigation / 9 minor).

5. **The 16 lossy fusion points touch 7 of the 12 ops.** LP-2 (attn HANDLED-discard) touches SUBSTITUTE-attn future and AUDIT-attn observer; LP-4 (audit lock) touches AUDIT; LP-5 (fairness_shm) touches FAIRNESS (overlay #24, not in this Core 12 set but adjacent); LP-7 (struct collision) touches CLASSIFY and any future op that pulls both `cipher_kernel_table.h` and `cipher_param_recovery.h`; LP-11 (binary confidence) touches CLASSIFY and ORACLE; LP-14 (hot-path budget) touches CLASSIFY; LP-15 (memory ordering) touches CLASSIFY and ARBITRATE. The 12-op port plan covers all of these in either Week 1 (LP-7), Week 2 (LP-2, LP-14 measurement), or v1.5 (LP-4 sharding, LP-5 fairness_shm-to-kmod migration).

---

**End of CORE_12_OP_VERIFICATION.md.**
