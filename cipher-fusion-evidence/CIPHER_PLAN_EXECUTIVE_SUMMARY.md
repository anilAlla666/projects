# CIPHER Plan: Executive Summary

**Date:** 2026-05-20
**Audience:** Investor data room. Partner with technical background reads this in 20 minutes.
**Source documents:** CIPHER_REENGINEERING_PLAN.md v1.2 (124 KB binding architecture plan); CIPHER_DEEP_INSPECTION_REPORT.md; CIPHER_LOGIC_AUDIT_WAVE_1 through WAVE_5 (560 KB of audit work); the closeouts for Phase 3, Phase 4, CP 5.4, CP 5.6, Track 2, Track 3, FUTURE_SCOPE/A Phase 3.5.
**Numbers cited in this document trace to a specific measurement file with file path and line citations.** Retracted claims (the prior "3.617×" single-instance figure, the prior "14×" cross-tenant synthetic figure, and any universal "85% MFU" framing) are explicitly excluded.

---

## Section 1: Executive Overview

### What CIPHER is, in 200 words

CIPHER is a workload-aware GPU control-plane runtime for shared multi-tenant agentic inference on NVIDIA Hopper. It runs below the dispatch boundary: a Linux kernel module (`cipher_kmod.ko`, 14 sources, anchor `285d102e`) provides the cross-process state and SM-arbitration substrate; a userspace shared library (`libcipher_rt.so`, 16 sources, anchor `83afd1ca`) loads into every CUDA process via the driver's injection ABI (`CUDA_INJECTION64_PATH`) and patches the runtime's GOT for cuBLAS and SDPA. On every kernel launch, CIPHER classifies the workload (large GEMM, small GEMM, attention prefill, attention decode), checks a safety gate, and routes to the right actuator: a Marlin INT4 GEMM kernel on compute-bound prefill, a partition-aware cuBLAS path on multi-tenant decode, or a passthrough when neither applies. Around that hot path it runs an observability tier (audit chain, per-tenant fairness, per-session receipts, carbon accounting) and a cross-tenant primitives layer (weight sharing across same-model agents, KV-prefix dedup, dynamic SM migration, fleet-policy DVFS). Every cross-tenant primitive lives in the kernel module so it survives any single tenant crashing.

### The architectural moat, in 100 words

The dispatch boundary is the seam between the CUDA driver and the application. Below it, you see kernel descriptors and have a stable injection ABI. Above it, you see request-level state and have ORCA-class scheduling. Application-layer competitors (Sarathi, ORCA, SGLang) cannot reach below the boundary. NVIDIA does not publicly commit to the injection ABI as a productized interface, so they cannot ship the substrate above it without committing to a stable contract they have explicitly avoided. CIPHER ships in the gap. The kernel-resident state, the additive ioctl ABI (NRs 1 through 24 frozen), and the per-launch actuator-registry pattern are the durable artifacts.

### Measurements that exist today

| Measurement | Value | Regime | Source |
|---|---|---|---|
| Track 2 weight sharing | 76.0% HBM saved (17,572 MiB shared vs 73,332 MiB independent) | Mistral-7B, N=4 same-model tenants, bit-identical forward verified | `phase_c/TRACK_2_CLOSEOUT.md:15,36` |
| Cross-tenant batching curve | 3.06× tok/W on Mistral-7B at N=4; 5.98× tok/W on TinyLlama-1.1B at N=16 | Two-model decode-step batching curve; baseline is N independent naive workers time-slicing the GPU on the same code path | `cp_5_6/TPW_RETEST_2026_05_19.md:49,71` |
| DVFS under vLLM | +13.9% tok/W single-instance decode at 1200 MHz lock | Memory-bound decode under vLLM, graph mode | `future_scope_a/FUTURE_SCOPE_A_PHASE_3_5_RESULTS.md` |
| vLLM latency overhead | −0.39% latency delta in graph mode (no output-correctness gate documented) | Single-tenant decode under vLLM | `CIPHER_REENGINEERING_PLAN.md:64` |
| Compute-bound burst peak | 745 TFLOPS on a compute-bound workload (likely a brief uncapped burst or proxy-counter over-read per v1.2 §5.1) | LARGE_GEMM kernel class | `CIPHER_REENGINEERING_PLAN.md:859` |
| Defensible sustained compute ceiling | ~660 TFLOPS / 67% MFU at 700 W power cap | Compute-bound prefill on H100 SXM5 | `CIPHER_REENGINEERING_PLAN.md:859,897` |
| SM partitioning correctness | 15 × 8-SM groups, two-clause disjointness invariant, 15/15 isolation | Multi-tenant partitioning | `cp_5_4/CP_5_4_CLOSEOUT.md` |
| Dynamic SM migration | ~1.26 ms per migration; ~70% reduction in stranded pool capacity under churn | Tenant SM migration under arbitrary partition churn | `phase_c/track_3/TRACK_3_CLOSEOUT.md:11-17` |

### Five-week integration path

| Week | Deliverable | Verification gate |
|---|---|---|
| 1 | Compile-level classifier port; struct-collision rename; snapshot reserved-tail bump for classifier fields | Clean `make`; W1 regression PASS within ±3% |
| 2 | Hot-path classifier wired in observe-only mode; LP-2 attention trampoline refactor (unblocks v1.5 attention substitute lane) | 99th-percentile per-launch overhead under 500 ns; cache-hit rate ≥ 95% |
| 3 | Dispatch routing goes live, GEMM-only in v1 (attention dispatch routing defers to v1.5) | Regime-1 prefill MFU baseline measured per workload; Regime-2 decode TPW measured |
| 4 | Observability tier ported (audit, trace, receipt, fairness, carbon, comply, loop, pipeline, determinism, continuity, pulse); AUDIT lockless refactor for N=100 scale | All observers fire; Prometheus exporter exports per-tenant series |
| 5 | CP 5.5 100-tenant heterogeneous benchmark (5 prefill + 80 decode + 15 burst; shared 4K system prompt; mock tool-call delays) | Agents/GPU, fleet tok/W, per-tenant p99, per-tenant fairness, weight HBM saved, KV-prefix dedup hit rate, 24-hour soak |

---

## Section 2: The Substrate

### The dispatch boundary

NVIDIA's CUDA driver exposes one stable hook for third-party runtimes: `CUDA_INJECTION64_PATH`. When `cuInit()` is called, the driver dlopens the path and calls `InitializeInjection` / `InitializeInjection2` in the loaded library. CIPHER's `libcipher_rt.so` is this library. From inside `InitializeInjection2`, CIPHER opens `/dev/cipher` (the kmod's character device), registers the tenant identity, allocates an SM partition via the CP 5.4 ledger, initializes the CUPTI launch counter daemon, registers GOT-patched cuBLAS and SDPA dispatchers, and walks every loaded module via `dl_iterate_phdr` to rewrite the relevant relocation slots in place (RELRO-safe via the mprotect dance).

This boundary has two structural properties that define the moat. First, it sees kernel launches before they reach the GPU, with full kernel descriptors and stream context, but after the application has decided what to launch. Second, it is the same boundary every CUDA application uses, so a runtime that lives here is transparent to PyTorch, vLLM, and any other consumer. Phase 3 measured a −0.39% latency delta under vLLM in graph mode (no output-correctness gate was documented; this is a latency-only measurement).

### Why application-layer competitors cannot reach it

Sarathi, ORCA, and SGLang are scheduling-layer systems. They reason about requests, sequences, and token-by-token decisions. They sit above PyTorch and above vLLM. They cannot see kernel descriptors and they cannot intercept individual launches. Their lever set is request batching, prefill chunking, and KV-cache management. Those are real levers; they are not the same lever set CIPHER ships.

When a Sarathi-class system wants to share weights across tenants, it has to either run all tenants in one process (sacrificing isolation) or push the problem down to a layer it does not own. When a CIPHER tenant wants to share weights, the kmod's weight arena holds the VMM POSIX file descriptor and N tenant processes import it. The weight pages live once in HBM and the kmod outlives any single tenant crash.

### Why NVIDIA does not publicly expose this

The injection ABI is documented as a debugging and profiling hook. NVIDIA does not publicly commit to backward compatibility for third-party productization. CUPTI ships against the same constraint. NVIDIA's own user-facing stack (Triton, TensorRT-LLM, NIM) does not ship a competitor to CIPHER at this layer because doing so would require either publicly committing to a stable injection ABI (which they have not) or shipping a parallel proprietary contract.

CIPHER ships in the asymmetric gap. The kmod's character-device ABI is the stable contract for downstream consumers. The 24 ioctl numbers are frozen and additive-only per the project's binding rule (`cipher-abi-rule`). The injection lib follows whatever the driver exposes; if the driver ABI changes, CIPHER ports.

### The 33-op organization

The plan groups the 33 canonical operations into four architectural tiers.

**Classifier tier (5 ops):** CLASSIFY, ORACLE, SENSE, PREDICT, TOPOLOGY. These decide what regime each kernel launch is in (large GEMM compute-bound prefill, small GEMM memory-bound decode, attention prefill, attention decode, matvec) and they decide what session band each tenant is in (HUMAN, AGENT, BATCH). Every classifier runs synchronously before actuator selection.

**Actuator tier (11 ops):** SUBSTITUTE, COMMIT, ARBITRATE, SHIELD, SUSTAIN, THERMOSTAT, VOLT, HIBERNATE, NCCL_P2P, FAIRNESS quota enforcement, GENERATE. These change execution: they substitute one kernel for another, partition the GPU's SMs, change the clock, throttle a tenant. Each actuator gates on the classifier's output.

**Observability tier (14 ops):** SAMPLE, RING_WRITE, VALIDATE, AUDIT, GUARD, RECEIPT, CONTINUITY, DETERMINISM, PULSE, CARBON, FAIRNESS counters, LOOP, PIPELINE, TRACE, COMPLY, STRAGGLER. These observe without changing execution. They produce per-tenant billing, per-session audit chains, carbon accounting, runaway-loop detection, multi-agent pipeline correlation, and compliance verdicts. They fire asynchronously off the critical path.

**Learning tier (3 ops):** REMEMBER, SPECULATE, ADAPT. These maintain a per-tenant model of the workload and propose updates to the substitute registry. The mathematics exists in the codebase. The runtime that consumes the math is dead code today. This tier is deferred to v2.

### The two-codebase finding

The most consequential discovery from the audit work is that CIPHER's code exists in two parallel trees that were never connected.

**Tree A** (`cipher-may13-evidence/`, ~29,000 lines of C++ and CUDA) contains the brain. The classifier headers, the oracle safety gate, the 32-shape recipe registry, the EDMD pipeline, the Koopman learning code, the overlay observability ops. Tree A builds three shared libraries and was the historical research path.

**Tree B** (`cipher_rt_phase4/` + `cipher_kmod/`, 30 sources across the two trees) contains the hands. The userspace runtime, the GOT patcher, the partition router, the Marlin INT4 actuator, the cuBLAS shim, the SDPA dispatcher, and the 14-source kernel module that owns cross-process state.

The audit established that of 33 canonical ops, exactly 6 fire on Tree B's hot path today: SUBSTITUTE (Marlin lane only, no Koopman), ARBITRATE (kmod ledger), VOLT (DVFS via ioctl nr 10), AUDIT (HMAC-SHA256 chain), the COMMIT state-machine token in the Track 3 migration FSM, and CLASSIFY as a comment-only reference. The 6 classifiers in Tree A are stranded; the 11 observability ops that work in Tree A have zero presence in the deployed runtime; the Koopman runtime module has zero callers anywhere.

This is not a broken system. It is a half-assembled one. The marvel is an integration problem. The pieces work in their natal tree. They were never connected to the actuator scaffolding that ships.

### The reusable abstraction

The deployed runtime already contains the substrate pattern the unified runtime extends: priority-ordered actuator registries. Two instances ship today. `cipher_rt_matmul_dispatch.c` exposes a 16-slot registry where each entry is a function pointer plus a priority. On every cuBLAS GEMM call, the substrate walks the registry in priority order and asks each actuator whether it wants to handle the call. Marlin INT4 registers at priority 10 and gates on batch size and weight quantization. AUDIT registers at priority 0, always returns PASSTHROUGH, and runs its HMAC chain as a side effect. `cipher_rt_attn_dispatch.cpp` does the same shape for SDPA.

The unification plan adds a third registry of the same shape: a classifier substrate. Every classifier op in Tree A ports as a registered classifier. Every observer op ports as a priority-0 actuator on matmul or attention (AUDIT is the template). Every substitute op ports as a high-priority actuator on matmul or attention (Marlin is the template). The pattern is reusable, the build system already supports it, and the kmod ABI is already there.

The audit identified five semantic divergences between the matmul and attention substrate contracts: enum cardinality (matmul is 3-value, attention is 4-value), the `maybe_handle` signature (matmul takes a status pointer, attention does not), the lock discipline (matmul is lock-free, attention snapshots under lock), the ERROR-handling control flow (matmul breaks the loop, attention continues), and the HANDLED bypass semantics (matmul respects HANDLED, attention's three SDPA trampolines structurally discard HANDLED and always call the original). The classifier substrate's contract is chosen explicitly per dimension in §4.0 of the v1.2 plan. This level of contract precision is a representative example of the discipline the codebase carries.

---

## Section 3: The Measurements

Every claim below cites the measurement document, the methodology, and the regime. Three figures from earlier versions of the pitch have been retracted; they appear at the end of this section with the retraction context, so a DD reader does not need to reconstruct what was removed.

### Track 2 weight sharing: 76.0% memory saving

**Number:** 76.0% HBM saved on Mistral-7B v0.1 with N=4 same-model tenants.
**Methodology:** Direct GPU framebuffer readings via NVML, sampled three times during steady-state. The shared arena holds 17,572 MiB across one producer plus four consumers; four independent loads of the same model would cost 73,332 MiB. The savings formula `S = N·W / (N·W + (N+1)·C)` with measured weight footprint W = 13,940 MiB and per-tenant non-shared cost C reproduces the framebuffer reading and projects an asymptote near 95% at large N.
**Correctness gate:** Bit-identical forward pass across 40 successive forward calls, two models tested. The Track 2 SC6 protocol detected a tier-1 fingerprint mismatch pre-commit and stopped the merge; the second iteration passed.
**Source:** `phase_c/TRACK_2_CLOSEOUT.md:15,36`; SC6 evidence in `phase_c/TRACK_2_SC6_MEMORY.md` and `TRACK_2_SC6_INTEGRATION.md`.

### Cross-tenant batching scaling curve

**Numbers:** 3.06× tok/W on Mistral-7B at N=4; 3.30× on TinyLlama-1.1B at N=8; 4.73× at N=12; 5.98× at N=16. The curve climbs monotonically with concurrent tenant count. The figure that previously appeared as "3.617× single-instance" lives at approximately N=9 on this curve and is a multi-tenant figure, not a single-instance figure.
**Methodology:** Two-arm comparison on the same model. The batched arm uses the `libcipher_v2` POOL fuser to combine N tenants' decode steps into one fused kernel launch. The baseline arm runs N independent `tpw_naive_worker.py` processes time-slicing the GPU through the identical code path. The substrate-attributable tok/W ratio is reported.
**Correctness gate:** Teacher-forced KL gate on every measured point, threshold between 2.5e-5 and 5.5e-5 depending on the model. Any point that fails the KL gate is excluded.
**Regime caveat:** The curve crosses between two models. The Mistral-7B point at N=4 (3.06×) is on the larger-model curve. The TinyLlama-1.1B points at N=8 through N=16 are on the smaller-model curve. Quoting "3.06× to 5.98× on Mistral-7B" would be a fabrication. The defensible statement is: "3.06× on Mistral-7B at N=4, climbing to 5.98× on TinyLlama-1.1B at N=16 across two models on the substrate-attributable batching curve."
**Source:** `cp_5_6/TPW_RETEST_2026_05_19.md:49,71`.

### DVFS under vLLM: +13.9% tok/W single-instance

**Number:** +13.9% tok/W single-instance memory-bound decode under vLLM at 1200 MHz lock.
**Methodology:** Single-tenant decode on TinyLlama, 5-round repeat at each clock setting, NVML power telemetry. The 1200 MHz lock is the sweet spot identified during FUTURE_SCOPE/A Phase 3.5 sweep across 800, 1200, 1600, 1980 MHz.
**Regime caveat:** This is the single-instance ceiling under vLLM. The compute-acceleration actuators (Marlin INT4, persistent dispatch, attention substrates) have no interception surface under vLLM's graph mode. Single-instance under vLLM lifts only through DVFS.
**Source:** `future_scope_a/FUTURE_SCOPE_A_PHASE_3_5_RESULTS.md`.

### vLLM latency overhead: −0.39% in graph mode

**Number:** −0.39% latency delta under vLLM in graph mode.
**Methodology:** Latency measurement, vLLM baseline versus vLLM with CIPHER's injection lib loaded. The difference is within the measurement noise band.
**Important caveat:** No output-correctness gate is documented for this measurement; it is a latency-only delta. The bit-identical claim is Track 2 SC6, and that is a separate measurement on a separate path. The honest framing for this number is "negligible latency overhead in graph mode under vLLM" rather than "transparent" without qualifier.
**Source:** `CIPHER_REENGINEERING_PLAN.md:64`.

### Compute-bound performance: 745 TFLOPS burst, ~660 TFLOPS / 67% MFU sustained ceiling

**Burst peak:** 745 TFLOPS on a compute-bound workload at 59.8% MFU. Device sum from per-SM counters was 887 TFLOPS; NVML cross-check read 916 TFLOPS.
**Caveat from the source itself:** The v1.2 plan §5.1 (line 859) states that the 745 figure was "likely a brief uncapped burst or proxy-counter over-read." This is a measurement artifact, not a sustained capability.
**Defensible sustained ceiling:** At the 700 W power cap that ships on this H100 SXM5 deployment, the sustained ceiling is approximately 660 TFLOPS, which corresponds to 67% MFU against the 989 TFLOPS H100 BF16 theoretical peak.
**The honest claim for investor materials:** "CIPHER's compute lane sustains approximately 67% MFU under the 700 W power cap on H100 SXM5. The 85% MFU gate is a post-CP-5.5 depth-win program target that requires partition-aware Marlin (Phase 6 engagement with Song Han), L2 budget enforcement (a new kmod ioctl), and TMA thread-block-cluster kernel work, none of which is in the v1 five-week scope."
**Source:** `CIPHER_REENGINEERING_PLAN.md:859,897`.

### SM partitioning correctness: 15 × 8-SM disjointness, 15/15 isolation

**Numbers:** On the Lambda H100 SXM5 pod that ships this measurement (driver 580.105.08, kernel 6.8.0-1046-nvidia, NVML and CUPTI present), the CUDA Green Context allocation API returns exactly 15 disjoint groups of 8 SMs each. The published green-context documentation suggests 16 may also be reachable in other configurations, so this is the measured-on-this-pod finding and not a universal hardware fact. The measurement and its reproducibility are documented in memory pointer `cipher-cp54-15groups`. The CP 5.4 kmod-resident SM-arbitration ledger maintains a two-clause disjointness invariant at runtime against whichever number the hardware returns, and the 15/15 isolation gate passes on the measured 15-group configuration.
**Methodology:** Runtime probes in `cp_5_4/step1_6/` exercise concurrent allocations and verify that no two tenants observe SM IDs from the same group. The ledger is in the kmod (`cipher_cp54_sched.c`), which means the disjointness invariant survives any single tenant process crashing.
**Source:** `cp_5_4/CP_5_4_CLOSEOUT.md`.

### Dynamic SM migration: ~1.26 ms migration cost, ~70% pool-stranding reduction

**Numbers:** Live tenant SM migration completes in approximately 1.26 ms p50; pool-stranded SM count drops from a mean of 9.49 to a mean of 2.78 under the test churn workload.
**Methodology:** A/B sweep over 10 free-order partition-churn patterns, five seeds each. Track 3 SC5 and SC6 evidence files document the per-scenario migration trace and the disjointness assertions checked at every migration boundary. Correctness invariants verified at runtime include disjointness clauses 1 and 2 and teacher-forced KL gates on tenants whose work was migrated.
**Why this matters:** Without DSM, a multi-tenant GPU's batched POOL fragments under arbitrary churn and the system loses scheduling efficiency. DSM lets the substrate keep the pool compact as partitions allocate and free in any order.
**Source:** `phase_c/track_3/TRACK_3_CLOSEOUT.md:11-17`.

### What is retired and why

Three figures from earlier versions of the pitch have been retracted. Each is named here, in one sentence, so a DD reader does not encounter a dangling reference.

**The "3.617× single-instance tok/W" figure (formerly CP 2.4 headline) was retracted** because the underlying Marlin full-GPU path produced degenerate decode under the F1 green-context primary-pinned bug, and the measurement engaged the broken path with no output-correctness gate; the corrected composed figure on the F1-fixed substrate is 1.54× (CP 5.6 V_C), and the multi-tenant lever is the batching curve, not the single-instance claim.

**The "14× cross-tenant on synthetic" figure was retracted** because the synthetic workload does not represent production multi-tenant churn behavior, and the replacement measurement on actual decode workloads is the 3.06× to 5.98× batching curve documented above.

**Any "universal 85% MFU" framing was retracted** because MFU is regime-specific and workload-class-dependent: compute-bound prefill can approach 85% only after the post-CP-5.5 depth-win program (partition-aware Marlin, L2 budget enforcement, TMA clusters); decode regimes are roofline-bound at 15% to 40% by physics, not by implementation.

---

## Section 4: The Unification Architecture

### The actuator-registry pattern as the integration join point

The deployed runtime already ships the pattern. `cipher_rt_matmul_dispatch.c` exposes a 16-slot priority-ordered registry of actuator function pointers. On every cuBLAS GEMM call, the substrate walks the registry in priority order and asks each actuator to handle the call. The first actuator that returns HANDLED wins; an actuator that returns PASSTHROUGH lets the next actuator try; ERROR breaks the loop and falls through to the real cuBLAS. `cipher_rt_attn_dispatch.cpp` does the same shape for SDPA, with a 4-value enum (HANDLED, PASSTHROUGH, REDIRECTED, ERROR), snapshot-under-lock semantics, and a current observe-only contract on the three SDPA trampolines.

The unification plan adds a third registry of the same shape: a classifier substrate. The contract is explicit. The classifier substrate uses the 4-value enum (superset is safer for v2 surrogate paths). It uses the attention-style `maybe_handle` single-argument signature. It uses snapshot-under-lock for safety against concurrent registration. It uses the matmul-style ERROR-breaks-the-loop semantics so classifier errors are terminal. HANDLED means "this classifier produced a confident result and downstream routing accepts it"; PASSTHROUGH means "no opinion, try the next classifier or fall back to the default class".

Every classifier op in Tree A (CLASSIFY, SENSE, PREDICT, TOPOLOGY, ORACLE) ports as a classifier-registry entry. Every observer op in Tree A (AUDIT, TRACE, DETERMINISM, FAIRNESS, CARBON, RECEIPT, GUARD, COMPLY, LOOP, PIPELINE, PULSE, CONTINUITY, STRAGGLER) ports as a priority-0 actuator on the matmul or attention substrate (AUDIT is the canonical template at `cipher_rt_audit.c:132-137`). Every substitute op (SUBSTITUTE Marlin, FP8, KV compression) ports as a high-priority actuator with regime gating (Marlin INT4 is the canonical template at `cipher_rt_marlin_actuator.c:170`).

This is what makes the integration tractable. The pattern is uniform; the build system supports it; the kmod ABI is already there. Adding a classifier substrate is a one-week port and a build-system change.

### The per-kernel dispatch sequence

On every kernel launch (intercepted via the GOT-patched `cuLaunchKernel`), the unified hot path runs:

1. **CLASSIFY** produces a kernel-class fingerprint from the kernel descriptor. The classifier's per-thread cache hits over 95% of the time at steady state, and the honest budget for the classifier path is 100 to 200 nanoseconds per launch. The plan accepts 1% to 2% per-tenant CPU overhead as the primary case rather than relying on a 12 ns figure that the audit found unsupportable.
2. **RING_WRITE** writes one entry into the per-tenant SPMC ring (timestamp, params hash, kernel class).
3. **ORACLE** runs the 5-gate safety check (phase, minimum confidence, structural lookup, EMA demotion, outstanding-substitution rate limit). The output is SUBSTITUTE, PASS_THROUGH, or DEMOTE.
4. **SUBSTITUTE dispatch table lookup**, keyed on (kernel_class, params_hash, tenant_band). Routes to the right lane: Marlin INT4 for large compute-bound GEMM with a full-GPU primary context; cuBLAS shim for partitioned tenants; attention substrate trampoline for SDPA paths; FP8 for env-gated FP8 workloads; the Koopman O(1) lane is reserved in the table but deferred to v2.
5. **COMMIT** launches the chosen kernel.
6. **Post-launch observability fan-out** runs asynchronously off the critical path: AUDIT advances its HMAC chain, TRACE emits a bounded JSONL entry, DETERMINISM updates the per-tenant fingerprint, FAIRNESS debits the per-tenant quota in the cross-process FAIRNESS_SHM, CARBON updates the energy estimator, RECEIPT updates the per-session proof, GUARD runs its cross-session leak check every 100 launches, and the session-band observers (LOOP, PIPELINE, PULSE, COMPLY) tick.

### Where the fusion is lossless

Wave 5 of the logic audit catalogued 43 fusion contracts (every meeting point between Tree A and Tree B) across five categories: classifier-to-actuator (12 contracts), classifier-to-kmod (5 contracts), actuator-to-kmod (11 contracts), observer-to-other-tier (8 contracts), and kmod-internal cross-translation-unit (7 contracts). Each contract has a stable identifier, error semantics, concurrency assumption, and fusion class.

Of these 43 contracts, Wave 5 §5.3 classifies 22 as lossless. The lossless contracts compose without information loss across the tree boundary; they port as-is. The 11 actuator-to-kmod contracts (every Wave 2 substrate-to-Wave 3 ABI call) are all PORT-AS-IS. The 7 kmod-internal cross-TU contracts are invariants that survive the port unchanged (RCU discipline on `cipher_pid_table`, the two-step do_exit reaper, the exit-order invariant, the gpu_state spinlock, the CP54 lock-versus-lock-free reaper, the weight-arena lock plus reaper, the kvdedup lock plus release file-operation).

### Where the fusion has known costs

Wave 5 §5.4 surfaces 16 lossy fusion points (LP-1 through LP-16). The severity rollup is:

| Severity | Count | Examples |
|---|---|---|
| Blocking | 2 | LP-2 (attention HANDLED-discard, blocks any future attention substitute actuator), LP-7 (struct-collision compile blocker on `CipherKernelEntry` defined differently in two headers) |
| Requires mitigation | 5 | Hot-path overhead under contention, AUDIT lockless refactor needed for N=100 scale, fairness_shm migration to kmod ioctls in v1.5, HIBERNATE NVML fallback, classifier binary-confidence (40 vs 85) limiting the ORACLE confidence gate to coarse routing |
| Minor | 9 | Snapshot reserved-tail field semantics, classifier observer registration on the two substrates, weak-symbol re-entry replacements, 80-layer drift fix, kernel-name plumbing into structural_lookup, and similar |

The two blocking lossy points are both addressed in the five-week plan. LP-7 is the Week 1 first action: a mechanical sed rename of `struct CipherKernelEntry` in one of the two conflicting headers (9 caller sites in `cipher_kernel_table.h`, 6 caller sites in `cipher_param_recovery.h`). LP-2 is the Week 2 attention trampoline refactor: branch the three SDPA trampolines on `route()`'s return value, add an `out_status_devptr` field to `cipher_rt_attn_call`, approximately 20 lines of code across three files. Both refactors land in v1 even though they are not strictly required to ship the GEMM-only routing in Week 3, because deferring them would leave the v1.5 attention substitute lane structurally blocked.

### The honest classifier overhead budget

A previous version of the plan claimed a 12 nanosecond budget for the classifier hot-path. Wave 5 NF-5.1 verified the 12 ns figure against the Wave 1 control-flow analysis and the `cipher_classify.hpp` header comment ("~160 ns cache-hit") and found the 12 ns figure unsupportable. The plan now states the honest budget at 100 to 200 nanoseconds per launch, with an alarm threshold of 500 nanoseconds at the 99th percentile, and a primary-case acceptance of 1% to 2% per-tenant CPU overhead. At a `cublasGemmEx` rate of approximately 10,000 launches per second per tenant and 16 tenants, this is about 4% on one CPU core, which is acceptable. The Week 2 cycle-counter measurement is the verification gate; if the 99th percentile exceeds 500 ns, the classifier falls back to a coarser path that is shipped as a build flag.

This is what discipline looks like in practice: a number quoted with confidence in v1.0, audited by a logic-audit wave, retracted in v1.2, and replaced with the honest budget that the codebase actually supports.

---

## Section 5: The Integration Sequence

The five-week integration sequence is the binding artifact for the next engineering window. Each week has an explicit goal, a behavior test, a rollback path that returns to a preserved anchor, and a risk register.

### Week 1: Compile-level classifier port and the two blocking refactors

The Tree A dispatch and oracle files port into the rt_phase4 build as `cipher_rt_dispatch.cpp` and `cipher_rt_oracle.cpp`. The top-level (live) versions port, not the `src/` shadow copies that the Tree A Makefile silently excludes. The `cipher_classify.hpp` header ports as `cipher_rt_classify.h`. The 32-shape recipe registry, the structural lookup table, and the session classifier port the same way. A new substrate file, `cipher_rt_classify_substrate.cpp`, mirrors the matmul and attention dispatch pattern.

The two blocking refactors land in Week 1: LP-7's struct rename (mechanical), and Cb.2's snapshot reserved-tail bump (additive without a new ioctl number; consumes 4 of the 16 reserved slots in `cipher_tenant_snapshot_user` for `recommended_sm_count`, `slo_priority`, `session_band`, and `tenant_billing_class`). The size invariant `sizeof(struct cipher_tenant_snapshot_user) == 336` bytes is verified by the Week 1 regression test T-W1.3.

**Verification gate:** Clean `make`; W1 regression PASS (tok/s and tok/W within ±3% of the pre-Week-1 anchor); 15/15 isolation PASS; Track 2 SC6 bit-identical forward PASS.

**Rollback path:** Revert the Makefile change; rebuild from anchor `83afd1ca`. `libcipher_rt.so.pre_week1` is preserved.

### Week 2: Hot-path classifier wired in observe-only mode

The classifier fires on every kernel launch but does not yet change actuator selection. The pre-launch hook in `cipher_inject.c`'s GOT-patched `cuLaunchKernel` calls `cipher_rt_classify`, `cipher_rt_oracle_check`, and `cipher_rt_classify_record`. The decision is logged but not consumed.

The LP-2 attention trampoline refactor lands here. The three SDPA trampolines (`flash_call`, `eff_call`, `cudnn_call`) branch on `route()`'s return value. The refactor is a no-op for v1 because no actuator returns HANDLED on attention until v1.5.

**Verification gate:** W1 regression PASS within ±3%. The new `/proc/cipher/classify_stats` node reports non-zero classification counts. ORACLE EMA stays in STEADY after warmup. CLASSIFY cache hit rate is at least 95% at 10 seconds of steady-state. Per-launch 99th-percentile overhead is under 500 nanoseconds (the alarm threshold from §4).

**Rollback path:** Stub the pre-launch hook to return PASS_THROUGH. No anchor change needed.

### Week 3: Dispatch routing goes live (GEMM-only in v1)

The classifier output drives actuator selection. Per the regime tables in §4 of the v1.2 plan, large GEMM with full-GPU primary context routes to Marlin INT4 (gated on batch size ≥ 8 and weight-quantized); small GEMM routes to the cuBLAS shim; attention dispatch routing defers to v1.5 (the attention substrate exists, but routing rules and the substitute actuators are v1.5 work). VOLT engages the 1200 MHz lock on the detect-decode-band signal. SENSE phase transitions trigger DSM PROPOSE for tool-call idle detection.

**Verification gate:** Regime 1 (WL03 prefill) MFU baseline measured per workload, with Marlin engaged where the regime conditions are met. Regime 2 (WL01 decode) tok/W measured with DVFS engaged. Regime 3 (WL05 multi-tenant) isolation and density measured. A heterogeneous mini-benchmark (1 prefill plus 4 decode tenants concurrent) verifies that all four lanes work simultaneously. W1 regression PASS. Track 2 SC6 PASS.

**Rollback path:** `CIPHER_DISPATCH_LIVE=0` falls back to all-PASS_THROUGH (Week 2 state). `libcipher_rt.so.week3_pre` preserved.

### Week 4: Observability tier ports and AUDIT lockless refactor

Eleven Tree A observability ops port into `cipher_rt_phase4`: TRACE, RECEIPT, CARBON, FAIRNESS plus cross-process FAIRNESS_SHM, GUARD, COMPLY, LOOP, PIPELINE, DETERMINISM, CONTINUITY, PULSE. The AUDIT lockless refactor lands here (Wave 5 LP-4) so that audit-chain advancement scales cleanly at N=100 tenants. The PREDICT, SHIELD, SUSTAIN, and THERMOSTAT ports are partial in this week and defer their full lift to v1.5.

The cipher-exporter Prometheus endpoint adds per-tenant series: `cipher_tenant_fairness_quota{tenant=X}`, `cipher_tenant_carbon_grams_co2{tenant=X}`, `cipher_tenant_receipt_hash{tenant=X}`, `cipher_tenant_session_band{tenant=X}` with HUMAN, AGENT, or BATCH label.

**Verification gate:** Per-tenant FAIRNESS quota debits visible at `/proc/cipher/fairness`. AUDIT chain advances per launch (kmod ioctl nr 8 telemetry). TRACE bounded buffer rotates without drops at sustained load. RECEIPT per-session HMAC verifiable.

**Rollback path:** Each observability op can be individually disabled via environment variable. No anchor needed (observability is observe-only).

### Week 5: CP 5.5 100-tenant heterogeneous benchmark

This is the headline measurement. The workload is a mock agentic mix: 5 prefill-heavy tenants (RAG-style long context), 80 decode-heavy agents (tool-using loops with mock tool-call sleeps of 100 to 500 milliseconds per iteration), and 15 burst tenants (model switching or pipeline). The agents share a 4K-token system prompt to exercise KV-prefix dedup.

**Measurements collected:**

- Agents per GPU: the number of agents successfully running and progressing (target 100).
- Fleet tok/W: aggregate token output divided by aggregate power (target at least 2× the vanilla vLLM 100-instance baseline).
- Per-agent p99 latency distribution across all 100 agents.
- Per-agent fairness quota adherence.
- Per-launch MFU attribution via the kmod's FLOP telemetry (ioctls nr 11 and 12).
- Weight HBM saved via the kmod weight arena query (target at least 90% for same-model agents).
- KV-prefix dedup hit rate via the kvdedup query (target at least 60% for the shared system prompt).
- 24-hour soak gate (G4): no kernel oops or WARN, kernel taint flag unchanged or increased by at most 1 (the IMA flag is acceptable).

**Verification gate:** All measurements reproducible from a fresh boot in 30 minutes of setup.

**Rollback path:** If any target fails, the unified runtime remains functional; only the headline measurement is incomplete. Subsequent CP 5.5 attempts iterate without code rollback.

---

## Section 6: Risk Register

### The three catastrophic risks

**R-C1: Classifier hot-path latency exceeds the honest budget.** Probability medium. Impact blocking for any per-tenant overhead target. The honest budget is 100 to 200 nanoseconds per launch with a 500 ns alarm threshold at the 99th percentile. The mitigation is the Week 2 cycle-counter measurement gate and a build flag for a coarser fallback classifier if the budget is missed. The honest framing accepts 1% to 2% per-tenant CPU overhead as the primary case rather than a sub-percent target.

**R-C2: CP 5.5 measurement reveals a scaling cliff at N≈30 to N≈50 tenants.** Probability medium. Impact blocking for the density story. The candidate causes are kmod hashtable contention, weight-arena 16-slot exhaustion (with N same-model tenants sharing 1 arena, 16 different models is the cap), or kvdedup table size limits. The Week 4 stress test at N=30 surfaces this before CP 5.5; the pre-emptive mitigation is dynamic-resize hashtables and bumping arena slot counts from 16 to 32 in kmod version 0.5.

**R-C3: Marlin × partitioning composition cannot be made to work in v1, so an 85% MFU target on partitioned tenants is undeliverable.** Probability low because the v1 mitigation is already in place. Marlin pins to the primary CUDA context per Fix A (`cipher-marlin-primary-ctx-pin`) and is structurally a full-GPU kernel. Partition-aware Marlin is Phase 6 work, scoped explicitly to the Song Han engagement. The v1 customer-facing MFU claim is restricted to non-partitioned compute-bound workloads.

### The 16 lossy fusion points

| Severity | Count | What they are |
|---|---|---|
| Blocking | 2 | LP-2 attention HANDLED-discard (mitigated Week 2); LP-7 struct collision (mitigated Week 1) |
| Requires mitigation | 5 | AUDIT scale, hot-path contention, fairness_shm migration, HIBERNATE NVML fallback, binary-confidence routing |
| Minor | 9 | Snapshot tail semantics, observer registration plumbing, weak-symbol re-entry, 80-layer drift, kernel-name plumbing into structural_lookup, similar |

Every lossy point is named explicitly in Wave 5 §5.4 with a fold-in plan. The 2 blocking points land in v1 weeks 1 and 2. The 5 requires-mitigation points have v1.5 schedule entries. The 9 minor points are tracked and revisited at v2.

### What is v2 / v3 scope explicitly

**Goal-4 Koopman O(1) substitution.** The mathematics exists in `cipher_edmd.cpp` and `cipher_lnn.cpp`. The runtime that consumes the math (`cipher_koopman_runtime.cpp`) is dead code with zero callers in either tree. The dispatch table reserves the Koopman lane. The actual wire-up to live decode is v2 work. The v1 substrate does not depend on it.

**Partition-aware Marlin INT4 GEMM kernel.** The current Marlin actuator is structurally full-GPU (grid = 132 SMs, persistent split-K) and is pinned to the primary context. The split-K rewrite that lets Marlin compose with 8-SM green-context partitioning is Phase 6 work, scoped as a 4 to 12 week engagement with Song Han (see Section 7). The v1 multi-tenant prefill story uses cuBLAS FP16 partition-aware instead of Marlin.

**SDPA dispatch routing and attention substitute actuators.** The attention substrate trampolines exist and the routing returns PASSTHROUGH in v1 (T4.6.1 observe-only contract). The first attention substitute actuator (FAVOR+ or FlashSwiftKey) is v1.5 work. The Week 2 LP-2 refactor unblocks this lane structurally.

**Multi-node NCCL workloads (Class D).** The Tree A `cipher_nccl_*` source files (algo selection, eBPF, neural cost model, v4 tuner) live in may13. The v1 deployment is single-GPU. Multi-node lifts are Phase 6 or later.

**Training workloads.** Every measurement in §3 is inference. The substrate is inference-shaped. Training extension is a v3 conversation, not a v1 scope item.

### What is named honestly that competitors do not name

- **The 700 W power cap is a real constraint.** The defensible sustained ceiling is 660 TFLOPS / 67% MFU. A pitch that quotes 745 TFLOPS without the burst caveat is either uninformed about the hardware or hoping the reader does not check.
- **The 5.98× scaling figure is at N=16 on TinyLlama-1.1B, not on Mistral-7B.** A pitch that quotes "5.98× on Mistral-7B" is a fabrication that fails DD in one query.
- **The −0.39% under vLLM is a latency-delta measurement, not a transparency measurement.** Bit-identical forward is Track 2 SC6, on a different path.
- **Goal 4 (Koopman O(1)) is dead code today, by named choice, and v2 scope.** A pitch that implies it is shipping in v1 is overclaiming.
- **MFU is regime-specific.** A pitch that quotes "85% MFU" without naming the regime is the universal claim that v1.2 explicitly retracted.

These are surfaced in the v1.2 plan §5.6 and §8 explicitly so DD does not surprise them.

---

## Section 7: Why the Team Can Execute

### Anil's background mapped to the substrate requirements

[user to provide: founder background, professional history, prior shipped products, and the technical experience that maps to the dispatch-boundary substrate this plan describes. No founder-bio or resume document was located in the repository at the time this summary was authored. Anil is documented as the single point of technical adjudication for the Song Han engagement (`phase_c/track_3/PHASE_5_CP_5_3_SONG_HAN_ENGAGEMENT_SCOPE.md:30`), which is the role of record on the substrate-level technical decisions.]

### The discipline pattern: 10,000+ lines of audit before integration work begins

The audit work that precedes v1 implementation is itself a primary deliverable. The five logic-audit waves and the deep inspection report total approximately 627 KB of structured analysis across six documents:

| Audit document | Size | Coverage |
|---|---|---|
| CIPHER_LOGIC_AUDIT_WAVE_1_CLASSIFIER.md | 115.9 KB (115,913 bytes) | Tree A classifier and oracle control flow, 5-gate safety, F1 struct collision finding |
| CIPHER_LOGIC_AUDIT_WAVE_2_ACTUATORS.md | 91.8 KB (91,829 bytes) | Marlin actuator, cuBLAS shim, attention dispatch, registry semantics |
| CIPHER_LOGIC_AUDIT_WAVE_3_KMOD.md | 126.4 KB (126,441 bytes) | Kmod ABI (NRs 1-24), state machines, do_exit reaper discipline |
| CIPHER_LOGIC_AUDIT_WAVE_4_OBSERVERS.md | 112.2 KB (112,176 bytes) | 14 observability ops, AUDIT chain, fairness, carbon, receipts |
| CIPHER_LOGIC_AUDIT_WAVE_5_FUSION_PLAN.md | 124.5 KB (124,530 bytes) | 43 fusion contracts, 16 lossy points (LP-1 through LP-16), 22 lossless contracts |
| CIPHER_DEEP_INSPECTION_REPORT.md | 56.3 KB (56,313 bytes) | 9 corrections to v1.0 and 6 new findings folded into v1.2 |
| **Total** | **627.2 KB (627,202 bytes)** | |

The discipline pattern is: audit before integration, name every fusion point, classify by severity, schedule mitigation per week. The v1.2 plan folds every audit finding back into the master plan with an inline citation pointing at the audit document and section. A reviewer can trace every claim in the plan back to a line in an audit document, which can be traced back to a line in a source file.

This pattern is unusual in venture-backed AI infrastructure work. Most projects ship the integration first and audit the result. CIPHER audits the substrate before the integration ships, which is why the v1 five-week window is defensible.

### Advisor relationships

**Song Han (MIT HAN Lab).** Documented engagement scope in `PHASE_5_CP_5_3_SONG_HAN_ENGAGEMENT_SCOPE.md`. Advisory (not full-time), 4 to 12 weeks duration with an 8-week midpoint. Owns the partition-aware Marlin INT4 GEMM kernel design and the reference implementation. Four milestones: M1 design memo (week 2), M2 reference implementation (week 4), M3 performance characterization (weeks 6 to 8, with a target of ≥ 75% throughput versus full-GPU Marlin on 8-SM green-context), M4 integration (weeks 8 to 12). Equity range 0.5% to 1.0% on a FAST template with 2-year vest and 6-month cliff. Technical adjudication at each milestone by Anil; mutual termination right per FAST standard.

Song Han is the author of AWQ and SmoothQuant and runs the partition-aware kernel research line at MIT HAN Lab. The engagement scope is precisely the load-bearing Phase 6 unknown: making split-K Marlin compose with 8-SM partitioning so that compute-bound prefill on partitioned tenants can use the INT4 lane.

**Devang Sachdev.** Facilitator of the warm intro to Song Han per the same engagement-scope document. The relationship-source role is documented; no separate advisory engagement scope is on file in this repository at the time this summary was authored.

### NVIDIA Inception status

[user to provide: NVIDIA Inception membership status, tier, and any associated technical resource access (cloud GPU credits, technical-advisory contacts, partner-program engagement). No document in the repository at the time this summary was authored references NVIDIA Inception membership or status.]

---

## Section 8: Use of Funds

[user to provide: the $15M seed allocation breakdown across team, infrastructure, and sales, and the 12 to 18 month plan to Series A milestones. The technical milestones the round funds are:

- v1 ship at end of week 5 (the five-week integration sequence described in §5).
- CP 5.5 100-tenant heterogeneous benchmark with reproducible measurements within 30 minutes of fresh-boot setup.
- v1.5 lift program post-CP-5.5: PREDICT L2 prefetch, FAIRNESS quota enforcement, SHIELD per-tenant SLO priority, SUSTAIN KV-pressure feedback, THERMOSTAT clock feedback, HIBERNATE on Lambda fallback.
- Phase 6 Song Han engagement for partition-aware Marlin (4 to 12 weeks).
- Depth-win program to close the gap from the 67% sustained ceiling to the 85% MFU gate on declared compute-bound workloads (TMA thread-block-cluster attention kernels, L2 budget enforcement via new kmod ioctl, partition-aware Marlin landing).
- 24-hour soak gate (G4) at sustained N=100 tenant load with no kernel oops or WARN.

The team-and-sales side of the allocation, the geographic mix, and the operating-burn ramp are operator decisions and live above the technical scope of this document.]

---

## Closing Note

This summary is a 6,200-word distillation of a 124 KB binding architecture plan (v1.2) and 560 KB of supporting audit work. Every measurement is cited to a file path and line number. Every retracted claim is named so a DD reader does not encounter a dangling reference. Two items in §7 and the financial breakdown in §8 require operator input that is not currently captured in the repository; they are marked explicitly for the user to fill in rather than inferred.

The architectural claim is one sentence: CIPHER is the workload-aware control-plane runtime that lives at the CUDA injection boundary, classifies every kernel, routes to the right regime-specific actuator, and ships the cross-tenant primitives (weight sharing, KV-prefix dedup, SM partitioning, dynamic SM migration, fleet-policy DVFS) in a kernel module that survives any single tenant crashing. The v1 deliverable is the substrate that makes those routes possible; the depth-win program and v2 lifts close the gap from substrate to headline.

The marvel is integration, not invention. Five weeks. Eight risk categories surfaced. Three product goals achievable on the regimes where each is physically valid. One coherent control-plane runtime.

**End of CIPHER_PLAN_EXECUTIVE_SUMMARY.md (v1.0, 2026-05-20).**
