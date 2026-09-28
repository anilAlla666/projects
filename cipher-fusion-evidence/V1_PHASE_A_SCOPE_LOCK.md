# V1 Phase A scope-lock : Driver-level worker-init hook for LD_PRELOAD-only deployment

**Date:** 2026-05-26
**Pre-state anchor:** `option-2-complete` plus `cp-5-5-pre-plugin-routed-feasibility`
- `cipher_rt_phase4` `52923af` tags `option-2-complete` plus `option-2-step-1-alpha-dtype-counter`
- `libcipher_rt.so` md5 `4bedf6488c9a7c0ae7eb793794b02907`
- `cipher_kmod` `8c643fc` tag `week-13-14-complete`
- `cipher_vllm_plugin/cipher_vllm_kv.py` md5 `b89a9b6e` (post-probe; probe hook is env-gated, no production side effect)
- `cipher-fusion-evidence` `f4b652c` tag `cp-5-5-pre-plugin-routed-feasibility`

**Authority:**
- Anil adjudication 2026-05-26 Goal 5 contract lock (Phase 1.5 §B option 4)
- Anil adjudication 2026-05-26 V1 substrate work sequence: Phase A through Phase D, then Phase E CP 5.5 scope-lock
- Anil adjudication 2026-05-26: abandon quant-method registration probe; no parallel track; substrate-restoration is the only workstream until Phase A lands
- Anil adjudication 2026-05-26 deployment-layer ledger discipline (gates all v1 + v1.5 substrate work going forward)

**Type:** v1 substrate-restoration work; Phase A of the 5-phase v1 sequence; precedes CP 5.5 scope-lock at Phase E. Tag `v1-substrate-driver-worker-init` at A.4 close. Paperwork commit tag `v1-goal5-contract-lock` ships ahead of A.1 substrate implementation per §4 sequencing.

---

## 1. Why this campaign

The v1 product target at `CIPHER_REENGINEERING_PLAN.md:98` states "driver-level multiplexing via `LD_PRELOAD` / `CUDA_INJECTION64_PATH`, zero application code changes." The Goal 5 contract at `CIPHER_REENGINEERING_PLAN.md:104` states "LD_PRELOAD-only deployment transparency, customer applications run unchanged." Per Anil 2026-05-26 lock, this contract is non-negotiable and supersedes the Option 2 close-out reframe at `option2-complete.md:22` ("LD_PRELOAD plus drop-in cipher_vllm_plugin, zero application code changes"). The reframe was a recorded intent; it never propagated to the plan; per 2026-05-26 lock it will not propagate. It is REVERTED.

Today's substrate at `option-2-complete` requires the `cipher-vllm-kv` pip-installed plugin to bring up cuBLAS GOT patches in the vLLM V1 EngineCore worker subprocess when the customer sets `LD_PRELOAD` only (without `CUDA_INJECTION64_PATH`). Per Option 2 Step 0 evidence (`option2-step0-closed.md` plus `WEEK_14_FOLLOWUP_OPTION_2_STEP_0_VLLM_WORKER_HOOK.md`), the plugin's `_install_cipher_rt_got_patches()` at `cipher_vllm_kv.py:649-706` force-dlopens libcublas, dlopens libcipher_rt, then calls `InitializeInjection2()`. Without the plugin, this sequence never runs in the worker; cuBLAS GOT slots are not patched; every decode GEMM passes unintercepted; substrate reads zero in the worker.

Phase A restores the Goal 5 contract by moving this responsibility from the plugin into libcipher_rt.so itself. After Phase A lands, the customer sets `LD_PRELOAD=libcipher_rt.so` only (or `CUDA_INJECTION64_PATH=libcipher_rt.so` if they prefer), no plugin install required, no pip dependency.

---

## 2. The strong prior

Option 2 Step 0 close-out (`option2-step0-closed.md`) confirmed three things that ground Phase A's design:

1. **`InitializeInjection2()` runs successfully in the worker when invoked.** Counter delta `cipher_rt_cublas_shim_calls` went 0 to 11658 in worker pid over TinyLlama-1.1B vLLM V1 N=1 128-token decode. The mechanism inside libcipher_rt is sound; the missing piece is the trigger from the worker subprocess.

2. **The CUDA driver invokes `InitializeInjection2()` automatically when `CUDA_INJECTION64_PATH` is set.** This path works today on this pod (`cipher-pod-environment` H100 driver 580.105.08). The plugin path was built for the LD_PRELOAD-only deployment case where the driver does not auto-invoke.

3. **The ordering constraint is: libcublas must be loaded before InitializeInjection2 runs.** Per `option2-step0-closed.md` and the plugin's force-dlopen sequence at `cipher_vllm_kv.py:675-686`, GOT-patching libcublas requires it to be mapped first. The plugin solves this by force-loading libcublas itself before calling InitializeInjection2. Phase A's substrate-side approach must solve the same ordering.

**The prior says Branch A is reachable.** All three ingredients exist; Phase A reroutes the trigger from plugin to substrate. Risk is at the ordering boundary (R-A.1) and at non-vLLM-V1 paths (R-A.5).

---

## 3. Three locked decisions

1. **Goal 5 contract is non-negotiable** (Anil 2026-05-26). Customer integration surface for v1:
   - Set `LD_PRELOAD=libcipher_rt.so`, OR
   - Set `CUDA_INJECTION64_PATH=libcipher_rt.so`
   Neither requires the other. Neither requires plugin install, CLI flag, config field, Python wrapping, or model code change. Phase A closes the LD_PRELOAD-only gap.

2. **Substrate-side worker-init mechanism per Anil spec.** Two candidate approaches surfaced:
   - **(a) `__attribute__((constructor))`** at `cipher_inject.c` or analogous compilation unit. Runs at libcipher_rt.so dlopen time in every process that has it LD_PRELOAD'd, including freshly-forked vLLM V1 EngineCore worker subprocesses.
   - **(b) LD_PRELOAD wrapper around `cuInit`** that interposes the CUDA driver call, runs InitializeInjection2 plus libcublas force-load on first cuInit, then calls real cuInit via `dlsym(RTLD_NEXT, "cuInit")`.
   Both are well-known LD_PRELOAD patterns. Implementation choice deferred to A.1 design memo; Branch A/B framing covers either outcome.

3. **HARD GATE at A.2 close** (Anil scope verbatim): pip uninstall `cipher-vllm-kv`; remove `vllm.general_plugins` entry point from importlib.metadata; run TinyLlama-1.1B vLLM V1 N=1 128-token decode under `LD_PRELOAD=libcipher_rt.so` only (no `CUDA_INJECTION64_PATH`, no plugin); verify `cipher_rt_cublas_shim_calls` in worker reaches baseline of approximately 11658 (matching Option 2 Step 0 §3 gate 2). If gate fails, STOP and surface; do not silently extend.

---

## 4. Five-substep sequence with file:line citations

### A.0 : Paperwork: Goal 5 contract lock revert

**Calendar:** 0.5 ED. Lands as own commit tagged `v1-goal5-contract-lock` ahead of A.1 implementation per Anil 2026-05-26 spec.

**Touches:**
- `CIPHER_REENGINEERING_PLAN.md:98` (v1 product target sentence): retain original "LD_PRELOAD-only" language verbatim; no edit needed, the existing text already matches the locked contract
- `CIPHER_REENGINEERING_PLAN.md:104` (Goal 5 bullet): retain original LD_PRELOAD-only deployment transparency framing verbatim; no edit needed
- `CIPHER_REENGINEERING_PLAN.md:1485, 1741` (R-G5.2 risk language): retain original "LD_PRELOAD-only Goal 5" naming verbatim; no edit needed
- `/home/ubuntu/.claude/projects/-home-ubuntu/memory/option2-complete.md:22, 36` and `:65-79` (the Goal 5 framing reframe block): edit to record that the reframe is REVERTED per 2026-05-26 contract lock; the substrate path identified as v1 work is Phase A driver-level worker-init, not plugin install
- `/home/ubuntu/.claude/projects/-home-ubuntu/memory/plan-v1.2.3.md:23, 30`: no edit needed; existing language is "LD_PRELOAD-only" which matches the locked contract
- `/home/ubuntu/.claude/projects/-home-ubuntu/memory/cipher-fusion-campaign.md`: scan for "LD_PRELOAD plus drop-in" or "LD_PRELOAD + drop-in" or "cipher_vllm_plugin reframe"; revert if present
- `/home/ubuntu/.claude/projects/-home-ubuntu/memory/pre-cp-5-5-plugin-routed-probe.md`: scan; revert if the reframe propagated here

**Audit finding before A.0 lands:** §1 of this scope-lock already verified plan §1 line 98 + 104 are still at original framing. The reframe propagated only to `option2-complete.md`. Paperwork is therefore one memory file edit (option2-complete.md) plus a scan-and-revert across the other named memory files.

**Verification gate at A.0 close:**
- `grep -rE "LD_PRELOAD \+ drop-in|cipher_vllm_plugin.*zero application" CIPHER_REENGINEERING_PLAN.md /home/ubuntu/.claude/projects/-home-ubuntu/memory/*.md` returns empty (modulo the option2-complete.md entry that records the REVERT)
- option2-complete.md carries an explicit "REVERTED 2026-05-26 per Goal 5 contract lock" note at the §5 reframe block

**Close tag:** `v1-goal5-contract-lock`. cipher-fusion-evidence commit lands this doc edit chain. No substrate change.

### A.1 : Substrate worker-init hook implementation

**Calendar:** 2 ED nominal (Branch A constructor approach), 4 ED if Branch B fallback to cuInit-wrapper triggers.

**Touches (Branch A constructor path):**
- `cipher_rt_phase4/cipher_inject.c` or `cipher_rt_phase4/cipher_rt_init.c` (TBD at design): add `__attribute__((constructor))` function `cipher_rt_auto_init_worker()` that:
  - Checks an idempotency flag (similar to InitializeInjection2's pthread_once guard); returns if already initialized
  - dlopen libcublas SONAMEs in order ("libcublas.so.12", "libcublas.so.11", "libcublas.so") with RTLD_GLOBAL plus RTLD_LAZY; mirrors plugin's force-dlopen at `cipher_vllm_kv.py:675-680`
  - dlopen libcudnn similarly per plugin's `:681-686`
  - Calls `InitializeInjection2()` (the CUDA-driver-mandated entrypoint name; direct symbol invocation since we are libcipher_rt.so itself, no dlopen needed)
  - Logs `[cipher_rt] auto-init worker: GOT patches installed pid=<pid>` to stderr at INFO level
- `cipher_rt_phase4/cipher_rt_init.h` or analogous: declare `cipher_rt_auto_init_worker` and the idempotency flag
- **Substrate counter dump port for verification harness.** Plugin's `_install_counter_dump_for_verification()` at `cipher_vllm_kv.py:709-792` provides the SIGUSR1 + atexit + install snapshot mechanism Option 2 used to read counters from the worker subprocess after vLLM teardown. Phase A must port this to substrate-side so A.2 verification can read counters without the plugin installed. Approach: env-gated by new var `CIPHER_RT_COUNTER_DUMP_PATH` (or reuse `CIPHER_VLLM_COUNTER_DUMP_PATH` for migration ease); when set, the auto-init constructor registers an atexit handler that writes a JSON snapshot of substrate counters (`cipher_rt_cublas_shim_calls`, `cipher_rt_matmul_calls_total/handled/passthrough`, `cipher_rt_koopman_calls_total/handled/skipped`, plus the Option-2 per-early-exit counters `cipher_rt_koopman_skip_dtype/_dim/_nullptr`) to `<env path>/cipher_rt_<pid>.json`. Zero production overhead when env unset (atexit not registered). Also installs a SIGUSR1 handler with the same snapshot logic for live-read support. ABI is the new env var; no new T-symbol since the snapshot reads existing T-symbols.
- Build system (`cipher_rt_phase4/Makefile`): no change expected; constructor attribute is compiler-level

**Touches (Branch B cuInit-wrapper fallback):**
- New file `cipher_rt_phase4/cipher_rt_cuinit_hook.c`: define `CUresult cuInit(unsigned int flags)` that intercepts via LD_PRELOAD symbol resolution, runs the auto-init sequence above, then calls real cuInit via `dlsym(RTLD_NEXT, "cuInit")`
- May require export list update if libcipher_rt.so uses a version script that hides symbols
- Counter dump port (above) is identical scope under Branch B

**ABI impact:** ABI-additive only. One new T-symbol `cipher_rt_auto_init_worker` (or `cuInit` interposition under Branch B). One new env var `CIPHER_RT_COUNTER_DUMP_PATH`. No removal, no signature change, no rename. Follows `cipher-abi-rule` discipline.

**Estimate detail (revised to include counter dump port):**
- Branch A constructor: ~80 LOC C constructor + ~120 LOC C counter dump port, ~2.5 ED (write plus self-test smoke plus regression spot-check)
- Branch B cuInit wrapper: ~120 LOC C interposition + ~120 LOC C counter dump port, ~4.5 ED total (write plus self-test plus regression)

**Close tag for A.1:** none yet; A.1 closes only after A.2 + A.3 gates PASS, at which point A.4 ships the close tag.

### A.2 : Verification harness rewrite (no-plugin path)

**Calendar:** 1 ED.

**New artifact:** `cipher-fusion-evidence/v1_phase_a/verify_no_plugin.py` extends Option 2 Step 0 `verify_step0.py` pattern (`cipher-fusion-evidence/option2/step_0/verify_step0.py`):
- Pre-condition assertion: `pip show cipher-vllm-kv` returns non-zero (uninstalled); `importlib.metadata.entry_points(group="vllm.general_plugins")` returns empty list
- Workload: TinyLlama-1.1B-Chat-v1.0 FP16 vLLM V1 N=1 128-token decode with `LD_PRELOAD=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so` only (no `CUDA_INJECTION64_PATH`)
- Telemetry: `CIPHER_RT_COUNTER_DUMP_PATH=/tmp/v1_phase_a_verify` (new env var introduced in A.1). Substrate counter dump fires from the worker subprocess via the constructor-installed atexit + SIGUSR1 handler (substrate-side port of the plugin's `_install_counter_dump_for_verification` pattern; landed in A.1 per the new sub-element above). This is the load-bearing dependency: without A.1's counter-dump port, A.2 has no way to read counters from the worker subprocess after vLLM tears it down.
- Hard gate per Anil scope verbatim: `cipher_rt_cublas_shim_calls` in worker >= 11000 (allow noise margin against the 11658 Option 2 baseline)
- Soft sanity gates: `cipher_rt_matmul_calls_total` >= shim_calls; `cipher_rt_koopman_calls_total` == 0 (koopman engine env-gated off in baseline run)

**Pre-condition setup steps documented in the harness preamble:**
```
# 1. pip uninstall cipher-vllm-kv  (run once; verify uninstall via pip show)
# 2. python -c "from importlib.metadata import entry_points; \
#      print([ep for ep in entry_points(group='vllm.general_plugins')])"
#    expected: [] (empty list)
# 3. unset CUDA_INJECTION64_PATH (verify in env)
# 4. export LD_PRELOAD=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
# 5. python verify_no_plugin.py --tokens 128 --out result.json
```

**Verification gate at A.2 close:**
- Gate (1) library load in worker: `/proc/<worker_pid>/maps` contains libcipher_rt.so (necessary, not sufficient; LD_PRELOAD inherits)
- Gate (2) **HARD per Anil scope**: `cipher_rt_cublas_shim_calls` in worker dump >= 11000; bit-identical decode token_ids vs vanilla-vLLM reference
- Gate (3) sanity: plugin path remains importable when manually pip-installed (idempotency check: re-installing plugin does not double-init the worker)

If Gate (2) fails, STOP and surface; do not silently extend to A.3.

### A.3 : Regression smoke against Track 2 plus Track 3

**Calendar:** 1 ED.

**Methodology:** re-run prior gate suites WITHOUT plugin installed:
- Track 2 SC2 through SC6 e2e per `cipher-fusion-evidence/TRACK3_SC1_SC6_E2E_RESULT.md` (note: prior baseline ran WITH plugin; A.3 reruns WITHOUT)
  - SC6 gate: 76 percent N=4 Mistral-7B HBM savings reproduces (cipher-track2-weight-sharing memory cites this as the headline)
  - Other SC gates per Track 2 close-out
- Track 3 DSM v1 SC1 through SC6 per `cipher-fusion-evidence/TRACK_3_CLOSEOUT.md`
  - Live SM migration correctness invariants must PASS
- W7-12 microbench regression spot-check (test_commit_atomicity, test_audit_chain, test_observe_publish, test_resolver, test_ring_write, test_g3_cross_model_keying, test_tc_probe, test_g5_va_density, test_l2_wireup); these run substrate-only, do not depend on plugin

**Estimate detail:** Track 2 SC6 alone is the load-bearing run; ~30 min wall clock per W6 bench harness. Track 3 SC1-SC6 ~30 min. Microbench regression ~10 min. Total ~1-1.5 hours measurement time; budget includes ~3 hours for diagnostic if any gate fails.

**Verification gate at A.3 close:**
- Track 2 SC6 PASS at 76 percent +/- 1 percent (matches Track 2 closeout baseline; small noise margin per `week6-bench-harness` ~0.5 percent floor)
- Track 3 SC1-SC6 all PASS per Track 3 closeout invariants
- W7-12 microbench regression matches W12 Step 3 baseline (matches `g6-audit-chain` Case 3 known carry-forward, `g3-g4-tc-probe` dlopen carry-forward; same baseline state)

### A.4 : Close-out with Branch A/B/C/D framing

**Calendar:** 0.5 ED.

**Touches:**
- `cipher-fusion-evidence/V1_PHASE_A_COMPLETE.md` (new): close-out doc with anchors, branch verdict, evidence chain, residue
- `/home/ubuntu/.claude/projects/-home-ubuntu/memory/v1-phase-a-driver-worker-init.md` (new memory file)
- `MEMORY.md` index update (one line)
- `cipher_rt_phase4` close-out commit + tag `v1-substrate-driver-worker-init` (substrate change IS required per A.1 scope; constructor lands new T-symbol or interposes cuInit, libcipher_rt.so md5 rotates)

**Verification:** scope-lock branch framing language (§8) maps to actual outcomes; honest residue declared; deployment-layer ledger row added per Q5 disposition.

---

## 5. Total estimate

A.1 estimates revised to include counter-dump port (~120 LOC C, ~0.5 ED added).

| Branch | A.0 | A.1 | A.2 | A.3 | A.4 | Total |
|---|---|---|---|---|---|---|
| **Branch A** (constructor lands clean, gates PASS) | 0.5 | 2.5 | 1 | 1 | 0.5 | **5.5 ED** approximately 1 cal-week |
| **Branch B** (constructor fails ordering; cuInit-wrapper ships as the working mechanism, constructor approach is documented as attempted, NOT layered together) | 0.5 | 4.5 | 1 | 1 | 0.5 | **7.5 ED** approximately 1.5 cal-weeks |
| **Branch C** (Branch A lands but regression in some path; conditional shipping) | 0.5 | 3.5 | 1 | 2 | 0.5 | **7.5 ED** approximately 1.5 cal-weeks |
| **Branch D** (both A and B fail to install reliably without plugin; HARD STOP) | 0.5 | 4.5-6.5 | 1 | 0 | 0.5 (surface) | **6.5-8.5 ED** approximately 1.5 cal-weeks before surface |

Range 5.5-8.5 ED matches Anil's 1-2 cal-week Phase A budget.

Branch A and Branch B are mutually exclusive shipping mechanisms per Q3 default (try-then-fallback, not ship-both-layered). If Anil picks ship-both-layered (Q3 option c), A.1 estimate rises to ~6.5 ED and Branch B effectively becomes the default close (constructor on top, cuInit-wrapper underneath as failsafe).

---

## 6. Risks

| ID | Risk | Likelihood | Impact | Mitigation | Detection (verification gate) |
|---|---|---|---|---|---|
| **R-A.1** | `__attribute__((constructor))` ordering issue: libcublas force-load races with the application's own CUDA init or pulls in CUDA contexts before the application is ready | MEDIUM | MAJOR (Branch B fallback) | Use `RTLD_LAZY` plus check-if-already-loaded via `dlsym` (no-op if libcublas symbols resolve already); make constructor idempotent and reentrant-safe. **Per Anil 2026-05-26 caveat on Q3 (a): if constructor ordering is mechanically fragile (libcublas not yet loaded at constructor time, or torch.cuda.init lock contention), SURFACE IMMEDIATELY. Do NOT push through ordering issues silently. Branch B switch is the documented fallback and requires surfacing-before-switching.** | A.2 hard gate; A.3 regression spot-check; A.1 design memo if ordering issue detected during implementation |
| **R-A.2** | Worker subprocess uses a non-cuInit CUDA entry (e.g., torch.cuda.init internal path bypasses standard cuInit) so neither constructor nor cuInit-wrapper installs before first GEMM | MEDIUM | MAJOR | Hook constructor at LD_PRELOAD-load (which fires unconditionally on libcipher_rt.so dlopen, regardless of CUDA call sequence). cuInit-wrapper is the Branch B fallback if constructor races; if BOTH fail, Branch D HARD STOP | A.2 counter delta is the empirical test |
| **R-A.3** | Constructor runs in non-CUDA processes (e.g., a process that LD_PRELOAD's libcipher_rt.so but never uses CUDA) and force-loads libcublas unnecessarily, costing process startup time and memory | LOW | MINOR | Constructor checks for CUDA library presence first via `dlsym` lookup; force-load only if not present; if no CUDA at all, no-op | A.3 regression includes a CPU-only Python smoke (no torch.cuda) |
| **R-A.4** | Existing `CUDA_INJECTION64_PATH` path (already working, return=1 per Option 2 Step 0) regresses due to constructor double-init | LOW | MAJOR | Idempotency flag plus pthread_once guard inside `cipher_rt_initialize_injection2`; CUDA_INJECTION64_PATH path detects already-initialized and no-ops | A.3 includes a CUDA_INJECTION64_PATH-only smoke (no LD_PRELOAD) |
| **R-A.5** | Other user-side LD_PRELOAD'd libraries (e.g., libcublasLt, custom shims) break due to constructor changes or interposed cuInit | LOW | MEDIUM | Constructor does minimal work, only force-loads named libs that match plugin's existing list; cuInit-wrapper falls through to RTLD_NEXT cleanly | A.3 includes a torch-only standalone smoke (no vLLM, no other LD_PRELOAD) |
| **R-A.6** | A.2 hard gate fails: cublasGemmEx interception does not reach baseline in worker without plugin | MEDIUM (per scope spec) | BLOCKER for Branch A | Branch B cuInit-wrapper fallback is the documented path; both fail equals Branch D HARD STOP and surface | A.2 itself is the test |
| **R-A.7** | (REMOVED, reframed below as expected baseline measurement, not a risk) | n/a | n/a | n/a | n/a |
| **R-A.8** | Paperwork revert (A.0) misses a propagation site, leaving stale "LD_PRELOAD plus drop-in cipher_vllm_plugin" language in some artifact | LOW | MINOR | A.0 verification gate runs the grep against the full cipher-fusion-evidence directory plus memory/, not just the named files; flag any other hits | A.0 grep returns approximately-empty (the expected REVERT-record entry in option2-complete.md is the only match) |

**Expected baseline measurement at A.3 (not a risk, mechanism-determined outcome):** Plugin's `_cipher_allocate_kv_cache_tensors` hook at `cipher_vllm_kv.py:305` is the KV-bridge load-bearing path for Track 2 SC6's 76 percent N=4 Mistral-7B savings (per `cipher-track2-weight-sharing` close). Uninstall the plugin and this hook never runs; vLLM falls back to native `torch.zeros` KV allocation per `cipher_vllm_kv.py:8-12` docstring. Track 2 SC6 measurement under no-plugin path will therefore show savings close to substrate-without-KV-bridge baseline (likely close to 0 percent cross-tenant savings since no shared weight arena, no cross-process page tagging). This is not a regression to be mitigated; it is the substrate-only baseline that Phase B's KV-bridge migration uplifts from. A.3 records the measured no-plugin number as Phase B scope-lock entry target. If the no-plugin number is reported as anything other than the substrate-only baseline (i.e., if savings surprise above expected), surface; otherwise the measurement is informational for Phase B planning.

---

## 7. Honest residue declared up-front

1. **`CUDA_INJECTION64_PATH` path is unaffected by Phase A.** That path already works today (Option 2 Step 0 return=1). Phase A is specifically for the LD_PRELOAD-without-CUDA_INJECTION64_PATH case.

2. **Plugin remains installable.** Phase A makes plugin install OPTIONAL, not removed. Customer deployment per Goal 5 contract uses libcipher_rt.so directly; dev / debug / advanced-feature paths can still install the plugin. Phase B is where plugin functionality migrates into substrate and the plugin becomes truly deprecated for v1 production.

3. **Track 2 SC6 may show reduced savings under no-plugin path** per R-A.7. Expected; Phase B addresses. A.3 records the measured number for Phase B scope-lock entry. Track 2 closeout headline 76 percent N=4 Mistral-7B savings is informationally retained as the WITH-plugin number until Phase B closes; the WITHOUT-plugin number from A.3 sets the Phase B uplift target.

4. **Mistral E.7 carry-forward is independent.** Phase A does not require Mistral; TinyLlama-1.1B is the gate workload. Mistral env block remains a separate concern queued for separate diagnostic prompt per Phase 1.5 §3 item 4.

5. **Paperwork revert disposition for option2-complete.md.** Anil's spec said to update the file noting REVERT, not delete the reframe block. The historical record of the reframe stays; the disposition flips from "LANDED" to "REVERTED per 2026-05-26". This preserves audit trail per `cipher-evidence-commit-discipline`.

6. **Branch labels (A/B/C/D) in this scope-lock are Phase A internal outcomes**, not to be confused with Anil's prior "5 product goals Branch A required in v1" adjudication. The Goal-Branch-A and Phase-Branch-A naming collision is unavoidable per the Option 2 followup pattern Anil endorsed. Doc disambiguates by always prefixing with "Phase A Branch X" or "Goal X Branch A" in body text.

---

## 8. Pre-commit framing per Phase A branch

Per `WEEK_14_FOLLOWUP_OPTION_2_SCOPE_LOCK.md` §3 decision 3 template.

**Phase A Branch A outcome statement template:**
> V1 Phase A driver-level worker-init landed at cipher_rt_phase4 [new commit] tag `v1-substrate-driver-worker-init`. With cipher-vllm-kv pip-uninstalled and `vllm.general_plugins` entry point absent, libcipher_rt.so [chosen mechanism, constructor or cuInit-wrapper per Q3 disposition] auto-initializes the worker subprocess GOT patches at [LD_PRELOAD load time or first cuInit]. TinyLlama-1.1B-Chat-v1.0 FP16 vLLM V1 N=1 128-token decode shows `cipher_rt_cublas_shim_calls=[X]` in worker (gate: >=11000; observed: [X]). Track 2 SC6 measurement under no-plugin path shows [Y] percent N=4 Mistral-7B savings (substrate-only baseline per §6 expected-baseline-measurement note; informs Phase B uplift target). Track 3 DSM v1 SC1-SC6 PASS. **Goal 5 contract restored at the cuBLAS GOT-intercept layer.** Full contract restoration through Phase D (Phase B migrates KV-bridge into substrate; Phase C adds BF16 Koopman; Phase D adds FP8 multi-route GOT-patch). Customer deployment that exercises only the cuBLAS-intercept layer is plugin-free at A.4 close; deployments that need Track 2 weight sharing or KV-dedup still require plugin install until Phase B closes.

**Phase A Branch B outcome statement template:**
> V1 Phase A driver-level worker-init landed at cipher_rt_phase4 [new commit] tag `v1-substrate-driver-worker-init`. The constructor approach hit [specific ordering issue, e.g., libcublas force-load deadlock with torch.cuda.init]; fallback to LD_PRELOAD wrapper on cuInit ships as the working mechanism and passes the A.2 hard gate per Branch A criteria. Constructor approach is documented at `V1_PHASE_A_COMPLETE.md` §[N] as attempted-but-not-shipped for future port reuse. Substrate ships with the cuInit-wrapper only (NOT ship-both-layered per Q3 default disposition). Phase A Branch B engineering result: substrate path is sharper than scope-lock anticipated; ordering issue informs Phase B / Phase C mechanism reuse. **Goal 5 contract restored at the cuBLAS GOT-intercept layer; same scope qualifier as Branch A.**

**Phase A Branch C outcome statement template:**
> V1 Phase A driver-level worker-init landed at cipher_rt_phase4 [new commit] tag `v1-substrate-driver-worker-init`. Branch A mechanism (constructor) lands cleanly for vLLM V1 but introduces regression in [specific path, e.g., torch-only standalone, vLLM V0 if still supported, custom-LD_PRELOAD-stack]. Substrate ships with conditional behavior: constructor enabled when vLLM V1 worker spawn pattern detected via [detection mechanism, e.g., `VLLM_USE_V1=1` env or fork-from-Python detection], otherwise no-op. **Goal 5 contract restored for vLLM V1 deployment at the cuBLAS GOT-intercept layer** (Phase B-D still pending); non-vLLM paths require either CUDA_INJECTION64_PATH or an explicit env opt-in. Surface to Anil whether the non-vLLM regression is acceptable in v1 or requires a v1.5 substep.

**Phase A Branch D outcome statement (HARD STOP):**
> V1 Phase A driver-level worker-init STOPPED. Both `__attribute__((constructor))` and LD_PRELOAD-wrapper-on-cuInit approaches failed to reliably install GOT patches in the vLLM V1 EngineCore worker subprocess without `cipher-vllm-kv` plugin install or `CUDA_INJECTION64_PATH` env. Goal 5 contract per 2026-05-26 lock cannot be restored at this layer with current substrate architecture. STOP and surface to Anil for adjudication: (a) accept CUDA_INJECTION64_PATH-only deployment with documented constraint that LD_PRELOAD-only does not work for vLLM V1 (narrows but does not violate Goal 5 contract since contract uses "or"); (b) accept plugin install as v1 deployment requirement (REVERSES 2026-05-26 contract lock, requires explicit Anil reversal); (c) re-architect the worker-init pattern, including possibly an LD_AUDIT-based or kernel-resident trigger that is not vLLM-V1-specific. No silent fallback to plugin install; Anil adjudication required.

---

## 9. No HARD STOP triggered at scope-lock landing

Scope-lock ready for Anil review. No prerequisite blocks Phase A entry. The pre-state anchors at §0 are verified. Next prompt on Anil approval is A.0 paperwork commit + tag `v1-goal5-contract-lock`, followed immediately by A.1 substrate implementation.

V1 substrate work sequence: **Phase A → Phase B → Phase C → Phase D → Phase E CP 5.5 scope-lock**. Each phase has its own scope-lock document per Anil's sequence spec.

---

## 10. Adjudication record (open items for Anil pre-commit review)

**Q1 (CLOSED 2026-05-26):** Goal 5 framing reverted to original per Anil 2026-05-26 contract lock. Option 2 reframe at `option2-complete.md:22` is REVERTED. Deployment-layer ledger discipline applies to Phase A through Phase D and folds into Phase E CP 5.5 scope-lock.

**Q2 (CLOSED 2026-05-26 per Anil):** option (b) confirmed. Three audit trails per `cipher-evidence-commit-discipline`:
- Scope-lock commit (this doc) ships standalone, no tag at this stage
- Paperwork commit ships next: option2-complete.md REVERT plus ledger creation; tagged `v1-goal5-contract-lock`
- A.1 substrate work ships as a separate commit when authorized

**Q3 (CLOSED 2026-05-26 per Anil):** option (a) confirmed. Constructor-first try-then-fallback. Reuses Option 2 plugin's proven force-dlopen libcublas plus InitializeInjection2 sequence (`cipher_vllm_kv.py:649-706` evidence trail).

**Anil caveat carried into R-A.1 mitigation language**: during A.1 design, if constructor ordering is mechanically fragile (e.g., libcublas not yet loaded at constructor time, or torch.cuda.init lock contention), surface IMMEDIATELY to Anil. Switch to Branch B (cuInit-wrapper) is the documented fallback; do NOT push through ordering issues silently. The Branch B framing at §8 is the recovery path; surfacing must precede silently shipping Branch B.

**Q4 (CLOSED 2026-05-26 per Anil):** option (a) confirmed. Accept Track 2 SC6 regression to substrate-only baseline under no-plugin as expected Phase A outcome. A.3 records the no-plugin number as Phase B entry target. Phase A stays at 5.5-8.5 ED per §5 estimate; Phase B absorbs KV-bridge migration per the locked sequence.

**Q5 (CLOSED 2026-05-26 per Anil):** option (b) confirmed. Standalone `V1_GOAL5_DEPLOYMENT_LEDGER.md` updated incrementally across phases A through D and into Phase E (CP 5.5 close).

Initial ledger structure per Anil 2026-05-26 (lands in A.0 paperwork commit, tag `v1-goal5-contract-lock`):

| Phase | Customer step | When added/removed | Contract violation? | Status |
|---|---|---|---|---|
| Pre-Phase-A | + pip install cipher-vllm-kv | Option 2 close 2026-05-25 | YES (Goal 5 contract 2026-05-26) | REMOVING in Phase A |
| Phase A | + set `LD_PRELOAD=libcipher_rt.so` | original framing, 2026-05-26 lock | NO | restoring |
| Phase B | (no change to customer step expected) | n/a | n/a | pending |
| Phase C | (no change to customer step expected) | n/a | n/a | pending |
| Phase D | (customer sets `--quantization fp8` for FP8 models per their own model choice) | quantization is model property, not CIPHER requirement | NO | pending |

**Ledger gate discipline (Anil 2026-05-26):** every substrate commit in phases A through E that touches deployment surface MUST update the ledger in the SAME commit. Any row marked "Contract violation? YES" blocks the substrate commit until resolved per Anil adjudication. No silent additions.

---

## 11. Related memory

- [[option2-complete]] : substrate baseline at Phase A entry; Goal 5 reframe REVERTED per A.0 paperwork
- [[option2-step0-closed]] : InitializeInjection2 mechanism evidence; Phase A reroutes the trigger from plugin to substrate
- [[vllm-v1-worker-subprocess]] : underlying gap (worker inherits LD_PRELOAD but does not auto-trigger InitializeInjection2); Phase A closes the gap at substrate side
- [[pre-cp-5-5-plugin-routed-probe]] : ruled out apply-wrapping path; informed the Goal 5 cumulative-drift catch that motivated this Phase A sequence
- [[plan-v1.2.3]] : five product goals; Goal 5 contract held at original per 2026-05-26 lock
- [[cipher-track2-weight-sharing]] : Track 2 SC6 76 percent baseline; A.3 regression measurement informs Phase B scope
- [[cipher-track3-dsm]] : Track 3 SC1-SC6 invariants; A.3 regression measurement
- [[cipher-abi-rule]] : A.1 substrate change is ABI-additive only
- [[cipher-evidence-commit-discipline]] : followed at A.0 paperwork commit and the scope-lock commit
- [[cipher-fresh-session-anchor-verify]] : followed at scope-lock §0 pre-state anchor verification this turn
- [[cipher-proceed-not-ask]] : "Proceed" plus detailed scope from Anil 2026-05-26 then drafted without re-litigation; surface-to-Anil-before-commit per Anil's explicit "Surface to Anil for review before commit" instruction
