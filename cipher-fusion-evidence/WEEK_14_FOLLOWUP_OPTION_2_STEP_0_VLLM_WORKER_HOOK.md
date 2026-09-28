# W14 Followup — Option-2 Step 0: vLLM V1 Worker-Subprocess GOT Patch Hook (CLOSED)

**Date:** 2026-05-25
**Pre-state anchor:** `week-13-14-complete` (verified on disk)
**Tag:** `option-2-step-0-vllm-worker-init-hook`
**Substrate-tree status:** unchanged (cipher_rt_phase4 25970f3, cipher_kmod 8c643fc, cipher_kv_bridge unchanged); plugin-only edit per scope-lock §4 Step 0

**Verdict:** PASS. Worker-subprocess GOT patches confirmed active. Step 1 calibration unblocked.

---

## 1. What this step delivered

Per scope-lock `WEEK_14_FOLLOWUP_OPTION_2_SCOPE_LOCK.md` §4 Step 0, the deliverable is: extend `cipher_vllm_plugin` so that when vLLM V1's `load_general_plugins()` runs `cipher_vllm_kv.register()` in the EngineCore worker subprocess, `InitializeInjection2()` is called and the libcipher_rt GOT patches install in the worker's address space. Without this fix, the worker inherits LD_PRELOAD'd libcipher_rt.so but the patches never apply, and every decode `cublasGemmEx` runs unintercepted (the prior null-measurement attempt 2026-05-24 at `/tmp/step13_3_baseline/option2/run_all.sh`).

The fix is two functions added to `cipher_vllm_kv.py`:

1. **`_install_cipher_rt_got_patches()`** — called unconditionally at the top of `register()`, before the `_enabled()` short-circuit. Force-dlopens libcublas SONAMEs (12, 11, generic) with `RTLD_GLOBAL` so the GOT patcher's `dl_iterate_phdr` walk finds them, then dlopens libcipher_rt and calls `InitializeInjection2()`. Idempotent via module-level flag + pthread_once inside the C side. Belt-and-suspenders for the Goal 5 LD_PRELOAD-only deployment mode (no `CUDA_INJECTION64_PATH`).

2. **`_install_counter_dump_for_verification()`** — env-gated on `CIPHER_VLLM_COUNTER_DUMP_PATH`; if unset, no-op. When set, writes per-pid JSON snapshots of substrate counters at install + SIGUSR1 + atexit. Zero production overhead. Permanent debug surface for future verification.

Plugin `cipher_vllm_kv.py` md5 at Step 0 close: `e77a3a58a5f075985c838110f9193585`. Snapshot at `plugin_snapshots/cipher_vllm_kv.py.w14_followup_step_0`.

---

## 2. Verification gates per scope-lock §4 Step 0

**Harness:** `option2/step_0/verify_step0.py` — runs TinyLlama-1.1B-Chat-v1.0 through vLLM V1 AsyncLLMEngine in-process (worker spawn forced via `VLLM_WORKER_MULTIPROC_METHOD=spawn`), under `LD_PRELOAD=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so` with `CUDA_INJECTION64_PATH` **unset** (the Goal 5 deployment mode the user's amendment is fixing). Decodes 128 tokens, snapshots worker counters before/after.

**Run:** `parent_pid=2032919`, single worker `pid=2033010`, 4 snapshots (install / sigusr1×2 / atexit), 11.12 s duration, 128-token decode in 1.39 s. Output: `" and kind queen. She had a daughter named Lily, who was a beautiful and kind-hea"`.

### Gate (1) — library-load presence in worker — **PASS**

Worker dump file `worker_2033010.json` exists (4 snapshots written). Existence of the file proves `cipher_vllm_kv.register()` ran in the worker subprocess, which proves `libcipher_rt.so` is loaded in worker (the dump helper dlopens it). `/proc/2033010/maps` no longer exists at gate-evaluation time (worker exited cleanly during shutdown), so direct grep is unavailable, but dump-file existence is the stronger discriminator.

Also corroborated by explicit worker log line (captured in stderr):
```
(EngineCore pid=2033010) [cipher-vllm-kv] InitializeInjection2 returned 1 (vLLM V1 worker GOT patches installed pid=2033010)
```

### Gate (2) — counter delta in worker — **PASS** (the strong discriminator)

`cipher_rt_cublas_shim_calls`: **0 → 11658** over the 128-token decode burst in worker pid 2033010. Counter strictly increased in the worker process specifically (the failing prior attempt `/tmp/option2_sweep_test.json` 2026-05-24 had this counter at 0 in the worker; today's run has it at 11658). GOT patches are intercepting `cublasGemmEx` in the worker process, not just the parent.

Also corroborated by worker-stderr exit summary:
```
[cipher_v2] MATMUL: exit totals — calls=11658 handled=0 passthrough=11658 (actuators=1)
[cipher_v2] GOT: patch applied -- 10 slot(s) across 363 module(s); 4 target(s) registered
```

### Gate (3) — `koopman_calls_handled` ≥ 1 at β=0.99 in worker — **INFORMATIONAL FAIL**

`cipher_rt_koopman_calls_total` and `cipher_rt_koopman_calls_handled` both stayed at 0 in the worker over the decode burst, even with `CIPHER_KOOPMAN=1 CIPHER_KOOPMAN_OOD_THRESHOLD=0.99` (always-fire-approx).

Per scope-lock §4 Step 0 verification text: *"Gate (3) is informational: confirms Koopman path reachable; β=0.99 is force-fire; production β=0.05 is Branch B prior."* The OVERALL gate is `gate_1 AND gate_2` (scope-lock specifies Gate 2 failure as the STOP-and-surface condition; Gate 3 failure is documented but not blocking).

**Significance of Gate 3 failure (preview for Step 1):** even at β=0.99, the Koopman engine was not entered for any of the 11658 cublasGemmEx calls in the worker. The substrate's MATMUL dispatcher classified 100% of decode GEMMs as PASSTHROUGH (`handled=0 passthrough=11658`). This is upstream evidence that on real TinyLlama V1 decode, Koopman either (a) doesn't reach the engine entry for any shape class (classifier filter at the MATMUL dispatcher level), or (b) the OOD residual_ratio is > 0.99 for every GEMM (no β in the sweep range will produce positive utility).

This is consistent with the W14 Step 2 E architectural-ceiling prior and presages Branch B outcome at Step 1 calibration. It does NOT bypass Step 1 — formal sweep with `cipher_rt_koopman_calls_total` and `_skipped` accounting is required for the documented gate evaluation.

**Full result:** `option2/step_0/verify_result.json`.

### Overall: PASS

```json
{
  "gate_1_library_load": true,
  "gate_2_counter_delta": true,
  "gate_3_handled_at_beta_099": false,
  "overall": true
}
```

Per scope-lock semantics: Step 1 unblocked.

---

## 3. Honest residue captured per scope-lock §7 #6 (Goal 5 framing)

Per Anil 2026-05-25 amendment, Step 0 surfaces a Goal 5 pitch-language residue:

> Memory #1's framing **"LD_PRELOAD-only deployment, zero customer code changes"** is incomplete as stated for vLLM V1 deployments. The cipher_vllm_plugin worker-init hook (this Step 0) is a deployment-side requirement (a small plugin install, not application/model code changes). The honest post-Step-0 reframe is: **"LD_PRELOAD + drop-in cipher_vllm_plugin, zero application code changes."**

This step doc is the record per scope-lock §7 honest residue #6. Pitch-language reconciliation deferred to post-Branch resolution at Step 4 close-out.

---

## 4. Surprises and process notes

### 4.1. Diagnosis reconciliation — CUDA_INJECTION64_PATH question

During Step 0 prep, advisor flagged: if the prior null-measurement attempt had `CUDA_INJECTION64_PATH` set, the CUDA driver would have auto-invoked `InitializeInjection2` in the worker via cuInit, making the diagnosis "worker never runs InitializeInjection2" incorrect. Inspected `/tmp/step13_3_baseline/option2/run_all.sh` (the prior runner script): uses `LD_PRELOAD=$LIB` only, no `CUDA_INJECTION64_PATH`. Diagnosis stands; Step 0 is the right fix.

Also surfaced: `cipher-fusion-evidence/week6/bench_v2/bench_llm.py` line 786-789 auto-sets `CUDA_INJECTION64_PATH` when `--cipher` is on. If a future campaign uses bench_llm.py with --cipher, the driver-invoked path is the active one and Step 0 plugin hook becomes belt-and-suspenders. Both paths now ship.

### 4.2. Scope-lock addendum during build

Strengthened scope-lock §4 Step 0 verification gates per advisor review (committed at `cipher-fusion-evidence` `7b54546`): added Gate (2) counter-delta discriminator because Gate (1) `/proc/<worker_pid>/maps` presence was necessary-not-sufficient (LD_PRELOAD inherits across spawn; the failing prior attempt also had libcipher_rt.so visible in worker /proc/maps). Same hygiene as W14 Step 2 addendum 6be1d4f and W14 Step 3 S3.C addendum ef8b830.

### 4.3. Commit hygiene error (acknowledged)

The scope-lock addendum commit `7b54546` used `git commit -am` which swept up 35 unrelated stale `phase_c/*.log` modifications from earlier sessions into a commit intended doc-only. Commit message says "scope-lock document only" but diff shows otherwise. No data harmed (logs from synthetic test runs; not sensitive); saved as feedback memory `cipher-evidence-commit-discipline` to prevent recurrence. Not reverted per safety protocol (destructive ops need user consent). Surface for user adjudication if cleanup-commit desired.

### 4.4. KV-bridge VA pool exhaustion (unrelated; isolated by CIPHER_KV_ALLOC=0)

First verification run crashed with `cipher_kv_bridge: slab_create failed (VA pool exhausted?)`. This is independent of Step 0 — the CIPHER KV-allocator's VA pool reservation was set too small (or stale from prior sessions). Isolated by setting `CIPHER_KV_ALLOC=0` in the verification harness; the Step 0 GOT-patch hook runs BEFORE the `_enabled()` check, so verification still tests the worker GOT path properly. KV-bridge issue is out-of-scope for Step 0 and queued for Step 1 entry checklist (Step 1 needs functional KV cache so the issue must be resolved or worked around there).

### 4.5. Counter-dump bug caught and fixed mid-build

First-pass `_install_counter_dump_for_verification()` only tried `ctypes.CDLL("libcipher_rt.so")` SONAME and failed in worker with "libcipher_rt.so not loadable". Fixed by mirroring `_install_cipher_rt_got_patches()` fallback chain (SONAME + absolute `/home/ubuntu/cipher_rt_phase4/libcipher_rt.so`). Plugin md5 advanced from `d41d7fbc...` (broken) to `e77a3a58...` (fixed).

---

## 5. Substrate state at Step 0 close

| Tree | Commit | Notes |
|---|---|---|
| `cipher_rt_phase4` | `25970f3` | unchanged (`week-13-14-complete`) — no substrate change in Step 0 |
| `cipher_kmod` | `8c643fc` | unchanged (`week-13-14-complete`) — no ABI change in Step 0 |
| `cipher_kv_bridge.so` | W12 Step 3 anchor | unchanged |
| `cipher_vllm_plugin/cipher_vllm_kv.py` | md5 `e77a3a58a5f075985c838110f9193585` | NEW for Step 0; snapshot at `plugin_snapshots/cipher_vllm_kv.py.w14_followup_step_0`; +~80 LOC across two functions |
| `cipher-fusion-evidence` | this commit lands `option-2-step-0-vllm-worker-init-hook` tag | `WEEK_14_FOLLOWUP_OPTION_2_STEP_0_VLLM_WORKER_HOOK.md` + `option2/step_0/` artifact tree |

---

## 6. Step 0 ED actual vs budgeted

**Budget:** 1-2 ED (scope-lock §4 Step 0 HARD BUDGET)

**Actual:** ~1 ED equivalent in one session (orientation + impl + addendum + verification + step doc). Within budget.

---

## 7. Step 1 prerequisites — entry checklist for next session

1. **Resolve KV-bridge VA pool exhaustion** (or work around with `CIPHER_KV_ALLOC=0` if Step 1 calibration doesn't need CIPHER KV allocator). Step 1 needs functional KV cache. Investigation candidate: stale VA reservation from prior session, or pool-size config update needed.
2. **Confirm bench_llm.py harness path** for Step 1: in-process AsyncLLMEngine (this verification's pattern) vs subprocess vLLM server vs custom phase_runner — pick one based on the calibration sweep methodology in scope-lock §4 Step 1.
3. **Substrate-overhead microbench prerequisites** per §10 Q1 (a): synthetic 1k-GEMM workload at `CIPHER_KOOPMAN=1 β=0.99` vs `CIPHER_KOOPMAN=0`. Existing test infra reuse candidate: `cipher_rt_phase4/test_g3_cross_model_keying` or `test_audit_chain` patterns.
4. **Capture Step 0's gate-3 finding** as Step 1 calibration baseline: even at β=0.99 on real TinyLlama V1 decode, the MATMUL dispatcher classified 100% PASSTHROUGH. Step 1 calibration formal sweep must surface why — classifier filter at dispatcher level, or residual_ratio > 0.99 on real inputs.

---

## 8. Related memory

- `vllm-v1-worker-subprocess` — the discovery this step addresses
- `cipher-fresh-session-anchor-verify` — verified at Step 0 start
- `cipher-evidence-commit-discipline` — surfaced during Step 0 build
- `w14-step-3-followup-mistral-tok-s` — queued-campaign pointer; Step 0 unblocks Step 1
- `w13-14-complete` — predecessor anchor

---

**Scope-lock §7 honest residue #6 (Goal 5 framing) — CAPTURED.**

**Step 1 prompt:** calibration gate on TinyLlama warmup per scope-lock §4 Step 1 (sweep β, +overhead microbench per §10 Q1 (a)).

Awaiting Anil adjudication for Step 0 close + Step 1 commencement.
