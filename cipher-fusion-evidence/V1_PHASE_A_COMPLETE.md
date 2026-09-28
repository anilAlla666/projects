# V1 Phase A COMPLETE (Branch D (a))

**Date:** 2026-05-26
**Close tag:** `v1-substrate-driver-worker-init` on cipher_rt_phase4 `8613812e`
**Substrate state at close:**
- cipher_rt_phase4 `8613812e` (rotated from `52923af`)
- libcipher_rt.so md5 `1d91e7da` (rotated from `4bedf648`)
- cipher_kmod `8c643fc` UNCHANGED (no kmod ABI delta in Phase A per scope-lock §3)
- cipher_vllm_plugin/cipher_vllm_kv.py md5 `b89a9b6e` UNCHANGED (probe hook from CP 5.5 pre-work env-gated; plugin status now OPTIONAL not REQUIRED for vLLM V1 deployment via Branch D (a) path)
- cipher-fusion-evidence this commit (close-out doc + ledger row update)

**Authority:** Anil 2026-05-26 V1 Phase A scope-lock (cipher-fusion-evidence `0eab9ef`) + Branch D (a) adjudication after Branch B HARD GATE FAIL surfaced the modern-CUDA cuGetProcAddress mechanism.

---

## 1. Branch D (a) outcome statement

V1 Phase A driver-level worker-init landed at cipher_rt_phase4 `8613812e` tag `v1-substrate-driver-worker-init`. For vLLM V1 deployments specifically, modern CUDA's lazy driver-symbol loading via `cuGetProcAddress` bypasses LD_PRELOAD interception by design. The locked Goal 5 contract preserves the "or" disjunction: customer sets `LD_PRELOAD=libcipher_rt.so` OR `CUDA_INJECTION64_PATH=libcipher_rt.so`. For vLLM V1, customer chooses `CUDA_INJECTION64_PATH=/path/to/libcipher_rt.so`. Counter delta verification confirms `InitializeInjection2` returns 1 with driver-mediated invocation; A.2 hard gate worker `cipher_rt_cublas_shim_calls` reaches 19090 (gate >= 11000) on TinyLlama-1.1B vLLM V1 N=1 128-token decode, plugin uninstalled. For non-vLLM-V1 CUDA workloads (custom inference servers, direct cuBLAS usage via the standard linker-resolved symbol path), the constructor plus cuInit-wrapper provides LD_PRELOAD-only operation. Both paths exercise the same substrate via the same `InitializeInjection2()` entry point. Goal 5 contract preserved per the 2026-05-26 lock.

---

## 2. What shipped

| Component | File | LOC | Purpose | Belt-and-suspenders? |
|---|---|---|---|---|
| Constructor worker-init | `cipher_inject.c:91-167` | ~80 (additive) | LD_PRELOAD-time force-dlopen libcublas/libcudnn + InitializeInjection2 | YES (non-vLLM-V1 paths) |
| Counter-dump port | `cipher_rt_counter_dump.{c,h}` | ~170 | substrate-side port of plugin's `_install_counter_dump_for_verification`; env-gated `CIPHER_RT_COUNTER_DUMP_PATH`; SIGUSR1 + atexit JSON dump | NO (independent verification tool; Phase B-D harnesses depend on it) |
| cuInit LD_PRELOAD wrapper | `cipher_rt_cuinit_hook.c` | ~150 | Branch B mechanism; exports cuInit T-symbol; force-dlopen + InitializeInjection2 + GOT-patch refresh + dlsym RTLD_NEXT chain; pthread_atfork child handler resets once-token on fork | YES (non-vLLM-V1 paths where direct symbol resolution works) |
| Makefile additions | `Makefile` | ~5 lines | new OBJS entries + build rules | n/a |

ABI delta is additive-only per `cipher-abi-rule`:
- New T-symbols: `cipher_rt_counter_dump_init`, `cuInit` (LD_PRELOAD interposition standard pattern)
- New env gates: `CIPHER_RT_COUNTER_DUMP_PATH`, `CIPHER_RT_DISABLE_AUTO_INIT`, `CIPHER_RT_DISABLE_CUINIT_HOOK`
- New static constructor `cipher_rt_auto_init_worker` (local symbol, fires at LD_PRELOAD load)

---

## 3. R-A.1 + R-A.2 mechanism finding (load-bearing for the v1 narrative)

V1 Phase A.1 surfaced two structural findings via empirical A.2 hard gate runs:

**R-A.1 (Branch A constructor structurally inadequate for vLLM V1 fork-based worker):**
- vLLM V1 forks the EngineCore worker subprocess by default per `vllm/envs.py:65` (`VLLM_WORKER_MULTIPROC_METHOD=fork`). `_maybe_force_spawn()` at `vllm/utils/system_utils.py:126-165` only escalates to spawn if CUDA is already initialized in parent at fork time, which is not the default state.
- `fork()` does not re-run constructors per POSIX. Worker inherits parent's already-mapped libcipher_rt.so but the constructor only fires once-per-process at LD_PRELOAD load time in the parent.
- At parent constructor time (LD_PRELOAD load), only ~24 base modules are loaded (libc, libm, libpthread). libcublas is loaded LATER when vLLM imports torch. The constructor's GOT-patch finds 1 slot to patch across the 24 modules, none of which are cublas callers.
- Empirical evidence: A.2 hard gate run with `LD_PRELOAD=libcipher_rt.so` only (no `CUDA_INJECTION64_PATH`, no plugin) shows worker `cipher_rt_cublas_shim_calls=0`, `got_slots_patched=1`, `got_modules_scanned=24`. Worker is verified to have libcipher_rt.so mapped via `/proc/<worker_pid>/maps` but no constructor re-fire happens in child.

**R-A.2 (Branch B cuInit LD_PRELOAD wrapper bypassed by modern CUDA lazy resolution):**
- Modern CUDA runtime (CUDA 11.3+) uses lazy driver-symbol loading via `cuGetProcAddress` / Driver Entry Point Access API per NVIDIA documentation.
- libcudart has zero undefined references to `cu*` driver symbols (verified `nm -D libcudart.so.13`).
- PyTorch and vLLM access the CUDA driver through libcudart. libcudart queries the driver via `cuGetProcAddress` and resolves function pointers directly from the driver's internal table.
- These pointers bypass LD_PRELOAD's symbol-resolution mechanism entirely. Verified via two minimal C tests:
  - `/tmp/test_cuinit_direct.c` (direct extern `cuInit` with `-lcuda`): our LD_PRELOAD wrapper fires; log line `[cipher_v2] cuinit-hook: setup complete pid=2060901` appears; GOT-patch refresh runs.
  - `/tmp/test_cuinit_intercept.c` (explicit `dlopen(libcuda) + dlsym(libcuda_handle, "cuInit")` then call): wrapper bypassed entirely; dlsym returns libcuda's internal cuInit address directly (`0x7884e1105cb0`); our log does NOT appear; real cuInit returns 0.
- Empirical A.2 evidence: with `LD_PRELOAD=libcipher_rt.so` + Branch B cuInit-wrapper exported as T-symbol at `0x1b680`, worker `cipher_rt_cublas_shim_calls` STILL = 0; `cuinit-hook: setup complete` log NEVER appears for worker pid. PyTorch's CUDA init path goes through cuGetProcAddress-mediated resolution that LD_PRELOAD cannot intercept.

**The Branch D (a) resolution:**
- `CUDA_INJECTION64_PATH` is the driver-mediated active-invocation equivalent. The CUDA driver itself reads the env var at cuInit time and explicitly invokes `InitializeInjection2()` on the named library regardless of dlsym or lazy-resolution paths. The driver actively calls us; we do not intercept it.
- Verified A.2 PASS under CUDA_INJECTION64_PATH: worker shim_calls = 19090, matmul_total = 19090 (gate >= 11000). Plugin uninstalled. vLLM V1 default mode (VLLM_COMPILE, fork).
- For non-vLLM-V1 CUDA workloads where the application code uses direct extern `cuInit` (resolved via the standard linker) rather than `dlsym(libcuda_handle, ...)`, the constructor + cuInit-wrapper belt-and-suspenders provides LD_PRELOAD-only operation. Both deployment paths trigger the same `InitializeInjection2` substrate entry, exercising the same actuator surface.

---

## 4. A.2 hard gate verdict (CUDA_INJECTION64_PATH path, Branch D (a) deployment)

| Variant | Path | shim_calls (worker) | matmul_total (worker) | Gate verdict |
|---|---|---|---|---|
| LD_PRELOAD-only, plugin uninstalled | `LD_PRELOAD=libcipher_rt.so`, NO `CUDA_INJECTION64_PATH` | 0 | 0 | FAIL by R-A.1 + R-A.2 mechanism |
| CUDA_INJECTION64_PATH, plugin uninstalled | `CUDA_INJECTION64_PATH=libcipher_rt.so`, NO `LD_PRELOAD` | **19090** | 19090 | **PASS** (gate >= 11000) |

Evidence: `v1_phase_a/results/a2_cuda_injection_PASS.json`. Read mechanism: collective_rpc-based live counter read (worker is SIGKILL'd at vLLM shutdown so atexit dump does not fire; live read via ctypes against explicit libcipher_rt.so path).

---

## 5. A.3 regression measurement

Per `v1_phase_a/results/a3_summary.md`:

| Test | Result | Note |
|---|---|---|
| A.2 hard gate under CUDA_INJECTION64_PATH | PASS shim_calls=19090 | substrate works on Branch D (a) path |
| W14 Step 3 test_step3_b0_producer | PASS p99 81 ns, rate 25.23 M/s | substrate integrity preserved at new commit |
| W14 Step 3 test_step3_b1_consumer | PASS cold-start + compose drop_pct 0.00% | same |
| Track 2 SC2-SC6 e2e four-tenant Mistral-7B | DEFERRED to Phase B entry | per Anil Q4 (a) confirmed; expected substrate-only baseline ~0% cross-tenant savings by mechanism (plugin's `_cipher_allocate_kv_cache_tensors` is what produces 76% number) |
| Track 3 v1 SC1-SC6 DSM correctness | DEFERRED to Phase B entry | kmod unchanged at `8c643fc`; no mechanism reason to expect regression; batched with Track 2 re-run at Phase B entry |
| W7-12 other microbenches (test_commit_atomicity, test_audit_chain, test_resolver, etc.) | NOT re-run this turn | substrate changes additive to specific .o; existing source for those microbenches unchanged |

Honest residue: full Track 2 SC6 measurement requires plugin-dependent orchestration that is now uninstalled per Branch D (a). The mechanism is dispositive (plugin hook never fires, savings collapse). Phase B entry checklist gets a "run Track 2 SC6 under no-plugin path first" item.

---

## 6. Goal 5 deployment-layer ledger (rows landed in same commit per ledger-gate discipline)

| Phase | Customer step | When added/removed | Contract violation? | Status |
|---|---|---|---|---|
| Phase A close | + set `CUDA_INJECTION64_PATH=libcipher_rt.so` (vLLM V1) OR `LD_PRELOAD=libcipher_rt.so` (other CUDA workloads) | Phase A close 2026-05-26 | NO (contract "or" disjunction preserved; one env var either way) | LANDED |
| Substrate mechanism note | R-A.1 + R-A.2 finding 2026-05-26: vLLM V1 modern CUDA uses cuGetProcAddress lazy resolution which bypasses LD_PRELOAD cuInit interception. CUDA_INJECTION64_PATH is the driver-mediated active-invocation equivalent. Constructor + cuInit-wrapper retained as belt-and-suspenders for non-vLLM-V1 deployments. | Phase A.1 Branch B mechanism finding 2026-05-26 | N/A (substrate-internal architecture) | informational |

`V1_GOAL5_DEPLOYMENT_LEDGER.md` updated in this same commit.

---

## 7. Plugin disposition (Branch D (a) close)

- cipher-vllm-kv was pip-uninstalled from `vllm_env` Python through A.1 + A.2 + A.3 per Anil Item 5
- Reinstall after this close-out commit per Anil Item 5 directive: restore baseline-as-shipped state for any future debug work and for Phase B development
- Reinstall command: `cd /home/ubuntu/cipher_vllm_plugin && /home/ubuntu/vllm_env/bin/pip install -e .`
- Plugin status semantically: OPTIONAL for v1 deployment via Branch D (a) path; INSTALLED for dev / debug / Phase B preparation
- Goal 5 contract NOT violated by plugin being installed during dev (the contract is about CUSTOMER deployment surface, not engineering workstation state)

---

## 8. Risk register status at A.4 close

| ID from scope-lock §6 | Status at A.4 |
|---|---|
| R-A.1 constructor ordering | FIRED, surfaced, Branch B fallback triggered, Branch B FIRED via R-A.2, Branch D (a) resolution |
| R-A.2 worker non-cuInit path | FIRED (cuGetProcAddress); resolved via CUDA_INJECTION64_PATH driver-mediated path |
| R-A.3 constructor in non-CUDA process | UNCHANGED LOW; smoke /bin/true PASS shows no crash |
| R-A.4 CUDA_INJECTION64_PATH double-init | UNCHANGED LOW; idempotency via pthread_once verified |
| R-A.5 other LD_PRELOAD libs break | UNCHANGED LOW; no torch-only standalone regression observed |
| R-A.6 A.2 hard gate fails | FIRED twice (Branch A + Branch B), resolution Branch D (a) per scope-lock §8 Branch D framing |
| R-A.7 Track 2 SC6 regression (reframed) | EXPECTED outcome per mechanism; deferred-measurement landed as Phase B entry checklist |
| R-A.8 paperwork miss | NOT FIRED; A.0 grep verified empty reframe-references post-revert |

---

## 9. Phase A close substep map

| Substep | Commit | Tag |
|---|---|---|
| Scope-lock | cipher-fusion-evidence `0eab9ef` | (no tag) |
| A.0 paperwork (contract lock revert + ledger creation) | cipher-fusion-evidence `95a2d0f` | `v1-goal5-contract-lock` |
| A.1 substrate code (constructor + counter-dump + cuInit-wrapper + Makefile) | cipher_rt_phase4 `8613812e` | `v1-substrate-driver-worker-init` |
| A.3 regression summary | `v1_phase_a/results/a3_summary.md` (lands in close-out commit) | n/a |
| A.4 close-out (this doc + ledger update) | cipher-fusion-evidence this commit | n/a (close-out is bookkeeping; substrate tag on cipher_rt_phase4 is the load-bearing tag) |

---

## 10. Next phase: B authorization gated by Anil

Phase A is closed. Phase B per Anil 2026-05-26 V1 substrate work sequence: "Move plugin functionality into substrate (~1-2 weeks). Scope: cipher_vllm_kv.py today handles KV-buffer ownership (CP 5.1) and KV-dedup auto-trigger (W6 carry). Both can move into libcipher_rt.so as CUDA driver-level hooks on cudaMalloc / cuMemAlloc / cuMemcpyDtoD. Plugin becomes optional / deprecated for v1 deployment."

Phase B entry prerequisites (folded into Phase B scope-lock when drafted):
1. Plugin reinstall (per Anil Item 5 directive)
2. Run Track 2 SC2-SC6 under no-plugin path first (records the substrate-only baseline that Phase B uplifts from; the A.3 deferral lands here)
3. Run Track 3 v1 SC1-SC6 under no-plugin path
4. Phase B substrate-migration design memo

Phase B does NOT start without explicit Anil "proceed to Phase B" signal.

---

## 11. Related memory

- [[v1-goal5-contract-lock]]: Phase A contract baseline and V1 substrate work sequence
- [[option2-complete]]: Branch B is what the v1.5 BF16 port queue at §6 inherits when it becomes the active substrate path post-Phase-D
- [[option2-step0-closed]]: prior plugin install of InitializeInjection2; Branch D (a) leverages the same `InitializeInjection2` entry via CUDA driver invocation
- [[vllm-v1-worker-subprocess]]: original worker-init gap; Phase A closes for vLLM V1 via CUDA_INJECTION64_PATH path and for non-vLLM-V1 paths via constructor + cuInit-wrapper
- [[plan-v1.2.3]]: five product goals; Goal 5 contract preserved
- [[cipher-track2-weight-sharing]]: Track 2 SC6 76% baseline; A.3 measurement deferred to Phase B entry per R-A.7
- [[cipher-track3-dsm]]: Track 3 v1 SC1-SC6; deferred batch with Track 2
- [[cipher-abi-rule]]: ABI-additive only at A.1 commit
- [[cipher-evidence-commit-discipline]]: followed at A.0 paperwork commit, A.1 substrate commit, A.4 close-out commit
- [[cipher-fresh-session-anchor-verify]]: anchor verification done at A.1 entry
- [[cipher-proceed-not-ask]]: Anil "Proceed to A.1" + "Proceed to Branch B" + Branch D (a) confirmation each executed without re-litigation
- [[pre-cp-5-5-plugin-routed-probe]]: prior plugin-routed mechanism investigation; ruled out apply-wrapping; informed Goal 5 cumulative-drift catch that motivated Phase A
