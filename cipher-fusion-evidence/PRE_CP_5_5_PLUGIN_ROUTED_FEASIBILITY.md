# CP 5.5 PRE-WORK plugin-routed Koopman feasibility probe

**Date:** 2026-05-26
**Scope:** Phase 1.5 of CP 5.5 scope-lock pre-work. Pre-Phase-2 substep per Anil adjudication 2026-05-26 of CP 5.5 Phase 1 surface Item B (BF16 vs FP8 vs plugin-routed-Koopman strategic call).
**Pre-state anchor:** `option-2-complete` (verified on disk at probe start)
- `cipher_rt_phase4` `52923af` tags `option-2-complete` + `option-2-step-1-alpha-dtype-counter`
- `libcipher_rt.so` md5 `4bedf6488c9a7c0ae7eb793794b02907`
- `cipher_kmod` `8c643fc` tags `week-13-14-complete` + `week-9-complete`
- `cipher_vllm_plugin/cipher_vllm_kv.py` md5 `8330506a` at probe start
- `cipher-fusion-evidence` `081dc7f` at probe start (W14 option-2 close-out §12 addendum)

**Authority:** Anil adjudication 2026-05-26 of CP 5.5 Phase 1 surface response §B 4th option ("Plugin-routed Koopman") and the named feasibility probe scope.

**Type:** read-only architectural probe with one ABI-additive plugin edit. No substrate (libcipher_rt.so / cipher_kmod) code change. Tag `cp-5-5-pre-plugin-routed-feasibility`.

---

## 1. Probe scope (per Anil 2026-05-26)

> Build minimal plugin hook in `cipher_vllm_plugin` that invokes a no-op
> substrate function (Python-to-C round trip only, no Koopman compute, no
> allocation, just call-and-return) at every per-layer matmul in vLLM
> model-runner forward path.

Three variants:
- `baseline`: no hook (production vLLM behavior unchanged)
- `python`: Python wrapper around `UnquantizedLinearMethod.apply` with per-call counter increment, no C crossing
- `ctypes`: Python wrapper plus single `ctypes` call to `cipher_rt_koopman_skip_dtype` (atomic uint64 load, no allocation)

Workload: TinyLlama-1.1B-Chat-v1.0 FP16 vLLM N=1 decode, 128 generated tokens, 7 reps per variant (rep 0 dropped as warmup), GPU clocks at 1830 MHz precedent (Option 2 anchor).

Gate per Anil scope:
- Overhead < 1%: plugin-routed Koopman viable, Item B answered
- Overhead 1-5%: marginal, surface to Anil
- Overhead > 5%: dead, fall back to GOT-patch per dtype

Hard budget: 2-3 ED. Surface if exceeded OR if unexpected substrate work surfaces.

---

## 2. Hook insertion point identified (no ambiguity)

`vllm.model_executor.layers.linear.UnquantizedLinearMethod.apply` at `vllm/model_executor/layers/linear.py:226-232`. Single hook captures all three production Linear forward methods via the `self.quant_method.apply(self, x, bias)` indirection:

- `ReplicatedLinear.forward` at `linear.py:392-401` calls `self.quant_method.apply(self, x, bias)` at L396
- `ColumnParallelLinear.forward` at `linear.py:582-604` calls `self.quant_method.apply(self, input_, bias)` at L588
- `RowParallelLinear.forward` at `linear.py:1544-1574` calls `self.quant_method.apply(self, input_parallel, bias_)` at L1561

`LinearBase.__init__` at `linear.py:262-285` instantiates `quant_method = UnquantizedLinearMethod()` at L277 when `quant_config is None` (TinyLlama-1.1B unquantized path). Probed via vLLM `collective_rpc` in the worker subprocess: 88 LinearBase instances (22 QKVParallelLinear, 44 RowParallelLinear, 22 MergedColumnParallelLinear), all 88 use `UnquantizedLinearMethod`. Class identity verified at runtime via `id(type(instance)) == id(patched_class)` (both equal in worker subprocess pid).

No multi-candidate ambiguity surfaced. Anil's "surface if multiple candidates" did not trigger.

---

## 3. Plumbing finding (separate from overhead finding)

Plugin install at `cipher_vllm_kv.py` `register()` time DOES install the wrapper on the worker subprocess's `UnquantizedLinearMethod.apply`. Class identity confirmed via `collective_rpc` inspection. Yet `crossings_total = 0` after 7 reps × 128 tokens of vLLM `generate()`. The wrapper is on the class but never invoked.

Diagnostic: when the same wrapper is installed via `llm.collective_rpc(install_fn)` AFTER `LLM()` returns (post-model-load), the wrapper fires correctly. Verified at `tokens=64`: `crossings = 5632 = 88 layers × 64 tokens`, exact match.

For the probe, this is a plumbing issue, not the load-bearing finding. The probe harness was modified to install via `collective_rpc` post-`LLM()` so accurate measurements could be taken. The plumbing finding is recorded here so any production plugin-routed Koopman implementation at this surface would need to install via a deferred hook (analogous to `_cipher_allocate_kv_cache_tensors` in the existing plugin, which fires post-model-load).

---

## 4. Load-bearing finding (the actual probe verdict at this hook surface)

### 4.1 Production mode (`enforce_eager=False`, V1 default, `VLLM_COMPILE` active)

| Variant | Mean tok/s | StDev | Crossings (6 reps × 128 tok) | tok/s Δ vs baseline |
|---|---|---|---|---|
| baseline | 579.66 | 1.88 | 0 | n/a |
| python | 591.81 | 1.12 | **0** | +2.09% |
| ctypes | 583.39 | 2.43 | **0** | +0.64% |

Hook installed correctly (verified via `collective_rpc` class-identity inspection) but `crossings = 0` for both `python` and `ctypes` variants across all 6 measured reps. The wrapped Python `apply()` function is never invoked during vLLM V1 production-mode `generate()`.

`tok/s` deltas (+2.09% python, +0.64% ctypes) are run-to-run noise. The hook never fires, so no overhead is being measured. These deltas are reproducibly within the C.3 vanilla self-consistency drift floor (`week6-bench-harness` documents ~0.5% power drift, similar magnitude on tok/s).

**Production-mode verdict at this hook surface: the Python `apply()` call is bypassed in V1 production execution.** The hook surface as wrapped is not reachable.

### 4.2 Eager mode (`enforce_eager=True`, no compile, no cudagraph)

| Variant | Mean tok/s | StDev | Crossings (6 reps × 128 tok) | tok/s Δ vs baseline |
|---|---|---|---|---|
| baseline | 107.98 | 0.26 | 0 | n/a |
| python | 104.98 | 0.44 | **67584** | **-2.78%** |
| ctypes | 104.52 | 0.46 | **67584** | **-3.20%** |

Hook fires correctly. `crossings = 67584 = 88 calls/token × 128 tokens × 6 measured reps`, exact match.

Per-call overhead derived from `(elapsed_hooked - elapsed_baseline) / tokens / calls_per_token`:
- python: **3.01 µs per call**
- ctypes: **3.48 µs per call** (ctypes call adds ~0.47 µs over Python-only)

Eager-mode overhead falls in Anil's MARGINAL gate (1-5%).

### 4.3 The eager-mode baseline is 5.4× slower than production

| Mode | Baseline tok/s |
|---|---|
| Production (V1 default) | 579.66 |
| Eager (no compile, no cudagraph) | 107.98 |

Eager mode loses 81% of production throughput. The 3.0-3.5% overhead measurement on eager mode does not transfer to production decisions because eager mode is not production-viable for vLLM V1 (graph optimizations are load-bearing).

### 4.4 Mechanism isolated (observed, not inferred)

A third configuration was tested to isolate which V1 default behavior causes the bypass:

| Config | `CompilationMode` | `enforce_eager` | cudagraph | Crossings (1 rep × 64 tok) |
|---|---|---|---|---|
| Production (default) | `VLLM_COMPILE` | False | enabled | 0 |
| Compile off, cudagraph on | `NONE` | False | enabled | **5632** (exact 88×64) |
| Eager | `VLLM_COMPILE` | True | disabled | 5632 (exact 88×64) |

**Result: `torch.compile` / `VLLM_COMPILE` is the bypass cause, not cudagraph.** With `CompilationMode.NONE` and cudagraph still enabled, the Python `apply()` call fires exactly the expected count. This is consistent with `torch.compile` tracing the model forward, inlining the `self.quant_method.apply(self, x, bias)` call into the compiled FX graph, and dispatching the inlined matmul directly per replay rather than re-invoking Python `apply()`.

This is observed evidence, not inferred mechanism. The throughput cost of `CompilationMode.NONE` versus default `VLLM_COMPILE` was NOT measured in this probe and is a separate question (out of scope per Anil's "stop and surface, do not silently extend" rule).

---

## 5. Probe verdict against Anil's gate

| Mode | Hook fires? | Overhead | Gate verdict |
|---|---|---|---|
| Production (V1 default) | NO (compile bypass) | unmeasurable | gate is mis-calibrated for this finding type |
| Eager | YES | 3.0-3.5% | MARGINAL band, mode not production-viable |
| Compile-off + cudagraph | YES (verified) | UNMEASURED in this probe | open follow-on |

**Strict gate application**: production-mode "overhead" measured as noise (+0.64% to +2.09%) is < 1% in magnitude but is meaningless because the hook never fires. Mechanically gate-passing on this number would commit Phase 2 to a plugin-routed approach at this surface whose actuator cannot fire in production.

**Honest gate application**: plugin-routed Koopman as wrapped at `UnquantizedLinearMethod.apply` is bypassed in V1 production at the default compilation mode. The "dead" gate (Anil scope >5% fallback condition) applies by structural reach rather than by overhead percentage. Anil's stated dead-path fallback was: "Overhead > 5% means plugin-routed path is dead. Fall back to GOT-patch per dtype. Phase 2 drafting proceeds with BF16-first as Item B answer."

The probe verdict is scoped to **this exact hook surface** (the apply-wrapping pattern). It does NOT rule out plugin-routed Koopman at other surfaces (see §6).

---

## 6. Adjacent paths NOT probed

These are NOT in scope for this probe per Anil's "stop and surface, do not silently extend" rule. Listed here so they are not re-derived later AND so they are not closed off by the §5 verdict above.

1. **Hook at a lower-level torch op (e.g., `torch.ops.aten.mm`, `F.linear`)**. `torch.compile` may preserve dispatches to registered custom ops. Plugin-routed Koopman as a custom `torch.ops.cipher.koopman_apply` op registered before compile would be compiled INTO the graph, then called per replay. Different surface, different probe.
2. **Inductor pre-pass / post-pass plugin**. `torch._inductor.config.fx_passes` or vLLM's `CompilationConfig.custom_ops` could insert Koopman dispatch into the compiled graph at compile time. Different abstraction layer, different complexity envelope.
3. **vLLM-native quantization method registration**. Add `CipherKoopmanLinearMethod(LinearMethodBase)` and register via `QUANTIZATION_METHODS`. vLLM would instantiate it at model load if `quant_config` selects it. **This pattern puts Koopman INSIDE the method's `apply` body, not in a wrapper around it.** The method's `apply` IS the call site that compile traces and inlines, so the Koopman dispatch is compiled into the graph by construction. The §5 verdict against the apply-wrapping pattern does NOT apply to this surface. Untested.
4. **`CompilationMode.NONE` deployment plus apply-wrapping plugin**. §4.4 verified the hook fires under `NONE` mode with cudagraph still on. The throughput cost of `NONE` mode versus default `VLLM_COMPILE` was not measured in this probe. If that cost is small (e.g., single-digit %), apply-wrapping plugin plus `NONE` mode is a real production option. Untested.
5. **Eager-mode-only deployment**. Accept the 5.4× throughput loss. Not recommended on its face, eliminates the production-mode advantage of vLLM almost entirely.
6. **GOT-patch per FP8 dispatch route** (Anil Item 1 fallback): `torch._scaled_mm`, `_C.cutlass_scaled_mm`, triton FP8 kernels, `deep_gemm`. The standard substrate intercept pattern, on different libtorch / vLLM `_C` GOT slots.
7. **GOT-patch on cuBLAS for BF16**: the Option 2 v1.5 BF16 port path already queued at `option2-complete` §6. Standard substrate pattern, well-understood, dtype-gate relaxation in `cipher_rt_koopman_engine.cpp` plus BF16 .cu kernel.

---

## 7. Substrate state at probe close

| Tree | Commit | md5 | Tag(s) | Δ vs probe entry |
|---|---|---|---|---|
| `cipher_rt_phase4` | `52923af` | `libcipher_rt.so` `4bedf648` | `option-2-complete`, `option-2-step-1-alpha-dtype-counter` | UNCHANGED |
| `cipher_kmod` | `8c643fc` | `cipher_kmod.ko` (unchanged) | `week-13-14-complete`, `week-9-complete` | UNCHANGED |
| `cipher_vllm_plugin/cipher_vllm_kv.py` | (file-only) | `8330506a` → `b89a9b6e` | snapshot at `plugin_snapshots/cipher_vllm_kv.py.pre_cp_5_5` | ROTATED (additive `_install_probe_hook()` + 1 call from `register()`) |
| `cipher-fusion-evidence` | this commit | n/a | tag `cp-5-5-pre-plugin-routed-feasibility` | ADDS this doc + `pre_cp_5_5/` probe artifacts + plugin snapshot |

Plugin edit is ABI-additive only:
- New env gate `CIPHER_PROBE_HOOK` (unset = no-op; `python` or `ctypes` = engages probe wrapper)
- New env gate `CIPHER_PROBE_HOOK_DUMP` (defaults to `/tmp`)
- New module-level state `_PROBE_HOOK_INSTALLED`, `_PROBE_CROSSINGS`
- One additional call in `register()` to `_install_probe_hook()`. When env unset, `_install_probe_hook()` returns immediately at line check. Zero production-path side effect.

The plugin-install path is documented per §3 as the structurally-bypassed path under the apply-wrapping pattern. The probe-harness `collective_rpc` install is the measurement path. Both are recorded so the production-vs-probe install distinction is unambiguous.

---

## 8. Reproducibility

Artifacts under `cipher-fusion-evidence/pre_cp_5_5/`:
- `probe_harness.py`: measurement script
- `run_probe.sh`: 3-variant driver
- `results/probe_result_production.json`: production-mode aggregate (V1 default)
- `results/probe_result_eager.json`: eager-mode aggregate
- `results/20260526_055349/`: production-mode raw per-variant JSONs
- `results/20260526_055558/`: eager-mode raw per-variant JSONs

Plugin snapshot at `plugin_snapshots/cipher_vllm_kv.py.pre_cp_5_5` mirrors the production plugin file at probe close (md5 `b89a9b6e`).

Re-run: `cd cipher-fusion-evidence/pre_cp_5_5 && VLLM_ALLOW_INSECURE_SERIALIZATION=1 ./run_probe.sh --reps 7 --tokens 128` for production. Add `--enforce-eager` for eager-mode comparison. Each run produces a timestamped subdirectory.

GPU state at probe time: H100 80GB SXM5, driver 580.105.08, kernel 6.8.0-1046-nvidia, gr_clock 1830 MHz, mem_clock 2619 MHz (Option 2 precedent), power 124-128 W idle, temp 39-40 °C.

---

## 9. Hard budget result

Wall-clock elapsed for the probe: ~50 minutes from anchor-verify to step-doc-draft (including the mechanism-isolation §4.4 follow-up). Well within the 2-3 ED budget. No silent extension.

The plumbing finding (§3) emerged mid-probe and was surfaced internally before the structural finding was committed. The structural finding (§4) was confirmed across production, eager, and compile-off+cudagraph modes with reproducible measurements.

---

## 10. Probe answer (what this probe established, what it did not)

**What the probe established:**
- Plugin-routed Koopman as an apply-wrapping pattern at `UnquantizedLinearMethod.apply` is bypassed in V1 production at default `VLLM_COMPILE` mode (§4.1).
- Eager-mode overhead at this same surface is 3.0-3.5% per call (§4.2) but eager mode costs 81% of production throughput (§4.3).
- The bypass is caused by `torch.compile` / `VLLM_COMPILE` specifically, not by cudagraph (§4.4).

**What the probe did NOT establish:**
- Whether plugin-routed Koopman at OTHER hook surfaces (§6 items 1-4) is viable.
- Whether the throughput cost of `CompilationMode.NONE` versus `VLLM_COMPILE` is acceptable (§6 item 4).
- Item B (BF16-first vs FP8-first vs both). Item B is a strategic call about 2026 versus 2027 customer positioning. This probe rules out exactly one path (apply-wrapping plugin-routed). It does NOT pick between Anil's three original options.

**Engineering recommendation (pre-probe, restated, not probe-supported):** BF16-first in v1, FP8 in v1.5. The engineering analysis behind this recommendation lives in the CP 5.5 Phase 1.5 response §B. The probe did not test FP8 or BF16 substrate paths.

**Anil's call for Phase 2 entry:**
- Pick between Item B options 1/2/3 (BF16-first / FP8-first / both) per Phase 1.5 surface, OR
- Authorize a follow-on probe for §6 item 3 (CipherKoopmanLinearMethod quant-method registration) or §6 item 4 (`CompilationMode.NONE` throughput cost), which would re-open the 4th option at a different surface, OR
- Both (parallel-track Phase 2 drafting on the BF16/FP8 decision while a follow-on probe runs).

---

## 11. Related memory

- [[option2-complete]]: substrate entry anchor, surfaces v1.5 BF16 port queue
- [[w13-14-complete]]: Koopman tier substrate baseline, FP16-only `.cu` kernel contract
- [[plan-v1.2.3]]: five product goals, Goal 4 Koopman in v1 narrow-domain
- [[week6-bench-harness]]: `bench_llm.py` methodology, ~0.5% drift floor cited at §4.1 noise interpretation
- [[cipher-evidence-commit-discipline]]: followed at the commit landing this doc + plugin edit
- [[cipher-proceed-not-ask]]: detailed Anil scope plus "Proceed with feasibility probe" then executed without re-litigation
