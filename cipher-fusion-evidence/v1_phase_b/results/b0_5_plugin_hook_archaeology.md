# V1 Phase B.0.5 plugin hook archaeology

**Date:** 2026-05-26
**Type:** read-only investigation, no substrate code change, blocks Phase B scope-lock drafting per Anil 2026-05-26 surface adjudication after B.0 finding contradicted CP 5.1 mechanism premise
**Budget:** 1-2 ED; this turn consumed ~0.5 ED (within budget)

---

## 1. What `_cipher_allocate_kv_cache_tensors` actually does at runtime (cipher_vllm_kv.py:305-397)

Reading the hook body from current source (cipher_vllm_kv.py:305-397 at md5 `b89a9b6e`):

**(a) W7-9 Step 1 G10 model registration** (lines 314-328): on first invocation per engine, calls `_register_model_with_kmod(model_path, hf_config)`. This issues `CIPHER_REGISTER_MODEL` ioctl (NR 27) to cipher_kmod, receives a stable 128-bit model_uuid keyed by `hf_config_hash`, caches the uuid at module scope. Downstream actuators (Marlin per `g4-marlin-tenant-kit`, KV-dedup per `g3-kv-dedup-model-keying`, Koopman per `g12-koopman-recipe-keying`) consume the uuid via cipher_kmod state.

**(b) W7-9 Step 5 stream registration** (lines 330-336): on first invocation, calls `_register_streams_with_kmod()`. This issues `CIPHER_REGISTER_STREAMS` ioctl (NR 29) to cipher_kmod with the worker's CUDA stream handles. Cipher_rt_phase4 uses this to resolve `(tgid, stream) -> tenant_id` in the cuBLAS dispatch hot path per `week-9-complete` multi-tenant resolver work.

**(c) W10-12 Step 3 G5 per-model VA sizing** (lines 338-364): pulls `max_model_len`, `dtype`, `max_batch`, `quantization` from `vllm.config.model_config` + `vllm.config.scheduler_config`. Calls `_ensure_bridge_init(hf_config=_hf, ...)` which invokes `cipher_kv_bridge.init(va_gib, ...)` with VA pool sized per the actual model footprint. Replaces the historical hardcoded 80 GiB default with per-model right-sizing (12.5x reduction at N=100 realistic mix per `g5-path-a-verified` memory).

**(d) CP 5.1 KV-cache buffer ownership transfer** (lines 366-397): the original CP 5.1 work. Replaces vLLM's `torch.zeros` allocation with `cipher_kv_bridge.vmm_zeros([nbytes], 1, "int8", _TENANT, 0, idx & 0xFFFF, 2)`. KV cache memory is now CIPHER-VMM-allocated (cuMemCreate-backed), page-tagged with tenant_id, and visible to cipher_rt_phase4 substrate routing. Per CP 5.1 close (`cipher-cp51-closed` memory) the gate measurement was correctness only (memo §4 #2): token-ID identity substrate-off vs -on (4 prompts × 64 tok greedy) on TinyLlama-1.1B and Mistral-7B-v0.1. Substrate-deliverable, not a measurement uplift.

## 2. What measurement(s) the hook is load-bearing for

Following the hook's documented chains via memory + source grep:

### 2.1 Week 5 cross-tenant KV-dedup (load-bearing, NOT what B.0 measured)

Per `week5-complete` memory + `cipher_vllm_kvdedup.py:7-9` docstring:

> "What it does: walks the CIPHER-VMM-backed KV pages allocated by CP 5.1's `cipher_vllm_kv` (which monkey-patches `GPUModelRunner._allocate_kv_cache_tensors` to source from `cipher_kv_bridge.vmm_zeros`), and on operator demand calls `cipher_rt_kv_dedup_alias` on each 2 MiB page."

This is unambiguous. Cross-tenant KV-dedup REQUIRES KV memory to be CIPHER-VMM-backed (via the plugin's `_cipher_allocate_kv_cache_tensors` hook). Without the hook, vLLM allocates KV via native `torch.zeros`; those pages are in PyTorch's caching allocator pool, NOT in `cipher_kv_bridge`'s VMM space; `cipher_rt_kv_dedup_alias` has nothing to walk.

Week 5 close headline numbers (per `week5-complete` memory §HEADLINE NUMBERS):
- **N=4 TinyLlama HBM saved: 45802 MiB (~45 GiB)** (Step 3.B)
- **N=2 TinyLlama HBM saved: 42306 MiB (~42 GiB)** (Step 2 exit gate)
- **N=4 Mistral-7B KL: 0.000000e+00** across all 6 pairs (Step 3.C bit-identity)
- **Hit rate: 83.20% at Mistral-7B N=4** (well above 60% gate)
- **False-positive rate: 0.015% absolute** at N=2 different-prompt
- Substrate: 67 unique physicals back 22968 virtuals at N=4 TinyLlama (4x sharing)

These numbers are the load-bearing Week 5 KV-dedup product story. They are DISTINCT from Track 2 SC6's 76% weight-sharing number. Both reproduce together under the plugin path. Track 2 SC6's 76% reproduces at substrate-only (B.0 finding). Week 5 KV-dedup numbers do NOT reproduce at substrate-only because the plugin's hook is the load-bearing chain link.

### 2.2 W7-9 Step 1 G10 model registration (load-bearing for vLLM workloads only)

Per `g10-abi-scaffold` memory: `CIPHER_REGISTER_MODEL` ioctl NR 27 + `model_registry` in cipher_kmod 0.5.5+. Plugin hook is where vLLM workloads issue this ioctl with the hf_config_hash. Without the plugin hook, vLLM workloads run with `model_uuid = MODEL_UNKNOWN` (all zeros) and:
- Marlin G4 weight-key falls back to single-model `(MODEL_UNKNOWN, w_ptr)` instead of `(model_uuid, w_ptr)` : limits the Marlin tenant-scope work
- KV-dedup G3 cross-model keying falls back to MODEL_UNKNOWN, weakening the algebraic substrate proxy per `g3-g4-tc-probe` Step 2
- Koopman G12 recipe registry falls back to MODEL_UNKNOWN, defeating the W13-14 model-keyed recipe seeding

These are all vLLM-workload-specific. Track 2 SC6 (uses transformers directly, not vLLM) is unaffected.

### 2.3 W7-9 Step 5 stream registration (load-bearing for vLLM workloads only)

Per `week9-complete` memory: `CIPHER_REGISTER_STREAMS` ioctl NR 29 + multi-tenant CUDA-stream resolver (tgid-keyed view-slot table, vmalloc_user'd 8192-slot open-addressing). Plugin hook is where vLLM workloads register their CUDA stream handles. Without the plugin hook, vLLM's substrate dispatch falls back to tenant=0 (per the docstring at `cipher_vllm_kv.py:336` "REGISTER_STREAMS skipped via exception ... tenant=0 fallback"). This defeats per-tenant routing in the cuBLAS dispatch hot path for vLLM workloads.

### 2.4 W10-12 Step 3 G5 per-model VA sizing (load-bearing for memory budget)

Per `g5-path-a-verified` + `w12-complete` memories: `compute_va_gib(hf_config, max_model_len, dtype_bytes, max_batch, quantization)` produces per-model VA right-sizing. N=100 realistic mix produces 640 GiB total VA, fitting 1 TiB envelope. Without the plugin hook, vLLM workloads use the hardcoded `va_gib=80` default from cipher_vllm_kv.py:58 or similar : would either over-reserve VA (per-process 80 GiB at N=100 = 8 TiB; exhausts host VA) OR not bring up KV bridge at all (depending on path).

Note: this is VA-pool sizing, not actual HBM. Different from Week 5 KV-dedup's HBM savings.

## 3. CP 5.1 close-out evidence

Per `cipher-cp51-closed` memory + `cp_5_1/CP_5_1_STEP_3_BUILD_LOG.md`:

CP 5.1 closed 2026-05-17 certifying memo §4 criterion **#2 only** (correctness): token-ID identity substrate-off vs -on. CP 5.1 explicitly did NOT certify criteria #1 (env-fail workloads), #3 (operator deployment recipe), or #4 (MFU/tok-W on live continuous-batching decode). The CP 5.1 close was a SUBSTRATE-DELIVERABLE close, NOT a measurement-uplift close.

The PURPOSE of the plugin hook at CP 5.1 was to make CIPHER own the KV memory so downstream actuators (Week 5 KV-dedup, future Koopman, future Marlin tenant-scope work) have a CIPHER-controlled buffer to operate on. The 76% Track 2 SC6 weight-sharing number was never a CP 5.1 deliverable; it's a Track 2 deliverable that uses cipher_kv_bridge weight arena directly without needing the CP 5.1 KV-cache hook.

**Anil 2026-05-26 V1 substrate work sequence spec's statement that "CP 5.1 KV-buffer ownership ... is the load-bearing path for Track 2 SC6's 76% N=4 Mistral-7B savings" appears to have conflated the cipher_kv_bridge weight arena (Track 2's load-bearing piece) with cipher_vllm_kv's CP 5.1 KV-cache hook (a separate chain).** Both use cipher_kv_bridge as the underlying VMM allocator, but Track 2 SC6 uses the weight-arena pattern (producer registers, consumers import via fd) while CP 5.1 uses the KV-cache buffer-ownership pattern (vmm_zeros at allocate time, page-tagged with tenant).

## 4. cipher_vllm_kvdedup.py (Week 5 / W6 auto-trigger) plugin role

Per `week6-kvdedup-autotrigger` memory + `cipher_vllm_kvdedup.py:254-352`:

The auto-trigger plugin runs in the EngineCore subprocess via `vllm.general_plugins`. It registers a wrapper on the already-CP-5.1-wrapped `_allocate_kv_cache_tensors` (R-W5.3 ordering enforced via setup.py entry-point order), records each returned tensor's VMM range, then via env-gated time-mode (`CIPHER_KVDEDUP_AUTOFLUSH`) or pressure-mode (`CIPHER_KVDEDUP_PRESSURE_THRESHOLD`) calls `cipher_rt_kv_dedup_alias` on each 2 MiB page when triggered.

Without cipher_vllm_kvdedup.py installed:
- No auto-trigger fires
- KV-dedup substrate (`cipher_rt_kv_dedup_alias` in cipher_kv_bridge.so) is never called on vLLM workloads
- Week 5 KV-dedup HBM savings DO NOT materialize on vLLM workloads regardless of cipher_vllm_kv.py state

So actually BOTH plugins (cipher_vllm_kv + cipher_vllm_kvdedup) are load-bearing for the Week 5 KV-dedup story. The chain is:
1. cipher_vllm_kv: makes KV memory CIPHER-VMM-owned (per `_cipher_allocate_kv_cache_tensors`)
2. cipher_vllm_kvdedup: walks that CIPHER-VMM memory and triggers substrate dedup (per `dedup_now()` time/pressure modes)
3. cipher_rt_kv_dedup_alias (substrate primitive in cipher_rt_phase4 ec0e005): does the cuMemUnmap + cuMemMap rebind

## 5. cipher_kv_offload.py (CP 5.2 KV offload) plugin role

Per `cipher-cp52-closed` memory + cipher_kv_offload.py:

CP 5.2 KV offload plugin monkey-patches THREE hooks: KV capture at `GPUModelRunner.initialize_kv_cache`, snapshot D2H on `Scheduler._preempt_request`, restore H2D on `Scheduler.schedule`. This is offload-to-host-DRAM under preempt; correctness gate PASS on TinyLlama + Mistral-7B. Installed under CP 5.1's `vllm.general_plugins` entry point (chained).

Without cipher_kv_offload installed: vLLM preemption uses native recompute (no CIPHER offload). This is a CORRECTNESS-equivalent path (no measurement uplift cited in CP 5.2 close per memory), so cipher_kv_offload removal does not affect any headline metric.

## 6. Outcome verdict: OUTCOME 1 : plugin hook IS load-bearing for non-SC6 measurements

The plugin hook chain is load-bearing for these measurements that B.0 did NOT exercise:

| Measurement | Plugin dependency | Source |
|---|---|---|
| Week 5 KV-dedup HBM savings (45 GiB N=4 TinyLlama; 42 GiB N=2) | cipher_vllm_kv.py + cipher_vllm_kvdedup.py BOTH required | `week5-complete` memory + cipher_vllm_kvdedup.py:7-9 docstring |
| Week 5 KV-dedup hit rate (83.20% Mistral-7B N=4) | Same chain | Same |
| Week 5 KV-dedup bit-identity (KL=0.0 Mistral-7B N=4 6 pairs) | Same chain | Same |
| vLLM-workload model_uuid registration (G10 / G3 / G4 / G12 actuators) | cipher_vllm_kv.py required | `g10-abi-scaffold` memory |
| vLLM-workload stream registration (multi-tenant resolver) | cipher_vllm_kv.py required | `week9-complete` memory |
| vLLM-workload per-model VA sizing (G5 right-sizing) | cipher_vllm_kv.py required | `g5-path-a-verified` memory |
| CP 5.2 KV offload (correctness-equivalent, no measurement uplift cited) | cipher_kv_offload.py required (chained on CP 5.1) | `cipher-cp52-closed` memory |

The plugin chain is NOT load-bearing for:

| Measurement | Why not |
|---|---|
| Track 2 SC2-SC6 weight-sharing (76% Mistral-7B N=4) | uses cipher_kv_bridge weight-arena pattern directly via transformers; vLLM not in process tree; B.0 verified |
| Track 3 v1 SC1-SC6 DSM | kmod-resident substrate; plugin-independent |
| Goal 5 LD_PRELOAD/CUDA_INJECTION64_PATH deployment transparency | Phase A close already achieved via constructor + cuInit-wrapper + CUDA_INJECTION64_PATH; plugin OPTIONAL post-Phase-A |
| W7-12 microbench regression (test_step3_b0, b1, etc.) | substrate-only, plugin-independent |

## 7. Phase B scope implication

Phase B's Branch A gate should NOT be Track 2 SC6's 76% (which substrate-only already produces). The realistic Branch A metric is **Week 5 KV-dedup reproduces under no-plugin path** (45 GiB N=4 TinyLlama HBM saved + 83% hit rate Mistral-7B N=4 + KL=0 bit-identity), since that's what the cipher_vllm_kv + cipher_vllm_kvdedup chain ACTUALLY delivers in production.

Secondary measurements that Phase B should preserve under no-plugin:
- vLLM-workload G10 model registration (downstream Marlin/KV-dedup/Koopman model-aware keying)
- vLLM-workload G5 stream registration (per-tenant cuBLAS routing)
- vLLM-workload G3 per-model VA sizing

These are not standalone metrics but they affect other measurements (Marlin tenant scope, KV-dedup cross-model accuracy, multi-tenant memory budget).

## 8. Substrate-port path analysis (engineering feasibility for Anil-spec'd cudaMalloc/cuMemAlloc intercept)

Anil's V1 substrate work sequence said: "implement as CUDA driver-level hooks on cudaMalloc / cuMemAlloc / cuMemcpyDtoD. Substrate intercepts the allocation calls vLLM uses for KV cache tensors and applies the same ownership / dedup logic the plugin does today, but from libcipher_rt.so instead of from a Python-layer plugin hook."

Engineering examination:

**(a) Direct cudaMalloc/cuMemAlloc GOT-patch.** PyTorch's caching allocator (`c10::cuda::CUDACachingAllocator`) sits between `torch.zeros` and `cudaMalloc`. A single `torch.zeros([N], device="cuda")` call may or may not trigger a fresh `cudaMalloc` depending on pool state; the allocator pools memory and may serve from cache. So intercepting `cudaMalloc` does NOT catch every `torch.zeros` call, and does NOT identify which `cudaMalloc` is for vLLM's KV cache (vs activations, weights, gradients).

**(b) PyTorch CachingAllocator-level intercept.** Intercept `c10::cuda::CUDACachingAllocator::malloc` or equivalent. PyTorch's allocator interface is C++ class methods; not LD_PRELOAD-friendly. Would need to GOT-patch the relevant C++ mangled symbols in libc10_cuda or libtorch_cuda. Fragile across PyTorch versions (private API).

**(c) vLLM allocator-level intercept (subprocess monkey-patch from substrate).** The substrate would need to embed libpython and call into the Python interpreter at the right timing (post-vLLM-imports, pre-KV-allocate) to monkey-patch `GPUModelRunner._allocate_kv_cache_tensors`. This is essentially what the plugin does, but moved into substrate via libpython. Heavyweight: substrate becomes Python-aware, adds libpython linkage, Python interpreter version sensitivity. Adds large surface to the substrate.

**(d) Hybrid: substrate provides the primitives; a tiny plugin-equivalent shim installs the Python hook automatically.** This is the existing architecture. The "plugin" is the right Python-shaped tool for the Python-shaped problem (vLLM's class methods); the substrate provides the cipher_kv_bridge.vmm_zeros / cipher_rt_kv_dedup_alias primitives. Moving plugin functionality wholesale into substrate may not improve the architecture; it just changes where Python-interpreter manipulation lives.

**(e) Alternative: ship the plugin AS PART OF substrate deployment.** Have libcipher_rt.so install the plugin entry point at substrate-init time (via libpython embedded in substrate) so customer doesn't pip-install separately; just bundle the plugin in the substrate's deb/wheel. Still a "drop-in plugin" architecturally but the customer-facing install is "install one .deb that gets both libcipher_rt.so + the plugin entry point". Same approach as cipher-platform.deb at CP 2.5 close (`cipher-cp25-closed` memory) which packaged libcipher_rt.so + libc10.so + kmod for one-shot install.

Engineering judgment: (a)/(b)/(c) are each significant scope and may not deliver on the 1-2 cal-week Phase B budget. (e) preserves the plugin's Python-shaped hook surface but eliminates the pip-install step from customer deployment. (e) is closer to a packaging change than substrate engineering.

## 9. Surface to Anil : recommended Phase B scope shape

Per NO AUTO SCOPE-DEGRADE, surfacing the analysis without picking. Anil adjudication required.

**Option (i) Phase B narrow: deprecate plugin chain (Outcome 2/3 framing).** Plugin remains pip-installed for v1 (no change from baseline-as-shipped). Phase B does NOT migrate to substrate. Branch A gate at close: "all current plugin-dependent measurements (Week 5 KV-dedup, G10/G5/G3 ABI calls, CP 5.2 offload) reproduce under cipher-vllm-kv installed at md5 b89a9b6e + CUDA_INJECTION64_PATH deployment". Phase B becomes documentation + ledger update (~0.5-1 ED).

The pre-Phase-A ledger row "+ pip install cipher-vllm-kv ... contract violation YES" gets reframed: per Phase A close ledger discipline, plugin is OPTIONAL for v1 deployment EXCEPT for the load-bearing measurements identified at §6. The contract preserves the "or" disjunction (LD_PRELOAD or CUDA_INJECTION64_PATH) for the cuBLAS shim layer; for KV-cache layer measurements (Week 5 KV-dedup), the plugin remains installed. Surface to Anil whether this re-violates the 2026-05-26 contract lock or fits within the contract's measured scope.

**Option (ii) Phase B pursue substrate port via approach (e) packaging.** Ship libcipher_rt.so + cipher_vllm_plugin together as a single drop-in unit (cipher-platform.deb style per CP 2.5 close). Customer installs one .deb; substrate is LD_PRELOAD'd + plugin entry point auto-registered. Plugin file lives on disk inside the substrate package; not a separate pip dependency. Goal 5 contract preserved if the .deb installation is treated as the customer-side step (one apt-get install instead of one pip install + one env var). Estimated 2-3 cal-weeks for packaging + verification.

**Option (iii) Phase B pursue substrate port via approach (c) libpython-embedded substrate.** Substrate calls into Python interpreter post-vLLM-imports to install the monkey-patch. Heavyweight; adds libpython linkage to substrate; Python-version sensitivity. Estimated 4-6 cal-weeks; exceeds original 1-2 cal-week Phase B budget. Not recommended.

**Option (iv) Phase B re-scoped to substrate-only-deliverable.** Drop the plugin-migration ambition; Phase B's scope becomes "verify substrate-only path covers all Goal 5 / Goal 1 deployment contracts that don't depend on vLLM-specific KV-cache machinery; document that vLLM-specific measurements require the cipher_vllm_plugin install as a separate deployment dependency that we DOCUMENT in the ledger". Plugin status: REQUIRED for vLLM-specific deployments, OPTIONAL for non-vLLM. Goal 5 contract reframed to acknowledge two deployment tiers. Estimated 1 cal-week.

**Option (v) Reverse the 2026-05-26 Goal 5 contract lock** to accept "LD_PRELOAD + drop-in cipher_vllm_plugin" as the v1 deployment surface for vLLM workloads. This is what the Option 2 close-out reframe originally proposed before Anil locked the contract on 2026-05-26. Requires explicit Anil reversal. Not recommended without Anil signal.

## 10. Honest residue at B.0.5 close

1. **The CP 5.1 vs Track 2 SC6 conflation in Anil's V1 substrate work sequence spec.** Anil's 2026-05-26 spec attributed the Track 2 SC6 76% to CP 5.1; empirically the 76% comes from cipher_kv_bridge weight arena (Track 2), not CP 5.1 KV-cache hook. This archaeology surfaces the conflation explicitly; Anil adjudication on whether the V1 substrate work sequence spec text needs an addendum.

2. **Plugin-OPTIONAL vs Plugin-REQUIRED nuance.** Post-Phase-A close ledger framed plugin as OPTIONAL for v1 deployment. B.0.5 surfaces that plugin is REQUIRED for the load-bearing Week 5 KV-dedup measurements. The ledger needs reconciliation: either acknowledge two-tier deployment (Goal 5 contract narrowing for vLLM workloads only), or commit to migrating the plugin functionality into substrate (Options ii/iii/iv from §9).

3. **Substrate counter-dump worked under CUDA_INJECTION64_PATH for B.0.** Substrate is functioning correctly under Branch D (a) deployment as verified at Phase A close. The remaining gap is exclusively the plugin-mediated chain identified above.

4. **Track 3 v1 SC1-SC6 still NOT located.** Per B.0 doc §9. Need to identify the test sequence as a Phase B scope-lock entry-checklist item; not blocking this archaeology surface.

5. **Engineering feasibility on Anil's spec'd cudaMalloc/cuMemAlloc approach is structurally challenged** per §8 (a)/(b)/(c). The original spec assumed cudaMalloc intercept covers vLLM's KV path; PyTorch's caching allocator pooling layer between `torch.zeros` and `cudaMalloc` makes this fragile.

## 11. Hard budget result

Wall-clock elapsed for B.0.5: ~30 minutes. Well within Anil's 1-2 ED budget. No silent extension.

## 12. Related memory

- [[cipher-cp51-closed]]: CP 5.1 closed certifies correctness only (memo §4 #2); criteria #1/#3/#4 NOT asserted; substrate-deliverable not measurement-uplift
- [[cipher-cp52-closed]]: CP 5.2 KV offload chained via CP 5.1 plugin; correctness gate; no measurement uplift cited
- [[week5-complete]]: Week 5 KV-dedup live wire delivered 45 GiB N=4 TinyLlama + 83% hit rate; plugin-dependent
- [[week6-kvdedup-autotrigger]]: cipher_vllm_kvdedup.py time/pressure auto-trigger; chained on cipher_vllm_kv
- [[g10-abi-scaffold]]: G10 CIPHER_REGISTER_MODEL ABI; plugin hook calls
- [[week9-complete]]: G5 CIPHER_REGISTER_STREAMS multi-tenant resolver; plugin hook calls
- [[g5-path-a-verified]]: per-model VA sizing; plugin hook compute_va_gib
- [[cipher-track2-weight-sharing]]: 76% Mistral-7B N=4 from cipher_kv_bridge weight arena, NOT CP 5.1
- [[v1-goal5-contract-lock]]: 2026-05-26 contract lock; plugin-uninstall framing
- [[v1-phase-a-driver-worker-init]]: Phase A close + Branch D (a) deployment path
- [[cipher-fusion-campaign]]: campaign discipline (NO AUTO SCOPE-DEGRADE, design-memo before code)
- [[cipher-proceed-not-ask]]: Anil "Proceed" + detailed scope = execute and document, surface findings
