# CP 5.1 — Step 2: vLLM KV cache allocation path — SCOPE MEMO

**Date:** 2026-05-17. **Status:** Step 2 of CP 5.1. Investigation complete —
scoped here for adjudication. **Step 2 is paper**: it maps the path and
recommends a composition option. The build is Step 3, gated on approval of §(d).

**Anchors held:** kmod `e2f50452`, libcipher_rt `c2c5d313`, libcipher_v2
`86618c30`. Step 2 ships no code.

**Context in:** Step 1 (`cp51_vllm_smoke.py`) PASS — vLLM **0.20.2** engine-init
clean on this pod (torch 2.11 / CUDA 13.0), coherent generation, `SMOKE OK`,
exit 0. Runlog: `cp_5_1/cp51_vllm_smoke.runlog`. Adjudicated `vLLM 0.20.2
ACCEPT`. Memo version line (`PHASE_5_CP_5_1_DESIGN_MEMO.md` §1, currently
"0.5/0.6/0.7") to be reconciled to 0.20.2 at CP 5.1 close.

---

## §(a) — vLLM v1 KV cache allocation path, as mapped

vLLM 0.20.2 uses the **v1 engine**. KV memory splits into two cleanly
separated layers:

**1. Physical buffer allocation — one-shot, at engine init.**
`GPUModelRunner.initialize_kv_cache_tensors()` →
`_allocate_kv_cache_tensors()` (`v1/worker/gpu_model_runner.py:6522`). The
*entire* GPU KV memory is a small set of raw buffers, each a single call:

```python
tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)   # :6537
```

one per `kv_cache_tensor` in `kv_cache_config.kv_cache_tensors` (typically one
per layer; `shared_by` lets layers alias one buffer). `_reshape_kv_cache_tensors()`
(`:6563`) then turns each raw int8 blob into the paged KV shape
`[2, num_blocks, block_size, num_kv_heads, head_size]` purely via
`.view()` / `torch.as_strided` — **no new allocation, just views**.

**2. Block management — pure Python, no GPU memory.**
`KVCacheManager`, `SingleTypeKVCacheManager`, `BlockPool` (`v1/core/`) —
verified to contain **zero** `torch`/`cuda`/`device=` calls. They are integer
bookkeeping: which logical block index of the pre-allocated buffer is assigned
to which request. "Paged attention" is addressing math over the §(a).1 buffers,
not runtime allocation.

**Consequence:** all KV GPU memory enters existence at exactly the
`torch.zeros` calls in `_allocate_kv_cache_tensors`. There is no per-block, no
per-step, no per-request GPU allocation. One interception point owns 100% of KV.

**Two allocation sites, not one.** `initialize_kv_cache_tensors` (`:6724`)
branches: the default path is `_allocate_kv_cache_tensors`; a second path,
`allocate_uniform_kv_caches` (`v1/worker/kv_connector_model_runner_mixin.py:193`),
fires only when `use_uniform_kv_cache` is true — i.e. a KV-connector
(disaggregated prefill) is configured. The Step 1 smoke (no connector) took the
default path. The uniform path is connector-only and out of the CP 5.1
single-pod baseline; flagged so Step 3 covers or explicitly defers it.

**Process boundary.** `_allocate_kv_cache_tensors` runs in the **EngineCore
subprocess** (Step 1 log: `(EngineCore pid=1475011)`), not the `LLM()` caller
process. Any Python-side install must execute inside that subprocess.

## §(b) — S2b transformers contract, recap

Op #1 S2b (`op_1_S2b_allocator_integration.md`, SHIPPED 2026-05-15) made CIPHER
own KV memory in **transformers 5.8.1** by *being the cache class*:
`StaticCache` builds per-layer `StaticLayer` / `StaticSlidingWindowLayer`
objects; each allocates its `keys`/`values` in `lazy_initialization()`. S2b
subclassed those layers, overrode `lazy_initialization` to allocate from CIPHER
VMM via the `cipher_kv_bridge` (`at::from_blob` over a VMM `CUdeviceptr`), and
`install()` monkey-patched the layer classes.

Why this mattered (`pause_note.md` §c/§d): data-ownership ports are
**coverage-immune** — CIPHER holds the KV tensor whether or not decode is
`torch.compile`-d / cudagraph-replayed, because there is no *call* to miss.
This is the property that must be preserved against vLLM; if the vLLM
integration degrades to interception it inherits Op #2's 43%/14% coverage
blocker.

`cipher_kv_bridge` API (built `.so` md5 `2cf82c06…`): `vmm_zeros(shape,
itemsize, tag…) -> torch.Tensor`, `page_info(devptr)`, `get_stats()`.
`vmm_zeros` already produces a `torch.Tensor` of arbitrary shape backed by a
tagged VMM allocation — reusable essentially as-is.

## §(c) — Composition options + coverage-immunity assessment

**Option A — buffer-ownership / `from_blob` injection at `_allocate_kv_cache_tensors`.**
Monkey-patch `GPUModelRunner._allocate_kv_cache_tensors` so each raw KV buffer
is sourced from `cipher_kv_bridge.vmm_zeros([kv_cache_tensor.size], itemsize=1,
tag=…)` (int8) instead of `torch.zeros`. vLLM's `_reshape_kv_cache_tensors`
views/strides are unaffected — they operate on whatever tensor it is handed.
- *Coverage:* **immune.** CIPHER *is* the buffer; every attention kernel —
  eager, compiled, or cudagraph-replayed — reads/writes CIPHER VMM memory.
  No call to miss. Same property as S2b.
- *Reuse:* `cipher_kv_bridge` reused ~as-is; only the call site changes.
- *Cost:* the patch + `cipher_kv_bridge.init()` must install inside the
  EngineCore subprocess (vLLM plugin entrypoint or `--worker-extension-cls`;
  resolved in Step 3). Plus the uniform-kv path (§a) to cover or defer.

**Option B — cache-class replacement (the literal S2b mechanism).**
*Finding, not an option:* vLLM v1 has **no per-layer KV cache class** to
subclass. Allocation is a `GPUModelRunner` method; paging is `KVCacheManager`
Python bookkeeping. There is no `StaticLayer` analogue. The literal S2b
mechanism does not transfer — Option A is its functional replacement (own the
*buffer* rather than the *class*; same coverage-immunity outcome). Recorded so
the adjudication trail shows this was checked, not skipped.

**Option C — interception fallback (driver-layer allocation hook).**
Hook the CUDA allocation beneath `torch.zeros` at the driver/allocator layer.
- *Rejected.* At that layer CIPHER cannot cleanly discriminate the KV buffer
  allocation from model-weight and activation allocations, and it reintroduces
  an interception dependency. Not coverage-clean, not targetable. Listed only
  to document that the interception path was considered and dismissed.

## §(d) — Recommendation for adjudication

**Recommended: Option A — buffer-ownership injection at
`_allocate_kv_cache_tensors`.** It is the true analogue of S2b's data-ownership
trick, preserves coverage-immunity (the load-bearing property from
`pause_note.md` §d), and reuses the `cipher_kv_bridge` `.so` with no rebuild —
only the call site moves from transformers' `StaticLayer` to vLLM's
`GPUModelRunner`. Step 3 (build) is then: install hook in the EngineCore
subprocess, source KV buffers from CIPHER VMM, gate on byte-identical vLLM
output substrate-on vs substrate-off (CP 5.1 §4 correctness bar).

**Known-unknowns to carry into Step 3 (not Step-2 blockers):**
1. **Page-tag granularity shift.** S2b tagged per-(layer, K/V-role) slabs.
   vLLM gives one int8 buffer per layer with K/V interleaved by stride — CIPHER
   would tag at per-layer-buffer granularity; K-vs-V is *inside* the buffer.
   This changes what T4.6.3 dedup keys on; assess before dedup wiring.
2. **EngineCore-subprocess install mechanism** — vLLM plugin entrypoint vs
   `--worker-extension-cls` vs the design memo's `CUDA_INJECTION64_PATH` (which
   injects C, not Python). To be resolved at Step 3 start.
3. **Uniform-kv-cache path** (`allocate_uniform_kv_caches`) — connector-only;
   cover or explicitly defer in Step 3 scope.
4. **vLLM memory profiler mis-accounts CIPHER VMM as free** — first-order
   Step-3 risk. CIPHER VMM is invisible to `torch.cuda.memory_allocated` (S2b
   measured KV delta `4.9 MiB → 0.0 MiB`). vLLM runs an active profile pass at
   engine init to size `num_gpu_blocks` from `gpu_memory_utilization` ×
   observed-free GPU memory. Under Option A the profiler sees CIPHER-held KV
   bytes as free and will overcommit or double-allocate the same physical
   memory. transformers' `StaticCache` had no profiler so S2b never hit this;
   for vLLM it is first-order. Mitigation (reserve CIPHER bytes from the
   profiler's accounting / scale `gpu_memory_utilization` / intercept the
   profile) is Step-3 work — surfaced here so adjudication sees the risk.
5. **VA pool sizing.** `cipher_rt_kv_alloc_init(va_pool_bytes)` was sized for
   S2b's transformers regime (per-layer slabs, `pages_resident_peak=128` ≈
   256 MiB). vLLM at H100 80 GB / typical `gpu_memory_utilization` wants the KV
   buffer in the tens-of-GiB range. Confirm at Step 3 the VA pool enlarges at
   `init()` with no hardcoded ceiling, or that `init()` takes the
   operator-chosen size.

**Gate to proceed:** adjudicate Option A. On approval, Step 3 builds the
EngineCore-subprocess KV buffer hook; on rejection, re-scope here. No code is
written until §(d) is approved.
