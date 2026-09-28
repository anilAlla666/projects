# Op #1 — S2b allocator integration — REPORT

**Status: SHIPPED — gate PASS.**

Op #1 of the fusion build: finish the T4.6.2 KV allocator integration that
the prior session paused on. Wires the VMM page allocator
(`cipher_rt_kv_alloc`) into transformers 5.8.1's `StaticCache` so CIPHER
owns and tags the KV cache memory — the foundation for T4.6.3 dedup and
T4.6.4 cuIpc cross-tenant export.

## What shipped

| Artifact | Lines | Purpose |
|---|---|---|
| `cipher_rt_phase4/cipher_rt_kv_alloc.{h,c}` | ~120 + ~310 | VMM page allocator (built in S2; this op added zero-on-map) |
| `cipher_rt_phase4/cipher_kv_bridge.cpp` | ~115 | pybind11/libtorch extension: `at::from_blob` wrap of a CIPHER VMM `CUdeviceptr` as a `torch.Tensor` |
| `cipher_rt_phase4/cipher_kv_cache.py` | ~110 | `CipherStaticLayer` / `CipherStaticSlidingWindowLayer` + `install()` monkey-patch |
| `cipher_rt_phase4/build_kv_bridge.sh` | ~30 | bridge build script |

Built artifact: `cipher_kv_bridge.cpython-310-x86_64-linux-gnu.so`
md5 `2cf82c06f3a7a4732ba59abb58f93b9c`.

**Note on scope:** S2b does NOT modify `libcipher_rt.so`. The bridge is a
separate Python-importable extension and the cache class is Python — both
operator-injected Python-side. There is therefore no new `libcipher_rt.so`
md5 to record for this op; the gate md5 is the bridge `.so` above.

## Integration model (resolved during the audit)

The T4.6.1 attention substrate intercepts SDPA *after* K/V are already
PyTorch-allocated — it cannot retroactively own KV memory. Resolution
(approved): own KV at the *cache* layer, not the attention layer.

transformers 5.8.1 `StaticCache` builds per-layer `StaticLayer` /
`StaticSlidingWindowLayer` objects; each allocates `self.keys`/`self.values`
in `lazy_initialization()` and writes in-place via `index_copy_`. CIPHER
subclasses these layers and overrides `lazy_initialization` to allocate the
backing tensors from VMM slabs via the bridge. `install()` monkey-patches
`transformers.cache_utils.StaticLayer`/`StaticSlidingWindowLayer` so
`StaticCache` builds CIPHER-backed layers, and wraps `StaticCache.__init__`
to stamp each layer with its index + a per-cache sequence id.

Customer code is unchanged; the operator injects `cipher_kv_cache.install()`
and (for production) defaults `cache_implementation=static` — see Deployment
notes.

## Three-indicator binding diagnostic — ALL THREE FIRE

Smoke: Mistral-7B-v0.1 (sliding-window → `CipherStaticSlidingWindowLayer`)
+ TinyLlama-1.1B (full-attention → `CipherStaticLayer`), 3 prompts ×
32 tokens, `cache_implementation=static`, baseline vs cipher.

| Indicator | Signal | Result |
|---|---|---|
| (a) allocator exercised by real decode | `cipher_kv_bridge.get_stats()` | `slabs_created=192` (3 prompts × 32 layers × 2 K/V), `pages_mapped=192`, `pages_resident_peak=128` — **FIRES** |
| (b) page tags correct | `page_info()` on layers 0/16/31 | tenant_id=7, seq_id=3, layer ∈ {0,16,31} matches index, role 0=K/1=V, head_kv=0xFFFF — **6/6 correct** |
| (c) KV bypasses PyTorch allocator | `torch.cuda.memory_allocated` Mistral KV delta | baseline **4.9 MiB** → cipher **0.0 MiB** — KV fully in CIPHER VMM pool — **FIRES** |

## Gate results

| Gate | Result |
|---|---|
| Byte-identical output — Mistral (sliding path) | ✅ `f4727c26db33f3e4` / `e27ab0e7a52354c2` / `51596ef1d51137b8` — identical baseline vs cipher |
| Byte-identical output — TinyLlama (plain path) | ✅ `fbfc4737ec8723fa` / `8a11cc62ec2136d3` / `9c814c98e164bc0e` — identical baseline vs cipher |
| Both cache-class paths exercised | ✅ `CipherStaticSlidingWindowLayer` (Mistral) + `CipherStaticLayer` (TinyLlama) |
| Allocator unit test (after zero-on-map edit) | ✅ 14/14 checks pass |
| Slab lifecycle clean | ✅ `slabs_created=192 == slabs_freed=192`, `pages_resident=0` at end (deleter fires on tensor GC) |
| Tolerance | strict byte-identical — S2b moves memory only, no math touched. Met. |
| Fallback anchors | ✅ `55ab8c0c…` / `86618c30…` unchanged |
| Kernel taint | ✅ 12288 |

## Adjustments applied (from the pre-build audit)

1. Overrode `lazy_initialization` (subclass) rather than post-hoc tensor
   substitution — no wasted `torch.zeros` allocation.
2. Subclassed **both** `StaticLayer` and `StaticSlidingWindowLayer` —
   Mistral-7B-v0.1 (`sliding_window=4096`) builds the sliding variant for
   all 32 layers; both verified byte-identical.
3. Deployment-policy note — see below.
4. Zero-on-map: the allocator `cuMemsetD8`s every freshly-mapped page.
   A 2 MiB page may be physical memory just released by another tenant's
   freed slab — without zeroing, the new owner could read the prior
   tenant's KV (cross-tenant leak) and `StaticLayer`'s zero-init semantics
   would break. Enforced centrally in `map_pages`, not per-caller.
5. The `from_blob` bridge is a pybind11/libtorch extension `.so`
   (`torch::from_blob` is C++-only) — confirmed correct artifact shape.

## Deployment notes (per adjudication)

1. CIPHER KV ownership engages only under `cache_implementation=static`.
2. Operator deployment defaults this via `GenerationConfig` injection at
   operator-config-load time (full-fusion: `cipher_inject.py`). Operator
   policy, not a customer code change — the customer Python script is
   byte-identical.
3. A tenant explicitly requesting `DynamicCache`: the allocator is simply
   not engaged for that session; all other ops still operate normally.
4. Per-tenant opt-out: `CIPHER_KV_ALLOC=0`.

## Honest gaps / deferred

- **No memory saving yet.** `StaticLayer` pre-allocates the full
  `[B,H,max_cache_len,D]` tensor and the attention kernel reads the whole
  span, so every page is mapped at slab-create. The allocator's on-demand
  page-growth path (`cipher_rt_kv_slab_ensure`) exists and is unit-tested,
  but the static-cache consumer does not exercise it. T4.6.2 delivers KV
  *ownership + tagging*, not a memory-footprint win — consistent with
  "foundation, not a moat number." Dedup (T4.6.3) is where saving appears.
- **TinyLlama `memory_allocated` delta is contaminated** by measuring two
  models in one process (Mistral still resident when TinyLlama's baseline
  is sampled). TinyLlama's role here was the functional/byte-identical
  check of the plain `StaticLayer` path, which passed; its memory number
  is not used as evidence. The Mistral delta (4.9 → 0.0 MiB) is clean
  (measured first, before TinyLlama loads).
- **Substrate composition (T4.5.1 cuBLAS + T4.6.1 attn) not yet tested
  with the CIPHER cache** — the smoke ran without `LD_PRELOAD`. That is
  the S4 composition check, tracked separately.

## Next

Op #2 — Op 9 AUDIT (simplest already-built op; validates the actuator
registration plumbing). Per protocol: AUDIT first, then PROPOSE, then WAIT.
