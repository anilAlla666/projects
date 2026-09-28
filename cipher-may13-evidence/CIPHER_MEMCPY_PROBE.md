# Path C — cudaMemcpy probe for RMSNorm weight discovery

**Headline**: `cudaMemcpyAsync` interception **does** see all 33 RMSNorm-sized weight loads during Llama-3.2-1B from_pretrained — the count, size, and timing match exactly. But **the destination is a single staging buffer**, not the per-layer weight addresses we need.

PyTorch's caching allocator pipelines small-tensor loads through a small pool of recycled staging buffers, then a separate (kernel-level) copy moves each weight to its final per-layer location. cudaMemcpyAsync sees only the first hop.

---

## What the probe showed

Llama-3.2-1B fp16 model load via `from_pretrained(device_map='cuda:0')`:

| size | count | unique dst addrs | implication |
|---|---|---|---|
| 4096 (RMSNorm) | 33 | **1** (0x1a48000000) | small-tensor staging |
| 2 MB (k/v_proj) | 32 | **1** | small-tensor staging |
| 8 MB (q/o_proj) | 32 | 8 | medium reuse |
| 32 MB (gate/up/down) | 48 | 10 | medium reuse |
| 525 MB (embed) | 1 | 1 | direct large-tensor allocation |

**33 of 4096-byte H2D copies = 32 RMSNorms × 2 + 1 final RMSNorm = 33 weights.** The count, size, and ordering exactly match. This is the right signal — we're seeing every RMSNorm weight load. But:

```
[CIPHER MEMCPY] #1   api=MemcpyAsync kind=H2D dst=0x1a48000000 src=0x7927069041a8 size=4096
[CIPHER MEMCPY] #5   api=MemcpyAsync kind=H2D dst=0x1a48000000 src=0x79270c9051a8 size=4096
[CIPHER MEMCPY] #10  api=MemcpyAsync kind=H2D dst=0x1a48000000 src=0x79270dd061a8 size=4096
... (33 total)
```

The src varies (each weight is at a different file offset in the mmap'd safetensors), but **dst is always 0x1a48000000**. PyTorch's caching allocator's small-block pool recycles a single ~MB staging chunk for all sub-MB tensors.

For **8 MB and 32 MB** tensors PyTorch DOES allocate per-tensor (8 / 10 unique dsts respectively) — those probably go directly to their final per-layer locations.

## Why this happens

PyTorch's `device_map='cuda:0'` weight-loading flow uses pinned host memory for `safetensors` mmap and runs many small H2D copies in parallel. For sub-MB tensors, the caching allocator's small-block pool recycles a single staging chunk; data is later copied (likely by a kernel or via cuMemcpyDtoD at the driver level — not visible in our cudaMemcpyAsync intercept) to the actual per-tensor location.

The 0 D2D copies in our log confirm the staging-to-final-location move doesn't go through `cudaMemcpyAsync`.

## What we still don't have

Per-layer RMSNorm weight addresses. The 33 copies tell us *how many* weights there are and *when* they're loaded, but not *where* each ends up.

## Three viable next paths (none in this delivery)

1. **Hook `cuMemcpy*` driver API** (`cuMemcpyHtoD`, `cuMemcpyDtoD`). PyTorch's caching allocator may use the driver API directly for the staging-to-final copy. ~30 min experiment, verifiable.

2. **Hook `cudaMalloc` and track stable allocations.** After `from_pretrained` completes, enumerate every live allocation that survived for >1s without being freed. Filter to size = 4096. The 33 stable 4 KB allocations ARE the RMSNorm weights. Order them by allocation sequence: in PyTorch's iteration order they correspond to layer 0 input_layernorm, layer 0 post_attention_layernorm, layer 1 input_layernorm, etc. Match to RMSNorm matcher's window count modulo 33 (or exact mapping if we track which tensors PyTorch creates).

   Caveat: PyTorch's caching allocator may ALSO sub-allocate small tensors from a slab. If so, `cudaMalloc` won't show 33 individual 4 KB allocations; it'll show a few large slabs. We'd need to walk further.

3. **Watch the kernel that runs right after each H2D #N.** If a kernel reads `0x1a48000000` (the staging buffer) and writes to a NEW address that wasn't previously allocated, that's the per-layer destination. Track these as the 33 RMSNorm weights.

## What's empirically clear now

- `cudaMemcpyAsync` H2D probe is the right vantage point for **counting** weight loads
- 33 4096-byte copies = exactly 33 RMSNorm weights (validated)
- Destination tracking from this signal alone gives us **the staging buffer**, not the per-layer weight pointers
- A second hop (kernel or driver-level copy) carries data from staging to per-layer

Path 3 (next-kernel watcher) seems cleanest — we already record every cudaLaunchKernel via `cipher_flow_recorder.cpp`. After each 4 KB H2D, the very next kernel (or a kernel within a few launches) on the same stream that writes a NEW dst pointer is the relocate. That dst is the RMSNorm weight address.

Estimate: ~1 day to wire path 3, including end-to-end correctness with output coherence verified.

## Files updated

- `src/cipher_intercept_cudart.cpp` — added `cudaMemcpy` and `cudaMemcpyAsync` shims, gated by `CIPHER_MEMCPY_PROBE=on`. Forward to real cudart implementations; log when probe is on.
- `exports.map` — added `cudaMemcpy` and `cudaMemcpyAsync` to the global symbol list so the LD_PRELOAD shim picks them up
- `CIPHER_MEMCPY_PROBE.md` — this report

## Reproducibility

```bash
CIPHER_MEMCPY_PROBE=on ./cipher_run.sh -c "<your model load>" 2>memcpy.log
grep 'CIPHER MEMCPY' memcpy.log | grep -oE 'size=[0-9]+' | sort | uniq -c | sort -rn
```

The output classifies every H2D copy by size — for Llama-3.2-1B you'll see 33 of 4096 (RMSNorms), 32 of 2 MB (k/v), 32 of 8 MB (q/o), 48 of 32 MB (gate/up/down), 1 of ~525 MB (embedding).
