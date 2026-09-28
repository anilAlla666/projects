# Path C — Deep dump of K5 lambda capture (256 bytes + 1-level indirection)

**Result**: searched the full 256-byte K5 args[1] window at every byte offset, followed every host-pointer-shaped value one level of indirection. **The per-layer RMSNorm weight pointer is not in the lambda capture** — neither directly nor at one level of indirection. Stable cross-window cuda pointers in this region resolve to a single shared address (probably `eps` or another model-global constant), not the layer-specific weight.

---

## What we did

1. Extended `scan_raw_dump_deep()` to read 256 bytes from `args[1]` (the `__nv_hdl_wrapper<lambda>` capture).
2. Walked **every byte offset** (1-byte stride) — args[1] address is `0x...197a`, not 8-byte aligned, so 8-byte-stride reads from the start would have missed aligned uint64 boundaries.
3. For every value classified as HOST pointer (`0x500000000000 .. 0x800000000000`), followed it: read 64 bytes from the target, scanned for cuda VA values (`0x100000000000 .. 0x400000000000`, the actual H100 user-VA range on this pod).

## What we found across 3 RMSNorm windows

```
[K5 args[1] @ 0x7ffdae7511fa, 256 bytes]
  +  2 = 0x00002000000074b4 [DEV]     (stale stack noise, 0x2000 = numel for adjacent kernel)
  +166 = 0x00007ffdae7515c0 [HOST] -> +34 = 0x00003f0000000000 [DEV]
  +190 = 0x00005738dccfc0b0 [HOST] -> +34 = 0x00003f0000000000 [DEV]      ← varies per window (0x5738dccfc0b0, 0x5738f61a2d10, 0x5738d7944d10)
  +222 = 0x000074b4184a5453 [HOST] -> +28 = 0x00003d481f760017 [DEV]
  scan summary: direct DEV=4, HOST→DEV=3
```

Cross-window observations:
- **+166 host pointer is identical across all 3 windows.** It points to a stack location that contains `0x3f0000000000` at offset +34. Same target across windows.
- **+190 host pointer is a DIFFERENT heap address per window** (`0x5738dccfc0b0`, `0x5738f61a2d10`, `0x5738d7944d10`). But **all three dereferences land at the same `0x3f0000000000`**.
- **+222 dereferences to `0x3d481f760017`** — also stable across windows.

## What this tells us

**Both stable cuda VA values (`0x3f0000000000` and `0x3d481f760017`) are SHARED across all observed RMSNorms.**

For Llama-3.2-1B with 33 distinct RMSNorm layers (32 layers × 2 + 1 final), each has its own 4096-byte weight tensor at a *different* device address. If the lambda capture were holding `weight.data_ptr()`, we'd expect ~33 different cuda values across calls — but we see only TWO unique ones, and both repeat for every RMSNorm.

So `0x3f0000000000` and `0x3d481f760017` are model-global constants, not per-layer weights. Most likely candidates:
- An `eps` value backing buffer (eps = 1e-5 stored once)
- Stride / shape info shared across all elementwise dispatches
- A TensorIterator workspace pointer

**The per-layer weight is not in K5's args[1] window, even one level of indirection deep.** It's somewhere further inside PyTorch's struct hierarchy, behind ≥ 2 indirections — exactly the version-pinned offset walking the original brief asked us to avoid.

## What remains untried at the driver level

1. **Walk PyTorch's TensorIterator at known offsets.** The lambda's captured TensorBase reference holds a chain `TensorBase → TensorImpl → StorageImpl → data_ptr_`. With version-pinned offsets we could chase it. Not portable across PyTorch versions.

2. **Allocation tracking + size correlation.** Hook `cudaMalloc`/`cudaFree`. Each layer's RMSNorm weight is a stable 4096-byte allocation (or sub-region of a slab). After model load, enumerate stable allocations of size 4096; pick the right one for each call by activation-tensor proximity in launch order. Model-aware but version-independent.

3. **Hook cudaMemcpy at model load time.** When PyTorch copies weight tensors to GPU during `from_pretrained`, capture each (host_src, device_dst, size). Build a map `{tensor_name → device_ptr}` keyed by tensor's role inferred from copy order. Then at inference time, pick the right weight by matching the RMSNorm's known position in the layer iteration.

None of these are simple. (1) breaks across PyTorch versions. (2) requires an additional Stage building the allocation table + correlation logic (the alloc-tracker the brief described). (3) requires identifying which copies are weight loads vs. anything else.

## Honest position

The brief's chain of reasoning — "MeanOps anchor + sequence-shape detection at driver level + extract operand pointers from kernel args" — works for **detection**. It does not work for **pointer extraction** in PyTorch's eager mode for Llama, because:

- PyTorch wraps kernel arguments in `__nv_hdl_wrapper<lambda>` and `TensorIterator` structs.
- The data pointers we need (X, W, Y) are inside those structs, behind 1-3 levels of indirection.
- One-level indirection scan finds **shared globals** (constants), not per-call tensors.
- The actual data pointers are at PyTorch-version-specific offsets within these structs.

Stages 1 + 2 + 3 ship as a **deterministic detector and substitution scaffold** — verified working at 100% sequence-shape match rate on Llama-3.2-1B and Llama-3.1-8B. The substitution call is wired and one-line-from-active once X / W / Y are sourced. The remaining gap is data-pointer resolution — and the empirical answer from this and the prior dumps is that it requires a separate mechanism (allocation tracking or version-pinned struct walking), not the kernel-arg scanning the brief proposed.

## Files updated

- `src/cipher_flow_substitute.cpp` — `scan_raw_dump_deep()` extended to 256 bytes, 1-byte stride, 1-level indirection, tight CUDA-VA filter (`>= 0x100000000000ULL`), with HOST→DEV chain logging
- `CIPHER_FLOW_DEEP_DUMP_FINAL.md` — this report

Reproducibility:
```bash
CIPHER_FLOW_MATCH=on CIPHER_FLOW_SUBSTITUTE=on CIPHER_FLOW_SCAN=on \
    ./cipher_run.sh -c "<inference>" 2>deep.log
grep -E "FLOW DEEP|HOST.*->|scan summary" deep.log
```
