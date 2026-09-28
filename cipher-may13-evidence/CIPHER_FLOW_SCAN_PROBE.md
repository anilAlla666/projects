# Path C — Pointer extraction probe via 128-byte scan of `args[i]`

**TL;DR**: the probe ships, runs cleanly (mincore-protected, no segfaults), and **finds real CUDA VA pointers** in the args[] window. But it **does not find the RMSNorm weight pointer**. The weight is hidden behind 4-5 levels of struct indirection (`TensorIterator → TensorBase → TensorImpl → StorageImpl → data_ptr_`) — outside the 128-byte one-level scan. Allocation tracking via `cudaMalloc`/`cudaFree` interception is the next step.

---

## What I built

`src/cipher_flow_substitute.cpp::scan_step_args(...)` and `scan_dump_and_classify(...)`. Wired into the existing window walker. Gated by `CIPHER_FLOW_SCAN=on`. For each of K1..K5 in a matched RMSNorm sequence:

- Read up to 128 bytes from `args[i]` for each i in 0..7
- mincore-check the page is resident before each read (no SIGSEGV)
- 8-byte stride; collect 8-byte-aligned values that fall in the canonical CUDA VA range (`0x100000000 .. 0x800000000000`)
- Filter out stack-looking pointers (`0x7f...` upper bits)
- After K5, cross-reference: which pointer appears in N steps? First-seen step? — labels each as `intermediate`, `X-or-intermediate`, `Y-or-W (K5 only)`, etc.

Stack-page reads are now protected with `mincore()` — no crashes across 60-token gens on Llama-3.2-1B and Llama-3.1-8B.

---

## What the probe found (3 consecutive RMSNorm windows on Llama-3.2-1B)

Real CUDA VA pointers (`0x1a48xxxxxx` range on this pod):

| pointer | which steps | classification | observation across windows |
|---|---|---|---|
| `0x1a48022400` | K2 + K3 | intermediate | stable across all 3 windows |
| `0x1a48022200` | K2 | candidate | stable across all 3 windows |
| `0x1a48026800` | K3 | candidate | stable across all 3 windows |
| `0x1a48027000` | K5 | "Y-or-W (K5 only)" | window 0 only |
| `0x1a4803a000` | K5 | "Y-or-W (K5 only)" | window 0 only |
| `0x1a48032000` | K5 | "Y-or-W (K5 only)" | windows 1 + 2 |
| `0x1a48040000` | K5 | "Y-or-W (K5 only)" | window 1 only |
| `0x1a4803c000` | K5 | "Y-or-W (K5 only)" | window 2 only |

Cross-reference with the cublasGemmEx log (same trace, just before the scan):
```
GEMM #0  A=q_proj_weight  B=0x1a4802b000  C=0x1a48027000
GEMM #6  A=...            B=0x1a48056000  C=0x1a48027000
```
`0x1a48027000` is a recycled activation buffer used as output by GEMMs in attention/MLP. It appears in K5 of window 0 — that's the RMSNorm OUTPUT (Y) for that call. Different windows see different Y's, as expected.

## What it didn't find

**The RMSNorm WEIGHT pointer is not present in any of the K1..K5 candidate sets.** Across 3 consecutive RMSNorm windows, no `0x1a48xxxxxx` pointer appears in K5 of all 3 — meaning the stable weight pointer (which by definition appears at the same address every RMSNorm call) is not visible in the 128-byte one-level scan.

This is consistent with PyTorch ATen's calling convention. For `vectorized_elementwise_kernel<8, BinaryFunctor<...,Mul>>`:
```
args[0] = numel (int)
args[1] = BinaryFunctor (functor struct, holds operation parameters by value)
args[2] = TensorIteratorOpaqueArray (struct containing pointer-array indirection)
```
The `TensorIteratorOpaqueArray` at args[2] doesn't directly contain `weight.data_ptr()`. It contains a pointer (or an offset) to a separately-allocated `data_` array somewhere on the heap, which contains the actual `char*` to weight memory. That's outside our 128-byte scan window — and chasing it requires version-specific offset knowledge.

## Why allocation tracking is the right next step (and what it should look like)

For Llama-3.2-1B with `hidden_dim = 2048`:
- RMSNorm weight: `2048 × 2 = 4096 bytes`, fp16, **stable** (allocated once at model load, never freed)
- Llama-3.2-1B has 33 RMSNorms (32 layers × 2 + 1 final) → 33 such 4096-byte allocations

If we hook `cudaMalloc`/`cudaFree`:
1. Build a live-allocation table `{ptr → size, alloc_seq, last_seen}`
2. Mark allocations as STABLE when they survive ≥ 1000 kernel launches without being freed
3. When the matcher detects an RMSNorm sequence with `rows × hidden_dim × 2` known, look up the unique stable allocation of size `hidden_dim × 2` that's "near" (in time/space) to the activations the kernel sees

This works without struct-parsing because we're correlating sizes, not offsets.

The 128-byte scan still has value: it can identify Y (output) by the activation buffer pointers it DID find in K5. Combined with allocation-table-based W identification, the substitution path becomes wirable:
- W ← stable-allocation-of-size-2H from cudaMalloc tracker
- Y ← K5-only candidate from scan probe (multiple per window; pick the largest likely-activation, or correlate with subsequent kernel reads)
- X ← either K0 candidate (we don't capture K0 currently) OR derive from "what was the last GEMM C= before this RMSNorm started"

## Caveat for caching allocators

PyTorch's caching allocator calls `cudaMalloc` for big slabs (e.g., 20 MB blocks) then sub-allocates internally. So the alloc tracker would see ~10-30 `cudaMalloc` calls totaling the model's weight + activation budget — NOT one per tensor. The 4096-byte RMSNorm weight isn't a separate `cudaMalloc`; it's an offset within a slab.

Solution for the alloc tracker: instead of looking for "stable allocation of size 2H", we'd need to look at **which slab the K5 stable pointer lands in**, and use the slab's `cuMemGetAddressRange` to get the slab base — then we know the slab contains weights, and the K5 pointer's offset within the slab identifies the RMSNorm weight. Multi-step but tractable.

Alternative cleaner path: tap PyTorch's allocator via `c10::cuda::CUDACachingAllocator::format_size_alloc_actions()` — but that's a PyTorch-side hook, not driver-level.

## Recommendation

Next-session build:
1. **`src/cipher_alloc_tracker.cpp`** — hook `cudaMalloc`/`cudaFree`, table of live allocations (~150 LOC).
2. **`cipher_flow_substitute.cpp`** — once a recipe is matched, query alloc tracker for stable-allocation-of-2H bytes. Combined with K5 scan candidates, infer X / Y by elimination.
3. **One-time correctness gate**: run with extracted X/W/Y, do real suppression for ONE RMSNorm call only, observe whether the next kernel's input still produces sane logits. If yes, scale to full suppression.

Estimated 2-3 days. The matcher and substitute scaffolding from Stage 2/3 are unchanged — pointer extraction is the missing piece.

---

## Reproducibility

```bash
CIPHER_FLOW_MATCH=on CIPHER_FLOW_SUBSTITUTE=on CIPHER_FLOW_SCAN=on \
    ./cipher_run.sh -c "<your inference code>"
```

Expected output: `[CIPHER FLOW SCAN] window=N candidates per step:` for the first 3 windows, with full per-step pointer dumps. Output of the model remains coherent (substitute is still in safe-stub mode).

Files updated:
- `src/cipher_flow_substitute.cpp` — added `scan_step_args` + `scan_dump_and_classify` + mincore-guarded reads
- `CIPHER_FLOW_SCAN_PROBE.md` — this report
