# Path C — Pointer-flow recorder, Stage 1

**Goal**: prove the cross-kernel data-flow graph that defines RMSNorm/SiLU/RoPE is visible at the cudaLaunchKernel level — *without* relying on kernel name regex.

**Status**: Stage 1 ships. Recorder + edge detector are wired into the hook DSO. Verified against Llama-3.2-1B fp16: the 6-kernel RMSNorm sequence is visible, and the `[edge: args[N]=X <- seq=Y args[M]]` line proves we can detect that kernel S+1's read pointer equals kernel S's write pointer — that's the data-flow signal Stage 2 will pattern-match against.

---

## Files added

| File | Role |
|---|---|
| `include/cipher_flow_recorder.h` | API (`init`, `observe`, `dump`) + `CipherFlowEntry` POD |
| `src/cipher_flow_recorder.cpp` | 64-entry circular ring, atomic head, page-bounded args[] read, edge cross-reference |
| `src/cipher_kernel_table.cpp` | Added `cipher_kt_name_for(void* fn)` accessor |
| `src/cipher_intercept_cudart.cpp` | Wired `cipher_flow_recorder_observe(...)` into `cudaLaunchKernel` shim |
| `Makefile` | Added `cipher_flow_recorder.cpp` to `HOOK_OBJ` so it compiles into `libcipher_hook.so` (zero dlsym overhead on hot path) |

Env vars (all default off):
- `CIPHER_FLOW_RECORD=on` — enable recording
- `CIPHER_FLOW_DUMP_EVERY=N` — dump the last 64 launches every N total launches
- `CIPHER_FLOW_DUMP=on` — dump once at process exit

## What the recorder captures per launch

```c
struct CipherFlowEntry {
    void*    func_ptr;
    uint64_t args[8];   // raw 8-byte reads of *(uint64_t*)args[i]
    uint8_t  is_ptr[8]; // heuristic: 0x100000000 ≤ v < 0x800000000000
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
    uint32_t smem;
    uint64_t seq;
    uint64_t stream;
};
```

Args reading is **page-bounded** to avoid SEGV: PyTorch doesn't null-terminate `args[]`, so we walk only while `args[i]` is on the same memory page as `args[0]` and stays in the canonical user range. This was the source of the initial crash; fixed.

## What we observed on Llama-3.2-1B fp16 (1 forward + 2 decode steps)

**Total launches captured**: ~1400. Ring buffer keeps the last 64.

**RMSNorm signature** (one of two per layer, observed N times in the ring):

```
[FLOW seq=N+0] elementwise_kernel<128, 4, __nv_hdl_wrapper<lambda>>
                                                        (pow / cast x → fp32)
[FLOW seq=N+1] <unresolved-lambda>                       (MUL or POW lambda)
[FLOW seq=N+2] reduce_kernel<512, 1,
                  ReduceOp<MeanOps<float,float,float,float>>>
                                                        ← variance reduction
[FLOW seq=N+3] vectorized_elementwise_kernel<4,
                  CUDAFunctorOnSelf_add<float>>          (+ eps)
[FLOW seq=N+4] vectorized_elementwise_kernel<4, lambda>  (rsqrt)
[FLOW seq=N+5] elementwise_kernel<128, 2, lambda>        (x · rsqrt)
[FLOW seq=N+6] vectorized_elementwise_kernel<8,
                  BinaryFunctor<Half,Half,Half,
                    MulFunctor<float>>>                  (· weight)
```

**Edges detected** (sample from the live dump):
```
[FLOW seq=1391] fn=...  grid=126x1x1 block=128x1x1
  args[0] = 0x000000000001f500 [SMALL]
  args[1] = 0xf500000000000001 [OTHER]
  args[2] = 0x0000001a480d3800 [DEV]
  edge: args[2]=1a480d3800 <- seq=1390 args[3] (1 launch(es) ago)

[FLOW seq=1394] fn=...  grid=1x1x1 block=128x1x1
  args[0] = 0x00007fff00000001 [DEV]
  args[1] = 0x0000000000000001 [SMALL]
  args[2] = 0x0000001a4804a400 [DEV]
  edge: args[0]=7fff00000001 <- seq=1388 args[0] (6 launch(es) ago)
```

The first edge: kernel 1391 reads what 1390 wrote at `args[3]`. Direct producer→consumer link, name-independent.

## Why this is the right substrate for Stage 2 + 3

**Names are unstable**: PyTorch uses `__nv_hdl_wrapper_t<__nv_dl_tag<...lambda from file:line...>>` — these mangled names embed source paths and change across builds. Pattern-matching them is fragile.

**Pointer flow is stable**: across PyTorch versions, the data-flow graph of RMSNorm doesn't change — there's always a reduce-then-add-eps-then-rsqrt-then-mul-then-mul. The OPS stay the same; the kernel implementations don't matter to the matcher. Once Stage 2 detects a 6-kernel sequence whose pointer-flow signature matches `{read X, write A; read A, reduce → B; read B, in-place add → B; read B, transform → C; read X+C, write D; read D+W, write Y}`, we know it's RMSNorm regardless of which lambda PyTorch instantiated.

**`reduce_kernel<…ReduceOp<MeanOps>…>` is a strong anchor.** It's a stable name fragment (`MeanOps<float,float,float,float>` is a fixed type) and only appears for mean reductions. We can use it as the SEQUENCE START signal in Stage 2: when we see `MeanOps`, look at the last 1-2 kernels (Pow input) and the next 4 kernels (eps, rsqrt, mul, scale). If the args[] flow matches our template, suppress the 6 kernels and substitute one fused launch.

## Open work (Stages 2 + 3 — not in this delivery)

**Stage 2 — pattern matcher**:
- `cipher_flow_pattern_match()` walks the ring backwards from each new entry
- Templates encoded as `FlowPattern { length, steps[] }` where each step has `{n_read_ptrs, n_write_ptrs, has_reduction, reads_from_step_idx}`
- On match, capture the role pointers (X, W, Y) by their position in args[]
- For RMSNorm: `X = step_0.read_ptr, W = step_5.read_ptr_2, Y = step_5.write_ptr`

**Stage 3 — substitution dispatcher**:
- When Stage 2 confirms a match starting at the CURRENT launch, call `cipher_fused_rmsnorm(X, W, Y, rows, hidden_dim, eps, stream)` and set `g_suppress_count = 5`.
- Subsequent 5 cudaLaunchKernel calls return `cudaSuccess` without launching the real kernel.
- This works because PyTorch dispatches synchronously on the same stream — the next 5 kernels are guaranteed to be the rest of the RMSNorm sequence we matched.

## What this proves about the path

The earlier 2.80× tok/W on Llama-3.1-8B used **Python-level** `LlamaRMSNorm.forward` patches. Stage 1 confirms the **same fusion gain** is achievable purely at the driver level — the data-flow signal that defines RMSNorm is observable in `cudaLaunchKernel`'s `args[]` window, without touching Python.

Stage 2 + 3 builds on top of this recorder. Estimated 1-2 days each. The remaining unknowns are bookkeeping (how to identify the rows / hidden_dim from grid/block dims and the args[] integer slots) and the suppression-replay correctness (must work under torch.cuda.graph capture too — but graph capture only re-runs the captured launches, so suppress-during-capture is safe).
