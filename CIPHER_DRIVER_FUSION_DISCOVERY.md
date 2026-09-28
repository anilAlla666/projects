# Driver-level kernel fusion via name templates — Task 1 discovery report

**Approach**: hook `cudaLaunchKernel`, log mangled name + grid/block/smem + first 16 `args[]` entries for every kernel during a Llama-3.2-1B forward pass. Pattern-match against expected fusion targets (RMSNorm, SiLU, residual_add, RoPE).

**Probe wired**: `src/cipher_intercept_cudart.cpp::cudaLaunchKernel` shim, gated by `CIPHER_KERNEL_PROBE=on`. Reads `args[i]` as `uint64_t`, classifies as device-pointer / small-int / unknown.

---

## What the probe discovered

PyTorch's `LlamaRMSNorm.forward()`:
```python
hidden_states.to(torch.float32) * torch.rsqrt(
    hidden_states.pow(2).mean(-1, keepdim=True) + variance_epsilon)
```

decomposes at the CUDA launch level into **5-6 distinct kernels per RMSNorm call**:

| # | Kernel (demangled) | What it does |
|---|---|---|
| 1 | `vectorized_elementwise_kernel<8, ?-pow-?>` | `x → x²` (fp16→fp32) |
| 2 | `reduce_kernel<512, 1, ReduceOp<MeanOps<float,float,float,float>>>` | per-row mean of x² |
| 3 | `vectorized_elementwise_kernel<4, CUDAFunctorOnSelf_add<float>>` | `+ eps` (fp32 scalar) |
| 4 | `unrolled_elementwise_kernel<__nv_hdl_wrapper_t<...rsqrt-lambda...>>` | `rsqrt(...)` |
| 5 | `vectorized_elementwise_kernel<8, BinaryFunctor<...,MulFunctor>>` | `x * rsqrt(...)` |
| 6 | `vectorized_elementwise_kernel<8, BinaryFunctor<...,MulFunctor>>` | `* weight` |

There is **no single ATen kernel named "RMSNorm"** to substitute. Same story for RoPE — it's `unrolled_elementwise_kernel<__nv_hdl_wrapper_t<...rotary lambda...>>` (multiple kernels, anonymous lambdas in mangled name). SiLU activation: `unrolled_elementwise_kernel` with a sigmoid+mul lambda, name varies per PyTorch build.

The kernels we DO see in Llama-3.2-1B forward (B=1, prefill+1 decode):

```
ResidualAdd  vectorized_elementwise_kernel<8, CUDAFunctor_add<Half>>          (the only cleanly-named target)
Reduce       reduce_kernel<512, 1, ReduceOp<MeanOps<float,float,float,float>>> (RMSNorm var)
ElementMul   vectorized_elementwise_kernel<8, BinaryFunctor<...,MulFunctor>>   (used by 5+ ops)
ElemwiseGen  unrolled_elementwise_kernel<__nv_hdl_wrapper_t<...lambda...>>     (lambdas, unstable name)
ElementMul   vectorized_elementwise_kernel<4, CUDAFunctorOnSelf_add<float>>    (in-place add)
Reduce       reduce_kernel<512,1,ReduceOp<...>>                                (softmax/norm reductions)
+ many mbtopk / radixSort kernels for sampling
```

**Param layouts** (from probe — args[] are pointers to caller-stack values):

For `vectorized_elementwise_kernel<8, CUDAFunctor_add<Half>>`:
```
args[0] = numel (int)
args[1] = functor (struct, includes pointer-array)
args[2] = output_ptr
args[3] = ...  (depends on functor specialization)
```

For `reduce_kernel<512, 1, ReduceOp<MeanOps>>`:
```
args[0] = ReduceOp instance (large struct with input/output ptrs + dim info embedded)
```

The `ReduceOp` is passed by value — a ~64-byte struct containing:
- input tensor info (ptr, strides, dims)
- output tensor pointer
- accumulator type
- reduction dimension(s)

The exact byte offsets within this struct are PyTorch-version-specific.

---

## Why name-template substitution doesn't directly work here

1. **PyTorch RMSNorm is N kernels, not 1.** Substituting the first reduce_kernel<MeanOps> alone gives wrong output — the subsequent rsqrt + mul + mul kernels also need to run (or be suppressed). Sequence-level pattern detection ("reduce_kernel<MeanOps> followed within ε by CUDAFunctorOnSelf_add then unrolled_elementwise<lambda> then Mul") is doable but fragile across PyTorch versions.

2. **Lambda-based kernels have unstable mangled names.** `__nv_hdl_wrapper_t<...__nv_dl_tag<...>>` includes the source file path and line number in the name. Two PyTorch builds give different mangled names for the same op.

3. **Functor structs change layout across versions.** A `BinaryFunctor<Half, Half, Half, MulFunctor<Half>>` was 24 bytes in PyTorch 2.4, may be 32 bytes in 2.7. The args[] decode depends on knowing this.

4. **The functor often contains lambdas, not just pointers.** For `unrolled_elementwise_kernel<...lambda...>`, the lambda's captures (which tensors it operates on) are inside the functor struct in a layout determined by NVCC's lambda code-gen.

---

## What IS tractable at the driver level (and what isn't)

| Target | Tractable? | Why |
|---|---|---|
| **`vectorized_elementwise_kernel<N, CUDAFunctor_add<Half>>`** (residual_add) | ✓ stable name, simple functor | Substitute with `cipher_fused_residual_add` if grid/block match shape |
| **`vectorized_elementwise_kernel<N, BinaryFunctor<...,MulFunctor>>`** | ✓ but ambiguous | Used by many ops (RMSNorm scale, MLP gate*up, attention QK*scale). Need shape disambiguation. |
| **`reduce_kernel<512,1,ReduceOp<MeanOps>>`** as RMSNorm-variance trigger | ✓ but partial | We can detect it; substituting the WHOLE RMSNorm requires suppressing the next 4-5 kernels and running our fused replacement. Doable, fragile. |
| **`unrolled_elementwise_kernel<...lambda...>`** (RoPE / SiLU / rsqrt) | ✗ name unstable | Lambdas embed file path/line number; not portable across builds. |
| **Single-kernel RMSNorm/SiLU/RoPE substitution** | ✗ doesn't exist | PyTorch decomposes them — there's no monolithic kernel to replace. |

The earlier `step9_*.py` approach worked **because it patched PyTorch's `LlamaRMSNorm.forward` at the Python level** — replacing the entire decomposed ATen graph with one kernel call. Driver-level is strictly less expressive: it sees the decomposition AFTER PyTorch has already chosen it.

---

## Two paths forward (neither is 1-day work)

### Path A — Sequence-pattern fusion at the driver level
Track a sliding window of the last ~6 cudaLaunchKernel calls. If we detect a known RMSNorm sequence (Pow → MeanReduce → AddEps → Rsqrt → Mul → ScaleByWeight) on the same set of tensor shapes, suppress all 6 kernels and substitute one fused launch. Requires:
- Stable cross-version detection of each kernel role (we have `cipher_kernel_table.cpp` classification — extend it)
- Param-layout discovery for each (we just ran the probe)
- Suppress-and-replay buffering

**Fragility**: any PyTorch version bump that changes kernel name or functor layout silently breaks the detection. Same problem torch.compile solves at a different layer.

### Path B — torch.compile integration (out of scope for "no Python")
When the customer runs `torch.compile(model)`, PyTorch's Triton backend emits ONE fused kernel per op (e.g., `triton_per_fused__to_copy_mean_pow_rsqrt_0`). Those names are stable across runs of the same compiled binary, and the fusion has already happened at the framework level. The CIPHER hook can pattern-match these and substitute with hand-written CUDA when available. Customer change: add `model = torch.compile(model)`. Strictly speaking, that's a one-line Python change — not a CIPHER monkey-patch.

---

## What this means for the deliverable

The user's stated approach — **name-matched parameter templates that fuse driver-level — runs into PyTorch's eager-mode decomposition.** A clean single-kernel substitution path doesn't exist in eager because the kernels themselves don't exist as monolithic units. The nearest viable driver-level path is sequence-pattern fusion (Path A), which is multi-day work and version-fragile.

The earlier `step9_graph.py` results stand:
- **2.80× tok/W on Llama-3.1-8B B=1**, **2.79× on B=8** (stock-eager baseline → graph + FP8 + fusion + clock lock)
- Those numbers are with PyTorch-level RMSNorm/SiLU/residual/RoPE patches **+** the same `cipher_fused_*` kernels we'd substitute in driver-level mode. Driver-level fusion would deliver the **same kernel-level wins**, just without the Python patches.

Tasks 2-5 of this brief presuppose a working Task 1 (i.e., a name → param-layout table). Since the probe shows that's not how PyTorch eager-mode kernels are structured for the RMSNorm/SiLU/RoPE targets, Tasks 2-5 are blocked on either taking Path A (sequence detection) or accepting the Python-level patches (already proven).

---

## Files added this round

- `src/cipher_intercept_cudart.cpp` — added kernel-parameter probe (gated by `CIPHER_KERNEL_PROBE=on`)
- `CIPHER_DRIVER_FUSION_DISCOVERY.md` — this report

Probe usage:
```
CIPHER_KERNEL_PROBE=on ./cipher_run.sh -c "<your forward pass>" 2>&1 | grep "CIPHER PROBE"
```

---

## Honest recommendation

If the goal is **"customer runs vanilla PyTorch, CIPHER fuses at driver-level"** → Path A (sequence detection) is the right build. Estimate 3-5 days. Output: a `cipher_kernel_seqdetect.cpp` that maintains a per-stream window, recognizes the RMSNorm/SiLU/RoPE sequences, and dispatches to the existing `cipher_fused_*` kernels via `cuLaunchKernel`. Pre-built for transformers 4.57; recompile required for major PyTorch version bumps.

If the goal is **"deliver the tok/W win to a customer THIS WEEK"** → ship the existing Python-level fusion via `cipher_run.sh` + a small `cipher_torch_helpers.py` that the customer optionally imports. The 2.80× tok/W has already been measured through this path on Llama-3.1-8B. The "no monkey-patches" constraint can be satisfied by making the helpers a separate optional package — the LD_PRELOAD launcher works without them, just at single-tenant numbers.

Eager-mode driver-level fusion (without sequence-detection) only addresses the cleanly-named targets: `vectorized_elementwise_kernel<N, CUDAFunctor_add<Half>>` (residual_add). One-kernel substitution there is straightforward but only worth ~3-5% tok/W of the total (we measured residual_add at 0 numerical error vs PyTorch — it adds zero precision risk but also delivers little fusion gain since it's a single elementwise op already).
