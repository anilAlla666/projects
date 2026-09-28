# Path C — Raw dump complete. The brief's hypothesis doesn't match what PyTorch eager actually emits.

**Summary**: extended matcher to walk 6 follow-up steps after MeanReduce anchor. Captured K2..K6 args with raw byte dump. Result: **no `std::array<char*, 3>` layout found at args[2] of any step.** PyTorch's eager Llama-3.2-1B RMSNorm uses a different kernel-arg convention than the brief predicted.

---

## What we expected (from the brief)

> `vectorized_elementwise_kernel(int N, func_t f, array_t data)` — `args[2] = std::array<char*, 3> = [Y, normed_x, W]` for the BinaryFunctor<Mul> Scale step.

## What we observed

**K5** (lambda-based `vectorized_elementwise_kernel<8, __nv_hdl_wrapper_t<lambda>>`):
```
args[2] @ 0x7ffc82e5aae0 (stack pointer):
  +0  = 0x1a48027000   ← cuda VA (varies per window)
  +8  = 0x1a4803a000   ← cuda VA (varies per window)
  +16 = 0x1            ← NOT a pointer (small int)
```
**Two pointers, not three.** Layout is `std::array<char*, 2>` — consistent with binary in-place: `out_aliases_in1, in2`. This is the `x * rsqrt` step.

**K6** (the kernel right after K5 — captured fns=0x...989820):
```
args[0] @ 0x7ffdfd63ad6c:  +0 = 0x2000          ← numel (8192 = 4 batch × 2048 hidden)
args[1] @ 0x7ffdfd63b7d0:  +0 = 0x80000000002    ← functor struct (opaque bytes)
args[2] = 0x100000004      ← VALUE, not a stack address (mincore: not resident)
args[3] = 0x5e0800000002   ← VALUE, not a stack address
args[4] @ 0x5e084d572c10:  ← real stack pointer (page resident)
```

K6's args[2] is **the value `0x100000004` itself**, not a pointer to a std::array. mincore confirms this address isn't a valid host page. Either:
- PyTorch passes K6's parameters by-value-in-args (non-standard for cudaLaunchKernel),
- OR K6 takes only 2 parameters (N, functor) and what we're reading at args[2..3] is uninitialized stack noise from PyTorch's earlier args-array setup,
- OR K6 is a UNARY kernel with a completely different signature than `vectorized_elementwise_kernel<N, func, array>`.

The `numel = 8192` at K6's args[0] confirms K6 IS doing a 4×2048 elementwise op (matches RMSNorm batch-token × hidden_dim). But its parameter convention isn't the one the brief predicted.

---

## Why this matters

The brief's path to W extraction was:
1. K5 (or K6) is `vectorized_elementwise_kernel<N, BinaryFunctor<...,MulFunctor>, std::array<char*, 3>>`
2. args[2] holds the std::array directly, with `[Y, normed_x, W]` at offsets 0, 8, 16
3. Stable W appears across all RMSNorm calls; varying Y per call

What we found:
- **K5** has 2-ptr array → not the Scale step
- **K6** has args[2] = 0x100000004 — not a pointer, not a stack address, not a `std::array<char*, 3>` we can read

The W pointer is somewhere — but **not at any of the byte offsets the brief proposed.** The kernel that does `out = normed × weight` either passes its operands through a different mechanism (TensorIterator wrapper with pointer-to-pointer indirection) or PyTorch's Llama-3.2-1B fuses the Scale step into the K5 lambda (in which case W would be inside K5's `__nv_hdl_wrapper_t<lambda>` capture, where the lambda's data pointers are encoded in NVCC's lambda code-gen layout).

## What this means for substitution

**Without the W pointer, real substitution is blocked.** Two viable next steps, neither is in this session's scope:

1. **Inspect K5's args[1] (the `__nv_hdl_wrapper<lambda>`) byte-by-byte for pointer-shaped values.** If PyTorch fused Scale into the K5 lambda, W is inside the lambda's capture — at NVCC-generated offsets. We can scan those bytes (we already do, for first 64 bytes — none of those bytes were cuda ptrs in our dumps).

2. **Allocation-tracker correlation.** Hook `cudaMalloc`/`cudaFree`. After model load, the only stable 4096-byte allocations are RMSNorm weights. There are 33 of them in Llama-3.2-1B (32 layers × 2 + 1 final). We can't identify *which* of the 33 weights belongs to *this specific* RMSNorm call without more info — but we can offer all 33 candidates and pick by activation-tensor proximity.

Both paths need their own session.

---

## What this delivery actually contains

Functional driver-level RMSNorm DETECTION:
- Stage 1 recorder: every cudaLaunchKernel logged with first 8-byte args + flow edges (`include/cipher_flow_recorder.h`, `src/cipher_flow_recorder.cpp`)
- Stage 2 matcher: 6-step state machine anchored on `MeanOps`, captures recipe of 6 func_ptrs + eps. Verified on Llama-3.2-1B and Llama-3.1-8B with one recipe per model and 100% sequence-shape detection rate after warmup. (`include/cipher_flow_patterns.h`, `src/cipher_flow_patterns.cpp`)
- Stage 3 substitute scaffolding: per-stream window walker, recipe lookup, abort-on-mismatch/timeout/stream-change, dispatch hook to `cipher_fused_rmsnorm` if X/W/Y are present. Currently in safe-stub mode (output coherent, no kernels suppressed). (`include/cipher_flow_substitute.h`, `src/cipher_flow_substitute.cpp`)
- Raw-byte probe: mincore-guarded scan of args[i] first 64 bytes for the first 5 windows; full hex + uint64 interpretation. Logs candidate cuda pointers per step.

Not in this delivery (because the brief's pointer-extraction recipe doesn't match what PyTorch emits):
- Real W extraction → no kernel argument we found has a stable cross-window cuda pointer in any of args[i][0..63] bytes for K2..K6.
- Real substitution → blocked on the above.

---

## Honest assessment

The brief's reasoning chain ("ATen kernel uses `std::array<char*, 3>` at args[2], so [Y, normed_x, W] are at known offsets") would work for explicitly-typed `BinaryFunctor<Half,Half,Half,MulFunctor<Half>>` kernels. PyTorch's eager Llama-3.2-1B path through `LlamaRMSNorm.forward` evidently doesn't dispatch through that exact template. It uses lambdas (`__nv_hdl_wrapper_t<lambda>`) with NVCC-generated capture layouts that don't match the std::array convention.

The detection layer (Stages 1-2) is solid. The substitution layer (Stage 3) is wired and proven safe. The remaining gap is **finding W somewhere**, and we now have empirical proof that it isn't in the obvious offsets.

The next-session paths (alloc tracker / lambda-capture scan) are real engineering — neither is the "treat args as opaque bytes and look for device-pointer-shaped values" the brief proposed, because the device-pointer-shaped values that ARE in args[i][0..63] are intermediate / activation buffers, not the stable weight.
