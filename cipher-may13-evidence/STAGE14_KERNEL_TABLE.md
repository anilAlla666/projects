# Stage 14 — Driver-level kernel table + the cuFuncGetParamInfo blocker

Date: 2026-04-30
Pod: H100 80GB SXM, CUDA 12.8, sm_90, PyTorch 2.7.

## Task 1 — kernel name recognition table: shipped

New hook-DSO module `src/cipher_kernel_table.cpp` (+ header).  Wired into
both intercept paths:

- `cuLaunchKernelEx`  — driver-API launches (cuBLAS, cuBLAS-LT, FlashAttention loaded via cu* modules)
- `cudaLaunchKernel`  — runtime-API launches (every PyTorch nn.Linear / RMSNorm / SiLU / RoPE / elementwise / etc.)

For every distinct CUfunction observed, the table records:
- name (via `cuFuncGetName` → fallback `dladdr` on the host stub),
- coarse category (GEMM / RMSNorm / SiLU / RoPE / FlashAttn / ResidualAdd / ElementMul / ElementGeneric / Reduce / Cast / Copy / Unknown),
- grid + block dimensions on first observation,
- per-param offsets/sizes via `cuFuncGetParamInfo` (CUDA 12+),
- launch count.

JSON dump at `/tmp/cipher_kernel_table.json` on demand or at process
exit (env `CIPHER_KERNEL_TABLE_VERBOSE=1`).

Default-off / zero-cost: each launch does a single linear-probe hash-
table lookup, single atomic increment on hit.  First-observation work
is one-shot per CUfunction.

## Confirmed measurement on Mistral-7B prefill+decode

Test: vanilla HuggingFace `from_pretrained(...)` + `model.generate(max_new_tokens=16)`
under `LD_PRELOAD=libcipher_hook.so`.  Customer code: zero CIPHER imports,
zero monkey-patches, just `LD_PRELOAD`.

```
Total CUfunctions observed: 62
Total launches:             21192

  category           count  launches   top kernel name
  ─────────          ─────  ────────   ───────────────
  ElementMul            18      8043   _ZN2at6native18elementwise_kernel...
  Unknown               27      7337   <unresolved-0x...>            (cuBLAS hgemm)
  ElementGeneric         5      2129   _ZN2at6native27unrolled_elementwise...
  ResidualAdd            4      2081   _ZN2at6native29vectorized_elementwise...
                                        CUDAFunctor_addINc10HalfEEE
  Reduce                 6      1090   _ZN2at6native13reduce_kernelILi512E
                                        ReduceOpIfMeanOps
  FlashAttn              2       512   _ZN13pytorch_flash16flash_fwd_kernel
                                        Flash_fwd_kernel_traitsILi128ELi128ELi64E
```

Name-based classification works: every FlashAttention launch is
correctly tagged `FlashAttn`, every elementwise add is tagged
`ResidualAdd` via the `CUDAFunctor_add` substring, every variance
reduction is tagged `Reduce` via `ReduceOpIfMeanOps`, etc.  The 27
"Unknown" entries are cuBLAS Hopper-optimized hgemm kernels where
both `cuFuncGetName` (returns failure) and `dladdr` on the host stub
(no symbol in libcublas) cannot recover a name — they're identifiable
by grid pattern (`(2, 64, 1)` block `(384, 1, 1)`, smem ~230 KB) but
not by string match.

## The blocker: `cuFuncGetParamInfo` returns 0 for ALL 62 kernels

```
Param-count distribution:
  param_count = 0  →  62 kernels  (100 %)
```

Every one of the 62 distinct kernels observed in Mistral-7B forward
returns `param_count = 0` from `cuFuncGetParamInfo`.  This includes
the kernels we'd want to substitute:

- The CUDAFunctor_add kernel (residual add) — `param_count = 0`.
- The flash_fwd_kernel (attention) — `param_count = 0`.
- Every elementwise variant (RMSNorm post-mul, SiLU, RoPE multiply) — `param_count = 0`.
- Every reduce kernel (RMSNorm variance) — `param_count = 0`.

This reproduces the prior session's finding (CLAUDE.md: *"`cuFuncGetParamInfo`
returns 0 params for all PyTorch runtime-API kernels — they're host
stubs registered via `__cudaRegisterFunction`, not real driver
CUfunctions, so parameter layout isn't queryable from the driver
API"*) on the current pod with PyTorch 2.7 / CUDA 12.8.

## Why this blocks Tasks 2–5

The user's plan for Tasks 2–5 (RMSNorm / SiLU / Residual / RoPE
substitution) reads:

> *"Read the kernel's parameters via the param layout from Task 1:
>  input_ptr, weight_ptr, output_ptr, hidden_dim, eps."*

Without `cuFuncGetParamInfo` returning offsets and sizes, we cannot
safely read those pointers from the `void**` array that
`cuLaunchKernel` passes — every PyTorch template (`elementwise_kernel`,
`vectorized_elementwise_kernel`, `unrolled_elementwise_kernel`,
`reduce_kernel`) is used for **dozens of distinct operations** with
**different parameter layouts**, decided at compile time by the host
stub.  We'd be guessing offsets per-template-per-PyTorch-version —
fragile, version-dependent, and unsafe (one wrong offset means we
write into the wrong device buffer).

Concrete example: `at::native::elementwise_kernel<128,4,...>` is the
top kernel in the table at 8 043 launches.  That single kernel
template implements: residual add, mul-by-scalar, mul-by-broadcast,
rsqrt, the post-RMSNorm weight multiply, RoPE-style cos/sin multiply,
fp16↔fp32 cast, etc.  Same CUfunction handle, different operations.
Without param-info we can't tell which.

## Recommendation: ship a Python-level shim

The user's earlier directive was *"CIPHER's value is ZERO application
changes."*  Two ways to honor that:

(a) **Strict driver-only** — what Tasks 2–5 specified.  Blocked at the
    `cuFuncGetParamInfo` foundation, per above.  Driver-level FP8 in
    `cublasGemmEx` works because cuBLAS exposes a documented API; the
    same does not hold for PyTorch's runtime kernels.

(b) **Driver-only + tiny Python shim loaded by `LD_PRELOAD`** — install
    the monkey-patches via a shim module the customer auto-imports.
    Two ways the customer ends up running it:
       - `python -m cipher.run -- <their_script.py>` (zero edits to
         their script),
       - or set `PYTHONSTARTUP=cipher_startup.py` in the same env where
         `LD_PRELOAD` is set (zero edits to their script),
       - or the LD_PRELOAD library injects a `_PyImport_AppendInittab`
         callback that auto-imports cipher's monkey-patcher when Python
         starts up.

Path (b) preserves the "no model code changes" property of the product.
The customer's `model = AutoModelForCausalLM.from_pretrained(...)` line
is unchanged; the patches happen invisibly at process start.  This is
how `vllm`'s in-place attention substitution works in practice.

## Files this round

- `include/cipher_kernel_table.h` — public API
- `src/cipher_kernel_table.cpp` — implementation (~280 LOC)
- `src/cipher_intercept_cudart.cpp` — wired `cipher_kt_observe` into
  both cuLaunchKernelEx and cudaLaunchKernel paths (a few-line insert
  on the hot path)
- `Makefile`, `exports.map` — added the new TU and 3 exports
  (`cipher_kt_observe / _size / _dump_json`)
- `tests/test_kernel_table.py` — vanilla HuggingFace driver, dumps the
  classification breakdown
- `STAGE14_KERNEL_TABLE.md` — this file

## Op regression

```
REGRESSION SUMMARY: 32 passed, 4 failed, 8 skipped
```

Test count rose by 1 (the new `test_kernel_table.py` joined the green
set).  Same 4 pre-existing failures (cipher_runtime missing, hardcoded
path from prior pod, persist_dispatch overhead gate).  No regression.

## What I'm not shipping until the user picks a path

Tasks 2 (RMSNorm), 3 (SiLU), 4 (residual_add), 5 (RoPE) — all four
need parameter offsets to read input/output/weight pointers from the
intercepted launch.  Without param-info that's fragile.

If the user picks path (b) I can ship a `cipher.run` launcher (or a
`PYTHONSTARTUP` shim) that applies the existing monkey-patches at
process start — preserving "no application code changes" while
sidestepping the `cuFuncGetParamInfo` blocker.

If the user insists on path (a) the unblocking work is:
- per-template parameter-layout templates (i.e., a database of which
  PyTorch op uses which offsets in `at::native::elementwise_kernel`)
  versioned against the PyTorch wheel,
- maintained per release,
- still fragile across PyTorch versions.

That's a multi-engineer-month project, not a session task.  Recommend
path (b).
