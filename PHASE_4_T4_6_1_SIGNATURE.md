# T4.6.1 — Verified PyTorch SDPA dispatcher signatures + drift-handling

This document records the exact ATen ::call signatures that the T4.6.1
attention substrate interposes on, the symbol-mangling that produced
the LD_PRELOAD trampolines, and the runtime behavior on signature drift.

## Verified-at stack

- **torch**: 2.11.0+cu130
- **cuDNN backend**: enabled (default on Hopper / CUDA 13)
- **Distribution**: `/home/ubuntu/.local/lib/python3.10/site-packages/torch`
- **SONAME**: `libtorch_cpu.so` (no version tag) and `libc10.so` (no version tag)

## Symbol-versioning posture

Unlike libcublas.so.13 (where `cublasGemmEx@@libcublas.so.13` is
version-tagged and T4.5.1 substrate uses `.symver` directives), the
ATen `_ops::_scaled_dot_product_*_attention::call` symbols in
libtorch_cpu.so are **plain mangled C++ symbols with no
`@@version` tag**. Verified empirically:

```
$ nm -D --with-symbol-versions libtorch_cpu.so | grep _scaled_dot_product_flash_attention
_ZN2at4_ops35_scaled_dot_product_flash_attention4callERKNS_6TensorES4_S4_dbbSt8optionalIdE
```

(No trailing `@@` annotation.) LD_PRELOAD interposition therefore works
with plain mangled-name shim definitions; no linker version map is
needed, and the existing `cublas_version.map` (which scopes only
libcublas.so.13 symbols) does not list the SDPA symbols. **Critical
linker note:** putting these symbols in a version map's `global:` clause
with `local: *;` would HIDE them and silently disable interposition.

## Three intercepted dispatcher entries

Each SDPA backend has its own `_ops::*::call` entry in libtorch_cpu.so.
The substrate trampolines all three; PyTorch's runtime backend selector
(`torch.nn.functional.scaled_dot_product_attention` + `sdpa_kernel`
context) routes a given attention call to exactly one of these.

### 1. Flash attention

**Mangled:**
```
_ZN2at4_ops35_scaled_dot_product_flash_attention4callERKNS_6TensorES4_S4_dbbSt8optionalIdE
```

**Demangled:**
```cpp
at::_ops::_scaled_dot_product_flash_attention::call(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale
) -> std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                c10::SymInt, c10::SymInt,
                at::Tensor, at::Tensor, at::Tensor>
```

**ATen schema string:**
```
_scaled_dot_product_flash_attention(Tensor query, Tensor key, Tensor value,
  float dropout_p=0.0, bool is_causal=False, bool return_debug_mask=False,
  *, float? scale=None)
  -> (Tensor output, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k,
      SymInt max_q, SymInt max_k, Tensor rng_state, Tensor unused,
      Tensor debug_attn_mask)
```

**Note:** Return tuple uses `c10::SymInt` for max_q/max_k (NOT
`int64_t`). Substituting `int64_t` would silently corrupt the
return slot — see SymInt note below.

### 2. Efficient attention (xFormers-style)

**Mangled:**
```
_ZN2at4_ops39_scaled_dot_product_efficient_attention4callERKNS_6TensorES4_S4_RKSt8optionalIS2_EbdbS5_IdE
```

**Demangled:**
```cpp
at::_ops::_scaled_dot_product_efficient_attention::call(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    bool compute_log_sumexp,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale
) -> std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
```

### 3. cuDNN attention (the one Mistral-7B actually hits on H100/cu13)

**Mangled:**
```
_ZN2at4_ops35_scaled_dot_product_cudnn_attention4callERKNS_6TensorES4_S4_RKSt8optionalIS2_EbdbbS5_IdE
```

**Demangled:**
```cpp
at::_ops::_scaled_dot_product_cudnn_attention::call(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    bool compute_log_sumexp,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale
) -> std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                c10::SymInt, c10::SymInt,
                at::Tensor, at::Tensor, at::Tensor>
```

## SymInt note

The flash and cuDNN return tuples carry `c10::SymInt max_q, max_k`.
`c10::SymInt` is a tagged union (concrete int OR pointer to a symbolic
node), not a plain `int64_t`. Substituting `int64_t` would either:
- Silently corrupt the return slot if sizes match but copy semantics
  differ (refcount-bumps vs trivially-copyable)
- Crash if sizes differ

The substrate uses the ATen header schema typedef directly
(`#include <ATen/ops/_scaled_dot_product_flash_attention_ops.h>`) so
binary layout matches what libtorch builds against, by construction.

## Runtime drift behavior

**Per-symbol lazy resolution.** The substrate does NOT eagerly dlsym
at libcipher_rt init (cuInit-time dlsym fails because libtorch_cpu was
loaded as a NEEDED dependency, but RTLD_NEXT from libcipher_rt at that
moment fails — possibly due to dlopen-order constraints when
CUDA_INJECTION64_PATH triggers a dlopen of the already-loaded
libcipher_rt). Instead, each trampoline resolves its own original on
first call via `dlsym(RTLD_NEXT, MANGLED_*)`. The resolved pointer is
cached in an `std::atomic` per backend.

**Build-time invariant:** libcipher_rt.so is linked with
`-Wl,--no-as-needed -ltorch_cpu`, forcing libtorch_cpu into DT_NEEDED
even though our trampolines satisfy the symbol names ourselves. Without
this the linker would drop the dependency (we'd appear to provide the
symbols), and dlsym RTLD_NEXT at runtime would find nothing to fall
through to.

**Failure mode on signature drift.** If a future torch upgrade renames
or re-mangles one of the symbols:

| Scenario | Behavior |
|---|---|
| Symbol exists with same mangling, same signature | Substrate operates normally |
| Symbol exists with same mangling, DIFFERENT signature | Undefined behavior — caller passes args matching our shim signature, callee expects different layout. Possible crash or corruption. **Operator must re-verify signature on torch upgrade.** |
| Symbol does not exist (renamed) | dlsym fails on first SDPA call. Trampoline prints `[cipher-attn] FATAL: <backend> dlsym never resolved` and calls `abort()`. Host application terminates with SIGABRT. |
| Symbol exists in different .so | dlsym RTLD_NEXT walks library order; if libtorch_cpu is in DT_NEEDED of libcipher_rt and exposes the symbol, resolution succeeds. |

## Why `abort()` on dlsym failure, not "register inactive and fall through"

The phase prompt's S5.1 suggested "register substrate as INACTIVE, fall
through to PyTorch's original" — this is **not possible by construction**
once libcipher_rt.so exports the mangled trampoline name. LD_PRELOAD
preempts the symbol unconditionally; there is no runtime "unregister"
once the linker has resolved a caller to our shim's address.

The only options at trampoline time when dlsym fails are:
1. **abort()** — host application terminates loudly with a clear error
2. **Return zero-initialized tuple** — would silently corrupt the model
   output, propagating NaNs / nonsense through downstream layers

**(1) is correct.** Loud failure on first SDPA call is far safer than
silent model corruption. Operators upgrading PyTorch must re-verify
T4.6.1 against the new mangled symbol names (via `nm -D libtorch_cpu.so
| c++filt | grep _scaled_dot_product_*_attention.*::call`) before
running CIPHER on the new torch.

The substrate's signature-drift safety is therefore: **loud crash on
incompatibility, not silent corruption.** This is the documented
contract.

## Coverage gap (acknowledged)

- **cuDNN trampoline empirically verified** on Mistral-7B / H100 /
  torch 2.11 / cu13 (3,072 cuDNN calls intercepted in S3 smoke;
  32,896 in S4 composition; byte-identical output across all runs).
- **Flash trampoline** built and symbol-exported, never hit at runtime
  on this stack (cuDNN is the default backend). First flash-backend
  workload through CIPHER will be the real smoke test.
- **Efficient trampoline** built and symbol-exported, never hit on
  this stack. Same caveat.

The flash and efficient trampolines are link-correct and signature-
matched against the verified ATen headers, but the lazy-dlsym +
trampoline-body path has not been exercised end-to-end in production.
Pre-Hopper hardware or non-cuDNN-default stacks will exercise these
on first encounter.

## Verification recipe (when torch is upgraded)

```bash
TORCH_LIB=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
for op in flash efficient cudnn; do
  echo "=== $op ==="
  nm -D --with-symbol-versions $TORCH_LIB/libtorch_cpu.so \
    | grep "_scaled_dot_product_${op}_attention.*::call\|_scaled_dot_product_${op}_attention4call" \
    | head -3
done
```

If the demangled signature differs from this document, regenerate the
mangled name (via `c++filt`) and update `MANGLED_FLASH`/`MANGLED_EFF`/
`MANGLED_CUDNN` constants in `cipher_rt_attn_dispatch.cpp`.
