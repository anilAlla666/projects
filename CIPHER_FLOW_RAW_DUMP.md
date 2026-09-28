# Path C — Raw hex dump of args[i] (Task 1 of brief)

User hypothesis: `args[2]` of the BinaryFunctor<Mul> Scale step contains `std::array<char*, 3> = [Y, normed_x, W]`. Verifying with raw byte dump, no filtering.

**Key finding: args[2] of the kernel my matcher captures as "K5" contains only 2 cuda pointers, not 3.** That kernel isn't BinaryFunctor<Mul> — it's the in-place `x * rsqrt` step. The actual Scale-by-Weight is **K6**, one step beyond my matcher's current 5-step walk. Extending to 6 steps + reading args[2] there should give us all three pointers including W.

---

## What the raw dump showed

### K2 (CUDAFunctorOnSelf_add — `+ eps`):

```
args[2] @ 0x7fffb2758f40 (64 bytes):
  +0  = 0x0000001a48022400   ← cuda ptr (stable across windows)
  +8  = 0x0000001a48022200   ← cuda ptr (stable across windows)
  +16 = 0x0000000000000010   (not a ptr)
```

In-place op. Two ptrs at +0/+8 are reused across all windows — these are TensorIterator's intermediate buffer recycling, not the actual variance-out.

### K5 (vectorized_elementwise_kernel<8, __nv_hdl_wrapper_t<lambda>> — `x * rsqrt`):

Window 0:
```
args[2] @ 0x7ffc82e5aae0:
  +0  = 0x0000001a48027000
  +8  = 0x0000001a4803a000
  +16 = 0x0000000000000001   (not a ptr — 0x1)
```

Window 1:
```
  +0  = 0x0000001a48032000
  +8  = 0x0000001a48040000
  +16 = 0x0000000000000001
```

Window 2:
```
  +0  = 0x0000001a48032000
  +8  = 0x0000001a4803c000
  +16 = 0x0000000000000001
```

**`std::array<char*, 2>`, not `<3>`.** Only +0 and +8 are cuda pointers; +16 is the small int `1`. This is consistent with an in-place binary op (`out = in × other` where `out` and one input alias, so the kernel needs only 2 distinct pointers to reach all 3 logical operands).

This K5 is clearly the **`x * rsqrt`** intermediate step, not the final Scale-by-Weight. Its name in the recipe was `vectorized_elementwise_kernel<8, __nv_hdl_wrapper_t<...lambda...>>` — a lambda template, not BinaryFunctor.

The +0/+8 pointers vary per window. They're activation buffers (X and intermediate-normed-output). One of them is X — confirmed cross-reference with the cublasGemmEx log:
```
[CIPHER GEMM-EX] #0 ... C=0x1a48027000   ← prior GEMM output
[CIPHER FLOW RAW] K5 ... args[2][+0] = 0x1a48027000   ← reads same buffer
```

So in window 0, `args[2][+0] = 0x1a48027000 = X` (the activation that came in as RMSNorm input).

---

## Why the matcher's K5 isn't the Scale step

PyTorch's eager Llama-3.2-1B decomposes RMSNorm into more than 5 follow-up kernels after MeanReduce. The user's brief listed:
```
K1 Pow → K2 Mean → K3 +eps → K4 rsqrt → K5 x*rsqrt → K6 *weight
```
Anchoring on `MeanOps` (= K2 in the brief), my walk goes K3 (+eps), K4 (rsqrt), K5 (x*rsqrt), and stops. **K6 (*weight) is not captured.** The recipe stores 5 funcptrs: anchor + 4 follow-ups.

The actual *weight step would be the next launch after my matcher's MATCHED state. That's the kernel whose args[2] should contain `std::array<char*, 3> = [Y, normed_x, W]`.

---

## What needs to change

1. **Extend the matcher state machine** from 5 steps to 6: SAW_MEAN → SAW_ADDEPS → SAW_RSQRT → SAW_MUL → SAW_NORMED → MATCHED. The new SAW_NORMED → MATCHED transition expects a kernel whose name contains `BinaryFunctor` or similar, AND whose args[2] has 3 cuda pointers.

2. **Capture K0 (Pow) from the recorder ring.** The ring already records every launch; at MeanOps anchor time, the launch one back is K0. Reading its args[2] would give `[A_intermediate, X]` → X is the second pointer.

3. **Once K6 is in the matcher and X is captured from K0, the recipe stores byte offsets**:
   - `x_at_arg2_offset_8` (within K0's args[2])
   - `w_at_arg2_offset_16` (within K6's args[2])
   - `y_at_arg2_offset_0` (within K6's args[2])
4. **Stage 3 substitute reads X/W/Y** at those offsets on every match, calls `cipher_fused_rmsnorm`, suppresses all 7 kernels.

---

## What this tells us about the brief's premise

The user wrote: *"args[2] at the Scale step contains exactly 3 device pointers"* — that's right for the BinaryFunctor<Mul> Scale kernel, but **my matcher wasn't capturing the Scale kernel as K5**. The 5-step walk from MeanOps stops at the multiply-by-rsqrt step.

Extending to a 6-step walk + raw-dumping K6 (the next kernel) is the right next step. That's a small matcher edit (~30 lines), one more dump, and confirmation that K6's args[2] has 3 cuda ptrs that include the **stable-across-windows** weight pointer.

If the K6 args[2] has the 3-ptr layout the brief predicted, then real substitution becomes wirable: extract X/W/Y at known offsets, call `cipher_fused_rmsnorm`, suppress the 7 kernels.

If K6 ALSO has only 2 pointers (i.e., PyTorch fuses Scale in-place too), then we'll need K7 — keep walking until a 3-pointer kernel appears. The dump tells us which kernel actually has the [Y, normed_x, W] triple.

---

## Files updated

- `src/cipher_flow_substitute.cpp` — added `scan_raw_dump_step()` (no filtering, full hex + uint64 interpretation), wired into K2/K4/K5 of the first 5 windows, bumped timeout to 50ms during scan probe (fprintf is slow enough to trip the 100us safety in normal mode)
- `CIPHER_FLOW_RAW_DUMP.md` — this report

## Reproducibility

```bash
CIPHER_FLOW_MATCH=on CIPHER_FLOW_SUBSTITUTE=on CIPHER_FLOW_SCAN=on \
    ./cipher_run.sh -c "<your inference>" 2>raw.log
grep -A 12 "K5-Scale" raw.log
```

## Next concrete step (small)

Edit `cipher_flow_patterns.cpp` to extend state machine to 6 follow-up steps; then re-run the raw dump on the new K6 (the kernel after my current K5). If K6's args[2] has 3 cuda ptrs and the third one is **stable across all windows**, that's the RMSNorm weight. Then wire real substitution.

Estimated 30 minutes of editing + verification.
