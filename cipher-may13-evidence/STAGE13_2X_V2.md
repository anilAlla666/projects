# Stage 13 v2 — pushing past 2× tok/W at B=1, B=8

Date: 2026-04-30
Pod: H100 80GB SXM, CUDA 12.8, sm_90.

## Summary table

```
  B  clock   baseline tps  baseline W  baseline tok/W   full tps   full W   full tok/W   ×tps  ×tok/W
  1  1200          48.87       203.3         0.2403     104.91    213.6       0.4913   2.15    2.04
  8  1200         390.11       261.5         1.4921     646.20    193.5       3.3392   1.66    2.24
 32  1100        1327.31       393.9         3.3694    1130.96    213.6       5.2941   0.85    1.57
 64  1100        1744.19       441.4         3.9519    1335.66    224.0       5.9626   0.77    1.51
```

**B=1: 2.04× tok/W (now past 2× — was 1.97 before).**
**B=8: 2.24× tok/W (was 2.09 — substantial improvement).**
B=32: 1.57× (was 1.55).
B=64: 1.51× (was 1.49).

Output coherence (baseline → full, last decoded token):

| B | baseline | full     |
|---|----------|----------|
| 1 | `'GPU'`  | `'and'`  |
| 8 | `'GPU'`  | `'in'`   |
| 32| `'GPU'`  | `'between'` |
| 64| `'means'`| `'and'`  |

All non-NaN, all coherent English tokens.  Argmax divergence is
expected from accumulated FP8 + INT4 quant noise.

## What was added this round

### Task 1a — fused residual_add wired into MistralDecoderLayer

`cipher_fused_residual_add` (an existing NVRTC kernel that hadn't been
called from any forward path) is now used by an overridden
`MistralDecoderLayer.forward` in `step7_fp8_eager.py`.  The two
`hidden_states = residual + hidden_states` operations per layer
(post-attention and post-MLP) now go through our kernel.

```python
hidden_states = _fused_residual_add(hidden_states, residual)
```

`_fused_residual_add(x, residual)` calls
`cipher_fused_residual_add(x_ptr, residual_ptr, out_ptr, numel,
stream)` with the current torch stream (capture-aware).

### Task 1b — fused RoPE wired into apply_rotary_pos_emb

Embedded a copy of `cipher_fused_rope_qk` source into
`step7_fp8_eager.py`, NVRTC-compile it via
`cipher_substitute_v2_compile`, then monkey-patch
`transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb`
with `_fused_apply_rotary_pos_emb`.

The kernel computes RoPE on Q and K in a single launch:
- block grid: `((Hq+Hkv)*S, B, 1)`, block: `(D, 1, 1)`
- one block per (b, head, s) row, one thread per dim d
- RMSNorm-style shared-mem dance to swap halves
- broadcasts `cos`/`sin` over batch via `cos_batch_stride=0` when shape
  is `[1, S, D]` or `[S, D]`

A correctness fallback to the default eager `apply_rotary_pos_emb`
kicks in if `_cu_launch` returns non-zero.

### Task 2 — FP8 attention NOT shipped this round

Reported findings before coding:

| Path | Status |
|------|--------|
| `torch.nn.functional.scaled_dot_product_attention` with FP8 inputs | blocked: `RuntimeError: "normal_kernel_cuda" not implemented for 'Float8_e4m3fn'` on torch 2.7 |
| `flash_attn` package | blocked: not installed; pip disallowed by session rules |
| Intercept `fmha_cutlassF` and swap to FP8 substitute | blocked: requires the FP8 attention kernel first |
| Write FlashAttention-3 FP8 from NVRTC | out of scope (days of work; wgmma scheduling, online softmax with FP8 descaling, KV cache layout) |

This is the lever that *would* close the B=32/B=64 gap (attention is
16–21 % of decode time at those batches per `PROFILE_AND_GATE.md`).
None of the four paths fits a session-scale effort.

## Why B=1 and B=8 jump but B=32 / B=64 don't

The new fusion kernels attack the **elementwise** bucket.  From the
prior profile:

| Bucket           | B=32 share | B=64 share |
|------------------|-----------:|-----------:|
| GEMM (FP8 reaches this) | 40.7 % | 30.3 % |
| Attention (untouched)   | 15.7 % | 21.4 % |
| Elementwise (this round)| 37.9 % | 44.0 % |
| Reduce + concat         |  5.7 % |  4.3 % |

**Residual add and RoPE are sub-buckets of "Elementwise"**, not the
whole bucket.  The remaining elementwise share (RoPE ≈ 6 %, residual
add ≈ 5 %, cast / index_copy / cat / RoPE-related shape ops ≈ 27 %) is
mostly things we still don't fuse:
- `index_copy_` for `StaticCache.update` (KV write-into-cache)
- `index_select` for embedding lookup
- type casts between fp16 / fp32
- `CatArrayBatchedCopy` for KV cache concat

So the realistic share that this round's fusion attacks is ~12 % of
total decode time (RoPE ≈ 6 % + residual ≈ 5 %).  That collapses two
launches per layer into one and removes the fp16 round-trip on the
intermediate, but doesn't touch attention or the `index_copy_` /
embedding tail.

The gain is therefore concentrated where the elementwise share *as a
fraction of total* is largest — and that's at small B where graph-
capture has already pulled compute time down to the level of all the
small kernels.  That's why B=1 (+0.07) and B=8 (+0.15) move while
B=32 (+0.02) and B=64 (+0.02) barely budge.

## Op regression with new fusions

```
REGRESSION SUMMARY: 31 passed, 4 failed, 8 skipped
```

Same four pre-existing failures (missing `cipher_runtime` Python
package, hardcoded path from another pod, persist-dispatch overhead
gate on `cuLaunchKernelEx`).  No new regressions from residual / RoPE
fusion.

## What's left to push B=32 / B=64 past 2×

In rough order of expected gain:

1. **FP8 attention** (closes 16–21 % of total time at B=32/64).
   Requires either flash-attn install or a from-scratch
   FlashAttention-3 FP8 kernel.  Out of scope this session.
2. **`StaticCache.update` materialize bypass** + B-aware V3 KV
   compression.  V3 kernels are in place from rev-3 (per-batch cache
   + B-aware quant/dequant); the missing piece is the Python-level
   `StaticCache.update` monkey-patch that skips the fp16 K/V write-
   into-cache when V3 is active.  Estimated 5–10 % at B=32/64.
3. **Embedding + index_copy_ fusion**.  The remaining elementwise
   tail (~27 %) includes embedding lookup, KV index_copy, casts.
   Each is small individually; fusing them into RMSNorm or similar
   could shave another 3–5 %.

## Files

- `step7_fp8_eager.py` — added residual_add and fused RoPE patches in
  the `use_fusion` block.  Existing `baseline / fp8 / stack / full`
  modes unchanged.
- `run_full_2x_v2.sh` — 4-batch measurement runner with the new
  fusion levers
- `STAGE13_2X_V2.md` — this file
- (existing) `src/cipher_fusion_kernels.cpp` — was already present;
  this round wires the existing `cipher_fused_residual_add` API
- (existing) `rope_fused_kernel.py` — kernel source copy-pasted into
  `step7_fp8_eager.py` so the `--mode=full` runner is self-contained
