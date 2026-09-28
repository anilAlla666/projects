# Session Summary — 2026-04-28 (Phase 2 + Phase 3)

This session began with Phase 2 already partially shipped: the Marlin INT4
GEMM was compiling via NVRTC, all 224 Mistral linears were repacked, and
correctness tests at all four Mistral shapes were green. The end-to-end
Mistral-7B sweep was reporting impressive numbers — batch=8 at 2102 tok/s
(2.47×), batch=64 at 8747 tok/s (1.84×). Three things still needed to be
done: fix the K=14336 down_proj layers (32 of 224 still falling back to
fp16), ship CUDA-Graph-safe Marlin, and replace the broken power numbers
(0 W readings at low batch).

What we discovered along the way is more important than what we set out
to fix.

---

## Phase 1 — Pre-existing infrastructure (verified, not changed)

- `cuLibraryLoadData` / `cuLibraryGetKernel` / `cuKernelGetFunction` LD_PRELOAD
  intercepts in `src/cipher_intercept_cudart.cpp` resolve cuBLAS-LT and CUTLASS
  kernels at module-load time; per-CUfunction parameter layouts come from
  `cuKernelGetParamInfo` on demand.
- PyTorch's runtime-API kernels (no driver-level name) covered via `dladdr`
  fallback against `libtorch_cuda.so`'s symbol table. 69 % named coverage on
  Mistral 32-token decode (up from 42 %).
- Stage 6 NVRTC pipeline (`cipher_substitute_v2`) compiles arbitrary CUDA
  source at runtime, indexes the resulting cubin in a 64-slot pool. This is
  what Marlin and the fused kernels ride on.

## Phase 2 — Marlin INT4 GEMM, fully wired

### K=14336 down_proj (the 32 missing layers)

`cipher_weight_compress_quantize()` short-circuited if the cached
`g_quant_out[idx].int4_buf` was non-null. PyTorch's caching allocator
reuses device pointers across freed tensors of different shapes
(`gate_proj` K=4096 N=14336 freed → `down_proj` K=14336 N=4096 allocated
at the same address). CIPHER returned the *previous* shape's
int4 / scales / Marlin layout for the new tensor. The rel_err of 1.4 at
K=14336 was the kernel running with the right INT4 inputs but the wrong
weight matrix.

Fix: shape-check the cached entry; on mismatch, `cudaFree` the stale
buffers and re-quantize (one-line condition guarding the early-return).
All 15 Mistral shapes now hit rel_err 0.115–0.118 (INT4 noise floor).

### Graph-capture safety in `cipher_weight_compress_marlin_gemm`

Two operations were illegal during CUDA stream capture:
1. **`cudaMalloc` of the workspace** — fired on the first call to a new N.
   Fixed: workspace pre-reserved at worst-case (16 KB, covers N up to 32 K)
   on first call only; subsequent calls reuse.
2. **`cudaMemset` (synchronous)** of the workspace before each launch.
   Fixed: `cudaMemsetAsync(d_ws, 0, ws_bytes, stream)` on the launch stream.

After this fix the full sweep ran without device-side asserts, and the
"Phase 2 v3" numbers landed: 4.55× at B=1, 4.17× at B=8, falling to 1.11×
at B=256.

### Power measurement
Graph-replay of a 32-token burst is too short (60–200 ms) for `nvidia-smi`
to sample (~150 ms cadence). Tried several extension paths:
- **Burst-reset (cache_pos.fill_ between captured replays)** — fails. Triggers
  a PyTorch `index_copy_` OOB; reproduced in plain Mistral with no CIPHER.
  Verified in `/tmp/test_cache_reset.py` — T1 (one burst) passes, T2 (reset
  then second burst) device-side asserts.
- **Larger StaticCache so the timed loop fits** — fails. With
  `MAX_LEN=6018`, attention scales linearly: SDPA matmuls full-cache K^T.
  Throughput drops 5× at low batch.
- **Eager-mode top-up loop after the timed graph replay** — works. Eager
  mode rebuilds the attention mask each forward, so cache_pos can wrap
  freely. tps is taken from the captured 32-token burst; power is the
  mean over a ≥10 s eager-mode top-up running the same batch.

This is a slight methodology compromise — graph replay throughput at the
power level eager-mode is drawing — but it correlates well: the dominant
kernels (Marlin / cuBLAS / FA / fused norm) are the same in both regimes,
so power profiles match within ~5 W.

## Phase 3 — Graph audit (the embarrassing surprise)

The Phase 2 numbers above (4.17× at B=8) felt too good. Marlin at M=8
should give roughly 1.5–2× over fp16 cuBLAS on the bandwidth axis,
RMSNorm fusion buys some single-digit percent of total time, and SiLU
fusion is below the noise floor. 4× total didn't line up.

Built `src/cipher_graph_inspect.cpp` to intercept `cudaGraphInstantiate*`
and walk the captured graph via `cuGraphGetNodes` →
`cuGraphKernelNodeGetParams` → `cuFuncGetName`, histogramming nodes by
name. First run at BATCH=8:

```
[CIPHER GRAPH-INSPECT #0] total=828  kernels=826  ...
   — top entries are RoPE muls/adds, FillFunctor, attention —
[CIPHER GRAPH-INSPECT #0] Marlin nodes=0  cuBLAS/cutlass GEMM nodes=33
```

**Zero Marlin nodes in the captured graph.** Also zero fp16 cuBLAS GEMM
nodes — only 1 cuBLAS-LT kernel and 32 attention kernels (the cutlass
attention kernel match was inflating the "cuBLAS/cutlass" counter). With
0 GEMM nodes, the "graph replay" was running the elementwise / index
operations *around* the linears but skipping the linears entirely.

Built a one-call repro (`/tmp/diag_capture_marlin.py`):

```
eager  rc=1  rel_err=0.1171
capture rc list = [1]
replay rel_err=1.0000  (after C.zero_ + g.replay)
```

Marlin returned `rc=1` from inside the capture context — the launch did
not error — but on replay (after zeroing the output buffer), rel_err
was 1.0. The Marlin output buffer kept its capture-time value; the
captured graph never replayed Marlin.

### Root cause
`cipher_weight_compress_marlin_gemm` was launched on `stream=NULL`
(legacy default stream). PyTorch's `torch.cuda.graph` captures on a
*different* stream — its CUDAGraph stream. Launches on the legacy
default stream during capture proceed eagerly and are NOT recorded.
Marlin executed once during capture, the output buffer got correct
data once, and the captured graph contained zero Marlin nodes. On
replay the graph re-ran every elementwise op around the linears, read
stale buffer outputs from "where the linear used to write," and
produced bit-identical output every replay. Argmax of stale logits
returned the same predicted token forever; tps measured no-op replay.

### Fix
One line per CIPHER call site:
```
ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
```
in place of `None`, repeated for `marlin_gemm`, `int4_gemv`,
`fused_rmsnorm`, `fused_silu_mul`. After the fix the BATCH=8 captured
graph contains 224 × `marlin_M1_N8_K8_G8` (32 layers × 7 linears),
65 × `cipher_rmsnorm_fp16`, 32 × `cipher_silu_mul_fp16`, and 32
MemEffAttention. Total node count went from 828 to 1373.

### Honest numbers

| batch | tok/s baseline | tok/s Phase 2 | × tok/s | × tok/W |
|---|---|---|---|---|
|   1   |  115.6 | 119.4 | 1.03× | 0.80× |
|   8   |  852.0 |1177.5 | **1.38×** | **1.79×** |
|  32   | 2805.4 |2857.6 | 1.02× | 1.24× |
|  64   | 4728.6 |5409.8 | 1.14× | 1.12× |
| 128   | 7166.0 |7918.1 | 1.10× | 0.95× |
| 256   | 9580.2 |10467.0| 1.09× | 1.00× |

The bench previously reported batch=8 at 3553 tok/s (4.17×). The honest
number is 1177 tok/s (1.38×). **The earlier 4× was a measurement
artifact of broken graph capture, not real speedup.**

### Marlin gating

At BATCH=64 (M=64), Marlin's compiled `m_blocks=4` template was
empirically slower than fp16 cuBLAS — running Marlin at M=64 gave 0.88×
vs baseline. Gated Marlin to M ≤ 32 in the linear; M=33..64 hits cuBLAS
in the captured graph and benefits only from fused norm / SiLU. That
gives 1.14× at B=64 (real, modest, honest).

## Phase 3.5 — Per-section profiling, the Amdahl ceiling

Where do the kernel-level wins (Marlin ≥2×, RMSNorm 15.5×, SiLU 3.4× in
microbench) actually surface end-to-end? Profiled with two tools:

### Eager-mode breakdown (proportions, BATCH=8)
| section | baseline % | Phase 2 % |
|---|---|---|
| **attn_net** (Q@K^T + softmax + AV + KV update) | **39.7 %** | 41.1 % |
| linears (qkv + o + gate_up + down) | 33.1 % | 33.9 % |
| rmsnorm × 65 | 14.9 % | 6.0 % |
| silu_mul × 32 | 2.7 % | 5.6 % |
| OTHER (rope, embed, residuals, mask, lm_head) | 9.7 % | 13.4 % |

### Graph-mode ablation (BATCH=8, per-iter µs)
| config | per-iter | × |
|---|---|---|
| baseline (CIPHER OFF) | 9360 | 1.000× |
| INT4 only             | 7956 | 1.176× |
| RMSNorm only          | 8227 | 1.138× |
| SiLU_mul only         | 9275 | 1.009× |
| ALL ON                | 6746 | **1.388×** |
| sum of singles        | (additive: 1404 + 1133 + 85 = 2622 µs ≈ 2614 µs ALL ON) |

Savings are perfectly additive. No hidden sync points / stream waits
between CIPHER kernels and PyTorch.

### Why kernel-level claims discount
| component | microbench × | end-to-end Δ at B=8 | of microbench |
|---|---|---|---|
| Marlin INT4 (M=8) | ≥2× | -17.6 % (1.18× linear-equiv) | ~80 % delivered |
| RMSNorm fused | 15.5× | -13.8 % (5.3× rmsnorm-equiv) | ~30 % delivered |
| SiLU_mul fused | 3.4× | -0.9 % (1.4× silu-equiv) | ~17 % delivered |

Microbench savings discount because graph capture **already eliminates
launch overhead**, which is the bulk of what fused kernels save in
eager mode. RMSNorm fusion was 15.5× because eager-mode PyTorch fires
many small kernels per norm; graph mode replays those for ~free, so
fused-RMSNorm's win shrinks to its compute / bandwidth advantage only.
INT4 holds up best because its win is HBM-bandwidth-bound (4× less
weight data per GEMM), not launch-bound.

### The Amdahl ceiling
CIPHER touches ~48 % of forward time (linears 33 % + RMSNorm 15 %).
At BATCH=8 we hit 1.388× = 72 % of the theoretical 1.92× ceiling that
infinitely fast INT4 + RMSNorm would give. **Attention is 40 % and
unchanged** — that's the unconquered hill.

## Lessons

1. **Verify capture, don't assume.** "It runs and the answers look right"
   is not the same as "the kernel is in the graph." A graph that's
   missing 224 layer matmuls but still produces a stable token stream
   *will* report a tps number — just not a meaningful one. The graph
   inspector is now permanent infrastructure.
2. **NULL stream is not the capture stream.** PyTorch `torch.cuda.graph`
   captures on its CUDAGraph stream, not the legacy default. Any C
   shim called from inside `with torch.cuda.graph(g):` must take and
   use the current stream. Resolved via `dlsym` from libcuda gives
   you the real `cuLaunchKernel`, but you still have to pass the right
   stream pointer to it.
3. **Microbench wins discount in graph mode.** A fused kernel that's
   15× faster in isolation may be ~5× faster once captured, because
   the capture already pays the launch cost only once at instantiate.
   Real model gains correlate with **bandwidth savings**, not launch
   savings.
4. **`cache_pos.fill_(...)` between captured replays is a PyTorch bug**
   (or at least a sharp edge). The captured `index_copy_` reads the
   buffer at replay time; resetting to a value lower than the captured
   value re-triggers an OOB check that doesn't fail in eager mode.
   Workaround: do power sampling in eager mode where the mask is
   rebuilt every forward.
5. **The K=14336 "Marlin bug" was actually a quant-cache bug.** Spent
   hours diffing Marlin's reference repack against ours, byte for byte,
   matching exactly — and the kernel still produced garbage. The
   Marlin code was always correct; the bug was upstream in the
   quantize-cache short-circuit returning the previous shape's
   buffers. **When a kernel fails for ONE shape but works for others,
   suspect the dispatcher / cache / lookup layer, not the kernel.**

## Files touched this session

### New
- `src/cipher_graph_inspect.cpp` (11 KB) — graph introspection LD_PRELOAD layer

### Modified
- `src/cipher_weight_compress.cpp` — graph-capture-safe Marlin workspace +
  quant-cache shape-invalidation
- `Makefile` — wire cipher_graph_inspect into HOOK build
- `exports.map` — 4 new symbols
- `CLAUDE.md` — Phase 2 final, Phase 3, Phase 3.5 sections
- `BUILD_STATE.md` — session inventory and build instructions

### Tests / diagnostics (in /tmp; not part of the build)
- `bench_phase2_table.py` — full sweep
- `diag_capture_marlin.py` — proves the stream-capture bug
- `diag_mistral_marlin.py` — eager-mode 224-layer correctness
- `test_cache_reset.py` — proves PyTorch `index_copy_` OOB
- `profile_b8.py` — per-section eager profiler
- `ablate_graph_b8.py` — graph-mode component ablation
- `phase2_table_v3.txt` — published table

## Where to go next

If pushing past 1.4× at B=8 is the goal, the lever is attention:
1. **KV redirect (Stage 8b/c)**, properly post-RoPE. The attention block
   is 40 % of total; reducing per-step KV bandwidth pays back on the
   dominant kernel.
2. **RoPE fusion into Q/K projections**. Eliminates 128 small kernel
   launches per layer, but in graph mode this saves only ~300–500 µs
   total — not the main lever.
3. **Marlin parallel-M kernel**. Compile the `m_blocks > 4` branch so
   batches 128 / 256 (M > 64) can use INT4 weights at all. Today they
   fall back to fp16 cuBLAS and only benefit from fused norm/SiLU.
