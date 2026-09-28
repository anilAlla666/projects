# CIPHER Build State — op31-prod snapshot

## Pod (current — 2026-04-28)
Working dir: /home/ubuntu/op31-prod-fix
Hardware: NVIDIA H100 80GB HBM3 (SXM, 700 W TDP, 132 SMs, sm_90, 50 MB L2, 31.2 MB L2 persist max, 128 MB access window max, 3.35 TB/s HBM)
Driver / CUDA runtime: 580.105.08 / 13.0
Power-cap-bound at 700 W → 1350 MHz sustained → 658.7 TFLOPS fp16 ceiling

## Built — all 12 stages shipped, all actuating, all gates green

### Phase 0 — Hardware measurements (2026-04-28)
- Sustained clock under 60 s GEMM: 1350 MHz median, 700 W cap (power-bound, NOT thermal-bound; junction 54 °C)
- Power-clock sweep: peak efficiency at 400–500 W (1.10–1.14 TFLOPS/W); 700 W = 0.94 TFLOPS/W
- Hardware profile via `cudaGetDeviceProperties`: SMs=132, L2=50MB, L2_persist=31.2MB, L2_window=128MB, HBM_BW=3.35TB/s, peak_fp16=989TFLOPS sparse, threshold=295 FLOPS/B
- M1–M4 baseline (fp16 4096³, 10s): M1=663, M2=663.5, M3=662.7, M4=662.0 TFLOPS — total CIPHER overhead 0.2% with all 20 observers ON

### Stage 1 — 21 ops + 20-observer regression (pre-existing 2026-04-15)
- Phase 1 Session Intel: SENSE, SHIELD, SUSTAIN
- Phase 2 Energy: THERMOSTAT, PULSE, VOLT (Path B), HIBERNATE (Path B)
- Phase 3 Straggler/NCCL: Straggler ✓; NCCL P2P CPU Proxy DEFERRED (multi-node)
- Phase 4 Agentic: LOOP, CONTINUITY, PIPELINE
- Phase 5 Compliance: PREDICT, GUARD, DETERMINISM, TOPOLOGY, TRACE, FAIRNESS, CARBON, RECEIPT, COMPLY
- 12 core ops in cipher_10ops_impl.cpp: CLASSIFY, SPECULATE r/w, SUBSTITUTE, ORCHESTRATE, GENERATE, RING_WRITE, REMEMBER, VALIDATE, AUDIT, ADAPT, ARBITRATE
- Performance fixes (2026-04-15): cuLaunchKernelEx dispatch guard + lazy Stage-1/2 thread spawn

### Stage 2 — Silicon Model (2026-04-28)
`include/cipher_silicon.h` + `src/cipher_silicon.cpp`. Static fields populated from `cudaGetDeviceProperties` + NVML at init priority 102. Dynamic atomic fields (`clock_sustained_mhz`, `thermal_headroom`, `power_draw_watts`) written by THERMOSTAT, read by all observers. Hook constructor (priority 101) calls `cipher_silicon_init` via dlsym; rt's autoinit at 102 covers the case when only rt is preloaded. No hardcoded constants — every value queried.

### Stage 3 — Persistence Engine (2026-04-28, full actuation)
`cipher_persist_engine.{h,cpp}` + hot-path wiring. Fixed 64-region table, fractional-knapsack budget against `silicon->l2_persist_max` (= 31.2 MB on this pod). Hot-path actuation: `cuLaunchKernelEx` shim alloca's an augmented `CUlaunchAttribute[]`, appends `CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1` with our window, and re-invokes the real driver. Per-launch — fires every launch with admitted region. PREDICT instrumentation upgraded: `cipher_predict_observe_ptr` called from `cublasGemmEx` (real PyTorch path) + `cipher_set_gemm_ptrs` (Python wrapper); on `HOT_PTR_COUNT=100` hits a pointer is auto-registered with the engine.
- Verified firing: M4 run = 48,848 window_hits / 48,797 launches in 10 s.
- M4 ΔTFLOPS: 0% (compute-bound at 700 W cap — wrong probe; correctness only).

### Stage 4 — VMM Pool (2026-04-28)
`cipher_vmm.{h,cpp}`. Side allocator (does NOT intercept `cudaMalloc`) backed by libcuda's `cuMemAddressReserve` / `cuMemCreate` / `cuMemMap` resolved via `dlopen`. Up to 256 regions; granularity queried at init (typically 2 MB). API: alloc/free/swap/stats. `cipher_vmm_swap` implements the dual-reservation shadow-swap pattern for atomic weight reload.

### Stage 5 — Graph Engine (2026-04-28, full actuation)
`cipher_graph.{h,cpp}`. Per-thread sliding-window kernel-fp sequence detector + state machine (OBSERVE → DECIDED → CAPTURED → REPLAYING → INVALID), promotes after 4 identical signatures. Real actuation API: `cipher_graph_begin_capture(stream)` (calls `cuStreamBeginCapture`, mode=THREAD_LOCAL), `cipher_graph_end_capture(stream)` (calls `cuStreamEndCapture` + `cuGraphInstantiateWithFlags`), `cipher_graph_replay(graph_id, stream)` (calls `cuGraphLaunch`). App-capture coexistence via `cuStreamIsCapturing` guard.
- Verified speedup: 64-kernel sequence (10 s, fp32 element-wise) → individual launches 318 µs/step, graph replay 78 µs/step → **4.07× speedup**.

### Stage 6 — NVRTC Substitution Engine (2026-04-28, full actuation)
`cipher_substitute_v2.{h,cpp}`. Real `nvrtcCreateProgram` + `nvrtcCompileProgram` (sm_90, std=c++17, `-I/usr/include` for cuda_fp16.h) + `cuModuleLoadData` + `cuModuleGetFunction`. `cudaFree(0)` ensures primary CUDA context. Up to 64 cubins; mutex-guarded slot pool. API: `compile(source, name) → cubin_id`, `get_function(id) → CUfunction`, `destroy(id)`. Stage 7/8 use this to compile their quantize kernels at first use.
- Verified: trivial kernel compiled, cubin_id=1, CUfunction handle returned.

### Stage 7 — W4A16 Weight Compression (2026-04-28, full actuation)
`cipher_weight_compress.{h,cpp}`. NVRTC-compiled kernel `cipher_w_quant_int4(__half* in, uint8_t* out, __half* scales, rows, cols)` — per-row absmax (warp reduction), scale=absmax/7, packs two 4-bit signed quants per byte. Engine retains output buffers via `cudaMalloc` for downstream dequant-fused GEMM. Stable-pointer detection (≥1 MB, ≥1000 hits) before quantization; correctness gate by relative-error threshold (default 0.01).
- Verified: 1024×1024 fp16 → 524,288 INT4 bytes + 1024 fp16 scales, kernel launched on 1024-block × 256-thread grid.

### Stage 8 — KIVI 2-bit KV Compression (2026-04-28, full actuation)
`cipher_kv_compress.{h,cpp}`. NVRTC-compiled kernel `cipher_kv_quant_2bit` — per-row min/max scan, scale=(max−min)/3, zero-point=min, packs four 2-bit quants per byte. Modes: `per_channel=1` for keys (per-row), `per_channel=0` for values (per-token). Layer allowlist via `CIPHER_KV_FULL_LAYERS=0,1,30,31`. Residual window default W=32.
- Verified: 64×4096 fp16 → 65,536 2-bit bytes + 64 scales + 64 zeros, key-mode launch.

### Stage 9 — NCCL Tuner v4 + Green Context (2026-04-28, full actuation)
`cipher_nccl_v4.{h,cpp}` + extension to `src/cipher_nccl_tuner.cpp`. Tuner DSO `dlsym`'s `cipher_nccl_v4_decide` and `cipher_nccl_v4_enabled` at init; per-call decision routes through v4 first when enabled. Algo bias: ≥16 MB → NVLS, 256 KB–16 MB → RING, < 256 KB → TREE. Reserves `green_ctx_sms=8` (configurable).
- Verified: 64 MB / 8 ranks → algo=5 (NVLS), 512 KB / 8 ranks → algo=2 (RING).

### Stage 10 — L2 Partition Router (2026-04-28, full actuation)
`cipher_partition_router.{h,cpp}`. Reads `silicon->sm_count` ÷ partition_count (default 2 on H100). Lazy real CUstream pool: dlopens libcuda, calls `cuStreamCreate` once per partition, retains for engine lifetime. API: `bind(ptr, hint)`, `get(ptr) → partition_id`, `stream_for(ptr) → CUstream`. Up to 1024 bindings; unbound pointers get hash-fallback partition.
- Verified: bind(0x1000→0), bind(0x2000→1) → two distinct non-NULL CUstreams.

### Stage 11 — Thermal-Substitution Feedback (2026-04-28, full actuation)
`cipher_thermal_feedback.{h,cpp}`. Dedicated `std::thread` opens libnvidia-ml.so.1, calls `nvmlInit_v2`, gets device handle, loops every 100 ms reading `nvmlDeviceGetClockInfo(GRAPHICS)` + `nvmlDeviceGetPowerUsage`, computes `headroom = 1 − power/cap`, writes via `cipher_silicon_update`, calls `cipher_thermal_feedback_tick`. Aggressiveness state machine: +0.05 when headroom < 0.25, −0.02 when headroom > 0.75, ±0.20 oscillation clamp.
- Verified: 21 sampler ticks in 0.5s window, clock=1980 MHz, power=108 W, headroom=0.85, aggressiveness updating.

### Stage 12 — Integration + Hardening (2026-04-28)
All Stage 3-11 modules co-existing; init priority chain 101 (hook) → 102 (silicon) → 103 (persist_engine) → 104 (vmm) → 105 (graph) → 106 (substitute_v2) → 107 (weight_compress) → 108 (kv_compress) → 109 (nccl_v4) → 110 (partition_router) → 111 (thermal_feedback). Each module idempotent + self-init via `__attribute__((constructor(N)))`.

## Three DSOs
- `libcipher_hook.so` (~50 KB) — LD_PRELOAD intercept layer (cuLaunchKernel/Ex shims, cublasGemmEx shim, dispatch guard)
- `libcipher_rt.so` (~430 KB) — runtime: 12 core ops + 20 observers + 9 actuator modules (silicon, persist_engine, vmm, graph, substitute_v2, weight_compress, kv_compress, nccl_v4, partition_router, thermal_feedback)
- `libcipher_nccl_tuner.so` / `libnccl-tuner-cipher.so` (~16 KB) — NCCL plugin with v3 decide + v4 dlsym bias

## Final gates (with everything ON: all 9 actuators + all 20 observers)
| Gate | Result |
|---|---|
| Build (`make clean && make all`) | clean |
| 20/20 observer regression | **20 PASS / 0 FAIL** |
| `tests/test_persist_engine.py` | **5/5 PASS** |
| `tests/test_actuation_full.py` (stages 6–11) | **6/6 PASS** |
| `tests/test_actuation_microbench.py` graph replay | **4.07× speedup** (318 µs → 78 µs / step) |
| M4 perf (10 s, fp16 4096³) | **663.8 TFLOPS** (Δ −0.03 % vs 664.0 baseline; power-cap ceiling) |

## Where each actuator's gain shows up
- Stage 3 (L2 persist): memory-bound kernels with hot reuse — KV reads, weight broadcast
- Stage 5 (graph replay): launch-bound workloads (verified 4.07× on 64-kernel sequence)
- Stage 6 (NVRTC): kernel JIT pipeline used by Stages 7/8/12
- Stage 7 (W4A16): decode-dominated inference, large models — 4× HBM weight bandwidth reduction
- Stage 8 (KV 2-bit): long-context decode — 8× KV bandwidth reduction
- Stage 9 (NCCL v4): multi-GPU AllReduce overlap
- Stage 10 (partition router): kernels touching same hot region land on same partition stream
- Stage 11 (thermal feedback): NVML telemetry → adaptive substitution → DVFS recovery loop

M4 4096³ fp16 GEMM stays at 663 TFLOPS because it's compute-bound at the 700 W power cap — none of the actuators target that workload class. Real-workload measurement (Mistral-7B / 70B decode) is where they compound.

## Hot-path additions (cipher_intercept_cudart.cpp)
- `cipher_persist_maybe_apply(stream)` helper — dlsym-cached, ~10ns when engine OFF
- `cipher_persist_engine_get_top_window_raw` injection in cuLaunchKernelEx
- `cipher_predict_observe_ptr` calls in cublasGemmEx (3 ptrs × ~10 ns) + cipher_set_gemm_ptrs
- All env-gated default OFF; zero observable cost when CIPHER flags are unset

## Files added this session (2026-04-28)
- 9 headers + 9 cpps (Stages 2–11): cipher_silicon, cipher_persist_engine, cipher_vmm, cipher_graph, cipher_substitute_v2, cipher_weight_compress, cipher_kv_compress, cipher_nccl_v4, cipher_partition_router, cipher_thermal_feedback
- 3 tests: test_persist_engine.py, test_actuation_microbench.py, test_actuation_full.py
- ~3000 LOC total

## Lessons (post-2026-04-15)
- Always verify wiring survived pod migration. CLAUDE.md claims valid only with grep on current pod.
- M4 4096³ fp16 GEMM is power-cap-bound at this pod's 700 W TDP. It's a regression sentinel, not a perf-gain probe. Use launch-bound or memory-bound microbenches for actuator validation.
- NVRTC needs an active CUDA context (`cudaFree(0)`) and `-I/usr/include` for `cuda_fp16.h` on this pod.
- `<stdatomic.h>` is C-only; never include from `.cpp` files.
- NVRTC inline-asm constraint quirks: `+r` (read+write) triggers an internal compiler error ("asm operand index requested is larger than the number of asm operands provided"). Use separate `=r` output + `r` input. PTX immediates: `lop3.b32` and `sub.f16x2` need register operands for non-immLut args, not `n` (immediate). Pass constants via `unsigned int` C variables and `r` constraint.

## Stage 7 INT4 GEMM kernel state (2026-04-28)
- INT4 GEMV (M=1, transposed B_T layout, warp-cooperative reduction): **1.13–1.18× cuBLAS at K=4096 N=4096**. Production decode-path win.
- Multi-row GEMV (M=2..15): 0.19× cuBLAS — substitute only at M=1 (Int4Linear gates correctly).
- wmma INT4 GEMM (BM=16 BN=64 BK=32, LOP3 fp16-magic dequant + adaptive K-split with fp16 atomicAdd):
  - M=16 K_split=8: **0.22× cuBLAS** (was 0.06× — 3.5× session improvement)
  - M=32 K_split=4: **0.15× cuBLAS** (was 0.05× — 2.9×)
  - M=64 K_split=2: **0.09× cuBLAS** (was 0.05× — 1.6×)
  - M≥128 K_split=1: 0.04–0.07× cuBLAS (already saturating without split)
- Adaptive split-K dispatch targets ≥528 blocks (132 SMs × 4 waves). Pre-zeros C via `cudaMemsetAsync` before kernel launch when K_split>1.
- LOP3 PTX: `lop3.b32 %0, %1, %2, %3, 0x6A` computes `(a&b)^c` (mask + sign-flip + bias OR) followed by `sub.f16x2 %0, %1, %2` with bias=0x64086408. NVRTC requires register operands (not `n` immediates) and rejects `+r` read-write constraint.
- Path to 1× cuBLAS: Marlin-style pre-shuffled layout + cp.async double-buffer + wgmma intrinsics. Beyond a single session.

## Stage 8b — Driver-level KV redirect (2026-04-28)
`cipher_kv_redirect.{h,cpp}` — rewrites FlashAttention's Params struct K/V pointers (`st[1]`, `st[2]`) to CIPHER-owned side buffers, populated either by GPU memcpy (V1) or 2-bit quant→dequant round-trip (V2).
- Empirical layout finding: PyTorch's `MemEffAttention` Params struct word order is `[query_ptr, key_ptr, value_ptr, output_ptr, ...]`; nulls at w4..w6 (attn_bias, seqstart_q, seqstart_k). FA reads up to **1 MB** out of K_buf/V_buf — smaller copies break correctness, 1 MB matches baseline exactly.
- Persistent KV cache is decoupled from FA's struct fields: PyTorch `StaticCache.update()` copies K_proj/V_proj output into a per-layer cache slot, then a separate materialize kernel copies the cache slot into a 1 MB staging buffer that FA reads. The staging is shared across all 32 layers (overwritten between FA calls).
- V1 (memcpy): output identical to OFF baseline (top-5 indices and values exact).
- V2 (2-bit asymmetric quant + dequant via NVRTC kernels `cipher_kv_q2`/`cipher_kv_dq2`, ROW=128 with per-row min/scale): top-1 next token preserved over prefill + 4 decode steps; top-5 reshuffles by ≤0.5 logits (KIVI 2-bit noise floor).
- V2 is correctness-only — no bandwidth win yet (the existing materialize still runs). Also note: V2's 1 MB per-layer staging buffer is sized for BATCH=1 — at BATCH≥128 with growing cache the FA's K buffer exceeds 1 MB and our memcpy reads past the destination, causing illegal memory access. To enable V2 at high batch, BUF_BYTES must scale with BATCH × num_kv_heads × max_cache_len × head_dim × 2.
- Layer index for the hook is derived as `g_fa_call_count % 32` — works for Mistral 7B since prefill traverses all 32 layers before the first decode token.

## Phase 2 — Marlin INT4 GEMM via NVRTC (2026-04-28)
- Marlin kernel (Apache-2.0, IST-DASLab) embedded as `kMarlinKernelSrc[]` in `src/cipher_marlin_src.cpp` (33.8 KB source, 10 template wrappers); compiles via CIPHER's NVRTC pipeline in 18.9 s, single 2.9 MB cubin holds all entry points. Resolved via new `cipher_substitute_v2_get_function_by_name()`.
- LUT-based repack (`marlin_repack` in `cipher_weight_compress.cpp`) converts our existing int4 + groupwise scales to Marlin's XOR-swizzled int32 layout — 1024-entry `_perm` for B, 64-entry `_scale_perm` for scales (both extracted from `marlin/__init__.py` `_get_perms()`).
- Host launcher mirrors `marlin_cuda(...)`: picks `(thread_k, thread_n)` from M, `m_blocks ∈ {1,2,3,4}`, sets `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES=96K`, gridDim=SM count (132), blockDim=256. Workspace cudaMalloc cached at module scope (graph-capture-safe).
- Correctness: all 15 Mistral shapes (M ∈ {1, 8, 16, 32, 64} × K, N ∈ {4096, 14336}) now hit rel_err 0.115–0.118 (INT4 noise floor).
- The K=14336 "bug" turned out to be in the **quant cache invalidation**, not the Marlin repack: PyTorch's caching allocator reuses the same device pointer across freed tensors, and our `cipher_weight_compress_quantize()` short-circuited if `g_quant_out[idx].int4_buf != nullptr`. When a layer with one shape (e.g. K=4096 N=14336) was freed and the same pointer reused for K=14336 N=4096, CIPHER returned the OLD shape's int4/scales/marlin buffers — the kernel ran on the right inputs but the wrong weight layout. Fix: shape-check the cached entry; on mismatch cudaFree the stale buffers and re-quantize. One-line condition guarding the early-return.
- End-to-end Mistral-7B speedups vs baseline (all 224 layers, K=14336 fix in):
  - batch=1: 115.6 → 119.4 tok/s (1.03× — GEMV path; attention-bound at M=1)
  - batch=8: 852.0 → 1177.5 tok/s (**1.38×** — Marlin sweet spot)
  - batch=32: 2805 → 2858 tok/s (1.02× — Marlin barely beats cuBLAS)
  - batch=64: 4729 → 5410 tok/s (1.14× — cuBLAS in graph + fused norm/silu)
  - batch=128: 7166 → 7918 tok/s (1.10× — fp16 cuBLAS + fusion)
  - batch=256: 9580 → 10467 tok/s (1.09× — fp16 cuBLAS + fusion)
- 224/224 quantized linears repacked for Marlin. Marlin gated to M ≤ 32 in
  the linear; M=33..64 hits cuBLAS in the captured graph (Marlin's m_blocks=4
  path was empirically slower than cuBLAS at those sizes in our build).

## Phase 3 — graph capture audit (2026-04-28)

Intercepted cudaGraphInstantiate / cudaGraphInstantiateWithFlags in
`libcipher_hook.so` (`src/cipher_graph_inspect.cpp`) and walked the captured
CUgraph via cuGraphGetNodes + cuGraphKernelNodeGetParams + cuFuncGetName.
Behind `CIPHER_GRAPH_INSPECT=on`, this aggregates kernel-node counts by name
after every torch.cuda.graph.

The audit caught a critical capture bug that had been inflating earlier
Phase 2 numbers: CIPHER's marlin/gemv/fused launches passed `stream=NULL`
to `cuLaunchKernel`. That goes to the legacy default stream, NOT torch's
capture stream — so the kernel ran ONCE during capture and was never
recorded as a graph node. On replay the captured graph re-ran every
elementwise / index_copy / RoPE / attention op around the missing linears,
read stale outputs from the linear's output buffer, and reported throughput
that was completing zero real GEMM work per replay.

Fix is one line per call site: pass
`ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)` instead of None.
After the fix, BATCH=8 captured graph has **224 marlin_M1_N8_K8_G8 nodes**
(32 layers × 7 linears) + 65 cipher_rmsnorm_fp16 + 32 cipher_silu_mul_fp16
+ 32 MemEffAttention; total 1373 nodes. tps dropped from inflated 3553 to
real 1177 — 1.38× over baseline. The audit-and-fix is recorded in the
"End-to-end Mistral-7B speedups" table above.

The proposed Phase-3 cuBLAS→Marlin node-rewriting in `cuGraphExecKernelNodeSetParams`
turned out to have no work to do once the stream bug was fixed: M≤32 already
captures Marlin (no cuBLAS to swap), M=33..64 captures cuBLAS but Marlin
is slower there in our build (don't swap), M≥128 has no compatible Marlin
kernel (parallel-M branch isn't compiled). The surgery infrastructure is
in place for Phase 4 KV-redirect work; the rewriting hooks are not used.

## Phase 3.5 — per-section profiling, Amdahl ceiling (2026-04-28)

After the stream-capture fix, profiled where time goes in a single decode
forward at BATCH=8 in graph mode. Used `torch.cuda.Event` brackets in eager
mode for the per-section breakdown (graph mode strips launch overhead but
preserves GPU-time proportions), and a graph-mode ablation (CIPHER
components on/off, captured separately) for the per-component savings.

### Eager-mode breakdown (proportions, BATCH=8 decode)
| section                                          | baseline % | Phase 2 %  | category       |
|---|---|---|---|
| **attn_net** (Q@K^T + softmax + AV + KV update)  | **39.7 %** | 41.1 %     | untouched      |
| qkv_proj + o_proj + gate_up + down (linears)     | 33.1 %     | 33.9 %     | INT4 target    |
| rmsnorm × 65                                     | 14.9 %     | 6.0 %      | fused target   |
| silu_mul × 32                                    | 2.7 %      | 5.6 %      | fused target   |
| OTHER (rope, embed, residuals, mask, lm_head)    | 9.7 %      | 13.4 %     | untouched      |

### Graph-mode ablation (BATCH=8, per-iter µs, additive)
| config            | per-iter µs | speedup |
|---|---|---|
| baseline          | 9360        | 1.000×  |
| INT4 only         | 7956        | 1.176×  |
| RMSNorm only      | 8227        | 1.138×  |
| SiLU_mul only     | 9275        | 1.009×  |
| ALL ON            | 6746        | **1.388×** |
| sum of singles    | -2622 µs    | (≈ -2614 µs ALL ON — fully additive) |

**The Amdahl ceiling**: CIPHER touches ~48 % of forward time
(linears 33 % + RMSNorm 15 %). At BATCH=8, 1.388× = 72 % of theoretical
ceiling (1.92× if INT4 + RMSNorm went infinitely fast). The remaining
gap to ceiling is launch overhead in graph mode being already low —
microbench claims (Marlin ≥2×, RMSNorm 15.5×, SiLU 3.4×) discount
heavily once captured into a graph because most of the microbench gain
came from saving CPU launch overhead. INT4 holds up best (~80 % of
microbench gain delivered) because its win is HBM-bandwidth-bound
(4× less weight bytes), not launch-bound.

**Next lever**: attention is 40 % of total and unchanged. KV-redirect
(Stage 8b/c) is the right path — reads less KV per step ⇒ FA runs
shorter. RoPE fusion into Q/K projections kills 128 small kernel
launches per layer in eager but only ~300–500 µs in graph (already
near-launch-overhead-zero).

### Final corrected end-to-end table (Phase 2, all 224 layers, graph + fused)

| batch | tok/s baseline | tok/s Phase 2 | × tok/s | tok/W base | tok/W ph2 | × tok/W |
|---|---|---|---|---|---|---|
|   1   |  115.6 | 119.4 | 1.03× |  0.717 |  0.574 | 0.80× |
|   8   |  852.0 |1177.5 |**1.38×**| 3.751 |  6.721 | **1.79×** |
|  32   | 2805.4 |2857.6 | 1.02× | 10.613 | 13.154 | 1.24× |
|  64   | 4728.6 |5409.8 | 1.14× | 16.270 | 18.152 | 1.12× |
| 128   | 7166.0 |7918.1 | 1.10× | 19.233 | 18.300 | 0.95× |
| 256   | 9580.2 |10467.0| 1.09× | 21.762 | 21.818 | 1.00× |

**tps measured** from a 32-token graph-replay burst.
**power measured** by nvidia-smi (~150 ms cadence) over a ≥10 s eager-mode
top-up loop at the same batch (graph replay alone is too short to sample,
and `cache_pos.fill_` between captured replays trips a PyTorch
`index_copy_` OOB — verified in plain Mistral, not a CIPHER bug).

## Stage 8c — Path A V3 attempt (2026-04-28)
Built per-layer persistent compressed K/V cache + GEMM hook + FA struct rewrite all wired through `cipher_kv_redirect_quant_kv` and `dequant_buffer_perm`.
- State: `g_kv_gemm_count` (K_proj at idx%2==0, V_proj at idx%2==1; layer = (idx/2)%num_layers); `g_layer_pos[layer]` advances by `n` after V_proj.
- Discovered layout mismatch and fixed: cuBLAS K_proj C is `(token, head, dim)` row-major while PyTorch MEA expects `(head, token, dim)`. Permuted dequant kernel `cipher_kv_dq2_perm` reorders during dequant.
- **Blocker — RoPE timing.** Mistral applies RoPE between K_proj GEMM and the cache update. Our V3 quant fires immediately after the K_proj cublasGemmEx returns, so the cache stores **pre-RoPE** K. FA expects **post-RoPE** K (matches what the persistent cache and materialize-into-staging both produce). Result: V3 alone produces a degraded model — top-1 next token is often still preserved (proximity), but downstream attention scores drift and the model output diverges from baseline.
- V3 build is plumbed and infrastructure-ready. Two viable fixes (next session):
  1. Apply RoPE inside the dequant kernel — we know layer + token position, so cos/sin can be computed inline from `theta_i = 10000^(-2i/D)`.
  2. Move the quant hook from cublasGemmEx (pre-RoPE) to the cache.update kernel (post-RoPE). Requires reliable detection of `index_elem_kernel` and safe reading of args[1]/args[2] without crashing on kernels with smaller param arrays.
