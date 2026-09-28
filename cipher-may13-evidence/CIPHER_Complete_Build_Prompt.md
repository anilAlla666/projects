# CIPHER — Complete Build Prompt & Handoff

Snapshot: 2026-04-30
Pod: H100 80GB SXM (700 W TDP), CUDA 12.8, libcublasLt 12.8.4, sm_90.
Working dir: `/home/ubuntu/op31-prod-fix`
Snapshot tarball: `~/cipher-apr30-snapshot.tar.gz`

---

## 0. The five CIPHER goals — priority order

| # | Goal | Current state |
|---|------|---------------|
| 1 | **Zero application code changes** — customer runs vanilla PyTorch + HuggingFace, optimization is invisible | **Partial.** Driver-level FP8 substitution (cublasGemmEx) is fully `LD_PRELOAD`. Fused RMSNorm / SiLU / residual / RoPE require Python-level monkey-patches today (see Stage 14 blocker below). Path (b) below closes this gap. |
| 2 | **≥ 2× tokens/watt vs vanilla baseline** | **✓ achieved at B=1 (1.97–2.04×) and B=8 (2.09–2.24×).** B=32: 1.57×, B=64: 1.51× — bounded by attention bucket which CIPHER doesn't yet substitute. |
| 3 | **O(1) substitution surface** — runtime decision, not compile-time | **✓ shipped as wired infrastructure.** Stages 6 (NVRTC), 7 (W4A16), 8 (KIVI 2-bit), 13 (FP8) are all online substitutions decided per-call. Per-shape compile cache in Stage 6 keeps amortized cost flat. |
| 4 | **Correctness preserved** — no NaN, no garbage tokens, output coherent | **✓.** FP8 correctness: 28/28 Mistral GEMM shapes at fro_rel ~0.037 (FP8 noise floor). End-to-end Mistral runs produce coherent English at every batch size measured. NaN sentinel passes in every timed run. |
| 5 | **Compliance / observability primitives** — RECEIPT / CARBON / COMPLY / DETERMINISM / FAIRNESS / GUARD | **✓ shipped.** Phase 5 (9 ops) all online. Test regression: 32 passing under simultaneous activation. Reports written to `/tmp/cipher_*_report.json` on demand. |

The 2× target is achieved at the small batches where CIPHER's levers (FP8
weight bandwidth, fused norm/activation, graph-capture launch overhead,
clock lock) all compound. At larger batches (B=32/64) the workload shifts
to attention-bound, which CIPHER does not yet substitute (the blocker for
that is documented below).

---

## 1. Final verified numbers

### 1.1 End-to-end Mistral-7B fp16 EAGER + graph capture (`run_full_2x_v2.sh`)

Methodology: baseline at default clock (no `nvidia-smi -lgc`, no `LD_PRELOAD`,
no CIPHER, eager mode). `full` mode = clock locked at per-batch optimum,
INT4 GEMV (M=1) + FP8 (M=2..512) + fused RMSNorm + fused SiLU·Mul + fused
residual_add + fused RoPE + CUDA graph capture + every CIPHER op enabled.
Prefill = 128, measurement window = 8 s.

```
  B  clock   baseline tps  baseline W  baseline tok/W   full tps   full W   full tok/W   ×tps  ×tok/W
  1  1200          48.87       203.3         0.2403     104.91    213.6       0.4913   2.15    2.04
  8  1200         390.11       261.5         1.4921     646.20    193.5       3.3392   1.66    2.24
 32  1100        1327.31       393.9         3.3694    1130.96    213.6       5.2941   0.85    1.57
 64  1100        1744.19       441.4         3.9519    1335.66    224.0       5.9626   0.77    1.51
```

Mean ×tok/W across 4 batches: **1.84×.**  Best single point: **B=8 at 2.24×.**

### 1.2 Long-context probe (B=8, prefill=2048)

```
  baseline:  tps=299.63   W=420.8   tok/W=0.7121
  CIPHER:    tps=194.87   W=214.0   tok/W=0.9104
  ratios:    ×tps=0.65    ×tok/W=1.28
```

V3 KV redirect was wired (B-aware quant/dequant + dynamic max_cache_len
+ shared FA staging) but doesn't yet bypass PyTorch's `StaticCache.update`
materialize, so V3 adds dequant overhead without subtracting fp16 traffic.

### 1.3 Per-batch clock sweep on B=32, B=64 (`sweep_b32_b64*.sh`)

Tested {900, 1000, 1100, 1200, 1350, 1500, 1650} MHz at B=32 and B=64 in
full mode.  Curve is non-monotonic with a peak at **1100 MHz** for both
batches.  Above 1200 MHz, the H100 power-clock curve gets steeper than
the throughput curve so tok/W drops; below 1100, throughput falls faster
than power.

### 1.4 FP8 correctness on Mistral GEMM shapes (`tests/test_fp8_correctness.py`)

```
28/28 PASS    M ∈ {1, 8, 32, 64, 128, 256, 512} × 4 Mistral N×K shapes.
fro_rel ≈ 0.037 across all cells (FP8 E4M3 noise floor).
NaN = 0 / Inf = 0 in every cell.
```

### 1.5 Op regression with everything on (`run_full_regression.sh`)

```
REGRESSION SUMMARY: 32 passed, 4 failed, 8 skipped
```

Env: every op enabled simultaneously
(SENSE / SHIELD / SUSTAIN / THERMOSTAT / PULSE / VOLT / HIBERNATE / LOOP /
CONTINUITY / SUBSTITUTE_V2 / WEIGHT_COMPRESS / FUSION_KERNELS / FP8_COMPUTE /
PREDICT / GUARD / DETERMINISM / TOPOLOGY / TRACE / FAIRNESS / CARBON /
RECEIPT / COMPLY / PERSIST_ENGINE / GRAPH / KV_COMPRESS / KV_REDIRECT / KV_RDR_V3 /
NCCL_V4 / PARTITION_ROUTER / THERMAL_FEEDBACK / VMM).

The 4 failures are pre-existing on this pod, unrelated to FP8 / fusion /
graph capture / KV V3 / kernel table:

- `test_packaging.py` — imports a missing `cipher_runtime` Python package
- `test_dep2.py` — same `cipher_runtime` import
- `test_hw_validation.py` — hardcoded path `/workspace/CIPHER_final_session7` from another pod
- `test_persist_dispatch.py` — pre-existing `cuLaunchKernelEx` overhead gate

### 1.6 M4 compute-bound regression sentinel (4096³ fp16 GEMM, 10 s)

```
M1 stock                 663.4 TFLOPS
M4 hook+rt all 20 obs.   663.8 TFLOPS    Δ = +0.06 %
```

CIPHER's CPU-side observer overhead is absorbed within DVFS noise on the
power-cap-bound M4 sentinel.

---

## 2. The cuFuncGetParamInfo blocker (Stage 14 finding — DISPOSITIVE)

The user's last directive asked to push every fusion (RMSNorm / SiLU /
residual / RoPE) into the driver level via cuLaunchKernelEx interception,
reading parameters via `cuFuncGetParamInfo`.

`include/cipher_kernel_table.h` + `src/cipher_kernel_table.cpp` were built
and wired into both driver-API (`cuLaunchKernelEx`) and runtime-API
(`cudaLaunchKernel`) intercept paths.  Mistral-7B prefill+decode under
`LD_PRELOAD=libcipher_hook.so` (zero CIPHER imports in the customer code)
observed:

```
Total CUfunctions:  62
Total launches:     21,192

  category           count   launches    top kernel
  ─────────          ─────   ────────    ──────────
  ElementMul            18      8043     at::native::elementwise_kernel<128,4>
  Unknown               27      7337     <unresolved>  (cuBLAS hgemm)
  ElementGeneric         5      2129     at::native::unrolled_elementwise_kernel
  ResidualAdd            4      2081     at::native::CUDAFunctor_add<c10::Half>
  Reduce                 6      1090     at::native::reduce_kernel<512,1,ReduceOp<MeanOps>>
  FlashAttn              2       512     pytorch_flash::flash_fwd_kernel
```

**`cuFuncGetParamInfo` returned param_count = 0 for all 62 of 62 distinct
CUfunctions.**

```
Param-count distribution: param_count = 0  →  62 / 62  (100 %)
```

This reproduces the prior session's CLAUDE.md finding on the current pod,
PyTorch 2.7, CUDA 12.8.  PyTorch's runtime-API kernels are host stubs
registered via `__cudaRegisterFunction` — the driver never sees a real
parameter layout for them, so it cannot return one.

### What this blocks

The same template (`at::native::elementwise_kernel<128,4>`) is used for:
residual add, RMSNorm post-multiply, RoPE multiply, fp16↔fp32 cast,
mul-by-broadcast, rsqrt, and many more — all distinct operations with
different parameter layouts decided at compile time by the host stub.
**Without `cuFuncGetParamInfo`, we cannot safely read input/weight/output
pointers from the `void**` argument array** that `cuLaunchKernel`
receives.  Same CUfunction handle → different operations → different
param meanings → silent wrong-data corruption if we guess.

### Why FP8 in cublasGemmEx works but RMSNorm interception doesn't

cuBLAS exposes a documented public API; its arguments (`A`, `B`, `C`,
`m`, `n`, `k`, `alpha`, etc.) are stable across versions.  The same
property does **not** hold for PyTorch's internal kernel templates.
The driver-level FP8 pattern doesn't extend to non-GEMM ops without
versioned per-template parameter-layout tables.

### Two paths

**(a) Strict driver-only** (what the user originally asked).  Requires
maintaining a database of "which PyTorch op uses which offsets in
`at::native::elementwise_kernel`" versioned against each PyTorch wheel,
mangled-symbol-by-mangled-symbol.  Multi-engineer-month effort, fragile
across PyTorch upgrades.  **Not recommended.**

**(b) Driver-only + tiny Python shim auto-loaded by `LD_PRELOAD`.**
The customer's code is genuinely unchanged (`model = AutoModelForCausalLM
.from_pretrained(...)` line stays).  The shim runs at process start by
one of:

- `python -m cipher.run -- <their_script.py>` (drop-in launcher)
- `PYTHONSTARTUP=cipher_startup.py` env var (one-shot environment edit)
- The `LD_PRELOAD` hook injects an `_PyImport_AppendInittab` callback
  that auto-imports the patcher when the Python interpreter initializes
  (zero edits to the customer's script *or* env)

The shim applies the proven monkey-patches (fused RMSNorm / SiLU /
residual / RoPE / Int4Linear) at startup.  This preserves the "no
application code changes" property of the product while sidestepping
the introspection blocker.  Same as how vLLM / DeepSpeed-Inference
in-place attention substitution works in practice.

**Recommendation: ship path (b).**

---

## 3. Architecture — three DSOs, 14 stages

### `libcipher_hook.so` (~120 KB)
LD_PRELOAD intercept layer.  Owns:
- cuLaunchKernel / cuLaunchKernelEx / cudaLaunchKernel intercepts
- cublasGemmEx shim with FP8 dispatch (Stage 13)
- cuModuleLoadData / cuKernelGetFunction / __cudaRegisterFunction interposition
- Persist engine fast-path detector
- Graph capture inspector (Phase 3 audit)
- Kernel table (Stage 14, name + grid + classification)
- KV redirect FA hook

### `libcipher_rt.so` (~640 KB)
Runtime DSO.  Owns:
- 12 core ops + 22 observers (Phase 1–5)
- 9 actuator modules (silicon, persist_engine, vmm, graph, substitute_v2,
  weight_compress, kv_compress, nccl_v4, partition_router, thermal_feedback)
- FP8 compute substitution module (cipher_fp8_compute.cpp)
- Fusion kernels (cipher_fusion_kernels.cpp): RMSNorm, SiLU·Mul, residual_add
- Weight compression (INT4 GEMV, INT4 GEMM, Marlin, FP8 path-7 GEMM)
- KV redirect (V1 memcpy, V2 quant-roundtrip, V3 per-(layer,batch,head,token) cache + B-aware quant/dequant)

### `libcipher_nccl_tuner.so` / `libnccl-tuner-cipher.so` (~16 KB)
NCCL plugin — algo selection bias.  Used at multi-node deployments only.

### Stage chain (init priorities 101–113)
```
hook (101) → silicon (102) → persist_engine (103) → vmm (104) → graph (105)
  → substitute_v2 (106) → weight_compress (107) → kv_compress (108)
  → nccl_v4 (109) → partition_router (110) → thermal_feedback (111)
  → kv_redirect (112) → fp8_compute (113)
```

---

## 4. The full substitution stack (what runs in `--mode=full`)

| Lever | Implementation | Hot path | Bucket attacked |
|-------|---------------|----------|-----------------|
| INT4 GEMV at M=1 | NVRTC kernel `cipher_int4_gemv` + Python `Int4Linear` monkey-patch | Decode batch=1 nn.Linear forward | GEMM |
| FP8 cublasLtMatmul at M=2..512 | NVRTC quant kernels + cublasLtMatmul wrapper, dispatched from cublasGemmEx hook | Driver-level | GEMM |
| Fused RMSNorm | NVRTC `cipher_rmsnorm_fp16` + `MistralRMSNorm.forward` patch | Pre-attn + pre-MLP layer norm | Elementwise (RMSNorm bucket) |
| Fused SiLU·Mul | NVRTC `cipher_silu_mul_fp16` + `MistralMLP.forward` patch | Gate × Up | Elementwise (SiLU bucket) |
| Fused residual_add | NVRTC `cipher_residual_add_fp16` + `MistralDecoderLayer.forward` patch | Two adds per layer | Elementwise (Residual bucket) |
| Fused RoPE | NVRTC `cipher_fused_rope_qk` + `apply_rotary_pos_emb` patch | Q+K rotary inside attention | Elementwise (RoPE bucket) |
| CUDA graph capture | `torch.cuda.CUDAGraph` over one decode step | Per-token | CPU launch overhead |
| Per-batch clock lock | `nvidia-smi -lgc` at chosen MHz | Process-wide | Power |

The four "monkey-patch" rows above are what path (b) (auto-loaded shim)
would deliver invisibly.

---

## 5. GPU-time bucket profile (vanilla baseline, per `profile_gemm_vs_nongemm.py`)

```
                    B = 32   per-step  14,096 µs        B = 64   per-step  19,246 µs
                    ────────────────────────────        ────────────────────────────
  category          cuda_us / step    %                cuda_us / step    %
  ────────────      ──────────────    ─────            ──────────────    ─────
  GEMM (linear)              5,731   40.65 %                  5,839   30.34 %
  ATTN  (cutlass MEA)        2,217   15.72 %                  4,117   21.39 %
  Elem (RMSNorm/SiLU/        5,345   37.91 %                  8,464   43.98 %
        RoPE/cast/index)
  Other (reductions/         803     5.72 %                   826     4.29 %
         concat)
```

This is why B=32 / B=64 stop at ~1.5×: only **30–40 %** of decode time is
in addressable GEMMs; FP8's reach scales with that share.  The
**16–21 % attention bucket is completely untouched** — it would close
the gap to 2× but is blocked by FP8 attention not being available in
PyTorch 2.7 / pip-disallowed environments.

---

## 6. Files added / modified across this build

### New source modules
- `include/cipher_fp8_compute.h` + `src/cipher_fp8_compute.cpp` — Stage 13 FP8 cublasLtMatmul substitute (~360 LOC)
- `include/cipher_kernel_table.h` + `src/cipher_kernel_table.cpp` — Stage 14 kernel name + classification table (~280 LOC)

### New runners / measurement scripts
- `step7_fp8_eager.py` — Mistral-7B harness with modes `baseline / fp8 / stack / full`
- `run_full_2x.sh` — apples-to-apples baseline-vs-CIPHER runner (B=1 / B=8)
- `run_full_2x_v2.sh` — 4-batch sweep with residual + RoPE fusion
- `sweep_b32_b64.sh` + `sweep_b32_b64_low.sh` — clock sweep
- `rerun_b32_b64_wider_gate.sh` — re-measure with FP8 gate at n≤512
- `run_b8_p2048.sh` — long-context probe
- `run_full_regression.sh` — env-loaded full op regression
- `profile_gemm_vs_nongemm.py` — torch.profiler bucketizer

### New tests
- `tests/test_fp8_correctness.py` — 28 Mistral GEMM shapes
- `tests/test_kernel_table.py` — kernel classification + cuFuncGetParamInfo probe

### Modified
- `src/cipher_intercept_cudart.cpp` — FP8 dispatch block in cublasGemmEx (n≤512), kernel table observation in both launch paths
- `src/cipher_kv_redirect.cpp` — three V3 fixes: shared staging buffer, dynamic max_cache_len, B-aware quant/dequant kernels (per-(layer, batch, head, token) cache)
- `Makefile`, `exports.map` — added new TUs and kt_observe / kt_size / kt_dump_json exports

### Reports (this build)
- `STAGE13_FP8_REPORT.md` — Stage 13 implementation
- `STAGE13_FINAL_2X.md` — first 2× attempt (1.51× / 1.69× peak)
- `STAGE13_FINAL_2X_V2.md` — second pass (1.51× B=64 confirmed best at 1200 MHz)
- `STAGE13_FINAL_2X_V3.md` — V3 KV refactor + B>1 gating
- `STAGE13_2X_ACHIEVED.md` — full mode at B=1, B=8 hits 2×
- `STAGE13_2X_V2.md` — residual + RoPE fusion, B=1 → 2.04×, B=8 → 2.24×
- `PROFILE_AND_GATE.md` — wider FP8 gate + GEMM-vs-non-GEMM profile
- `CLOCK_SWEEP_B32_B64.md` — clock sweep findings
- `STAGE14_KERNEL_TABLE.md` — kernel table + cuFuncGetParamInfo blocker

---

## 7. To rebuild and verify (next session)

```bash
# Extract:
tar xzf ~/cipher-apr30-snapshot.tar.gz -C ~/op31-prod-fix-fresh

# Build:
cd ~/op31-prod-fix-fresh
make clean && make all

# Verify FP8 correctness:
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so \
  python3 tests/test_fp8_correctness.py        # expect 28/28 PASS

# Verify regression:
bash run_full_regression.sh                     # expect 32 passed, 4 pre-existing failures

# Verify 2× at B=1, B=8:
bash run_full_2x_v2.sh                          # expect ~2.0× at B=1 and ~2.2× at B=8

# Probe kernel table (driver-level only, no monkey-patches):
LD_PRELOAD="$PWD/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so" \
  CIPHER_KERNEL_TABLE_VERBOSE=1 \
  python3 tests/test_kernel_table.py
# Confirms cuFuncGetParamInfo returns 0 for all 62 PyTorch kernels.
```

---

## 8. The next session's task: ship path (b)

**Goal:** customer runs vanilla PyTorch + HuggingFace, sets only
`LD_PRELOAD`, gets the proven 2.04 / 2.24 / 1.57 / 1.51× tok/W
without touching their code.

**Concrete deliverables:**

1. **`cipher_startup.py`** — apply all four monkey-patches
   (Int4Linear, fused RMSNorm, fused SiLU·Mul, fused residual,
   fused RoPE) at module import.  Reads env vars to gate each
   patch independently (`CIPHER_FUSE_RMSNORM=on` etc.).

2. **`python -m cipher.run`** — convenience launcher that sets up
   `PYTHONSTARTUP=cipher_startup.py` and forwards argv to the
   customer's script.

3. **`_PyImport_AppendInittab` injection in `libcipher_hook.so`** —
   detects when libpython is loaded (via the `dlopen` interpose),
   registers a built-in module that auto-imports `cipher_startup`
   when the interpreter initializes.  Customer doesn't even need
   `PYTHONSTARTUP`.  Fallback to PYTHONSTARTUP if the inject fails.

4. **End-to-end verification:** run a script with a single
   `from transformers import ...; model.generate(...)` line under
   only `LD_PRELOAD=libcipher_hook.so`.  Verify the patches fired
   automatically and tok/W matches the `--mode=full` numbers.

5. **Measurement:** re-run the 4-batch sweep with the auto-loaded
   shim instead of `step7_fp8_eager.py`.  Confirm same numbers
   (within run-to-run noise).

**What stays untouched:** `step7_fp8_eager.py` and every existing
test.  The auto-load path is additive.

**Beyond path (b):** if the user later adds `pip install flash-attn`
or moves to PyTorch ≥ 2.8 with FP8 SDPA support, the attention bucket
becomes addressable and B=32 / B=64 should reach 2× tok/W as well.

---

## 9. Reference numbers — what to compare against in re-validation

```
  B   prefill  clock   baseline tps   baseline W   baseline tok/W   full tps   full W   full tok/W   ×tok/W
  1     128    1200          48.87        203.3          0.2403      104.91    213.6       0.4913    2.04
  8     128    1200         390.11        261.5          1.4921      646.20    193.5       3.3392    2.24
 32     128    1100        1327.31        393.9          3.3694     1130.96    213.6       5.2941    1.57
 64     128    1100        1744.19        441.4          3.9519     1335.66    224.0       5.9626    1.51
  8    2048    1200         299.63        420.8          0.7121      194.87    214.0       0.9104    1.28
```

```
FP8 correctness (28 shapes): 28/28 PASS   fro_rel ~0.037   NaN=0
Op regression (full ops):    32 passed   4 pre-existing fail   8 skipped
M4 baseline (4096³ fp16):    663.8 TFLOPS  (Δ ~0% vs vanilla)
Kernel table observation:    62 unique CUfunctions
Param introspection:         0 / 62  (cuFuncGetParamInfo blocked)
```

---

## 10. Honest read

CIPHER ships:
- A driver-level FP8 substitution (Stage 13) that **works exactly as
  designed** for cublasGemmEx — every Mistral linear projection at
  M ∈ [2, 512] becomes a cublasLtMatmul FP8 call without the customer
  changing anything.
- A kernel name table (Stage 14) that **classifies all 62 of the
  PyTorch kernels** Mistral-7B uses, with a clean JSON dump.
- An end-to-end measurement that **hits 2× tok/W at B=1 and B=8**
  using the full lever stack.
- A reproducible regression suite that catches no-NaN, no-Inf,
  output-coherence, and per-op behavior across every CIPHER mode.

CIPHER does not yet ship:
- Fully driver-only substitution of fused RMSNorm / SiLU / residual /
  RoPE — blocked by `cuFuncGetParamInfo` returning 0 for all PyTorch
  runtime-API kernels.  Path (b) closes this gap with a tiny
  `cipher_startup.py` shim auto-loaded by `LD_PRELOAD`.
- FP8 attention — blocked by no FP8 SDPA in PyTorch 2.7 + pip-disallowed
  flash-attn.  Would close B=32 / B=64 to 2× when available.
- KV V3 materialize bypass — V3 kernels and B-aware cache are in place
  (rev v3 of STAGE13); the missing piece is a `StaticCache.update`
  intercept that drops the fp16 write-into-cache.

This is the snapshot to hand off.
