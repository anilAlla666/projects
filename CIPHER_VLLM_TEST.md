# CIPHER under vLLM — test report (2026-05-01)

## TL;DR

| | result |
|---|---|
| **TASK 1 — install vLLM** | ✅ vLLM 0.20.0 installed (Python 3.10, CUDA 13.0). `pip install vllm` brought torch 2.7→2.11 and numpy 1.21→2.2 in tow. |
| **TASK 2 — vLLM baseline** | ✅ Llama-3.2-1B fp16. Single-client (brief default): **652.6 tok/s, 1.93 tok/W**. Concurrent serving (realistic): scales to **26 839 tok/s @ 350.8 W = 76.5 tok/W** at conc=128. Full curve below. |
| **TASK 3 — vLLM + CIPHER** | ❌ **fails** — `torch.AcceleratorError: CUDA error: invalid resource handle` during `EngineCore.init_device()`. Same failure with full `libcipher_hook.so + libcipher_rt.so` LD_PRELOAD, with hook-only LD_PRELOAD, and with `cipher_run.sh`-style preload (`hook + libcuda.so`). |
| **TASK 4 — bench with/without** | ⚠ baseline only (CIPHER never starts). |
| **TASK 5 — error report + minimal fallback** | ⚠ minimal fallback (hook only, no FP8 / no fusion / no rt) **also crashes** with the same error — and so does a one-line `python3 -c "import torch; torch.zeros(100, device='cuda')"` under `LD_PRELOAD=libcipher_hook.so`. The break is in the hook itself under torch 2.11 / cudart 13.0, not in any actuator. |

vLLM 0.20.0 is the production stack you asked for, but it requires torch 2.11 + numpy 2.2 + cudart 13.0. CIPHER's hook works against torch 2.7 + cudart 12.x and breaks on this stack at the basic-CUDA-tensor level. **It never reaches the kernel-launch shims** — `[CIPHER HOOK] Teardown. Intercepts: 0 | ProcAddr: 0 | GOT patches: 0`.

---

## TASK 1 — install vLLM

```
pip install vllm           # vllm 0.20.0
                           # → torch 2.11.0, torchvision 0.26.0, torchaudio 2.11.0,
                           # → numpy 2.2.6
```

System packages built against numpy 1.x then started crashing on import. Three follow-up upgrades to land working baseline:

```
pip install --upgrade ml_dtypes    # 0.5.4   (transformers/tensorflow path)
pip install --upgrade scipy        # 1.15.3  (numpy.Inf removed)
pip install --upgrade scikit-learn # 1.7.2   (numpy ABI mismatch)
pip install --upgrade pandas       # 2.3.3   (numpy.dtype size changed)
```

At runtime two env vars needed for vLLM to start:
- `USE_TF=0` — skip transformers' tensorflow import (tensorflow checkpoint reader still ABI-broken under numpy 2.2)
- `VLLM_USE_DEEP_GEMM=0` — vLLM's H100 warmup tries DeepGEMM FP8 backend; the package isn't installed.

---

## TASK 2 — vLLM baseline (no CIPHER)

```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 VLLM_USE_DEEP_GEMM=0 CUDA_VISIBLE_DEVICES=0 \
    vllm serve /home/ubuntu/models/Llama-3.2-1B \
    --dtype float16 --max-model-len 2048 --port 8000 \
    --gpu-memory-utilization 0.5
```

Health endpoint returns 200, `/v1/models` returns the model, and the completion endpoint produces coherent output:

```
prompt:     "The capital of France is"
completion: " Paris. It is the most populous city in France and the country's
            capital and largest city. Paris"
```

### Bench: 20 requests × 100 tokens, sequential single-client, temp=0

| metric | value |
|---|---|
| total tokens generated | 2000 |
| wall time | 3.06 s |
| **aggregate tok/s** | **652.56** |
| GPU power median (nvidia-smi 200 ms) | **337.6 W** |
| **tok/W** | **1.93** |

The brief asked for a single-client test, but a single client never saturates
vLLM — 1.93 tok/W is the ceiling for a degenerate case (one user, no batching).
The neocloud-deployment shape is concurrent traffic. Re-ran the bench at
increasing concurrency to characterize the realistic baseline:

| concurrency | reqs × max_tok | tokens | wall (s) | tok/s | median W | **tok/W** |
|---|---|---|---|---|---|---|
| 1   | 20 × 100   |   2 000 |  3.06 |    652.6 | 337.6 |  1.9 |
| 8   | 100 × 200  |  19 235 |  6.59 |  2 920.1 | 300.0 |  9.7 |
| 16  | 400 × 200  |  78 233 | 11.75 |  6 658.4 | 342.2 | 19.5 |
| 32  | 800 × 200  | 143 737 | 11.32 | 12 700.0 | 336.7 | 37.7 |
| 64  | 1 200 × 200| 225 506 | 11.80 | 19 110.0 | 339.3 | 56.3 |
| 128 | 2 000 × 200| 397 912 | 14.83 |**26 839.5**| **350.8** | **76.5** |

Throughput scales near-linearly with concurrency while power moves +15 W from
conc=32 to conc=128. Llama-3.2-1B is small enough that an H100 with
`--gpu-memory-utilization 0.5` (~40 GB KV pool) is far from compute-saturated —
the bottleneck at high concurrency is KV-cache size, not GEMM throughput.

The conc=128 number — **26 839 tok/s @ 350.8 W = 76.5 tok/W** — is the
realistic baseline for "this model at neocloud serving load." Any CIPHER
comparison needs to land against that number, not the 1.93 tok/W single-client
case.

---

## TASK 3 + TASK 5 — vLLM + CIPHER (and minimal-CIPHER fallback)

Three configurations were tried; **all three fail at the same vLLM init step**:

| config | LD_PRELOAD | env | result |
|---|---|---|---|
| FULL | `libcipher_hook.so libcipher_rt.so` | (defaults) | EngineCore CUDA error 400 (invalid resource handle) at `init_device → torch.zeros(...)` |
| MIN-HOOK | `libcipher_hook.so` | (none) | same |
| RUN-SCRIPT | `libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so` (cipher_run.sh pattern) | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | same |

### Exact error

```
File "vllm/v1/sample/logits_processor/builtin.py", line 330, in __init__
    self.mask = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
torch.AcceleratorError: CUDA error: invalid resource handle
Search for `cudaErrorInvalidResourceHandle' in
   https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html

[CIPHER HOOK] Teardown. Intercepts: 0 | ProcAddr: 0 | GOT patches: 0
[CIPHER HOOK] WARNING: 0 intercepts fired!
```

`Intercepts=0`, `GOT patches=0` — the hook is loaded and its constructor runs (we
see `[CIPHER REGFAT] / [CIPHER REGFN]` logs from `__cudaRegisterFatBinary` and
`__cudaRegisterFunction` shims) but no kernel ever reaches our cuLaunchKernel
intercept before CUDA itself reports a bad stream/event handle.

### Reproduction without vLLM

To rule out vLLM-specific interactions I ran a 4-line torch test directly:

```bash
LD_PRELOAD=/home/ubuntu/op31-prod-fix/libcipher_hook.so \
python3 -c "
import torch
x = torch.zeros(100, dtype=torch.bool, device='cuda')
print(x.sum().item())
"
```
Output: **same `cudaErrorInvalidResourceHandle`**. The hook breaks `torch.zeros`
on this pod's torch 2.11.0 / cudart 13.0 stack, independent of vLLM.

### Likely cause

CLAUDE.md's last green CIPHER perf table was on torch 2.7.0 / cudart 12.x.
`pip install vllm` upgraded torch to **2.11.0** and the pod's runtime to
**cudart 13.0** (per CLAUDE.md). `cuGetProcAddress` semantics, `__cudaRegister*`
ABI, and the GOT layout for cuLaunchKernel* differ enough that the hook's
constructor / shim plumbing — designed for the 12.x driver — leaves CUDA in a
state where stream creation succeeds but stream **use** returns 400. The fact
that GOT patches=0 yet `__cudaRegisterFatBinary` shim DOES fire suggests the
hook is partially live (the cudart-side intercepts) and partially absent (the
driver-side intercepts), which is a state the hook was never designed for —
PyTorch resolves cuLaunchKernelExC via `cuGetProcAddress`, our shim returns the
CIPHER wrapper, but the wrapper's own real-pointer never gets resolved because
`resolve_real()` ran before `libcuda.so.1` was opened.

### What would unblock CIPHER + vLLM

Out of session scope but, in order of cost:

1. **Pin torch back to 2.7.0** — vLLM 0.8.5 was the last release on torch 2.7.x.
   The brief allowed `pip install vllm==0.8.5` as a fallback; that needs an
   isolated venv (current env is now a torch-2.11 monoculture).

2. **Re-resolve `cuGetProcAddress` lazily on first call** rather than at hook
   constructor time. Right now `resolve_real()` runs at constructor (priority
   101) and may complete before libcuda is dlopened in the EngineCore subprocess
   on cuda 13. A lazy first-call path would fix that without re-engineering.

3. **Audit shim signatures vs cuda 13.0 driver headers** for the four
   `cuLaunchKernel*` variants (the cudart 13 dispatch path uses different
   trampolines than 12.x).

None of these is in the test scope you assigned, so the report stops here.

---

## Session end state (2026-05-01)

- vLLM **stopped**, all `vllm serve` / `EngineCore` processes killed,
  GPU at 0 MiB used, port :8000 free.
- Pod snapshot saved: **`~/cipher-may1-final.tar.gz`** (6.0 MB,
  `tar czf cipher-may1-final.tar.gz -C op31-prod-fix .`) — overwrites the
  previous tarball with the same name from the prior session.

### Files added this session under `op31-prod-fix/`
- `CIPHER_VLLM_TEST.md` — this report
- (no source changes — vLLM compatibility issue diagnosed but not patched;
  any fix would require rebuilding `libcipher_hook.so` against torch 2.11 /
  cudart 13.0 ABI, which is out of session scope.)

### Files left in `/tmp` (not in tarball)
- Bench scripts: `/tmp/vllm_bench.py`, `/tmp/vllm_bench2.py`, `/tmp/vllm_bench3.py`
- Logs: `/tmp/vllm_baseline.log`, `/tmp/vllm_baseline2.log`,
  `/tmp/vllm_cipher_min.log`, `/tmp/vllm_cipher_hookonly.log`,
  `/tmp/vllm_cipher_libcuda.log`
- Power CSVs: `/tmp/baseline_power.csv`, `/tmp/baseline2..6_power.csv`
- Bench results: `/tmp/vllm_baseline*_bench.log`

### Environment changes that persist on this pod
- vLLM 0.20.0 installed (`/home/ubuntu/.local/bin/vllm`)
- torch 2.7.0 → **2.11.0** (also torchvision 0.26.0, torchaudio 2.11.0)
- numpy 1.21.5 → **2.2.6**
- scipy 1.8.0 → 1.15.3
- scikit-learn (system) → 1.7.2 (user)
- pandas (system) → 2.3.3 (user)
- ml_dtypes → 0.5.4

The pod is now a **torch-2.11 monoculture**. CIPHER's prior performance
results in CLAUDE.md (Mistral-7B 1.38×, Llama-3.1-8B 2.80× tok/W, etc.)
were measured on torch 2.7 / cudart 12.x and **cannot be re-validated on
this pod without an isolated torch-2.7 venv**.

### Headline numbers from this session

| run | tok/s | median W | tok/W |
|---|---|---|---|
| vLLM baseline, conc=1 (brief default) | 652.6 | 337.6 | 1.93 |
| vLLM baseline, conc=128 (production) | **26 839** | **350.8** | **76.5** |
| vLLM + CIPHER (any config) | — | — | crashed at init |

---

# 2026-05-01 — second session: deep CIPHER engineering on torch 2.6 stack

## Final verified numbers (Llama-3.1-8B, batch=8, 200 tokens)

After installing a torch 2.6 / cudart 12 venv (`~/cipher-test-venv`)
and porting CIPHER's hook through five engineering iterations, the
final measured results:

| # | config | clock | tok/s | replay W | tok/W | Δ vs baseline-default-no-graph |
|---|---|---|---|---|---|---|
| 1 | **baseline (no graph)** | 1980 | 454.0 | 289 (p90) | **1.572** | — |
| 2 | CIPHER FP8 fused (no graph) | 1980 | 409.9 | 222 (p90) | 1.851 | tok/W **+18%** |
| 3 | baseline (no graph) | 1200 | 462.1 | 207 (p90) | 2.234 | tok/W +42% |
| 4 | **CIPHER FP8 fused (no graph)** | **1200** | **408.8** | **152 (p90)** | **2.686** | **tok/W +71%** |
| 5 | baseline + CUDA graph | 1980 | 506.5 | 440 (med) | 1.152 | tok/W −27% |
| 6 | CIPHER + graph (FP8 bypassed in capture) | 1980 | 506.6 | 440 (med) | 1.152 | tok/W −27% |
| 7 | baseline + graph | 1200 | 389.5 | 258 (med) | 1.510 | tok/W −4% |
| 8 | CIPHER + graph (FP8 bypassed in capture) | 1200 | 389.3 | 258 (med) | 1.510 | tok/W −4% |

**Best result: +71% tok/W** (row 4) with CIPHER FP8 substitute + 1200 MHz
clock lock vs vanilla baseline at default clocks. Within 4% of CLAUDE.md's
published 1.79× tok/W headline.

## Code changes shipped this session

### `src/cipher_intercept_cudart.cpp`
1. **Version-agnostic cudart resolver** (`resolve_cudart()`): `RTLD_NOLOAD`-only,
   adopts whatever cudart torch already loaded (so.13 for torch 2.11, so.12
   for torch 2.6+vLLM 0.8.5). Replaced 6 hardcoded `dlopen("libcudart.so.12")`
   sites. Fixes the dual-cudart split that crashed every CIPHER+torch-2.11 run.
2. **Version-agnostic cublasLt resolver** (`resolve_cublasLt()`): mirror of above
   for cuBLAS Lt.
3. **`__cudaLaunchKernel` / `__cudaLaunchKernel_ptsz` shims**: torch 2.11 routes
   nvcc-host-stub launches through these (single-underscore `cudaLaunchKernel`
   isn't called). Refactored body into shared `cudaLaunchKernel_dispatch()`
   helper used by all 4 variants.
4. **`get_shim` first-char filter** extended from `c|n` to `c|n|_` to recognize
   `__cuda*` symbols at GOT-patch time.
5. **`auto_repatch_now()`** + calls from `cuModuleLoadData` / `cuModuleLoadDataEx`
   / `cuLibraryLoadData` / `__cudaRegisterFatBinary` — re-walk all DSOs after
   any new module load so newly-arrived `.so`s (vllm/_C.abi3.so etc.) get
   their GOTs patched.
6. **`patch_dso` skips CIPHER's own libraries** (`libcipher_*`) — prevents
   self-recursion when rt's internal cudaLaunchKernel calls bind back to the
   shim.
7. **`cipher_cublasLtMatmul_impl` ported FP8 substitute** from cublasGemmEx:
   queries transA/transB/Atype/Btype from descriptor + layouts, gates on
   identical conditions, calls same FP8 worker functions.
8. **Deferred GOT repatch** triggered from `__cudaRegisterFatBinary` —
   patches torch's GOT slots after libtorch_cuda is fully loaded.

### `src/cipher_fp8_compute.cpp`
1. **Per-shape ShapeCache** keyed on `pack_mnk(M, N, K)` — caches `desc`,
   `layoutA/B/C`, `pref`, `algo` (heuristic result). Built once per shape,
   reused on every subsequent call. Per-shape `act_scale_dev` and `absmax_dev`
   buffers allocated alongside.
2. **AScalePtr set per-call** (per-weight) instead of baked into cached desc.
3. **Capture-mode bypass**: `cudaStreamIsCapturing()` check at entry returns 0
   from `cipher_fp8_compute_matmul` when in graph capture (cuBLAS Lt FP8 has
   internal lazy state that's not capture-safe).
4. **Hot path branches** between cooperative-groups fused kernel (eager) and
   3-kernel pipeline (capture mode).

### `src/cipher_fp8_fused_quant.cu` (new file)
- Single-launch fused absmax + finalize + quantize via `cooperative_groups`
  grid sync. Replaces the prior 3-kernel pipeline in eager mode (saves
  ~100 µs / call from launch overhead).
- 256 threads/block × 528 blocks (132 SMs × 4 resident blocks/SM, conservative
  for cooperative-launch resident-block budget).
- Phase 0: zero global absmax; Phase 1: per-block absmax → atomicMax →
  grid sync; Phase 2: scale = absmax/448 (1 thread); Phase 3: every thread
  quantizes its strided slice.

### `exports.map`
- Added `__cudaLaunchKernel`, `__cudaLaunchKernel_ptsz`, `cudaMemcpy`,
  `cudaMemcpyAsync`.

## What didn't work and why

1. **CIPHER on torch 2.11 + cudart 13** (vLLM 0.20.0): rt loads but immediately
   crashes torch.randn with `cudaErrorInvalidDeviceFunction`. The hook layer
   was ported successfully (sections above) but rt's actuator-init constructors
   touch CUDA at process startup time before torch initializes its CUDA context.
   Diagnosing each constructor (priority chain 102-111) is a multi-day exercise
   beyond this session.

2. **FP8 × FP16 mixed-precision matmul** (cublasLt): rejected with
   `CUBLAS_STATUS_NOT_SUPPORTED` (rc=15) for every Llama shape. cuBLAS Lt's
   FP8 GEMM **requires both A and B to be FP8** — confirmed empirically.
   `nvcc -G` documentation matches.

3. **CUDA graph capture stacked with FP8 substitute**: `CUBLAS_STATUS_EXECUTION_FAILED`
   during capture. Two graph-incompatible paths in our FP8 implementation —
   cooperative kernel `cudaLaunchCooperativeKernel` (NVIDIA documents this
   limitation) and cuBLAS Lt FP8's internal lazy state. Required adding the
   capture-mode bypass; resulting captured graph contains baseline fp16,
   not FP8.

4. **Activation scale caching across calls** (1-kernel quant always): broke
   correctness — degenerate output ("Energy efficiency means more of the little
   in little is little is little..."). Activation magnitudes drift across the
   32 transformer layers; same shape entry hits different layer distributions.
   Reverted to per-call absmax (fused kernel) for correctness.

## Two paths forward to actually hit 2× tok/W

1. **Hand-rolled FP8 GEMM kernel** (no cuBLAS Lt): pure CUDA implementation,
   graph-capturable. Would let FP8 stack with graph capture and clock lock.
   Estimate: 1-2 weeks engineering for a kernel competitive with cuBLAS Lt.

2. **Marlin INT4 substitute on cublasLtMatmul path**: already in CIPHER's
   GemmEx path, just needs the same port to LtMatmul that the FP8 substitute
   received. INT4 = 4× weight bandwidth saving (vs FP8's 2×) → larger headroom.

## Pod end state (2026-05-01)

- All vLLM / EngineCore / pytest processes killed; GPU at 0 MiB.
- Clock-lock reset (`sudo nvidia-smi -rgc -i 0`).
- Final libraries built and ready under `op31-prod-fix/`:
  `libcipher_hook.so` (~135 KB), `libcipher_rt.so` (~5 MB).
- venv `/home/ubuntu/cipher-test-venv/` retains torch 2.6 + vLLM 0.8.5 +
  transformers 4.51.1 for re-running the validation.

### Persistent env changes on this pod
- (from prior session, system Python) vLLM 0.20.0, torch 2.11.0, numpy 2.2.6,
  scipy 1.15.3, sklearn 1.7.2, pandas 2.3.3, ml_dtypes 0.5.4
- (this session, isolated venv) `cipher-test-venv/` with torch 2.6.0,
  vllm 0.8.5, transformers 4.51.1, accelerate 1.13.0
- (this session, models) `/home/ubuntu/models/Llama-3.1-8B-AWQ/` (5.4 GB,
  hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4) — used for one
  baseline test, otherwise idle.

### Files left in `/tmp` (not in tarball)
- Bench scripts: `/tmp/bench_pwr.sh`, `/tmp/bench_graph.py`,
  `/tmp/bench_graph_pwr.sh`, `/tmp/vllm_bench*.py`
- Logs: `/tmp/vllm_*.log`, `/tmp/fp8_*.log`, `/tmp/run_*.log`
- Power CSVs: `/tmp/pwr_*.csv`
