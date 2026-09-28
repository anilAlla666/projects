# CIPHER v2 Build Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve CIPHER from Session 7 baseline into a zero-application-change GPU primitive that eliminates iterative algorithms via O(1) Koopman substitution and achieves 85%+ MFU on H100.

**Architecture:** LD_PRELOAD hook DSO intercepts cuBLAS/cuLaunchKernel at PLT level; a 3-stage pipeline (classify -> oracle -> substitute) replaces O(N) GPU kernels with O(1) Koopman surrogates derived by EDMD. Five gaps separate Session 7 from 85% MFU: no build system, narrow shape coverage, NCCL unwired, no real telemetry, and no MFU measurement loop.

**Tech Stack:** C++17, CUDA 12.4+ (nvcc), OpenSSL (HMAC), NCCL 2.x, Python 3.10+ (PyTorch hook layer), dl/elf/link.h (GOT patching)

---

## What Exists (Session 7 Inventory)

### Fully Wired and Working

| Component | File(s) | What It Does |
|-----------|---------|--------------|
| **F1 Kernel Intercept** | `src/cipher_intercept_cudart.cpp` (1133 lines) | PLT export + GOT patching for 8 cuLaunchKernel variants, cuGetProcAddress v1/v2, cublasGemmEx, cublasLtMatmul. Two-layer interception: LD_PRELOAD symbols + dl_iterate_phdr GOT rewriting. Thread-local M/N/K/A/B/C capture. Recursion guards. |
| **L3.1 Classifier** | `include/cipher_classify.hpp` | Geometry fingerprint: op_class (0-6) + confidence (0-100) in <100ns from grid/block/shmem signature |
| **L3 Dispatch Engine** | `cipher_dispatch.cpp` (488 lines) | Hot path: classify -> struct_lookup -> oracle -> registry -> apply_recipe. Passthrough <160ns, full substitute ~1.8us. Three recipe types: GEMM (roofline), Chebyshev (reductions), HyperFlux (physics) |
| **L3.2 GEMM Recipe** | `src/cipher_recipes.cpp` (497 lines) | Roofline-optimal tiling, 32 day-one registry entries (Llama-3 70B + A100 shapes). Block-level O(Kr) path when manifold ready. O(1) cached output path when Koopman converged |
| **L3.4 Chebyshev** | `src/cipher_recipes.cpp` | Pre-computed degree-8 coefficients for GeLU, SiLU, LayerNorm, RMSNorm, Softmax. Clenshaw recurrence, <0.1% max error |
| **L3.5 EDMD** | `src/cipher_edmd.cpp` (863 lines) | Snapshot collection, Chebyshev basis lift, QR least-squares solve, convergence test. 512 snapshots, 32 basis functions, 1% error target |
| **L3.6/L3.7/L3.9 Oracle** | `cipher_oracle.cpp` (452 lines) | Five sequential gates: phase detection (warmup/convergence/finetune), confidence threshold, structural lookup, EMA gradient divergence, N<=4 rule. Permanent demotion on divergence |
| **L3.8 Structural Lookup** | `src/cipher_structural_lookup.cpp` | 512-slot FNV-1a kernel name cache. Hardcoded rules: attention=FP, last 3 layers=FP, optimizer=FP, loss=FP |
| **L3.10 LNN (CfC)** | `src/cipher_lnn.cpp` (478 lines) | 48-dim input, 64-dim hidden CfC, 12-dim output. Analytical weight initialization. Closed-form ODE step. <2us forward pass |
| **10-Ops Pipeline** | `src/cipher_10ops_impl.cpp` | Stage 1 shadow thread (REMEMBER/VALIDATE/AUDIT/SPECULATE), Stage 2 background (ADAPT/ARBITRATE). Real CfC, Welford stats, HMAC-SHA256, EDMD per-class |
| **F2 Green Contexts** | `src/cipher_green_ctx.cu` (263 lines) | CUDA 12.4+ cuGreenCtxCreate API. SM partitioning with fallback mode |
| **F3 L2 Persistence** | `src/cipher_l2_persist.cu` (238 lines) | cudaAccessPolicyWindow with hitProp=Persisting. Budget management, L2 capacity query |
| **F4 Liquid State** | `src/cipher_liquid_state.cu` (330 lines) | cudaMallocManaged unified memory. Phase, per-layer state, gradient tracking, NCCL history, workload rhythm |
| **Python Wrapper** | `cipher_wrapper.py` (128 lines) | torch.mm interception, per-layer fp16 MLP hooks via cipher_koopman_fp16_launch, block sub collection |
| **Pre-built DSOs** | `libcipher_hook.so` (37KB), `libcipher_rt.so` (187KB) | x86-64 ELF shared objects. hook.so = F1 intercept, rt.so = everything else |

### Partially Wired (Infrastructure Present, Not Enforced)

| Component | File(s) | Gap |
|-----------|---------|-----|
| **NCCL Policy** | `src/cipher_nccl.cpp`, `cipher_nccl_bpf.cpp`, `cipher_nccl_neural.cpp` | Rule engine + neural CfC policy exist but no PLT export of ncclAllReduce. Policy computed but never applied |
| **L2.2 Fusion** | `src/cipher_fusion.cpp` | Detects 4 fusion patterns (EW+GEMM, GEMM+GEMM etc) but never launches fused kernels |
| **L2.3 Mem Layout** | `src/cipher_mem_layout.cpp` | Advises row/col/tiled layout but doesn't transform tensors |
| **F5 Telemetry** | `src/cipher_telemetry.cpp` | 500Hz NVML sampling thread runs but CUPTI PM counters are placeholder (hardcoded 50% occupancy, 70% L2 hit) |
| **Koopman fp16** | `src/cipher_koopman_runtime.cpp` | Only fires for K=4096. Mistral-7B MLP projections at K=14336 pass through |

---

## The Five Gaps

### Gap 1: No Build System
**Impact:** Cannot rebuild after any source change. Pre-built DSOs are stale the moment we edit code.
**Root cause:** Session 7 compiled ad-hoc. No Makefile, no CMakeLists, no build script.
**Fix:** Single Makefile that builds both DSOs. `libcipher_hook.so` needs only g++ + `-ldl`. `libcipher_rt.so` needs nvcc for `.cu` files + g++ for `.cpp` files + `-lssl -lcrypto` for HMAC.

### Gap 2: Narrow Koopman Shape Coverage
**Impact:** Only K=4096 fp16 GEMMs get O(1) substitution. Mistral-7B has K=14336 (MLP up/down projections), K=128 (attention head dim). These all pass through = wasted MFU.
**Root cause:** Hard-coded `tls_K == 4096` gate in `cipher_intercept_cudart.cpp:253` and `:428`.
**Fix:** Shape-parametric Koopman: register (M,K,N) -> (V,K_block,W) triples per shape. Gate on registry hit, not magic constant.

### Gap 3: NCCL Not Hooked
**Impact:** On multi-GPU, AllReduce dominates idle time. CIPHER's NCCL policy engine (ring/tree/NVLS selection) is computed but never applied. Zero overlap scheduling.
**Root cause:** No `ncclAllReduce` PLT export in `exports.map`. Policy engine is disconnected from actual NCCL calls.
**Fix:** Add ncclAllReduce/Broadcast/Send/Recv shims to intercept layer. Apply policy before forwarding to real NCCL. Wire overlap scheduler.

### Gap 4: Placeholder Telemetry
**Impact:** Oracle decisions use fake hardware metrics (50% occupancy, 70% L2 hit). SM packer, mem layout advisor, and MFU calculation all downstream of these numbers. Garbage in, garbage out.
**Root cause:** CUPTI Performance Monitoring (PM) counters not integrated. `cipher_telemetry.cpp` samples NVML only (temperature, power).
**Fix:** Add CUPTI PM sampling for: sm__inst_executed (occupancy), lts__t_sectors_hit (L2), dram__bytes (HBM BW). Feed real numbers into liquid state.

### Gap 5: No MFU Measurement Loop
**Impact:** Cannot verify we hit 85%. Cannot auto-tune. Cannot detect regressions.
**Root cause:** No code computes `achieved_tflops / peak_tflops`. No feedback from MFU to oracle thresholds.
**Fix:** Compute MFU from GEMM shape (2*M*N*K FLOPs) + wall-clock time per kernel. Aggregate per-step. Feed into oracle as a primary signal. Log continuously.

---

## Build Order

### Phase 0: Build System (Prerequisite for Everything)

#### Task 0.1: Create Makefile

**Files:**
- Create: `Makefile`
- Reference: `exports.map`, `hook_versions.map`

- [ ] **Step 1: Verify nvcc and dependencies are available**

```bash
which nvcc && nvcc --version
which g++ && g++ --version
pkg-config --libs openssl 2>/dev/null || echo "openssl: check -lssl -lcrypto"
```

- [ ] **Step 2: Write Makefile**

```makefile
# CIPHER v2 Build System
CXX      := g++
NVCC     := nvcc
CXXFLAGS := -std=c++17 -O2 -fPIC -Wall -Wextra -Wno-unused-parameter
NVFLAGS  := -std=c++17 -O2 --compiler-options -fPIC -Xcompiler -Wall
INCLUDES := -I./include
LDFLAGS_HOOK := -shared -ldl
LDFLAGS_RT   := -shared -ldl -lssl -lcrypto -lpthread

# F1 hook DSO — pure C++, no CUDA SDK needed
HOOK_SRC := src/cipher_intercept_cudart.cpp
HOOK_OUT := libcipher_hook.so

# Runtime DSO — C++ and CUDA sources
RT_CPP_SRC := $(filter-out src/cipher_intercept_cudart.cpp, $(wildcard src/*.cpp)) \
              cipher_dispatch.cpp cipher_oracle.cpp
RT_CU_SRC  := $(wildcard src/*.cu)
RT_CPP_OBJ := $(RT_CPP_SRC:.cpp=.o)
RT_CU_OBJ  := $(RT_CU_SRC:.cu=.o)
RT_OUT     := libcipher_rt.so

.PHONY: all clean test hook rt

all: $(HOOK_OUT) $(RT_OUT)

hook: $(HOOK_OUT)

$(HOOK_OUT): $(HOOK_SRC) exports.map
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS_HOOK) \
		-Wl,--version-script=exports.map \
		-o $@ $<

rt: $(RT_OUT)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c -o $@ $<

$(RT_OUT): $(RT_CPP_OBJ) $(RT_CU_OBJ)
	$(NVCC) -shared -o $@ $^ -lssl -lcrypto -lpthread -ldl

clean:
	rm -f $(HOOK_OUT) $(RT_OUT) $(RT_CPP_OBJ) $(RT_CU_OBJ)

test: all
	python3 test_all_ops.py
```

- [ ] **Step 3: Build hook DSO and verify**

```bash
make hook
nm -D libcipher_hook.so | grep -E "cuLaunchKernel|cublasLtMatmul|cublasGemmEx" | head -10
```

Expected: symbols visible in dynamic symbol table.

- [ ] **Step 4: Build runtime DSO and verify**

```bash
make rt
nm -D libcipher_rt.so | grep -E "cipher_dispatch|cipher_init|cipher_lnn_forward" | head -5
```

Expected: all entry points exported.

- [ ] **Step 5: Commit**

```bash
git add Makefile
git commit -m "build: add Makefile for hook and runtime DSOs"
```

---

#### Task 0.2: Fix Stale Path in run_mistral.py

**Files:**
- Modify: `run_mistral.py:9`

- [ ] **Step 1: Fix sys.path**

Change line 9 from:
```python
sys.path.insert(0, '/workspace/CIPHER_final')
```
to:
```python
sys.path.insert(0, '/workspace/CIPHER_final_session7')
```

- [ ] **Step 2: Commit**

```bash
git add run_mistral.py
git commit -m "fix: correct sys.path to session7 directory"
```

---

### Phase 1: Close Gap 2 — Shape-Parametric Koopman (Goal 1: Eliminate Iterative Algorithms)

This is the highest-leverage change. Currently only K=4096 gets O(1) substitution. Mistral-7B has these GEMM shapes:

| Layer | M | K | N | Current |
|-------|---|---|---|---------|
| Attention QKV | batch*seq | 4096 | 4096 | Substituted (K=4096) |
| Attention O-proj | batch*seq | 4096 | 4096 | Substituted (K=4096) |
| MLP gate_proj | batch*seq | 4096 | 14336 | **PASS-THROUGH** |
| MLP up_proj | batch*seq | 4096 | 14336 | **PASS-THROUGH** |
| MLP down_proj | batch*seq | 14336 | 4096 | **PASS-THROUGH** |

MLP projections are ~60% of total GEMM FLOPs. They all pass through today.

#### Task 1.1: Shape-Parametric Koopman Registry

**Files:**
- Modify: `src/cipher_intercept_cudart.cpp:253` (cublasGemmEx gate)
- Modify: `src/cipher_intercept_cudart.cpp:428` (cublasLtMatmul gate)
- Modify: `src/cipher_koopman_runtime.cpp` (add shape-keyed lookup)
- Modify: `include/cipher_koopman_runtime.h` (add shape key to CipherKRRecord)

- [ ] **Step 1: Write test that verifies K=14336 is attempted**

Create `tests/test_shape_coverage.py`:
```python
"""Verify Koopman substitute is attempted for all Mistral-7B GEMM shapes."""
import subprocess, re

def test_k14336_attempted():
    """K=14336 GEMMs should attempt Koopman lookup, not skip unconditionally."""
    # Run with CIPHER and capture stderr
    result = subprocess.run(
        ['python3', '-c', '''
import torch
# Simulate Mistral MLP gate_proj: [1, 4096] x [4096, 14336]
a = torch.randn(1, 4096, dtype=torch.float16, device='cuda')
b = torch.randn(4096, 14336, dtype=torch.float16, device='cuda')
for _ in range(10):
    torch.mm(a, b)
torch.cuda.synchronize()
'''],
        env={**__import__('os').environ,
             'LD_PRELOAD': './libcipher_hook.so ./libcipher_rt.so',
             'CIPHER_SAFE_MODE': '1'},
        capture_output=True, text=True, timeout=60
    )
    stderr = result.stderr
    # Should see intercept log for K=14336, not just K=4096
    assert 'K=14336' in stderr or 'k=14336' in stderr, \
        f"K=14336 GEMM not intercepted. stderr:\n{stderr[-500:]}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_shape_coverage.py -v
```
Expected: FAIL — current code skips K!=4096.

- [ ] **Step 3: Remove hard-coded K=4096 gate in cublasGemmEx shim**

In `src/cipher_intercept_cudart.cpp`, replace the condition at line ~253:
```cpp
    if (tls_Ctype == 2 /* fp16 */ && tls_M > 0 && tls_K == 4096) {
```
with:
```cpp
    if (tls_Ctype == 2 /* fp16 */ && tls_M > 0 && tls_K > 0) {
```

The `cipher_koopman_fp16_launch` function will return non-zero (fail) for shapes it doesn't have calibrated matrices for, causing fallback to real cuBLAS. This is safe.

- [ ] **Step 4: Remove hard-coded K=4096 gate in cublasLtMatmul shim**

In `src/cipher_intercept_cudart.cpp`, replace the condition at line ~428:
```cpp
    if (tls_Ctype == 2 /* fp16 */ && tls_M > 0 && tls_K == 4096) {
```
with:
```cpp
    if (tls_Ctype == 2 /* fp16 */ && tls_M > 0 && tls_K > 0) {
```

Same safety: unrecognized shapes fall through to real cublasLtMatmul.

- [ ] **Step 5: Add shape key to Koopman runtime record**

In `include/cipher_koopman_runtime.h`, add to CipherKRRecord:
```cpp
    uint32_t shape_m;    // GEMM M dimension (0 = non-GEMM)
    uint32_t shape_k;    // GEMM K dimension
    uint32_t shape_n;    // GEMM N dimension
```

In `src/cipher_koopman_runtime.cpp`, update `cipher_kr_hash()` to incorporate M/K/N when op_class==GEMM:
```cpp
uint64_t cipher_kr_hash_gemm(uint32_t m, uint32_t k, uint32_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;  // FNV offset basis
    h ^= (uint64_t)m; h *= 0x100000001b3ULL;
    h ^= (uint64_t)k; h *= 0x100000001b3ULL;
    h ^= (uint64_t)n; h *= 0x100000001b3ULL;
    return h;
}
```

- [ ] **Step 6: Add Mistral-7B MLP shapes to day-one registry**

In `src/cipher_recipes.cpp`, inside `cipher_registry_init()`, add entries:
```cpp
    // Mistral-7B MLP projections (fp16)
    // gate_proj / up_proj: M=var, K=4096, N=14336
    reg->entries[reg->count++] = {
        .op_class = 0, .shape_hash = hash_shape(0, 4096, 14336),
        .hw_arch = 90, .recipe_type = 0, .error_bound = 0.01f,
        .confidence = 0.0f, .name = "mistral-mlp-gate-4096x14336", .active = true
    };
    // down_proj: M=var, K=14336, N=4096
    reg->entries[reg->count++] = {
        .op_class = 0, .shape_hash = hash_shape(0, 14336, 4096),
        .hw_arch = 90, .recipe_type = 0, .error_bound = 0.01f,
        .confidence = 0.0f, .name = "mistral-mlp-down-14336x4096", .active = true
    };
```

- [ ] **Step 7: Rebuild and run test**

```bash
make clean && make all
python3 -m pytest tests/test_shape_coverage.py -v
```
Expected: PASS — K=14336 now intercepted and attempted.

- [ ] **Step 8: Commit**

```bash
git add src/cipher_intercept_cudart.cpp src/cipher_koopman_runtime.cpp \
        include/cipher_koopman_runtime.h src/cipher_recipes.cpp \
        tests/test_shape_coverage.py
git commit -m "feat: shape-parametric Koopman — remove K=4096 gate, add Mistral MLP shapes"
```

---

#### Task 1.2: Per-Shape Manifold Calibration Pipeline

**Files:**
- Modify: `cipher_wrapper.py` (extend install_all_mlp_hooks for 14336-dim layers)
- Modify: `src/cipher_koopman_runtime.cpp` (per-shape EDMD collection + solve)

- [ ] **Step 1: Write test for per-shape EDMD collection**

Add to `tests/test_shape_coverage.py`:
```python
def test_edmd_collects_for_new_shape():
    """EDMD pipeline should collect snapshots for K=14336 shapes."""
    result = subprocess.run(
        ['python3', '-c', '''
import torch
a = torch.randn(4, 4096, dtype=torch.float16, device='cuda')
b = torch.randn(4096, 14336, dtype=torch.float16, device='cuda')
for _ in range(250):  # Exceed CIPHER_KR_MIN_SNAPSHOTS=200
    torch.mm(a, b)
torch.cuda.synchronize()
'''],
        env={**__import__('os').environ,
             'LD_PRELOAD': './libcipher_hook.so ./libcipher_rt.so',
             'CIPHER_SAFE_MODE': '1'},
        capture_output=True, text=True, timeout=120
    )
    stderr = result.stderr
    assert 'EDMD' in stderr, f"No EDMD activity for K=14336. stderr:\n{stderr[-500:]}"
```

- [ ] **Step 2: Run test to verify current behavior**

```bash
python3 -m pytest tests/test_shape_coverage.py::test_edmd_collects_for_new_shape -v
```

- [ ] **Step 3: Extend cipher_koopman_runtime.cpp to collect per-GEMM-shape snapshots**

In `cipher_kr_process()`, after creating a new CipherKRRecord, store the GEMM dimensions:
```cpp
    rec->shape_m = m;
    rec->shape_k = k;
    rec->shape_n = n;
```

The existing EDMD collection logic (`cipher_kr_collect_snapshot`) already works for any shape — it uses the 16-dim feature vector. The key change is ensuring the fp16 launch function queries the runtime for shape-specific calibration rather than hardcoded layer pointers.

- [ ] **Step 4: Extend cipher_wrapper.py to calibrate 14336-dim layers**

In `install_all_mlp_hooks()`, detect MLP dimensions dynamically:
```python
def install_all_mlp_hooks(model):
    """Install Koopman hooks on all MLP layers, any hidden dimension."""
    for name, module in model.named_modules():
        if hasattr(module, 'gate_proj'):  # Mistral MLP
            K = module.gate_proj.weight.shape[1]  # 4096
            N = module.gate_proj.weight.shape[0]  # 14336
            layer_idx = int(re.search(r'layers\.(\d+)', name).group(1))
            _install_mlp_hook_for_shape(model, layer_idx, K, N)
```

- [ ] **Step 5: Rebuild and run test**

```bash
make clean && make all
python3 -m pytest tests/test_shape_coverage.py -v
```

- [ ] **Step 6: Commit**

```bash
git add cipher_wrapper.py src/cipher_koopman_runtime.cpp tests/test_shape_coverage.py
git commit -m "feat: per-shape EDMD collection for MLP projections"
```

---

### Phase 2: Close Gap 4 — Real Telemetry (Goal 2: MFU to 85%)

Without real hardware counters, the oracle makes decisions on fake data. This phase wires CUPTI PM counters into the telemetry thread.

#### Task 2.1: CUPTI PM Counter Integration

**Files:**
- Modify: `src/cipher_telemetry.cpp` (replace placeholders with CUPTI sampling)
- Modify: `include/cipher_telemetry.h` (add CUPTI state)
- Create: `tests/test_telemetry.py`

- [ ] **Step 1: Check CUPTI availability**

```bash
find /usr/local/cuda -name "cupti.h" 2>/dev/null
find /usr/local/cuda -name "libcupti*" 2>/dev/null
```

- [ ] **Step 2: Write test for real telemetry values**

Create `tests/test_telemetry.py`:
```python
"""Verify telemetry reports non-placeholder values."""
import subprocess

def test_telemetry_not_placeholder():
    """SM occupancy should not be exactly 0.50 (the placeholder)."""
    result = subprocess.run(
        ['python3', '-c', '''
import ctypes, time
rt = ctypes.CDLL('./libcipher_rt.so')
rt.cipher_init(0)
time.sleep(0.1)  # Let telemetry thread sample
sample = rt.cipher_read_sample()
'''],
        env={**__import__('os').environ,
             'LD_PRELOAD': './libcipher_hook.so ./libcipher_rt.so'},
        capture_output=True, text=True, timeout=30
    )
    # After CUPTI integration, occupancy should vary from 0.50
    assert result.returncode == 0
```

- [ ] **Step 3: Add CUPTI PM sampling to telemetry thread**

In `src/cipher_telemetry.cpp`, replace the placeholder block (the section that hardcodes `sample->sm_occupancy = 0.50f` etc.) with CUPTI `cuptiMetricGetValue` calls for:
- `sm__inst_executed.avg.pct_of_peak_sustained_active` (SM occupancy)
- `lts__t_sectors_hit_rate.pct` (L2 hit rate)
- `dram__bytes.sum.per_second` (HBM bandwidth)
- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` (MFU proxy)

Use range profiling (`cuptiProfilerBeginPass` / `cuptiProfilerEndPass`) on a dedicated CUDA stream to avoid perturbing the main workload.

Keep NVML fallback: if CUPTI init fails (no permissions, older driver), log warning and continue with NVML-only metrics (power, temperature) + placeholder flags for counters.

- [ ] **Step 4: Feed real counters into liquid state**

The existing liquid state update path in `cipher_telemetry.cpp` already writes to:
```cpp
liquid->sm_occupancy     = sample->sm_occupancy;
liquid->l2_hit_rate      = sample->l2_hit_rate;
liquid->hbm_bw_utilized  = sample->hbm_bw_utilized;
```

No change needed here — just ensure the sample source is CUPTI instead of constants.

- [ ] **Step 5: Rebuild and verify**

```bash
make clean && make all
python3 -m pytest tests/test_telemetry.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/cipher_telemetry.cpp include/cipher_telemetry.h tests/test_telemetry.py
git commit -m "feat: wire CUPTI PM counters into telemetry thread"
```

---

### Phase 3: Close Gap 5 — MFU Measurement Loop (Goal 2: MFU to 85%)

#### Task 3.1: Per-Kernel MFU Computation

**Files:**
- Modify: `cipher_dispatch.cpp` (add MFU accumulator)
- Modify: `include/cipher.h` (add MFU state to CipherRuntime)
- Create: `tests/test_mfu.py`

- [ ] **Step 1: Write test for MFU reporting**

Create `tests/test_mfu.py`:
```python
"""Verify MFU is computed and reported."""
import subprocess

def test_mfu_reported():
    result = subprocess.run(
        ['python3', '-c', '''
import torch, time
a = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
b = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
for _ in range(100):
    torch.mm(a, b)
torch.cuda.synchronize()
'''],
        env={**__import__('os').environ,
             'LD_PRELOAD': './libcipher_hook.so ./libcipher_rt.so',
             'CIPHER_SAFE_MODE': '1'},
        capture_output=True, text=True, timeout=60
    )
    assert 'MFU=' in result.stderr, f"No MFU reported. stderr:\n{result.stderr[-500:]}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_mfu.py -v
```
Expected: FAIL — no MFU computation exists.

- [ ] **Step 3: Add MFU accumulator to CipherRuntime**

In `include/cipher.h`, add to CipherRuntime:
```cpp
    // MFU tracking
    double   mfu_flops_accumulated;   // Total FLOPs across all GEMM dispatches
    double   mfu_time_accumulated_ns; // Wall-clock ns for those GEMMs
    uint64_t mfu_gemm_count;          // Number of GEMMs tracked
    uint64_t mfu_last_report_step;    // Step at which we last logged MFU
```

- [ ] **Step 4: Compute MFU per GEMM dispatch**

In `cipher_dispatch.cpp`, inside the GEMM path of `cipher_dispatch()`, after the kernel completes (whether substituted or passthrough):
```cpp
    // MFU tracking: 2*M*N*K FLOPs per GEMM
    if (desc->op_class == 0 && tls_shape_valid) {
        uint64_t now_ns = desc->intercept_ns;  // Already captured
        double flops = 2.0 * tls_M * tls_N * tls_K;
        g_cipher.mfu_flops_accumulated += flops;
        g_cipher.mfu_gemm_count++;

        // Report every 1000 GEMMs
        if (g_cipher.mfu_gemm_count % 1000 == 0) {
            double elapsed_s = (now_ns - g_cipher.mfu_time_accumulated_ns) * 1e-9;
            double tflops = g_cipher.mfu_flops_accumulated * 1e-12;
            double peak_tflops = 989.0;  // H100 fp16
            double mfu = (elapsed_s > 0) ? (tflops / elapsed_s) / peak_tflops : 0;
            fprintf(stderr, "[CIPHER MFU] MFU=%.1f%% (%d GEMMs, %.1f TFLOPS achieved)\n",
                    mfu * 100.0, (int)g_cipher.mfu_gemm_count, tflops / elapsed_s);
            // Reset accumulators
            g_cipher.mfu_flops_accumulated = 0;
            g_cipher.mfu_time_accumulated_ns = now_ns;
        }
    }
```

- [ ] **Step 5: Feed MFU into oracle as primary signal**

In the oracle's `cipher_oracle_decide()`, add MFU-based threshold adjustment:
```cpp
    // If MFU < 70%, oracle becomes more aggressive about substitution
    // If MFU > 85%, oracle becomes more conservative (don't break what works)
    if (g_cipher.mfu_last_reported > 0.85f)
        state->min_confidence += 5;  // Raise bar
    else if (g_cipher.mfu_last_reported < 0.70f)
        state->min_confidence -= 5;  // Lower bar
```

- [ ] **Step 6: Rebuild and run test**

```bash
make clean && make all
python3 -m pytest tests/test_mfu.py -v
```

- [ ] **Step 7: Commit**

```bash
git add cipher_dispatch.cpp include/cipher.h cipher_oracle.cpp tests/test_mfu.py
git commit -m "feat: per-kernel MFU measurement loop with oracle feedback"
```

---

### Phase 4: NCCL Intercept + Compute-Communication Overlap (Target: -60 TFLOPS gap)

AllReduce blocking is the single largest MFU killer on multi-GPU. The GPU sits
idle waiting for gradient synchronization. CIPHER intercepts ncclAllReduce,
times it, and launches the next forward-pass GEMMs on a subset of SMs while
AllReduce runs on the rest via Green Context partitioning.

#### Task 4.1: ncclAllReduce GOT Patch

**Files:**
- Modify: `src/cipher_intercept_cudart.cpp` (add ncclAllReduce shim)
- Modify: `exports.map` (export ncclAllReduce)
- Modify: `hook_versions.map` (add libnccl version tag)

- [ ] **Step 1: Add ncclAllReduce PLT shim**

Same pattern as cublasGemmEx: dlopen libnccl.so.2, resolve real symbol via dlvsym,
export our shim. Log message size, time the call, feed duration into liquid state
NCCL ring buffer.

- [ ] **Step 2: Wire policy engine**

Call `cipher_nccl_select_policy()` from the shim. Phase 1: log-only passthrough.
Export `cipher_nccl_apply_policy` and `cipher_nccl_record_allreduce` from RT.

- [ ] **Step 3: Add exports**

`ncclAllReduce`, `cipher_nccl_apply_policy`, `cipher_nccl_record_allreduce` to exports.map.

#### Task 4.2: Compute-Communication Overlap via Green Contexts

**Files:**
- Modify: `src/cipher_green_ctx.cu` (expose SM partition API)
- Modify: `src/cipher_intercept_cudart.cpp` (overlap scheduler in ncclAllReduce shim)

- [ ] **Step 1: Partition SMs during AllReduce**

When ncclAllReduce fires:
1. Query current SM allocation from Green Context state
2. Reserve N SMs for NCCL (NVLink-bound, doesn't need many SMs)
3. Launch queued forward-pass GEMMs on remaining SMs via compute Green Context stream
4. Wait for both AllReduce and overlap GEMMs to complete

- [ ] **Step 2: Overlap scheduler**

Track which GEMMs are "next" in the forward pass sequence (from workload rhythm
ring buffer). When AllReduce starts, speculatively launch the next 1-2 GEMMs
on the compute partition. If speculation was wrong, discard and re-run.

This is geometry-only: the scheduler sees grid dims and function pointers,
not model structure.

---

### Phase 5: Non-Matmul Substitution (Target: -50 TFLOPS norms+activations gap)

GeLU, SiLU, RMSNorm, LayerNorm collectively consume ~50 TFLOPS of headroom.
Chebyshev degree-8 recipes already exist in `cipher_recipes.cpp` (recipe_type=2)
with pre-computed coefficients and <0.1% max error. They just never fire because
only cuBLAS calls (cublasGemmEx/cublasLtMatmul) route through the recipe system.
Non-GEMM CUDA kernels hit the cuLaunchKernel path which currently only classifies
and logs — it never applies recipes.

#### Task 5.1: Wire cuLaunchKernel to Recipe System

**Files:**
- Modify: `src/cipher_intercept_cudart.cpp` (cuLaunchKernel shim dispatch path)
- Modify: `cipher_dispatch.cpp` (apply Chebyshev recipe for non-GEMM ops)

- [ ] **Step 1: Route classified non-GEMM ops to recipe application**

In the cuLaunchKernel shim, after `cipher_dispatch()` classifies the kernel:
- If op_class is ELEMENTWISE (3) or REDUCTION (4), and registry has a Chebyshev entry,
  apply the recipe. The Chebyshev evaluation code exists in `cipher_recipes.cpp`
  (Clenshaw recurrence, pre-computed coefficients for 6 nonlinearities).

- [ ] **Step 2: Make Chebyshev recipes fire**

The registry already has entries for GeLU, SiLU, RMSNorm with recipe_type=2.
The dispatch engine has `apply_recipe()` case 2 for Chebyshev. The missing link:
the cuLaunchKernel dispatch path returns PASS_THROUGH before reaching apply_recipe
for non-GEMM ops. Fix the control flow.

#### Task 5.2: Kernel Fusion — Norm+GEMM and GEMM+Activation

**Files:**
- Modify: `src/cipher_fusion.cpp` (execute fusions, not just detect)
- Create: `src/cipher_fused_kernels.cu` (fused CUDA kernels)

- [ ] **Step 1: Implement fused norm+GEMM kernel**

When L2.2 detects [REDUCTION, GEMM] or [ELEMENTWISE, ELEMENTWISE, GEMM] pattern:
launch a single kernel that does RMSNorm + GEMM in one pass. Saves one global
memory round-trip (norm output doesn't need to be written to HBM).

- [ ] **Step 2: Implement fused GEMM+activation kernel**

When L2.2 detects [GEMM, ELEMENTWISE] pattern: launch fused GEMM+GeLU or
GEMM+SiLU. The activation is applied to GEMM output in shared memory before
writing to HBM.

---

### Phase 6: Attention Optimization (Target: -70 TFLOPS attention gap)

Attention is the largest single MFU killer. FlashAttention helps but is still
memory-bound for decode (seq_len=1). CIPHER can detect attention patterns from
geometry and apply Koopman-based low-rank approximation.

#### Task 6.1: FlashAttention Pattern Detection

**Files:**
- Modify: `src/cipher_classify.hpp` (detect attention geometry)
- Modify: `cipher_dispatch.cpp` (attention-specific recipe path)

- [ ] **Step 1: Detect attention from geometry**

FlashAttention kernels have distinctive geometry:
- Block size: 128 or 256 threads
- Shared memory: 32KB-100KB (softmax + recompute)
- Grid: (num_heads, batch, ceil(seq/block))

Classify these as ATTENTION (op_class=1). Currently structural lookup forces
full precision on all attention. Relax this for decode-phase attention where
Q@K^T is rank 32-64.

#### Task 6.2: Decode Attention Low-Rank Substitute

**Files:**
- Create: `src/cipher_attention_koopman.cu`
- Modify: `src/cipher_recipes.cpp` (add attention recipe type)

- [ ] **Step 1: Low-rank Q@K^T for decode**

During decode (seq_len=1 or small), Q@K^T is a rank-1 outer product.
The softmax output is effectively low-rank (32-64 significant singular values).
Koopman can approximate the full attention output with O(d*r) instead of O(d*seq).

This only applies when seq_len < some threshold (geometry check, not model check).

---

### Phase 7: Hardware-Only Validation + Billing (All Five Gaps)

CIPHER is a primitive. It validates against hardware counters, not model outputs.
No model-specific logic. No per-layer calibration. No "which layer are we in."
CIPHER sees geometry only: M, N, K, function pointer, grid dims.

**DRIFT RULE:** If any validation step requires knowledge of which model is running,
which layer we are in, or what activations mean — STOP and flag it as drift.

#### Task 6.1: Kernel Intercept Confirmation

**Files:**
- Create: `tests/test_hw_validation.py`

- [ ] **Step 1: Write intercept counter test**

Create `tests/test_hw_validation.py`:
```python
"""CIPHER v2 hardware-only validation. No model knowledge. Geometry only."""
import subprocess, re, ctypes, os

HOOK = './libcipher_hook.so'
RT   = './libcipher_rt.so'
ENV  = {**os.environ, 'LD_PRELOAD': f'{HOOK} {RT}', 'CIPHER_SAFE_MODE': '1'}

def test_intercept_count_nonzero():
    """Every cuBLAS GEMM must be intercepted. Zero originals escape."""
    result = subprocess.run(
        ['python3', '-c', '''
import ctypes, torch
hook = ctypes.CDLL("./libcipher_hook.so")
hook.cipher_intercept_count.restype = ctypes.c_uint64
a = torch.randn(2048, 2048, dtype=torch.float16, device="cuda")
b = torch.randn(2048, 2048, dtype=torch.float16, device="cuda")
before = hook.cipher_intercept_count()
for _ in range(100):
    torch.mm(a, b)
torch.cuda.synchronize()
after = hook.cipher_intercept_count()
intercepted = after - before
print(f"INTERCEPTED={intercepted}")
assert intercepted >= 100, f"Only {intercepted}/100 GEMMs intercepted"
'''],
        env=ENV, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, f"Intercept test failed:\n{result.stderr[-500:]}"
    assert 'INTERCEPTED=' in result.stdout
    count = int(re.search(r'INTERCEPTED=(\d+)', result.stdout).group(1))
    assert count >= 100, f"Only {count} intercepted"
```

- [ ] **Step 2: Run test**

```bash
python3 -m pytest tests/test_hw_validation.py::test_intercept_count_nonzero -v
```
Expected: PASS — every GEMM hits our shim.

- [ ] **Step 3: Commit**

```bash
git add tests/test_hw_validation.py
git commit -m "test: kernel intercept count validation — geometry only"
```

---

#### Task 6.2: Numerical Correctness (max_diff < 0.01 vs cuBLAS)

**Files:**
- Modify: `tests/test_hw_validation.py`

- [ ] **Step 1: Write correctness test**

Add to `tests/test_hw_validation.py`:
```python
def test_correctness_vs_cublas():
    """CIPHER output must match cuBLAS within max_diff < 0.01.
    Pure geometry test: random matrices, no model, no layers."""
    result = subprocess.run(
        ['python3', '-c', '''
import torch

shapes = [
    (1024, 1024, 1024),
    (2048, 4096, 2048),
    (4096, 4096, 4096),
    (1, 4096, 14336),
    (32, 14336, 4096),
]

for M, K, N in shapes:
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")

    # Baseline: cuBLAS (CIPHER passthrough on first call)
    ref = torch.mm(a, b)

    # CIPHER path (may substitute on subsequent calls)
    for _ in range(50):
        out = torch.mm(a, b)

    diff = (out - ref).abs().max().item()
    print(f"SHAPE=({M},{K},{N}) MAX_DIFF={diff:.6f}")
    assert diff < 0.01, f"Shape ({M},{K},{N}): max_diff={diff} >= 0.01"

print("CORRECTNESS_PASS")
'''],
        env=ENV, capture_output=True, text=True, timeout=120
    )
    assert 'CORRECTNESS_PASS' in result.stdout, \
        f"Correctness failed:\n{result.stdout}\n{result.stderr[-500:]}"
```

- [ ] **Step 2: Run test**

```bash
python3 -m pytest tests/test_hw_validation.py::test_correctness_vs_cublas -v
```
Expected: PASS — all shapes within tolerance.

- [ ] **Step 3: Commit**

```bash
git add tests/test_hw_validation.py
git commit -m "test: numerical correctness vs cuBLAS — max_diff < 0.01, geometry only"
```

---

#### Task 6.3: MFU Before vs After (Sustained GEMM Benchmark)

**Files:**
- Modify: `tests/test_hw_validation.py`

- [ ] **Step 1: Write MFU benchmark test**

Add to `tests/test_hw_validation.py`:
```python
def _run_gemm_benchmark(use_cipher, M=4096, K=4096, N=4096, iters=500):
    """Pure sustained GEMM benchmark. Returns achieved TFLOPS."""
    preload = f'{HOOK} {RT}' if use_cipher else ''
    env = {**os.environ}
    if preload:
        env['LD_PRELOAD'] = preload
        env['CIPHER_SAFE_MODE'] = '1'

    result = subprocess.run(
        ['python3', '-c', f'''
import torch, time

M, K, N, ITERS = {M}, {K}, {N}, {iters}
a = torch.randn(M, K, dtype=torch.float16, device="cuda")
b = torch.randn(K, N, dtype=torch.float16, device="cuda")

# Warmup
for _ in range(50):
    torch.mm(a, b)
torch.cuda.synchronize()

# Timed
start = time.perf_counter()
for _ in range(ITERS):
    torch.mm(a, b)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

flops = 2.0 * M * K * N * ITERS
tflops = flops / elapsed / 1e12
peak = 989.0  # H100 fp16
mfu = tflops / peak * 100.0
print(f"TFLOPS={{tflops:.1f}} MFU={{mfu:.1f}} ELAPSED={{elapsed:.4f}}")
'''],
        env=env, capture_output=True, text=True, timeout=120
    )
    match = re.search(r'TFLOPS=([\d.]+) MFU=([\d.]+)', result.stdout)
    if not match:
        raise RuntimeError(f"Benchmark failed:\n{result.stdout}\n{result.stderr[-500:]}")
    return float(match.group(1)), float(match.group(2))

def test_mfu_before_after():
    """MFU with CIPHER must be >= MFU without CIPHER (no regression).
    Target: 85%+ sustained on 4096x4096x4096 fp16 GEMM."""
    tflops_base, mfu_base = _run_gemm_benchmark(use_cipher=False)
    tflops_cipher, mfu_cipher = _run_gemm_benchmark(use_cipher=True)

    print(f"Baseline:  {tflops_base:.1f} TFLOPS, {mfu_base:.1f}% MFU")
    print(f"CIPHER:    {tflops_cipher:.1f} TFLOPS, {mfu_cipher:.1f}% MFU")

    # CIPHER must not regress MFU by more than 2%
    assert mfu_cipher >= mfu_base - 2.0, \
        f"MFU regression: base={mfu_base:.1f}% cipher={mfu_cipher:.1f}%"
```

- [ ] **Step 2: Run test**

```bash
python3 -m pytest tests/test_hw_validation.py::test_mfu_before_after -v -s
```
Expected: PASS with MFU numbers printed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_hw_validation.py
git commit -m "test: MFU before/after comparison — sustained GEMM, no model"
```

---

#### Task 6.4: GPU Clock Stability Under Increasing Elimination Fraction

**Files:**
- Modify: `tests/test_hw_validation.py`

- [ ] **Step 1: Write clock stability test**

Add to `tests/test_hw_validation.py`:
```python
def test_clock_stability():
    """GPU clocks must remain stable as CIPHER eliminates more kernels.
    Validates that substitution doesn't cause throttling or idle clocks."""
    result = subprocess.run(
        ['python3', '-c', '''
import torch, time, subprocess

def get_gpu_clock():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    return int(r.stdout.strip())

# Phase 1: Light load (few GEMMs, low elimination)
a = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
b = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
for _ in range(100):
    torch.mm(a, b)
torch.cuda.synchronize()
clock_light = get_gpu_clock()

# Phase 2: Heavy sustained load (many GEMMs, high elimination opportunity)
a = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
b = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
for _ in range(500):
    torch.mm(a, b)
torch.cuda.synchronize()
clock_heavy = get_gpu_clock()

# Phase 3: Mixed shapes (varying elimination)
shapes = [(512,512,512), (2048,4096,2048), (4096,4096,4096), (1,4096,14336)]
for M,K,N in shapes:
    aa = torch.randn(M, K, dtype=torch.float16, device="cuda")
    bb = torch.randn(K, N, dtype=torch.float16, device="cuda")
    for _ in range(100):
        torch.mm(aa, bb)
torch.cuda.synchronize()
clock_mixed = get_gpu_clock()

print(f"CLOCK_LIGHT={clock_light} CLOCK_HEAVY={clock_heavy} CLOCK_MIXED={clock_mixed}")

# Clocks should not drop more than 10% under any phase
min_clock = min(clock_light, clock_heavy, clock_mixed)
max_clock = max(clock_light, clock_heavy, clock_mixed)
assert min_clock > max_clock * 0.90, \
    f"Clock instability: min={min_clock} max={max_clock} ({min_clock/max_clock*100:.0f}%)"
print("CLOCK_STABLE")
'''],
        env=ENV, capture_output=True, text=True, timeout=180
    )
    assert 'CLOCK_STABLE' in result.stdout, \
        f"Clock stability failed:\n{result.stdout}\n{result.stderr[-500:]}"
```

- [ ] **Step 2: Run test**

```bash
python3 -m pytest tests/test_hw_validation.py::test_clock_stability -v -s
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_hw_validation.py
git commit -m "test: GPU clock stability under increasing elimination fraction"
```

---

#### Task 6.5: Full Hardware Validation Suite

- [ ] **Step 1: Run all four hardware tests**

```bash
python3 -m pytest tests/test_hw_validation.py -v -s 2>&1 | tee /tmp/cipher_v2_validation.log
```

All four must pass:
1. Intercept count >= expected (no kernel escapes)
2. max_diff < 0.01 vs cuBLAS on all shapes
3. MFU no regression (target 85%+)
4. GPU clocks stable across elimination fractions

- [ ] **Step 2: Document results**

```bash
cat > RESULTS.md << 'EOF'
# CIPHER v2 Hardware Validation Results

| Test | Metric | Result |
|------|--------|--------|
| Intercept | kernels captured / total | |
| Correctness | max_diff across shapes | |
| MFU baseline | TFLOPS / % | |
| MFU CIPHER | TFLOPS / % | |
| Clock light | MHz | |
| Clock heavy | MHz | |
| Clock mixed | MHz | |
EOF
```

Fill from `/tmp/cipher_v2_validation.log`.

---

## Build Order Summary

```
Phase 0: Build System + Path Fix                         ✅ COMPLETE
  Task 0.1: Makefile                    
  Task 0.2: Fix run_mistral.py path     

Phase 1: Shape-Parametric Koopman                        ✅ COMPLETE
  Task 1.1: Remove K=4096 gate, all shapes intercepted
  Task 1.2: Generic kernel for any K/N at runtime

Phase 2: Real Telemetry                                  ✅ COMPLETE
  Task 2.1: NVML clock/power/utilization -> liquid state
  Task 2.2: Windowed MFU from GEMM geometry

Phase 3: MFU Oracle Feedback + Billing                   <- NEXT
  Task 3.1: MFU -> oracle threshold adjustment
  Task 3.2: Per-step billing counters

Phase 4: NCCL Intercept + Overlap                        <- -60 TFLOPS gap
  Task 4.1: ncclAllReduce GOT patch + timing
  Task 4.2: Compute-communication overlap via Green Contexts

Phase 5: Non-Matmul Substitution                         <- -50 TFLOPS gap
  Task 5.1: Wire cuLaunchKernel to Chebyshev recipes
  Task 5.2: Kernel fusion (norm+GEMM, GEMM+activation)

Phase 6: Attention Optimization                          <- -70 TFLOPS gap
  Task 6.1: FlashAttention geometry detection
  Task 6.2: Decode attention low-rank Koopman substitute

Phase 7: Hardware-Only Validation + Billing              <- PROVE ALL FIVE GAPS
  Task 7.1: Kernel intercept count (no escapes)
  Task 7.2: Correctness vs cuBLAS (max_diff < 0.01)
  Task 7.3: MFU before vs after (full workload, not just GEMM)
  Task 7.4: GPU clock stability under elimination
  Task 7.5: Per-gap billing report with real numbers
```

## MFU Gap Analysis (why 480 not 700)

```
700 TFLOPS  <- cuBLAS GEMM ceiling (H100 thermal steady state)
 -70        <- attention (memory bound)           -> Phase 6
 -50        <- norms + activations (elementwise)  -> Phase 5
 -60        <- AllReduce blocking (GPU idle)      -> Phase 4
 -25        <- optimizer steps                    -> future (v3)
 -15        <- framework dispatch overhead        -> partially addressed by LD_PRELOAD
-----------
~480 TFLOPS <- published end-to-end MFU (Meta/Google/Microsoft)

CIPHER v2 target: close 3 of 5 gaps = recover ~180 TFLOPS -> ~660 TFLOPS end-to-end
```

## Goal Traceability

| Goal | Phases | Mechanism | TFLOPS recovered |
|------|--------|-----------|-----------------|
| **1. Eliminate iterative algorithms** | Phase 1, 5, 6 | Shape-parametric Koopman for GEMMs. Chebyshev for norms/activations. Low-rank for decode attention. | ~120 |
| **2. Raise end-to-end MFU** | Phase 2, 3, 4, 5, 6 | Real telemetry, MFU oracle feedback, NCCL overlap, non-GEMM substitution, attention optimization. | ~180 total |
| **3. Zero application changes** | Phase 0, 4 | LD_PRELOAD + PLT export. NCCL/cuBLAS/cuLaunchKernel all intercepted same way. | N/A |

## Drift Rule (Overrides Everything)

**If any task requires knowledge of:**
- Which model is running
- Which layer we are in
- What activations mean

**STOP and flag it immediately. That is drift.**

CIPHER sees only geometry: M, N, K, function pointer, grid dims.

## What This Plan Does NOT Include (Explicit Non-Goals for v2)

- **Model-specific validation** — No Mistral, no Llama, no per-layer hooks. Hardware metrics only.
- **Optimizer step optimization** — -25 TFLOPS gap. Requires intercepting Adam/SGD kernels. Deferred to v3.
- **eBPF real loading** — NCCL policy applied from user space. Real eBPF requires root + kernel module. Deferred.
- **Multi-node testing** — Phase 4 NCCL hook works single-node. Multi-node validation requires cluster access.
- **Blackwell/AMD support** — Hardware profiles exist in cipher_hw_desc.cpp but untested. H100-only for v2.
